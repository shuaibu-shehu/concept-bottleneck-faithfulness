"""Train the head on the supervised concept layer: ERM, the mitigations and the penalty.

Includes the Group-DRO and penalty-weight sweeps (Appendix D).

Run:    python -m src.experiments.mitigations_supervised
Writes: results/mitigations_supervised.json
"""
import json
import os
import time

import numpy as np
import torch

from src.concept_layer import (GROUPS, concept_scores, fit_probes, load_ddi,
                               load_fitz, signs_for)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# identical to mitigations_zeroshot.py
from src.training import (ETAS, FINAL_SEEDS, GAMMAS, GAMMA_AUROC_FLOOR, SWEEP_SEEDS,
                          WDS, make_trainer)
from src.metrics import (auc_np, mean_of, mono_np, paired_ci, probs, sens_global,
                         sens_pergroup, summarise, worst_auc)


# --- training harness ---
def build_evaluator(Cva, Cte, Cddi, yva, gva, yte, gte, yddi, gddi, signs):
    def evaluate(P):
        pv, pt, pd_ = probs(P, Cva), probs(P, Cte), probs(P, Cddi)
        rec = {'val_worst_auc': worst_auc(pv, yva, gva),
               'val_mono': mono_np(P, Cva, signs),
               'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
               'test_mono': {k: mono_np(P, Cte[torch.tensor(gte == k)], signs) for k in GROUPS},
               'ddi_auc': {k: auc_np(yddi[gddi == k], pd_[gddi == k]) for k in GROUPS},
               'ddi_mono': {k: mono_np(P, Cddi[torch.tensor(gddi == k)], signs) for k in GROUPS},
               'sens@spec_global_thr': sens_global(pv, yva, pt, yte, gte),
               'sens@spec_pergroup_thr': sens_pergroup(pv, yva, gva, pt, yte, gte)}
        rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
        rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
        rec['test_mono_gap'] = rec['test_mono']['light'] - rec['test_mono']['dark']
        return rec
    return evaluate


# --- 0. ERM regression check ---
def erm_regression_check(F, D):
    """Re-run the 7-concept ERM row of mitigations_zeroshot.py and require identical results."""
    ref_path = os.path.join(REPO, 'results/mitigations_zeroshot.json')
    if not os.path.exists(ref_path):
        print('WARNING: results/mitigations_zeroshot.json absent -- regression check SKIPPED')
        return {'ran': False, 'reason': 'mitigations_zeroshot.json absent'}
    ref = json.load(open(ref_path))['methods']['erm']

    tr, va, te = F['split'] == 'train', F['split'] == 'val', F['split'] == 'test'
    C = torch.tensor(F['C7'])
    Y = torch.tensor(F['y'].astype(np.float32))
    sg7 = np.ones(7, dtype=int)          # all 7 checklist criteria are malignancy-suggestive
    train = make_trainer(C[tr], Y[tr], F['g'][tr], sg7)
    ev = build_evaluator(C[va], C[te], torch.tensor(D['C7']),
                         F['y'][va], F['g'][va], F['y'][te], F['g'][te],
                         D['y'], D['g'], sg7)
    runs = [ev(train('erm', s)[0]) for s in FINAL_SEEDS]
    got = summarise(runs)

    diffs = {}
    for key in ['test_worst_auc', 'test_auc_gap', 'test_mono_gap']:
        diffs[key] = abs(got[key]['mean'] - ref[key]['mean'])
    for blk in ['test_auc', 'test_mono']:
        for k in GROUPS:
            diffs['%s.%s' % (blk, k)] = abs(got[blk][k]['mean'] - ref[blk][k]['mean'])
    mx = max(diffs.values())
    print('ERM regression check vs mitigations_zeroshot.json: max abs diff %.3e  -> %s'
          % (mx, 'PASS' if mx == 0.0 else ('CLOSE' if mx < 1e-9 else 'FAIL')))
    for k, v in sorted(diffs.items(), key=lambda kv: -kv[1])[:3]:
        print('    %-22s %.3e' % (k, v))
    assert mx < 1e-9, (
        'ERM row does not reproduce mitigations_zeroshot.json (max abs diff %.3e) -- run void. '
        'Before assuming the harness is wrong, check the thread count: this ran with '
        'torch.get_num_threads()=%d, OMP_NUM_THREADS=%s. Bit-identical reproduction '
        'requires the default thread count; at 2 threads this check fails at ~3.5e-3 '
        'with correct code.' % (mx, torch.get_num_threads(),
                                os.environ.get('OMP_NUM_THREADS', 'unset')))
    return {'ran': True, 'max_abs_diff': float(mx),
            'torch_num_threads': torch.get_num_threads(),
            'per_quantity': {k: float(v) for k, v in diffs.items()}}


# --- main ---
def main():
    t0 = time.time()
    F, D = load_fitz(), load_ddi()
    names = F['names']
    print('Fitzpatrick %d | concepts %d | DDI join %s'
          % (len(F['y']), len(names), 'OK' if D['join_ok'] else 'UNAVAILABLE'))

    print('\n' + '=' * 78)
    print('0. ERM REGRESSION CHECK (7 zero-shot concepts, must reproduce the paper)')
    print('=' * 78)
    reg = erm_regression_check(F, D)

    # supervised concept layer, probes fit on the training split only
    tr, va, te = F['split'] == 'train', F['split'] == 'val', F['split'] == 'test'
    probes = fit_probes(F['emb'], F['L'], F['has'], np.where(tr)[0])
    S = concept_scores(probes, F['emb'])
    Sd = concept_scores(probes, D['emb'])
    signs = signs_for(names, 'clinical')
    n_scored = int((signs != 0).sum())
    print('\nsupervised concept layer: %d concepts, %d scored (+1 x%d, -1 x%d)'
          % (len(names), n_scored, int((signs > 0).sum()), int((signs < 0).sum())))
    print('train %d  val %d  test %d  | ddi %d' % (tr.sum(), va.sum(), te.sum(), len(D['g'])))

    Ct = torch.tensor(S[tr], dtype=torch.float32)
    Cv = torch.tensor(S[va], dtype=torch.float32)
    Ce = torch.tensor(S[te], dtype=torch.float32)
    Cd = torch.tensor(Sd, dtype=torch.float32)
    Yt = torch.tensor(F['y'][tr].astype(np.float32))
    train = make_trainer(Ct, Yt, F['g'][tr], signs)
    ev = build_evaluator(Cv, Ce, Cd, F['y'][va], F['g'][va], F['y'][te], F['g'][te],
                         D['y'], D['g'], signs)

    out = {'erm_regression_check': reg,
           'bottleneck': {'kind': 'supervised SkinCon probes on frozen DermLIP embedding',
                          'n_concepts': len(names), 'n_scored': n_scored,
                          'concepts': names,
                          'signs': {names[j]: int(signs[j]) for j in range(len(names))},
                          'probes_fit_on': 'official train split only'}}

    # 1. Group-DRO sweep
    print('\n' + '=' * 78)
    print('1. GROUP-DRO SWEEP (selection on VAL worst-group AUROC)')
    print('=' * 78)
    sweep = []
    for eta in ETAS:
        for wd in WDS:
            runs, qs = [], []
            for s in SWEEP_SEEDS:
                P, q = train('gdro', s, eta=eta, wd=wd)
                runs.append(ev(P)); qs.append(q)
            vw, _ = mean_of(runs, ['val_worst_auc'])
            qm = np.mean(qs, axis=0)
            sweep.append({'eta': eta, 'wd': wd, 'val_worst_auc': vw,
                          'final_q_mean': [float(v) for v in qm], 'q_max': float(qm.max())})
            print('  eta=%-7g wd=%-6g  val worst %.4f   q = [%.3f %.3f %.3f]  max %.3f'
                  % (eta, wd, vw, qm[0], qm[1], qm[2], qm.max()))
    best = max(sweep, key=lambda r: r['val_worst_auc'])
    print('selected: eta=%g wd=%g (val worst %.4f)  [%.0fs]'
          % (best['eta'], best['wd'], best['val_worst_auc'], time.time() - t0))
    out['gdro_sweep'] = sweep
    out['gdro_selected'] = best
    out['n_configs_with_collapsed_q'] = int(sum(1 for r in sweep if r['q_max'] > 0.99))

    # 2. penalty-weight (gamma) sweep
    print('\n' + '=' * 78)
    print('2. GAMMA SWEEP for mono_global  (declared selection rule)')
    print('=' * 78)
    erm_val_runs = [ev(train('erm', s)[0]) for s in SWEEP_SEEDS]
    erm_val_worst, _ = mean_of(erm_val_runs, ['val_worst_auc'])
    print('  ERM val worst-group AUROC = %.4f ; floor = %.4f'
          % (erm_val_worst, erm_val_worst - GAMMA_AUROC_FLOOR))
    gsweep = []
    for gam in GAMMAS:
        runs = [ev(train('mono_global', s, gamma=gam)[0]) for s in SWEEP_SEEDS]
        vw, _ = mean_of(runs, ['val_worst_auc'])
        vm, _ = mean_of(runs, ['val_mono'])
        ok = vw >= erm_val_worst - GAMMA_AUROC_FLOOR
        gsweep.append({'gamma': gam, 'val_worst_auc': vw, 'val_mono': vm,
                       'meets_auroc_floor': bool(ok)})
        print('  gamma=%-5g  val worst %.4f  val mono %.4f   %s'
              % (gam, vw, vm, 'eligible' if ok else 'below floor'))
    eligible = [r for r in gsweep if r['meets_auroc_floor']]
    if eligible:
        gbest = max(eligible, key=lambda r: r['val_mono'])
        floor_met = True
    else:
        gbest = min(gsweep, key=lambda r: r['gamma'])
        floor_met = False
        print('  NO configuration met the floor -- falling back to smallest gamma, '
              'as declared')
    print('selected gamma = %g  (val mono %.4f, val worst %.4f)  [%.0fs]'
          % (gbest['gamma'], gbest['val_mono'], gbest['val_worst_auc'], time.time() - t0))
    out['gamma_sweep'] = {'erm_val_worst_auc': erm_val_worst,
                          'floor': erm_val_worst - GAMMA_AUROC_FLOOR,
                          'curve': gsweep, 'selected': gbest,
                          'floor_constraint_met': floor_met}

    # 3. method comparison
    print('\n' + '=' * 78)
    print('3. METHOD COMPARISON (%d seeds)' % len(FINAL_SEEDS))
    print('=' * 78)
    G = gbest['gamma']
    METHODS = [('erm', {}, 'erm'),
               ('gdro', {'eta': 1.0, 'wd': 0.0}, 'gdro_as_published'),
               ('gdro', {'eta': best['eta'], 'wd': best['wd']}, 'gdro_tuned'),
               ('resample', {}, 'resample'),
               ('reweight', {}, 'reweight'),
               ('mono_global', {'gamma': 2.0}, 'mono_global_gamma2'),
               ('mono_global', {'gamma': G}, 'mono_global_tuned'),
               ('mono_pergroup', {'gamma': G}, 'mono_pergroup_tuned')]

    results, per_seed = {}, {}
    print('%-22s %7s %7s %7s %7s %7s %7s' % ('method', 'worstAU', 'aucGap', 'monoGap',
                                             'mono_lt', 'mono_dk', 'sensDk'))
    for m, kw, name in METHODS:
        runs = [ev(train(m, s, **kw)[0]) for s in FINAL_SEEDS]
        per_seed[name] = {
            'test_worst_auc': [r['test_worst_auc'] for r in runs],
            'test_mono_dark': [r['test_mono']['dark'] for r in runs],
            'test_mono_light': [r['test_mono']['light'] for r in runs],
            'test_mono_gap': [r['test_mono_gap'] for r in runs],
            'test_auc_gap': [r['test_auc_gap'] for r in runs],
            'sens_dark': [r['sens@spec_global_thr']['dark'] for r in runs]}
        rec = summarise(runs, {'method': m, 'kwargs': kw})
        results[name] = rec
        print('%-22s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f'
              % (name, rec['test_worst_auc']['mean'], rec['test_auc_gap']['mean'],
                 rec['test_mono_gap']['mean'], rec['test_mono']['light']['mean'],
                 rec['test_mono']['dark']['mean'], rec['sens@spec_global_thr']['dark']['mean']))
    out['methods'] = results
    out['per_seed'] = per_seed

    # 4. paired contrasts with ERM
    print('\n' + '=' * 78)
    print('4. PAIRED vs ERM  (95% bootstrap CI on the per-seed difference)')
    print('=' * 78)
    vs = {}
    for name in results:
        if name == 'erm':
            continue
        vs[name] = {q: paired_ci(per_seed[name][q], per_seed['erm'][q])
                    for q in per_seed[name]}
        d = vs[name]['test_mono_dark']
        print('  %-22s mono_dark %+.4f [%+.4f, %+.4f]  (up in %d/%d)  %s'
              % (name, d['mean'], d['lo'], d['hi'], d['n_positive'], d['n'],
                 'RAISES dark mono' if d['lo'] > 0 else
                 ('LOWERS dark mono' if d['hi'] < 0 else 'CI covers 0')))
    out['paired_vs_erm'] = vs

    # pre-specified check: does an accuracy-directed method raise dark-skin monotonicity
    # above ERM, with a paired CI excluding zero?
    acc_methods = ['gdro_as_published', 'gdro_tuned', 'resample', 'reweight']
    weakened = [m for m in acc_methods if m in vs and vs[m]['test_mono_dark']['lo'] > 0]
    out['prespecified_check'] = {
        'accuracy_methods_raising_dark_mono_with_CI_excluding_zero': weakened,
        'dissociation_weakened': bool(weakened)}
    print('\n  Pre-specified check: accuracy-directed methods that raise dark monotonicity '
          '(CI excludes 0): %s' % (weakened if weakened else 'none'))

    dest = os.path.join(REPO, 'results/mitigations_supervised.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs total]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
