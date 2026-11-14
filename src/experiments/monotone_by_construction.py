"""A head that is monotone in the signed concepts by construction (Section 5.5).

For f(x) = W2 . relu(W1 x + b1) + b2, df/dx_i = sum_k W2_k relu'_k W1[i,k], so W2 >= 0 and
W1[i,:] >= 0 make f non-decreasing in x_i. Inputs are sign-aligned first (x_i = c_i for
sign +1, 1 - c_i for sign -1), and only the 14 signed concepts are constrained.

Run:    python -m src.experiments.monotone_by_construction
Writes: results/monotone_by_construction.json
"""
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

from src.concept_layer import GROUPS, concept_scores, fit_probes, load_ddi, load_fitz, signs_for
from src.training import FINAL_SEEDS, ITERS, LR, fwd, make_trainer, mono_terms
from src.metrics import auc_np, paired_ci, sens_global, sens_pergroup, summarise, worst_auc

HID = 16
NEGLIGIBLE = 0.01


def align(X, signs):
    """Sign-align so that asserting any scored concept always INCREASES its coordinate."""
    Z = X.clone()
    for i in range(X.shape[1]):
        if signs[i] < 0:
            Z[:, i] = 1.0 - X[:, i]
    return Z


def mono_fwd(Z, P):
    """P = [V1, b1, V2, b2]; positivity is imposed inside, so the parameters stay free."""
    V1, b1, V2, b2, pos_mask = P
    W1 = torch.where(pos_mask, torch.nn.functional.softplus(V1), V1)
    W2 = torch.nn.functional.softplus(V2)
    return (torch.relu(Z @ W1 + b1) @ W2 + b2).squeeze(1)


def train_constrained(Ztr, ytr, signs, seed, wtr=None):
    gg = torch.Generator().manual_seed(seed)
    n_in = Ztr.shape[1]
    V1 = (torch.randn(n_in, HID, generator=gg) * 0.3).requires_grad_(True)
    b1 = torch.zeros(HID, requires_grad=True)
    V2 = (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True)
    b2 = torch.zeros(1, requires_grad=True)
    # rows of the signed concepts are non-negative; the others stay free
    pos_mask = torch.tensor([[signs[i] != 0] * HID for i in range(n_in)], dtype=torch.bool)
    P = [V1, b1, V2, b2, pos_mask]
    opt = torch.optim.Adam([V1, b1, V2, b2], lr=LR)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    pw = torch.tensor(max(float((ytr == 0).sum()), 1.0) / max(float((ytr == 1).sum()), 1.0))
    for _ in range(ITERS):
        per = bce(mono_fwd(Ztr, P), ytr, pos_weight=pw, reduction='none')
        loss = per.mean() if wtr is None else (per * wtr).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return [V1.detach(), b1.detach(), V2.detach(), b2.detach(), pos_mask]


def mono_rate_constrained(P, Z, signs):
    """Monotonicity in the ALIGNED space: asserting = coordinate 1, negating = 0."""
    outs = []
    for i in range(Z.shape[1]):
        if signs[i] == 0:
            continue
        hi, lo = Z.clone(), Z.clone()
        hi[:, i], lo[:, i] = 1.0, 0.0
        with torch.no_grad():
            outs.append((torch.sigmoid(mono_fwd(hi, P)) >= torch.sigmoid(mono_fwd(lo, P)) - 1e-6)
                        .float().numpy())
    return float(np.mean(outs)), np.array(outs)


def effects_constrained(P, Z, signs):
    out = []
    for i in range(Z.shape[1]):
        if signs[i] == 0:
            continue
        hi, lo = Z.clone(), Z.clone()
        hi[:, i], lo[:, i] = 1.0, 0.0
        with torch.no_grad():
            out.append((torch.sigmoid(mono_fwd(hi, P)) - torch.sigmoid(mono_fwd(lo, P))).numpy())
    return np.array(out)


def main():
    t0 = time.time()
    mit = json.load(open(os.path.join(REPO, 'results/mitigations_supervised.json')))
    GAM = mit['gamma_sweep']['selected']['gamma']
    print('gamma = %g (validation-selected, for the soft comparator)' % GAM)

    F, D = load_fitz(), load_ddi()
    names = F['names']
    signs = signs_for(names, 'clinical')
    n_scored = int((signs != 0).sum())
    tr, va, te = F['split'] == 'train', F['split'] == 'val', F['split'] == 'test'
    probes = fit_probes(F['emb'], F['L'], F['has'], np.where(tr)[0])
    S = concept_scores(probes, F['emb'])
    Sd = concept_scores(probes, D['emb'])

    Xt = torch.tensor(S[tr], dtype=torch.float32)
    Xv = torch.tensor(S[va], dtype=torch.float32)
    Xe = torch.tensor(S[te], dtype=torch.float32)
    Xd = torch.tensor(Sd, dtype=torch.float32)
    Zt, Zv, Ze, Zd = (align(x, signs) for x in (Xt, Xv, Xe, Xd))
    yt = torch.tensor(F['y'][tr].astype(np.float32))
    yva, gva = F['y'][va].astype(np.float32), F['g'][va]
    yte, gte = F['y'][te].astype(np.float32), F['g'][te]
    print('%d concepts, %d scored (constrained), %d free' % (len(names), n_scored,
                                                             len(names) - n_scored))

    # group weights for the reweighted constrained variant
    gm = [torch.tensor(F['g'][tr] == k) for k in GROUPS]
    inv = torch.tensor([1.0 / max(float(m.sum()), 1.0) for m in gm])
    w = torch.zeros(len(yt))
    for k, m in enumerate(gm):
        w[m] = inv[k]
    w = w / w.mean()

    def ev_constrained(P):
        with torch.no_grad():
            pv = torch.sigmoid(mono_fwd(Zv, P)).numpy()
            pt = torch.sigmoid(mono_fwd(Ze, P)).numpy()
            pd_ = torch.sigmoid(mono_fwd(Zd, P)).numpy()
        rec = {'val_worst_auc': worst_auc(pv, yva, gva),
               'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
               'test_mono': {}, 'ddi_auc': {k: auc_np(D['y'].astype(np.float32)[D['g'] == k],
                                                     pd_[D['g'] == k]) for k in GROUPS},
               'sens@spec_global_thr': sens_global(pv, yva, pt, yte, gte),
               'sens@spec_pergroup_thr': sens_pergroup(pv, yva, gva, pt, yte, gte)}
        for k in GROUPS:
            m = torch.tensor(gte == k)
            rec['test_mono'][k] = mono_rate_constrained(P, Ze[m], signs)[0]
        rec['val_mono'] = mono_rate_constrained(P, Zv, signs)[0]
        rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
        rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
        rec['test_mono_gap'] = rec['test_mono']['light'] - rec['test_mono']['dark']
        rec['ddi_mono'] = {k: float('nan') for k in GROUPS}
        return rec

    runs, heads = {}, {}
    print('\ntraining monotone-by-construction heads ...')
    heads['constrained'] = [train_constrained(Zt, yt, signs, s) for s in FINAL_SEEDS]
    runs['constrained'] = [ev_constrained(P) for P in heads['constrained']]
    heads['constrained_reweight'] = [train_constrained(Zt, yt, signs, s, wtr=w)
                                     for s in FINAL_SEEDS]
    runs['constrained_reweight'] = [ev_constrained(P) for P in heads['constrained_reweight']]

    # assert the guarantee
    mono_all = [r['test_mono'][k] for r in runs['constrained'] for k in GROUPS]
    print('monotonicity of the constrained head: min %.6f max %.6f over %d (seed, tone) cells'
          % (min(mono_all), max(mono_all), len(mono_all)))
    assert min(mono_all) > 0.999999, (
        'construction is broken: monotonicity should be exactly 1.0 by design, got %.6f'
        % min(mono_all))
    print('  -> 1.0 BY CONSTRUCTION, as designed (asserted, not measured)')

    out = {'construction_verified': True,
           'monotonicity_min_over_cells': float(min(mono_all)),
           'methods': {}, 'per_seed': {}}
    for m, rr in runs.items():
        out['methods'][m] = summarise(rr, {'method': m})
        out['per_seed'][m] = {
            'test_worst_auc': [r['test_worst_auc'] for r in rr],
            'test_auc_gap': [r['test_auc_gap'] for r in rr],
            'test_mono_dark': [r['test_mono']['dark'] for r in rr],
            'sens_dark': [r['sens@spec_global_thr']['dark'] for r in rr]}

    # soft comparators, retrained on the same seeds for paired contrasts
    train_soft = make_trainer(Xt, yt, F['g'][tr], signs)

    def ev_soft(P):
        with torch.no_grad():
            pv = torch.sigmoid(fwd(Xv, P)).numpy(); pt = torch.sigmoid(fwd(Xe, P)).numpy()
        rec = {'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS}}
        rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
        rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
        rec['sens_dark'] = sens_global(pv, yva, pt, yte, gte)['dark']
        return rec

    print('training soft comparators ...')
    soft = {}
    for nm, meth, kw in [('mono_global', 'mono_global', {'gamma': GAM}),
                         ('erm', 'erm', {})]:
        rr = [ev_soft(train_soft(meth, s, **kw)[0]) for s in FINAL_SEEDS]
        soft[nm] = {'test_worst_auc': [r['test_worst_auc'] for r in rr],
                    'test_auc_gap': [r['test_auc_gap'] for r in rr],
                    'sens_dark': [r['sens_dark'] for r in rr]}
    # reweight + mono, retrained on the same seeds
    def reweight_mono(seed):
        gg = torch.Generator().manual_seed(seed)
        P = [(torch.randn(Xt.shape[1], HID, generator=gg) * 0.3).requires_grad_(True),
             torch.zeros(HID, requires_grad=True),
             (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True),
             torch.zeros(1, requires_grad=True)]
        opt = torch.optim.Adam(P, lr=LR)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        pw = torch.tensor(max(float((yt == 0).sum()), 1.0) / max(float((yt == 1).sum()), 1.0))
        for _ in range(ITERS):
            per = bce(fwd(Xt, P), yt, pos_weight=pw, reduction='none')
            loss = (per * w).mean() + GAM * mono_terms(Xt, P, signs).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return [p.detach() for p in P]
    rr = [ev_soft(reweight_mono(s)) for s in FINAL_SEEDS]
    soft['reweight_mono'] = {'test_worst_auc': [r['test_worst_auc'] for r in rr],
                             'test_auc_gap': [r['test_auc_gap'] for r in rr],
                             'sens_dark': [r['sens_dark'] for r in rr]}
    out['soft_comparators'] = {k: {q: float(np.mean(v)) for q, v in d.items()}
                               for k, d in soft.items()}

    # is monotonicity 1.0 only because effects vanish?
    E = np.mean([effects_constrained(P, Ze, signs) for P in heads['constrained']], axis=0)
    out['effect_magnitude'] = {
        'mean_abs_effect': float(np.abs(E).mean()),
        'frac_negligible_per_concept': [float(v) for v in (np.abs(E) < NEGLIGIBLE).mean(1)],
        'n_concepts_negligible_over_90pct': int(((np.abs(E) < NEGLIGIBLE).mean(1) > 0.90).sum()),
        'scored_concepts': [names[j] for j in range(len(names)) if signs[j] != 0]}
    mech = json.load(open(os.path.join(REPO, 'results/monotonicity_mechanism.json')))
    soft_mag = mech['mono']['mean_abs_effect']
    out['effect_magnitude']['soft_objective_mean_abs_effect'] = soft_mag
    out['effect_magnitude']['ratio_constrained_over_soft'] = float(np.abs(E).mean() / soft_mag)

    # tables and the pre-specified verdict
    print('\n' + '=' * 92)
    print('%-24s %9s %9s %9s %9s %9s' % ('method', 'worstAUC', 'aucGap', 'monoDark',
                                         'sensDark', '|effect|'))
    print('=' * 92)
    for m in ('constrained', 'constrained_reweight'):
        r = out['methods'][m]
        print('%-24s %9.4f %9.4f %9.4f %9.4f %9.4f'
              % (m + ' (=1.0 by design)', r['test_worst_auc']['mean'],
                 r['test_auc_gap']['mean'], r['test_mono']['dark']['mean'],
                 r['sens@spec_global_thr']['dark']['mean'],
                 out['effect_magnitude']['mean_abs_effect']))
    for m in ('mono_global', 'reweight_mono', 'erm'):
        d = soft[m]
        print('%-24s %9.4f %9.4f %9s %9.4f %9s'
              % (m + ' (soft)', float(np.mean(d['test_worst_auc'])),
                 float(np.mean(d['test_auc_gap'])),
                 '%.4f' % mech['mono']['monotonicity'] if m != 'erm' else '-',
                 float(np.mean(d['sens_dark'])),
                 '%.4f' % soft_mag if m != 'erm' else '-'))

    print('\n' + '=' * 92)
    print('EXPECTED: the constrained head reaches 1.0')
    print('but LOSES worst-group AUROC against the soft objective.')
    print('=' * 92)
    verdicts = {}
    for hard in ('constrained', 'constrained_reweight'):
        for softm in ('mono_global', 'reweight_mono'):
            ci = paired_ci(out['per_seed'][hard]['test_worst_auc'],
                           soft[softm]['test_worst_auc'])
            key = '%s_vs_%s' % (hard, softm)
            verdicts[key] = {**ci, 'hard_loses': bool(ci['hi'] < 0),
                             'hard_wins': bool(ci['lo'] > 0)}
            print('  %-42s %+.4f [%+.4f, %+.4f]  %s'
                  % (key, ci['mean'], ci['lo'], ci['hi'],
                     'HARD LOSES' if ci['hi'] < 0 else
                     ('HARD WINS -- soft objective is DOMINATED' if ci['lo'] > 0
                      else 'CI covers 0')))
    out['paired_hard_minus_soft'] = verdicts
    held = all(v['hard_loses'] for v in verdicts.values())
    dominated = any(v['hard_wins'] for v in verdicts.values())
    out['verdict'] = {
        'prediction_held': bool(held), 'soft_objective_dominated': bool(dominated),
        'statement': ('PREDICTION HELD -- the hard constraint attains 1.0 but costs '
                      'worst-group AUROC against every soft variant, so the soft penalty '
                      'is a Pareto choice and the redistribution mechanism explains why'
                      if held else
                      ('SOFT OBJECTIVE DOMINATED -- a monotone-by-construction head '
                       'matches or beats it while attaining 1.0. This goes in the abstract.'
                       if dominated else
                       'MIXED -- some contrasts cover zero; report the full grid'))}
    print('\n  VERDICT: %s' % out['verdict']['statement'])
    print('\n  effect magnitude: constrained %.4f vs soft %.4f (%.2fx) | concepts '
          'negligible on >90%% of samples: %d of %d'
          % (out['effect_magnitude']['mean_abs_effect'], soft_mag,
             out['effect_magnitude']['ratio_constrained_over_soft'],
             out['effect_magnitude']['n_concepts_negligible_over_90pct'], n_scored))

    dest = os.path.join(REPO, 'results/monotone_by_construction.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
