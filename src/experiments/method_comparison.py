"""Score every method on every metric and dataset (Table 1).

Run:    python -m src.experiments.method_comparison
Writes: results/method_comparison.json
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
from src.experiments.concept_grounded_faithfulness import support_levels
from src import last_layer as base


# --- metrics ---
def mono_pf(pf, Cm, signs, levels=None):
    """Monotonicity per sample; levels=None uses 0 and 1, otherwise the observed levels."""
    if len(Cm) == 0:
        return float('nan')
    outs = []
    for i in range(Cm.shape[1]):
        if signs[i] == 0:
            continue
        lo_v, hi_v = (0.0, 1.0) if levels is None else levels[i]
        hi, lo = Cm.clone(), Cm.clone()
        a, b = (hi_v, lo_v) if signs[i] > 0 else (lo_v, hi_v)
        hi[:, i], lo[:, i] = a, b
        outs.append((pf(hi) >= pf(lo) - 1e-6).astype(float))
    return float(np.mean(outs))


def gt_agreement(pf, Cm, signs, levels, L_sub, has_sub):
    """Agreement with the clinical direction on image-concept pairs annotated present."""
    n = len(Cm)
    hit, cnt = np.zeros(n), np.zeros(n)
    for i in range(Cm.shape[1]):
        if signs[i] == 0:
            continue
        lo_v, hi_v = levels[i]
        hi, lo = Cm.clone(), Cm.clone()
        a, b = (hi_v, lo_v) if signs[i] > 0 else (lo_v, hi_v)
        hi[:, i], lo[:, i] = a, b
        ok = (pf(hi) >= pf(lo) - 1e-6).astype(float)
        present = has_sub & np.isfinite(L_sub[:, i]) & (L_sub[:, i] == 1)
        hit[present] += ok[present]; cnt[present] += 1
    out = np.full(n, np.nan)
    m = cnt > 0
    out[m] = hit[m] / cnt[m]
    return out


def by_tone(v, g):
    m = np.isfinite(v)
    r = {k: (float(v[m & (g == k)].mean()) if (m & (g == k)).any() else float('nan'))
         for k in GROUPS}
    r['gap'] = r['light'] - r['dark']
    return r


def main():
    t0 = time.time()
    mit = json.load(open(os.path.join(REPO, 'results/mitigations_supervised.json')))
    GAM = mit['gamma_sweep']['selected']['gamma']
    gdro_cfg = mit['gdro_selected']
    print('gamma = %g | tuned Group-DRO eta=%g wd=%g'
          % (GAM, gdro_cfg['eta'], gdro_cfg['wd']))

    F, D = load_fitz(), load_ddi()
    names = F['names']
    signs = signs_for(names, 'clinical')
    tr, va, te = F['split'] == 'train', F['split'] == 'val', F['split'] == 'test'
    itr = np.where(tr)[0]
    probes = fit_probes(F['emb'], F['L'], F['has'], itr)
    S = concept_scores(probes, F['emb'])
    Sd = concept_scores(probes, D['emb'])
    levels, _ = support_levels(S, F['L'], F['has'], itr, names)

    Xt = torch.tensor(S[tr], dtype=torch.float32)
    Xv = torch.tensor(S[va], dtype=torch.float32)
    Xe = torch.tensor(S[te], dtype=torch.float32)
    Xd = torch.tensor(Sd, dtype=torch.float32)
    yt = torch.tensor(F['y'][tr].astype(np.float32))
    yva, gva = F['y'][va].astype(np.float32), F['g'][va]
    yte, gte = F['y'][te].astype(np.float32), F['g'][te]
    L_te, has_te = F['L'][te], F['has'][te]
    train = make_trainer(Xt, yt, F['g'][tr], signs)

    # Derm7pt, clinical photographs (same modality as Fitzpatrick)
    d7 = np.load(os.path.join(REPO, 'caches/fairness_cache/features_derm7pt.npz'),
                 allow_pickle=True)
    mel = d7['mel_label'].astype(int)
    X7 = torch.tensor(concept_scores(probes, d7['emb_clinic'].astype(np.float64)),
                      dtype=torch.float32)

    # point the last-layer methods at this concept layer
    base.C = torch.tensor(S, dtype=torch.float32)
    base.tr, base.va, base.te = tr, va, te
    base.Xtr, base.ytr, base.gtr = Xt, yt, F['g'][tr]
    base.Cva, base.yva, base.gva = Xv, yva, gva
    base.yte, base.gte = yte, gte
    base.Cddi, base.Gddi = Xd, D['g']
    base.yddi_np = D['y'].astype(np.float32)

    def evaluate(pf):
        pv, pt, pd_, p7 = pf(Xv), pf(Xe), pf(Xd), pf(X7)
        gt = gt_agreement(pf, Xe, signs, levels, L_te, has_te)
        rec = {
            'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
            'test_mono': {k: mono_pf(pf, Xe[torch.tensor(gte == k)], signs) for k in GROUPS},
            'test_mono_insupport': {k: mono_pf(pf, Xe[torch.tensor(gte == k)], signs, levels)
                                    for k in GROUPS},
            'test_gt_agreement': by_tone(gt, gte),
            'ddi_auc': {k: auc_np(base.yddi_np[D['g'] == k], pd_[D['g'] == k]) for k in GROUPS},
            'ddi_mono': {k: mono_pf(pf, Xd[torch.tensor(D['g'] == k)], signs) for k in GROUPS},
            'derm7pt_melanoma_auc': auc_np(mel.astype(np.float32), p7),
            'derm7pt_mono': mono_pf(pf, X7, signs),
            'sens@spec_global_thr': sens_global(pv, yva, pt, yte, gte),
            'sens@spec_pergroup_thr': sens_pergroup(pv, yva, gva, pt, yte, gte),
            'val_worst_auc': worst_auc(pv, yva, gva),
            'val_mono': mono_pf(pf, Xv, signs)}
        rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
        rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
        rec['test_mono_gap'] = rec['test_mono']['light'] - rec['test_mono']['dark']
        rec['test_mono_insupport_gap'] = (rec['test_mono_insupport']['light']
                                          - rec['test_mono_insupport']['dark'])
        return rec
    pf_head = lambda P: (lambda Cm: torch.sigmoid(fwd(Cm, P)).detach().numpy())  # noqa: E731

    def reweight_mono(seed, gamma):
        gg = torch.Generator().manual_seed(seed)
        P = [(torch.randn(Xt.shape[1], 16, generator=gg) * 0.3).requires_grad_(True),
             torch.zeros(16, requires_grad=True),
             (torch.randn(16, 1, generator=gg) * 0.3).requires_grad_(True),
             torch.zeros(1, requires_grad=True)]
        opt = torch.optim.Adam(P, lr=LR)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        pw = torch.tensor(max(float((yt == 0).sum()), 1.0) / max(float((yt == 1).sum()), 1.0))
        gmask = [torch.tensor(F['g'][tr] == g) for g in GROUPS]
        inv = torch.tensor([1.0 / max(float(m.sum()), 1.0) for m in gmask])
        w = torch.zeros(len(yt))
        for k, m in enumerate(gmask):
            w[m] = inv[k]
        w = w / w.mean()
        for _ in range(ITERS):
            per = bce(fwd(Xt, P), yt, pos_weight=pw, reduction='none')
            loss = (per * w).mean() + gamma * mono_terms(Xt, P, signs).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return [p.detach() for p in P]

    # check the result format on one quick head before the long training runs
    _probe = evaluate(pf_head(base.train_erm(0)))
    _need_scalar = ['test_worst_auc', 'test_auc_gap', 'test_mono_gap', 'val_worst_auc',
                    'val_mono']
    _need_block = ['test_auc', 'test_mono', 'ddi_auc', 'ddi_mono',
                   'sens@spec_global_thr', 'sens@spec_pergroup_thr']
    _missing = ([k for k in _need_scalar if k not in _probe]
                + [k for k in _need_block if k not in _probe]
                + ['%s.%s' % (b, k) for b in _need_block if b in _probe
                   for k in GROUPS if k not in _probe[b]])
    assert not _missing, 'evaluate() is missing keys summarise() requires: %s' % _missing
    print('contract check: evaluate() satisfies summarise() (%d scalars, %d blocks)'
          % (len(_need_scalar), len(_need_block)))

    print('\ntraining base ERM heads ...')
    P_erm = [base.train_erm(s) for s in FINAL_SEEDS]

    runs = {}
    print('full-head methods ...')
    runs['erm'] = [evaluate(pf_head(P)) for P in P_erm]
    runs['gdro_tuned'] = [evaluate(pf_head(train('gdro', s, eta=gdro_cfg['eta'],
                                                 wd=gdro_cfg['wd'])[0])) for s in FINAL_SEEDS]
    runs['resample'] = [evaluate(pf_head(train('resample', s)[0])) for s in FINAL_SEEDS]
    runs['reweight'] = [evaluate(pf_head(train('reweight', s)[0])) for s in FINAL_SEEDS]
    print('last-layer methods ...')
    runs['dfr'] = [evaluate(pf_head(base.dfr(P_erm[i], s)[0]))
                   for i, s in enumerate(FINAL_SEEDS)]
    runs['afr'] = [evaluate(pf_head(base.afr(P_erm[i], s)[0]))
                   for i, s in enumerate(FINAL_SEEDS)]
    from src import prototypes as dpe  # imported AFTER rebinding so its `base` alias sees new data
    runs['dpe'] = [evaluate(dpe.fit_and_select(P_erm[i], s)[0].prob_fn())
                   for i, s in enumerate(FINAL_SEEDS)]
    print('faithfulness-targeted methods ...')
    runs['mono_global'] = [evaluate(pf_head(train('mono_global', s, gamma=GAM)[0]))
                           for s in FINAL_SEEDS]
    runs['reweight_mono'] = [evaluate(pf_head(reweight_mono(s, GAM))) for s in FINAL_SEEDS]

    ORDER = ['erm', 'gdro_tuned', 'resample', 'reweight', 'dfr', 'afr', 'dpe',
             'mono_global', 'reweight_mono']
    out = {'method_order': ORDER, 'methods': {}, 'per_seed': {}}
    for m in ORDER:
        rr = runs[m]
        rec = summarise(rr, {'method': m})
        for scalar in ['derm7pt_melanoma_auc', 'derm7pt_mono', 'test_mono_insupport_gap']:
            vals = [r[scalar] for r in rr]
            rec[scalar] = {'mean': float(np.nanmean(vals)), 'sd': float(np.nanstd(vals))}
        for blk in ['test_mono_insupport', 'test_gt_agreement']:
            rec[blk] = {}
            for k in (GROUPS + ['gap'] if blk == 'test_gt_agreement' else GROUPS):
                vals = [r[blk][k] for r in rr]
                rec[blk][k] = {'mean': float(np.nanmean(vals)), 'sd': float(np.nanstd(vals))}
        # per-seed DDI aggregates, needed for their standard deviations
        ddi = {'ddi_auc_gap': [r['ddi_auc']['light'] - r['ddi_auc']['dark'] for r in rr],
               'ddi_mono_gap': [r['ddi_mono']['light'] - r['ddi_mono']['dark'] for r in rr],
               'ddi_auc_mean': [float(np.nanmean([r['ddi_auc'][g] for g in GROUPS]))
                                for r in rr],
               'ddi_mono_mean': [float(np.nanmean([r['ddi_mono'][g] for g in GROUPS]))
                                 for r in rr]}
        for _k, _v in ddi.items():
            rec[_k] = {'mean': float(np.nanmean(_v)), 'sd': float(np.nanstd(_v))}
        out['methods'][m] = rec
        out['per_seed'][m] = dict(ddi, **{
            'test_worst_auc': [r['test_worst_auc'] for r in rr],
            'test_auc_gap': [r['test_auc_gap'] for r in rr],
            'test_mono_dark': [r['test_mono']['dark'] for r in rr],
            'test_mono_light': [r['test_mono']['light'] for r in rr],
            'insupport_dark': [r['test_mono_insupport']['dark'] for r in rr],
            'gt_agreement_dark': [r['test_gt_agreement']['dark'] for r in rr],
            'derm7pt_auc': [r['derm7pt_melanoma_auc'] for r in rr],
            'derm7pt_mono': [r['derm7pt_mono'] for r in rr],
            'sens_dark': [r['sens@spec_global_thr']['dark'] for r in rr],
            'ddi_auc_light': [r['ddi_auc']['light'] for r in rr],
            'ddi_auc_dark': [r['ddi_auc']['dark'] for r in rr],
            'ddi_mono_light': [r['ddi_mono']['light'] for r in rr],
            'ddi_mono_dark': [r['ddi_mono']['dark'] for r in rr]})

    # --- tables ---
    def row(m, fields):
        r = out['methods'][m]
        vals = []
        for f in fields:
            v = r
            for p in f.split('.'):
                v = v[p]
            vals.append(v['mean'] if isinstance(v, dict) and 'mean' in v else v)
        return vals

    print('\n' + '=' * 100)
    print('TABLE 1  Fitzpatrick17k test split (10 seeds)')
    print('=' * 100)
    print('%-16s %8s %8s %8s %8s %8s %8s' % ('method', 'worstAU', 'aucGap', 'mono_lt',
                                             'mono_dk', 'monoGap', 'sensDk'))
    for m in ORDER:
        v = row(m, ['test_worst_auc', 'test_auc_gap', 'test_mono.light', 'test_mono.dark',
                    'test_mono_gap', 'sens@spec_global_thr.dark'])
        print('%-16s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % tuple([m] + v))

    print('\n' + '=' * 100)
    print('TABLE 2  UN-OPTIMIZED faithfulness criteria (never in any training objective)')
    print('=' * 100)
    print('%-16s %10s %10s %10s %10s %10s' % ('method', 'insup_lt', 'insup_dk', 'gt_lt',
                                              'gt_dk', 'gt_gap'))
    for m in ORDER:
        v = row(m, ['test_mono_insupport.light', 'test_mono_insupport.dark',
                    'test_gt_agreement.light', 'test_gt_agreement.dark',
                    'test_gt_agreement.gap'])
        print('%-16s %10.4f %10.4f %10.4f %10.4f %10.4f' % tuple([m] + v))

    print('\n' + '=' * 100)
    print('TABLE 3  EXTERNAL transfer -- DDI and Derm7pt (clinical photographs)')
    print('=' * 100)
    print('%-16s %9s %9s %9s %9s %9s %9s' % ('method', 'ddiAU_lt', 'ddiAU_dk', 'ddiM_lt',
                                             'ddiM_dk', 'd7_AUC', 'd7_mono'))
    for m in ORDER:
        v = row(m, ['ddi_auc.light', 'ddi_auc.dark', 'ddi_mono.light', 'ddi_mono.dark',
                    'derm7pt_melanoma_auc', 'derm7pt_mono'])
        print('%-16s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f' % tuple([m] + v))

    # --- paired contrasts with ERM ---
    print('\n' + '=' * 100)
    print('PAIRED vs ERM (95% bootstrap CI); "up"/"dn" = CI excludes zero')
    print('=' * 100)
    vs = {}
    keys = ['test_worst_auc', 'test_mono_dark', 'insupport_dark', 'gt_agreement_dark',
            'derm7pt_auc', 'derm7pt_mono', 'ddi_auc_gap', 'ddi_auc_dark']
    print('%-16s %-22s %-22s %-22s' % ('method', 'worstAUC', 'gt_agreement_dark',
                                       'derm7pt_AUC'))
    for m in ORDER:
        if m == 'erm':
            continue
        vs[m] = {k: paired_ci(out['per_seed'][m][k], out['per_seed']['erm'][k]) for k in keys}
        cells = []
        for k in ['test_worst_auc', 'gt_agreement_dark', 'derm7pt_auc']:
            c = vs[m][k]
            tag = 'up' if c['lo'] > 0 else ('dn' if c['hi'] < 0 else '--')
            cells.append('%+.4f [%+.4f,%+.4f]%s' % (c['mean'], c['lo'], c['hi'], tag))
        print('%-16s %-22s %-22s %-22s' % tuple([m] + cells))
    out['paired_vs_erm'] = vs

    # do accuracy-directed methods also change the criteria they do not optimise?
    acc = ['gdro_tuned', 'resample', 'reweight', 'dfr', 'afr', 'dpe']
    movers = {k: [m for m in acc if vs[m][k]['lo'] > 0]
              for k in ['insupport_dark', 'gt_agreement_dark', 'derm7pt_mono']}
    out['accuracy_methods_improving_unoptimized'] = movers
    print('\naccuracy-directed methods improving each UN-OPTIMIZED criterion '
          '(CI excludes 0):')
    for k, v in movers.items():
        print('  %-22s %s' % (k, v if v else 'none'))

    dest = os.path.join(REPO, 'results/method_comparison.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
