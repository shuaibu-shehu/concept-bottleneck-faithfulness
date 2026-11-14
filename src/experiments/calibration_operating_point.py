"""Per-tone operating point and calibration for the nine methods of Table 1 (Appendix E).

Retrains the nine methods as method_comparison.py does and reports, per tone on the test
split: sensitivity and specificity at the shared and per-tone thresholds, ECE before and
after one tone-agnostic Platt recalibration, and observed against expected malignant
cases. Checks its per-seed results against results/method_comparison.json.

Run:    python -m src.experiments.calibration_operating_point
Writes: results/calibration_operating_point.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

from src.concept_layer import GROUPS, concept_scores, fit_probes, load_ddi, load_fitz, signs_for
from src.training import FINAL_SEEDS, ITERS, LR, fwd, make_trainer, mono_terms
from src.metrics import SPEC, auc_np, sens_global
from src import last_layer as base

N_BINS = 10
N_NULL = 1000         # Bernoulli draws for the perfect-calibration ECE baseline
Z_CRIT = 1.96         # two-sided 5% level for the calibration-in-the-large z-score
CKPT = os.path.join(REPO, 'caches', 'calibration_operating_point')
ORDER = ['erm', 'gdro_tuned', 'resample', 'reweight', 'dfr', 'afr', 'dpe',
         'mono_global', 'reweight_mono']


# --- metrics ---
def thresholds(pv, yva, gva):
    """Shared threshold over all validation negatives, and one per tone (shared if <= 10)."""
    shared = float(np.quantile(pv[yva == 0], SPEC))
    per = {}
    for k in GROUPS:
        mv = (gva == k) & (yva == 0)
        per[k] = float(np.quantile(pv[mv], SPEC)) if mv.sum() > 10 else shared
    return shared, per


def operating_point(pt, yte, gte, thr):
    out = {}
    for k in GROUPS:
        t = thr[k] if isinstance(thr, dict) else thr
        m = gte == k
        out[k] = {'sensitivity': float((pt[m & (yte == 1)] > t).mean()),
                  'specificity': float((pt[m & (yte == 0)] <= t).mean())}
    return out


def bin_index(p, equal_mass):
    if equal_mass:
        idx = np.empty(len(p), dtype=int)
        idx[np.argsort(p, kind='stable')] = np.arange(len(p)) * N_BINS // len(p)
        return idx
    return np.clip((p * N_BINS).astype(int), 0, N_BINS - 1)


def reliability(p, y, equal_mass=False):
    """ECE and the per-bin (n, mean prediction, observed malignancy rate)."""
    idx = bin_index(p, equal_mass)
    ece, rows = 0.0, []
    for b in range(N_BINS):
        m = idx == b
        if not m.any():
            continue
        conf, freq = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - freq)
        rows.append({'bin': b, 'n': int(m.sum()), 'mean_pred': conf, 'frac_malignant': freq})
    return float(ece), rows


def null_ece(p, rng):
    """Mean and 95th percentile of the ECE a perfectly calibrated predictor reaches
    at this sample size.
    """
    draws = [reliability(p, (rng.random(len(p)) < p).astype(float))[0] for _ in range(N_NULL)]
    return float(np.mean(draws)), float(np.percentile(draws, 95))


def calibration_in_the_large(p, y):
    """Observed malignant cases against the count the probabilities predict."""
    expected, var = float(p.sum()), float((p * (1 - p)).sum())
    observed = float(y.sum())
    return {'observed': observed, 'expected': expected, 'var': var,
            'observed_over_expected': observed / expected,
            'z': (observed - expected) / np.sqrt(var)}


def dark_vs_light(cl):
    """Dark minus light difference in observed/expected malignant cases."""
    d, l = cl['dark'], cl['light']
    diff = d['observed_over_expected'] - l['observed_over_expected']
    se = np.sqrt(d['var'] / d['expected'] ** 2 + l['var'] / l['expected'] ** 2)
    return {'diff_observed_over_expected': float(diff), 'z': float(diff / se)}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(pv, yva):
    """One recalibration for all tones: logistic regression on the validation logit."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv)[:, None], yva)
    return lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]


def score_run(pv, pt, yva, gva, yte, gte):
    shared, per = thresholds(pv, yva, gva)
    pt_cal = platt(pv, yva)(pt)
    rec = {'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
           'sens_global_check': sens_global(pv, yva, pt, yte, gte),
           'shared_threshold': operating_point(pt, yte, gte, shared),
           'pergroup_threshold': operating_point(pt, yte, gte, per)}
    for tag, p in [('raw', pt), ('platt', pt_cal)]:
        for mass in (False, True):
            key = 'ece_%s%s' % (tag, '_equal_mass' if mass else '')
            rec[key] = {k: reliability(p[gte == k], yte[gte == k], mass)[0] for k in GROUPS}
        rec['mean_pred_%s' % tag] = {k: float(p[gte == k].mean()) for k in GROUPS}
    rng = np.random.default_rng(0)
    null = {k: null_ece(pt_cal[gte == k], rng) for k in GROUPS}
    rec['ece_platt_null'] = {k: null[k][0] for k in GROUPS}
    rec['ece_platt_null_p95'] = {k: null[k][1] for k in GROUPS}
    rec['calibration_in_the_large'] = {k: calibration_in_the_large(pt_cal[gte == k],
                                                                   yte[gte == k])
                                       for k in GROUPS}
    rec['dark_vs_light'] = dark_vs_light(rec['calibration_in_the_large'])
    rec['_pt'], rec['_pt_cal'] = pt, pt_cal
    return rec


def cached(method, fit):
    """Validation and test predictions per seed, cached so an interrupted run can resume."""
    path = os.path.join(CKPT, '%s_seeds%s.npz' % (method, '-'.join(map(str, FINAL_SEEDS))))
    if os.path.exists(path):
        z = np.load(path)
        return list(zip(z['pv'], z['pt']))
    out = fit()
    os.makedirs(CKPT, exist_ok=True)
    np.savez(path, pv=np.stack([pv for pv, _ in out]), pt=np.stack([pt for _, pt in out]))
    return out


def mean_sd(vals):
    return {'mean': float(np.nanmean(vals)), 'sd': float(np.nanstd(vals))}


# --- main ---
def main(dest=None):
    """dest overrides the output path, so a trial run cannot overwrite the results file."""
    t0 = time.time()
    mit = json.load(open(os.path.join(REPO, 'results/mitigations_supervised.json')))
    GAM = mit['gamma_sweep']['selected']['gamma']
    gdro_cfg = mit['gdro_selected']
    print('gamma = %g | tuned Group-DRO eta=%g wd=%g' % (GAM, gdro_cfg['eta'], gdro_cfg['wd']))

    # identical setup to method_comparison.py
    F, D = load_fitz(), load_ddi()
    names = F['names']
    signs = signs_for(names, 'clinical')
    tr, va, te = F['split'] == 'train', F['split'] == 'val', F['split'] == 'test'
    itr = np.where(tr)[0]
    probes = fit_probes(F['emb'], F['L'], F['has'], itr)
    S = concept_scores(probes, F['emb'])
    Sd = concept_scores(probes, D['emb'])

    Xt = torch.tensor(S[tr], dtype=torch.float32)
    Xv = torch.tensor(S[va], dtype=torch.float32)
    Xe = torch.tensor(S[te], dtype=torch.float32)
    Xd = torch.tensor(Sd, dtype=torch.float32)
    yt = torch.tensor(F['y'][tr].astype(np.float32))
    yva, gva = F['y'][va].astype(np.float32), F['g'][va]
    yte, gte = F['y'][te].astype(np.float32), F['g'][te]
    train = make_trainer(Xt, yt, F['g'][tr], signs)

    base.C = torch.tensor(S, dtype=torch.float32)
    base.tr, base.va, base.te = tr, va, te
    base.Xtr, base.ytr, base.gtr = Xt, yt, F['g'][tr]
    base.Cva, base.yva, base.gva = Xv, yva, gva
    base.yte, base.gte = yte, gte
    base.Cddi, base.Gddi = Xd, D['g']
    base.yddi_np = D['y'].astype(np.float32)

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

    def preds(pf):
        return pf(Xv), pf(Xe)

    P_erm = []

    def erm_heads():
        if not P_erm:
            P_erm.extend(base.train_erm(s) for s in FINAL_SEEDS)
        return P_erm

    from src import prototypes as dpe  # imported AFTER rebinding so its `base` alias sees new data
    fits = {
        'erm': lambda: [preds(pf_head(P)) for P in erm_heads()],
        'gdro_tuned': lambda: [preds(pf_head(train('gdro', s, eta=gdro_cfg['eta'],
                                                   wd=gdro_cfg['wd'])[0])) for s in FINAL_SEEDS],
        'resample': lambda: [preds(pf_head(train('resample', s)[0])) for s in FINAL_SEEDS],
        'reweight': lambda: [preds(pf_head(train('reweight', s)[0])) for s in FINAL_SEEDS],
        'dfr': lambda: [preds(pf_head(base.dfr(erm_heads()[i], s)[0]))
                        for i, s in enumerate(FINAL_SEEDS)],
        'afr': lambda: [preds(pf_head(base.afr(erm_heads()[i], s)[0]))
                        for i, s in enumerate(FINAL_SEEDS)],
        'dpe': lambda: [preds(dpe.fit_and_select(erm_heads()[i], s)[0].prob_fn())
                        for i, s in enumerate(FINAL_SEEDS)],
        'mono_global': lambda: [preds(pf_head(train('mono_global', s, gamma=GAM)[0]))
                                for s in FINAL_SEEDS],
        'reweight_mono': lambda: [preds(pf_head(reweight_mono(s, GAM))) for s in FINAL_SEEDS],
    }
    runs = {}
    for m in ORDER:
        print('%-14s ...  [%.0fs]' % (m, time.time() - t0))
        runs[m] = [score_run(pv, pt, yva, gva, yte, gte) for pv, pt in cached(m, fits[m])]

    # --- regression check against method_comparison.json ---
    ref = json.load(open(os.path.join(REPO, 'results/method_comparison.json')))
    worst = 0.0
    for m in ORDER:
        mine_worst = [min(r['test_auc'].values()) for r in runs[m]]
        mine_sens = [r['sens_global_check']['dark'] for r in runs[m]]
        worst = max(worst,
                    np.max(np.abs(np.array(mine_worst) - ref['per_seed'][m]['test_worst_auc'])),
                    np.max(np.abs(np.array(mine_sens) - ref['per_seed'][m]['sens_dark'])))
        for r in runs[m]:
            for k in GROUPS:
                assert r['shared_threshold'][k]['sensitivity'] == r['sens_global_check'][k]
    print('\nregression check vs method_comparison.json: max |diff| over per-seed worst-group '
          'AUROC and dark sensitivity = %.2e' % worst)

    # --- aggregate ---
    counts = {k: {'n': int((gte == k).sum()), 'n_malignant': int(((gte == k) & (yte == 1)).sum()),
                  'n_benign': int(((gte == k) & (yte == 0)).sum())} for k in GROUPS}
    out = {'protocol': __doc__.split('\n\n')[1].strip(), 'n_bins': N_BINS, 'spec': SPEC,
           'n_seeds': len(FINAL_SEEDS), 'test_counts': counts,
           'regression_check_max_abs_diff': float(worst), 'method_order': ORDER, 'methods': {}}
    for m in ORDER:
        rr = runs[m]
        rec = {}
        for op in ['shared_threshold', 'pergroup_threshold']:
            rec[op] = {k: {q: mean_sd([r[op][k][q] for r in rr])
                           for q in ['sensitivity', 'specificity']} for k in GROUPS}
        for key in ['ece_raw', 'ece_raw_equal_mass', 'ece_platt', 'ece_platt_equal_mass',
                    'ece_platt_null', 'ece_platt_null_p95', 'mean_pred_raw', 'mean_pred_platt']:
            rec[key] = {k: mean_sd([r[key][k] for r in rr]) for k in GROUPS}
            rec[key]['dark_minus_light'] = mean_sd([r[key]['dark'] - r[key]['light']
                                                    for r in rr])
        rec['seeds_ece_above_null_p95'] = {
            k: int(sum(r['ece_platt'][k] > r['ece_platt_null_p95'][k] for r in rr))
            for k in GROUPS}
        rec['observed_malignancy_rate'] = {k: float(yte[gte == k].mean()) for k in GROUPS}
        rec['calibration_in_the_large'] = {}
        for k in GROUPS:
            cl = [r['calibration_in_the_large'][k] for r in rr]
            rec['calibration_in_the_large'][k] = {
                'observed': cl[0]['observed'],
                'expected': mean_sd([c['expected'] for c in cl]),
                'observed_over_expected': mean_sd([c['observed_over_expected'] for c in cl]),
                'z': mean_sd([c['z'] for c in cl]),
                'seeds_abs_z_above_crit': int(sum(abs(c['z']) > Z_CRIT for c in cl))}
        dl = [r['dark_vs_light'] for r in rr]
        rec['dark_vs_light'] = {
            'diff_observed_over_expected': mean_sd([c['diff_observed_over_expected']
                                                    for c in dl]),
            'z': mean_sd([c['z'] for c in dl]),
            'seeds_z_above_crit': int(sum(c['z'] > Z_CRIT for c in dl))}
        # reliability curves pool the ten seeds' test predictions
        for tag, field in [('raw', '_pt'), ('platt', '_pt_cal')]:
            rec['reliability_%s' % tag] = {}
            for k in GROUPS:
                p = np.concatenate([r[field][gte == k] for r in rr])
                y = np.concatenate([yte[gte == k]] * len(rr))
                rec['reliability_%s' % tag][k] = reliability(p, y)[1]
        out['methods'][m] = rec

    # --- tables ---
    print('\nTest counts by tone: %s' % counts)
    print('\nOPERATING POINT at the shared threshold (90%% specificity on pooled validation '
          'negatives), mean over %d seeds' % len(FINAL_SEEDS))
    print('%-14s %8s %8s %8s   %8s %8s %8s' % ('method', 'sens_lt', 'sens_md', 'sens_dk',
                                              'spec_lt', 'spec_md', 'spec_dk'))
    for m in ORDER:
        s = out['methods'][m]['shared_threshold']
        print('%-14s %8.3f %8.3f %8.3f   %8.3f %8.3f %8.3f'
              % tuple([m] + [s[k]['sensitivity']['mean'] for k in GROUPS]
                      + [s[k]['specificity']['mean'] for k in GROUPS]))
    print('\nCALIBRATION  ECE by tone (equal-width bins): raw | after one tone-agnostic '
          'Platt recalibration | perfect-calibration baseline at the same sample size')
    print('%-14s %7s %7s %7s   %7s %7s %7s   %7s %7s %7s'
          % ('method', 'raw_lt', 'raw_md', 'raw_dk', 'cal_lt', 'cal_md', 'cal_dk',
             'nul_lt', 'nul_md', 'nul_dk'))
    for m in ORDER:
        r = out['methods'][m]
        print('%-14s %7.3f %7.3f %7.3f   %7.3f %7.3f %7.3f   %7.3f %7.3f %7.3f'
              % tuple([m] + [r['ece_raw'][k]['mean'] for k in GROUPS]
                      + [r['ece_platt'][k]['mean'] for k in GROUPS]
                      + [r['ece_platt_null'][k]['mean'] for k in GROUPS]))
    print('\nseeds (of %d) whose recalibrated ECE exceeds the 95th percentile of the '
          'perfect-calibration baseline' % len(FINAL_SEEDS))
    for m in ORDER:
        print('  %-14s %s' % (m, out['methods'][m]['seeds_ece_above_null_p95']))

    print('\nCALIBRATION IN THE LARGE after recalibration: observed / expected malignant cases '
          'and z (seeds with |z| > %.2f)' % Z_CRIT)
    print('%-14s %19s %19s %19s   %s' % ('method', 'light', 'mid', 'dark',
                                         'dark-light O/E, z (seeds z>crit)'))
    for m in ORDER:
        c = out['methods'][m]['calibration_in_the_large']
        dl = out['methods'][m]['dark_vs_light']
        print('%-14s %s   %+.3f, %+.2f (%d)' % (m, ' '.join(
            '%4.0f/%5.1f %+5.2f(%d)' % (c[k]['observed'], c[k]['expected']['mean'],
                                       c[k]['z']['mean'], c[k]['seeds_abs_z_above_crit'])
            for k in GROUPS), dl['diff_observed_over_expected']['mean'], dl['z']['mean'],
            dl['seeds_z_above_crit']))

    dest = dest or os.path.join(REPO, 'results/calibration_operating_point.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
