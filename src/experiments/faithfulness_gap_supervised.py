"""Primary endpoint: light-dark monotonicity gap on the supervised concept layer (Section 5.2).

Same protocol as faithfulness_gap_zeroshot.py, with the probe layer in place of the prompts;
analyse() is copied verbatim so that both layers use identical statistics. Probes are refit
inside each fold, and assert_no_leakage checks that no probe sees a test image.

Run:    python -m src.experiments.faithfulness_gap_supervised
Reads:  caches/fairness_cache/features_{fitz,ddi}.npz, results/fitz_rowmeta.parquet,
        data/skincon_fitz.csv, caches/ddi_metadata.csv
Writes: results/faithfulness_gap_supervised.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from src.concept_layer import (GROUPS, assert_no_leakage, concept_scores, fit_probes,
                           load_ddi, load_fitz, signs_for)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# same constants as faithfulness_gap_zeroshot.py, checked by check_protocol_match()
N_FOLDS, N_SEEDS, B = 5, 20, 4000
HID, ITERS, LR = 16, 1500, 0.05
N_PERM = 20000

ESTIMAND = (
    "E[ mono(light) - mono(dark) ] for a 2-layer SUPERVISED-SKINCON-CONCEPT -> malignancy "
    "head trained by this procedure on Fitzpatrick17k, expectation taken over training "
    "seeds and over the patient population, evaluated out-of-fold over all 16,012 images "
    "under 5-fold cross-validation with folds grouped by near-duplicate cluster. Reported "
    "as (a) a 95% CI on that mean, (b) a 95% predictive interval for one randomly seeded "
    "head, and (c) a permutation p-value. Monotonicity is scored under the declared "
    "CLINICAL sign table; the train-derived sign is scored as a sensitivity analysis. "
    "This is a separate endpoint from the zero-shot-concept one, which it does not "
    "supersede."
)


def check_protocol_match():
    """Assert that the protocol constants match faithfulness_gap_zeroshot.py."""
    src = open(os.path.join(HERE, 'faithfulness_gap_zeroshot.py')).read()
    want = {'N_FOLDS, N_SEEDS, B = 5, 20, 4000': (N_FOLDS, N_SEEDS, B),
            'HID, ITERS, LR = 16, 1500, 0.05': (HID, ITERS, LR),
            'N_PERM = 20000': (N_PERM,)}
    for line in want:
        assert line in src, 'protocol drift: faithfulness_gap_zeroshot.py no longer contains %r' % line
    print('protocol constants match faithfulness_gap_zeroshot.py exactly')


# --- model (as in faithfulness_gap_zeroshot.py) ---
def fwd(C, P):
    W1, b1, W2, b2 = P
    return (torch.relu(C @ W1 + b1) @ W2 + b2).squeeze(1)


def train_head(Xtr, ytr, seed):
    gg = torch.Generator().manual_seed(seed)
    P = [(torch.randn(Xtr.shape[1], HID, generator=gg) * 0.3).requires_grad_(True),
         torch.zeros(HID, requires_grad=True),
         (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True),
         torch.zeros(1, requires_grad=True)]
    opt = torch.optim.Adam(P, lr=LR)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    pw = torch.tensor(max(float((ytr == 0).sum()), 1.0) / max(float((ytr == 1).sum()), 1.0))
    for _ in range(ITERS):
        loss = bce(fwd(Xtr, P), ytr, pos_weight=pw, reduction='mean')
        opt.zero_grad(); loss.backward(); opt.step()
    return P


def mono_per_sample(P, C, signs):
    """Per-sample monotonicity over the signed concepts.

    Asserting a concept sets its score to 1 for sign +1 and to 0 for sign -1; concepts with
    sign 0 stay model inputs but are not scored.
    """
    inds = []
    for i in range(C.shape[1]):
        if signs[i] == 0:
            continue
        hi, lo = C.clone(), C.clone()
        a, b = (1.0, 0.0) if signs[i] > 0 else (0.0, 1.0)
        hi[:, i], lo[:, i] = a, b
        inds.append((torch.sigmoid(fwd(hi, P)) >= torch.sigmoid(fwd(lo, P)) - 1e-6)
                    .float().detach().numpy())
    return np.mean(inds, axis=0)


def auc_np(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --- statistics (as in faithfulness_gap_zeroshot.py) ---
def analyse(M, Pr, g, y, name):
    idx = {k: np.where(g == k)[0] for k in GROUPS}
    per_seed_mono = {k: M[:, idx[k]].mean(1) for k in GROUPS}
    gaps = per_seed_mono['light'] - per_seed_mono['dark']

    rng = np.random.default_rng(0)
    boot_mean = np.empty(B)
    for b in range(B):
        se = rng.integers(0, M.shape[0], M.shape[0])
        il = idx['light'][rng.integers(0, len(idx['light']), len(idx['light']))]
        idk = idx['dark'][rng.integers(0, len(idx['dark']), len(idx['dark']))]
        e = M[se].mean(0)
        boot_mean[b] = e[il].mean() - e[idk].mean()

    boot_pred = np.empty(B)
    for b in range(B):
        e = M[rng.integers(0, M.shape[0])]
        il = idx['light'][rng.integers(0, len(idx['light']), len(idx['light']))]
        idk = idx['dark'][rng.integers(0, len(idx['dark']), len(idx['dark']))]
        boot_pred[b] = e[il].mean() - e[idk].mean()

    ens = M.mean(0)
    pool = np.concatenate([idx['light'], idx['dark']])
    vals = ens[pool]
    nl = len(idx['light'])
    obs = vals[:nl].mean() - vals[nl:].mean()
    perm = np.empty(N_PERM)
    for b in range(N_PERM):
        v = rng.permutation(vals)
        perm[b] = v[:nl].mean() - v[nl:].mean()
    p_one = (int((perm >= obs).sum()) + 1) / (N_PERM + 1)
    p_two = (int((np.abs(perm) >= abs(obs)).sum()) + 1) / (N_PERM + 1)

    var_seed = float(gaps.var(ddof=1))
    var_total_pred = float(boot_pred.var(ddof=1))
    var_patient = max(var_total_pred - var_seed, 0.0)

    auc_gaps = np.array([auc_np(y[idx['light']], Pr[s][idx['light']])
                         - auc_np(y[idx['dark']], Pr[s][idx['dark']]) for s in range(M.shape[0])])
    ok = np.isfinite(auc_gaps) & np.isfinite(gaps)
    corr = float(np.corrcoef(auc_gaps[ok], gaps[ok])[0, 1]) if ok.sum() > 2 else float('nan')

    rec = {
        'n_by_group': {k: int(len(idx[k])) for k in GROUPS},
        'mono_mean': {k: float(per_seed_mono[k].mean()) for k in GROUPS},
        'mono_sd_over_seeds': {k: float(per_seed_mono[k].std(ddof=1)) for k in GROUPS},
        'gap_mean': float(gaps.mean()), 'gap_sd_over_seeds': float(gaps.std(ddof=1)),
        'gap_frac_positive': float((gaps > 0).mean()),
        'ci_on_mean_gap': {'lo': float(np.percentile(boot_mean, 2.5)),
                           'hi': float(np.percentile(boot_mean, 97.5)),
                           'excludes_zero': bool(np.percentile(boot_mean, 2.5) > 0)},
        'predictive_interval_single_head': {'lo': float(np.percentile(boot_pred, 2.5)),
                                            'hi': float(np.percentile(boot_pred, 97.5))},
        'permutation': {'observed': float(obs), 'n_perm': N_PERM,
                        'null_sd': float(perm.std()), 'null_max': float(perm.max()),
                        'p_one_sided': p_one, 'p_two_sided': p_two},
        'variance_decomposition': {'var_seed': var_seed, 'var_patient_resid': var_patient,
                                   'pct_seed': (100 * var_seed / var_total_pred
                                                if var_total_pred > 0 else float('nan'))},
        'auroc_mean': {k: float(np.mean([auc_np(y[idx[k]], Pr[s][idx[k]])
                                         for s in range(M.shape[0])])) for k in GROUPS},
        'auroc_gap_mean': float(np.nanmean(auc_gaps)),
        'corr_auroc_gap_vs_mono_gap': corr,
    }
    print('\n== %s ==' % name)
    print('  n: %s' % rec['n_by_group'])
    print('  mono  light %.4f  mid %.4f  dark %.4f'
          % tuple(rec['mono_mean'][k] for k in GROUPS))
    print('  auroc light %.4f  mid %.4f  dark %.4f  (gap %+.4f)'
          % (tuple(rec['auroc_mean'][k] for k in GROUPS) + (rec['auroc_gap_mean'],)))
    print('  gap   %.4f  (sd over seeds %.4f, +ve in %d/%d)'
          % (rec['gap_mean'], rec['gap_sd_over_seeds'],
             int(rec['gap_frac_positive'] * M.shape[0]), M.shape[0]))
    c = rec['ci_on_mean_gap']
    print('  (a) 95%% CI on MEAN gap        [%.4f, %.4f]  %s'
          % (c['lo'], c['hi'], 'EXCLUDES 0' if c['excludes_zero'] else 'contains 0'))
    q = rec['predictive_interval_single_head']
    print('  (b) 95%% predictive (one head) [%.4f, %.4f]' % (q['lo'], q['hi']))
    pp = rec['permutation']
    print('  (c) permutation p (1-sided) = %.2e   null sd %.4f' % (pp['p_one_sided'], pp['null_sd']))
    print('  variance: %.0f%% seed / %.0f%% patient'
          % (rec['variance_decomposition']['pct_seed'],
             100 - rec['variance_decomposition']['pct_seed']))
    print('  corr(AUROC gap, mono gap) over seeds = %+.3f' % rec['corr_auroc_gap_vs_mono_gap'])
    return rec


# --- run ---
def main():
    check_protocol_match()
    F, D = load_fitz(), load_ddi()
    n, names = len(F['y']), F['names']
    print('Fitzpatrick cohort: %d  %s' % (n, {k: int((F['g'] == k).sum()) for k in GROUPS}))
    print('concepts: %d supervised SkinCon | DDI join %s'
          % (len(names), 'OK' if D['join_ok'] else 'UNAVAILABLE (%s)' % D.get('join_reason')))

    sg_clin = signs_for(names, 'clinical')
    print('clinical signs: +1 x%d, -1 x%d, excluded x%d'
          % (int((sg_clin > 0).sum()), int((sg_clin < 0).sum()), int((sg_clin == 0).sum())))

    strata = np.array([GROUPS.index(t) * 2 + int(v) for t, v in zip(F['g'], F['y'])])
    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
                 .split(np.zeros(n), strata, F['cluster']))
    print('duplicate clusters: %d distinct -> leakage-aware folds' % len(np.unique(F['cluster'])))

    RULES = ['clinical', 'derived', 'derived14']
    M = {r: np.zeros((N_SEEDS, n), dtype=np.float32) for r in RULES}
    Pr = np.zeros((N_SEEDS, n), dtype=np.float32)
    M_ddi = {r: np.zeros((N_SEEDS, len(D['g'])), dtype=np.float32) for r in RULES}
    Pr_ddi = np.zeros((N_SEEDS, len(D['g'])), dtype=np.float32)
    derived_signs_by_fold = []

    t0 = time.time()
    for fi, (itr, ite) in enumerate(folds):
        assert_no_leakage(itr, ite, F['cluster'])
        probes = fit_probes(F['emb'], F['L'], F['has'], itr)
        S = concept_scores(probes, F['emb'])
        sg_der = signs_for(names, 'derived', y=F['y'], S=S, train_idx=itr)
        derived_signs_by_fold.append({names[j]: int(sg_der[j]) for j in range(len(names))})
        # derived signs, limited to the 14 concepts the clinical rule signs
        sg_der14 = np.where(sg_clin == 0, 0, sg_der)
        SG = {'clinical': sg_clin, 'derived': sg_der, 'derived14': sg_der14}

        Xtr = torch.tensor(S[itr], dtype=torch.float32)
        ytr = torch.tensor(F['y'][itr], dtype=torch.float32)
        Xte = torch.tensor(S[ite], dtype=torch.float32)
        for s in range(N_SEEDS):
            P = train_head(Xtr, ytr, s)
            for r in RULES:
                M[r][s, ite] = mono_per_sample(P, Xte, SG[r])
            Pr[s, ite] = torch.sigmoid(fwd(Xte, P)).detach().numpy()
        print('  fold %d/%d done (%.0fs)' % (fi + 1, N_FOLDS, time.time() - t0))

    # DDI is external: probes and heads are fit on all of Fitzpatrick17k
    print('training %d heads on full Fitzpatrick for the DDI external set ...' % N_SEEDS)
    probes_full = fit_probes(F['emb'], F['L'], F['has'], np.arange(n))
    S_full = concept_scores(probes_full, F['emb'])
    S_ddi = concept_scores(probes_full, D['emb'])
    sg_der_full = signs_for(names, 'derived', y=F['y'], S=S_full, train_idx=np.arange(n))
    SGf = {'clinical': sg_clin, 'derived': sg_der_full,
           'derived14': np.where(sg_clin == 0, 0, sg_der_full)}
    Xf = torch.tensor(S_full, dtype=torch.float32)
    yf = torch.tensor(F['y'], dtype=torch.float32)
    Xd = torch.tensor(S_ddi, dtype=torch.float32)
    for s in range(N_SEEDS):
        P = train_head(Xf, yf, s)
        for r in RULES:
            M_ddi[r][s] = mono_per_sample(P, Xd, SGf[r])
        Pr_ddi[s] = torch.sigmoid(fwd(Xd, P)).detach().numpy()
    print('training complete (%.0fs)' % (time.time() - t0))

    out = {'estimand': ESTIMAND,
           'bottleneck': {'kind': 'supervised SkinCon probes on frozen DermLIP embedding',
                          'n_concepts': len(names), 'concepts': names,
                          'clinical_signs': {names[j]: int(sg_clin[j]) for j in range(len(names))},
                          'derived_signs_by_fold': derived_signs_by_fold,
                          'probes_fit_inside_each_fold': True},
           'duplicate_aware_folds': True, 'sets': {}}
    for r in RULES:
        tag = {'clinical': 'PRIMARY', 'derived': 'declared sensitivity',
               'derived14': 'POST-HOC decomposition, not declared'}[r]
        out['sets']['fitz_oof__%s' % r] = analyse(
            M[r], Pr, F['g'], F['y'], 'Fitzpatrick17k out-of-fold, ALL images [%s sign, %s]' % (r, tag))
        out['sets']['ddi_external__%s' % r] = analyse(
            M_ddi[r], Pr_ddi, D['g'], D['y'], 'DDI external [%s sign, %s]' % (r, tag))

    dest = os.path.join(REPO, 'results', 'faithfulness_gap_supervised.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved', os.path.relpath(dest, REPO))


if __name__ == '__main__':
    main()
