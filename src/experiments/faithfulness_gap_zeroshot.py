"""Primary endpoint protocol on the zero-shot checklist layer (Section 5.2).

Run:    python -m src.experiments.faithfulness_gap_zeroshot
Reads:  caches/fairness_cache/features_{fitz,ddi}.npz, results/fitz_rowmeta.parquet
Writes: results/faithfulness_gap_zeroshot.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
GROUPS = ['light', 'mid', 'dark']

N_FOLDS, N_SEEDS, B = 5, 20, 4000
HID, ITERS, LR = 16, 1500, 0.05
N_PERM = 20000

ESTIMAND = (
    "E[ mono(light) - mono(dark) ] for a 2-layer concept->malignancy head trained "
    "by this procedure on Fitzpatrick17k, expectation taken over training seeds and "
    "over the patient population. Reported as (a) a 95% CI on that mean and "
    "(b) a 95% predictive interval for one randomly seeded head. Declared before "
    "estimation; not revised after seeing the result."
)


# --- data ---
def load(tag):
    z = np.load(os.path.join(REPO, 'caches/fairness_cache/features_%s.npz' % tag),
                allow_pickle=True)
    C = z['concept_scores'].astype(np.float32)
    g = z['group'].astype('U8')
    ok = np.isfinite(C).all(1) & np.isin(g, GROUPS)
    return {'C': torch.tensor(C[ok]), 'y': torch.tensor(z['malignant'][ok].astype(np.float32)),
            'g': g[ok], 'ok': ok}


fitz, ddi = load('fitz'), load('ddi')
n = len(fitz['g'])
print('Fitzpatrick cohort: %d  %s' % (n, {k: int((fitz['g'] == k).sum()) for k in GROUPS}))

rowmeta_path = os.path.join(REPO, 'results', 'fitz_rowmeta.parquet')
if os.path.exists(rowmeta_path):
    import pandas as pd
    clusters = pd.read_parquet(rowmeta_path)['cluster'].values
    assert len(clusters) == n, 'rowmeta rows (%d) != cohort (%d)' % (len(clusters), n)
    print('duplicate clusters loaded: %d distinct -> leakage-aware folds'
          % len(np.unique(clusters)))
    USE_GROUPS = True
else:
    clusters = np.arange(n)
    print('WARNING: results/fitz_rowmeta.parquet missing -- folds are NOT duplicate-aware.')
    USE_GROUPS = False

strata = np.array([GROUPS.index(g) * 2 + int(v) for g, v in zip(fitz['g'], fitz['y'].numpy())])
splitter = (StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
            if USE_GROUPS else StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0))
folds = list(splitter.split(np.zeros(n), strata, clusters) if USE_GROUPS
             else splitter.split(np.zeros(n), strata))


# --- model ---
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


def mono_per_sample(P, C):
    inds = []
    for i in range(C.shape[1]):
        v1, v0 = C.clone(), C.clone()
        v1[:, i], v0[:, i] = 1.0, 0.0
        inds.append((torch.sigmoid(fwd(v1, P)) >= torch.sigmoid(fwd(v0, P)) - 1e-6)
                    .float().detach().numpy())
    return np.mean(inds, axis=0)


def auc_np(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --- out-of-fold run ---
M = np.zeros((N_SEEDS, n), dtype=np.float32)       # per-seed, per-sample monotonicity
Pr = np.zeros((N_SEEDS, n), dtype=np.float32)      # per-seed, per-sample P(malignant)
M_ddi = np.zeros((N_SEEDS, len(ddi['g'])), dtype=np.float32)
Pr_ddi = np.zeros((N_SEEDS, len(ddi['g'])), dtype=np.float32)

t0 = time.time()
for fi, (itr, ite) in enumerate(folds):
    Xtr, ytr = fitz['C'][itr], fitz['y'][itr]
    for s in range(N_SEEDS):
        P = train_head(Xtr, ytr, s)
        M[s, ite] = mono_per_sample(P, fitz['C'][ite])
        Pr[s, ite] = torch.sigmoid(fwd(fitz['C'][ite], P)).detach().numpy()
    print('  fold %d/%d done (%.0fs)' % (fi + 1, N_FOLDS, time.time() - t0))

# DDI is external: heads are trained on all of Fitzpatrick17k and evaluated on all of DDI
print('training %d heads on full Fitzpatrick for the DDI external set ...' % N_SEEDS)
for s in range(N_SEEDS):
    P = train_head(fitz['C'], fitz['y'], s)
    M_ddi[s] = mono_per_sample(P, ddi['C'])
    Pr_ddi[s] = torch.sigmoid(fwd(ddi['C'], P)).detach().numpy()
print('training complete (%.0fs)' % (time.time() - t0))


# --- statistics ---
def analyse(M, Pr, g, y, name):
    idx = {k: np.where(g == k)[0] for k in GROUPS}
    per_seed_mono = {k: M[:, idx[k]].mean(1) for k in GROUPS}
    gaps = per_seed_mono['light'] - per_seed_mono['dark']

    # (a) CI on the mean gap: resample seeds and patients
    rng = np.random.default_rng(0)
    boot_mean = np.empty(B)
    for b in range(B):
        se = rng.integers(0, M.shape[0], M.shape[0])
        il = idx['light'][rng.integers(0, len(idx['light']), len(idx['light']))]
        idk = idx['dark'][rng.integers(0, len(idx['dark']), len(idx['dark']))]
        e = M[se].mean(0)
        boot_mean[b] = e[il].mean() - e[idk].mean()

    # (b) predictive interval for one head: one seed, patients resampled
    boot_pred = np.empty(B)
    for b in range(B):
        e = M[rng.integers(0, M.shape[0])]
        il = idx['light'][rng.integers(0, len(idx['light']), len(idx['light']))]
        idk = idx['dark'][rng.integers(0, len(idx['dark']), len(idx['dark']))]
        boot_pred[b] = e[il].mean() - e[idk].mean()

    # (c) permutation null: shuffle tone among the light and dark images
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

    # variance decomposition of the gap
    var_seed = float(gaps.var(ddof=1))
    var_total_pred = float(boot_pred.var(ddof=1))
    var_patient = max(var_total_pred - var_seed, 0.0)

    # per-seed correlation of the AUROC gap with the monotonicity gap
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
        'auroc_gap_mean': float(np.nanmean(auc_gaps)),
        'corr_auroc_gap_vs_mono_gap': corr,
    }
    print('\n== %s ==' % name)
    print('  n: %s' % rec['n_by_group'])
    print('  mono  light %.4f  mid %.4f  dark %.4f'
          % tuple(rec['mono_mean'][k] for k in GROUPS))
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
    print('  AUROC gap %.4f ; corr(AUROC gap, mono gap) over seeds = %+.3f'
          % (rec['auroc_gap_mean'], rec['corr_auroc_gap_vs_mono_gap']))
    return rec


out = {'estimand': ESTIMAND,
       'duplicate_aware_folds': bool(USE_GROUPS), 'sets': {}}
out['sets']['fitz_oof'] = analyse(M, Pr, fitz['g'], fitz['y'].numpy(), 'Fitzpatrick17k (out-of-fold, ALL images)')
out['sets']['ddi_external'] = analyse(M_ddi, Pr_ddi, ddi['g'], ddi['y'].numpy(), 'DDI (external)')

dest = os.path.join(REPO, 'results', 'faithfulness_gap_zeroshot.json')
json.dump(out, open(dest, 'w'), indent=2)
print('\nsaved', dest)
