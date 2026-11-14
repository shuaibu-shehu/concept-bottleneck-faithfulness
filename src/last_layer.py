"""Last-layer retraining baselines: DFR and AFR.

DFR (Kirichenko et al., 2023) refits the last layer on group-balanced subsets of the
validation split; AFR (Qiu et al., 2023) refits it with weights exp(-gamma * p_ERM(y|x)) and
no group labels. The caches load at import, because other modules import the fitting helpers.

Run:    python -m src.last_layer
Writes: results/last_layer.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
GROUPS = ['light', 'mid', 'dark']

# same constants as mitigations_zeroshot.py, for an identical ERM row
HID, ITERS, LR = 16, 1500, 0.05
FINAL_SEEDS = list(range(10))
SPEC = 0.90

# DFR / AFR protocol constants
DFR_C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]     # inverse l1 strength
AFR_GAMMA_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
N_SUBSETS = 10                                    # balanced refits averaged
ERM_TOLERANCE = 1e-6                              # regression check vs mitigations_zeroshot.json


# --- data ---
z = np.load(os.path.join(REPO, 'caches/fairness_cache/features_fitz.npz'), allow_pickle=True)
Cn = z['concept_scores'].astype(np.float32)
gall = z['group'].astype('U8')
ok = np.isfinite(Cn).all(1) & np.isin(gall, GROUPS)
C = torch.tensor(Cn[ok])
Y = torch.tensor(z['malignant'][ok].astype(np.float32))
G = gall[ok]
SP = z['split'].astype('U8')[ok]

ddi = np.load(os.path.join(REPO, 'caches/fairness_cache/features_ddi.npz'), allow_pickle=True)
Cd = ddi['concept_scores'].astype(np.float32)
gd = ddi['group'].astype('U8')
okd = np.isfinite(Cd).all(1) & np.isin(gd, GROUPS)
Cddi = torch.tensor(Cd[okd])
Yddi = torch.tensor(ddi['malignant'][okd].astype(np.float32))
Gddi = gd[okd]

tr, va, te = SP == 'train', SP == 'val', SP == 'test'
Xtr, ytr, gtr = C[tr], Y[tr], G[tr]
yva, gva = Y[va].numpy(), G[va]
yte, gte = Y[te].numpy(), G[te]
yddi_np = Yddi.numpy()
Cva = C[va]

print('train %d  val %d  test %d  | ddi %d' % (tr.sum(), va.sum(), te.sum(), len(Gddi)))
cells = {'%s/%d' % (g, c): int(((gva == g) & (yva == c)).sum())
         for g in GROUPS for c in (0, 1)}
print('val (tone x malignancy) cells:', cells)
print('binding cell for group-balanced refit: %d -> %d refit rows'
      % (min(cells.values()), 6 * min(cells.values())))


# --- model ---
def fwd(Cm, P):
    W1, b1, W2, b2 = P
    return (torch.relu(Cm @ W1 + b1) @ W2 + b2).squeeze(1)


def hidden(Cm, P):
    """The 16-d activation DFR treats as the feature layer."""
    W1, b1, _, _ = P
    return torch.relu(Cm @ W1 + b1).detach().numpy()


def train_erm(seed):
    """The 'erm' branch of mitigations_zeroshot.train(), reproduced exactly."""
    gg = torch.Generator().manual_seed(seed)
    P = [(torch.randn(Xtr.shape[1], HID, generator=gg) * 0.3).requires_grad_(True),
         torch.zeros(HID, requires_grad=True),
         (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True),
         torch.zeros(1, requires_grad=True)]
    opt = torch.optim.Adam(P, lr=LR, weight_decay=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    pw = torch.tensor(max(float((ytr == 0).sum()), 1.0) / max(float((ytr == 1).sum()), 1.0))
    for _ in range(ITERS):
        loss = bce(fwd(Xtr, P), ytr, pos_weight=pw, reduction='none').mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return [p.detach() for p in P]


# --- metrics (as in mitigations_zeroshot.py, which runs a long sweep on import) ---
def auc_np(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mono_np(P, Cm):
    if len(Cm) == 0:
        return float('nan')
    outs = []
    for i in range(Cm.shape[1]):
        v1, v0 = Cm.clone(), Cm.clone()
        v1[:, i], v0[:, i] = 1.0, 0.0
        outs.append((torch.sigmoid(fwd(v1, P)) >= torch.sigmoid(fwd(v0, P)) - 1e-6)
                    .float().detach().numpy())
    return float(np.mean(outs))


def probs(P, Cm):
    return torch.sigmoid(fwd(Cm, P)).detach().numpy()


def worst_auc(p, y, g):
    a = [auc_np(y[g == k], p[g == k]) for k in GROUPS]
    a = [v for v in a if not np.isnan(v)]
    return min(a) if a else float('nan')


def sens_at_spec_global(p_val, y_val, p_te, y_te, g_te, spec=SPEC):
    thr = np.quantile(p_val[y_val == 0], spec)
    return {k: float((p_te[(g_te == k) & (y_te == 1)] > thr).mean()) for k in GROUPS}


def evaluate(P):
    pv, pt, pd_ = probs(P, Cva), probs(P, C[te]), probs(P, Cddi)
    rec = {'val_worst_auc': worst_auc(pv, yva, gva),
           'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
           'test_mono': {k: mono_np(P, C[te][torch.tensor(gte == k)]) for k in GROUPS},
           'ddi_auc': {k: auc_np(yddi_np[Gddi == k], pd_[Gddi == k]) for k in GROUPS},
           'ddi_mono': {k: mono_np(P, Cddi[torch.tensor(Gddi == k)]) for k in GROUPS},
           'sens@spec_global_thr': sens_at_spec_global(pv, yva, pt, yte, gte)}
    rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
    rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
    rec['test_mono_gap'] = rec['test_mono']['light'] - rec['test_mono']['dark']
    rec['ddi_mono_gap'] = rec['ddi_mono']['light'] - rec['ddi_mono']['dark']
    return rec


def mean_of(runs, path):
    vals = []
    for r in runs:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


# --- DFR and AFR ---
def balanced_index(idx, g, y, rng, by_class=True):
    """DFR footnote 3: keep all of the smallest cell, subsample the rest to match."""
    keys = ([(k, c) for k in GROUPS for c in (0, 1)] if by_class else [(k, None) for k in GROUPS])
    pools = []
    for k, c in keys:
        m = (g[idx] == k) if c is None else ((g[idx] == k) & (y[idx] == c))
        pools.append(idx[m])
    n = min(len(p) for p in pools)
    if n == 0:
        raise ValueError('empty cell in balanced refit')
    return np.concatenate([rng.choice(p, n, replace=False) for p in pools]), n


def fold_into_head(w, b, mu, sd):
    """Fold the feature standardisation into the weights, keeping a plain two-layer head."""
    w2 = (w / sd).astype(np.float32)
    b2 = float(b - np.sum(w * mu / sd))
    return w2, b2


def fit_last_layer(H, yv, C_or_gamma, mode, sample_w=None):
    """One l1 (DFR) or l2 (AFR) logistic fit on standardized features."""
    mu, sd = H.mean(0), H.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Hs = (H - mu) / sd
    # random_state makes the liblinear fit deterministic
    if mode == 'dfr':
        lr = LogisticRegression(penalty='l1', C=C_or_gamma, solver='liblinear',
                                max_iter=2000, random_state=0)
    else:
        lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000,
                                random_state=0)
    if len(np.unique(yv)) < 2:
        raise ValueError('single-class refit set')
    lr.fit(Hs, yv, sample_weight=sample_w)
    return fold_into_head(lr.coef_[0], float(lr.intercept_[0]), mu, sd)


def make_head(P_erm, w2, b2):
    return [P_erm[0], P_erm[1],
            torch.tensor(w2.reshape(-1, 1)), torch.tensor([b2], dtype=torch.float32)]


def dfr(P_erm, seed, by_class=True):
    """DFR^Val_Tr. Returns a new head and the selected C."""
    rng = np.random.default_rng(1000 + seed)
    H = hidden(Cva, P_erm)
    all_idx = np.arange(len(yva))

    # tune C: fit on one half of validation, score worst-group AUROC on the other
    perm = rng.permutation(all_idx)
    A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]
    scores = {}
    for c in DFR_C_GRID:
        try:
            bi, _ = balanced_index(A, gva, yva, rng, by_class)
            sw = None
            if not by_class:
                cw = {0: 1.0, 1: float((yva[bi] == 0).sum()) / max((yva[bi] == 1).sum(), 1)}
                sw = np.array([cw[int(v)] for v in yva[bi]])
            w2, b2 = fit_last_layer(H[bi], yva[bi], c, 'dfr', sw)
        except ValueError:
            scores[c] = float('nan')
            continue
        pB = probs(make_head(P_erm, w2, b2), Cva[torch.tensor(B)])
        scores[c] = worst_auc(pB, yva[B], gva[B])
    best_c = max(DFR_C_GRID, key=lambda c: (-np.inf if np.isnan(scores[c]) else scores[c]))

    # refit on all of validation at the best C, averaging N_SUBSETS balanced draws
    W, Bs, ns = [], [], []
    for _ in range(N_SUBSETS):
        bi, n = balanced_index(all_idx, gva, yva, rng, by_class)
        sw = None
        if not by_class:
            cw = {0: 1.0, 1: float((yva[bi] == 0).sum()) / max((yva[bi] == 1).sum(), 1)}
            sw = np.array([cw[int(v)] for v in yva[bi]])
        w2, b2 = fit_last_layer(H[bi], yva[bi], best_c, 'dfr', sw)
        W.append(w2); Bs.append(b2); ns.append(n)
    return (make_head(P_erm, np.mean(W, 0), float(np.mean(Bs))),
            {'selected_C': best_c, 'val_worst_auc_by_C': scores,
             'cell_size': int(ns[0]), 'refit_rows': int(ns[0] * (6 if by_class else 3))})


def afr(P_erm, seed):
    """AFR: refit with weights exp(-gamma * p_ERM(true class)); group labels only tune gamma."""
    rng = np.random.default_rng(2000 + seed)
    H = hidden(Cva, P_erm)
    p1 = probs(P_erm, Cva)
    p_true = np.where(yva == 1, p1, 1.0 - p1)
    all_idx = np.arange(len(yva))
    perm = rng.permutation(all_idx)
    A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]

    def weights(idx, gamma):
        w = np.exp(-gamma * p_true[idx])
        return w / w.mean()

    scores = {}
    for gm in AFR_GAMMA_GRID:
        w2, b2 = fit_last_layer(H[A], yva[A], None, 'afr', weights(A, gm))
        pB = probs(make_head(P_erm, w2, b2), Cva[torch.tensor(B)])
        scores[gm] = worst_auc(pB, yva[B], gva[B])
    best_g = max(AFR_GAMMA_GRID, key=lambda g: (-np.inf if np.isnan(scores[g]) else scores[g]))
    w2, b2 = fit_last_layer(H, yva, None, 'afr', weights(all_idx, best_g))
    return make_head(P_erm, w2, b2), {'selected_gamma': best_g,
                                      'val_worst_auc_by_gamma': scores}


# --- run ---
def main():
    t0 = time.time()
    print('\ntraining %d ERM base heads ...' % len(FINAL_SEEDS))
    bases = [train_erm(s) for s in FINAL_SEEDS]

    methods = {'erm': [], 'dfr': [], 'dfr_tone_only': [], 'afr': []}
    aux = {'dfr': [], 'dfr_tone_only': [], 'afr': []}
    for i, (s, P) in enumerate(zip(FINAL_SEEDS, bases)):
        methods['erm'].append(evaluate(P))
        for name, fn in (('dfr', lambda p, sd: dfr(p, sd, True)),
                         ('dfr_tone_only', lambda p, sd: dfr(p, sd, False)),
                         ('afr', lambda p, sd: afr(p, sd))):
            head, info = fn(P, s)
            methods[name].append(evaluate(head))
            aux[name].append(info)
        print('  seed %d done [%.0fs]' % (s, time.time() - t0))

    keys = [('worst_auc', ['test_worst_auc']), ('auc_gap', ['test_auc_gap']),
            ('mono_gap', ['test_mono_gap']), ('ddi_mono_gap', ['ddi_mono_gap'])]
    summary = {}
    for name, runs in methods.items():
        rec = {}
        for label, path in keys:
            rec[label], rec[label + '_sd'] = mean_of(runs, path)
        for g in GROUPS:
            rec['mono_' + g] = mean_of(runs, ['test_mono', g])[0]
            rec['auc_' + g] = mean_of(runs, ['test_auc', g])[0]
            rec['sens@90_' + g] = mean_of(runs, ['sens@spec_global_thr', g])[0]
        summary[name] = rec

    # regression check against mitigations_zeroshot.json
    check = {'ran': False}
    mj = os.path.join(REPO, 'results', 'mitigations_zeroshot.json')
    if os.path.exists(mj):
        prev = json.load(open(mj))
        try:
            ref = prev['methods']['erm']
            pairs = {'worst_auc': 'test_worst_auc', 'auc_gap': 'test_auc_gap',
                     'mono_gap': 'test_mono_gap'}
            diffs = {k: abs(summary['erm'][k] - ref[j]['mean']) for k, j in pairs.items()}
            for g in GROUPS:
                diffs['mono_' + g] = abs(summary['erm']['mono_' + g]
                                         - ref['test_mono'][g]['mean'])
            check = {'ran': True, 'abs_diff_vs_mitigations_json': diffs,
                     'passed': all(v < ERM_TOLERANCE for v in diffs.values())}
        except (KeyError, TypeError) as e:
            check = {'ran': False, 'reason': 'mitigations_zeroshot.json schema mismatch: %r' % (e,)}
    print('\nERM regression check vs mitigations_zeroshot.json:', check)

    # pre-specified test, evaluated in code
    e, d = summary['erm'], summary['dfr']
    closure = 1.0 - abs(d['mono_gap']) / abs(e['mono_gap']) if e['mono_gap'] else float('nan')
    verdict = {
        'rule': ('claim "no accuracy-directed method repairs faithfulness" is FALSIFIED '
                 'iff gap closes >50% AND dark monotonicity rises above ERM'),
        'gap_closure_fraction': closure,
        'condition_a_gap_closes_over_50pct': bool(closure > 0.5),
        'dark_mono_erm': e['mono_dark'], 'dark_mono_dfr': d['mono_dark'],
        'condition_b_dark_mono_rises': bool(d['mono_dark'] > e['mono_dark']),
    }
    verdict['claim_falsified'] = bool(verdict['condition_a_gap_closes_over_50pct']
                                      and verdict['condition_b_dark_mono_rises'])
    verdict['levelling_down'] = bool(verdict['condition_a_gap_closes_over_50pct']
                                     and not verdict['condition_b_dark_mono_rises'])

    out = {'summary': summary, 'per_seed': methods,
           'method_details': aux, 'erm_regression_check': check,
           'predeclared_verdict': verdict,
           'val_cells': cells}
    path = os.path.join(REPO, 'results', 'last_layer.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)

    print('\n%-16s %8s %8s %8s %8s %8s %8s' %
          ('method', 'worstAUC', 'aucGap', 'monoL', 'monoD', 'monoGap', 'sens90D'))
    for name in ['erm', 'dfr', 'dfr_tone_only', 'afr']:
        r = summary[name]
        print('%-16s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' %
              (name, r['worst_auc'], r['auc_gap'], r['mono_light'], r['mono_dark'],
               r['mono_gap'], r['sens@90_dark']))
    print('\nVERDICT:', json.dumps(verdict, indent=1))
    print('\nwrote', path, '[%.0fs]' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
