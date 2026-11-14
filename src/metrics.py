"""Scoring: accuracy by group, monotonicity and paired intervals.

Monotonicity is the fraction of image-concept interventions in which asserting a concept
does not move predicted malignancy against its clinical direction.
"""
import numpy as np
import torch

from src.concept_layer import GROUPS
from src.training import fwd

# Specificity at which sensitivity is reported.
SPEC = 0.90


# --- metrics ---
def auc_np(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mono_np(P, Cm, signs):
    if len(Cm) == 0:
        return float('nan')
    outs = []
    for i in range(Cm.shape[1]):
        if signs[i] == 0:
            continue
        hi, lo = Cm.clone(), Cm.clone()
        a, b = (1.0, 0.0) if signs[i] > 0 else (0.0, 1.0)
        hi[:, i], lo[:, i] = a, b
        outs.append((torch.sigmoid(fwd(hi, P)) >= torch.sigmoid(fwd(lo, P)) - 1e-6)
                    .float().detach().numpy())
    return float(np.mean(outs))


def probs(P, Cm):
    return torch.sigmoid(fwd(Cm, P)).detach().numpy()


def worst_auc(p, y, g):
    a = [auc_np(y[g == k], p[g == k]) for k in GROUPS]
    a = [v for v in a if not np.isnan(v)]
    return min(a) if a else float('nan')


def sens_global(p_val, y_val, p_te, y_te, g_te, spec=SPEC):
    thr = np.quantile(p_val[y_val == 0], spec)
    return {k: float((p_te[(g_te == k) & (y_te == 1)] > thr).mean()) for k in GROUPS}


def sens_pergroup(p_val, y_val, g_val, p_te, y_te, g_te, spec=SPEC):
    out = {}
    for k in GROUPS:
        mv = (g_val == k) & (y_val == 0)
        thr = np.quantile(p_val[mv], spec) if mv.sum() > 10 else np.quantile(p_val[y_val == 0], spec)
        out[k] = float((p_te[(g_te == k) & (y_te == 1)] > thr).mean())
    return out


def mean_of(runs, path):
    vals = []
    for r in runs:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def summarise(runs, extra=None):
    rec = dict(extra or {})
    rec['n_seeds'] = len(runs)
    for key in ['test_worst_auc', 'test_auc_gap', 'test_mono_gap', 'val_worst_auc', 'val_mono']:
        mu, sd = mean_of(runs, [key]); rec[key] = {'mean': mu, 'sd': sd}
    for blk in ['test_auc', 'test_mono', 'ddi_auc', 'ddi_mono',
                'sens@spec_global_thr', 'sens@spec_pergroup_thr']:
        rec[blk] = {}
        for k in GROUPS:
            mu, sd = mean_of(runs, [blk, k]); rec[blk][k] = {'mean': mu, 'sd': sd}
    return rec


def paired_ci(a, b, B=4000, seed=0):
    """Percentile bootstrap CI on the paired per-seed difference a-b."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return {'mean': float(d.mean()), 'lo': float(np.percentile(bs, 2.5)),
            'hi': float(np.percentile(bs, 97.5)),
            'n_positive': int((d > 0).sum()), 'n': int(len(d))}
