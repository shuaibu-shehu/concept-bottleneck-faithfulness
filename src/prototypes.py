"""Diverse Prototypical Ensembles (DPE; To et al., 2025), an accuracy baseline.

Replaces the classification layer with an ensemble of prototype classifiers, fitted on
class-balanced bootstraps of the validation split. With 16-dimensional features it uses
Adam at batch 64 for 20 epochs (the authors' repository defaults) and selects (tau, alpha)
on validation.

Run:    python -m src.prototypes
Writes: results/prototypes.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from src import last_layer as base

REPO, GROUPS, FINAL_SEEDS = base.REPO, base.GROUPS, base.FINAL_SEEDS

N_MEMBERS = 15                    # paper: saturates by 15; repo default num_stages 16
# tau grid: the authors' stated range, plus a low anchor
TAU_GRID = [10.0, 20.0, 30.0, 40.0]
# alpha grid: the authors' range rescaled from 512-2048 to 16 dimensions; 0 tests
# bootstrap-only diversity
ALPHA_GRID = [0.0, 1e1, 1e2, 1e3, 1e4]
EPOCHS, BATCH, LR = 20, 64, 1e-3   # repo Stage-1+ defaults


# --- evaluation for a head given as a callable (DPE is not a two-layer head) ---
def g_mono(pf, Cm):
    if len(Cm) == 0:
        return float('nan')
    outs = []
    for i in range(Cm.shape[1]):
        v1, v0 = Cm.clone(), Cm.clone()
        v1[:, i], v0[:, i] = 1.0, 0.0
        outs.append((pf(v1) >= pf(v0) - 1e-6).astype(float))
    return float(np.mean(outs))


def g_evaluate(pf):
    pv, pt, pd_ = pf(base.Cva), pf(base.C[base.te]), pf(base.Cddi)
    yte, gte, gva, yva = base.yte, base.gte, base.gva, base.yva
    rec = {'val_worst_auc': base.worst_auc(pv, yva, gva),
           'test_auc': {k: base.auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
           'test_mono': {k: g_mono(pf, base.C[base.te][torch.tensor(gte == k)])
                         for k in GROUPS},
           'ddi_auc': {k: base.auc_np(base.yddi_np[base.Gddi == k], pd_[base.Gddi == k])
                       for k in GROUPS},
           'ddi_mono': {k: g_mono(pf, base.Cddi[torch.tensor(base.Gddi == k)])
                        for k in GROUPS},
           'sens@spec_global_thr': base.sens_at_spec_global(pv, yva, pt, yte, gte)}
    rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
    rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
    rec['test_mono_gap'] = rec['test_mono']['light'] - rec['test_mono']['dark']
    rec['ddi_mono_gap'] = rec['ddi_mono']['light'] - rec['ddi_mono']['dark']
    return rec


# --- the method ---
def class_balanced(idx, y, rng):
    """Class-balanced bootstrap subset for one member; no group labels are used."""
    pools = [idx[y[idx] == c] for c in (0, 1)]
    n = min(len(p) for p in pools)
    return np.concatenate([rng.choice(p, n, replace=True) for p in pools])


class DPE:
    """Sequentially grown ensemble of 2-prototype classifiers over the 16-d layer."""

    def __init__(self, P_erm, tau, alpha):
        self.P, self.tau, self.alpha = P_erm, tau, alpha
        self.protos, self.scales = [], []

    def _feat(self, Cm):
        h = torch.relu(Cm @ self.P[0] + self.P[1])
        return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _dist(x, p, ds):
        """IsoMax+ distance: |d_s| * || x_hat - p_hat ||_2, BOTH sides normalized."""
        ph = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return ds.abs() * torch.cdist(x, ph)

    def fit_member(self, H, y, seed):
        g = torch.Generator().manual_seed(seed)
        p = (torch.randn(2, H.shape[1], generator=g) * 0.1).requires_grad_(True)
        ds = torch.tensor(1.0, requires_grad=True)
        opt = torch.optim.Adam([p, ds], lr=LR)
        prev = torch.stack(self.protos) if self.protos else None   # (n_prev, 2, d)
        n = len(H)
        for _ in range(EPOCHS):
            order = torch.randperm(n, generator=g)
            for b in range(0, n, BATCH):
                idx = order[b:b + BATCH]
                # IsoMax: the entropic scale multiplies the distance in the loss only
                loss = torch.nn.functional.cross_entropy(
                    -self.tau * self._dist(H[idx], p, ds), y[idx])
                if prev is not None:
                    # penalise cosine similarity between prototypes, since their norms do not
                    # enter the distance
                    ph = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    pv = prev / prev.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    ips = sum((pv[:, k, :] @ ph[k]).abs().sum() for k in (0, 1))
                    loss = loss + self.alpha * ips / (prev.shape[0] * H.shape[1])
                opt.zero_grad(); loss.backward(); opt.step()
        self.protos.append(p.detach())
        self.scales.append(ds.detach())

    def prob_fn(self):
        """Average the members' probabilities, each keeping tau: P(y|x) = softmax(-tau * D)."""
        def pf(Cm):
            x = self._feat(Cm)
            ps = [torch.softmax(-self.tau * self._dist(x, p, s), dim=1)[:, 1]
                  for p, s in zip(self.protos, self.scales)]
            return torch.stack(ps).mean(0).detach().numpy()
        return pf


def run_dpe(P_erm, seed, tau, alpha, idx_pool, rng):
    H_all = torch.tensor(base.hidden(base.Cva, P_erm))
    H_all = H_all / H_all.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    m = DPE(P_erm, tau, alpha)
    for j in range(N_MEMBERS):
        bi = class_balanced(idx_pool, base.yva, rng)
        m.fit_member(H_all[bi], torch.tensor(base.yva[bi], dtype=torch.long),
                     seed * 100 + j)
    return m


def fit_and_select(P_erm, seed):
    rng = np.random.default_rng(3000 + seed)
    all_idx = np.arange(len(base.yva))
    perm = rng.permutation(all_idx)
    A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]

    scores = {}
    for tau in TAU_GRID:
        for alpha in ALPHA_GRID:
            m = run_dpe(P_erm, seed, tau, alpha, A, np.random.default_rng(seed))
            pB = m.prob_fn()(base.Cva[torch.tensor(B)])
            scores['%g|%g' % (tau, alpha)] = base.worst_auc(pB, base.yva[B], base.gva[B])
    best = max(scores, key=lambda k: (-np.inf if np.isnan(scores[k]) else scores[k]))
    tau, alpha = (float(v) for v in best.split('|'))
    final = run_dpe(P_erm, seed, tau, alpha, all_idx, np.random.default_rng(seed + 7))
    return final, {'selected_tau': tau, 'selected_alpha': alpha,
                   'val_worst_auc_by_config': scores,
                   'refit_rows_per_member': int(2 * min((base.yva == 0).sum(),
                                                        (base.yva == 1).sum()))}


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    h = {3: 4.303, 10: 2.262}[n] * sd / np.sqrt(n)
    return {'mean': m, 'ci95': [m - h, m + h], 'excludes_zero': bool((m - h) * (m + h) > 0)}


def main():
    t0 = time.time()
    print('training %d ERM base heads ...' % len(FINAL_SEEDS))
    bases = [base.train_erm(s) for s in FINAL_SEEDS]

    # self-check: the generic evaluator must reproduce base.evaluate on a linear head
    ref, gen = base.evaluate(bases[0]), g_evaluate(lambda Cm: base.probs(bases[0], Cm))
    diffs = {k: abs(ref[k] - gen[k]) for k in ('test_worst_auc', 'test_auc_gap',
                                               'test_mono_gap')}
    diffs.update({'mono_' + g: abs(ref['test_mono'][g] - gen['test_mono'][g])
                  for g in GROUPS})
    selfcheck = {'max_abs_diff': max(diffs.values()), 'per_quantity': diffs,
                 'tolerance': 1e-6,
                 'passed': max(diffs.values()) < 1e-6}
    print('self-check (generic vs base.evaluate on a linear head): max diff %.2e -> %s'
          % (selfcheck['max_abs_diff'], 'PASS' if selfcheck['passed'] else 'FAIL'))

    runs, aux = [], []
    for s, P in zip(FINAL_SEEDS, bases):
        m, info = fit_and_select(P, s)
        runs.append(g_evaluate(m.prob_fn()))
        aux.append(info)
        print('  seed %d done  tau=%g alpha=%g  [%.0fs]'
              % (s, info['selected_tau'], info['selected_alpha'], time.time() - t0))

    prev = json.load(open(os.path.join(REPO, 'results', 'last_layer.json')))
    erm, dfr = prev['per_seed']['erm'], prev['per_seed']['dfr']

    def col(rs, path):
        out = []
        for r in rs:
            v = r
            for p in path:
                v = v[p]
            out.append(v)
        return out

    P_ = {'mono_gap_vs_erm': paired(col(runs, ['test_mono_gap']),
                                    col(erm, ['test_mono_gap'])),
          'mono_dark_vs_erm': paired(col(runs, ['test_mono', 'dark']),
                                     col(erm, ['test_mono', 'dark'])),
          'mono_light_vs_erm': paired(col(runs, ['test_mono', 'light']),
                                      col(erm, ['test_mono', 'light'])),
          'worst_auc_vs_erm': paired(col(runs, ['test_worst_auc']),
                                     col(erm, ['test_worst_auc'])),
          'worst_auc_vs_dfr': paired(col(runs, ['test_worst_auc']),
                                     col(dfr, ['test_worst_auc'])),
          'sens90_dark_vs_erm': paired(col(runs, ['sens@spec_global_thr', 'dark']),
                                       col(erm, ['sens@spec_global_thr', 'dark']))}
    a = P_['mono_gap_vs_erm']['mean'] < 0 and P_['mono_gap_vs_erm']['excludes_zero']
    b = P_['mono_dark_vs_erm']['mean'] > 0 and P_['mono_dark_vs_erm']['excludes_zero']
    verdict = {
        'rule': 'claim "no accuracy-directed method repairs faithfulness" is FALSIFIED '
                'iff (a) paired mono-gap vs ERM is negative with CI excluding 0 AND '
                '(b) paired dark monotonicity vs ERM is positive with CI excluding 0',
        'condition_a_gap_closes': bool(a), 'condition_b_dark_rises': bool(b),
        'claim_falsified': bool(a and b),
        'levelling_down': bool(a and not b),
        'dark_mono_rose_in_n_seeds': int(sum(
            1 for x, y in zip(col(runs, ['test_mono', 'dark']),
                              col(erm, ['test_mono', 'dark'])) if x > y)),
    }

    summary = {}
    for label, path in (('worst_auc', ['test_worst_auc']), ('auc_gap', ['test_auc_gap']),
                        ('mono_gap', ['test_mono_gap']), ('ddi_mono_gap', ['ddi_mono_gap'])):
        summary[label] = float(np.nanmean(col(runs, path)))
        summary[label + '_sd'] = float(np.nanstd(col(runs, path)))
    for g in GROUPS:
        summary['mono_' + g] = float(np.nanmean(col(runs, ['test_mono', g])))
        summary['auc_' + g] = float(np.nanmean(col(runs, ['test_auc', g])))
        summary['sens@90_' + g] = float(np.nanmean(
            col(runs, ['sens@spec_global_thr', g])))

    out = {'config': {'n_members': N_MEMBERS, 'tau_grid': TAU_GRID,
                      'alpha_grid': ALPHA_GRID, 'epochs': EPOCHS, 'batch': BATCH,
                      'lr': LR, 'seeds': FINAL_SEEDS},
           'summary': summary, 'per_seed': runs, 'method_details': aux,
           'self_check': selfcheck, 'paired': P_, 'predeclared_verdict': verdict}
    path = os.path.join(REPO, 'results', 'prototypes.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)

    print('\n%-8s %9s %9s %9s %9s %9s %9s' % ('method', 'worstAUC', 'aucGap', 'monoL',
                                              'monoD', 'monoGap', 'sens90D'))
    for nm, r in (('ERM', prev['summary']['erm']), ('DFR', prev['summary']['dfr']),
                  ('DPE', summary)):
        print('%-8s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f'
              % (nm, r['worst_auc'], r['auc_gap'], r['mono_light'], r['mono_dark'],
                 r['mono_gap'], r['sens@90_dark']))
    print('\nPAIRED:', json.dumps({k: {'mean': round(v['mean'], 4),
                                       'ci': [round(c, 4) for c in v['ci95']]}
                                   for k, v in P_.items()}, indent=1))
    print('\nVERDICT:', json.dumps(verdict, indent=1))
    print('\nwrote', path, '[%.0fs]' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
