"""Train the head on the zero-shot checklist layer: ERM, the mitigations and a Group-DRO sweep.

Later steps reuse its ERM run as a regression check.

Run:    python -m src.experiments.mitigations_zeroshot
Writes: results/mitigations_zeroshot.json
"""
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
GROUPS = ['light', 'mid', 'dark']

HID, ITERS, LR = 16, 1500, 0.05
SWEEP_SEEDS, FINAL_SEEDS = [0, 1, 2], list(range(10))
ETAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
WDS = [0.0, 1e-4, 1e-2]
SPEC = 0.90


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
Cddi, Yddi, Gddi = torch.tensor(Cd[okd]), torch.tensor(ddi['malignant'][okd].astype(np.float32)), gd[okd]

tr, va, te = SP == 'train', SP == 'val', SP == 'test'
Xtr, ytr, gtr = C[tr], Y[tr], G[tr]
print('train %d  val %d  test %d  | ddi %d' % (tr.sum(), va.sum(), te.sum(), len(Gddi)))
print('val is used for model selection here; no code in the repo had ever touched it.')

gmask_tr = [torch.tensor(gtr == g) for g in GROUPS]


# --- model ---
def fwd(Cm, P):
    W1, b1, W2, b2 = P
    return (torch.relu(Cm @ W1 + b1) @ W2 + b2).squeeze(1)


def mono_terms(base, P):
    """Per-sample hinge on monotonicity violations, averaged over the 7 concepts."""
    pen = 0.0
    for i in range(base.shape[1]):
        v1, v0 = base.clone(), base.clone()
        v1[:, i], v0[:, i] = 1.0, 0.0
        pen = pen + torch.relu(torch.sigmoid(fwd(v0, P)) - torch.sigmoid(fwd(v1, P)))
    return pen / base.shape[1]


def train(method, seed, eta=1.0, wd=0.0, gamma=2.0):
    gg = torch.Generator().manual_seed(seed)
    P = [(torch.randn(Xtr.shape[1], HID, generator=gg) * 0.3).requires_grad_(True),
         torch.zeros(HID, requires_grad=True),
         (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True),
         torch.zeros(1, requires_grad=True)]
    opt = torch.optim.Adam(P, lr=LR, weight_decay=wd)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    pw = torch.tensor(max(float((ytr == 0).sum()), 1.0) / max(float((ytr == 1).sum()), 1.0))
    q = torch.ones(len(GROUPS)) / len(GROUPS)

    # group-balanced resampling: fixed index set, equal draws per tone group
    if method == 'resample':
        rs = np.random.default_rng(seed)
        per = int(max(m.sum() for m in gmask_tr))
        idx = np.concatenate([rs.choice(np.where(m.numpy())[0], per, replace=True)
                              for m in gmask_tr])
        idx = torch.tensor(idx)
    # importance reweighting: inverse group frequency, mean 1
    if method == 'reweight':
        inv = torch.tensor([1.0 / max(float(m.sum()), 1.0) for m in gmask_tr])
        w_of_sample = torch.zeros(len(ytr))
        for k, m in enumerate(gmask_tr):
            w_of_sample[m] = inv[k]
        w_of_sample = w_of_sample / w_of_sample.mean()

    for _ in range(ITERS):
        if method == 'resample':
            per_loss = bce(fwd(Xtr[idx], P), ytr[idx], pos_weight=pw, reduction='none')
            loss = per_loss.mean()
        else:
            per_loss = bce(fwd(Xtr, P), ytr, pos_weight=pw, reduction='none')
            if method == 'erm':
                loss = per_loss.mean()
            elif method == 'reweight':
                loss = (per_loss * w_of_sample).mean()
            elif method.startswith('gdro'):
                lg = torch.stack([per_loss[m].mean() if m.any() else torch.tensor(0.0)
                                  for m in gmask_tr])
                q = q * torch.exp(eta * lg.detach())
                q = q / q.sum()
                loss = (q * lg).sum()
            elif method == 'mono_global':
                loss = per_loss.mean() + gamma * mono_terms(Xtr, P).mean()
            elif method == 'mono_pergroup':
                # penalise the worst group's violation rate
                mt = mono_terms(Xtr, P)
                per_g = torch.stack([mt[m].mean() if m.any() else torch.tensor(0.0)
                                     for m in gmask_tr])
                loss = per_loss.mean() + gamma * per_g.max()
            else:
                raise ValueError(method)
        opt.zero_grad(); loss.backward(); opt.step()
    return P, q.detach().numpy().tolist()


# --- metrics ---
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
    """Per-group test sensitivity at one threshold set on validation for `spec` specificity."""
    thr = np.quantile(p_val[y_val == 0], spec)
    return {k: float((p_te[(g_te == k) & (y_te == 1)] > thr).mean()) for k in GROUPS}


def sens_at_spec_pergroup(p_val, y_val, g_val, p_te, y_te, g_te, spec=SPEC):
    """Per-group test sensitivity with a threshold per group set on validation."""
    out = {}
    for k in GROUPS:
        mv = (g_val == k) & (y_val == 0)
        thr = np.quantile(p_val[mv], spec) if mv.sum() > 10 else np.quantile(p_val[y_val == 0], spec)
        out[k] = float((p_te[(g_te == k) & (y_te == 1)] > thr).mean())
    return out


yva, gva = Y[va].numpy(), G[va]
yte, gte = Y[te].numpy(), G[te]
yddi_np = Yddi.numpy()


def evaluate(P):
    pv, pt, pd_ = probs(P, C[va]), probs(P, C[te]), probs(P, Cddi)
    rec = {'val_worst_auc': worst_auc(pv, yva, gva),
           'test_auc': {k: auc_np(yte[gte == k], pt[gte == k]) for k in GROUPS},
           'test_mono': {k: mono_np(P, C[te][torch.tensor(gte == k)]) for k in GROUPS},
           'ddi_auc': {k: auc_np(yddi_np[Gddi == k], pd_[Gddi == k]) for k in GROUPS},
           'ddi_mono': {k: mono_np(P, Cddi[torch.tensor(Gddi == k)]) for k in GROUPS},
           'sens@spec_global_thr': sens_at_spec_global(pv, yva, pt, yte, gte),
           'sens@spec_pergroup_thr': sens_at_spec_pergroup(pv, yva, gva, pt, yte, gte)}
    rec['test_worst_auc'] = min(v for v in rec['test_auc'].values() if not np.isnan(v))
    rec['test_auc_gap'] = rec['test_auc']['light'] - rec['test_auc']['dark']
    rec['test_mono_gap'] = rec['test_mono']['light'] - rec['test_mono']['dark']
    return rec


def mean_of(runs, path):
    vals = []
    for r in runs:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


# --- 1. Group-DRO sweep ---
print('\n' + '=' * 78)
print('GROUP-DRO SWEEP  (selection on VAL worst-group AUROC)')
print('=' * 78)
sweep = []
t0 = time.time()
for eta in ETAS:
    for wd in WDS:
        runs, qs = [], []
        for s in SWEEP_SEEDS:
            P, q = train('gdro', s, eta=eta, wd=wd)
            runs.append(evaluate(P)); qs.append(q)
        vw, _ = mean_of(runs, ['val_worst_auc'])
        qm = np.mean(qs, axis=0)
        sweep.append({'eta': eta, 'wd': wd, 'val_worst_auc': vw,
                      'final_q_mean': [float(v) for v in qm],
                      'q_max': float(qm.max())})
        print('  eta=%-7g wd=%-6g  val worst AUROC %.4f   final q = [%.3f %.3f %.3f]  max %.3f'
              % (eta, wd, vw, qm[0], qm[1], qm[2], qm.max()))
best = max(sweep, key=lambda r: r['val_worst_auc'])
print('\nselected: eta=%g wd=%g  (val worst AUROC %.4f)  [%.0fs]'
      % (best['eta'], best['wd'], best['val_worst_auc'], time.time() - t0))
collapsed = [r for r in sweep if r['q_max'] > 0.99]
print('configs where q collapsed onto one group (q_max>0.99): %d / %d'
      % (len(collapsed), len(sweep)))

# --- 2. all methods ---
print('\n' + '=' * 78)
print('METHOD COMPARISON  (%d seeds)' % len(FINAL_SEEDS))
print('=' * 78)
METHODS = [('erm', {}),
           ('gdro', {'eta': 1.0, 'wd': 0.0}),  # fixed step size, no tuning
           ('gdro', {'eta': best['eta'], 'wd': best['wd']}),  # tuned on validation
           ('resample', {}), ('reweight', {}),
           ('mono_global', {}), ('mono_pergroup', {})]
NAMES = ['erm', 'gdro_as_published', 'gdro_tuned', 'resample', 'reweight',
         'mono_global', 'mono_pergroup']

results = {}
print('%-20s %7s %7s %7s %7s %7s %7s' % ('method', 'worstAU', 'aucGap', 'monoGap',
                                         'mono_lt', 'mono_dk', 'sensDk'))
for (m, kw), name in zip(METHODS, NAMES):
    runs = [evaluate(train(m, s, **kw)[0]) for s in FINAL_SEEDS]
    rec = {'method': m, 'kwargs': kw, 'n_seeds': len(FINAL_SEEDS)}
    for key in ['test_worst_auc', 'test_auc_gap', 'test_mono_gap', 'val_worst_auc']:
        mu, sd = mean_of(runs, [key]); rec[key] = {'mean': mu, 'sd': sd}
    for blk in ['test_auc', 'test_mono', 'ddi_auc', 'ddi_mono',
                'sens@spec_global_thr', 'sens@spec_pergroup_thr']:
        rec[blk] = {}
        for k in GROUPS:
            mu, sd = mean_of(runs, [blk, k]); rec[blk][k] = {'mean': mu, 'sd': sd}
    results[name] = rec
    print('%-20s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f'
          % (name, rec['test_worst_auc']['mean'], rec['test_auc_gap']['mean'],
             rec['test_mono_gap']['mean'], rec['test_mono']['light']['mean'],
             rec['test_mono']['dark']['mean'], rec['sens@spec_global_thr']['dark']['mean']))

# --- 3. verdicts ---
print('\n' + '=' * 78)
print('VERDICTS (vs ERM, mean over seeds)')
print('=' * 78)
e = results['erm']
verdict = {}
for name in NAMES[1:]:
    r = results[name]
    verdict[name] = {
        'd_worst_auc': r['test_worst_auc']['mean'] - e['test_worst_auc']['mean'],
        'd_auc_gap': r['test_auc_gap']['mean'] - e['test_auc_gap']['mean'],
        'd_mono_gap': r['test_mono_gap']['mean'] - e['test_mono_gap']['mean'],
        'd_mono_dark': r['test_mono']['dark']['mean'] - e['test_mono']['dark']['mean'],
        'd_sens_dark': (r['sens@spec_global_thr']['dark']['mean']
                        - e['sens@spec_global_thr']['dark']['mean'])}
    v = verdict[name]
    print('%-20s worstAUROC %+.4f  aucGap %+.4f  monoGap %+.4f  mono(dark) %+.4f  sens(dark) %+.4f'
          % (name, v['d_worst_auc'], v['d_auc_gap'], v['d_mono_gap'],
             v['d_mono_dark'], v['d_sens_dark']))

print('\nper-group vs global threshold, dark-group sensitivity @%.0f%% spec (ERM):' % (100 * SPEC))
print('  global thr   %.4f' % e['sens@spec_global_thr']['dark']['mean'])
print('  per-group thr %.4f' % e['sens@spec_pergroup_thr']['dark']['mean'])

out = {'gdro_sweep': sweep, 'gdro_selected': best,
       'n_configs_with_collapsed_q': len(collapsed), 'methods': results, 'verdict_vs_erm': verdict}
dest = os.path.join(REPO, 'results', 'mitigations_zeroshot.json')
json.dump(out, open(dest, 'w'), indent=2)
print('\nsaved', dest)
