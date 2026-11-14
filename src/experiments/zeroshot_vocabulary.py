"""The 22 SkinCon concepts read with one prompt pair each (Section 5.1).

Separates the vocabulary from the readout: checklist prompts, SkinCon prompts and SkinCon
probes. Reports concept detection and trains the head on each layer.

Run:    python -m src.experiments.zeroshot_vocabulary
Reads:  caches/fairness_cache/features_fitz.npz and the DermLIP text encoder
Writes: results/zeroshot_vocabulary.json
"""
import json
import os

import numpy as np
import torch

from src.concept_layer import (GROUPS, load_fitz, fit_probes, concept_scores,
                               signs_for)
from src.metrics import auc_np, mono_np, probs
from src.training import make_trainer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
MODEL_NAME = 'hf-hub:redlessone/DermLIP_ViT-B-16'
SEEDS = list(range(10))
TAU = 100.0          # the encoder's own learned logit scale, as used at encoding time

# one plain present/absent prompt pair per SkinCon concept
SKINCON_PROMPTS = {
    'Papule': ('a skin lesion with a papule', 'a skin lesion without a papule'),
    'Plaque': ('a skin lesion with a plaque', 'a skin lesion without a plaque'),
    'Pustule': ('a skin lesion with a pustule', 'a skin lesion without a pustule'),
    'Bulla': ('a skin lesion with a bulla', 'a skin lesion without a bulla'),
    'Patch': ('a skin lesion with a patch', 'a skin lesion without a patch'),
    'Nodule': ('a skin lesion with a nodule', 'a skin lesion without a nodule'),
    'Ulcer': ('a skin lesion with an ulcer', 'a skin lesion without an ulcer'),
    'Crust': ('a skin lesion with a crust', 'a skin lesion without a crust'),
    'Erosion': ('a skin lesion with an erosion', 'a skin lesion without an erosion'),
    'Atrophy': ('a skin lesion with atrophy', 'a skin lesion without atrophy'),
    'Exudate': ('a skin lesion with exudate', 'a skin lesion without exudate'),
    'Telangiectasia': ('a skin lesion with telangiectasia',
                       'a skin lesion without telangiectasia'),
    'Scale': ('a skin lesion with scale', 'a skin lesion without scale'),
    'Scar': ('a skin lesion with a scar', 'a skin lesion without a scar'),
    'Friable': ('a friable skin lesion', 'a skin lesion that is not friable'),
    'Dome-shaped': ('a dome-shaped skin lesion', 'a skin lesion that is not dome-shaped'),
    'Brown(Hyperpigmentation)': ('a brown hyperpigmented skin lesion',
                                 'a skin lesion without brown hyperpigmentation'),
    'White(Hypopigmentation)': ('a white hypopigmented skin lesion',
                                'a skin lesion without white hypopigmentation'),
    'Purple': ('a purple skin lesion', 'a skin lesion that is not purple'),
    'Yellow': ('a yellow skin lesion', 'a skin lesion that is not yellow'),
    'Black': ('a black skin lesion', 'a skin lesion that is not black'),
    'Erythema': ('a skin lesion with erythema', 'a skin lesion without erythema'),
}


def softmax2(a, b):
    """Identical to the encoder's scoring: softmax over two cosines at the logit scale."""
    z = np.stack([a, b], 1) * TAU
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def encode_prompts(names):
    """Unit-norm text embeddings for the present and absent prompt of each concept."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.eval()
    print('logit_scale.exp() = %.2f   (prompts scored at TAU = %.1f)'
          % (float(model.logit_scale.exp().detach()), TAU))
    present = [SKINCON_PROMPTS[n][0] for n in names]
    absent = [SKINCON_PROMPTS[n][1] for n in names]
    with torch.no_grad():
        t = model.encode_text(tokenizer(present + absent)).float()
        t = t / t.norm(dim=-1, keepdim=True)
    t = t.numpy()
    return t[:len(names)], t[len(names):]


def checkpoint_logit_scale():
    """The released checkpoint's learned temperature, which TAU rounds."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    return float(model.logit_scale.exp().detach())


def zeroshot_scores(emb, names):
    pres, absn = encode_prompts(names)
    S = np.zeros((len(emb), len(names)))
    for j in range(len(names)):
        S[:, j] = softmax2(absn[j] @ emb.T, pres[j] @ emb.T)[:, 1]
    return S


def detection(S, L, has, idx, g):
    """Per-concept detection AUROC against the SkinCon annotations, overall and by tone."""
    out = {}
    lab = idx[has[idx]]
    for j in range(S.shape[1]):
        t = L[lab, j]
        m = np.isfinite(t)
        if len(np.unique(t[m])) < 2:
            continue
        row = {'auroc_all': auc_np(t[m], S[lab][m, j])}
        for k in GROUPS:
            mk = m & (g[lab] == k)
            if len(np.unique(t[mk])) == 2:
                row['auroc_' + k] = auc_np(t[mk], S[lab][mk, j])
        out[j] = row
    return out


def head_metrics(X, y, g, tr, te, signs):
    """Train the two-layer head on one concept layer; report accuracy and monotonicity."""
    Xtr = torch.tensor(X[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    Xte = torch.tensor(X[te], dtype=torch.float32)
    train = make_trainer(Xtr, ytr, g[tr], signs)
    runs = []
    for s in SEEDS:
        P, _ = train('erm', s)
        p = probs(P, Xte)
        r = {'auc': {k: auc_np(y[te][g[te] == k], p[g[te] == k]) for k in GROUPS},
             'mono': {}}
        for k in GROUPS:
            r['mono'][k] = mono_np(P, Xte[torch.tensor(g[te] == k)], signs)
        r['auc_gap'] = r['auc']['light'] - r['auc']['dark']
        r['mono_gap'] = r['mono']['light'] - r['mono']['dark']
        r['worst_auc'] = min(r['auc'].values())
        runs.append(r)

    def agg(path):
        v = [r[path[0]][path[1]] if len(path) == 2 else r[path[0]] for r in runs]
        return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}

    return {
        'auc': {k: agg(['auc', k]) for k in GROUPS},
        'mono': {k: agg(['mono', k]) for k in GROUPS},
        'auc_gap': agg(['auc_gap']),
        'mono_gap': agg(['mono_gap']),
        'worst_auc': agg(['worst_auc']),
        'n_seeds': len(SEEDS),
    }


def main():
    F = load_fitz()
    names, L, has, emb, y, g = F['names'], F['L'], F['has'], F['emb'], F['y'], F['g']
    tr = np.where(F['split'] == 'train')[0]
    te = np.where(F['split'] == 'test')[0]
    print('cohort %d | SkinCon annotated %d | concepts %d'
          % (len(y), int(has.sum()), len(names)))

    missing = [n for n in names if n not in SKINCON_PROMPTS]
    assert not missing, 'no prompt pair for: %s' % missing

    print('\nencoding %d prompt pairs ...' % len(names))
    Z22 = zeroshot_scores(emb, names)
    probes = fit_probes(emb, L, has, tr)
    S22 = concept_scores(probes, emb)

    # --- concept detection against the SkinCon annotations ---
    dz = detection(Z22, L, has, te, g)
    ds = detection(S22, L, has, te, g)
    mz = [v['auroc_all'] for v in dz.values()]
    ms = [v['auroc_all'] for v in ds.values()]
    print('\nconcept detection AUROC on held-out labelled test rows')
    print('  zero-shot prompts : mean %.4f  median %.4f  min %.4f  max %.4f'
          % (np.mean(mz), np.median(mz), min(mz), max(mz)))
    print('  supervised probes : mean %.4f  median %.4f  min %.4f  max %.4f'
          % (np.mean(ms), np.median(ms), min(ms), max(ms)))
    print('  per concept (zero-shot vs supervised):')
    for j, n in enumerate(names):
        if j in dz and j in ds:
            print('    %-26s %.3f   %.3f' % (n, dz[j]['auroc_all'], ds[j]['auroc_all']))

    # --- the same head on each concept layer ---
    sg22 = signs_for(names, 'clinical')
    sg7 = np.ones(F['C7'].shape[1], dtype=int)
    layers = {
        'zeroshot7': (F['C7'].astype(np.float64), sg7),
        'zeroshot22': (Z22, sg22),
        'supervised22': (S22, sg22),
    }
    res = {}
    for tag, (X, sg) in layers.items():
        print('\ntraining head on %s (%d concepts, %d seeds) ...'
              % (tag, X.shape[1], len(SEEDS)))
        res[tag] = head_metrics(X, y, g, tr, te, sg)
        r = res[tag]
        print('  AUROC  light %.4f  mid %.4f  dark %.4f   light-dark %.4f'
              % (r['auc']['light']['mean'], r['auc']['mid']['mean'],
                 r['auc']['dark']['mean'], r['auc_gap']['mean']))
        print('  mono   light %.4f  mid %.4f  dark %.4f   light-dark %.4f'
              % (r['mono']['light']['mean'], r['mono']['mid']['mean'],
                 r['mono']['dark']['mean'], r['mono_gap']['mean']))

    print('\n--- vocabulary against readout ---')
    print('%-14s %-9s %-10s %10s %10s' % ('layer', 'vocab', 'readout',
                                          'AUROC gap', 'mono gap'))
    meta = {'zeroshot7': ('checklist', 'prompts'),
            'zeroshot22': ('SkinCon', 'prompts'),
            'supervised22': ('SkinCon', 'probes')}
    for tag in ('zeroshot7', 'zeroshot22', 'supervised22'):
        v, rd = meta[tag]
        print('%-14s %-9s %-10s %10.4f %10.4f'
              % (tag, v, rd, res[tag]['auc_gap']['mean'], res[tag]['mono_gap']['mean']))

    out = {
        'prompts': {n: list(SKINCON_PROMPTS[n]) for n in names},
        'tau': TAU,
        'checkpoint_logit_scale': checkpoint_logit_scale(),
        'n_seeds': len(SEEDS),
        'detection': {
            'zeroshot22': {names[j]: v for j, v in dz.items()},
            'supervised22': {names[j]: v for j, v in ds.items()},
            'mean_auroc': {'zeroshot22': float(np.mean(mz)),
                           'supervised22': float(np.mean(ms))},
        },
        'head': res,
    }
    path = os.path.join(REPO, 'results', 'zeroshot_vocabulary.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\nwrote %s' % path)


if __name__ == '__main__':
    main()
