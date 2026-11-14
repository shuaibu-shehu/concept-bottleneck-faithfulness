"""Six-template prompt ensemble for the 22 SkinCon concepts (Section 5.1).

Averages the unit-norm text embeddings of six paraphrase templates per concept and pole,
and scores the concepts on the same embedding and split as zeroshot_vocabulary.py.

Run:    python -m src.experiments.prompt_ensemble
Reads:  caches/fairness_cache/features_fitz.npz and the DermLIP text encoder
Writes: results/prompt_ensemble.json
"""
import json
import os

import numpy as np
import torch

from src.concept_layer import GROUPS, load_fitz, fit_probes, concept_scores, signs_for
from src.experiments.zeroshot_vocabulary import (MODEL_NAME, SEEDS, detection, head_metrics,
                                                 softmax2, zeroshot_scores)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# one noun phrase per concept, inserted into each template below
PHRASE = {
    'Papule': 'a papule', 'Plaque': 'a plaque', 'Pustule': 'a pustule', 'Bulla': 'a bulla',
    'Patch': 'a patch', 'Nodule': 'a nodule', 'Ulcer': 'an ulcer', 'Crust': 'a crust',
    'Erosion': 'an erosion', 'Atrophy': 'atrophy', 'Exudate': 'exudate',
    'Telangiectasia': 'telangiectasia', 'Scale': 'scale', 'Scar': 'a scar',
    'Friable': 'a friable surface', 'Dome-shaped': 'a dome shape',
    'Brown(Hyperpigmentation)': 'brown hyperpigmentation',
    'White(Hypopigmentation)': 'white hypopigmentation',
    'Purple': 'purple colour', 'Yellow': 'yellow colour', 'Black': 'black colour',
    'Erythema': 'erythema',
}

TEMPLATES = [
    ('a skin lesion with %s', 'a skin lesion without %s'),
    ('a photo of a skin lesion with %s', 'a photo of a skin lesion without %s'),
    ('a clinical photograph of skin showing %s', 'a clinical photograph of skin showing no %s'),
    ('a dermatology image with %s', 'a dermatology image with no %s'),
    ('this skin lesion has %s', 'this skin lesion does not have %s'),
    ('skin with %s', 'skin without %s'),
]


def ensemble_directions(names):
    """Mean unit-norm text embedding over templates, per concept and polarity."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.eval()
    pres, absn = [], []
    for n in names:
        p = [tp % PHRASE[n] for tp, _ in TEMPLATES]
        a = [ta % PHRASE[n] for _, ta in TEMPLATES]
        with torch.no_grad():
            tp = model.encode_text(tokenizer(p)).float()
            ta = model.encode_text(tokenizer(a)).float()
        tp = tp / tp.norm(dim=-1, keepdim=True)
        ta = ta / ta.norm(dim=-1, keepdim=True)
        tp = tp.mean(0); ta = ta.mean(0)
        pres.append((tp / tp.norm()).numpy())
        absn.append((ta / ta.norm()).numpy())
    return np.stack(pres), np.stack(absn)


def main():
    F = load_fitz()
    names, L, has, emb, y, g = F['names'], F['L'], F['has'], F['emb'], F['y'], F['g']
    tr = np.where(F['split'] == 'train')[0]
    te = np.where(F['split'] == 'test')[0]
    missing = [n for n in names if n not in PHRASE]
    assert not missing, 'no phrase for: %s' % missing

    print('encoding %d concepts x %d templates x 2 polarities ...'
          % (len(names), len(TEMPLATES)))
    pres, absn = ensemble_directions(names)
    E = np.zeros((len(emb), len(names)))
    for j in range(len(names)):
        E[:, j] = softmax2(absn[j] @ emb.T, pres[j] @ emb.T)[:, 1]

    Z = zeroshot_scores(emb, names)                 # the single-pair baseline
    probes = fit_probes(emb, L, has, tr)
    S = concept_scores(probes, emb)

    de, dz, ds = (detection(X, L, has, te, g) for X in (E, Z, S))
    me = [v['auroc_all'] for v in de.values()]
    mz = [v['auroc_all'] for v in dz.values()]
    ms = [v['auroc_all'] for v in ds.values()]

    print('')
    print('concept detection AUROC on held-out labelled test rows')
    print('  single prompt pair : mean %.4f  below chance %d/22' % (np.mean(mz), sum(x < 0.5 for x in mz)))
    print('  prompt ensemble    : mean %.4f  below chance %d/22' % (np.mean(me), sum(x < 0.5 for x in me)))
    print('  supervised probes  : mean %.4f  below chance %d/22' % (np.mean(ms), sum(x < 0.5 for x in ms)))
    print('')
    print('  %-26s %8s %9s %9s' % ('concept', 'single', 'ensemble', 'probe'))
    for j, n in enumerate(names):
        if j in de and j in dz:
            print('  %-26s %8.3f %9.3f %9.3f'
                  % (n, dz[j]['auroc_all'], de[j]['auroc_all'], ds[j]['auroc_all']))

    sg = signs_for(names, 'clinical')
    print('')
    print('training head on the ensemble layer (%d seeds) ...' % len(SEEDS))
    head = head_metrics(E, y, g, tr, te, sg)
    print('  AUROC  light %.4f  mid %.4f  dark %.4f   light-dark %.4f'
          % (head['auc']['light']['mean'], head['auc']['mid']['mean'],
             head['auc']['dark']['mean'], head['auc_gap']['mean']))
    print('  mono   light %.4f  mid %.4f  dark %.4f   light-dark %.4f'
          % (head['mono']['light']['mean'], head['mono']['mid']['mean'],
             head['mono']['dark']['mean'], head['mono_gap']['mean']))

    out = {
        'templates': TEMPLATES, 'phrases': PHRASE, 'n_templates': len(TEMPLATES),
        'detection': {
            'single_pair': {names[j]: v for j, v in dz.items()},
            'ensemble': {names[j]: v for j, v in de.items()},
            'supervised': {names[j]: v for j, v in ds.items()},
            'mean_auroc': {'single_pair': float(np.mean(mz)),
                           'ensemble': float(np.mean(me)),
                           'supervised': float(np.mean(ms))},
            'below_chance': {'single_pair': int(sum(x < 0.5 for x in mz)),
                             'ensemble': int(sum(x < 0.5 for x in me)),
                             'supervised': int(sum(x < 0.5 for x in ms))},
        },
        'head_ensemble': head,
    }
    path = os.path.join(REPO, 'results', 'prompt_ensemble.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('')
    print('wrote %s' % path)


if __name__ == '__main__':
    main()
