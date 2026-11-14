"""Feature sets compared by one logistic regression each (Table 2, Figure 2a).

Fits one balanced logistic regression per feature set on the training rows and scores the
test split: the raw embedding, PCA with 7 and 22 components, the checklist prompts, the
SkinCon prompts (one pair and the ensemble) and the SkinCon probes.

Run:    python -m src.experiments.readout_comparison
Writes: results/readout_comparison.json
"""
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.concept_layer import GROUPS, load_fitz, fit_probes, concept_scores
from src.experiments.prompt_ensemble import ensemble_directions
from src.experiments.zeroshot_vocabulary import softmax2, zeroshot_scores
from src.metrics import auc_np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# values printed in the paper, checked on every run
PAPER = {
    'raw embedding (512-d)': (0.9405, -0.0014),
    '7 unsupervised PCA': (0.9226, 0.0327),
    '7 zero-shot checklist': (0.7730, 0.1463),
    '22 supervised SkinCon': (0.9150, 0.0302),
}


def evaluate(X, y, g, tr, te):
    """One balanced logistic regression, fit on the training rows and scored on the test split."""
    lr = LogisticRegression(max_iter=5000, class_weight='balanced',
                            random_state=0).fit(X[tr], y[tr])
    p = lr.predict_proba(X[te])[:, 1]
    out = {k: auc_np(y[te][g[te] == k], p[g[te] == k]) for k in GROUPS}
    out['auroc'] = auc_np(y[te], p)
    out['gap'] = out['light'] - out['dark']
    out['worst'] = min(out[k] for k in GROUPS)
    return out


def main():
    F = load_fitz()
    names, emb, y, g = F['names'], F['emb'], F['y'], F['g']
    tr = np.where(F['split'] == 'train')[0]
    te = np.where(F['split'] == 'test')[0]
    print('cohort %d | train %d | test %d' % (len(y), len(tr), len(te)))

    print('building the seven feature sets ...')
    P7 = PCA(n_components=7, random_state=0).fit(emb[tr]).transform(emb)
    P22 = PCA(n_components=22, random_state=0).fit(emb[tr]).transform(emb)
    Z22 = zeroshot_scores(emb, names)
    pres, absn = ensemble_directions(names)
    E22 = np.zeros((len(emb), len(names)))
    for j in range(len(names)):
        E22[:, j] = softmax2(absn[j] @ emb.T, pres[j] @ emb.T)[:, 1]
    S22 = concept_scores(fit_probes(emb, F['L'], F['has'], tr), emb)

    feats = [
        ('raw embedding (512-d)', emb),
        ('7 unsupervised PCA', P7),
        ('22 unsupervised PCA', P22),
        ('7 zero-shot checklist', F['C7'].astype(np.float64)),
        ('22 zero-shot SkinCon', Z22),
        ('22 zero-shot SkinCon ensemble', E22),
        ('22 supervised SkinCon', S22),
    ]

    res, ok = {}, True
    print('')
    print('%-32s %7s %7s %7s %8s %9s %10s'
          % ('feature set', 'light', 'mid', 'dark', 'AUROC', 'light-dark', 'vs paper'))
    for name, X in feats:
        r = evaluate(X, y, g, tr, te)
        res[name] = r
        chk = ''
        if name in PAPER:
            pa, pg = PAPER[name]
            match = abs(r['auroc'] - pa) < 5e-4 and abs(r['gap'] - pg) < 5e-4
            ok &= match
            chk = 'match' if match else 'DIFFERS'
        print('%-32s %7.3f %7.3f %7.3f %8.4f %9.4f %10s'
              % (name, r['light'], r['mid'], r['dark'], r['auroc'], r['gap'], chk))

    print('')
    print('all four published rows reproduce Section 5.1: %s' % ok)

    out = {'protocol': ('one balanced logistic probe per feature set, fit on the training rows, '
                        'scored on the Fitzpatrick17k test split; identical to Section 5.1'),
           'reproduces_section_5_1': bool(ok), 'paper_values': PAPER, 'results': res}
    path = os.path.join(REPO, 'results', 'readout_comparison.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote %s' % path)


if __name__ == '__main__':
    main()
