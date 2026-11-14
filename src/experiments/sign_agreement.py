"""Agreement between the clinical directions and the label-derived rule (Appendix C).

A concept agrees when the derived sign matches the clinical one in at least three of the
five folds. Also reports each concept's training AUROC against malignancy.

Run:    python -m src.experiments.sign_agreement
Reads:  results/faithfulness_gap_supervised.json
Writes: results/sign_agreement.json
"""
import io
import json
import os
from math import comb

import numpy as np

from src.concept_layer import load_fitz, fit_probes, concept_scores, CLINICAL_SIGNS
from src.metrics import auc_np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def binomial_upper_tail(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2): agreement this high from arbitrary signs."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    R = json.load(io.open(os.path.join(REPO, 'results', 'faithfulness_gap_supervised.json'),
                          encoding='utf-8'))
    names = R['bottleneck']['concepts']
    by_fold = R['bottleneck']['derived_signs_by_fold']

    # training AUROC of each concept score against malignancy
    F = load_fitz()
    tr = np.where(F['split'] == 'train')[0]
    probes = fit_probes(F['emb'], F['L'], F['has'], tr)
    S = concept_scores(probes, F['emb'])
    auc_vs_malignancy = {n: auc_np(F['y'][tr], S[tr, j]) for j, n in enumerate(names)}

    print('%-26s %6s %8s %10s %9s' % ('concept', 'clin', 'folds+', 'AUROC(y)', 'agrees'))
    rows = []
    for n in names:
        c = CLINICAL_SIGNS[n]
        if c == 0:
            continue
        npos = sum(1 for f in by_fold if f[n] > 0)
        n_agree = sum(1 for f in by_fold if f[n] == c)
        majority = 1 if npos >= 3 else -1
        rows.append({'concept': n, 'clinical_sign': c, 'folds_derived_positive': npos,
                     'folds_agreeing': n_agree, 'n_folds': len(by_fold),
                     'agrees': bool(majority == c),
                     'auroc_vs_malignancy_train': auc_vs_malignancy[n]})
        print('%-26s %6d %8s %10.3f %9s'
              % (n, c, '%d/%d' % (npos, len(by_fold)), auc_vs_malignancy[n],
                 'yes' if majority == c else 'NO'))

    agree = sum(r['agrees'] for r in rows)
    p = binomial_upper_tail(agree, len(rows))
    print('\nsigned concepts: %d | clinical and derived agree on %d, disagree on %d'
          % (len(rows), agree, len(rows) - agree))
    print('one-sided exact binomial p against arbitrary signs: %.4f  (%d expected by chance)'
          % (p, len(rows) // 2))
    print('\nfor the disagreeing concepts, how decisive is the data?')
    for r in rows:
        if not r['agrees']:
            a = r['auroc_vs_malignancy_train']
            print('  %-26s AUROC vs malignancy %.3f  (margin from chance %.3f)'
                  % (r['concept'], a, abs(a - 0.5)))

    out = {'rule': 'agrees when the derived sign matches the clinical sign in at least '
                   'three of the five folds',
           'n_signed': len(rows), 'n_agree': int(agree),
           'binomial_p_one_sided': p, 'concepts': rows}
    dest = os.path.join(REPO, 'results', 'sign_agreement.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s' % os.path.relpath(dest, REPO))


if __name__ == '__main__':
    main()
