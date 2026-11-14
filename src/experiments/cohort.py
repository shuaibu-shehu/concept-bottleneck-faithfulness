"""Count the Fitzpatrick17k and DDI cohorts by skin tone, malignancy and split.

Run:    python -m src.experiments.cohort
Writes: results/cohort.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
GROUPS = ['light', 'mid', 'dark']


def summarise(tag, rowmeta=None):
    z = np.load(os.path.join(REPO, 'caches/fairness_cache/features_%s.npz' % tag),
                allow_pickle=True)
    C = z['concept_scores'].astype(np.float32)
    g = z['group'].astype('U8')
    ok = np.isfinite(C).all(1) & np.isin(g, GROUPS)
    g, y = g[ok], z['malignant'][ok].astype(int)
    rec = {'n_total': int(ok.sum()), 'n_malignant': int(y.sum()),
           'n_dropped_unusable_tone': int((~ok).sum()), 'by_group': {}}
    for k in GROUPS:
        m = g == k
        rec['by_group'][k] = {'n': int(m.sum()), 'n_malignant': int(y[m].sum())}
        if rowmeta is not None:
            part = rowmeta['three_partition_label'].values[m]
            rec['by_group'][k]['pct_non_neoplastic'] = round(
                100.0 * float((part == 'non-neoplastic').mean()), 1)
    return rec


rowmeta = None
p = os.path.join(REPO, 'results', 'fitz_rowmeta.parquet')
if os.path.exists(p):
    rowmeta = pd.read_parquet(p)

out = {'fitzpatrick17k': summarise('fitz', rowmeta),
       'ddi': summarise('ddi')}

for name, rec in [('Fitzpatrick17k', out['fitzpatrick17k']), ('DDI', out['ddi'])]:
    print('%s: %d images (%d malignant), %d dropped for unusable tone'
          % (name, rec['n_total'], rec['n_malignant'], rec['n_dropped_unusable_tone']))
    for k in GROUPS:
        b = rec['by_group'][k]
        extra = ('  non-neoplastic %.1f%%' % b['pct_non_neoplastic']) if 'pct_non_neoplastic' in b else ''
        print('  %-5s %5d (%4d malignant)%s' % (k, b['n'], b['n_malignant'], extra))

dest = os.path.join(REPO, 'results', 'cohort.json')
json.dump(out, open(dest, 'w'), indent=2)
print('\nsaved', dest)
