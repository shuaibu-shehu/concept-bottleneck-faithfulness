"""Supervised concept layer: one logistic probe per SkinCon concept on the frozen embedding.

Probes are fit on labelled training rows only; assert_no_leakage checks this.

Run:    python -m src.concept_layer
Writes: results/concept_layer.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
GROUPS = ['light', 'mid', 'dark']

MIN_POSITIVES = 50  # minimum positive annotations for a concept to be kept
PROBE_SEED = 0
PROBE_MAX_ITER = 3000

# --- clinical direction of each concept, for any skin malignancy ---
#  +1  asserting the feature must not lower predicted malignancy
#  -1  asserting the feature must not raise predicted malignancy
#   0  no consensus direction: still a model input, but not scored
CLINICAL_SIGNS = {
    'Nodule': +1, 'Ulcer': +1, 'Telangiectasia': +1, 'Friable': +1,
    'Brown(Hyperpigmentation)': +1, 'Black': +1, 'Erosion': +1,
    'Dome-shaped': +1, 'Papule': +1,
    'Pustule': -1, 'Bulla': -1, 'Exudate': -1, 'Erythema': -1, 'Yellow': -1,
    'Plaque': 0, 'Patch': 0, 'Atrophy': 0, 'Crust': 0,
    'Scale': 0, 'Scar': 0, 'White(Hypopigmentation)': 0, 'Purple': 0,
}

SKINCON_META = ('Unnamed: 0', 'ImageID', 'md5hash', 'DDI_file',
                'Do not consider this image')


# --- data loading ---
def _cohort_mask(z):
    """The paper's cohort filter, identical to faithfulness_gap_zeroshot.load()."""
    C = z['concept_scores'].astype(np.float32)
    g = z['group'].astype('U8')
    return np.isfinite(C).all(1) & np.isin(g, GROUPS)


def load_fitz():
    """Fitzpatrick17k cohort with SkinCon labels, joined by md5hash.

    Returns a dict with:
      emb  (n,512)   frozen DermLIP embedding
      C7   (n,7)     zero-shot checklist scores
      y,g,split      malignancy, tone group, train/val/test split
      cluster        near-duplicate cluster id, for duplicate-aware folds
      L    (n,K)     SkinCon labels, NaN where unannotated
      has  (n,)      True where SkinCon annotates the row
      names [K]      concepts kept (at least MIN_POSITIVES positives)
    """
    z = np.load(os.path.join(REPO, 'caches/fairness_cache/features_fitz.npz'),
                allow_pickle=True)
    ok = _cohort_mask(z)
    emb = z['emb'].astype(np.float64)[ok]
    C7 = z['concept_scores'].astype(np.float32)[ok]
    y = z['malignant'][ok].astype(int)
    g = z['group'].astype('U8')[ok]
    split = z['split'].astype('U8')[ok]

    rm = pd.read_parquet(os.path.join(REPO, 'results/fitz_rowmeta.parquet'))
    assert len(rm) == len(y), 'rowmeta rows (%d) != cohort (%d)' % (len(rm), len(y))
    assert (rm['group'].to_numpy() == g).all(), 'rowmeta tone disagrees with cache'
    assert (rm['malignant'].to_numpy() == y).all(), 'rowmeta label disagrees with cache'

    sk = pd.read_csv(os.path.join(REPO, 'data/skincon_fitz.csv'))
    sk['md5hash'] = sk['ImageID'].str.replace('.jpg', '', regex=False)
    sk = sk[sk['Do not consider this image'] == 0]
    concepts = [c for c in sk.columns if c not in SKINCON_META]

    m = rm[['md5hash']].merge(sk[['md5hash'] + concepts], on='md5hash',
                              how='left', indicator=True)
    assert len(m) == len(y), 'skincon merge changed row count (duplicate md5hash?)'
    has = (m['_merge'] == 'both').to_numpy()
    Lall = m[concepts].to_numpy(dtype=float)

    keep = [i for i, c in enumerate(concepts)
            if np.nansum(Lall[has, i]) >= MIN_POSITIVES]
    names = [concepts[i] for i in keep]
    return {'emb': emb, 'C7': C7, 'y': y, 'g': g, 'split': split,
            'cluster': rm['cluster'].to_numpy(),
            'three_partition': rm['three_partition_label'].to_numpy(),
            'L': Lall[:, keep], 'has': has, 'names': names}


def load_ddi():
    """DDI cohort with SkinCon labels, joined by the row order of caches/ddi_metadata.csv.

    The join is checked: tone and malignancy must agree on all 656 rows, otherwise the
    labels are returned as None.
    """
    z = np.load(os.path.join(REPO, 'caches/fairness_cache/features_ddi.npz'),
                allow_pickle=True)
    ok = _cohort_mask(z)
    out = {'emb': z['emb'].astype(np.float64)[ok],
           'C7': z['concept_scores'].astype(np.float32)[ok],
           'y': z['malignant'][ok].astype(int),
           'g': z['group'].astype('U8')[ok],
           'L': None, 'names': None, 'join_ok': False}

    csv = os.path.join(REPO, 'caches/ddi_metadata.csv')
    if not os.path.exists(csv):
        out['join_reason'] = 'caches/ddi_metadata.csv absent'
        return out

    d = pd.read_csv(csv)
    if len(d) != int(ok.sum()) or not ok.all():
        out['join_reason'] = 'row count mismatch (csv %d, cohort %d)' % (len(d), int(ok.sum()))
        return out

    tone2g = {12: 'light', 34: 'mid', 56: 'dark'}
    g_csv = np.array([tone2g.get(v, 'unknown') for v in d['skin_tone']])
    y_csv = d['malignant'].astype(int).to_numpy()
    ag_g = float((g_csv == out['g']).mean())
    ag_y = float((y_csv == out['y']).mean())
    out['join_agreement'] = {'group': ag_g, 'malignant': ag_y, 'n': len(d)}
    if ag_g != 1.0 or ag_y != 1.0:
        out['join_reason'] = ('row-order gate FAILED (group %.4f, malignant %.4f)'
                              % (ag_g, ag_y))
        return out

    sk = pd.read_csv(os.path.join(REPO, 'data/skincon_ddi.csv'))
    sk = sk.rename(columns={'ImageID': 'DDI_file'})
    if 'Do not consider this image' in sk.columns:
        sk = sk[sk['Do not consider this image'] == 0]
    concepts = [c for c in sk.columns if c not in SKINCON_META]
    m = d[['DDI_file']].merge(sk[['DDI_file'] + concepts], on='DDI_file',
                              how='left', indicator=True)
    out['L_all'] = m[concepts].to_numpy(dtype=float)
    out['names_all'] = concepts
    out['has'] = (m['_merge'] == 'both').to_numpy()
    out['join_ok'] = True
    return out


# --- the concept layer ---
def fit_probes(emb, L, has, train_idx):
    """Fit one probe per concept on the labelled rows within train_idx.

    In cross-validation, train_idx must be the fold's training rows.
    """
    tr = np.zeros(len(emb), dtype=bool)
    tr[train_idx] = True
    fit_on = tr & has
    probes = []
    for j in range(L.shape[1]):
        t = L[:, j]
        m = fit_on & np.isfinite(t)
        if len(np.unique(t[m])) < 2:
            probes.append(None)          # degenerate in this fold; scored as 0.5
            continue
        probes.append(LogisticRegression(max_iter=PROBE_MAX_ITER,
                                         class_weight='balanced',
                                         random_state=PROBE_SEED).fit(emb[m], t[m]))
    return probes


def concept_scores(probes, emb):
    """Apply the probes to every row, labelled or not."""
    S = np.full((len(emb), len(probes)), 0.5)
    for j, p in enumerate(probes):
        if p is not None:
            S[:, j] = p.predict_proba(emb)[:, 1]
    return S


def signs_for(names, rule='clinical', y=None, S=None, train_idx=None):
    """Direction per concept, in {-1, 0, +1}; 0 is excluded from scoring only.

    rule='clinical': the declared table (primary analysis).
    rule='derived':  +1 if the concept's AUROC against malignancy on train_idx is at least
                     0.5, otherwise -1 (pre-specified sensitivity analysis).
    """
    if rule == 'clinical':
        missing = [n for n in names if n not in CLINICAL_SIGNS]
        assert not missing, 'no declared sign for: %s' % missing
        return np.array([CLINICAL_SIGNS[n] for n in names], dtype=int)
    if rule == 'derived':
        assert y is not None and S is not None and train_idx is not None
        return np.array([1 if _auc(y[train_idx], S[train_idx, j]) >= 0.5 else -1
                         for j in range(S.shape[1])], dtype=int)
    raise ValueError(rule)


def assert_no_leakage(train_idx, test_idx, cluster):
    """Probes must not train on a test row, nor on a duplicate of one."""
    assert not (set(train_idx) & set(test_idx)), 'probe train/test indices overlap'
    assert not (set(cluster[train_idx]) & set(cluster[test_idx])), \
        'probe training rows share a near-duplicate cluster with test rows'


def _auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --- report ---
def main():
    F = load_fitz()
    names, L, has = F['names'], F['L'], F['has']
    print('Fitzpatrick cohort %d | SkinCon annotated %d | by tone %s'
          % (len(F['y']), int(has.sum()),
             {k: int((has & (F['g'] == k)).sum()) for k in GROUPS}))
    print('concepts with >= %d positives: %d' % (MIN_POSITIVES, len(names)))

    sg = signs_for(names, 'clinical')
    print('signs: +1 x%d, -1 x%d, excluded x%d'
          % (int((sg > 0).sum()), int((sg < 0).sum()), int((sg == 0).sum())))

    # held-out concept detection on the test split
    tr = np.where(F['split'] == 'train')[0]
    te = np.where(F['split'] == 'test')[0]
    probes = fit_probes(F['emb'], L, has, tr)
    S = concept_scores(probes, F['emb'])

    te_lab = te[has[te]]
    det = {}
    for j, n in enumerate(names):
        t = L[:, j]
        row = {'n_pos_labelled': int(np.nansum(t[has])),
               'sign': int(sg[j]),
               'auroc_all': _auc(t[te_lab].astype(int), S[te_lab, j])}
        for k in GROUPS:
            m = te_lab[F['g'][te_lab] == k]
            row['auroc_' + k] = _auc(t[m].astype(int), S[m, j]) if len(m) else float('nan')
        row['light_minus_dark'] = row['auroc_light'] - row['auroc_dark']
        row['below_0.60'] = bool(row['auroc_all'] < 0.60)
        det[n] = row

    aur = np.array([det[n]['auroc_all'] for n in names])
    gaps = np.array([det[n]['light_minus_dark'] for n in names])
    print('detection AUROC: mean %.4f  min %.4f  max %.4f  | below 0.60: %d'
          % (aur.mean(), aur.min(), aur.max(), int((aur < 0.60).sum())))
    print('detection light-dark gap: mean %+.4f  positive in %d/%d concepts'
          % (np.nanmean(gaps), int((gaps > 0).sum()), len(gaps)))

    out = {'cohort': {'n': int(len(F['y'])), 'n_skincon': int(has.sum()),
                      'by_tone': {k: int((has & (F['g'] == k)).sum()) for k in GROUPS}},
           'concepts': names,
           'signs': {n: int(sg[j]) for j, n in enumerate(names)},
           'detection': det,
           'detection_summary': {'mean_auroc': float(aur.mean()),
                                 'min_auroc': float(aur.min()),
                                 'max_auroc': float(aur.max()),
                                 'n_below_0.60': int((aur < 0.60).sum()),
                                 'mean_light_minus_dark': float(np.nanmean(gaps)),
                                 'n_gaps_positive': int((gaps > 0).sum())}}

    # concept detection on DDI
    D = load_ddi()
    if D['join_ok']:
        Sd = concept_scores(probes, D['emb'])
        idx = {n: i for i, n in enumerate(D['names_all'])}
        ext = {}
        for j, n in enumerate(names):
            if n not in idx:
                continue
            t = D['L_all'][:, idx[n]]
            m = D['has'] & np.isfinite(t)
            if len(np.unique(t[m])) < 2 or int(np.nansum(t[m])) < 20:
                continue
            ext[n] = {'n_pos': int(np.nansum(t[m])),
                      'auroc': _auc(t[m].astype(int), Sd[m, j])}
        va = [v['auroc'] for v in ext.values()]
        print('DDI concept detection (probes trained on Fitzpatrick only): '
              '%d concepts, mean AUROC %.4f' % (len(ext), float(np.mean(va))))
        out['ddi_detection'] = {'join_agreement': D['join_agreement'],
                                'per_concept': ext,
                                'mean_auroc': float(np.mean(va)),
                                'n_concepts': len(ext)}
    else:
        print('DDI SkinCon arm unavailable: %s' % D.get('join_reason'))
        out['ddi_detection'] = {'available': False,
                                'reason': D.get('join_reason', 'unknown')}

    dst = os.path.join(REPO, 'results/concept_layer.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', os.path.relpath(dst, REPO))


if __name__ == '__main__':
    main()
