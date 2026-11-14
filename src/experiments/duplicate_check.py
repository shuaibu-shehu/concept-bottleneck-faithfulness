"""Find near-duplicate images in Fitzpatrick17k from the cached embeddings.

Pairs with cosine similarity of at least COS_THRESH are linked into clusters, which the
duplicate-aware folds keep together. Also recovers md5hash and the diagnosis labels of each
cached row by re-reading fitzpatrick17k.csv with the encoder's filter and row order.

Run:    python -m src.experiments.duplicate_check
Reads:  caches/fairness_cache/features_fitz.npz (downloads the labels CSV if needed)
Writes: results/duplicate_check.json, results/fitz_rowmeta.parquet
"""
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
GROUPS = ['light', 'mid', 'dark']
COS_THRESH = 0.98          # near-duplicate cutoff on unit-norm DermLIP embeddings
CSV_URL = 'https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv'


def tone_group(v):
    try:
        s = int(float(v))
    except Exception:
        return 'unknown'
    return {1: 'light', 2: 'light', 3: 'mid', 4: 'mid', 5: 'dark', 6: 'dark'}.get(s, 'unknown')


# --- load the cache ---
z = np.load(os.path.join(REPO, 'caches/fairness_cache/features_fitz.npz'), allow_pickle=True)
C_all = z['concept_scores'].astype(np.float32)
g_all = z['group'].astype('U8')
n_rows = len(g_all)
print('cache rows: %d' % n_rows)

# --- rejoin the labels CSV ---
local = os.path.join(REPO, 'caches', 'fitzpatrick17k.csv')
if os.path.exists(local):
    f = pd.read_csv(local)
    print('labels csv: local cache')
else:
    f = pd.read_csv(CSV_URL)
    f.to_csv(local, index=False)
    print('labels csv: downloaded -> %s' % local)

part_col = 'three_partition_label' if 'three_partition_label' in f.columns else 'nine_partition_label'
tone_col = next(c for c in ['fitzpatrick_scale', 'fitzpatrick', 'fitzpatrick_centaur'] if c in f.columns)
id_col = next(c for c in ['md5hash', 'hasher', 'image', 'filename'] if c in f.columns)

# exactly the filter and ordering encode_features.py applies
f = f[f[part_col].notna() & f[tone_col].notna()].reset_index(drop=True)
print('csv rows after the encoder filter: %d  (cache rows: %d)' % (len(f), n_rows))
aligned = len(f) == n_rows
print('row-order alignment: %s' % ('OK' if aligned else 'MISMATCH -- do not trust the rejoin'))

if aligned:
    csv_group = np.array([tone_group(v) for v in f[tone_col]])
    agree = float((csv_group == g_all).mean())
    print('tone-label agreement cache vs csv: %.4f  (1.0 confirms the alignment)' % agree)
else:
    agree = float('nan')

# --- near-duplicate clusters ---
ok = np.isfinite(C_all).all(1) & np.isin(g_all, GROUPS)
emb = z['emb'][ok].astype(np.float32)
split = z['split'].astype('U8')[ok]
grp = g_all[ok]
idx_global = np.where(ok)[0]
n = len(emb)
print('\nanalysis cohort: %d images' % n)

norms = np.linalg.norm(emb, axis=1)
print('embedding norms: min %.4f max %.4f (expect ~1.0)' % (norms.min(), norms.max()))

parent = np.arange(n)


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


CH = 1024
n_pairs = 0
t0 = time.time()
for s in range(0, n, CH):
    block = emb[s:s + CH] @ emb.T                      # (b, n) cosine, unit-norm
    for r in range(block.shape[0]):
        i = s + r
        hits = np.where(block[r] >= COS_THRESH)[0]
        hits = hits[hits > i]                          # upper triangle only
        n_pairs += len(hits)
        for j in hits:
            union(i, int(j))
print('pairs >= %.2f cosine: %d   (%.1fs)' % (COS_THRESH, n_pairs, time.time() - t0))

cluster = np.array([find(i) for i in range(n)])
_, cluster = np.unique(cluster, return_inverse=True)
sizes = np.bincount(cluster)
n_clusters = len(sizes)
n_multi = int((sizes > 1).sum())
n_in_multi = int(sizes[sizes > 1].sum())
print('clusters: %d  (%d contain >1 image, covering %d images = %.1f%%)'
      % (n_clusters, n_multi, n_in_multi, 100 * n_in_multi / n))
print('largest cluster: %d images' % sizes.max())

# --- leakage across splits ---
leaky = 0
leaky_tr_te = 0
for c in np.where(sizes > 1)[0]:
    sp = set(split[cluster == c])
    if len(sp) > 1:
        leaky += 1
        if 'train' in sp and 'test' in sp:
            leaky_tr_te += 1
print('\nclusters straddling >1 split: %d' % leaky)
print('clusters straddling train AND test: %d   <-- direct leakage' % leaky_tr_te)

n_test_leaked = int(sum(((cluster == c) & (split == 'test')).sum()
                        for c in np.where(sizes > 1)[0]
                        if {'train', 'test'} <= set(split[cluster == c])))
print('test images with a near-duplicate in train: %d / %d (%.1f%%)'
      % (n_test_leaked, int((split == 'test').sum()),
         100 * n_test_leaked / max(int((split == 'test').sum()), 1)))

by_tone = {}
for t in GROUPS:
    m = (split == 'test') & (grp == t)
    leak = int(sum(((cluster == c) & m).sum() for c in np.where(sizes > 1)[0]
                   if {'train', 'test'} <= set(split[cluster == c])))
    by_tone[t] = {'n_test': int(m.sum()), 'n_leaked': leak,
                  'pct': round(100 * leak / max(int(m.sum()), 1), 2)}
    print('  %-5s %d / %d test images leaked (%.1f%%)' % (t, leak, int(m.sum()), by_tone[t]['pct']))

# --- save ---
if aligned:
    rowmeta = pd.DataFrame({
        'md5hash': f[id_col].astype(str).values[ok],
        'three_partition_label': f[part_col].astype(str).values[ok],
        'tone_scale': f[tone_col].values[ok],
        'group': grp, 'split': split,
        'malignant': z['malignant'][ok].astype(int),
        'cluster': cluster, 'row_in_cache': idx_global})
    if 'nine_partition_label' in f.columns:
        rowmeta['nine_partition_label'] = f['nine_partition_label'].astype(str).values[ok]
    if 'label' in f.columns:
        rowmeta['label'] = f['label'].astype(str).values[ok]
    dest_pq = os.path.join(REPO, 'results', 'fitz_rowmeta.parquet')
    rowmeta.to_parquet(dest_pq)
    print('\nsaved', dest_pq)

out = {'n_cohort': int(n),
       'csv_rows_after_filter': int(len(f)), 'row_order_aligned': bool(aligned),
       'tone_agreement_cache_vs_csv': agree,
       'n_pairs_above_thresh': int(n_pairs), 'n_clusters': int(n_clusters),
       'n_multi_image_clusters': int(n_multi), 'n_images_in_multi_clusters': int(n_in_multi),
       'largest_cluster': int(sizes.max()),
       'clusters_straddling_splits': int(leaky),
       'clusters_straddling_train_test': int(leaky_tr_te),
       'test_images_leaked_from_train': int(n_test_leaked),
       'leakage_by_tone': by_tone}
dest = os.path.join(REPO, 'results', 'duplicate_check.json')
json.dump(out, open(dest, 'w'), indent=2)
print('saved', dest)
