"""Characterise the cohort: source atlases, tone composition and concept base rates.

Run:    python -m src.experiments.cohort_characterisation
Writes: results/cohort_characterisation.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

GROUPS = ['light', 'mid', 'dark']
# ordinal colour ramp for the three tone groups
TONE_C = {'light': '#86b6ef', 'mid': '#2a78d6', 'dark': '#104281'}
ATLAS_C = {'dermaamin.com': '#2a78d6', 'atlasdermatologico.com.br': '#eb6834'}


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    from src.concept_layer import load_fitz, fit_probes, concept_scores, signs_for

    F = load_fitz()
    names, L, has = F['names'], F['L'], F['has']
    g, y, sp = F['g'], F['y'], F['split']
    n = len(y)

    rm = pd.read_parquet(os.path.join(REPO, 'results/fitz_rowmeta.parquet'))
    lab = pd.read_csv(os.path.join(REPO, 'caches/fitzpatrick17k.csv')).set_index('md5hash')
    src = (lab['url'].astype(str).str.extract(r'https?://([^/]+)')[0]
           .reindex(rm['md5hash']).to_numpy())
    atlas = np.where(src == 'atlasdermatologico.com.br', 'atlasderm', 'dermaamin')
    dx = lab['label'].reindex(rm['md5hash']).astype(str).to_numpy()
    cluster = rm['cluster'].to_numpy()
    tp = rm['three_partition_label'].to_numpy()

    out = {}

    # 1. composition
    print('=' * 78)
    print('1. COHORT COMPOSITION')
    print('=' * 78)
    comp = {}
    print('%-8s %7s %9s %9s %9s %9s' % ('tone', 'n', 'malig%', 'atlas%', 'nonneo%', 'skincon%'))
    for k in GROUPS:
        m = g == k
        comp[k] = {'n': int(m.sum()), 'malignant_pct': float(100 * y[m].mean()),
                   'atlasderm_pct': float(100 * (atlas[m] == 'atlasderm').mean()),
                   'non_neoplastic_pct': float(100 * (tp[m] == 'non-neoplastic').mean()),
                   'skincon_annotated_pct': float(100 * has[m].mean())}
        c = comp[k]
        print('%-8s %7d %9.1f %9.1f %9.1f %9.1f'
              % (k, c['n'], c['malignant_pct'], c['atlasderm_pct'],
                 c['non_neoplastic_pct'], c['skincon_annotated_pct']))
    out['composition_by_tone'] = comp

    # 2. atlas confound
    print()
    print('=' * 78)
    print('2. THE ATLAS CONFOUND')
    print('=' * 78)
    a_is = atlas == 'atlasderm'
    print('dermaamin %d | atlasdermatologico %d' % ((~a_is).sum(), a_is.sum()))
    print('malignancy rate: dermaamin %.3f | atlasderm %.3f' % (y[~a_is].mean(), y[a_is].mean()))

    tr, te = sp == 'train', sp == 'test'
    E, C7 = F['emb'], F['C7'].astype(np.float64)
    pa = LogisticRegression(max_iter=4000, class_weight='balanced',
                            random_state=0).fit(E[tr], a_is[tr])
    a_emb = auc(a_is[te].astype(int), pa.predict_proba(E[te])[:, 1])
    pc = LogisticRegression(max_iter=4000, class_weight='balanced',
                            random_state=0).fit(C7[tr], a_is[tr])
    a_c7 = auc(a_is[te].astype(int), pc.predict_proba(C7[te])[:, 1])
    print('atlas predictable from the 512-d embedding: AUROC %.4f' % a_emb)
    print('  ... from the 7 zero-shot concepts only  : AUROC %.4f' % a_c7)
    print('  -> the backbone encodes WHICH WEBSITE the image came from almost perfectly')

    # the main contrasts, re-measured within each atlas
    probes = fit_probes(F['emb'], L, has, np.where(tr)[0])
    S22 = concept_scores(probes, F['emb'])
    # PCA with 7 components: a width control without clinical meaning, fit on train only
    pca = PCA(n_components=7, random_state=0).fit(E[tr])
    P7 = pca.transform(E)
    feats = {'raw embedding (512-d)': E, '7 zero-shot concepts': C7,
             '7 unsupervised PCA': P7, '22 supervised SkinCon': S22}
    within, preds = {}, {}
    subsets = [('all', te), ('dermaamin', te & ~a_is), ('atlasderm', te & a_is)]
    print()
    print('%-24s %-16s %8s %8s %8s %9s'
          % ('features', 'subset', 'light', 'mid', 'dark', 'L-D') + '   pooled')
    for fname, X in feats.items():
        lr = LogisticRegression(max_iter=5000, class_weight='balanced',
                                random_state=0).fit(X[tr], y[tr])
        p = np.zeros(n); p[te] = lr.predict_proba(X[te])[:, 1]
        preds[fname] = p
        within[fname] = {}
        for sub, msk in subsets:
            d = {k: auc(y[msk & (g == k)], p[msk & (g == k)]) for k in GROUPS}
            d['gap'] = d['light'] - d['dark']
            # pooled and worst-group AUROC on the same split as the gap
            d['auroc'] = auc(y[msk], p[msk])
            d['worst'] = min(d[k] for k in GROUPS)
            d['n'] = int(msk.sum())
            within[fname][sub] = d
            print('%-24s %-16s %8.4f %8.4f %8.4f %+9.4f %8.4f'
                  % (fname if sub == 'all' else '', sub, d['light'], d['mid'],
                     d['dark'], d['gap'], d['auroc']))

    # bootstrap intervals for the within-atlas gaps (test images resampled within each tone)
    counts = {sub: {k: {'n': int((msk & (g == k)).sum()),
                        'n_malignant': int((msk & (g == k) & (y == 1)).sum())}
                    for k in GROUPS}
              for sub, msk in subsets}
    B = 2000
    boot = {}
    print()
    print('light-dark gap, %d bootstrap draws over test images (95%% interval)' % B)
    for sub, msk in subsets:
        rng = np.random.default_rng(0)
        idx = {k: np.where(msk & (g == k))[0] for k in ('light', 'dark')}
        draws = {f: np.empty(B) for f in feats}
        for b in range(B):
            s = {k: rng.choice(v, len(v)) for k, v in idx.items()}
            for f in feats:
                draws[f][b] = (auc(y[s['light']], preds[f][s['light']])
                               - auc(y[s['dark']], preds[f][s['dark']]))
        diff = draws['7 zero-shot concepts'] - draws['22 supervised SkinCon']
        diff = diff[np.isfinite(diff)]
        ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
        boot[sub] = {
            'gap_ci95': {f: ci(v) for f, v in draws.items()},
            'checklist_minus_probes': {
                'point': (within['7 zero-shot concepts'][sub]['gap']
                          - within['22 supervised SkinCon'][sub]['gap']),
                'ci95': ci(diff), 'share_above_zero': float((diff > 0).mean()),
                'n_draws': int(len(diff))}}
        c = boot[sub]['checklist_minus_probes']
        print('  %-10s malignant dark test images %3d | checklist - probes %+.3f [%+.3f, %+.3f]'
              % (sub, counts[sub]['dark']['n_malignant'], c['point'], c['ci95'][0],
                 c['ci95'][1]))

    cl, pr = within['7 zero-shot concepts'], within['22 supervised SkinCon']
    out['atlas'] = {
        'counts': {'dermaamin': int((~a_is).sum()), 'atlasderm': int(a_is.sum())},
        'malignancy_rate': {'dermaamin': float(y[~a_is].mean()),
                            'atlasderm': float(y[a_is].mean())},
        'predictability_auroc': {'from_embedding': a_emb, 'from_7_concepts': a_c7},
        'within_atlas_auroc': within,
        'within_atlas_test_counts': counts,
        'within_atlas_gap_bootstrap': boot,
        'bootstrap_protocol': ('%d draws of the test images, resampled with replacement '
                               'within light and within dark skin, the fitted probes held '
                               'fixed; percentile intervals' % B),
        'interpretation': (
            'Atlas is predictable from the embedding at %.4f AUROC and is associated with '
            'tone (dark is %.0f%% atlasderm vs light %.0f%%) and label (malignancy %.1f%% '
            'on dermaamin vs %.1f%% on atlasderm). Within atlas, the checklist gap is %+.4f '
            'against %+.4f for the '
            'probes on dermaamin and %+.4f against %+.4f on atlasderm, on %d and %d '
            'malignant dark-skin test images. The bootstrap 95%% intervals of the '
            'difference are [%+.3f, %+.3f] and [%+.3f, %+.3f]. The raw embedding gap is '
            '%+.4f pooled and %+.4f / %+.4f within atlas.'
            % (a_emb, 100 * (atlas[g == 'dark'] == 'atlasderm').mean(),
               100 * (atlas[g == 'light'] == 'atlasderm').mean(),
               100 * y[~a_is].mean(), 100 * y[a_is].mean(),
               cl['dermaamin']['gap'], pr['dermaamin']['gap'],
               cl['atlasderm']['gap'], pr['atlasderm']['gap'],
               counts['dermaamin']['dark']['n_malignant'],
               counts['atlasderm']['dark']['n_malignant'],
               *boot['dermaamin']['checklist_minus_probes']['ci95'],
               *boot['atlasderm']['checklist_minus_probes']['ci95'],
               within['raw embedding (512-d)']['all']['gap'],
               within['raw embedding (512-d)']['dermaamin']['gap'],
               within['raw embedding (512-d)']['atlasderm']['gap']))}


    # 3. SkinCon subset
    print()
    print('=' * 78)
    print('3. IS THE SkinCon-ANNOTATED SUBSET REPRESENTATIVE?')
    print('=' * 78)
    rep = {}
    print('%-22s %12s %12s %10s' % ('', 'annotated', 'unannotated', 'diff'))
    rows = [('n', has.sum(), (~has).sum(), None),
            ('malignant %', 100 * y[has].mean(), 100 * y[~has].mean(), None),
            ('dark %', 100 * (g[has] == 'dark').mean(), 100 * (g[~has] == 'dark').mean(), None),
            ('light %', 100 * (g[has] == 'light').mean(), 100 * (g[~has] == 'light').mean(), None),
            ('atlasderm %', 100 * (atlas[has] == 'atlasderm').mean(),
             100 * (atlas[~has] == 'atlasderm').mean(), None),
            ('non-neoplastic %', 100 * (tp[has] == 'non-neoplastic').mean(),
             100 * (tp[~has] == 'non-neoplastic').mean(), None)]
    for lbl, a_, b_, _ in rows:
        d = None if lbl == 'n' else a_ - b_
        rep[lbl] = {'annotated': float(a_), 'unannotated': float(b_),
                    'diff': None if d is None else float(d)}
        print('%-22s %12.1f %12.1f %10s'
              % (lbl, a_, b_, '' if d is None else '%+.1f' % d))
    out['skincon_representativeness'] = rep
    big = [k for k, v in rep.items() if v['diff'] is not None and abs(v['diff']) > 5]
    print('  -> attributes differing by >5 points: %s' % (big if big else 'none'))
    out['skincon_representativeness_flags'] = big

    # 4. split hygiene
    print()
    print('=' * 78)
    print('4. SPLIT HYGIENE')
    print('=' * 78)
    cells = {}
    print('%-8s %8s %8s %8s   %s' % ('split', 'n', 'malig%', 'dark%', 'dark x malignant'))
    for s in ['train', 'val', 'test']:
        m = sp == s
        dm = int(((g == 'dark') & (y == 1) & m).sum())
        cells[s] = {'n': int(m.sum()), 'malignant_pct': float(100 * y[m].mean()),
                    'dark_pct': float(100 * (g[m] == 'dark').mean()),
                    'dark_malignant_n': dm}
        print('%-8s %8d %8.1f %8.1f   %d' % (s, m.sum(), 100 * y[m].mean(),
                                             100 * (g[m] == 'dark').mean(), dm))
    # duplicate clusters straddling splits
    df = pd.DataFrame({'cluster': cluster, 'split': sp})
    straddle = df.groupby('cluster')['split'].nunique()
    n_straddle = int((straddle > 1).sum())
    leaked = int(df[df['cluster'].isin(straddle[straddle > 1].index)].shape[0])
    print('duplicate clusters straddling splits: %d (%d images)' % (n_straddle, leaked))
    print('  -> the dark x malignant VAL cell (%d) is DFR\'s binding constraint'
          % cells['val']['dark_malignant_n'])
    out['split_hygiene'] = {'cells': cells, 'clusters_straddling_splits': n_straddle,
                            'images_in_straddling_clusters': leaked}

    # 5. diagnoses
    print()
    print('=' * 78)
    print('5. DIAGNOSIS COMPOSITION BY TONE (the Fitzpatrick analogue of DDI mycosis fungoides)')
    print('=' * 78)
    dxo = {}
    for k in GROUPS:
        m = (g == k) & (y == 1)
        vc = pd.Series(dx[m]).value_counts().head(5)
        dxo[k] = {'n_malignant': int(m.sum()),
                  'top_malignant_diagnoses': {a: int(b) for a, b in vc.items()}}
        print('%s (n=%d malignant): %s'
              % (k, m.sum(), ', '.join('%s(%d)' % (a, b) for a, b in vc.items())))
    # share of each group's malignancies in its most common diagnosis
    for k in GROUPS:
        tot = dxo[k]['n_malignant']
        top = max(dxo[k]['top_malignant_diagnoses'].values()) if tot else 0
        dxo[k]['top_diagnosis_share'] = float(top / tot) if tot else float('nan')
        print('  %-6s top diagnosis is %.0f%% of the group\'s malignancies'
              % (k, 100 * dxo[k]['top_diagnosis_share']))
    out['diagnosis_by_tone'] = dxo

    # 6. concept prevalence
    print()
    print('=' * 78)
    print('6. CONCEPT PREVALENCE BY TONE (SkinCon-annotated images)')
    print('=' * 78)
    signs = signs_for(names, 'clinical')
    prev = {}
    print('%-26s %5s %8s %8s %8s %9s' % ('concept', 'sign', 'light', 'mid', 'dark', 'L-D'))
    for j, nm in enumerate(names):
        r = {}
        for k in GROUPS:
            m = has & (g == k) & np.isfinite(L[:, j])
            r[k] = float(L[m, j].mean()) if m.any() else float('nan')
        r['gap'] = r['light'] - r['dark']
        r['sign'] = int(signs[j])
        prev[nm] = r
    for nm, r in sorted(prev.items(), key=lambda kv: -abs(kv[1]['gap']))[:10]:
        print('%-26s %5d %8.3f %8.3f %8.3f %+9.3f'
              % (nm, r['sign'], r['light'], r['mid'], r['dark'], r['gap']))
    out['concept_prevalence_by_tone'] = prev


    # 7. score distributions
    print()
    print('=' * 78)
    print('7. CONCEPT SCORE DISTRIBUTIONS (zero-shot vs supervised)')
    print('=' * 78)
    dists = {}
    for nm, X in [('zero_shot_7', C7), ('supervised_22', S22)]:
        insup = float(((X > 0.05) & (X < 0.95)).mean())
        dists[nm] = {'frac_in_0.05_0.95': insup,
                     'frac_below_0.05': float((X < 0.05).mean()),
                     'frac_above_0.95': float((X > 0.95).mean()),
                     'median_per_concept': [float(v) for v in np.median(X, 0)],
                     'sd_per_concept': [float(v) for v in X.std(0)]}
        print('%-16s in (0.05,0.95): %5.1f%%   saturated low %4.1f%%  high %4.1f%%  '
              'mean sd %.3f'
              % (nm, 100 * insup, 100 * dists[nm]['frac_below_0.05'],
                 100 * dists[nm]['frac_above_0.95'], float(X.std(0).mean())))
    print('  -> the {0,1} intervention endpoints extrapolate wherever coverage is high')
    out['score_distributions'] = dists


    dest = os.path.join(REPO, 'results/cohort_characterisation.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
