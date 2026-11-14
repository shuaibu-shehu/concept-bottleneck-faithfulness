"""Two diagnostics: DDI dark-skin AUROC by diagnosis (Section 5.6) and the effect of the sign rule.

Run:    python -m src.experiments.diagnostics
Writes: results/diagnostics.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

from src.concept_layer import (GROUPS, _auc, concept_scores, fit_probes, load_ddi, load_fitz,
                           signs_for)
from src.training import fwd, make_trainer

SEEDS = [0, 1, 2, 3, 4]
TONE2G = {12: 'light', 34: 'mid', 56: 'dark'}


def main():
    t0 = time.time()
    F, D = load_fitz(), load_ddi()
    names = F['names']
    clin = signs_for(names, 'clinical')
    tr, te = F['split'] == 'train', F['split'] == 'test'
    itr = np.where(tr)[0]
    probes = fit_probes(F['emb'], F['L'], F['has'], itr)
    S = concept_scores(probes, F['emb'])
    Sd = concept_scores(probes, D['emb'])
    der = signs_for(names, 'derived', y=F['y'], S=S, train_idx=itr)

    out = {}

    # 1. DDI
    print('=' * 78)
    print('1. DDI DARK-SKIN COLLAPSE: disease spectrum, not skin tone')
    print('=' * 78)

    # 1a. concept detection on DDI by tone
    idx = {n: i for i, n in enumerate(D['names_all'])}
    det = {}
    for j, n in enumerate(names):
        if n not in idx:
            continue
        t = D['L_all'][:, idx[n]]
        m = D['has'] & np.isfinite(t)
        if int(np.nansum(t[m])) < 20:
            continue
        r = {}
        for k in GROUPS:
            mk = m & (D['g'] == k)
            vals = t[mk][np.isfinite(t[mk])]
            r[k] = _auc(t[mk].astype(int), Sd[mk, j]) if len(set(vals)) > 1 else float('nan')
        r['n_pos'] = int(np.nansum(t[m]))
        det[n] = r
    mean_by_tone = {k: float(np.nanmean([det[n][k] for n in det])) for k in GROUPS}
    print('concept DETECTION AUROC on DDI (%d concepts): light %.4f mid %.4f dark %.4f '
          '(L-D %+.4f)' % (len(det), mean_by_tone['light'], mean_by_tone['mid'],
                           mean_by_tone['dark'],
                           mean_by_tone['light'] - mean_by_tone['dark']))
    print('  -> the concept layer is NOT failing dark skin on DDI')

    # 1b. malignancy transfer, with and without mycosis fungoides
    Xt = torch.tensor(S[tr], dtype=torch.float32)
    yt = torch.tensor(F['y'][tr].astype(np.float32))
    train = make_trainer(Xt, yt, F['g'][tr], clin)
    Cd = torch.tensor(Sd, dtype=torch.float32)
    pr = np.mean([torch.sigmoid(fwd(Cd, train('erm', s)[0])).detach().numpy()
                  for s in SEEDS], axis=0)

    meta = pd.read_csv(os.path.join(REPO, 'caches/ddi_metadata.csv'))
    g = np.array([TONE2G[v] for v in meta['skin_tone']])
    assert (g == D['g']).all(), 'ddi_metadata row order disagrees with the cache'
    dx = meta['disease'].astype(str).str.lower().values
    mf = np.char.find(dx.astype(str), 'mycosis') >= 0
    y = D['y']
    print('\nmycosis fungoides in DDI: %d images | by tone %s | malignant %d'
          % (mf.sum(), {k: int((mf & (g == k)).sum()) for k in GROUPS},
             int((mf & (y == 1)).sum())))
    print('  share of each tone group\'s MALIGNANCIES that are MF: %s'
          % {k: ('%.0f%%' % (100.0 * (mf & (g == k) & (y == 1)).sum()
                             / ((g == k) & (y == 1)).sum())
                 if ((g == k) & (y == 1)).sum() else 'n/a') for k in GROUPS})

    def auroc_by_tone(keep):
        return {k: (_auc(y[keep & (g == k)], pr[keep & (g == k)])
                    if len(set(y[keep & (g == k)])) > 1 else float('nan')) for k in GROUPS}
    a_all = auroc_by_tone(np.ones(len(y), bool))
    a_no = auroc_by_tone(~mf)
    print('\n%-32s %8s %8s %8s' % ('DDI malignancy AUROC', 'light', 'mid', 'dark'))
    print('%-32s %8.4f %8.4f %8.4f' % ('all 656 images', a_all['light'], a_all['mid'],
                                       a_all['dark']))
    print('%-32s %8.4f %8.4f %8.4f' % ('excluding mycosis fungoides', a_no['light'],
                                       a_no['mid'], a_no['dark']))
    print('  dark: %.4f -> %.4f  (%+.4f) after removing %d MF images'
          % (a_all['dark'], a_no['dark'], a_no['dark'] - a_all['dark'],
             int((mf & (g == 'dark')).sum())))

    ranks = {'dark_MF_malignant': float(pr[mf & (g == 'dark') & (y == 1)].mean()),
             'dark_other_malignant': float(pr[~mf & (g == 'dark') & (y == 1)].mean()),
             'dark_benign': float(pr[(g == 'dark') & (y == 0)].mean())}
    print('\nmean predicted malignancy, dark group:')
    for k, v in ranks.items():
        print('  %-24s %.4f' % (k, v))
    print('  -> MF malignancies rank BELOW benign lesions: the head has learned '
          '"malignant = tumour-like", and MF is not')

    out['ddi_case_mix'] = {
        'concept_detection_by_tone': {'per_concept': det, 'mean_by_tone': mean_by_tone,
                                      'n_concepts': len(det)},
        'mycosis_fungoides': {
            'n_total': int(mf.sum()),
            'by_tone': {k: int((mf & (g == k)).sum()) for k in GROUPS},
            'n_malignant': int((mf & (y == 1)).sum()),
            'share_of_group_malignancies': {
                k: (float((mf & (g == k) & (y == 1)).sum() / ((g == k) & (y == 1)).sum())
                    if ((g == k) & (y == 1)).sum() else None) for k in GROUPS}},
        'auroc_all': a_all, 'auroc_excluding_mf': a_no,
        'dark_auroc_change': float(a_no['dark'] - a_all['dark']),
        'mean_predicted_prob_dark': ranks,
        'interpretation': ('The DDI dark-skin AUROC collapse is a disease-spectrum shift, '
                           'not a concept-layer failure: detection is equal across tones, '
                           'and removing mycosis fungoides recovers dark AUROC from '
                           'below chance. A second instance of the case-mix confound the '
                           'paper documents for Fitzpatrick.')}

    # 2. sign rule
    print('\n' + '=' * 78)
    print('2. SIGN RULE: decomposing the clinical-vs-derived disagreement')
    print('=' * 78)
    disagree = [names[j] for j in range(len(names)) if clin[j] != 0 and clin[j] != der[j]]
    excluded_scored = [names[j] for j in range(len(names)) if clin[j] == 0]
    print('clinical scores %d concepts; derived scores %d (derived never returns 0)'
          % (int((clin != 0).sum()), len(names)))
    print('  sign disagreements among the %d clinically-scored: %d %s'
          % (int((clin != 0).sum()), len(disagree), disagree))
    print('  concepts clinical EXCLUDES but derived scores: %d' % len(excluded_scored))

    Xe = torch.tensor(S[te], dtype=torch.float32)
    gte = F['g'][te]
    acc = {}
    for s in SEEDS:
        P = train('erm', s)[0]
        for tag, sg in (('clin', clin), ('der', der)):
            for i in range(Xe.shape[1]):
                if sg[i] == 0:
                    continue
                hi, lo = Xe.clone(), Xe.clone()
                a, b = (1.0, 0.0) if sg[i] > 0 else (0.0, 1.0)
                hi[:, i], lo[:, i] = a, b
                with torch.no_grad():
                    ok = (torch.sigmoid(fwd(hi, P)) >= torch.sigmoid(fwd(lo, P)) - 1e-6)
                acc.setdefault(tag, {}).setdefault(i, []).append(ok.float().numpy())

    def subset(tag, idxs):
        M = np.mean([np.mean([acc[tag][i][s] for i in idxs], axis=0)
                     for s in range(len(SEEDS))], axis=0)
        r = {k: float(M[gte == k].mean()) for k in GROUPS}
        r['gap'] = r['light'] - r['dark']
        return r

    rows = {
        'clinical_14_PRIMARY': subset('clin', list(acc['clin'])),
        'derived_all_22': subset('der', list(acc['der'])),
        'derived_same_14': subset('der', [i for i in acc['der'] if clin[i] != 0]),
        'derived_only_the_8_excluded': subset('der', [i for i in acc['der'] if clin[i] == 0]),
    }
    print('\n%-32s %8s %8s %8s %9s' % ('subset', 'light', 'mid', 'dark', 'L-D'))
    for k, r in rows.items():
        print('%-32s %8.4f %8.4f %8.4f %+9.4f' % (k, r['light'], r['mid'], r['dark'], r['gap']))
    print('\n  -> within THIS protocol (5 seeds, test split) the sign rule barely moves '
          'the gap. That does NOT carry over: the derived14 arm of faithfulness_gap_supervised.json, run '
          'out-of-fold over all images with 20 seeds, puts 87% of the clinical-vs-derived '
          'disagreement in the SIGNS. Believe that arm -- it shares a protocol with the '
          'endpoint it explains, and this one does not.')

    per_concept = {}
    for i in sorted(acc['clin']):
        v = np.mean(acc['clin'][i], axis=0)
        r = {k: float(v[gte == k].mean()) for k in GROUPS}
        r['gap'] = r['light'] - r['dark']
        r['sign'] = int(clin[i])
        per_concept[names[i]] = r
    top = sorted(per_concept.items(), key=lambda kv: -abs(kv[1]['gap']))
    print('\nper-concept tone gaps (clinical signs), largest first:')
    print('%-28s %5s %8s %8s %8s %9s' % ('concept', 'sign', 'light', 'mid', 'dark', 'L-D'))
    for n, r in top:
        print('%-28s %5d %8.4f %8.4f %8.4f %+9.4f'
              % (n, r['sign'], r['light'], r['mid'], r['dark'], r['gap']))
    pos = sum(r['gap'] for _, r in top if r['gap'] > 0)
    neg = sum(r['gap'] for _, r in top if r['gap'] < 0)
    print('\n  sum of POSITIVE per-concept gaps %+.4f ; NEGATIVE %+.4f ; net %+.4f'
          % (pos, neg, pos + neg))
    print('  -> the aggregate gap is a near-cancellation of large opposing per-concept '
          'gaps, not a broad phenomenon')

    contradicted = {n: r for n, r in per_concept.items() if r['light'] < 0.25 or r['light'] > 0.99}
    print('\nconcepts where the head\'s conditional behaviour is near-deterministic:')
    for n, r in contradicted.items():
        print('  %-26s sign %+d  monotonicity(light) %.4f  %s'
              % (n, r['sign'], r['light'],
                 'head CONTRADICTS the clinical prior' if r['light'] < 0.25 else
                 'head agrees almost always'))

    out['sign_rule'] = {
        'n_scored_clinical': int((clin != 0).sum()), 'n_scored_derived': len(names),
        'sign_disagreements_among_clinically_scored': disagree,
        'concepts_clinical_excludes': excluded_scored,
        'subsets': rows, 'per_concept_clinical': per_concept,
        'sum_positive_gaps': float(pos), 'sum_negative_gaps': float(neg),
        'interpretation': ('SUPERSEDED, and kept as a record of how it was got wrong. '
                           'This block once concluded that the clinical-vs-derived '
                           'disagreement was a CONCEPT SET difference, because at 5 seeds '
                           'on the test split the two rules agree closely on the same 14 '
                           'concepts. The h3_skincon derived14 arm settles it on the '
                           'endpoint\'s own protocol -- out-of-fold, all images, 20 seeds '
                           '-- and finds the opposite: 87% of the difference is the SIGNS '
                           '(0.0017 clinical -> 0.0144 derived-on-the-same-14), and only '
                           '13% is the eight extra concepts. Two concepts carry it, '
                           'Exudate and Yellow, which track malignancy in this cohort and '
                           'so are signed positive against the clinical prior. Cross-'
                           'protocol reasoning is what produced the wrong answer here.')}

    dest = os.path.join(REPO, 'results/diagnostics.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
