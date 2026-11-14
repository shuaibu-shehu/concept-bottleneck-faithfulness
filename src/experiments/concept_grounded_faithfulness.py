"""Monotonicity in support and on annotated images (Section 5.4, criteria (i) and (ii)).

Run:    python -m src.experiments.concept_grounded_faithfulness
Reads:  results/mitigations_supervised.json (the selected penalty weight)
Writes: results/concept_grounded_faithfulness.json
"""
import json
import os
import time

import numpy as np
import torch

from src.concept_layer import GROUPS, concept_scores, fit_probes, load_fitz, signs_for
from src.training import FINAL_SEEDS, fwd, make_trainer
from src.metrics import paired_ci

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def support_levels(S, L, has, train_idx, names):
    """Per concept, the median training score of images annotated absent and present.

    Falls back to 0 and 1 for concepts with too few annotations.
    """
    tr = np.zeros(len(S), dtype=bool)
    tr[train_idx] = True
    lv, fell_back = [], []
    for j in range(S.shape[1]):
        t = L[:, j]
        m = tr & has & np.isfinite(t)
        pos, neg = S[m & (t == 1), j], S[m & (t == 0), j]
        if len(pos) < 20 or len(neg) < 20:
            lv.append((0.0, 1.0)); fell_back.append(names[j]); continue
        lv.append((float(np.median(neg)), float(np.median(pos))))
    return lv, fell_back


def mono_in_support(P, Cm, signs, levels):
    """Monotonicity with concepts set to their observed levels instead of 0 and 1."""
    if len(Cm) == 0:
        return np.array([])
    outs = []
    for i in range(Cm.shape[1]):
        if signs[i] == 0:
            continue
        lo_v, hi_v = levels[i]
        hi, lo = Cm.clone(), Cm.clone()
        a, b = (hi_v, lo_v) if signs[i] > 0 else (lo_v, hi_v)
        hi[:, i], lo[:, i] = a, b
        outs.append((torch.sigmoid(fwd(hi, P)) >= torch.sigmoid(fwd(lo, P)) - 1e-6)
                    .float().detach().numpy())
    return np.mean(outs, axis=0)


def ground_truth_agreement(P, Cm, signs, levels, L_sub, has_sub):
    """Agreement with the clinical direction on image-concept pairs annotated present.

    Returns one rate per image; images without such a pair return NaN.
    """
    n = len(Cm)
    hit = np.zeros(n); cnt = np.zeros(n)
    for i in range(Cm.shape[1]):
        if signs[i] == 0:
            continue
        lo_v, hi_v = levels[i]
        hi, lo = Cm.clone(), Cm.clone()
        a, b = (hi_v, lo_v) if signs[i] > 0 else (lo_v, hi_v)
        hi[:, i], lo[:, i] = a, b
        ok = (torch.sigmoid(fwd(hi, P)) >= torch.sigmoid(fwd(lo, P)) - 1e-6).float().numpy()
        present = has_sub & np.isfinite(L_sub[:, i]) & (L_sub[:, i] == 1)
        hit[present] += ok[present]; cnt[present] += 1
    out = np.full(n, np.nan)
    m = cnt > 0
    out[m] = hit[m] / cnt[m]
    return out


def by_tone(vals, g, mask=None):
    m = np.isfinite(vals) if mask is None else (np.isfinite(vals) & mask)
    rec = {}
    for k in GROUPS:
        s = vals[m & (g == k)]
        rec[k] = float(s.mean()) if len(s) else float('nan')
    rec['gap'] = rec['light'] - rec['dark']
    rec['n'] = {k: int((m & (g == k)).sum()) for k in GROUPS}
    return rec


def main():
    t0 = time.time()
    mit_path = os.path.join(REPO, 'results/mitigations_supervised.json')
    assert os.path.exists(mit_path), \
        'run mitigations_supervised.py first -- the selected gamma comes from it'
    mit = json.load(open(mit_path))
    GAM = mit['gamma_sweep']['selected']['gamma']
    print('using gamma = %g, selected on val in mitigations_supervised.json' % GAM)

    F = load_fitz()
    names = F['names']
    signs = signs_for(names, 'clinical')
    tr, va, te = F['split'] == 'train', F['split'] == 'val', F['split'] == 'test'
    itr = np.where(tr)[0]

    probes = fit_probes(F['emb'], F['L'], F['has'], itr)
    S = concept_scores(probes, F['emb'])
    levels, fell_back = support_levels(S, F['L'], F['has'], itr, names)
    print('support levels estimated on train; fell back to {0,1} for %d concepts%s'
          % (len(fell_back), (': ' + ', '.join(fell_back)) if fell_back else ''))
    scored = [names[j] for j in range(len(names)) if signs[j] != 0]
    print('scored concepts: %d | median (lo, hi) over scored = (%.3f, %.3f)'
          % (len(scored),
             float(np.median([levels[j][0] for j in range(len(names)) if signs[j] != 0])),
             float(np.median([levels[j][1] for j in range(len(names)) if signs[j] != 0]))))

    Ct = torch.tensor(S[tr], dtype=torch.float32)
    Ce = torch.tensor(S[te], dtype=torch.float32)
    Yt = torch.tensor(F['y'][tr].astype(np.float32))
    train = make_trainer(Ct, Yt, F['g'][tr], signs)

    gte = F['g'][te]
    L_te, has_te = F['L'][te], F['has'][te]
    print('test rows %d | SkinCon-annotated among them %d' % (te.sum(), int(has_te.sum())))

    out = {'support_levels': {names[j]: {'lo': levels[j][0], 'hi': levels[j][1],
                                         'sign': int(signs[j])} for j in range(len(names))},
           'conditions': {}}

    per_seed = {}
    for name, kw in [('erm', {}), ('mono_global', {'gamma': GAM})]:
        m = 'erm' if name == 'erm' else 'mono_global'
        A_tone, B_tone, End_tone = [], [], []
        for s in FINAL_SEEDS:
            P, _ = train(m, s, **kw)
            a = mono_in_support(P, Ce, signs, levels)
            b = ground_truth_agreement(P, Ce, signs, levels, L_te, has_te)
            e = mono_in_support(P, Ce, signs, [(0.0, 1.0)] * len(names))
            A_tone.append(by_tone(a, gte)); B_tone.append(by_tone(b, gte))
            End_tone.append(by_tone(e, gte))
        per_seed[name] = {'A': A_tone, 'B': B_tone, 'endpoint': End_tone}

        def agg(rows):
            return {k: float(np.nanmean([r[k] for r in rows])) for k in GROUPS + ['gap']}
        rec = {'in_support_monotonicity': agg(A_tone),
               'ground_truth_agreement': agg(B_tone),
               'endpoint_monotonicity_reference': agg(End_tone),
               'n_by_tone': A_tone[0]['n'],
               'n_by_tone_ground_truth': B_tone[0]['n']}
        out['conditions'][name] = rec
        print('\n== %s ==' % name)
        for lbl, key in [('endpoint (trained-for)', 'endpoint_monotonicity_reference'),
                         ('A in-support   (un-opt)', 'in_support_monotonicity'),
                         ('B ground-truth (un-opt)', 'ground_truth_agreement')]:
            r = rec[key]
            print('  %-24s light %.4f  mid %.4f  dark %.4f  | gap %+.4f'
                  % (lbl, r['light'], r['mid'], r['dark'], r['gap']))

    # paired contrasts with ERM on these criteria
    print('\n== paired mono_global - ERM (95%% bootstrap CI over seeds) ==')
    contrasts = {}
    for key, lbl in [('A', 'in_support_monotonicity'), ('B', 'ground_truth_agreement'),
                     ('endpoint', 'endpoint_monotonicity_reference')]:
        contrasts[lbl] = {}
        for k in GROUPS + ['gap']:
            a = [r[k] for r in per_seed['mono_global'][key]]
            b = [r[k] for r in per_seed['erm'][key]]
            contrasts[lbl][k] = paired_ci(a, b)
        d = contrasts[lbl]['gap']
        dk = contrasts[lbl]['dark']
        print('  %-34s gap %+.4f [%+.4f, %+.4f] | dark %+.4f [%+.4f, %+.4f]'
              % (lbl, d['mean'], d['lo'], d['hi'], dk['mean'], dk['lo'], dk['hi']))
    out['paired_mono_minus_erm'] = contrasts

    verdict = {}
    for lbl in ['in_support_monotonicity', 'ground_truth_agreement']:
        g_erm = out['conditions']['erm'][lbl]['gap']
        g_mono = out['conditions']['mono_global'][lbl]['gap']
        ci = contrasts[lbl]['gap']
        verdict[lbl] = {'gap_erm': g_erm, 'gap_mono': g_mono,
                        'paired_change': ci['mean'], 'ci': [ci['lo'], ci['hi']],
                        'gap_narrowed_significantly': bool(ci['hi'] < 0),
                        'dark_raised_significantly': bool(contrasts[lbl]['dark']['lo'] > 0)}
    out['verdict'] = verdict
    print('\n== verdict on un-optimized criteria ==')
    for lbl, v in verdict.items():
        print('  %-32s gap %.4f -> %.4f  (%s) ; dark raised: %s'
              % (lbl, v['gap_erm'], v['gap_mono'],
                 'narrowed, CI excludes 0' if v['gap_narrowed_significantly'] else 'CI covers 0',
                 'YES' if v['dark_raised_significantly'] else 'no'))

    dest = os.path.join(REPO, 'results/concept_grounded_faithfulness.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
