"""Does the penalised head still rely on its concepts? Effect sizes and attribution (Section 5.5).

Effects are averaged over seeds before thresholding, so values differ slightly from the
per-head means in mitigations_supervised.json.

Run:    python -m src.experiments.monotonicity_mechanism
Reads:  results/mitigations_supervised.json (the selected penalty weight)
Writes: results/monotonicity_mechanism.json
"""
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

from src.concept_layer import GROUPS, concept_scores, fit_probes, load_fitz, signs_for
from src.training import fwd, make_trainer
from src.metrics import auc_np

SEEDS = [0, 1, 2, 3, 4]
NEGLIGIBLE = 0.01
# pre-specified pass thresholds
MIN_EFFECT_RATIO = 0.5
MAX_SINGLE_CONCEPT_SHARE = 0.60


def effects(P, Cm, signs):
    """Signed intervention effect p(asserted) - p(negated) per concept and sample.

    Asserting a sign -1 concept sets its score to 0, so a head that respects every direction
    has non-negative effects.
    """
    out = []
    for i in range(Cm.shape[1]):
        if signs[i] == 0:
            continue
        hi, lo = Cm.clone(), Cm.clone()
        a, b = (1.0, 0.0) if signs[i] > 0 else (0.0, 1.0)
        hi[:, i], lo[:, i] = a, b
        with torch.no_grad():
            out.append((torch.sigmoid(fwd(hi, P)) - torch.sigmoid(fwd(lo, P))).numpy())
    return np.array(out)


def summarise(E):
    """E: (n_scored, n_samples) signed effects from the seed ensemble."""
    return {'violation_rate_per_concept': [float(v) for v in (E < 0).mean(1)],
            'mean_effect_per_concept': [float(v) for v in E.mean(1)],
            'mean_abs_effect_per_concept': [float(v) for v in np.abs(E).mean(1)],
            'frac_negligible_per_concept': [float(v) for v in (np.abs(E) < NEGLIGIBLE).mean(1)],
            'monotonicity': float((E >= -1e-6).mean()),
            'mean_abs_effect': float(np.abs(E).mean())}


def main():
    t0 = time.time()
    mit = json.load(open(os.path.join(REPO, 'results/mitigations_supervised.json')))
    GAM = mit['gamma_sweep']['selected']['gamma']
    print('gamma = %g (validation-selected in mitigations_supervised.json)' % GAM)

    F = load_fitz()
    names = F['names']
    signs = signs_for(names, 'clinical')
    scored = [names[j] for j in range(len(names)) if signs[j] != 0]
    sgn = [int(signs[j]) for j in range(len(names)) if signs[j] != 0]
    tr, te = F['split'] == 'train', F['split'] == 'test'
    probes = fit_probes(F['emb'], F['L'], F['has'], np.where(tr)[0])
    S = concept_scores(probes, F['emb'])

    Xtr = torch.tensor(S[tr], dtype=torch.float32)
    Xte = torch.tensor(S[te], dtype=torch.float32)
    ytr = torch.tensor(F['y'][tr].astype(np.float32))
    train = make_trainer(Xtr, ytr, F['g'][tr], signs)
    gte, yte = F['g'][te], F['y'][te]
    print('scored concepts: %d of %d | test rows %d' % (len(scored), len(names), te.sum()))

    heads = {'erm': [train('erm', s)[0] for s in SEEDS],
             'mono': [train('mono_global', s, gamma=GAM)[0] for s in SEEDS]}
    E = {k: np.mean([effects(P, Xte, signs) for P in v], axis=0) for k, v in heads.items()}
    rec = {k: summarise(v) for k, v in E.items()}

    # AUROC of each concept score against malignancy, for context (not concept detection)
    sep = []
    k = 0
    for j in range(len(names)):
        if signs[j] == 0:
            continue
        sep.append(auc_np(yte, S[te][:, j])); k += 1

    # share of the reduction in violations per concept
    dv = np.array(rec['erm']['violation_rate_per_concept']) - \
        np.array(rec['mono']['violation_rate_per_concept'])
    share = dv / dv.sum() if dv.sum() != 0 else dv * 0

    print('\nPER-CONCEPT: violation rate and intervention-effect MAGNITUDE, ERM vs +mono')
    print('%-26s %4s %7s %7s %7s %8s %8s %7s %7s'
          % ('concept', 'sgn', 'sepAUC', 'vERM', 'vMONO', '|e|ERM', '|e|MONO', 'negl%', 'share%'))
    for j, n in enumerate(scored):
        print('%-26s %4d %7.3f %7.3f %7.3f %8.4f %8.4f %7.1f %7.1f'
              % (n, sgn[j], sep[j],
                 rec['erm']['violation_rate_per_concept'][j],
                 rec['mono']['violation_rate_per_concept'][j],
                 rec['erm']['mean_abs_effect_per_concept'][j],
                 rec['mono']['mean_abs_effect_per_concept'][j],
                 100 * rec['mono']['frac_negligible_per_concept'][j],
                 100 * share[j]))

    ratio = rec['mono']['mean_abs_effect'] / rec['erm']['mean_abs_effect']
    print('\nmonotonicity (seed-ensemble)  ERM %.4f -> MONO %.4f'
          % (rec['erm']['monotonicity'], rec['mono']['monotonicity']))
    print('mean |effect|                 ERM %.4f -> MONO %.4f   (%.2fx)'
          % (rec['erm']['mean_abs_effect'], rec['mono']['mean_abs_effect'], ratio))

    # per tone group
    per_tone = {}
    for g in GROUPS:
        m = gte == g
        per_tone[g] = {k: {'monotonicity': float((E[k][:, m] >= -1e-6).mean()),
                           'mean_abs_effect': float(np.abs(E[k][:, m]).mean())}
                       for k in ('erm', 'mono')}
    print('\nper tone   mono: %s' % ' '.join(
        '%s %.4f->%.4f' % (g, per_tone[g]['erm']['monotonicity'],
                           per_tone[g]['mono']['monotonicity']) for g in GROUPS))
    print('per tone   |e| : %s' % ' '.join(
        '%s %.4f->%.4f' % (g, per_tone[g]['erm']['mean_abs_effect'],
                           per_tone[g]['mono']['mean_abs_effect']) for g in GROUPS))

    # pre-specified pass conditions
    newly_negligible = [scored[j] for j in range(len(scored))
                        if rec['mono']['frac_negligible_per_concept'][j] > 0.90
                        and rec['erm']['frac_negligible_per_concept'][j] <= 0.90]
    top_share = float(np.max(share)) if len(share) else 0.0
    top_concept = scored[int(np.argmax(share))] if len(share) else None
    c1 = bool(ratio >= MIN_EFFECT_RATIO)
    c2 = bool(not newly_negligible)
    c3 = bool(top_share <= MAX_SINGLE_CONCEPT_SHARE)
    verdict = ('REPAIR -- all three declared conditions met' if (c1 and c2 and c3)
               else 'CONSTRAINT-SATISFACTION -- at least one condition failed; report '
                    '0.986 as constraint satisfaction, not repair, in the abstract')
    print('\n' + '=' * 78)
    print('PASS CONDITIONS')
    print('=' * 78)
    print('  1. |effect| ratio >= %.2f            : %.2fx  -> %s'
          % (MIN_EFFECT_RATIO, ratio, 'PASS' if c1 else 'FAIL'))
    print('  2. no NEWLY negligible concept       : %s  -> %s'
          % (newly_negligible if newly_negligible else 'none', 'PASS' if c2 else 'FAIL'))
    print('  3. top concept share <= %.0f%%         : %.1f%% (%s) -> %s'
          % (100 * MAX_SINGLE_CONCEPT_SHARE, 100 * top_share, top_concept,
             'PASS' if c3 else 'FAIL'))
    print('\n  VERDICT: %s' % verdict)

    out = {'scored_concepts': scored, 'signs': sgn,
           'separability_auroc_vs_malignancy': [float(v) for v in sep],
           'separability_caveat': ('AUROC of a concept score against the MALIGNANCY '
                                   'label, entangled with case mix. NOT concept '
                                   'detection quality -- see concept_layer.json.'),
           'erm': rec['erm'], 'mono': rec['mono'],
           'effect_ratio_mono_over_erm': float(ratio),
           'violation_reduction_share_per_concept': [float(v) for v in share],
           'per_tone': per_tone,
           'pass_conditions': {
               'effect_ratio': {'value': float(ratio), 'threshold': MIN_EFFECT_RATIO,
                                'pass': c1},
               'no_newly_negligible': {'newly_negligible': newly_negligible, 'pass': c2},
               'top_concept_share': {'value': top_share, 'concept': top_concept,
                                     'threshold': MAX_SINGLE_CONCEPT_SHARE, 'pass': c3},
               'all_pass': bool(c1 and c2 and c3)},
           'verdict': verdict}
    dest = os.path.join(REPO, 'results/monotonicity_mechanism.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
