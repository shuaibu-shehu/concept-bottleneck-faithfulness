"""Run every analysis in dependency order, skipping steps whose output already exists.

    python run_pipeline.py                 # run whatever is missing
    python run_pipeline.py --list          # show the plan and stop
    python run_pipeline.py --force NAME    # recompute one step (repeatable)
    python run_pipeline.py --force all     # recompute everything
    python run_pipeline.py --skip-slow     # leave out the long steps

Needs the feature caches written by encode_features.py. Each step writes one file to
results/, and later steps read earlier ones, so the order of STEPS matters. Keep the default
torch thread count: it sets the floating-point reduction order, which shifts monotonicity
by up to 3.5e-3.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

# (module, output file, rough time budget in seconds, title, description)
STEPS = [
    ('src.experiments.duplicate_check', 'duplicate_check.json', 7,
     'Near-duplicate audit',
     'Finds near-duplicate images from the cached embeddings and writes the per-row '
     'metadata every later join uses. Must run first.'),

    ('src.experiments.cohort', 'cohort.json', 7,
     'Cohort composition',
     'Counts the cohort by skin tone, malignancy and split, for both datasets.'),

    ('src.experiments.faithfulness_gap_zeroshot', 'faithfulness_gap_zeroshot.json', 660,
     'Endpoint on the zero-shot concept layer',
     'Runs the out-of-fold endpoint protocol on the seven zero-shot checklist concepts: '
     'per-group monotonicity, the light-dark gap, and its interval.'),

    ('src.experiments.mitigations_zeroshot', 'mitigations_zeroshot.json', 1620,
     'Mitigations on the zero-shot concept layer',
     'ERM plus Group-DRO, resampling, reweighting and per-group thresholds. Group-DRO is '
     'tuned over a 15-config sweep selected on the validation split.'),

    ('src.last_layer', 'last_layer.json', 128,
     'Last-layer retraining: DFR and AFR',
     'Refits the last layer on a group-balanced held-out set, with and without group '
     'labels, and reports accuracy and monotonicity for both.'),

    ('src.prototypes', 'prototypes.json', 879,
     'Diverse prototypical ensembles',
     'Fits the DPE ensemble with its diversity and temperature knobs selected on the '
     'validation split, paired against the same base seeds as DFR.'),

    ('src.concept_layer', 'concept_layer.json', 40,
     'Build the supervised concept layer',
     'Fits one logistic probe per concept on the frozen embedding from expert '
     'annotations and training images only, then scores every image. Reports per-concept '
     'detection AUROC overall and by tone.'),

    ('src.experiments.faithfulness_gap_supervised',
     'faithfulness_gap_supervised.json', 1200,
     'Primary endpoint',
     'The light-dark monotonicity gap out of fold over all 16,012 images, with '
     'duplicate-aware folds and 20 seeds: the mean gap with a bootstrap interval, a '
     'predictive interval and a permutation p-value. Pre-specified.'),

    ('src.experiments.mitigations_supervised', 'mitigations_supervised.json', 3000,
     'Train the head, the mitigations, and the objective',
     'ERM, four mitigations and the monotonicity penalty on the supervised concept '
     'layer, plus a sweep over the penalty weight selected by a rule fixed in advance.'),

    ('src.experiments.concept_grounded_faithfulness',
     'concept_grounded_faithfulness.json', 400,
     'Monotonicity in support and on annotated images',
     'Scores monotonicity between concept states that actually occur, and agreement with '
     'expert annotations. Neither quantity is what the penalty minimises.'),

    ('src.experiments.monotonicity_mechanism', 'monotonicity_mechanism.json', 900,
     'Concept reliance under the penalty',
     'Reports mean absolute intervention effect, how many concepts fall below a '
     'negligible-effect threshold, and how much of the change each concept accounts for, '
     'for the penalised head against ERM.'),

    ('src.experiments.monotone_by_construction', 'monotone_by_construction.json', 900,
     'Monotone-by-construction head',
     'Fits a head constrained to be monotone in the signed concepts, and reports its '
     'accuracy and monotonicity against the penalised and unpenalised heads.'),

    ('src.experiments.diagnostics', 'diagnostics.json', 900,
     'External accuracy by diagnosis, and the sign rule',
     'Breaks the external dark-skin AUROC down by diagnosis, and measures how much of '
     'the clinical-vs-derived difference comes from the signs rather than the concept '
     'set.'),

    ('src.experiments.method_comparison', 'method_comparison.json', 1800,
     'All methods, all datasets, one table',
     'Scores all nine methods on accuracy, worst-group accuracy, monotonicity and the '
     'unoptimized criteria, across the three datasets, with paired intervals against '
     'ERM.'),

    ('src.experiments.cohort_characterisation', 'cohort_characterisation.json', 900,
     'Cohort characterisation',
     'Source atlases and how strongly tone tracks them, concept base rates by tone, and '
     'four feature sets compared on the test split, pooled and within each atlas, with '
     'bootstrap intervals for the within-atlas light-dark gaps.'),

    ('src.experiments.zeroshot_vocabulary', 'zeroshot_vocabulary.json', 900,
     'Vocabulary and readout separated',
     'Reads the 22 SkinCon concepts with one text-prompt pair each, filling the missing '
     'cell between the zero-shot checklist and the supervised probes, and reports concept '
     'detection against the annotations and the head trained on each layer.'),

    ('src.experiments.prompt_ensemble', 'prompt_ensemble.json', 900,
     'Prompt ensembling',
     'Repeats the SkinCon prompt layer with six paraphrase templates averaged, to test '
     'whether the prompt readout fails as a technique or only with one wording.'),

    ('src.experiments.readout_comparison', 'readout_comparison.json', 600,
     'Seven feature sets, one logistic regression each',
     'One balanced logistic regression per feature set, fit on the training rows and '
     'scored on the test split (Table 2 and Figure 2a).'),

    ('src.experiments.faithfulness_gap_prompt_layers', 'faithfulness_gap_prompt_layers.json',
     1500,
     'Endpoint protocol on the prompt-read layers',
     'Puts both prompt-read SkinCon layers through the primary endpoint protocol, so all '
     'four concept layers share one estimand. Post hoc.'),

    ('src.experiments.sign_agreement', 'sign_agreement.json', 120,
     'Clinical signs against the data',
     'Compares each declared clinical sign with the direction the training labels imply in '
     'each fold, with an exact binomial test against arbitrary signs.'),

    ('src.experiments.calibration_operating_point', 'calibration_operating_point.json', 4200,
     'Per-tone operating point and calibration',
     'Retrains the nine Table 1 methods and reports sensitivity and specificity by tone at '
     'the shared and per-tone thresholds, and calibration by tone before and after one '
     'tone-agnostic recalibration.'),
]

# long steps, left out by --skip-slow
SLOW = {'src.experiments.mitigations_zeroshot',
        'src.experiments.faithfulness_gap_supervised',
        'src.experiments.mitigations_supervised',
        'src.experiments.method_comparison',
        'src.experiments.faithfulness_gap_prompt_layers',
        'src.experiments.calibration_operating_point'}


def rule(ch='-'):
    print(ch * 78)


def run(script):
    """Execute one step; return True if it succeeded."""
    t0 = time.time()
    r = subprocess.run([sys.executable, '-m', script], cwd=HERE)
    dt = time.time() - t0
    if r.returncode:
        print('  FAILED after %.0fs (exit %d)' % (dt, r.returncode))
        return False
    print('  done in %.0fs' % dt)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--force', action='append', default=[])
    ap.add_argument('--skip-slow', action='store_true')
    a = ap.parse_args()

    force_all = 'all' in a.force
    forced = set(a.force)

    plan = []
    for script, out, budget, title, why in STEPS:
        if a.skip_slow and script in SLOW:
            continue
        have = os.path.exists(os.path.join(RESULTS, out))
        todo = force_all or script in forced or not have
        plan.append((script, title, why, todo))

    rule('=')
    print('ANALYSIS PLAN  (%d steps, %d to run)' % (len(plan), sum(1 for p in plan if p[-1])))
    rule('=')
    for script, title, why, todo in plan:
        print('  [%s] %-44s %s' % ('run ' if todo else 'skip', script, title))
    if a.list:
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    failed = []
    for script, title, why, todo in plan:
        if not todo:
            continue
        rule()
        print('%s  (%s)' % (title, script))
        print('  %s' % why)
        if not run(script):
            failed.append(script)
            print('  -> later steps may depend on this; continuing so the rest still run')

    rule('=')
    if failed:
        print('FAILED: %s' % ', '.join(failed))
        return 1
    print('pipeline complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
