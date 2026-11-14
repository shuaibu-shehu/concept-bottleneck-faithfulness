"""Primary endpoint protocol on the two prompt-read SkinCon layers (Section 5.2, post hoc).

Reads the 22 SkinCon concepts with one prompt pair per concept and with the six-template
ensemble, then applies the endpoint protocol of faithfulness_gap_supervised.py, whose head,
scorer, folds and analyse() are imported unchanged.

Run:    python -m src.experiments.faithfulness_gap_prompt_layers
Reads:  caches/fairness_cache/features_{fitz,ddi}.npz and the DermLIP text encoder
Writes: results/faithfulness_gap_prompt_layers.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from src.concept_layer import GROUPS, assert_no_leakage, load_ddi, load_fitz, signs_for
from src.experiments.faithfulness_gap_supervised import (N_FOLDS, N_SEEDS, analyse, fwd,
                                                         mono_per_sample, train_head)
from src.experiments.prompt_ensemble import ensemble_directions
from src.experiments.zeroshot_vocabulary import encode_prompts, softmax2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

LAYERS = {
    'skincon_single_prompt': 'SkinCon vocabulary, one present/absent prompt pair per concept',
    'skincon_prompt_ensemble': 'SkinCon vocabulary, six paraphrase templates averaged',
}


def prompt_scores(emb, pres, absn):
    S = np.zeros((len(emb), len(pres)))
    for j in range(len(pres)):
        S[:, j] = softmax2(absn[j] @ emb.T, pres[j] @ emb.T)[:, 1]
    return S


def main():
    F, D = load_fitz(), load_ddi()
    n, names = len(F['y']), F['names']
    sg = signs_for(names, 'clinical')
    print('Fitzpatrick cohort: %d | DDI %d | scored concepts %d of %d'
          % (n, len(D['y']), int((sg != 0).sum()), len(names)))

    directions = {'skincon_single_prompt': encode_prompts(names),
                  'skincon_prompt_ensemble': ensemble_directions(names)}

    # identical fold construction to faithfulness_gap_supervised.py
    strata = np.array([GROUPS.index(t) * 2 + int(v) for t, v in zip(F['g'], F['y'])])
    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
                 .split(np.zeros(n), strata, F['cluster']))

    out = {'protocol': 'faithfulness_gap_supervised.py, clinical sign table; post hoc',
           'n_seeds': N_SEEDS, 'n_folds': N_FOLDS, 'duplicate_aware_folds': True,
           'layers': {}}
    t0 = time.time()
    for layer, desc in LAYERS.items():
        pres, absn = directions[layer]
        S = prompt_scores(F['emb'], pres, absn)
        S_ddi = prompt_scores(D['emb'], pres, absn)

        M = np.zeros((N_SEEDS, n), dtype=np.float32)
        Pr = np.zeros((N_SEEDS, n), dtype=np.float32)
        for fi, (itr, ite) in enumerate(folds):
            assert_no_leakage(itr, ite, F['cluster'])
            Xtr = torch.tensor(S[itr], dtype=torch.float32)
            ytr = torch.tensor(F['y'][itr], dtype=torch.float32)
            Xte = torch.tensor(S[ite], dtype=torch.float32)
            for s in range(N_SEEDS):
                P = train_head(Xtr, ytr, s)
                M[s, ite] = mono_per_sample(P, Xte, sg)
                Pr[s, ite] = torch.sigmoid(fwd(Xte, P)).detach().numpy()
            print('  %s: fold %d/%d done (%.0fs)' % (layer, fi + 1, N_FOLDS, time.time() - t0),
                  flush=True)

        # DDI is external: heads are trained on all of Fitzpatrick17k
        M_ddi = np.zeros((N_SEEDS, len(D['g'])), dtype=np.float32)
        Pr_ddi = np.zeros((N_SEEDS, len(D['g'])), dtype=np.float32)
        Xf = torch.tensor(S, dtype=torch.float32)
        yf = torch.tensor(F['y'], dtype=torch.float32)
        Xd = torch.tensor(S_ddi, dtype=torch.float32)
        for s in range(N_SEEDS):
            P = train_head(Xf, yf, s)
            M_ddi[s] = mono_per_sample(P, Xd, sg)
            Pr_ddi[s] = torch.sigmoid(fwd(Xd, P)).detach().numpy()

        out['layers'][layer] = {
            'description': desc,
            'fitz_oof': analyse(M, Pr, F['g'], F['y'],
                                'Fitzpatrick17k out-of-fold, ALL images [%s]' % layer),
            'ddi_external': analyse(M_ddi, Pr_ddi, D['g'], D['y'], 'DDI external [%s]' % layer)}

    dest = os.path.join(REPO, 'results', 'faithfulness_gap_prompt_layers.json')
    json.dump(out, open(dest, 'w'), indent=2)
    print('\nsaved %s  [%.0fs]' % (os.path.relpath(dest, REPO), time.time() - t0))


if __name__ == '__main__':
    main()
