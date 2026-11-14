"""The concept-to-malignancy head and the objectives used to train it.

A two-layer MLP on the concept scores. mono_terms is the monotonicity penalty: for each
signed concept, the amount by which asserting it lowers predicted malignancy.
"""
import numpy as np
import torch

from src.concept_layer import GROUPS

# Head width, Adam steps and learning rate. Fixed across every experiment.
HID, ITERS, LR = 16, 1500, 0.05

# Seeds: a small set while sweeping hyper-parameters, ten for the reported runs.
SWEEP_SEEDS, FINAL_SEEDS = [0, 1, 2], list(range(10))

# Group-DRO step sizes and weight decays swept on the validation split.
ETAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
WDS = [0.0, 1e-4, 1e-2]

# Penalty weights, and how far validation worst-group AUROC may fall below ERM before a
# setting is disqualified.
GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
GAMMA_AUROC_FLOOR = 0.01


# --- model ---
def fwd(Cm, P):
    W1, b1, W2, b2 = P
    return (torch.relu(Cm @ W1 + b1) @ W2 + b2).squeeze(1)


def mono_terms(base, P, signs):
    """Per-sample hinge on monotonicity violations, averaged over the signed concepts only."""
    pen, k = 0.0, 0
    for i in range(base.shape[1]):
        if signs[i] == 0:
            continue
        hi, lo = base.clone(), base.clone()
        a, b = (1.0, 0.0) if signs[i] > 0 else (0.0, 1.0)
        hi[:, i], lo[:, i] = a, b
        pen = pen + torch.relu(torch.sigmoid(fwd(lo, P)) - torch.sigmoid(fwd(hi, P)))
        k += 1
    return pen / max(k, 1)


def make_trainer(Xtr, ytr, gtr, signs):
    gmask_tr = [torch.tensor(gtr == g) for g in GROUPS]

    def train(method, seed, eta=1.0, wd=0.0, gamma=2.0):
        gg = torch.Generator().manual_seed(seed)
        P = [(torch.randn(Xtr.shape[1], HID, generator=gg) * 0.3).requires_grad_(True),
             torch.zeros(HID, requires_grad=True),
             (torch.randn(HID, 1, generator=gg) * 0.3).requires_grad_(True),
             torch.zeros(1, requires_grad=True)]
        opt = torch.optim.Adam(P, lr=LR, weight_decay=wd)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        pw = torch.tensor(max(float((ytr == 0).sum()), 1.0) / max(float((ytr == 1).sum()), 1.0))
        q = torch.ones(len(GROUPS)) / len(GROUPS)

        if method == 'resample':
            rs = np.random.default_rng(seed)
            per = int(max(m.sum() for m in gmask_tr))
            idx = torch.tensor(np.concatenate(
                [rs.choice(np.where(m.numpy())[0], per, replace=True) for m in gmask_tr]))
        if method == 'reweight':
            inv = torch.tensor([1.0 / max(float(m.sum()), 1.0) for m in gmask_tr])
            w = torch.zeros(len(ytr))
            for k, m in enumerate(gmask_tr):
                w[m] = inv[k]
            w = w / w.mean()

        for _ in range(ITERS):
            if method == 'resample':
                loss = bce(fwd(Xtr[idx], P), ytr[idx], pos_weight=pw, reduction='none').mean()
            else:
                per_loss = bce(fwd(Xtr, P), ytr, pos_weight=pw, reduction='none')
                if method == 'erm':
                    loss = per_loss.mean()
                elif method == 'reweight':
                    loss = (per_loss * w).mean()
                elif method.startswith('gdro'):
                    lg = torch.stack([per_loss[m].mean() if m.any() else torch.tensor(0.0)
                                      for m in gmask_tr])
                    q = q * torch.exp(eta * lg.detach())
                    q = q / q.sum()
                    loss = (q * lg).sum()
                elif method == 'mono_global':
                    loss = per_loss.mean() + gamma * mono_terms(Xtr, P, signs).mean()
                elif method == 'mono_pergroup':
                    mt = mono_terms(Xtr, P, signs)
                    per_g = torch.stack([mt[m].mean() if m.any() else torch.tensor(0.0)
                                         for m in gmask_tr])
                    loss = per_loss.mean() + gamma * per_g.max()
                else:
                    raise ValueError(method)
            opt.zero_grad(); loss.backward(); opt.step()
        return P, q.detach().numpy().tolist()
    return train
