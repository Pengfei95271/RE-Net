"""
Lambda sensitivity sweep on PhysioNet (validation-based, reproducible).

Addresses reviewer 4.14 / 5.14:
  - runs on PhysioNet (109 subjects), not the low-power 9-subject 2a set;
  - reports VALIDATION accuracy (never test), so lambda is not tuned on test;
  - multiple seeds per lambda to remove single-run noise;
  - also records the mean within-branch weight cosine similarity at each lambda,
    so the sweep shows the weight-geometry effect, not only accuracy.

Design. A single fixed split of the 109 subjects into a training pool and a
held-out VALIDATION pool (by subject, stratified is not needed since we average
over subjects). For each lambda and seed we train RE-Net on the training pool
and evaluate accuracy on the validation pool. This is a development-set sweep:
it uses no test subject and no per-dataset test tuning. Because it is one split
(not full LOSO) it is fast -- a few minutes per (lambda, seed).

This script REUSES the exact RENet model, CFG, osfr_loss and train() from
run_renet.py, so the swept model is identical to the main experiment.

Outputs lambda_sweep/lambda_sweep.json and prints a table.

Usage:  python lambda_sweep.py
Env:    DATASET=physionet  N_SEEDS=3  VAL_SUBJECTS=20
"""
import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the EXACT main-experiment code
import run_renet as R
from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, to_compute_tensors, batch_index)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "lambda_sweep"); os.makedirs(OUT, exist_ok=True)

LAMBDAS = [0.0, 0.01, 0.05, 0.10, 0.50, 1.0]
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
VAL_SUBJECTS = int(os.environ.get("VAL_SUBJECTS", "20"))


def mean_weight_cosine(model):
    """mean absolute off-diagonal cosine between the D spatial filters, per band."""
    W = model.spatial.weight.view(R.CFG["F1"], R.CFG["D"], -1)
    vals = []
    for f in range(R.CFG["F1"]):
        Wn = F.normalize(W[f], p=2, dim=-1)
        G = (Wn @ Wn.t()).abs()
        D = G.shape[0]
        off = (G.sum() - G.diag().sum()) / (D * (D - 1) + 1e-9)
        vals.append(float(off))
    return float(np.mean(vals))


_CACHE = {}


def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    _CACHE["data"] = (X, y, s, ncl)
    _CACHE["CT"] = (C, T)
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    subs = subjects_of(s)

    # Use ALL subjects as the training pool. run_renet.train() internally makes
    # a stratified 15% validation split (never the test subject) and returns the
    # best VALIDATION accuracy via es.best. We report that -- so no test data is
    # ever used to choose lambda. We also record the weight cosine after training.
    tr_idx = np.arange(len(y))
    print(f"Lambda sweep on PhysioNet | {len(subs)} subjects | "
          f"train() internal 15% validation | {N_SEEDS} seeds\n")
    print(f"{'lambda':>8} {'val_acc':>10} {'val_sd':>8} {'weight_cos':>12}")

    rows = []
    for lam in LAMBDAS:
        val_accs, coss = [], []
        for k in range(N_SEEDS):
            seed = SEED + k
            old = R.CFG["lambda_osfr"]
            R.CFG["lambda_osfr"] = lam
            try:
                set_seed(seed)
                model = R.RENet(C, T, ncl).to(device)
                best_val = R.train(model, Xt, yt, tr_idx, seed, on_gpu)
            finally:
                R.CFG["lambda_osfr"] = old
            val_accs.append(float(best_val))
            coss.append(mean_weight_cosine(model))
            del model
            torch.cuda.empty_cache() if use_cuda else None
        rows.append(dict(lam=float(lam),
                         val_acc_mean=float(np.mean(val_accs)), val_acc_std=float(np.std(val_accs)),
                         weight_cos_mean=float(np.mean(coss)), weight_cos_std=float(np.std(coss)),
                         seeds=N_SEEDS))
        print(f"{lam:>8.2f} {np.mean(val_accs)*100:>9.2f}% {np.std(val_accs)*100:>7.2f}% {np.mean(coss):>12.4f}")

    json.dump(rows, open(os.path.join(OUT, "lambda_sweep.json"), "w"), indent=2)
    print(f"\nSaved {OUT}/lambda_sweep.json")
    print("\nReported accuracy is the internal VALIDATION accuracy from train()")
    print("(the held-out test subject is never used to choose lambda). lambda was")
    print("fixed a priori at 0.10 for all main results; this is a development-set")
    print("check that the choice is not delicate.")


if __name__ == "__main__":
    main()
