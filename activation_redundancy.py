"""
Activation-level redundancy check (reviewer 4.3).

The paper's interpretability argument moves from WEIGHT orthogonality
(W-bar W-bar^T ~ I) to a claim of non-redundant, separable spatial
representations. Reviewer 4.3 correctly notes that weight orthogonality does
NOT by itself imply that the two filters' OUTPUT signals are uncorrelated,
because the outputs depend on the (spatially coloured) input covariance:
orthogonal filters applied to correlated channels can still yield correlated
outputs. This script tests the implication directly, at the activation level.

For each band f and its D=2 spatial filters, we take the filter OUTPUT signals
z_1(t), z_2(t) = (spatial conv output before DSA/pooling) on held-out data, and
measure:
  - |Pearson correlation| between z_1 and z_2 (temporal, per trial, averaged)
  - the normalized redundancy of the 2-D output via its covariance eigenvalues
    (1 - effective_rank/2; 0 = both dimensions used equally, ->1 = collapsed)
We compare RE-Net (OSFR) vs the unconstrained EEGNet-style backbone (lambda=0).

Interpretation:
  - If OSFR outputs are MORE decorrelated than the unconstrained ones, weight
    orthogonality does translate to activation-level non-redundancy: the
    interpretability claim is supported at the level that matters.
  - If they are similar, the paper must restrict its claim to WEIGHT geometry
    and NOT assert output/source non-redundancy. Either way we report honestly.

Reuses the exact RENet model and train() from run_renet.py.

Outputs activation_run/activation_redundancy.json and prints a summary.

Usage:  python activation_redundancy.py
Env:    DATASET=physionet  TEST_K=15
"""
import os, json, warnings
import numpy as np
import torch
import torch.nn.functional as F

import run_renet as R
from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, to_compute_tensors, batch_index)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "activation_run"); os.makedirs(OUT, exist_ok=True)
TEST_K = int(os.environ.get("TEST_K", "15"))


def spatial_outputs(model, Xt, idx, on_gpu, batch=128):
    """return spatial-conv outputs (N, F1*D, T) on CPU, before DSA/pooling."""
    outs = []
    with torch.no_grad():
        for j in range(0, len(idx), batch):
            bx = batch_index(Xt, torch.as_tensor(idx[j:j+batch]), on_gpu)
            z = model.spatial(model.block1(bx.unsqueeze(1)))  # (b, F1*D, 1, T)
            outs.append(z.squeeze(2).cpu())
            del bx, z
    return torch.cat(outs, 0)  # (N, F1*D, T)


def output_redundancy(Z, F1, D):
    """per band: |corr(z1,z2)| and covariance-based redundancy of the D outputs."""
    N = Z.shape[0]
    corrs, redun = [], []
    for f in range(F1):
        chans = [f*D + j for j in range(D)]
        zf = Z[:, chans, :].numpy()  # (N, D, T)
        # temporal Pearson |corr| between the two filter outputs, averaged over trials
        cs = []
        for n in range(N):
            a, b = zf[n, 0], zf[n, 1]
            a = a - a.mean(); b = b - b.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
            cs.append(abs(float(a @ b) / denom))
        corrs.append(np.mean(cs))
        # covariance eigen redundancy across the D output dims (flatten trials x time)
        M = zf.transpose(1, 0, 2).reshape(D, -1)  # (D, N*T)
        M = M - M.mean(axis=1, keepdims=True)
        cov = (M @ M.T) / (M.shape[1] - 1)
        ev = np.clip(np.linalg.eigvalsh(cov).real, 1e-12, None)
        p = ev / ev.sum()
        eff_rank = float(np.exp(-(p * np.log(p)).sum()))
        redun.append(1.0 - eff_rank / D)  # 0=full rank, ->1 collapsed
    return float(np.mean(corrs)), float(np.mean(redun))


def train_one(Xt, yt, tr_idx, lam, seed, C, T, ncl, on_gpu):
    old = R.CFG["lambda_osfr"]; R.CFG["lambda_osfr"] = lam
    try:
        set_seed(seed)
        m = R.RENet(C, T, ncl).to(device)
        R.train(m, Xt, yt, tr_idx, seed, on_gpu)
    finally:
        R.CFG["lambda_osfr"] = old
    m.eval(); return m


def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    subs = subjects_of(s)
    F1, D = R.CFG["F1"], R.CFG["D"]
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Activation-level redundancy | {len(test_subs)} held-out subjects\n")
    print(f"{'subj':>5} {'OSFR |corr|':>12} {'EEG |corr|':>12} {'OSFR redun':>12} {'EEG redun':>12}")

    o_corr, e_corr, o_red, e_red = [], [], [], []
    for subj in test_subs:
        tr_idx = np.where(s != subj)[0]
        te_idx = np.where(s == subj)[0]
        # OSFR model
        mo = train_one(Xt, yt, tr_idx, 0.10, SEED, C, T, ncl, on_gpu)
        Zo = spatial_outputs(mo, Xt, te_idx, on_gpu)
        oc, orr = output_redundancy(Zo, F1, D)
        del mo; torch.cuda.empty_cache() if use_cuda else None
        # unconstrained model
        me = train_one(Xt, yt, tr_idx, 0.0, SEED, C, T, ncl, on_gpu)
        Ze = spatial_outputs(me, Xt, te_idx, on_gpu)
        ec, err = output_redundancy(Ze, F1, D)
        del me; torch.cuda.empty_cache() if use_cuda else None
        o_corr.append(oc); e_corr.append(ec); o_red.append(orr); e_red.append(err)
        print(f"{subj:>5} {oc:>12.4f} {ec:>12.4f} {orr:>12.4f} {err:>12.4f}")

    from scipy.stats import wilcoxon
    oc, ec = np.array(o_corr), np.array(e_corr)
    orr, err = np.array(o_red), np.array(e_red)
    summary = dict(
        osfr_output_corr_mean=float(oc.mean()), osfr_output_corr_std=float(oc.std()),
        eegnet_output_corr_mean=float(ec.mean()), eegnet_output_corr_std=float(ec.std()),
        osfr_output_redun_mean=float(orr.mean()), eegnet_output_redun_mean=float(err.mean()),
        n=len(oc))
    try:
        _, p_corr = wilcoxon(oc, ec, alternative="less")   # OSFR corr < EEGNet corr ?
        summary["wilcoxon_p_osfr_less_corr"] = float(p_corr)
    except Exception: pass
    json.dump(dict(summary=summary,
                   osfr_corr=o_corr, eegnet_corr=e_corr,
                   osfr_redun=o_red, eegnet_redun=e_red),
              open(os.path.join(OUT, "activation_redundancy.json"), "w"), indent=2)

    print("\n" + "=" * 66)
    print("ACTIVATION-LEVEL output redundancy (does weight orthogonality carry over?)")
    print(f"  |corr(z1,z2)|   RE-Net(OSFR) {oc.mean():.4f}  vs  EEGNet {ec.mean():.4f}")
    print(f"  output redundancy RE-Net {orr.mean():.4f}  vs  EEGNet {err.mean():.4f}")
    if "wilcoxon_p_osfr_less_corr" in summary:
        print(f"  Wilcoxon (OSFR output-corr < EEGNet) p = {summary['wilcoxon_p_osfr_less_corr']:.4f}")
    print("=" * 66)
    print("If OSFR output corr/redundancy is clearly lower, weight orthogonality")
    print("does translate to activation-level non-redundancy (claim supported).")
    print("If similar, restrict the claim to weight geometry only. Report honestly.")


if __name__ == "__main__":
    main()
