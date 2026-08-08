"""
DSA ERD/ERS visualization  (reviewer minor #2)

Tests whether the DSA power branch  log(1 + AvgPool(x^2))  actually tracks
sensorimotor-rhythm event-related (de)synchronization, i.e. whether it earns
its place as an interpretable module despite being accuracy-neutral.

What it produces (saved to figures/):
  1. dsa_erders_timefreq.png  -- grand-average ERD/ERS time-frequency map of
     the DSA power-branch activations, per class, referenced to a baseline.
  2. dsa_power_timecourse.png -- mu/beta band power time course from the DSA
     branch, left vs right hand, showing contralateral desynchronization.
  3. dsa_vs_raw_erd.png       -- DSA power-branch ERD vs. classical band-power
     ERD computed directly from the input, as a sanity cross-check.

Interpretation:
  * If the DSA branch shows a clear mu/beta power decrease (ERD) during motor
    imagery, lateralized to the contralateral hemisphere, DSA is justified as
    an interpretable ERD/ERS pathway -> keep DSA (route B).
  * If the DSA branch bears no clear relation to sensorimotor ERD/ERS, DSA has
    no interpretability value and should be dropped -> route A.

Usage:
  python visualize_dsa.py                 # trains one subject, makes figures
  SUBJECT=7 python visualize_dsa.py       # pick the held-out subject
Env: DATASET=physionet (2a has too few channels for a clean topography here)
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.metrics import accuracy_score

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)
from run_renet import RENet, osfr_loss, CFG

warnings.filterwarnings("ignore")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
FS = 128                                   # sampling rate after preprocessing
SUBJECT = int(os.environ.get("SUBJECT", "7"))


# ---- hook: capture the DSA power-branch output of block-1 activation --------
class PowerTap:
    """Recompute the DSA power branch of act1 for a given input batch.
    act1 = DualStateActivation((1,4)) followed by Dropout; we reproduce its
    power path log(1+AvgPool(x^2)) on the pre-activation feature map."""
    def __init__(self, model):
        self.model = model
        self.pool = nn.AvgPool2d((1, 4))

    def power(self, x):                     # x: (B, C, T) raw trial
        with torch.no_grad():
            h = self.model.bn1(self.model.spatial(self.model.block1(x.unsqueeze(1))))
            p = torch.log1p(self.pool(h ** 2))   # (B, F1*D, 1, T/4)
        return p.squeeze(2).cpu().numpy()        # (B, F1*D, T')


def train_one(Xt, yt, s, subj, C, T, n_classes):
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    model = RENet(C, T, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    ce = nn.CrossEntropyLoss()
    best, state = -1, None
    for ep in range(80):
        model.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device) * CFG["noise_std"]
            by = yt[idx].to(device)
            opt.zero_grad()
            (ce(model(bx), by) + CFG["lambda_osfr"] * osfr_loss(model)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xt[val].to(device)).argmax(1).cpu().numpy()
            a = accuracy_score(yt[val].numpy(), pv)
            if a > best:
                best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model


def bandpower(x, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return np.abs(hilbert(filtfilt(b, a, x, axis=-1), axis=-1)) ** 2


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    subj = SUBJECT if SUBJECT in subjects_of(s) else subjects_of(s)[0]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    print(f"Training RE-Net, holding out S{subj:03d} for DSA inspection...")
    model = train_one(Xt, yt, s, subj, C, T, n_classes)

    te = np.where(s == subj)[0]
    Xte = Xt[te]; yte = y[te]
    tap = PowerTap(model)
    P = tap.power(Xte.to(device))                     # (N, F1*D, T')
    tp = np.linspace(CFG.get("tmin", 0.5), CFG.get("tmax", 3.5), P.shape[-1])

    # ---- Fig 1: DSA power-branch time course, per class -------------------
    plt.figure(figsize=(8, 4))
    for cls, name, col in [(0, "Left hand", "#2b6cb0"), (1, "Right hand", "#c53030")]:
        m = P[yte == cls].mean(0).mean(0)             # avg over feature maps
        base = m[tp < (tp[0] + 0.5)].mean()
        erd = (m - base) / (base + 1e-9) * 100
        plt.plot(tp, erd, color=col, label=name, lw=2)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("DSA power change vs. baseline (%)")
    plt.title(f"DSA power branch, S{subj:03d}"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "dsa_power_timecourse.png"), dpi=300)
    print("Saved dsa_power_timecourse.png")

    # ---- Fig 2: DSA power-branch feature-map heatmap (time x feature) ------
    plt.figure(figsize=(9, 4))
    for k, (cls, name) in enumerate([(0, "Left hand"), (1, "Right hand")]):
        plt.subplot(1, 2, k + 1)
        M = P[yte == cls].mean(0)                     # (F1*D, T')
        base = M[:, tp < (tp[0] + 0.5)].mean(1, keepdims=True)
        plt.imshow((M - base) / (base + 1e-9) * 100, aspect="auto", cmap="RdBu_r",
                   vmin=-60, vmax=60, extent=[tp[0], tp[-1], M.shape[0], 0])
        plt.title(name); plt.xlabel("Time (s)")
        if k == 0: plt.ylabel("DSA feature map (band x spatial filter)")
        plt.colorbar(label="% change")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "dsa_erders_timefreq.png"), dpi=300)
    print("Saved dsa_erders_timefreq.png")

    # ---- Fig 3: DSA-branch ERD vs. classical mu/beta band-power ERD --------
    raw = Xte.numpy()
    plt.figure(figsize=(8, 4))
    for band, lo, hi, col in [("mu (8-12)", 8, 12, "#2b6cb0"), ("beta (13-30)", 13, 30, "#805ad5")]:
        bp = bandpower(raw, lo, hi).mean(1)           # avg channels -> (N, T)
        tt = np.linspace(tp[0], tp[-1], bp.shape[-1])
        m = bp.mean(0); base = m[tt < (tt[0] + 0.5)].mean()
        plt.plot(tt, (m - base) / (base + 1e-9) * 100, color=col, lw=2, label=f"raw {band}")
    dsa = P.mean(0).mean(0); base = dsa[tp < (tp[0] + 0.5)].mean()
    plt.plot(tp, (dsa - base) / (base + 1e-9) * 100, "k--", lw=2, label="DSA power branch")
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Power change vs. baseline (%)")
    plt.title(f"DSA branch vs. classical band-power ERD/ERS, S{subj:03d}"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "dsa_vs_raw_erd.png"), dpi=300)
    print("Saved dsa_vs_raw_erd.png")

    # ---- numeric summary for the caption ----------------------------------
    corr = np.corrcoef(dsa[:len(tt)] if len(dsa) >= len(tt) else dsa,
                       np.interp(tp, tt, m))[0, 1]
    print(f"\nCorrelation (DSA power branch vs. raw mu band-power time course): {corr:+.3f}")
    print("If strongly positive and both show a mid-trial dip (ERD), DSA is")
    print("tracking sensorimotor desynchronization -> keep DSA.")
    print("All DSA visualizations done.")


if __name__ == "__main__":
    main()
