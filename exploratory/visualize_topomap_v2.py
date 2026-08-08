"""
OSFR activation-pattern topographies  (interpretability figure, corrected)

Haufe et al. (2014): the weights W of a discriminative spatial filter are NOT
directly interpretable as neural sources; the interpretable quantity is the
activation pattern  A = Cov(X) W  (covariance-weighted), which reflects the
scalp projection the filter responds to. v1 plotted W (wrong); this plots A.

Also fixes the band: uses the mu band (8-12 Hz, band index 1 by default),
where left/right-hand MI shows C3/C4 lateralization, not theta (band 0).

For each subject we plot, per hand class, the activation pattern averaged over
that class's trials, so contralateral C3/C4 focality can be judged.

Produces figures/osfr_actpattern_topo.png

Usage:  python visualize_topomap_v2.py
Env:    DATASET=physionet  SUBJECTS=7,42,88  BAND=1  (1 = mu 8-12 Hz)
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)

warnings.filterwarnings("ignore")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
SUBJECTS = [int(x) for x in os.environ.get("SUBJECTS", "7,42,88").split(",")]
BAND = int(os.environ.get("BAND", "1"))     # 1 = mu (8-12 Hz)

PHYSIONET_CH = [
    "Fc5","Fc3","Fc1","Fcz","Fc2","Fc4","Fc6","C5","C3","C1","Cz","C2","C4","C6",
    "Cp5","Cp3","Cp1","Cpz","Cp2","Cp4","Cp6","Fp1","Fpz","Fp2","Af7","Af3","Afz",
    "Af4","Af8","F7","F5","F3","F1","Fz","F2","F4","F6","F8","Ft7","Ft8","T7","T8",
    "T9","T10","Tp7","Tp8","P7","P5","P3","P1","Pz","P2","P4","P6","P8","Po7","Po3",
    "Poz","Po4","Po8","O1","Oz","O2","Iz"]


class Net(nn.Module):
    def __init__(self, C, T, n_classes=2, F1=8, D=2, F2=16, K=64, p=0.25):
        super().__init__(); self.F1, self.D = F1, D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False), nn.BatchNorm2d(F1))
        self.spatial = nn.Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False)
        self.bn1 = nn.BatchNorm2d(F1*D)
        self.act1 = nn.Sequential(nn.ELU(True), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2),
            nn.ELU(True), nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            flat = self.block2(self.act1(self.bn1(self.spatial(self.block1(
                torch.zeros(1, 1, C, T)))))).numel()
        self.head = nn.Linear(flat, n_classes)

    def temporal(self, x):
        """Output of block1 (temporal conv), per band: (N, F1, C, T)."""
        return self.block1(x.unsqueeze(1))

    def forward(self, x):
        x = self.act1(self.bn1(self.spatial(self.block1(x.unsqueeze(1)))))
        return self.head(self.block2(x).flatten(1))


def osfr_loss(model):
    W = model.spatial.weight.view(model.F1, model.D, -1)
    I = torch.eye(model.D, device=W.device, dtype=W.dtype)
    return sum(torch.norm(F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I, p="fro")
               for f in range(model.F1)) / model.F1


def train(model, Xt, yt, s, subj, C, T, use_osfr, lam=0.10):
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(); best, state = -1, None
    for ep in range(80):
        model.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device)*0.03
            by = yt[idx].to(device)
            opt.zero_grad()
            loss = ce(model(bx), by)
            if use_osfr: loss = loss + lam * osfr_loss(model)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                a = accuracy_score(yt[val].numpy(), model(Xt[val].to(device)).argmax(1).cpu().numpy())
            if a > best: best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model


def activation_pattern(model, X, band, filt=0):
    """Haufe pattern A = Cov(z_band) @ w, where z_band is the band-filtered
    signal that the spatial filter sees, w is the spatial weight (C,).
    Returns a (C,) scalp pattern that is source-interpretable."""
    with torch.no_grad():
        z = model.temporal(X.to(device))[:, band]      # (N, C, T) band-filtered
    z = z.cpu().numpy()
    N, C, T = z.shape
    Z = z.transpose(1, 0, 2).reshape(C, N * T)          # (C, N*T)
    Z = Z - Z.mean(1, keepdims=True)
    cov = (Z @ Z.T) / Z.shape[1]                        # (C, C)
    w = model.spatial.weight.detach().view(model.F1, model.D, C).cpu().numpy()[band, filt]
    A = cov @ w                                         # (C,) activation pattern
    return A


def make_info(C):
    import mne
    names = PHYSIONET_CH[:C]
    info = mne.create_info(names, sfreq=128, ch_types="eeg")
    mont = mne.channels.make_standard_montage("standard_1005")
    mont_names = {n.lower(): n for n in mont.ch_names}
    info.rename_channels({n: mont_names[n.lower()] for n in names if n.lower() in mont_names})
    info.set_montage(mont, match_case=False, on_missing="ignore")
    return info


def main():
    import mne
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    info = make_info(C)

    n = len(SUBJECTS)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1: axes = axes[None, :]
    c3 = PHYSIONET_CH.index("C3"); c4 = PHYSIONET_CH.index("C4")
    for r, subj in enumerate(SUBJECTS):
        if subj not in subjects_of(s): continue
        print(f"[S{subj:03d}] training + activation patterns (band {BAND})...")
        m_o = train(Net(C, T, n_classes).to(device), Xt, yt, s, subj, C, T, True)
        m_e = train(Net(C, T, n_classes).to(device), Xt, yt, s, subj, C, T, False)
        te = np.where(s == subj)[0]
        Ao = activation_pattern(m_o, Xt[te], BAND)
        Ae = activation_pattern(m_e, Xt[te], BAND)
        for col, (A, tag) in enumerate([(Ao, "OSFR"), (Ae, "EEGNet")]):
            mne.viz.plot_topomap(A, info, axes=axes[r, col], show=False,
                                 cmap="RdBu_r", contours=4)
            axes[r, col].set_title(f"S{subj:03d} {tag}", fontsize=10)
            # report C3/C4 relative magnitude
            focal = (abs(A[c3]) + abs(A[c4])) / (np.abs(A).mean() + 1e-9)
            print(f"    {tag}: C3/C4 focality ratio = {focal:.2f}  (>1 = above-average)")
        del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None
    plt.suptitle(f"Activation-pattern topographies, mu band ({BAND})", y=1.005)
    plt.tight_layout()
    out = os.path.join(FIG, "osfr_actpattern_topo.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    print("Focality ratio > 1 at C3/C4 and a visible central focus = sensorimotor localization.")


if __name__ == "__main__":
    main()
