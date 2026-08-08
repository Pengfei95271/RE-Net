"""
OSFR spatial-filter topographies  (interpretability figure)

Trains RE-Net (EEGNet + OSFR) on a few held-out subjects, then renders the
learned depthwise spatial filters as scalp topographies using the standard
10-10 montage. Motor-imagery filters should localize over sensorimotor cortex
(C3/C4). Produces:

  figures/osfr_topomaps.png     grid of the lowest-band OSFR filters (per subj)
  figures/osfr_vs_eegnet_topo.png  OSFR vs unconstrained EEGNet, same band

Channel positions come from MNE's standard_1005 montage, matched by name to
the PhysioNet 64-channel layout. No moabb needed (works in the base env).

Usage:  python visualize_topomap.py
Env:    DATASET=physionet   SUBJECTS=7,42,88   BAND=0  (0=lowest freq filter)
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
BAND = int(os.environ.get("BAND", "0"))

# PhysioNet eegmmidb 64-channel order (EEG MMIDB, 10-10 names as MNE expects).
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


def make_info(C):
    import mne
    names = PHYSIONET_CH[:C] if C <= len(PHYSIONET_CH) else \
            PHYSIONET_CH + [f"EEG{i}" for i in range(C - len(PHYSIONET_CH))]
    info = mne.create_info(names, sfreq=128, ch_types="eeg")
    mont = mne.channels.make_standard_montage("standard_1005")
    # case-insensitive rename to montage convention
    rename = {}
    mont_names = {n.lower(): n for n in mont.ch_names}
    for n in names:
        if n.lower() in mont_names:
            rename[n] = mont_names[n.lower()]
    info.rename_channels(rename)
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
    for r, subj in enumerate(SUBJECTS):
        if subj not in subjects_of(s): continue
        print(f"[S{subj:03d}] training OSFR + EEGNet for topographies...")
        m_o = train(Net(C, T, n_classes).to(device), Xt, yt, s, subj, C, T, True)
        m_e = train(Net(C, T, n_classes).to(device), Xt, yt, s, subj, C, T, False)
        # spatial weights: (F1*D, C); take band BAND, first spatial filter
        Wo = m_o.spatial.weight.detach().view(8, 2, C).cpu().numpy()[BAND, 0]
        We = m_e.spatial.weight.detach().view(8, 2, C).cpu().numpy()[BAND, 0]
        for c, (W, title) in enumerate([(Wo, f"S{subj:03d} OSFR"),
                                        (We, f"S{subj:03d} EEGNet")]):
            mne.viz.plot_topomap(W, info, axes=axes[r, c], show=False,
                                 cmap="RdBu_r", contours=4)
            axes[r, c].set_title(title, fontsize=10)
        del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None
    plt.suptitle(f"Spatial filter topographies (band {BAND})", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG, "osfr_topomaps.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    print("Look for C3/C4 (central) focal points in the OSFR column.")


if __name__ == "__main__":
    main()
