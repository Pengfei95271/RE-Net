"""
S007 analyzability worked-example figure  (reviewer M1, method A)

Produces one clean figure for the worked example in the paper: for subject
S007 at band 1 (mu), the two within-band spatial filters of RE-Net (OSFR)
side by side with the two of the unconstrained EEGNet, using a diverging
colormap so polarity is visible, annotated with the inter-filter correlation.

The figure supports exactly the claim the text makes -- RE-Net's two filters
are near-orthogonal and separable (|r|=0.009), EEGNet's are strongly
correlated and entangled (|r|=0.703) -- WITHOUT asserting any left/right or
focal localization, since the patterns are distributed. Filters are shown as
distributed diverging maps; the message is separable vs. overlapping, not a
spatial template.

Produces figures/analyzability_s007.png
Usage:  python analyzability_s007.py
Env:    DATASET=physionet  SUBJECT=7  BAND=0
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
SUBJECT = int(os.environ.get("SUBJECT", "7"))
BAND = int(os.environ.get("BAND", "0"))   # 0-indexed band

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


def train(Xt, yt, s, subj, C, T, n_classes, use_osfr):
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    model = Net(C, T, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(); best, state = -1, None
    for ep in range(80):
        model.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device) * 0.03
            by = yt[idx].to(device)
            opt.zero_grad()
            loss = ce(model(bx), by)
            if use_osfr: loss = loss + 0.10 * osfr_loss(model)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                a = accuracy_score(yt[val].numpy(), model(Xt[val].to(device)).argmax(1).cpu().numpy())
            if a > best: best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model


def two_filters(model, band, C):
    W = model.spatial.weight.detach().view(model.F1, model.D, C).cpu().numpy()[band]
    return W[0], W[1], abs(np.corrcoef(W[0], W[1])[0, 1])


def make_info(C):
    import mne
    names = PHYSIONET_CH[:C]
    info = mne.create_info(names, sfreq=128, ch_types="eeg")
    mont = mne.channels.make_standard_montage("standard_1005")
    mn = {n.lower(): n for n in mont.ch_names}
    info.rename_channels({n: mn[n.lower()] for n in names if n.lower() in mn})
    info.set_montage(mont, match_case=False, on_missing="ignore")
    return info


def main():
    import mne
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    info = make_info(C)

    if SUBJECT not in subjects_of(s):
        print(f"Subject {SUBJECT} not found; available e.g. {sorted(subjects_of(s))[:10]}")
        return
    print(f"Training RE-Net + EEGNet for S{SUBJECT:03d} ...")
    m_o = train(Xt, yt, s, SUBJECT, C, T, n_classes, True)
    m_e = train(Xt, yt, s, SUBJECT, C, T, n_classes, False)
    o0, o1, oc = two_filters(m_o, BAND, C)
    e0, e1, ec = two_filters(m_e, BAND, C)
    print(f"RE-Net |corr|={oc:.3f}   EEGNet |corr|={ec:.3f}")

    # symmetric color scale per model for fair polarity display
    def vlim(a, b):
        m = max(np.abs(a).max(), np.abs(b).max()); return -m, m

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    panels = [
        (o0, "RE-Net filter 1", vlim(o0, o1)),
        (o1, "RE-Net filter 2", vlim(o0, o1)),
        (e0, "EEGNet filter 1", vlim(e0, e1)),
        (e1, "EEGNet filter 2", vlim(e0, e1)),
    ]
    for ax, (w, title, (vmn, vmx)) in zip(axes, panels):
        mne.viz.plot_topomap(w, info, axes=ax, show=False, cmap="RdBu_r",
                             vlim=(vmn, vmx), contours=4)
        ax.set_title(title, fontsize=11)

    # correlation annotations under each pair
    axes[0].text(1.05, -0.16, f"RE-Net (OSFR): $|r|={oc:.3f}$  —  separable",
                 transform=axes[0].transAxes, ha="center", fontsize=10, color="#2b6cb0")
    axes[2].text(1.05, -0.16, f"EEGNet (no OSFR): $|r|={ec:.3f}$  —  entangled",
                 transform=axes[2].transAxes, ha="center", fontsize=10, color="#c53030")
    plt.suptitle(f"Subject S{SUBJECT:03d}, mu band: within-band spatial filters", y=1.02, fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG, "analyzability_s007.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    print("\nBoth patterns are spatially distributed; the point is that the two")
    print("OSFR filters are near-orthogonal (separable) while the two EEGNet")
    print("filters are strongly correlated (entangled). No left/right claim.")


if __name__ == "__main__":
    main()
