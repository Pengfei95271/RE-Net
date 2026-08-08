"""
OSFR reinforcement #2 — OSFR filters vs. FBCSP spatial patterns  (reviewer #3)

Turns the "OSFR is CSP-like" analogy into measured evidence. For each frequency
band, we compare the subspace spanned by RE-Net's OSFR spatial filters with the
subspace spanned by FBCSP's spatial patterns (from pyriemann/CSP) using
principal angles. Small principal angles = the two spatial subspaces align =
OSFR learns CSP-like projections. We contrast against unconstrained EEGNet
filters as a baseline.

Usage:  python osfr_vs_csp.py
Env:    DATASET=physionet   SUBJECTS=7,42,88
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.linalg import subspace_angles
from sklearn.metrics import accuracy_score

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split, CFG_DATA)

warnings.filterwarnings("ignore")
SUBJECTS = [int(x) for x in os.environ.get("SUBJECTS", "7,42,88").split(",")]
FS = CFG_DATA["resample"]
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 40)]  # F1=8


# ---- RE-Net (EEGNet + OSFR), reused from osfr_dimensions style ----
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


def fbcsp_patterns(X, y, lo, hi, n=2):
    """Band-limited CSP spatial patterns (C x n) via pyriemann CSP on covariances."""
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    Xb = filtfilt(b, a, X, axis=-1).astype(np.float32)
    cov = Covariances("oas").fit_transform(Xb)
    csp = CSP(nfilter=n, log=True).fit(cov, y)
    # CSP.patterns_ is (n_filters, C); take first n rows as the spatial subspace
    P = getattr(csp, "patterns_", None)
    if P is None:
        P = csp.filters_[:n]
    return np.asarray(P[:n]).T          # (C, n)


def subspace_alignment(A, B):
    """1 - mean(sin(principal angles)); 1 = identical subspaces, 0 = orthogonal."""
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
    ang = subspace_angles(A, B)
    return float(np.mean(np.cos(ang)))   # mean cosine of principal angles


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    print("=" * 60)
    print(f"OSFR vs FBCSP subspace alignment  (subjects {SUBJECTS})")
    print("higher = filters align with CSP patterns (1=identical)")
    print("=" * 60)
    print(f"{'band (Hz)':>12} | {'RE-Net~CSP':>11} | {'EEGNet~CSP':>11}")
    print("-" * 60)

    align_osfr, align_eeg = [[] for _ in BANDS], [[] for _ in BANDS]
    for subj in SUBJECTS:
        if subj not in subjects_of(s): continue
        tr = np.where(s != subj)[0]
        m_o = train(Net(C, T, n_classes, D=2).to(device), Xt, yt, s, subj, C, T, True)
        m_e = train(Net(C, T, n_classes, D=2).to(device), Xt, yt, s, subj, C, T, False)
        Wo = m_o.spatial.weight.detach().view(8, 2, C).cpu().numpy()
        We = m_e.spatial.weight.detach().view(8, 2, C).cpu().numpy()
        for bi, (lo, hi) in enumerate(BANDS):
            try:
                Pcsp = fbcsp_patterns(X[tr], y[tr], lo, hi, n=2)     # (C,2)
                align_osfr[bi].append(subspace_alignment(Wo[bi].T, Pcsp))
                align_eeg[bi].append(subspace_alignment(We[bi].T, Pcsp))
            except Exception as e:
                print(f"  band {lo}-{hi} failed: {e}")
        del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None

    for bi, (lo, hi) in enumerate(BANDS):
        if align_osfr[bi]:
            print(f"{f'{lo}-{hi}':>12} | {np.mean(align_osfr[bi]):>11.3f} | "
                  f"{np.mean(align_eeg[bi]):>11.3f}")
    all_o = np.mean([np.mean(a) for a in align_osfr if a])
    all_e = np.mean([np.mean(a) for a in align_eeg if a])
    print("-" * 60)
    print(f"{'mean':>12} | {all_o:>11.3f} | {all_e:>11.3f}")
    print("\nInterpretation: if RE-Net~CSP alignment exceeds EEGNet~CSP, OSFR makes")
    print("the learned spatial subspace measurably more CSP-like — evidence, not analogy.")


if __name__ == "__main__":
    main()
