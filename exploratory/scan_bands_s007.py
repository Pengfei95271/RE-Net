"""
Scan all bands for the cleanest RE-Net vs EEGNet contrast (one subject).

Trains RE-Net and EEGNet ONCE for the subject, then reports the inter-filter
correlation for every band, so you can pick the band where OSFR is clearly
separable (low) and the unconstrained backbone is clearly entangled (high).

Usage:  python scan_bands_s007.py
Env:    DATASET=physionet  SUBJECT=7
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)

warnings.filterwarnings("ignore")
SUBJECT = int(os.environ.get("SUBJECT", "7"))


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


def all_band_corrs(model, C):
    W = model.spatial.weight.detach().view(model.F1, model.D, C).cpu().numpy()
    return [abs(np.corrcoef(W[f, 0], W[f, 1])[0, 1]) for f in range(model.F1)]


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    if SUBJECT not in subjects_of(s):
        print(f"Subject {SUBJECT} not available"); return
    print(f"Training RE-Net + EEGNet once for S{SUBJECT:03d} ...")
    m_o = train(Xt, yt, s, SUBJECT, C, T, n_classes, True)
    m_e = train(Xt, yt, s, SUBJECT, C, T, n_classes, False)
    oc = all_band_corrs(m_o, C)
    ec = all_band_corrs(m_e, C)

    print(f"\n{'band':>4} | {'RE-Net |r|':>10} | {'EEGNet |r|':>10} | {'gap':>6}")
    print("-" * 42)
    best_band, best_gap = None, -1
    for b in range(len(oc)):
        gap = ec[b] - oc[b]
        flag = ""
        if oc[b] < 0.10 and ec[b] > 0.40:
            flag = "  <-- clean contrast"
            if gap > best_gap: best_gap, best_band = gap, b
        print(f"{b:>4} | {oc[b]:>10.3f} | {ec[b]:>10.3f} | {gap:>+6.3f}{flag}")

    print("\n" + "=" * 42)
    if best_band is not None:
        print(f"BEST band for the figure: BAND={best_band}  "
              f"(RE-Net {oc[best_band]:.3f}, EEGNet {ec[best_band]:.3f})")
        print(f"Run:  BAND={best_band} SUBJECT={SUBJECT} python analyzability_s007.py")
    else:
        print("No band with RE-Net<0.10 AND EEGNet>0.40 for this subject.")
        print("Consider a different subject, e.g. SUBJECT=42 or SUBJECT=88,")
        print("or fall back to the 240-pair distribution figure as the M1 evidence.")


if __name__ == "__main__":
    main()
