"""
OSFR reinforcement #1 — orthogonality vs. number of filters D  (reviewer #5)

Shows OSFR is NOT trivially satisfied because D=2. Trains RE-Net (EEGNet+OSFR)
with D in {2,4,6} spatial filters per band and reports, for each:
  * mean off-diagonal inter-filter correlation (lower = more orthogonal)
  * the same for an unconstrained EEGNet at that D (contrast)
  * LOSO accuracy (a few subjects), to show raising D does not hurt.

If OSFR keeps correlations low even at D=6 while unconstrained filters drift
up, the orthogonality result is non-trivial.

Usage:  python osfr_dimensions.py
Env:    DATASET=physionet   SUBJECTS=7,42,88 (few subjects; this is a probe)
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
SUBJECTS = [int(x) for x in os.environ.get("SUBJECTS", "7,42,88").split(",")]
DS = [2, 4, 6]
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)


class Net(nn.Module):
    """EEGNet backbone with configurable D; OSFR applied externally if use_osfr."""
    def __init__(self, C, T, n_classes=2, F1=8, D=2, F2=16, K=64, p=0.25):
        super().__init__()
        self.F1, self.D = F1, D
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


def mean_offdiag_corr(model):
    """Mean absolute off-diagonal inter-filter correlation, averaged over bands."""
    W = model.spatial.weight.detach().view(model.F1, model.D, -1).cpu()
    vals = []
    for f in range(model.F1):
        w = F.normalize(W[f], p=2, dim=-1)
        c = (w @ w.t()).numpy()
        off = c[~np.eye(model.D, dtype=bool)]
        vals.append(np.abs(off).mean())
    return float(np.mean(vals))


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
                pv = model(Xt[val].to(device)).argmax(1).cpu().numpy()
            a = accuracy_score(yt[val].numpy(), pv)
            if a > best: best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    te = np.where(s == subj)[0]
    with torch.no_grad():
        pt = model(Xt[te].to(device)).argmax(1).cpu().numpy()
    return accuracy_score(yt[te].numpy(), pt)


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    print("=" * 68)
    print(f"OSFR orthogonality vs. D  (subjects {SUBJECTS})")
    print("=" * 68)
    print(f"{'D':>3} | {'OSFR corr':>10} | {'EEGNet corr':>12} | {'OSFR acc':>9} | {'EEGNet acc':>11}")
    print("-" * 68)
    results = {}
    for D in DS:
        oc, ec, oa, ea = [], [], [], []
        for subj in SUBJECTS:
            if subj not in subjects_of(s): continue
            m_o = Net(C, T, n_classes, D=D).to(device)
            oa.append(train(m_o, Xt, yt, s, subj, C, T, use_osfr=True))
            oc.append(mean_offdiag_corr(m_o))
            m_e = Net(C, T, n_classes, D=D).to(device)
            ea.append(train(m_e, Xt, yt, s, subj, C, T, use_osfr=False))
            ec.append(mean_offdiag_corr(m_e))
            del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None
        results[D] = dict(oc=np.mean(oc), ec=np.mean(ec), oa=np.mean(oa), ea=np.mean(ea))
        print(f"{D:>3} | {np.mean(oc):>10.3f} | {np.mean(ec):>12.3f} | "
              f"{np.mean(oa):>8.1%} | {np.mean(ea):>10.1%}")

    print("\nInterpretation: if OSFR corr stays low (e.g. <0.1) as D grows to 6,")
    print("while EEGNet corr rises, the orthogonality is non-trivial and OSFR is")
    print("actively enforcing it — not an artifact of D=2.")

    # bar chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = np.arange(len(DS)); w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(xs - w/2, [results[D]["oc"] for D in DS], w, label="RE-Net (OSFR)", color="#2b6cb0")
    plt.bar(xs + w/2, [results[D]["ec"] for D in DS], w, label="EEGNet (no OSFR)", color="#c53030")
    plt.axhline(0.1, color="green", ls="--", alpha=0.6, label="0.1 reference")
    plt.xticks(xs, [f"D={D}" for D in DS]); plt.ylabel("mean |inter-filter corr|")
    plt.title("OSFR keeps filters orthogonal as D grows"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "osfr_vs_D.png"), dpi=300)
    print("Saved osfr_vs_D.png")


if __name__ == "__main__":
    main()
