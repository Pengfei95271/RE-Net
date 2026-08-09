"""
FINAL ATTEMPT -- Analyzability put to work: does spatial-filter collapse
predict decoding failure?  (reviewer #1: show the enabled interpretation
producing a concrete, non-trivial finding)

The idea. OSFR makes each band's two spatial filters separable, so their
degree of *residual* collapse (how close the two filters come to sharing a
direction, despite the orthogonality prior) is a readable, per-band quantity.
For the unconstrained EEGNet this quantity is meaningless -- its filters are
entangled everywhere -- so the following analysis is one that ONLY the
separable OSFR model licenses.

Hypothesis (from the reviewer): subjects the model decodes poorly are subjects
where, despite OSFR, the spatial filters collapse most -- i.e. residual
inter-filter correlation is higher for low-accuracy subjects. If so, the
separable structure has *revealed a failure signature*: collapse predicts poor
decoding, a statement one cannot even formulate on the entangled backbone.

Design (cheap: no full LOSO re-run). We take the 20 lowest- and 20
highest-accuracy subjects under the existing RE-Net LOSO results, retrain the
RE-Net model for each (LOSO fold, train only -- we already know its accuracy),
extract the 8-band spatial filters, and compute per-subject collapse =
mean over bands of |corr(filter_1, filter_2)|. We then test whether collapse
is higher in the low-accuracy group (one-sided Mann-Whitney) and whether
collapse correlates with accuracy across all 40 subjects (Spearman).

Outputs finalshot_run/collapse.json and prints the verdict.

Usage:  python final_collapse.py
Env:    DATASET=physionet
"""
import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "finalshot_run"); os.makedirs(OUT, exist_ok=True)

# two extremes from the existing RE-Net LOSO results
LOW = [20, 36, 87, 99, 3, 5, 81, 109, 28, 37, 53, 67, 80, 89, 97, 100, 4, 6, 9, 18]
HIGH = [61, 73, 93, 102, 32, 44, 31, 71, 42, 65, 94, 7, 43, 75, 72, 62, 2, 29, 85, 55]
F1 = 8


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


def train_fold(Xt, yt, s, subj, C, T, ncl):
    """train RE-Net for the LOSO fold that holds out `subj` (train only)."""
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    m = Net(C, T, ncl).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(); best, st = -1, None
    from sklearn.metrics import accuracy_score
    for ep in range(80):
        m.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device) * 0.03
            by = yt[idx].to(device)
            opt.zero_grad(); loss = ce(m(bx), by) + 0.10 * osfr_loss(m)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (ep + 1) % 5 == 0:
            m.eval()
            with torch.no_grad():
                a = accuracy_score(yt[val].numpy(), m(Xt[val].to(device)).argmax(1).cpu().numpy())
            if a > best: best, st = a, {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(st); m.eval(); return m


def collapse_of(model):
    """per-subject collapse = mean over bands of |corr(filter_1, filter_2)|."""
    W = model.spatial.weight.detach().view(F1, 2, -1).cpu().numpy()  # (F1,2,C)
    cs = []
    for f in range(F1):
        a, b = W[f, 0], W[f, 1]
        a = a - a.mean(); b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        cs.append(abs(float(a @ b) / denom))
    return float(np.mean(cs)), cs


def main():
    accs = json.load(open(os.path.join(BASE, "renet_run/results/loso_renet.json")))
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    rows = []
    both = [("low", LOW), ("high", HIGH)]
    total = sum(len(g) for _, g in both)
    done = 0
    for group, subs in both:
        for subj in subs:
            if subj not in subjects_of(s):
                continue
            m = train_fold(Xt, yt, s, subj, C, T, ncl)
            col, per_band = collapse_of(m)
            acc = accs[str(subj)]["acc"]
            rows.append(dict(subject=subj, group=group, acc=acc,
                             collapse=col, per_band=per_band))
            del m; torch.cuda.empty_cache() if use_cuda else None
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{total} | last: S{subj} {group} acc={acc:.2f} collapse={col:.4f}")

    lo_c = np.array([r["collapse"] for r in rows if r["group"] == "low"])
    hi_c = np.array([r["collapse"] for r in rows if r["group"] == "high"])
    all_acc = np.array([r["acc"] for r in rows])
    all_col = np.array([r["collapse"] for r in rows])

    # low-accuracy group should have HIGHER collapse if hypothesis holds
    U, p_mw = mannwhitneyu(lo_c, hi_c, alternative="greater")
    rho, p_sp = spearmanr(all_acc, all_col)  # expect negative rho (higher acc -> lower collapse)

    summary = dict(
        low_collapse_mean=float(lo_c.mean()), low_collapse_std=float(lo_c.std()),
        high_collapse_mean=float(hi_c.mean()), high_collapse_std=float(hi_c.std()),
        mannwhitney_U=float(U), mannwhitney_p_onesided=float(p_mw),
        spearman_rho=float(rho), spearman_p=float(p_sp),
        n=len(rows),
    )
    json.dump(dict(summary=summary, rows=rows),
              open(os.path.join(OUT, "collapse.json"), "w"), indent=2)

    print("\n" + "=" * 64)
    print("Does OSFR filter collapse predict decoding failure?")
    print(f"  low-accuracy  subjects: collapse {lo_c.mean():.4f} +/- {lo_c.std():.4f}")
    print(f"  high-accuracy subjects: collapse {hi_c.mean():.4f} +/- {hi_c.std():.4f}")
    print(f"  Mann-Whitney (low > high) one-sided p = {p_mw:.4f}")
    print(f"  Spearman(acc, collapse) rho = {rho:+.3f}, p = {p_sp:.4f}")
    print("=" * 64)
    if p_mw < 0.05 and rho < 0:
        print("FINDING: collapse is significantly higher for low-accuracy subjects,")
        print("and accuracy correlates negatively with collapse. The separable OSFR")
        print("structure reveals a failure signature -- a statement the entangled")
        print("backbone cannot express. THIS IS A POSITIVE, USABLE RESULT.")
    else:
        print("NEGATIVE: no significant collapse-failure link. The separable structure")
        print("does not reveal a failure signature on this data; report honestly and")
        print("do not add to the paper as a positive claim.")


if __name__ == "__main__":
    main()
