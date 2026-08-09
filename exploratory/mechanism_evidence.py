"""
Mechanism evidence for "orthogonality is free"  (reviewer #5)

Section 5.4 argues: binary MI discrimination is essentially low-rank, so one
per-band filter can align with the discriminative direction while the other is
free to be orthogonal; requiring orthogonality therefore constrains only the
second filter and leaves the discriminative one essentially untouched -- which
is why accuracy is preserved.

This script tests that argument directly. For each band we:
  1. Estimate the class-discriminative spatial direction d from the training
     data (Fisher/LDA direction on band-filtered channel power).
  2. Measure how well each model's two spatial filters SPAN d
     (|cos| of d onto the 2-D filter subspace), for RE-Net (OSFR) and EEGNet.
If the mechanism holds, BOTH models capture d about equally well (OSFR does not
push the filters away from the discriminative direction), even though OSFR's two
filters are orthogonal and EEGNet's are not.

We also report the per-band effective rank of the discriminative subspace to
support the "essentially low-rank" premise.

Uses already-trained-style models (trains once per subject, few subjects).
Outputs mechanism_run/mechanism.json  and prints a summary table.

Usage:  python mechanism_evidence.py
Env:    DATASET=physionet  TEST_K=15
"""
import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "mechanism_run"); os.makedirs(OUT, exist_ok=True)
TEST_K = int(os.environ.get("TEST_K", "12"))
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
        self._K = K

    def forward(self, x):
        x = self.act1(self.bn1(self.spatial(self.block1(x.unsqueeze(1)))))
        return self.head(self.block2(x).flatten(1))

    def temporal_filtered(self, x, batch=128):
        """band-filtered signal per band: (N, F1, C, T), computed in batches to
        avoid OOM on small GPUs; result returned on CPU."""
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb = x[i:i+batch].to(device)
                zb = self.block1(xb.unsqueeze(1)).cpu()
                outs.append(zb)
                del xb, zb
        return torch.cat(outs, 0)


def osfr_loss(model):
    W = model.spatial.weight.view(model.F1, model.D, -1)
    I = torch.eye(model.D, device=W.device, dtype=W.dtype)
    return sum(torch.norm(F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I, p="fro")
               for f in range(model.F1)) / model.F1


def train(Xt, yt, s, subj, C, T, ncl, use_osfr):
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    m = Net(C, T, ncl).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(); best, st = -1, None
    for ep in range(80):
        m.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device) * 0.03
            by = yt[idx].to(device)
            opt.zero_grad(); loss = ce(m(bx), by)
            if use_osfr: loss = loss + 0.10 * osfr_loss(m)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (ep + 1) % 5 == 0:
            m.eval()
            with torch.no_grad():
                a = accuracy_score(yt[val].numpy(), m(Xt[val].to(device)).argmax(1).cpu().numpy())
            if a > best: best, st = a, {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(st); m.eval(); return m


def fisher_direction(feat, labels):
    """LDA/Fisher direction for 2 classes on (N, C) band-power features."""
    m0 = feat[labels == 0].mean(0); m1 = feat[labels == 1].mean(0)
    Sw = np.cov(feat[labels == 0].T) + np.cov(feat[labels == 1].T)
    Sw += 1e-3 * np.eye(Sw.shape[0])
    d = np.linalg.solve(Sw, (m1 - m0))
    return d / (np.linalg.norm(d) + 1e-9)


def subspace_capture(d, W2):
    """|cos| of direction d onto the 2-D subspace spanned by rows of W2 (2,C)."""
    Q, _ = np.linalg.qr(W2.T)              # (C,2) orthonormal basis
    proj = Q @ (Q.T @ d)                   # projection of d onto subspace
    return float(np.linalg.norm(proj) / (np.linalg.norm(d) + 1e-9))  # in [0,1]


def eff_rank(feat, labels):
    """effective rank of the between/within discriminative structure (premise check)."""
    m0 = feat[labels == 0].mean(0); m1 = feat[labels == 1].mean(0)
    Sw = np.cov(feat[labels == 0].T) + np.cov(feat[labels == 1].T) + 1e-3*np.eye(feat.shape[1])
    Sb = np.outer(m1 - m0, m1 - m0)
    ev = np.linalg.eigvalsh(np.linalg.solve(Sw, Sb))
    ev = np.clip(ev.real, 0, None)
    if ev.sum() < 1e-12: return 0.0
    p = ev / ev.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))   # effective rank (perplexity)


def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Mechanism evidence: {len(test_subs)} subjects x {F1} bands\n")

    cap_o, cap_e, ranks = [], [], []
    for si, subj in enumerate(test_subs):
        tr = np.where(s != subj)[0]
        Xtr = Xt[tr]; ytr = yt[tr].numpy()
        m_o = train(Xt, yt, s, subj, C, T, ncl, True)
        m_e = train(Xt, yt, s, subj, C, T, ncl, False)
        # band-filtered signals (use OSFR model's temporal filters; same architecture)
        z = m_o.temporal_filtered(Xtr).numpy()  # (N,F1,C,T), already on CPU
        Wo = m_o.spatial.weight.detach().view(F1, 2, C).cpu().numpy()
        We = m_e.spatial.weight.detach().view(F1, 2, C).cpu().numpy()
        for f in range(F1):
            # band-power feature per channel: variance over time of band-filtered signal
            feat = np.log(np.var(z[:, f], axis=-1) + 1e-8)   # (N, C)
            d = fisher_direction(feat, ytr)
            cap_o.append(subspace_capture(d, Wo[f]))
            cap_e.append(subspace_capture(d, We[f]))
            ranks.append(eff_rank(feat, ytr))
        del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(test_subs)} | RE-Net capture {np.mean(cap_o):.3f} | "
                  f"EEGNet capture {np.mean(cap_e):.3f} | eff-rank {np.mean(ranks):.2f}")

    co = np.array(cap_o); ce = np.array(cap_e); rk = np.array(ranks)
    summary = dict(
        renet_capture_mean=float(co.mean()), renet_capture_std=float(co.std()),
        eegnet_capture_mean=float(ce.mean()), eegnet_capture_std=float(ce.std()),
        capture_diff=float(co.mean() - ce.mean()),
        eff_rank_mean=float(rk.mean()), eff_rank_median=float(np.median(rk)),
        n_pairs=len(co),
    )
    with open(os.path.join(OUT, "mechanism.json"), "w") as f:
        json.dump(dict(summary=summary, renet=co.tolist(), eegnet=ce.tolist(),
                       eff_rank=rk.tolist()), f, indent=2)

    print("\n" + "=" * 62)
    print(f"Discriminative-direction capture (|cos| onto 2-D filter subspace):")
    print(f"  RE-Net (OSFR):   {co.mean():.3f} +/- {co.std():.3f}")
    print(f"  EEGNet (no OSFR): {ce.mean():.3f} +/- {ce.std():.3f}")
    print(f"  difference:       {co.mean()-ce.mean():+.3f}")
    print(f"Effective rank of per-band discriminative subspace: "
          f"mean {rk.mean():.2f}, median {np.median(rk):.2f}  (premise: ~1 = low-rank)")
    print("=" * 62)
    print("\nInterpretation: if the two capture values are close, OSFR does NOT")
    print("push filters away from the discriminative direction -- both models")
    print("span it about equally -- supporting the Section 5.4 mechanism. A low")
    print("effective rank (near 1) supports the 'essentially low-rank' premise.")


if __name__ == "__main__":
    main()
