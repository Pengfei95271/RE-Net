"""
Cross-subject spatial-filter consistency  (reviewer M1, method C)

Hypothesis: because OSFR makes each band's two spatial filters orthogonal and
unit-norm, the 2-D subspace they span may be more consistent / aggregable
ACROSS subjects than the entangled EEGNet subspaces, which vary in how
correlated their two filters are.

Honest caveat built in: orthogonality within a subject does NOT by itself
force the subspace ORIENTATION to agree across subjects. So we compare three
things per band:
  (1) RE-Net  cross-subject subspace consistency
  (2) EEGNet  cross-subject subspace consistency
  (3) RANDOM orthonormal 2-D subspaces (control): the consistency you'd get
      from orthogonality ALONE with no learned structure.
If RE-Net > EEGNet AND RE-Net > random, the consistency is real learned
structure. If RE-Net ~ random, the "advantage" is just an artifact of
orthonormalization and we should not claim it.

Consistency metric per band:
  For each subject, the two filters (unit-normalized) span a 2-D subspace with
  orthonormal basis Q_s (C x 2). Between two subjects s,t the subspace affinity
  is the mean of squared principal-subspace cosines:
      aff(s,t) = ||Q_s^T Q_t||_F^2 / 2   in [0,1]  (1 = identical subspace)
  Band consistency = mean over all subject pairs of aff(s,t).
Higher = more consistent across subjects.

No topographies. Uses only the learned spatial weights (+ a random control).

Usage:  python crosssubj_consistency.py
Env:    DATASET=physionet   TEST_K=20
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
OUT = os.path.join(BASE, "crosssubj_run"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
TEST_K = int(os.environ.get("TEST_K", "20"))
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


def ortho_basis(W2):
    """W2: (2, C) -> orthonormal basis Q: (C, 2) via QR."""
    Q, _ = np.linalg.qr(W2.T)   # (C,2)
    return Q[:, :2]


def subspace_affinity(Qs, Qt):
    """mean squared principal cosines in [0,1]; 1 = identical 2-D subspace."""
    M = Qs.T @ Qt              # (2,2)
    return float(np.sum(M**2) / 2.0)


def band_consistency(bases):
    """mean pairwise affinity over a list of (C,2) bases."""
    n = len(bases)
    if n < 2: return float("nan")
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(subspace_affinity(bases[i], bases[j]))
    return float(np.mean(vals))


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Cross-subject consistency: {len(test_subs)} subjects x {F1} bands\n")

    # collect per-band bases for each model
    renet_bases = {b: [] for b in range(F1)}
    eeg_bases = {b: [] for b in range(F1)}
    for si, subj in enumerate(test_subs):
        m_o = train(Xt, yt, s, subj, C, T, n_classes, True)
        m_e = train(Xt, yt, s, subj, C, T, n_classes, False)
        Wo = m_o.spatial.weight.detach().view(F1, 2, C).cpu().numpy()
        We = m_e.spatial.weight.detach().view(F1, 2, C).cpu().numpy()
        for b in range(F1):
            renet_bases[b].append(ortho_basis(Wo[b]))
            eeg_bases[b].append(ortho_basis(We[b]))
        del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None
        if (si+1) % 5 == 0: print(f"  {si+1}/{len(test_subs)} subjects done")

    # random orthonormal control: same number of subjects, random 2-D subspaces
    rctrl = np.random.RandomState(123)
    rand_bases = {b: [] for b in range(F1)}
    for b in range(F1):
        for _ in range(len(test_subs)):
            Wr = rctrl.randn(2, C)
            rand_bases[b].append(ortho_basis(Wr))

    renet_c = np.array([band_consistency(renet_bases[b]) for b in range(F1)])
    eeg_c   = np.array([band_consistency(eeg_bases[b]) for b in range(F1)])
    rand_c  = np.array([band_consistency(rand_bases[b]) for b in range(F1)])

    summary = dict(
        renet_mean=float(renet_c.mean()), eeg_mean=float(eeg_c.mean()),
        random_mean=float(rand_c.mean()),
        renet_per_band=renet_c.tolist(), eeg_per_band=eeg_c.tolist(),
        random_per_band=rand_c.tolist(),
    )
    with open(os.path.join(OUT, "crosssubj_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 58)
    print(f"{'band':>4} | {'RE-Net':>8} | {'EEGNet':>8} | {'random':>8}")
    print("-" * 58)
    for b in range(F1):
        print(f"{b:>4} | {renet_c[b]:>8.3f} | {eeg_c[b]:>8.3f} | {rand_c[b]:>8.3f}")
    print("-" * 58)
    print(f"{'mean':>4} | {renet_c.mean():>8.3f} | {eeg_c.mean():>8.3f} | {rand_c.mean():>8.3f}")
    print("=" * 58)
    print("\nReading (subspace affinity in [0,1], higher = more consistent):")
    print("  Claim holds ONLY if  RE-Net > EEGNet  AND  RE-Net > random.")
    print("  If RE-Net ~ random, the apparent consistency is just an artifact")
    print("  of orthonormalization, not learned structure -> do not claim it.")
    print("  If RE-Net ~ EEGNet, OSFR gives no cross-subject consistency gain.")

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(F1); w = 0.27
    ax.bar(x - w, renet_c, w, label="RE-Net (OSFR)", color="#2b6cb0")
    ax.bar(x,     eeg_c,   w, label="EEGNet",        color="#c53030")
    ax.bar(x + w, rand_c,  w, label="random ortho",  color="#999999")
    ax.set_xlabel("frequency band"); ax.set_ylabel("cross-subject subspace affinity")
    ax.set_title("Cross-subject spatial-filter consistency")
    ax.legend(); plt.tight_layout()
    out = os.path.join(FIG, "crosssubj_consistency.png")
    plt.savefig(out, dpi=300); print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
