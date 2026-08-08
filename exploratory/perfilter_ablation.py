"""
Per-filter attribution via ablation  (reviewer M1, method B)

Turns "analyzable" into an operational capability: because OSFR makes the two
within-band spatial filters orthogonal, the contribution of each filter to the
decision can be attributed cleanly and additively. For the entangled EEGNet
filters it cannot -- ablating one filter is largely compensated by the other,
so single-filter ablations do not add up and cannot be attributed.

For each subject we train RE-Net (OSFR) and EEGNet, then at TEST time, per band,
we measure the accuracy drop from zeroing:
  - filter 1 only        (drop1)
  - filter 2 only        (drop2)
  - both filters         (drop_both)
Additivity gap = (drop1 + drop2) - drop_both.
  * Near 0  => contributions are separable/additive => attributable per filter.
  * Large   => strong overlap/redundancy => single-filter ablation not attributable.

We report, averaged over subjects and bands:
  - mean |additivity gap| for RE-Net vs EEGNet
  - correlation between additivity gap and inter-filter |r| (redundancy predicts
    non-additivity)

No topographies, no retraining beyond the models. Uses accuracy only.

Usage:  python perfilter_ablation.py
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
OUT = os.path.join(BASE, "perfilter_run"); os.makedirs(OUT, exist_ok=True)
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

    def _spatial_out(self, x):
        return self.spatial(self.block1(x.unsqueeze(1)))   # (N, F1*D, 1, T)

    def forward_from_spatial(self, sp):
        return self.head(self.block2(self.act1(self.bn1(sp))).flatten(1))

    def forward(self, x):
        return self.forward_from_spatial(self._spatial_out(x))


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


@torch.no_grad()
def acc_with_ablation(model, Xte, yte, band, which):
    """which: 'none' | 'f1' | 'f2' | 'both' -> zero those filter channels of `band`."""
    sp = model._spatial_out(Xte.to(device))   # (N, F1*D, 1, T)
    D = model.D
    ch1 = band * D + 0
    ch2 = band * D + 1
    if which in ("f1", "both"): sp[:, ch1] = 0
    if which in ("f2", "both"): sp[:, ch2] = 0
    pred = model.forward_from_spatial(sp).argmax(1).cpu().numpy()
    return accuracy_score(yte, pred)


def band_corr(model, band, C):
    W = model.spatial.weight.detach().view(model.F1, model.D, C).cpu().numpy()[band]
    return abs(np.corrcoef(W[0], W[1])[0, 1])


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Per-filter ablation: {test_subs and len(test_subs)} subjects x {F1} bands\n")

    rows = {"renet": [], "eegnet": []}   # each: dict(gap, corr, drop1, drop2, drop_both)
    for si, subj in enumerate(test_subs):
        for name, use_osfr in [("renet", True), ("eegnet", False)]:
            model = train(Xt, yt, s, subj, C, T, n_classes, use_osfr)
            te = np.where(s == subj)[0]
            Xte, yte = Xt[te], y[te]
            base = acc_with_ablation(model, Xte, yte, 0, "none")  # base same for all bands
            for b in range(F1):
                a_none = base
                a1 = acc_with_ablation(model, Xte, yte, b, "f1")
                a2 = acc_with_ablation(model, Xte, yte, b, "f2")
                ab = acc_with_ablation(model, Xte, yte, b, "both")
                drop1, drop2, drop_both = a_none - a1, a_none - a2, a_none - ab
                gap = (drop1 + drop2) - drop_both
                rows[name].append(dict(subj=subj, band=b, corr=float(band_corr(model, b, C)),
                                       drop1=float(drop1), drop2=float(drop2),
                                       drop_both=float(drop_both), gap=float(gap)))
            del model; torch.cuda.empty_cache() if use_cuda else None
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(test_subs)} subjects done")

    def summarize(name):
        g = np.array([r["gap"] for r in rows[name]])
        c = np.array([r["corr"] for r in rows[name]])
        return dict(
            mean_abs_gap=float(np.mean(np.abs(g))),
            median_abs_gap=float(np.median(np.abs(g))),
            mean_corr=float(np.mean(c)),
            corr_gap_vs_r=float(np.corrcoef(np.abs(g), c)[0, 1]) if len(g) > 2 else float("nan"),
        )

    summ = {k: summarize(k) for k in rows}
    with open(os.path.join(OUT, "perfilter_results.json"), "w") as f:
        json.dump(dict(summary=summ, rows=rows), f, indent=2)

    print("\n" + "=" * 66)
    print(f"{'model':>8} | {'mean|additivity gap|':>20} | {'mean |r|':>9} | {'corr(gap,|r|)':>13}")
    print("-" * 66)
    for k in ("renet", "eegnet"):
        d = summ[k]
        print(f"{k:>8} | {d['mean_abs_gap']*100:>18.2f}pp | {d['mean_corr']:>9.3f} | {d['corr_gap_vs_r']:>13.3f}")
    print("=" * 66)
    print("\nReading:")
    print("  additivity gap = (drop_f1 + drop_f2) - drop_both, per band.")
    print("  ~0  => the two filters' contributions add up => attributable separately.")
    print("  >0  => overlap: ablating one is compensated by the other => not")
    print("         cleanly attributable.")
    print("  If RE-Net's mean |gap| is small and EEGNet's is larger, and if")
    print("  corr(gap,|r|)>0 (redundancy predicts non-additivity), then OSFR")
    print("  delivers per-filter attributability that the backbone does not.")

    # simple bar figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [summ["renet"]["mean_abs_gap"]*100, summ["eegnet"]["mean_abs_gap"]*100]
    ax.bar([0, 1], vals, color=["#2b6cb0", "#c53030"], alpha=0.8, width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RE-Net\n(OSFR)", "EEGNet\n(no OSFR)"])
    ax.set_ylabel("Mean |additivity gap| (percentage points)")
    ax.set_title("Per-filter attributability\n(smaller = contributions add up cleanly)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG, "perfilter_attribution.png")
    plt.savefig(out, dpi=300); print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
