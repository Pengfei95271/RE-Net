"""
Analyzability put to use: per-filter discriminative-contribution diagnosis
(reviewer #1 -- make analyzability *do* something, not just be demonstrated)

Because OSFR makes the two within-band spatial filters orthogonal, the
discriminative contribution of EACH filter can be attributed cleanly and
independently. We use this to perform a concrete diagnostic that the entangled
EEGNet cannot support: for every (subject, band, filter) we measure how much
class-discriminative information that single filter's output carries (AUC of a
1-D logistic on the filter's band-power), and we ask whether the two filters of
a band carry SEPARATE, non-redundant information.

Diagnostic quantities per band:
  auc1, auc2         : single-filter discriminative AUC (each filter alone)
  auc_both           : both filters together
  redundancy         : (auc1 + auc2 - 0.5) - auc_both   (how much the two overlap)
                       ~0  => contributions are separable/additive (clean attribution)
                       >0  => the two filters carry overlapping info (entangled)

Under OSFR the two filters are orthogonal, so their contributions should be
cleanly separable (redundancy ~ 0) and each filter is individually interpretable
as "carrying X% of the band's discriminative power". Under EEGNet the entangled
filters should overlap (redundancy > 0), so a per-filter attribution is not
trustworthy -- you cannot say what each filter contributes on its own.

Concretely, this lets an analyst DIAGNOSE a model: with OSFR you can point to a
band and say "filter 1 carries the discriminative signal here, filter 2 is
near-chance and could be pruned" -- a statement the entangled backbone does not
license. We report how often such a clean per-filter verdict is possible.

Uses already-trained-style models (trains once per subject, few subjects).
Outputs diag_run/diagnosis.json and prints a summary.

Usage:  python analyzability_diagnosis.py
Env:    DATASET=physionet  TEST_K=12
"""
import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "diag_run"); os.makedirs(OUT, exist_ok=True)
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

    def forward(self, x):
        x = self.act1(self.bn1(self.spatial(self.block1(x.unsqueeze(1)))))
        return self.head(self.block2(x).flatten(1))

    def spatial_out(self, x, batch=128):
        """output of the depthwise spatial conv, per (band,filter): (N, F1*D, T).
        Batched to avoid OOM; returned on CPU."""
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb = x[i:i+batch].to(device)
                z = self.spatial(self.block1(xb.unsqueeze(1))).squeeze(2).cpu()  # (b,F1*D,T)
                outs.append(z); del xb, z
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
    from sklearn.metrics import accuracy_score
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


def bandpower_feat(sp, band, D):
    """log band-power of each of the D filters of `band`: returns (N, D)."""
    ch = [band*D + j for j in range(D)]
    x = sp[:, ch, :].numpy()                    # (N, D, T)
    return np.log(np.var(x, axis=-1) + 1e-8)    # (N, D)


def auc_1d(x1, y):
    """AUC of a single feature via 1-D logistic (sign-agnostic)."""
    try:
        clf = LogisticRegression(max_iter=200).fit(x1.reshape(-1, 1), y)
        p = clf.predict_proba(x1.reshape(-1, 1))[:, 1]
        return max(roc_auc_score(y, p), roc_auc_score(y, -p))
    except Exception:
        return 0.5


def auc_2d(x2, y):
    try:
        clf = LogisticRegression(max_iter=200).fit(x2, y)
        p = clf.predict_proba(x2)[:, 1]
        return roc_auc_score(y, p)
    except Exception:
        return 0.5


def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Analyzability diagnosis: {len(test_subs)} subjects x {F1} bands\n")

    red_o, red_e = [], []          # redundancy per band
    clean_o, clean_e = 0, 0        # count of clean per-filter verdicts
    total = 0
    for si, subj in enumerate(test_subs):
        te = np.where(s == subj)[0]
        Xte, yte = Xt[te], y[te]
        for name, use_osfr, redlist in [("renet", True, red_o), ("eegnet", False, red_e)]:
            m = train(Xt, yt, s, subj, C, T, ncl, use_osfr)
            sp = m.spatial_out(Xte)             # (N, F1*D, T) on CPU
            for f in range(F1):
                feat = bandpower_feat(sp, f, 2)  # (N,2)
                a1 = auc_1d(feat[:, 0], yte)
                a2 = auc_1d(feat[:, 1], yte)
                ab = auc_2d(feat, yte)
                redundancy = (a1 + a2 - 0.5) - ab
                redlist.append(float(redundancy))
                # "clean verdict": one filter clearly discriminative (>0.6),
                # the other clearly near-chance (<0.55) -> can name each filter's role
                hi, lo = max(a1, a2), min(a1, a2)
                clean = (hi > 0.60 and lo < 0.55)
                if name == "renet":
                    clean_o += int(clean); total += 1
                else:
                    clean_e += int(clean)
            del m; torch.cuda.empty_cache() if use_cuda else None
        if (si+1) % 4 == 0:
            print(f"  {si+1}/{len(test_subs)} | RE-Net redun {np.mean(red_o):.3f} | "
                  f"EEGNet redun {np.mean(red_e):.3f}")

    ro = np.array(red_o); re = np.array(red_e)
    summary = dict(
        renet_redundancy_mean=float(ro.mean()), renet_redundancy_std=float(ro.std()),
        eegnet_redundancy_mean=float(re.mean()), eegnet_redundancy_std=float(re.std()),
        renet_clean_verdict_frac=float(clean_o/total) if total else 0.0,
        eegnet_clean_verdict_frac=float(clean_e/total) if total else 0.0,
        n_bands=total,
    )
    with open(os.path.join(OUT, "diagnosis.json"), "w") as f:
        json.dump(dict(summary=summary, renet_redundancy=ro.tolist(),
                       eegnet_redundancy=re.tolist()), f, indent=2)

    print("\n" + "=" * 64)
    print("Per-filter attribution redundancy  (0 = separable/clean, >0 = overlap):")
    print(f"  RE-Net (OSFR):   {ro.mean():+.3f} +/- {ro.std():.3f}")
    print(f"  EEGNet (no OSFR): {re.mean():+.3f} +/- {re.std():.3f}")
    print(f"\nFraction of bands with a CLEAN per-filter verdict")
    print(f"(one filter discriminative >0.60 AUC, the other near-chance <0.55):")
    print(f"  RE-Net (OSFR):   {clean_o/total:.1%}" if total else "  n/a")
    print(f"  EEGNet (no OSFR): {clean_e/total:.1%}" if total else "  n/a")
    print("=" * 64)
    print("\nUse: with OSFR, a clean per-filter verdict lets an analyst point to a")
    print("band and say 'filter 1 carries the discriminative signal, filter 2 is")
    print("near-chance and could be pruned' -- a diagnostic the entangled backbone")
    print("does not license. If RE-Net's redundancy is lower and its clean-verdict")
    print("rate higher, analyzability is not just shown but USED.")


if __name__ == "__main__":
    main()
