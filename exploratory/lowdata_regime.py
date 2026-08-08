"""
Low-data regime experiment  (does the OSFR prior earn its keep?)

A geometric prior should matter most when calibration data is scarce.
This experiment holds the LOSO test protocol fixed but caps the number of
TRAINING subjects available to each fold: N_train in {15, 30, 50, 80, all}.
For each cap it compares RE-Net (EEGNet + OSFR) against the plain EEGNet
backbone under the identical, leakage-free protocol used in the paper.

If the RE-Net minus EEGNet gap widens as N_train shrinks, OSFR provides
increasing value in the low-calibration regime that dominates real BCI
deployment -- the paper's missing "so what". If the gap stays flat, we report
that honestly and the prior remains a zero-cost non-redundancy device only.

To keep runtime feasible we evaluate on a fixed random subset of TEST subjects
(default 30 of 109) -- the same test set across all conditions and both models,
so comparisons are paired and fair. Everything else (val split from training
subjects, seeding, early stopping) matches common.py exactly.

Usage:
  python lowdata_regime.py                 # physionet, 30 test subjects
  TEST_K=40 python lowdata_regime.py
Env: DATASET=physionet
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
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(BASE, "lowdata_run"); os.makedirs(OUT, exist_ok=True)

TEST_K   = int(os.environ.get("TEST_K", "30"))     # held-out test subjects to average over
CAPS     = [15, 30, 50, 80, 0]                       # 0 = use all available training subjects
LAMBDA   = 0.10
NOISE    = 0.03
WD       = 0.01
EPOCHS   = 80


class Net(nn.Module):
    """EEGNet backbone; OSFR applied externally when use_osfr=True."""
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


def train_eval(Xt, yt, s, test_subj, train_pool, C, T, n_classes, use_osfr, seed):
    """Train on train_pool (capped), validate on split from train_pool,
    test on test_subj. Returns test accuracy."""
    set_seed(seed)
    tr_all = np.where(np.isin(s, train_pool))[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, seed)
    tr, val = tr_all[ti], tr_all[vi]
    te = np.where(s == test_subj)[0]

    model = Net(C, T, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WD)
    ce = nn.CrossEntropyLoss(); best, state = -1, None
    for ep in range(EPOCHS):
        model.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device) * NOISE
            by = yt[idx].to(device)
            opt.zero_grad()
            loss = ce(model(bx), by)
            if use_osfr:
                loss = loss + LAMBDA * osfr_loss(model)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                a = accuracy_score(yt[val].numpy(), model(Xt[val].to(device)).argmax(1).cpu().numpy())
            if a > best: best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        pt = model(Xt[te].to(device)).argmax(1).cpu().numpy()
    return accuracy_score(yt[te].numpy(), pt)


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)

    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Low-data regime: {len(test_subs)} test subjects, caps={CAPS}")
    print(f"(cap 0 = all remaining training subjects)\n")

    results = {}   # cap -> {'renet':[accs], 'eegnet':[accs]}
    for cap in CAPS:
        rn, eg = [], []
        for tsub in test_subs:
            pool = [u for u in subs if u != tsub]
            if cap and cap < len(pool):
                # deterministic per-(test,cap) training subset for paired comparison
                r = np.random.RandomState(SEED + tsub + cap)
                pool = sorted(r.choice(pool, size=cap, replace=False).tolist())
            seed = SEED + tsub
            rn.append(train_eval(Xt, yt, s, tsub, pool, C, T, n_classes, True,  seed))
            eg.append(train_eval(Xt, yt, s, tsub, pool, C, T, n_classes, False, seed))
        rn, eg = np.array(rn), np.array(eg)
        label = "all" if cap == 0 else str(cap)
        results[label] = dict(renet=rn.tolist(), eegnet=eg.tolist(),
                              renet_mean=float(rn.mean()*100), eegnet_mean=float(eg.mean()*100),
                              delta=float((rn-eg).mean()*100), delta_std=float((rn-eg).std()*100))
        print(f"N_train={label:>3} | RE-Net {rn.mean()*100:5.2f} | EEGNet {eg.mean()*100:5.2f} "
              f"| delta {(rn-eg).mean()*100:+5.2f} (±{(rn-eg).std()*100:4.2f})")

    with open(os.path.join(OUT, "lowdata_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}/lowdata_results.json")

    # plot delta vs N_train
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = [k for k in ["15","30","50","80","all"] if k in results]
    xs = np.arange(len(labels))
    deltas = [results[k]["delta"] for k in labels]
    rmean  = [results[k]["renet_mean"] for k in labels]
    emean  = [results[k]["eegnet_mean"] for k in labels]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, rmean, "o-", color="#2b6cb0", label="RE-Net (OSFR)")
    ax[0].plot(xs, emean, "s-", color="#c53030", label="EEGNet")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labels)
    ax[0].set_xlabel("# training subjects"); ax[0].set_ylabel("LOSO accuracy (%)")
    ax[0].set_title("Accuracy vs. training-set size"); ax[0].legend()
    ax[1].plot(xs, deltas, "o-", color="#2f855a")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels)
    ax[1].set_xlabel("# training subjects"); ax[1].set_ylabel("RE-Net − EEGNet (pp)")
    ax[1].set_title("OSFR advantage vs. training-set size")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "lowdata_regime.png"), dpi=300)
    print(f"Saved {FIG}/lowdata_regime.png")

    print("\nInterpretation:")
    print("  If the right panel trends UP as training subjects shrink,")
    print("  OSFR provides increasing value in the low-calibration regime")
    print("  -> the paper's practical 'so what'. If flat, report honestly.")


if __name__ == "__main__":
    main()
