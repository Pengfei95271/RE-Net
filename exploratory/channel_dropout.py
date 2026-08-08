"""
Channel-dropout robustness  (M1: a possible practical value of OSFR)

Hypothesis: because OSFR spreads information across mutually orthogonal,
complementary spatial filters (topographies are less focal than the
unconstrained backbone), RE-Net may degrade more gracefully when EEG channels
are missing at test time -- a common real-world failure (loose/broken
electrodes) in zero-calibration BCI.

Design (no retraining beyond the models themselves):
  * Train RE-Net (EEGNet+OSFR) and plain EEGNet under the identical
    leakage-free LOSO protocol, on a subset of held-out test subjects.
  * At TEST time only, randomly zero out k of the C channels (k = 0,5,10,20),
    averaging over several random channel masks, and measure accuracy.
  * Compare the accuracy drop vs. k for the two models. If RE-Net's curve
    decays more slowly, OSFR confers channel-dropout robustness.

The channel masks are shared between the two models for each (subject, k, rep)
so the comparison is paired and fair. Everything else matches common.py.

Usage:  python channel_dropout.py
Env:    DATASET=physionet   TEST_K=20   REPS=5
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
OUT = os.path.join(BASE, "chandrop_run"); os.makedirs(OUT, exist_ok=True)

TEST_K = int(os.environ.get("TEST_K", "20"))   # held-out test subjects
REPS   = int(os.environ.get("REPS", "5"))       # random masks per k
KS     = [0, 5, 10, 20]                          # channels dropped
LAMBDA, NOISE, WD, EPOCHS = 0.10, 0.03, 0.01, 80


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


def train(Xt, yt, s, test_subj, C, T, n_classes, use_osfr):
    set_seed(SEED + test_subj)
    tr_all = np.where(s != test_subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + test_subj)
    tr, val = tr_all[ti], tr_all[vi]
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
            if use_osfr: loss = loss + LAMBDA * osfr_loss(model)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                a = accuracy_score(yt[val].numpy(), model(Xt[val].to(device)).argmax(1).cpu().numpy())
            if a > best: best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model


def eval_with_mask(model, Xte, yte, drop_idx):
    Xm = Xte.clone()
    if len(drop_idx):
        Xm[:, drop_idx, :] = 0.0
    with torch.no_grad():
        p = model(Xm.to(device)).argmax(1).cpu().numpy()
    return accuracy_score(yte, p)


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Channel-dropout robustness: {len(test_subs)} test subjects, "
          f"k in {KS}, {REPS} masks each\n")

    # acc[model][k] = list over (subject,rep)
    acc = {"renet": {k: [] for k in KS}, "eegnet": {k: [] for k in KS}}
    for tsub in test_subs:
        m_o = train(Xt, yt, s, tsub, C, T, n_classes, True)
        m_e = train(Xt, yt, s, tsub, C, T, n_classes, False)
        te = np.where(s == tsub)[0]
        Xte, yte = Xt[te], y[te]
        for k in KS:
            if k == 0:
                acc["renet"][k].append(eval_with_mask(m_o, Xte, yte, []))
                acc["eegnet"][k].append(eval_with_mask(m_e, Xte, yte, []))
            else:
                r = np.random.RandomState(SEED + tsub + k)
                for _ in range(REPS):
                    drop = r.choice(C, size=k, replace=False)   # shared mask both models
                    acc["renet"][k].append(eval_with_mask(m_o, Xte, yte, drop))
                    acc["eegnet"][k].append(eval_with_mask(m_e, Xte, yte, drop))
        del m_o, m_e; torch.cuda.empty_cache() if use_cuda else None

    print(f"{'k':>3} | {'RE-Net':>16} | {'EEGNet':>16} | {'drop RN':>8} | {'drop EG':>8}")
    print("-" * 64)
    base_rn = np.mean(acc["renet"][0]) * 100
    base_eg = np.mean(acc["eegnet"][0]) * 100
    summary = {}
    for k in KS:
        rn = np.mean(acc["renet"][k]) * 100
        eg = np.mean(acc["eegnet"][k]) * 100
        d_rn = rn - base_rn
        d_eg = eg - base_eg
        summary[k] = dict(renet=rn, eegnet=eg, drop_renet=d_rn, drop_eegnet=d_eg)
        print(f"{k:>3} | {rn:6.2f} ± {np.std(acc['renet'][k])*100:4.2f} | "
              f"{eg:6.2f} ± {np.std(acc['eegnet'][k])*100:4.2f} | "
              f"{d_rn:+7.2f} | {d_eg:+7.2f}")

    with open(os.path.join(OUT, "chandrop_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT}/chandrop_results.json")

    # plot accuracy vs k
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rn = [summary[k]["renet"] for k in KS]
    eg = [summary[k]["eegnet"] for k in KS]
    plt.figure(figsize=(6, 4))
    plt.plot(KS, rn, "o-", color="#2b6cb0", label="RE-Net (OSFR)")
    plt.plot(KS, eg, "s-", color="#c53030", label="EEGNet")
    plt.xlabel("# channels dropped at test time"); plt.ylabel("Accuracy (%)")
    plt.title("Robustness to missing channels"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "channel_dropout.png"), dpi=300)
    print(f"Saved {FIG}/channel_dropout.png")

    print("\nInterpretation:")
    print("  If RE-Net's accuracy drops LESS than EEGNet's as k grows")
    print("  (drop RN closer to 0 than drop EG), OSFR confers channel-dropout")
    print("  robustness -> a concrete practical value. If the drops match,")
    print("  report honestly and rely on the analyzability argument for M1.")


if __name__ == "__main__":
    main()
