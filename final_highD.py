"""
FINAL ACCURACY TEST -- Does OSFR help accuracy in the high-capacity regime?

Rationale. At D=2 (default) OSFR is accuracy-neutral: the discriminative signal
needs ~one direction, so orthogonalizing the second filter is free but useless
for accuracy. The one regime where OSFR *could* help is high D, where the
unconstrained backbone's extra filters collapse into redundant duplicates
(inter-filter corr rises to 0.29 at D=6, Table osfr-D) -- wasting capacity and
risking overfit -- while OSFR forces them into new orthogonal directions that
may capture secondary discriminative structure.

This script runs, on PhysioNet LOSO, D=6 for:
  (1) RE-Net  : EEGNet backbone + OSFR   (osfr on)
  (2) EEGNet  : same backbone, no OSFR   (osfr off)
  (3) EEGNet+WD: same backbone, no OSFR, but STRONGER weight decay matched to
                 OSFR's regularization strength -- the control that separates
                 "geometric prior" from "just more regularization".

If RE-Net(D=6) significantly beats EEGNet(D=6) AND also beats EEGNet+WD(D=6),
the gain is attributable to the orthogonality prior, not generic regularization.
If EEGNet+WD closes the gap, the effect is just regularization (report honestly).
If nothing separates, the discriminative signal is low-dim and OSFR gives no
accuracy at any D (report honestly, close the question).

Outputs highD_run/results_{tag}.json (resumable per subject) and a summary.

Usage:
  MODE=renet   python final_highD.py    # RE-Net D=6, OSFR on
  MODE=eegnet  python final_highD.py    # EEGNet D=6, OSFR off
  MODE=eegnetwd python final_highD.py   # EEGNet D=6, no OSFR, strong weight decay
Then:
  python final_highD.py summary         # aggregate + significance vs existing D=2

Env: DATASET=physionet  SEED=2024  D_MULT=6
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import wilcoxon

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "highD_run"); os.makedirs(OUT, exist_ok=True)
D_MULT = int(os.environ.get("D_MULT", "6"))


class Net(nn.Module):
    def __init__(self, C, T, n_classes=2, F1=8, D=6, F2=16, K=64, p=0.25):
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


def train_eval(Xt, yt, s, subj, C, T, ncl, use_osfr, wd):
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    te = np.where(s == subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    m = Net(C, T, ncl, D=D_MULT).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    ce = nn.CrossEntropyLoss(); best, st = -1, None
    for ep in range(200):
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
            # early stop
            if ep > 60 and a < best - 0.15: break
    m.load_state_dict(st); m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xt[te][j:j+256].to(device)).argmax(1).cpu()
                          for j in range(0, len(te), 256)]).numpy()
    yte = yt[te].numpy()
    del m; torch.cuda.empty_cache() if use_cuda else None
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def run(mode):
    cfgs = {
        "renet":    dict(use_osfr=True,  wd=0.01),
        "eegnet":   dict(use_osfr=False, wd=0.01),
        "eegnetwd": dict(use_osfr=False, wd=0.05),  # stronger WD control
    }
    cfg = cfgs[mode]
    res_file = os.path.join(OUT, f"results_{mode}_D{D_MULT}.json")
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    subs = subjects_of(s)
    print(f"MODE={mode} D={D_MULT} osfr={cfg['use_osfr']} wd={cfg['wd']} | {len(subs)} subjects")
    for sub in [sb for sb in subs if str(sb) not in done]:
        t0 = time.time()
        acc, f1 = train_eval(Xt, yt, s, sub, C, T, ncl, cfg["use_osfr"], cfg["wd"])
        done[str(sub)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        json.dump(done, open(res_file, "w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} | {len(done)}/{len(subs)} {time.time()-t0:.0f}s")
    accs = [v["acc"] for v in done.values()]
    print(f"\n{mode} D={D_MULT}: {np.mean(accs):.2%} +/- {np.std(accs):.2%} ({len(accs)} subj)")


def summary():
    def load(tag):
        f = os.path.join(OUT, f"results_{tag}_D{D_MULT}.json")
        return json.load(open(f)) if os.path.exists(f) else {}
    r, e, ewd = load("renet"), load("eegnet"), load("eegnetwd")
    common = sorted(set(r) & set(e), key=int)
    if not common:
        print("No overlapping subjects yet."); return
    ra = np.array([r[k]["acc"] for k in common])
    ea = np.array([e[k]["acc"] for k in common])
    print("=" * 60)
    print(f"High-D accuracy test (D={D_MULT}), {len(common)} subjects")
    print(f"  RE-Net (OSFR):    {ra.mean():.2%} +/- {ra.std():.2%}")
    print(f"  EEGNet (no OSFR): {ea.mean():.2%} +/- {ea.std():.2%}")
    print(f"  RE-Net - EEGNet:  {(ra.mean()-ea.mean())*100:+.2f} pts")
    try:
        _, p = wilcoxon(ra, ea, alternative="greater")
        print(f"  Wilcoxon (RE-Net > EEGNet) p = {p:.4f}")
    except Exception as ex:
        print("  Wilcoxon failed:", ex)
    if ewd:
        cw = sorted(set(r) & set(ewd), key=int)
        wa = np.array([ewd[k]["acc"] for k in cw])
        rw = np.array([r[k]["acc"] for k in cw])
        print(f"  EEGNet+strongWD:  {wa.mean():.2%} +/- {wa.std():.2%}")
        try:
            _, p2 = wilcoxon(rw, wa, alternative="greater")
            print(f"  Wilcoxon (RE-Net > EEGNet+WD) p = {p2:.4f}")
        except Exception:
            pass
    print("=" * 60)
    print("Recall D=2: RE-Net 64.36 vs EEGNet 63.46 (+0.90, ns).")
    print("Verdict guide:")
    print("  - RE-Net > EEGNet AND > EEGNet+WD, gap larger than at D=2")
    print("    => OSFR helps accuracy at high D via the orthogonality prior. POSITIVE.")
    print("  - EEGNet+WD closes the gap => effect is generic regularization, not geometry.")
    print("  - No separation => discriminative signal is low-dim; OSFR accuracy-neutral")
    print("    at all D. Close the question honestly.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary()
    else:
        mode = os.environ.get("MODE", "renet")
        assert mode in ("renet", "eegnet", "eegnetwd")
        run(mode)
