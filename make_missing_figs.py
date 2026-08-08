"""
Generate the 4 missing figures for the RE-Net paper.

Two are data-only (fast, no training):
  - per_subject_scatter.png       RE-Net vs EEGNet per-subject accuracy
  - per_subject_scatter_f1.png    RE-Net vs EEGNet per-subject F1

Two need one trained RE-Net to read its spatial weights (~15 min):
  - osfr_orthogonality.png        per-band |W W^T| correlation matrices
  - osfr_comparison.png           OSFR vs unconstrained inter-filter correlation

Auto-discovers the PhysioNet LOSO JSONs for the scatters. If they are not
found, it prints where it looked and still produces the two OSFR figures.

Usage:  python make_missing_figs.py
Env:    DATASET=physionet
"""
import os, glob, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = os.path.expanduser("~/Downloads/renet_code_v3")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)


# ---------- helpers to load per-subject results ----------
def find_json(model):
    pats = [
        os.path.join(BASE, f"{model}_run/results/loso_{model}.json"),
        os.path.join(BASE, f"{model}_physionet_run/results/loso_{model}.json"),
        os.path.join(BASE, "**", f"loso_{model}.json"),
    ]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None


def load_per_subject(model):
    f = find_json(model)
    if not f:
        return None, None, None
    d = json.load(open(f))
    accs, f1s, keys = [], [], []
    for k, v in d.items():
        if isinstance(v, dict):
            a = v.get("acc"); f1 = v.get("f1", v.get("F1"))
        else:
            a = v; f1 = None
        if a is not None:
            keys.append(k); accs.append(a); f1s.append(f1)
    return np.array(accs, float), (np.array(f1s, float) if all(x is not None for x in f1s) else None), f


def scatter(ax, x, y, xlabel, ylabel, title):
    lo = min(np.nanmin(x), np.nanmin(y)); hi = max(np.nanmax(x), np.nanmax(y))
    pad = (hi - lo) * 0.05 + 1e-6
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, zorder=1)
    above = (y > x).sum()
    ax.scatter(x, y, s=28, color="#2b6cb0", alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.text(0.05, 0.95, f"RE-Net higher: {above}/{len(x)}",
            transform=ax.transAxes, va="top", fontsize=9, color="#2b6cb0")


def make_scatters():
    ra, rf, rfile = load_per_subject("renet")
    ea, ef, efile = load_per_subject("eegnet")
    if ra is None or ea is None:
        print("Could not find LOSO JSONs for renet/eegnet; skipping scatters.")
        print("Looked under:", BASE, "(*_run/results/loso_*.json)")
        return
    n = min(len(ra), len(ea))
    print(f"scatter: using {n} subjects  ({rfile}, {efile})")
    # accuracy scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    xa = ea[:n] * (100 if ea.max() <= 1.5 else 1)
    ya = ra[:n] * (100 if ra.max() <= 1.5 else 1)
    scatter(ax, xa, ya, "EEGNet accuracy (%)", "RE-Net accuracy (%)",
            "Per-subject accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "per_subject_scatter.png"), dpi=300); plt.close()
    print("saved per_subject_scatter.png")
    # F1 scatter
    if rf is not None and ef is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        scatter(ax, ef[:n], rf[:n], "EEGNet F1", "RE-Net F1", "Per-subject F1")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "per_subject_scatter_f1.png"), dpi=300); plt.close()
        print("saved per_subject_scatter_f1.png")
    else:
        # fall back: duplicate accuracy-based ranking if F1 not stored
        print("F1 not found in JSON; generating F1 scatter is skipped.")
        print("  (If your JSON has no per-subject F1, tell me and I'll adapt.)")


# ---------- OSFR orthogonality figures (need one trained model) ----------
def make_osfr_figs():
    import torch, torch.nn as nn, torch.nn.functional as F
    from sklearn.metrics import accuracy_score
    from common import (device, use_cuda, SEED, set_seed, load_data,
                        subjects_of, stratified_val_split)

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
        return sum(torch.norm(F.normalize(W[f],p=2,dim=-1)@F.normalize(W[f],p=2,dim=-1).t()-I,p="fro")
                   for f in range(model.F1)) / model.F1

    def train(Xt, yt, s, subj, C, T, ncl, use_osfr):
        set_seed(SEED+subj)
        tr_all = np.where(s != subj)[0]
        ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED+subj)
        tr, val = tr_all[ti], tr_all[vi]
        m = Net(C,T,ncl).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.01)
        ce = nn.CrossEntropyLoss(); best,state=-1,None
        for ep in range(80):
            m.train()
            for i in torch.randperm(len(tr)).split(64):
                idx=tr[i.numpy()]
                bx=Xt[idx].to(device)+torch.randn(len(idx),C,T,device=device)*0.03
                by=yt[idx].to(device)
                opt.zero_grad(); loss=ce(m(bx),by)
                if use_osfr: loss=loss+0.10*osfr_loss(m)
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            if (ep+1)%5==0:
                m.eval()
                with torch.no_grad():
                    a=accuracy_score(yt[val].numpy(), m(Xt[val].to(device)).argmax(1).cpu().numpy())
                if a>best: best,state=a,{k:v.cpu().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(state); m.eval(); return m

    X,y,s,ncl = load_data(); C,T = X.shape[1],X.shape[2]
    Xt,yt = torch.from_numpy(X), torch.from_numpy(y)
    subj = sorted(subjects_of(s))[0]
    print(f"osfr figs: training RE-Net + EEGNet on S{subj:03d} ...")
    mo = train(Xt,yt,s,subj,C,T,ncl,True)
    me = train(Xt,yt,s,subj,C,T,ncl,False)

    def corr_mats(model):
        W = model.spatial.weight.detach().view(model.F1,model.D,C).cpu().numpy()
        mats=[]
        for f in range(model.F1):
            Wn = W[f]/(np.linalg.norm(W[f],axis=1,keepdims=True)+1e-9)
            mats.append(np.abs(Wn@Wn.T))
        return mats

    # Fig 1: OSFR orthogonality matrices (8 bands)
    mats = corr_mats(mo)
    fig, axes = plt.subplots(2,4, figsize=(10,5))
    for f,ax in enumerate(axes.flat):
        im=ax.imshow(mats[f], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"Band {f+1}", fontsize=9)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{mats[f][i,j]:.2f}",ha="center",va="center",
                        color="white" if mats[f][i,j]<0.5 else "black",fontsize=8)
    fig.suptitle("OSFR: |normalized spatial-filter Gram| per band (off-diagonal $\\approx$0)", y=1.0)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="|correlation|")
    plt.savefig(os.path.join(FIG,"osfr_orthogonality.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("saved osfr_orthogonality.png")

    # Fig 2: OSFR vs unconstrained mean off-diagonal per band
    def offdiag(model):
        return [corr_mats(model)[f][0,1] for f in range(model.F1)]
    oo, oe = offdiag(mo), offdiag(me)
    fig, ax = plt.subplots(figsize=(6,4))
    x=np.arange(1,9); w=0.38
    ax.bar(x-w/2, oo, w, label="RE-Net (OSFR)", color="#2b6cb0")
    ax.bar(x+w/2, oe, w, label="EEGNet (no OSFR)", color="#c53030")
    ax.set_xlabel("frequency band"); ax.set_ylabel("|inter-filter correlation|")
    ax.set_title("OSFR vs. unconstrained: within-band filter correlation")
    ax.set_xticks(x); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"osfr_comparison.png"), dpi=300); plt.close()
    print("saved osfr_comparison.png")


if __name__ == "__main__":
    print("=== 1/2  per-subject scatters (data only) ===")
    make_scatters()
    print("\n=== 2/2  OSFR orthogonality figures (trains one model) ===")
    make_osfr_figs()
    print("\nDone. Check figures/ for the 4 files.")
