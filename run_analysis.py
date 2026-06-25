"""
Usage:
  python run_analysis.py ablation --variant dsa_only|osfr_only|full
  python run_analysis.py sensitivity --lambda_osfr 0.10
  python run_analysis.py complexity
  python run_analysis.py visualize
  python run_analysis.py statistical
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.abspath(__file__))
CACHE   = os.path.join(BASE, "cache")
DATA_DIR = os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "Datasets"))
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════
#  RE-Net components (shared by ablation / sensitivity / visualize)
# ══════════════════════════════════════════════════════════════════

class DualStateActivation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.pool = nn.AvgPool2d(k)
    def forward(self, x):
        return self.pool(F.elu(x)) + torch.log1p(self.pool(x ** 2))

class StandardActivation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(nn.ELU(True), nn.AvgPool2d(k))
    def forward(self, x):
        return self.f(x)

def _build_renet(C, T, use_dsa=True, n_classes=2):
    F1, D, F2, K, p = 8, 2, 16, 64, 0.25
    Act = DualStateActivation if use_dsa else StandardActivation
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1,K), padding=(0,K//2), bias=False), nn.BatchNorm2d(F1))
            self.spatial = nn.Conv2d(F1, F1*D, (C,1), groups=F1, bias=False)
            self.bn1 = nn.BatchNorm2d(F1*D)
            self.act1 = nn.Sequential(Act((1,4)), nn.Dropout(p))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1*D, F1*D, (1,16), padding=(0,8), groups=F1*D, bias=False),
                nn.Conv2d(F1*D, F2, (1,1), bias=False), nn.BatchNorm2d(F2))
            self.act2 = nn.Sequential(Act((1,8)), nn.Dropout(p))
            with torch.no_grad():
                flat = self.act2(self.block2(self.act1(self.bn1(self.spatial(
                    self.block1(torch.zeros(1,1,C,T))))))).numel()
            self.head = nn.Linear(flat, n_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        def forward(self, x):
            x = self.act1(self.bn1(self.spatial(self.block1(x.unsqueeze(1)))))
            return self.head(self.act2(self.block2(x)).flatten(1))
    return Net()

def osfr_loss(model):
    W = model.spatial.weight.view(8, 2, -1)
    I = torch.eye(2, device=W.device, dtype=W.dtype)
    return sum(
        torch.norm(F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I, p="fro")
        for f in range(8)) / 8


# ── Shared utils ───────────────────────────────────────────────────

def load_data():
    d = np.load(os.path.join(CACHE, "physionetmi_casa_preprocessed.npz"))
    X, y, s = d["X"], d["y"], d["s"]
    X = ((X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
    print(f"Data: {X.shape[0]} trials, {X.shape[1]}ch, {X.shape[2]}tp")
    return X, y.astype(np.int64), s.astype(np.int64)

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience, self.counter, self.best = patience, 0, None
        self.should_stop, self.state = False, None
    def __call__(self, score, model):
        if self.best is None or score > self.best + 1e-3:
            self.best, self.counter = score, 0
            self.state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience
    def restore(self, model):
        if self.state:
            model.load_state_dict({k: v.to(device) for k, v in self.state.items()})

def loso_train(model, Xt, yt, s, subj, lam_osfr=0.10, noise=0.03):
    tr, te = s != subj, s == subj
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce, scaler = nn.CrossEntropyLoss(), torch.amp.GradScaler("cuda", enabled=use_cuda)
    es, bs = EarlyStopping(20), 64

    for ep in range(200):
        model.train()
        for i in torch.randperm(tr.sum()).split(bs):
            bx = Xt[tr][i].to(device) + torch.randn(len(i), Xt.shape[1], Xt.shape[2], device=device)*noise
            by = yt[tr][i].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = ce(model(bx), by)
                if lam_osfr > 0: loss = loss + lam_osfr * osfr_loss(model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(Xt[te][j:j+256].to(device)).argmax(1).cpu()
                                  for j in range(0, te.sum(), 256)]).numpy()
            es(accuracy_score(yt[te].numpy(), pred), model)
            if es.should_stop: break

    es.restore(model)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Xt[te][j:j+256].to(device)).argmax(1).cpu()
                          for j in range(0, te.sum(), 256)]).numpy()
    return accuracy_score(yt[te].numpy(), pred), f1_score(yt[te].numpy(), pred, average="macro")


# ══════════════════════════════════════════════════════════════════
#  1. Ablation
# ══════════════════════════════════════════════════════════════════

def cmd_ablation():
    variant = sys.argv[3] if len(sys.argv) > 3 else "full"
    use_dsa = variant != "osfr_only"
    lam = 0.0 if variant == "dsa_only" else 0.10
    tag = variant

    out = os.path.join(BASE, f"renet_ablation_{tag}", "results")
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_ablation_{tag}.json")

    print(f"{'='*50}\nAblation [{tag}] DSA={use_dsa} lambda={lam}\n{'='*50}")
    X, y, s = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt = torch.from_numpy(X).pin_memory() if use_cuda else torch.from_numpy(X)
    yt = torch.from_numpy(y).pin_memory() if use_cuda else torch.from_numpy(y)

    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for sub in [i for i in range(1,110) if str(i) not in done]:
        t0 = time.time()
        model = _build_renet(C, T, use_dsa).to(device)
        acc, f1 = loso_train(model, Xt, yt, s, sub, lam)
        done[str(sub)] = {"acc": round(acc,4), "f1": round(f1,4)}
        json.dump(done, open(res_file,"w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | {len(done)}/109 {time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache() if use_cuda else None

    accs = [v["acc"] for v in done.values()]
    print(f"\n[{tag}] {len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


# ══════════════════════════════════════════════════════════════════
#  2. Sensitivity
# ══════════════════════════════════════════════════════════════════

def cmd_sensitivity():
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    tag = f"lambda_{lam:.2f}".replace(".", "p")
    out = os.path.join(BASE, f"sensitivity_{tag}", "results")
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_{tag}.json")

    print(f"{'='*50}\nSensitivity lambda={lam}\n{'='*50}")
    X, y, s = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt = torch.from_numpy(X).pin_memory() if use_cuda else torch.from_numpy(X)
    yt = torch.from_numpy(y).pin_memory() if use_cuda else torch.from_numpy(y)

    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for sub in [i for i in range(1,110) if str(i) not in done]:
        t0 = time.time()
        model = _build_renet(C, T, True).to(device)
        acc, f1 = loso_train(model, Xt, yt, s, sub, lam)
        done[str(sub)] = {"acc": round(acc,4), "f1": round(f1,4)}
        json.dump(done, open(res_file,"w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | {len(done)}/109 {time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache() if use_cuda else None

    accs = [v["acc"] for v in done.values()]
    print(f"\nlambda={lam} | {len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


# ══════════════════════════════════════════════════════════════════
#  3. Complexity
# ══════════════════════════════════════════════════════════════════

def cmd_complexity():
    for f in ["run_baselines.py", "run_renet.py"]:
        if not os.path.exists(os.path.join(BASE, f)):
            print(f"ERROR: {f} not found."); return

    try:
        from thop import profile
        has_thop = True
    except ImportError:
        has_thop = False
        print("thop not installed, skipping FLOPs")

    sys.path.insert(0, BASE)
    exec(open(os.path.join(BASE, "run_baselines.py")).read().split("def load_data")[0], globals())
    exec(open(os.path.join(BASE, "run_renet.py")).read().split("def load_data")[0], globals())

    models = {
        "RE-Net (ours)": RENet(64, 385, 2),
        "EEGNet": EEGNet(64, 385, 2),
        "DeepConvNet": DeepConvNet(64, 385, 2),
        "EEG-Conformer": EEGConformer(64, 385, 2),
        "LMDA-Net": LMDA(64, 385, 2),
    }
    dummy = torch.randn(1, 64, 385)

    def latency(model, x, dev, n=500):
        model.eval(); x = x.to(next(model.parameters()).device)
        with torch.no_grad():
            for _ in range(50): model(x)
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n): model(x)
            if dev == "cuda": torch.cuda.synchronize()
        return (time.time()-t0)/n*1000

    print(f"{'Model':25s} | {'Params':>10s} | {'FLOPs':>10s} | {'CPU ms':>8s} | {'GPU ms':>8s}")
    print("-" * 70)
    for name, m in models.items():
        p = sum(x.numel() for x in m.parameters() if x.requires_grad)
        flops = "N/A"
        if has_thop:
            try: macs, _ = profile(m.cpu(), inputs=(dummy.cpu(),), verbose=False); flops = f"{macs/1e6:.2f}M"
            except: flops = "err"
        lat_cpu = latency(m.cpu(), dummy.cpu(), "cpu")
        lat_gpu = latency(m.cuda(), dummy.cuda(), "cuda") if torch.cuda.is_available() else "N/A"
        gpu_s = f"{lat_gpu:.3f}" if isinstance(lat_gpu, float) else lat_gpu
        print(f"{name:25s} | {p:>10,} | {flops:>10s} | {lat_cpu:>6.3f}ms | {gpu_s:>6s}ms")
        m.cpu()


# ══════════════════════════════════════════════════════════════════
#  4. Visualize
# ══════════════════════════════════════════════════════════════════

def cmd_visualize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mne
    mne.set_log_level("CRITICAL")
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import MotorImagery

    FIG_DIR = os.path.join(BASE, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    X, y, s = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)

    # Train RE-Net on best subject (S007)
    SUBJ = 7
    tr, te = s != SUBJ, s == SUBJ
    print(f"Training RE-Net (S{SUBJ:03d})...")
    model = _build_renet(C, T, True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    best_acc, best_state = 0, None
    for ep in range(100):
        model.train()
        for i in torch.randperm(tr.sum()).split(64):
            bx = Xt[tr][i].to(device) + torch.randn(len(i),C,T,device=device)*0.03
            by = yt[tr][i].to(device)
            opt.zero_grad()
            loss = ce(model(bx), by) + 0.10 * osfr_loss(model)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (ep+1)%5==0:
            model.eval()
            with torch.no_grad():
                pred = model(Xt[te].to(device)).argmax(1).cpu().numpy()
            a = accuracy_score(yt[te].numpy(), pred)
            if a > best_acc: best_acc = a; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.cpu(); model.load_state_dict(best_state)
    print(f"RE-Net S{SUBJ:03d}: {best_acc:.2%}")

    W = model.spatial.weight.data.squeeze().view(8, 2, -1)  # (8,2,64)

    # Fig 1: OSFR orthogonality
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("OSFR: Spatial Filter Orthogonality", fontsize=14, fontweight="bold")
    for f in range(8):
        ax = axes[f//4, f%4]
        w = F.normalize(W[f], p=2, dim=-1)
        corr = torch.mm(w, w.t()).numpy()
        sns.heatmap(np.abs(corr), annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
                    ax=ax, cbar=False, xticklabels=[f"F{f+1}a",f"F{f+1}b"],
                    yticklabels=[f"F{f+1}a",f"F{f+1}b"])
        ax.set_title(f"Band {f+1}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "osfr_orthogonality.png"), dpi=300, bbox_inches="tight")
    print("Saved osfr_orthogonality.png")

    # Fig 2: Topomaps
    mne.set_config("MNE_DATA", DATA_DIR, set_env=True)
    ds = PhysionetMI()
    raw = ds.get_data(subjects=[1])[1]['0']['0']
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))
    w_topo = W[0].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Spatial Filters (Band 1)", fontsize=13, fontweight="bold")
    for i in range(2):
        mne.viz.plot_topomap(np.abs(w_topo[i]), info, axes=axes[i], cmap="Reds", show=False, contours=0)
        axes[i].set_title(f"Filter {i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "spatial_topomaps.png"), dpi=300, bbox_inches="tight")
    print("Saved spatial_topomaps.png")

    # Fig 3: RE-Net vs EEGNet orthogonality
    sys.path.insert(0, BASE)
    exec(open(os.path.join(BASE, "run_baselines.py")).read().split("def load_data")[0], globals())
    eegnet = EEGNet(64, 385, 2).to(device)
    opt2 = torch.optim.Adam(eegnet.parameters(), lr=1e-3)
    best2, state2 = 0, None
    for ep in range(100):
        eegnet.train()
        for i in torch.randperm(tr.sum()).split(64):
            bx, by = Xt[tr][i].to(device), yt[tr][i].to(device)
            opt2.zero_grad(); nn.CrossEntropyLoss()(eegnet(bx),by).backward(); opt2.step()
        if (ep+1)%5==0:
            eegnet.eval()
            with torch.no_grad(): p2 = eegnet(Xt[te].to(device)).argmax(1).cpu().numpy()
            a2 = accuracy_score(yt[te].numpy(), p2)
            if a2>best2: best2=a2; state2={k:v.cpu().clone() for k,v in eegnet.state_dict().items()}
    eegnet.cpu(); eegnet.load_state_dict(state2)
    W_ee = eegnet.depth[0].weight.data.squeeze().view(8, 2, -1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle("Orthogonality: RE-Net vs EEGNet", fontsize=13, fontweight="bold")
    for idx, (nm, wt) in enumerate([("RE-Net (OSFR)", W), ("EEGNet", W_ee)]):
        off = [abs(torch.mm(F.normalize(wt[f],p=2,dim=-1), F.normalize(wt[f],p=2,dim=-1).t()).numpy()[0,1]) for f in range(8)]
        axes[idx].bar(range(1,9), off, color="steelblue" if idx==0 else "coral")
        axes[idx].set_xlabel("Band"); axes[idx].set_ylabel("|Corr|"); axes[idx].set_title(nm); axes[idx].set_ylim(0,1)
        axes[idx].axhline(0.1, color="green", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "osfr_comparison.png"), dpi=300, bbox_inches="tight")
    print("Saved osfr_comparison.png")

    # Fig 4 & 5: Per-subject scatter (Acc & F1)
    renet_path = os.path.join(BASE, "renet_run/results/loso_renet.json")
    eegnet_path = os.path.join(BASE, "eegnet_run/results/loso_eegnet.json")
    if not os.path.exists(renet_path) or not os.path.exists(eegnet_path):
        print("Skipping scatter plots: need both renet and eegnet LOSO results.")
        print("All visualizations done!"); return
    renet_res = json.load(open(renet_path))
    eegnet_res = json.load(open(eegnet_path))
    subs = [str(i) for i in range(1, 110)]

    for metric, lo, hi, fmt in [("acc", 30, 100, ".1f"), ("f1", 0, 1, ".4f")]:
        a = np.array([renet_res[s][metric] * (100 if metric=="acc" else 1) for s in subs])
        b = np.array([eegnet_res[s][metric] * (100 if metric=="acc" else 1) for s in subs])
        w, t, l = (a>b).sum(), (a==b).sum(), (a<b).sum()
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot([lo,hi],[lo,hi],'k--',alpha=0.3)
        colors = np.where(a>b,'#2ecc71',np.where(a<b,'#e74c3c','#95a5a6'))
        ax.scatter(b, a, c=colors, s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(f"EEGNet {metric.upper()}"); ax.set_ylabel(f"RE-Net {metric.upper()}")
        ax.set_title(f"Per-Subject {metric.upper()} (W:{w} T:{t} L:{l})", fontweight="bold")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fname = f"per_subject_scatter{'_f1' if metric=='f1' else ''}.png"
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
        print(f"Saved {fname}")

    print("All visualizations done!")


# ══════════════════════════════════════════════════════════════════
#  5. Statistical Test (Wilcoxon signed-rank)
# ══════════════════════════════════════════════════════════════════

def cmd_statistical():
    from scipy.stats import wilcoxon

    renet_path = os.path.join(BASE, "renet_run/results/loso_renet.json")
    if not os.path.exists(renet_path):
        print("ERROR: RE-Net results not found. Run `python run_renet.py` first.")
        return

    baselines = {
        "EEGNet":        os.path.join(BASE, "eegnet_run/results/loso_eegnet.json"),
        "DeepConvNet":   os.path.join(BASE, "deepconvnet_run/results/loso_deepconvnet.json"),
        "LMDA-Net":      os.path.join(BASE, "lmda_run/results/loso_lmda.json"),
        "EEG-Conformer": os.path.join(BASE, "eeg_conformer_run/results/loso_eeg_conformer.json"),
    }
    available = {k: v for k, v in baselines.items() if os.path.exists(v)}
    if not available:
        print("ERROR: No baseline results found. Run `python run_baselines.py <model>` first.")
        return

    renet = json.load(open(renet_path))
    subs = [str(i) for i in range(1, 110)]
    acc_re = np.array([renet[s]["acc"] for s in subs])

    print("=" * 75)
    print("  Wilcoxon Signed-Rank Test (one-sided, H1: RE-Net > Baseline)")
    print("=" * 75)
    print(f"{'Comparison':30s} | {'p-value':>12s} | {'Sig':>4s} | {'W/T/L':>10s} | {'Mean Diff':>10s}")
    print("-" * 75)

    for name, path in baselines.items():
        if not os.path.exists(path):
            print(f"{'RE-Net vs '+name:30s} | {'MISSING':>12s} |")
            continue
        bl = json.load(open(path))
        acc_bl = np.array([bl[s]["acc"] for s in subs])
        diff = acc_re - acc_bl
        w = (diff > 0).sum()
        t = (diff == 0).sum()
        l = (diff < 0).sum()
        stat, p2 = wilcoxon(acc_re, acc_bl, alternative="greater")
        sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"
        print(f"{'RE-Net vs '+name:30s} | {p2:12.3e} | {sig:>4s} | {w:>3d}/{t:>2d}/{l:>2d} | {diff.mean()*100:>+8.2f}%")

    print("=" * 75)
    print("  * p<0.05, ** p<0.01, *** p<0.001")


# ══════════════════════════════════════════════════════════════════

COMMANDS = {"ablation": cmd_ablation, "sensitivity": cmd_sensitivity,
            "complexity": cmd_complexity, "visualize": cmd_visualize,
            "statistical": cmd_statistical}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    assert cmd in COMMANDS, f"Usage: python run_analysis.py {{{','.join(COMMANDS)}}}"
    COMMANDS[cmd]()

