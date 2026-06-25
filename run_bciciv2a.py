"""
BCI Competition IV Dataset 2a — RE-Net & Baselines
=====================================================
Dataset : BNCI2014001 via MOABB
Subjects: 9, Sessions: 2, Classes: 4 (left hand / right hand / feet / tongue)
Channels: 22 EEG, Sampling rate: 250 Hz → resampled to 128 Hz
Epoch   : 2.0 – 6.0 s post-cue (4 s window, 512 → 513 tp at 128 Hz)

Evaluation protocol (standard in literature):
  - Within-subject, cross-session:
      Train on Session 1 (288 trials), test on Session 2 (288 trials)
  - No cross-subject split — this is the established BCI-IV 2a benchmark

Usage
-----
  python run_bciciv2a.py renet
  python run_bciciv2a.py eegnet
  python run_bciciv2a.py deepconvnet
  python run_bciciv2a.py conformer
  python run_bciciv2a.py lmda
  python run_bciciv2a.py all          ← runs every model sequentially
"""

import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
CACHE    = os.path.join(BASE, "cache")
DATA_DIR = os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "Datasets"))

# ── Device ─────────────────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cudnn.benchmark = True

# ── Dataset constants ──────────────────────────────────────────────
N_SUBJECTS  = 9
N_CLASSES   = 4
SUBJECTS    = list(range(1, N_SUBJECTS + 1))
EVENTS      = ["left_hand", "right_hand", "feet", "tongue"]
FMIN, FMAX  = 4, 40
TMIN, TMAX  = 2.0, 6.0          # standard 4 s window
RESAMPLE    = 128
N_CHANNELS  = 22


# ══════════════════════════════════════════════════════════════════
#  Model Definitions
# ══════════════════════════════════════════════════════════════════

# ── RE-Net ────────────────────────────────────────────────────────
class DualStateActivation(nn.Module):
    """DSA: phase (ELU) + power (x² → log)."""
    def __init__(self, pool_kernel):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_kernel)

    def forward(self, x):
        return self.pool(F.elu(x)) + torch.log1p(self.pool(x ** 2))


class RENet(nn.Module):
    def __init__(self, C, T, n_classes=N_CLASSES,
                 F1=8, D=2, F2=16, K=64, p=0.25):
        super().__init__()
        self.F1, self.D = F1, D

        self.block1  = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.spatial = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
        self.bn1     = nn.BatchNorm2d(F1 * D)
        self.act1    = nn.Sequential(DualStateActivation((1, 4)), nn.Dropout(p))

        self.block2  = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1),   bias=False),
            nn.BatchNorm2d(F2))
        self.act2    = nn.Sequential(DualStateActivation((1, 8)), nn.Dropout(p))

        with torch.no_grad():
            d = self.act2(self.block2(self.act1(self.bn1(self.spatial(
                self.block1(torch.zeros(1, 1, C, T)))))))
            flat = d.numel()
        self.head = nn.Linear(flat, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.bn1(self.spatial(self.block1(x.unsqueeze(1)))))
        return self.head(self.act2(self.block2(x)).flatten(1))


def osfr_loss(model):
    """Frobenius penalty on spatial filter Gram deviation from identity."""
    W = model.spatial.weight.view(model.F1, model.D, -1)   # (F1, D, C)
    I = torch.eye(model.D, device=W.device, dtype=W.dtype)
    return sum(
        torch.norm(
            F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I,
            p="fro")
        for f in range(model.F1)
    ) / model.F1


# ── EEGNet ────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, C, T, n_classes=N_CLASSES,
                 F1=8, D=2, F2=16, K=64, p=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False),
            nn.BatchNorm2d(F1))
        self.depth  = nn.Sequential(
            nn.Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(True),
            nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.sep    = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2,   (1, 1),  bias=False),
            nn.BatchNorm2d(F2), nn.ELU(True),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            flat = self.sep(self.depth(self.block1(torch.zeros(1, 1, C, T)))).numel()
        self.head = nn.Linear(flat, n_classes)

    def forward(self, x):
        return self.head(self.sep(self.depth(self.block1(x.unsqueeze(1)))).flatten(1))


# ── DeepConvNet ───────────────────────────────────────────────────
class DeepConvNet(nn.Module):
    def __init__(self, C, T, n_classes=N_CLASSES, p=0.5):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 25, (1,5), bias=True),
            nn.Conv2d(25, 25, (C,1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(True),
            nn.MaxPool2d((1,3), stride=(1,3)))
        self.b2 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(25, 50, (1,5), bias=False),
            nn.BatchNorm2d(50), nn.ELU(True),
            nn.MaxPool2d((1,3), stride=(1,3)))
        self.b3 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(50, 100, (1,5), bias=False),
            nn.BatchNorm2d(100), nn.ELU(True),
            nn.MaxPool2d((1,3), stride=(1,3)))
        self.b4 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(100, 200, (1,5), bias=False),
            nn.BatchNorm2d(200), nn.ELU(True),
            nn.MaxPool2d((1,3), stride=(1,3)))
        with torch.no_grad():
            d = self.b4(self.b3(self.b2(self.b1(torch.zeros(1, 1, C, T)))))
            fl = d.size(-1)
        self.head = nn.Conv2d(200, n_classes, (1, fl))

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x.unsqueeze(1)))))
        return self.head(x).squeeze(-1).squeeze(-1)


# ── EEG-Conformer ─────────────────────────────────────────────────
class EEGConformer(nn.Module):
    def __init__(self, C, T, n_classes=N_CLASSES, emb=40, depth=6, heads=10, p=0.5):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1,25)), nn.Conv2d(40, 40, (C,1)),
            nn.BatchNorm2d(40), nn.ELU(True),
            nn.AvgPool2d((1,75),(1,15)), nn.Dropout(p))
        self.proj = nn.Conv2d(40, emb, (1,1))
        enc = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=emb*4,
            dropout=p, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        with torch.no_grad():
            d = self.proj(self.patch(torch.zeros(1,1,C,T))).squeeze(2).permute(0,2,1)
            flat = self.transformer(d).reshape(1,-1).size(1)
        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.ELU(True), nn.Dropout(0.5),
            nn.Linear(256, 32),  nn.ELU(True), nn.Dropout(0.3),
            nn.Linear(32, n_classes))

    def forward(self, x):
        x = self.proj(self.patch(x.unsqueeze(1))).squeeze(2).permute(0,2,1)
        return self.head(self.transformer(x).reshape(x.size(0), -1))


# ── LMDA-Net ──────────────────────────────────────────────────────
class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super().__init__()
        self.C = C
        self.pool    = nn.AdaptiveAvgPool2d((1, W))
        self.conv    = nn.Conv2d(1, 1, (k,1), padding=(k//2,0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        y = self.softmax(self.conv(self.pool(x).transpose(-2,-3))).transpose(-2,-3)
        return y * self.C * x


class LMDA(nn.Module):
    def __init__(self, C, T, n_classes=N_CLASSES,
                 depth=9, d1=24, d2=9, K=75, pool=5):
        super().__init__()
        self.cw = nn.Parameter(torch.randn(depth, 1, C))
        nn.init.xavier_uniform_(self.cw.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, d1, (1,1), bias=False), nn.BatchNorm2d(d1),
            nn.Conv2d(d1, d1, (1,K), groups=d1, bias=False),
            nn.BatchNorm2d(d1), nn.GELU())
        self.chan_conv = nn.Sequential(
            nn.Conv2d(d1, d2, (1,1), bias=False), nn.BatchNorm2d(d2),
            nn.Conv2d(d2, d2, (C,1), groups=d2, bias=False),
            nn.BatchNorm2d(d2), nn.GELU())
        self.norm = nn.Sequential(nn.AvgPool3d((1,1,pool)), nn.Dropout(0.65))
        with torch.no_grad():
            d = torch.einsum('bdcw,hdc->bhcw', torch.ones(1,1,C,T), self.cw)
            d = self.time_conv(d); _, Cf, _, W = d.size()
        self.da = EEGDepthAttention(W, Cf, k=7)
        with torch.no_grad():
            flat = self.norm(self.chan_conv(d)).numel()
        self.head = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = torch.einsum('bdcw,hdc->bhcw', x.unsqueeze(1), self.cw)
        x = self.da(self.time_conv(x))
        return self.head(self.norm(self.chan_conv(x)).flatten(1))


# ── Model registry ─────────────────────────────────────────────────
MODELS = {
    "renet":       dict(cls=RENet,        lr=1e-3, wd=0.01, opt="adam",  lam=0.10, noise=0.03),
    "eegnet":      dict(cls=EEGNet,       lr=1e-3, wd=0,    opt="adam",  lam=0,    noise=0),
    "deepconvnet": dict(cls=DeepConvNet,  lr=1e-3, wd=0,    opt="adam",  lam=0,    noise=0),
    "conformer":   dict(cls=EEGConformer, lr=2e-4, wd=0,    opt="adam",  lam=0,    noise=0),
    "lmda":        dict(cls=LMDA,         lr=1e-3, wd=1e-2, opt="adamw", lam=0,    noise=0),
}


# ══════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════

def load_data():
    """
    Load BCI-IV 2a data via MOABB (BNCI2014001).
    Returns X (trials, C, T), y (labels), session (array of 1/2).

    Cache layout:
      cache/bciciv2a_preprocessed_subXX.npz  — per-subject, two sessions
    """
    import mne, moabb
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    mne.set_log_level("CRITICAL")
    moabb.set_log_level("CRITICAL")
    mne.set_config("MNE_DATA", DATA_DIR, set_env=True)
    os.makedirs(CACHE, exist_ok=True)

    par = MotorImagery(
        events=EVENTS, n_classes=N_CLASSES,
        fmin=FMIN, fmax=FMAX,
        tmin=TMIN, tmax=TMAX,
        resample=RESAMPLE)

    all_X, all_y, all_sub, all_ses = [], [], [], []

    for sid in SUBJECTS:
        cache_path = os.path.join(CACHE, f"bciciv2a_sub{sid:02d}.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path, allow_pickle=True)
            X, y, ses = d["X"], d["y"], d["ses"]
        else:
            print(f"  Preprocessing S{sid:02d}...", flush=True)
            X, y, meta = par.get_data(dataset=BNCI2014_001(), subjects=[sid])
            le  = LabelEncoder()
            y   = le.fit_transform(y).astype(np.int64)
            ses = meta["session"].values          # '0train' / '1test' → encode to 1/2
            # MOABB session names vary; map to int 1 / 2
            unique_ses = sorted(set(ses))
            ses_map = {s: i+1 for i, s in enumerate(unique_ses)}
            ses = np.array([ses_map[s] for s in ses], dtype=np.int64)
            np.savez_compressed(cache_path,
                X=X.astype(np.float32), y=y.astype(np.int64), ses=ses)

        all_X.append(X.astype(np.float32))
        all_y.append(y.astype(np.int64))
        all_sub.append(np.full(len(y), sid, dtype=np.int64))
        all_ses.append(ses.astype(np.int64))

    X   = np.concatenate(all_X,   axis=0)
    y   = np.concatenate(all_y,   axis=0)
    sub = np.concatenate(all_sub, axis=0)
    ses = np.concatenate(all_ses, axis=0)

    # Trial-level z-normalisation
    X = ((X - X.mean(-1, keepdims=True)) /
         (X.std(-1, keepdims=True) + 1e-6)).astype(np.float32)

    print(f"BCI-IV 2a loaded: {X.shape[0]} trials | "
          f"{X.shape[1]} ch | {X.shape[2]} tp | {N_CLASSES} classes")
    return X, y, sub, ses


# ══════════════════════════════════════════════════════════════════
#  Training Utilities
# ══════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=30):
        self.patience   = patience
        self.counter    = 0
        self.best       = None
        self.should_stop = False
        self.state      = None

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


def train_model(model, X_tr, y_tr, X_te, y_te,
                lr=1e-3, wd=0.0, opt_name="adam",
                lam_osfr=0.0, noise_std=0.0,
                n_epochs=300, bs=64, patience=30, eval_interval=5):
    """
    Generic training loop.  Returns best validation accuracy.
    """
    OptCls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    opt    = OptCls(model.parameters(), lr=lr, weight_decay=wd)
    ce     = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    es     = EarlyStopping(patience)

    for ep in range(n_epochs):
        model.train()
        for i in torch.randperm(len(X_tr)).split(bs):
            bx = X_tr[i].to(device)
            if noise_std > 0:
                bx = bx + torch.randn_like(bx) * noise_std
            by = y_tr[i].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(bx)
                loss   = ce(logits, by)
                if lam_osfr > 0 and hasattr(model, "spatial"):
                    loss = loss + lam_osfr * osfr_loss(model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        if (ep + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([
                    model(X_te[j:j+bs*4].to(device)).argmax(1).cpu()
                    for j in range(0, len(X_te), bs*4)]).numpy()
            es(accuracy_score(y_te.numpy(), pred), model)
            if es.should_stop:
                break

    es.restore(model)
    return es.best or 0.0


def evaluate(model, X_te, y_te, bs=256):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([
            model(X_te[j:j+bs].to(device)).argmax(1).cpu()
            for j in range(0, len(X_te), bs)]).numpy()
    y_np = y_te.numpy()
    return (accuracy_score(y_np, pred),
            f1_score(y_np, pred, average="macro"),
            cohen_kappa_score(y_np, pred))


# ══════════════════════════════════════════════════════════════════
#  Cross-Session Evaluation Loop
# ══════════════════════════════════════════════════════════════════

def run(model_name: str):
    """
    Within-subject, cross-session evaluation:
      Train on Session 1 → Test on Session 2
    for all 9 subjects independently.
    """
    cfg = MODELS[model_name]
    out = os.path.join(BASE, f"{model_name}_bciciv2a", "results")
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"session_{model_name}.json")

    print("=" * 55)
    print(f"  {model_name.upper()} | BCI-IV 2a | Cross-Session (S1→S2)")
    print("=" * 55)

    X, y, sub, ses = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt   = torch.from_numpy(X).pin_memory() if use_cuda else torch.from_numpy(X)
    yt   = torch.from_numpy(y).pin_memory() if use_cuda else torch.from_numpy(y)

    dummy_model = cfg["cls"](C, T, N_CLASSES)
    n_params = sum(p.numel() for p in dummy_model.parameters())
    print(f"  Parameters: {n_params:,}")
    del dummy_model

    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    todo = [s for s in SUBJECTS if str(s) not in done]

    for sid in todo:
        t0 = time.time()

        # Session masks for this subject
        is_sub = sub == sid
        tr_mask = is_sub & (ses == 1)     # Session 1 → train
        te_mask = is_sub & (ses == 2)     # Session 2 → test

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            print(f"  S{sid:02d} — session data missing, skipping.")
            continue

        model = cfg["cls"](C, T, N_CLASSES).to(device)
        train_model(
            model,
            Xt[tr_mask], yt[tr_mask],
            Xt[te_mask], yt[te_mask],
            lr=cfg["lr"], wd=cfg["wd"], opt_name=cfg["opt"],
            lam_osfr=cfg["lam"], noise_std=cfg["noise"],
            n_epochs=300, patience=30)

        acc, f1, kappa = evaluate(model, Xt[te_mask], yt[te_mask])
        done[str(sid)] = {
            "acc":   round(acc,   4),
            "f1":    round(f1,    4),
            "kappa": round(kappa, 4)}
        json.dump(done, open(res_file, "w"), indent=2)

        dt = time.time() - t0
        eta = dt * (len(todo) - list(todo).index(sid) - 1) / 60
        print(f"  S{sid:02d} | Acc:{acc:.2%}  F1:{f1:.4f}  κ:{kappa:.4f}"
              f" | {len(done)}/{N_SUBJECTS}  {dt:.0f}s  ETA:{eta:.0f}m")

        del model
        if use_cuda: torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────
    accs   = [v["acc"]   for v in done.values()]
    f1s    = [v["f1"]    for v in done.values()]
    kappas = [v["kappa"] for v in done.values()]
    print(f"\n{'─'*55}")
    print(f"  {model_name.upper()} | {len(accs)} subjects")
    print(f"  Acc   : {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    print(f"  F1    : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Kappa : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
    print(f"{'─'*55}\n")


# ══════════════════════════════════════════════════════════════════
#  Statistical Summary across all trained models
# ══════════════════════════════════════════════════════════════════

def summary():
    """Print a comparison table of all models that have results."""
    from scipy.stats import wilcoxon

    model_dirs = {
        "RE-Net":        os.path.join(BASE, "renet_bciciv2a/results/session_renet.json"),
        "EEGNet":        os.path.join(BASE, "eegnet_bciciv2a/results/session_eegnet.json"),
        "DeepConvNet":   os.path.join(BASE, "deepconvnet_bciciv2a/results/session_deepconvnet.json"),
        "EEG-Conformer": os.path.join(BASE, "conformer_bciciv2a/results/session_conformer.json"),
        "LMDA-Net":      os.path.join(BASE, "lmda_bciciv2a/results/session_lmda.json"),
    }

    results = {}
    for name, path in model_dirs.items():
        if os.path.exists(path):
            d = json.load(open(path))
            results[name] = {
                "acc":   [d[str(s)]["acc"]   for s in SUBJECTS if str(s) in d],
                "kappa": [d[str(s)]["kappa"] for s in SUBJECTS if str(s) in d],
            }

    if not results:
        print("No results found. Run experiments first."); return

    print(f"\n{'='*65}")
    print(f"  BCI-IV 2a Results — Cross-Session (9 subjects)")
    print(f"{'='*65}")
    header = f"  {'Model':18s}  {'Acc (%)':>10s}  {'±':>6s}  {'Kappa':>8s}  {'±':>6s}"
    print(header)
    print(f"  {'-'*58}")

    re_accs = results.get("RE-Net", {}).get("acc", [])
    for name, r in results.items():
        a = np.array(r["acc"])
        k = np.array(r["kappa"])
        marker = " ◄" if name == "RE-Net" else ""
        print(f"  {name:18s}  {np.mean(a)*100:8.2f}%  "
              f"±{np.std(a)*100:4.2f}  "
              f"{np.mean(k):8.4f}  "
              f"±{np.std(k):4.4f}{marker}")

    if re_accs and len(results) > 1:
        print(f"\n  Wilcoxon signed-rank (one-sided, H1: RE-Net > Baseline):")
        print(f"  {'Comparison':30s}  {'p-value':>12s}  {'Sig':>4s}  {'ΔAcc':>8s}")
        print(f"  {'-'*58}")
        for name, r in results.items():
            if name == "RE-Net": continue
            bl = np.array(r["acc"])
            re = np.array(re_accs[:len(bl)])
            if len(re) != len(bl) or len(re) < 2: continue
            try:
                _, p = wilcoxon(re, bl, alternative="greater")
            except ValueError:
                p = 1.0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            diff = (np.mean(re) - np.mean(bl)) * 100
            print(f"  {'RE-Net vs '+name:30s}  {p:12.3e}  {sig:>4s}  {diff:>+6.2f}%")
    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if cmd == "all":
        for name in MODELS:
            run(name)
        summary()
    elif cmd == "summary":
        summary()
    elif cmd in MODELS:
        run(cmd)
        summary()
    else:
        print(__doc__)
        print(f"Available models: {list(MODELS.keys())} | all | summary")
