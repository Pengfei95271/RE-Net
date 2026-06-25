import os, json, time, sys, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Paths & Device ─────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
CACHE   = os.path.join(BASE, "cache")
DATA_DIR = os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "Datasets"))
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cudnn.benchmark = True

SUBJECTS = list(range(1, 110))


# ══════════════════════════════════════════════════════════════════
#  Models
# ══════════════════════════════════════════════════════════════════

# ── EEGNet (Lawhern et al., J. Neural Eng., 2018) ─────────────────
class EEGNet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        F1, D, F2, K, p = 8, 2, 16, 64, 0.25
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False), nn.BatchNorm2d(F1))
        self.depth = nn.Sequential(
            nn.Conv2d(F1, F1*D, (C,1), groups=F1, bias=False), nn.BatchNorm2d(F1*D),
            nn.ELU(True), nn.AvgPool2d((1,4)), nn.Dropout(p))
        self.sep = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1,16), padding=(0,8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1,1), bias=False), nn.BatchNorm2d(F2),
            nn.ELU(True), nn.AvgPool2d((1,8)), nn.Dropout(p))
        with torch.no_grad():
            flat = self.sep(self.depth(self.block1(torch.zeros(1,1,C,T)))).numel()
        self.head = nn.Linear(flat, n_classes)

    def forward(self, x):
        return self.head(self.sep(self.depth(self.block1(x.unsqueeze(1)))).flatten(1))


# ── DeepConvNet (Schirrmeister et al., HBM, 2017) ─────────────────
class DeepConvNet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        p = 0.5
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1,5), bias=True),
            nn.Conv2d(25, 25, (C,1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(True), nn.MaxPool2d((1,3), stride=(1,3)))
        self.block2 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(25, 50, (1,5), bias=False),
            nn.BatchNorm2d(50), nn.ELU(True), nn.MaxPool2d((1,3), stride=(1,3)))
        self.block3 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(50, 100, (1,5), bias=False),
            nn.BatchNorm2d(100), nn.ELU(True), nn.MaxPool2d((1,3), stride=(1,3)))
        self.block4 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(100, 200, (1,5), bias=False),
            nn.BatchNorm2d(200), nn.ELU(True), nn.MaxPool2d((1,3), stride=(1,3)))
        with torch.no_grad():
            d = self.block4(self.block3(self.block2(self.block1(torch.zeros(1,1,C,T)))))
            fl = d.size(-1)
        self.head = nn.Conv2d(200, n_classes, (1, fl))

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x.unsqueeze(1)))))
        return self.head(x).squeeze(-1).squeeze(-1)


# ── EEG-Conformer (Song et al., IEEE TNSRE, 2022) ─────────────────
class EEGConformer(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        emb, depth, heads, p = 40, 6, 10, 0.5

        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1,25)), nn.Conv2d(40, 40, (C,1)),
            nn.BatchNorm2d(40), nn.ELU(True),
            nn.AvgPool2d((1,75), (1,15)), nn.Dropout(p))
        self.proj = nn.Conv2d(40, emb, (1,1))

        enc = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=emb*4,
            dropout=p, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)

        with torch.no_grad():
            d = self.proj(self.patch(torch.zeros(1,1,C,T))).squeeze(2).permute(0,2,1)
            d = self.transformer(d)
            flat = d.reshape(1,-1).size(1)

        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.ELU(True), nn.Dropout(0.5),
            nn.Linear(256, 32), nn.ELU(True), nn.Dropout(0.3),
            nn.Linear(32, n_classes))

    def forward(self, x):
        x = self.proj(self.patch(x.unsqueeze(1))).squeeze(2).permute(0,2,1)
        return self.head(self.transformer(x).reshape(x.size(0), -1))


# ── LMDA-Net (Miao et al., NeuroImage, 2023) ──────────────────────
class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super().__init__()
        self.C = C
        self.pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, (k,1), padding=(k//2,0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        y = self.softmax(self.conv(self.pool(x).transpose(-2,-3))).transpose(-2,-3)
        return y * self.C * x


class LMDA(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        depth, d1, d2, K, pool = 9, 24, 9, 75, 5

        self.cw = nn.Parameter(torch.randn(depth, 1, C))
        nn.init.xavier_uniform_(self.cw.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, d1, (1,1), bias=False), nn.BatchNorm2d(d1),
            nn.Conv2d(d1, d1, (1,K), groups=d1, bias=False), nn.BatchNorm2d(d1), nn.GELU())
        self.chan_conv = nn.Sequential(
            nn.Conv2d(d1, d2, (1,1), bias=False), nn.BatchNorm2d(d2),
            nn.Conv2d(d2, d2, (C,1), groups=d2, bias=False), nn.BatchNorm2d(d2), nn.GELU())
        self.norm = nn.Sequential(nn.AvgPool3d((1,1,pool)), nn.Dropout(0.65))

        with torch.no_grad():
            d = torch.einsum('bdcw,hdc->bhcw', torch.ones(1,1,C,T), self.cw)
            d = self.time_conv(d)
            _, Cf, _, W = d.size()
        self.da = EEGDepthAttention(W, Cf, k=7)
        with torch.no_grad():
            flat = self.norm(self.chan_conv(d)).numel()
        self.head = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = torch.einsum('bdcw,hdc->bhcw', x.unsqueeze(1), self.cw)
        x = self.da(self.time_conv(x))
        return self.head(self.norm(self.chan_conv(x)).flatten(1))


# ══════════════════════════════════════════════════════════════════
#  Model Registry
# ══════════════════════════════════════════════════════════════════

MODELS = {
    "eegnet":      dict(cls=EEGNet,       lr=1e-3, wd=0,    opt="adam"),
    "deepconvnet": dict(cls=DeepConvNet,   lr=1e-3, wd=0,    opt="adam"),
    "conformer":   dict(cls=EEGConformer,  lr=2e-4, wd=0,    opt="adam"),
    "lmda":        dict(cls=LMDA,          lr=1e-3, wd=1e-2, opt="adamw"),
}


# ══════════════════════════════════════════════════════════════════
#  Shared Data & Training
# ══════════════════════════════════════════════════════════════════

def load_data():
    cache = os.path.join(CACHE, "physionetmi_casa_preprocessed.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y, s = d["X"], d["y"], d["s"]
    else:
        import mne, moabb
        from moabb.datasets import PhysionetMI
        from moabb.paradigms import MotorImagery
        mne.set_log_level("CRITICAL"); moabb.set_log_level("CRITICAL")
        mne.set_config("MNE_DATA", DATA_DIR, set_env=True)
        par = MotorImagery(events=["left_hand","right_hand"], n_classes=2,
                           fmin=4, fmax=40, tmin=0.5, tmax=3.5, resample=128)
        X, y, meta = par.get_data(dataset=PhysionetMI(), subjects=SUBJECTS)
        s = meta["subject"].values.astype(int)
        y = LabelEncoder().fit_transform(y)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(cache, X=X.astype(np.float32), y=y.astype(np.int64), s=s.astype(np.int64))
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


def train(model, X_tr, y_tr, X_te, y_te, lr, wd, opt_name):
    OptCls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    opt = OptCls(model.parameters(), lr=lr, weight_decay=wd)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    es, bs = EarlyStopping(20), 64

    for ep in range(200):
        model.train()
        for i in torch.randperm(len(X_tr)).split(bs):
            bx, by = X_tr[i].to(device), y_tr[i].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = ce(model(bx), by)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(X_te[j:j+256].to(device)).argmax(1).cpu()
                                  for j in range(0, len(X_te), 256)]).numpy()
            es(accuracy_score(y_te.numpy(), pred), model)
            if es.should_stop: break

    es.restore(model)
    return es.best or 0.0


# ══════════════════════════════════════════════════════════════════
#  LOSO Loop
# ══════════════════════════════════════════════════════════════════

def run(model_name):
    cfg = MODELS[model_name]
    out = os.path.join(BASE, f"{model_name}_run", "results")
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_{model_name}.json")

    print("=" * 50)
    print(f"{model_name.upper()} LOSO")
    print("=" * 50)

    X, y, s = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt = torch.from_numpy(X).pin_memory() if use_cuda else torch.from_numpy(X)
    yt = torch.from_numpy(y).pin_memory() if use_cuda else torch.from_numpy(y)

    print(f"Params: {sum(p.numel() for p in cfg['cls'](C,T).parameters()):,}")

    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    todo = [sub for sub in SUBJECTS if str(sub) not in done]

    for sub in todo:
        t0 = time.time()
        tr, te = s != sub, s == sub
        model = cfg["cls"](C, T).to(device)
        train(model, Xt[tr], yt[tr], Xt[te], yt[te], cfg["lr"], cfg["wd"], cfg["opt"])

        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xt[te][j:j+256].to(device)).argmax(1).cpu()
                              for j in range(0, te.sum(), 256)]).numpy()
        acc = accuracy_score(y[te], pred)
        f1  = f1_score(y[te], pred, average="macro")

        done[str(sub)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        json.dump(done, open(res_file, "w"), indent=2)

        dt = time.time() - t0
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | "
              f"{len(done)}/{len(SUBJECTS)} {dt:.0f}s ETA:{dt*(len(SUBJECTS)-len(done))/60:.0f}m")
        del model; torch.cuda.empty_cache() if use_cuda else None

    accs = [v["acc"] for v in done.values()]
    print(f"\n{len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "eegnet"
    assert name in MODELS, f"Choose from: {list(MODELS.keys())}"
    run(name)
