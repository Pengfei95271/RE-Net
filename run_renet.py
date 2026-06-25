import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Paths (auto-detect from script location) ──────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
PROJECT   = os.path.join(BASE, "renet_run")
CACHE     = os.path.join(BASE, "cache")
RESULTS   = os.path.join(PROJECT, "results")
DATA_DIR  = os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "Datasets"))
os.makedirs(RESULTS, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cudnn.benchmark = True

# ── Config ─────────────────────────────────────────────────────────
CFG = dict(
    F1=8, D=2, F2=16, kernel_length=64, dropout=0.25,
    lr=1e-3, weight_decay=0.01, batch_size=64,
    n_epochs=200, patience=20, eval_interval=5, grad_clip=1.0,
    lambda_osfr=0.10, noise_std=0.03,
    subjects=list(range(1, 110)),
    fmin=4, fmax=40, tmin=0.5, tmax=3.5, resample=128,
)

# ══════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════

class DualStateActivation(nn.Module):
    """DSA: phase (ELU) + power (x² → log)."""
    def __init__(self, pool_kernel):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_kernel)

    def forward(self, x):
        return self.pool(F.elu(x)) + torch.log1p(self.pool(x ** 2))


class RENet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        F1, D, F2 = CFG["F1"], CFG["D"], CFG["F2"]
        K, p = CFG["kernel_length"], CFG["dropout"]

        # Block 1: temporal → spatial → DSA
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
        self.bn1     = nn.BatchNorm2d(F1 * D)
        self.act1    = nn.Sequential(DualStateActivation((1, 4)), nn.Dropout(p))

        # Block 2: separable conv → DSA
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
        )
        self.act2 = nn.Sequential(DualStateActivation((1, 8)), nn.Dropout(p))

        # Classifier (auto-compute flat dim)
        with torch.no_grad():
            d = self.act2(self.block2(self.act1(self.bn1(self.spatial(self.block1(
                torch.zeros(1, 1, C, T)))))))
            flat = d.numel()
        self.head = nn.Linear(flat, n_classes)
        self._init()

    def _init(self):
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
    W = model.spatial.weight.view(CFG["F1"], CFG["D"], -1)   # (F1, D, C)
    I = torch.eye(CFG["D"], device=W.device, dtype=W.dtype)
    return sum(
        torch.norm(F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I, p="fro")
        for f in range(CFG["F1"])
    ) / CFG["F1"]


# ══════════════════════════════════════════════════════════════════
#  Data & Training Utilities
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
                           fmin=CFG["fmin"], fmax=CFG["fmax"],
                           tmin=CFG["tmin"], tmax=CFG["tmax"], resample=CFG["resample"])
        X, y, meta = par.get_data(dataset=PhysionetMI(), subjects=CFG["subjects"])
        s = meta["subject"].values.astype(int)
        y = LabelEncoder().fit_transform(y)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(cache, X=X.astype(np.float32), y=y.astype(np.int64), s=s.astype(np.int64))
    # trial-wise z-norm
    X = ((X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
    print(f"Data: {X.shape[0]} trials, {X.shape[1]}ch, {X.shape[2]}tp")
    return X, y.astype(np.int64), s.astype(np.int64)


class EarlyStopping:
    def __init__(self, patience):
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


def train(model, X_tr, y_tr, X_te, y_te):
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    ce  = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    es  = EarlyStopping(CFG["patience"])
    bs  = CFG["batch_size"]

    for ep in range(CFG["n_epochs"]):
        model.train()
        for i in (idx := torch.randperm(len(X_tr))).split(bs):
            bx = X_tr[i].to(device) + torch.randn(len(i), *X_tr.shape[1:], device=device) * CFG["noise_std"]
            by = y_tr[i].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = ce(model(bx), by) + CFG["lambda_osfr"] * osfr_loss(model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update()

        if (ep + 1) % CFG["eval_interval"] == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(X_te[j:j+bs*4].to(device)).argmax(1).cpu()
                                  for j in range(0, len(X_te), bs*4)]).numpy()
            es(accuracy_score(y_te.numpy(), pred), model)
            if es.should_stop: break

    es.restore(model)
    return es.best or 0.0


# ══════════════════════════════════════════════════════════════════
#  LOSO Loop
# ══════════════════════════════════════════════════════════════════

def run():
    print("=" * 50)
    print(f"RE-Net LOSO | DSA + OSFR (lambda={CFG['lambda_osfr']})")
    print("=" * 50)

    X, y, s = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt = torch.from_numpy(X).pin_memory() if use_cuda else torch.from_numpy(X)
    yt = torch.from_numpy(y).pin_memory() if use_cuda else torch.from_numpy(y)

    print(f"Params: {sum(p.numel() for p in RENet(C,T).parameters()):,}")

    res_file = os.path.join(RESULTS, "loso_renet.json")
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    todo = [sub for sub in CFG["subjects"] if str(sub) not in done]


    for sub in todo:
        t0 = time.time()
        tr, te = s != sub, s == sub
        model = RENet(C, T).to(device)
        best = train(model, Xt[tr], yt[tr], Xt[te], yt[te])

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
              f"{len(done)}/{len(CFG['subjects'])} {dt:.0f}s ETA:{dt*(len(CFG['subjects'])-len(done))/60:.0f}m")
        del model; torch.cuda.empty_cache() if use_cuda else None

    accs = [v["acc"] for v in done.values()]
    print(f"\n{len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


if __name__ == "__main__":
    run()
