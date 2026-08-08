"""
Baselines under the SAME LOSO protocol as RE-Net.
  python run_baselines.py <model>
  model in: eegnet deepconvnet conformer lmda shallow fbcnet eegtcnet atcnet
Env: DATASET=physionet|bci2a|bci2b   SEED=2024
"""
import os, sys, json, time, warnings
import numpy as np
from scipy.signal import firwin
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import (BASE, device, use_cuda, SEED, set_seed,
                    load_data, subjects_of, stratified_val_split,
                    to_compute_tensors, batch_index, result_dir,
                    MATCHED_PROTOCOL, MATCHED_NOISE_STD, MATCHED_WEIGHT_DECAY)

warnings.filterwarnings("ignore")


# ── EEGNet ────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        F1, D, F2, K, p = 8, 2, 16, 64, 0.25
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False), nn.BatchNorm2d(F1))
        self.depth = nn.Sequential(
            nn.Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1*D),
            nn.ELU(True), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.sep = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2),
            nn.ELU(True), nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            flat = self.sep(self.depth(self.block1(torch.zeros(1, 1, C, T)))).numel()
        self.head = nn.Linear(flat, n_classes)
    def forward(self, x):
        return self.head(self.sep(self.depth(self.block1(x.unsqueeze(1)))).flatten(1))


# ── DeepConvNet ───────────────────────────────────────────────────
class DeepConvNet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__(); p = 0.5
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), bias=True), nn.Conv2d(25, 25, (C, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(True), nn.MaxPool2d((1, 3), stride=(1, 3)))
        self.block2 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(25, 50, (1, 5), bias=False),
            nn.BatchNorm2d(50), nn.ELU(True), nn.MaxPool2d((1, 3), stride=(1, 3)))
        self.block3 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(50, 100, (1, 5), bias=False),
            nn.BatchNorm2d(100), nn.ELU(True), nn.MaxPool2d((1, 3), stride=(1, 3)))
        self.block4 = nn.Sequential(
            nn.Dropout(p), nn.Conv2d(100, 200, (1, 5), bias=False),
            nn.BatchNorm2d(200), nn.ELU(True), nn.MaxPool2d((1, 3), stride=(1, 3)))
        with torch.no_grad():
            fl = self.block4(self.block3(self.block2(self.block1(torch.zeros(1,1,C,T))))).size(-1)
        self.head = nn.Conv2d(200, n_classes, (1, fl))
    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x.unsqueeze(1)))))
        return self.head(x).squeeze(-1).squeeze(-1)


# ── EEG-Conformer ─────────────────────────────────────────────────
class EEGConformer(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__(); emb, depth, heads, p = 40, 6, 10, 0.5
        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)), nn.Conv2d(40, 40, (C, 1)),
            nn.BatchNorm2d(40), nn.ELU(True), nn.AvgPool2d((1, 75), (1, 15)), nn.Dropout(p))
        self.proj = nn.Conv2d(40, emb, (1, 1))
        enc = nn.TransformerEncoderLayer(d_model=emb, nhead=heads, dim_feedforward=emb*4,
            dropout=p, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        with torch.no_grad():
            d = self.proj(self.patch(torch.zeros(1,1,C,T))).squeeze(2).permute(0,2,1)
            flat = self.transformer(d).reshape(1, -1).size(1)
        self.head = nn.Sequential(nn.Linear(flat, 256), nn.ELU(True), nn.Dropout(0.5),
            nn.Linear(256, 32), nn.ELU(True), nn.Dropout(0.3), nn.Linear(32, n_classes))
    def forward(self, x):
        x = self.proj(self.patch(x.unsqueeze(1))).squeeze(2).permute(0, 2, 1)
        return self.head(self.transformer(x).reshape(x.size(0), -1))


# ── LMDA-Net ──────────────────────────────────────────────────────
class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super().__init__(); self.C = C
        self.pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, (k, 1), padding=(k//2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)
    def forward(self, x):
        y = self.softmax(self.conv(self.pool(x).transpose(-2, -3))).transpose(-2, -3)
        return y * self.C * x

class LMDA(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__(); depth, d1, d2, K, pool = 9, 24, 9, 75, 5
        self.cw = nn.Parameter(torch.randn(depth, 1, C)); nn.init.xavier_uniform_(self.cw.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, d1, (1, 1), bias=False), nn.BatchNorm2d(d1),
            nn.Conv2d(d1, d1, (1, K), groups=d1, bias=False), nn.BatchNorm2d(d1), nn.GELU())
        self.chan_conv = nn.Sequential(
            nn.Conv2d(d1, d2, (1, 1), bias=False), nn.BatchNorm2d(d2),
            nn.Conv2d(d2, d2, (C, 1), groups=d2, bias=False), nn.BatchNorm2d(d2), nn.GELU())
        self.norm = nn.Sequential(nn.AvgPool3d((1, 1, pool)), nn.Dropout(0.65))
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


# ── ShallowConvNet ────────────────────────────────────────────────
class ShallowConvNet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.spatial  = nn.Conv2d(40, 40, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(40); self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(0.5)
        with torch.no_grad():
            flat = self._feat(torch.zeros(1, 1, C, T)).numel()
        self.head = nn.Linear(flat, n_classes)
    def _feat(self, x):
        x = self.bn(self.spatial(self.temporal(x)))
        x = torch.clamp(x, -1e6, 1e6) ** 2
        x = self.pool(x)
        return self.drop(torch.log(torch.clamp(x, 1e-6, 1e6)))
    def forward(self, x):
        return self.head(self._feat(x.unsqueeze(1)).flatten(1))


# ── FBCNet ─ fixed FIR filter bank + log-variance ─────────────────
class _FilterBank(nn.Module):
    def __init__(self, fs=128, bands=None, taps=65):
        super().__init__()
        if bands is None:
            bands = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32),(32,36),(36,40)]
        ker = [firwin(taps, [lo, hi], pass_zero=False, fs=fs).astype(np.float32) for lo, hi in bands]
        self.register_buffer("kernel", torch.tensor(np.stack(ker))[:, None, :])
        self.taps, self.nBands = taps, len(bands)
    def forward(self, x):
        B, C, T = x.shape
        y = F.conv1d(x.reshape(B*C, 1, T), self.kernel, padding=self.taps//2)
        return y[..., :T].reshape(B, C, self.nBands, T).permute(0, 2, 1, 3)

class _LogVarLayer(nn.Module):
    def __init__(self, nWin=4): super().__init__(); self.nWin = nWin
    def forward(self, x):
        B, Fc, _, T = x.shape; w = T // self.nWin
        x = x[..., :w*self.nWin].reshape(B, Fc, 1, self.nWin, w)
        return torch.log(torch.clamp(x.var(dim=-1), 1e-6, 1e6))

class FBCNet(nn.Module):
    def __init__(self, C, T, n_classes=2, m=32, nWin=4, fs=128):
        super().__init__()
        self.fb = _FilterBank(fs=fs); nB = self.fb.nBands
        self.scb = nn.Sequential(nn.Conv2d(nB, nB*m, (C, 1), groups=nB, bias=False),
                                 nn.BatchNorm2d(nB*m), nn.SiLU())
        self.var = _LogVarLayer(nWin); self.head = nn.Linear(nB*m*nWin, n_classes)
    def forward(self, x):
        return self.head(self.var(self.scb(self.fb(x))).flatten(1))


# ── EEG-TCNet ─────────────────────────────────────────────────────
class _TCNBlock(nn.Module):
    def __init__(self, ch, k=4, d=1, p=0.3):
        super().__init__(); pad = (k - 1) * d
        self.c1, self.b1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=d), nn.BatchNorm1d(ch)
        self.c2, self.b2 = nn.Conv1d(ch, ch, k, padding=pad, dilation=d), nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        n = x.size(-1)
        y = self.drop(F.elu(self.b1(self.c1(x)[..., :n])))
        y = self.drop(F.elu(self.b2(self.c2(y)[..., :n])))
        return F.elu(x + y)

class EEGTCNet(nn.Module):
    def __init__(self, C, T, n_classes=2, F1=8, D=2, K=64, tcn_ch=12):
        super().__init__(); F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False), nn.BatchNorm2d(F1))
        self.depth = nn.Sequential(
            nn.Conv2d(F1, F2, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(True), nn.AvgPool2d((1, 8)), nn.Dropout(0.3))
        self.proj = nn.Conv1d(F2, tcn_ch, 1)
        self.tcn = nn.Sequential(_TCNBlock(tcn_ch, d=1), _TCNBlock(tcn_ch, d=2))
        self.head = nn.Linear(tcn_ch, n_classes)
    def forward(self, x):
        x = self.depth(self.block1(x.unsqueeze(1))).squeeze(2)
        return self.head(self.tcn(self.proj(x))[..., -1])


# ── ATCNet (compact) ──────────────────────────────────────────────
class ATCNet(nn.Module):
    def __init__(self, C, T, n_classes=2, F1=16, D=2, K=64, n_win=5, heads=2):
        super().__init__(); F2 = F1 * D
        self.conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(True), nn.AvgPool2d((1, 8)), nn.Dropout(0.3),
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False), nn.BatchNorm2d(F2),
            nn.ELU(True), nn.AvgPool2d((1, 4)), nn.Dropout(0.3))
        self.n_win, self.F2 = n_win, F2
        self.attn = nn.MultiheadAttention(F2, heads, dropout=0.3, batch_first=True)
        self.tcn  = _TCNBlock(F2, d=1)
        self.fc   = nn.ModuleList([nn.Linear(F2, n_classes) for _ in range(n_win)])
    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).squeeze(2).permute(0, 2, 1)
        Tc = x.size(1); wl = Tc - self.n_win + 1; outs = []
        for i in range(self.n_win):
            w = x[:, i:i+wl, :]
            a, _ = self.attn(w, w, w); w = w + a
            t = self.tcn(w.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(self.fc[i](t[:, -1, :]))
        return torch.stack(outs, 0).mean(0)


MODELS = {
    "eegnet":      dict(cls=EEGNet,        lr=1e-3, wd=0,    opt="adam"),
    "deepconvnet": dict(cls=DeepConvNet,   lr=1e-3, wd=0,    opt="adam"),
    "conformer":   dict(cls=EEGConformer,  lr=2e-4, wd=0,    opt="adam"),
    "lmda":        dict(cls=LMDA,          lr=1e-3, wd=1e-2, opt="adamw"),
    "shallow":     dict(cls=ShallowConvNet, lr=1e-3, wd=0,   opt="adam"),
    "fbcnet":      dict(cls=FBCNet,        lr=1e-3, wd=0,    opt="adam"),
    "eegtcnet":    dict(cls=EEGTCNet,      lr=1e-3, wd=0,    opt="adam"),
    "atcnet":      dict(cls=ATCNet,        lr=1e-3, wd=1e-2, opt="adamw"),
}


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


def train(model, Xt, yt, tr_idx, lr, wd, opt_name, seed, on_gpu):
    """tr_idx: numpy indices of the training subjects. A stratified validation
    split is carved out of them; the test subject is never seen here."""
    y_tr_all = yt[tr_idx].cpu().numpy()
    ti, vi = stratified_val_split(y_tr_all, 0.15, seed)
    tr, val = tr_idx[ti], tr_idx[vi]
    if MATCHED_PROTOCOL:                       # reviewer Major #6
        wd = MATCHED_WEIGHT_DECAY
        opt_name = "adam"
    OptCls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    opt = OptCls(model.parameters(), lr=lr, weight_decay=wd)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    es, bs = EarlyStopping(20), 64
    val_t = torch.as_tensor(val)
    for ep in range(200):
        model.train()
        for i in torch.randperm(len(tr)).split(bs):
            idx = torch.as_tensor(tr[i.numpy()])
            bx = batch_index(Xt, idx, on_gpu)
            by = batch_index(yt, idx, on_gpu)
            if MATCHED_PROTOCOL:               # RE-Net's noise augmentation
                bx = bx + torch.randn_like(bx) * MATCHED_NOISE_STD
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
                pred = torch.cat([
                    model(batch_index(Xt, val_t[j:j+256], on_gpu)).argmax(1).cpu()
                    for j in range(0, len(val), 256)]).numpy()
            es(accuracy_score(yt[val].cpu().numpy(), pred), model)   # validation
            if es.should_stop: break
    es.restore(model)
    return es.best or 0.0


def run(model_name):
    cfg = MODELS[model_name]
    out, tag = result_dir(model_name)
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_{model_name}.json")
    print("=" * 56)
    print(f"{model_name.upper()} LOSO | SEED={SEED} | "
          f"protocol={'MATCHED' if MATCHED_PROTOCOL else 'original'} | out={tag}_run")
    print("=" * 56)
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]; subjects = subjects_of(s)
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    set_seed(SEED)
    print(f"Params: {sum(p.numel() for p in cfg['cls'](C, T, n_classes).parameters()):,}")
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for sub in [sb for sb in subjects if str(sb) not in done]:
        set_seed(SEED + sub)
        t0 = time.time()
        tr_idx = np.where(s != sub)[0]
        te_idx = torch.as_tensor(np.where(s == sub)[0])
        model = cfg["cls"](C, T, n_classes).to(device)
        train(model, Xt, yt, tr_idx, cfg["lr"], cfg["wd"], cfg["opt"], SEED + sub, on_gpu)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([
                model(batch_index(Xt, te_idx[j:j+256], on_gpu)).argmax(1).cpu()
                for j in range(0, len(te_idx), 256)]).numpy()
        yte = y[s == sub]
        acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")
        done[str(sub)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        json.dump(done, open(res_file, "w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | {len(done)}/{len(subjects)} {time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache() if use_cuda else None
    accs = [v["acc"] for v in done.values()]
    print(f"\n{len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "eegnet"
    assert name in MODELS, f"Choose from: {list(MODELS.keys())}"
    run(name)
