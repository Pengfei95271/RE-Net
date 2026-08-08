"""RE-Net (DSA + OSFR) LOSO. Usage: python run_renet.py  Env: DATASET, SEED"""
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import (BASE, device, use_cuda, SEED, set_seed,
                    load_data, subjects_of, stratified_val_split,
                    to_compute_tensors, batch_index, result_dir)

warnings.filterwarnings("ignore")

CFG = dict(F1=8, D=2, F2=16, kernel_length=64, dropout=0.25,
           lr=1e-3, weight_decay=0.01, batch_size=64,
           n_epochs=200, patience=20, eval_interval=5, grad_clip=1.0,
           lambda_osfr=0.10, noise_std=0.03)


class DualStateActivation(nn.Module):
    def __init__(self, pool_kernel):
        super().__init__(); self.pool = nn.AvgPool2d(pool_kernel)
    def forward(self, x):
        return self.pool(F.elu(x)) + torch.log1p(self.pool(x ** 2))


class RENet(nn.Module):
    def __init__(self, C, T, n_classes=2):
        super().__init__()
        F1, D, F2 = CFG["F1"], CFG["D"], CFG["F2"]
        K, p = CFG["kernel_length"], CFG["dropout"]
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding=(0, K // 2), bias=False), nn.BatchNorm2d(F1))
        self.spatial = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
        self.bn1  = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.Sequential(DualStateActivation((1, 4)), nn.Dropout(p))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2))
        self.act2 = nn.Sequential(DualStateActivation((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            flat = self.act2(self.block2(self.act1(self.bn1(self.spatial(self.block1(
                torch.zeros(1, 1, C, T))))))).numel()
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
    W = model.spatial.weight.view(CFG["F1"], CFG["D"], -1)
    I = torch.eye(CFG["D"], device=W.device, dtype=W.dtype)
    return sum(
        torch.norm(F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I, p="fro")
        for f in range(CFG["F1"])) / CFG["F1"]


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


def train(model, Xt, yt, tr_idx, seed, on_gpu):
    ti, vi = stratified_val_split(yt[tr_idx].cpu().numpy(), 0.15, seed)
    tr, val = tr_idx[ti], tr_idx[vi]
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    ce  = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    es = EarlyStopping(CFG["patience"]); bs = CFG["batch_size"]
    val_t = torch.as_tensor(val)
    for ep in range(CFG["n_epochs"]):
        model.train()
        for i in torch.randperm(len(tr)).split(bs):
            idx = torch.as_tensor(tr[i.numpy()])
            bx = batch_index(Xt, idx, on_gpu)
            bx = bx + torch.randn_like(bx) * CFG["noise_std"]
            by = batch_index(yt, idx, on_gpu)
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
                pred = torch.cat([
                    model(batch_index(Xt, val_t[j:j+256], on_gpu)).argmax(1).cpu()
                    for j in range(0, len(val), 256)]).numpy()
            es(accuracy_score(yt[val].cpu().numpy(), pred), model)      # validation
            if es.should_stop: break
    es.restore(model)
    return es.best or 0.0


def run():
    out, tag = result_dir("renet")
    print("=" * 56)
    print(f"RE-Net LOSO | DSA + OSFR (lambda={CFG['lambda_osfr']}) | SEED={SEED} | out={tag}_run")
    print("=" * 56)
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    subjects = subjects_of(s)
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    set_seed(SEED)
    print(f"Params: {sum(p.numel() for p in RENet(C, T, n_classes).parameters()):,}")
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, "loso_renet.json")
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for sub in [sb for sb in subjects if str(sb) not in done]:
        set_seed(SEED + sub)
        t0 = time.time()
        tr_idx = np.where(s != sub)[0]
        te_idx = torch.as_tensor(np.where(s == sub)[0])
        model = RENet(C, T, n_classes).to(device)
        train(model, Xt, yt, tr_idx, SEED + sub, on_gpu)
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
    run()
