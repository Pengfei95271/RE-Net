"""Lambda sweep with PURE-OSFR model (no DSA). Env: DATASET=physionet N_SEEDS=3"""
import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import run_analysis as A
from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    to_compute_tensors, batch_index, stratified_val_split)
warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "lambda_sweep_nodsa"); os.makedirs(OUT, exist_ok=True)
LAMBDAS = [0.0, 0.01, 0.05, 0.10, 0.50, 1.0]
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
F1, D = 8, 2

def weight_cos(model):
    W = model.spatial.weight.view(F1, D, -1)
    vals = []
    for f in range(F1):
        Wn = F.normalize(W[f], p=2, dim=-1)
        G = (Wn @ Wn.t()).abs()
        vals.append(float((G.sum() - G.diag().sum()) / (D * (D - 1) + 1e-9)))
    return float(np.mean(vals))

def train_full(Xt, yt, tr_idx, C, T, ncl, on_gpu, lam, seed):
    set_seed(seed)
    model = A._build_renet(C, T, use_dsa=False, n_classes=ncl).to(device)
    ti, vi = stratified_val_split(yt[tr_idx].cpu().numpy(), 0.15, seed)
    tr, val = tr_idx[ti], tr_idx[vi]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    es = A.EarlyStopping(20); bs = 64
    val_t = torch.as_tensor(val)
    for ep in range(200):
        model.train()
        for i in torch.randperm(len(tr)).split(bs):
            idx = torch.as_tensor(tr[i.numpy()])
            bx = batch_index(Xt, idx, on_gpu)
            bx = bx + torch.randn_like(bx) * 0.03
            by = batch_index(yt, idx, on_gpu)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = ce(model(bx), by)
                if lam > 0:
                    loss = loss + lam * A.osfr_loss(model)
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
            es(accuracy_score(yt[val].cpu().numpy(), pred), model)
            if es.should_stop:
                break
    es.restore(model); model.eval()
    return model, (es.best or 0.0)

def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    tr_idx = np.arange(len(y))
    print(f"Lambda sweep (NO-DSA / pure OSFR) | {N_SEEDS} seeds\n")
    print(f"{'lambda':>8} {'val_acc':>10} {'val_sd':>8} {'weight_cos':>12}")
    rows = []
    for lam in LAMBDAS:
        accs, coss = [], []
        for k in range(N_SEEDS):
            model, best = train_full(Xt, yt, tr_idx, C, T, ncl, on_gpu, lam, SEED + k)
            accs.append(float(best)); coss.append(weight_cos(model))
            del model
            if use_cuda: torch.cuda.empty_cache()
        rows.append(dict(lam=float(lam),
                         val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)),
                         weight_cos_mean=float(np.mean(coss)), weight_cos_std=float(np.std(coss)),
                         seeds=N_SEEDS))
        print(f"{lam:>8.2f} {np.mean(accs)*100:>9.2f}% {np.std(accs)*100:>7.2f}% {np.mean(coss):>12.4f}")
    json.dump(rows, open(os.path.join(OUT, "lambda_sweep.json"), "w"), indent=2)
    print(f"\nSaved {OUT}/lambda_sweep.json")

if __name__ == "__main__":
    main()
