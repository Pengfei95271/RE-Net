"""Full-cohort weight/activation redundancy, PURE-OSFR (no DSA). Env: START STOP SEED"""
import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import run_analysis as A
from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, to_compute_tensors, batch_index,
                    stratified_val_split)
warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "fullcohort_nodsa"); os.makedirs(OUT, exist_ok=True)
JSON = os.path.join(OUT, "fullcohort.json")
START = int(os.environ.get("START", "0"))
STOP = int(os.environ.get("STOP", "1000"))
F1, D = 8, 2

def train_fold(Xt, yt, s, subj, C, T, ncl, on_gpu, lam=0.10):
    set_seed(SEED)
    model = A._build_renet(C, T, use_dsa=False, n_classes=ncl).to(device)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].cpu().numpy(), 0.15, seed=int(subj))
    tr, val = tr_all[ti], tr_all[vi]
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
    return model

def weight_cos(model):
    W = model.spatial.weight.view(F1, D, -1)
    vals = []
    for f in range(F1):
        Wn = F.normalize(W[f], p=2, dim=-1)
        G = (Wn @ Wn.t()).abs()
        vals.append(float((G.sum() - G.diag().sum()) / (D * (D - 1) + 1e-9)))
    return float(np.mean(vals))

def act_redundancy(model, Xt, idx, on_gpu):
    outs = []
    with torch.no_grad():
        for j in range(0, len(idx), 128):
            bx = batch_index(Xt, torch.as_tensor(idx[j:j+128]), on_gpu)
            z = model.spatial(model.block1(bx.unsqueeze(1))).squeeze(2).cpu()
            outs.append(z)
    Z = torch.cat(outs, 0)
    corrs, redun = [], []
    for f in range(F1):
        ch = [f*D + j for j in range(D)]
        zf = Z[:, ch, :].numpy()
        cs = []
        for n in range(zf.shape[0]):
            a, b = zf[n, 0], zf[n, 1]
            a = a - a.mean(); b = b - b.mean()
            cs.append(abs(float(a @ b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
        corrs.append(np.mean(cs))
        M = zf.transpose(1, 0, 2).reshape(D, -1); M = M - M.mean(1, keepdims=True)
        cov = (M @ M.T) / (M.shape[1]-1)
        ev = np.clip(np.linalg.eigvalsh(cov).real, 1e-12, None); p = ev/ev.sum()
        redun.append(1.0 - float(np.exp(-(p*np.log(p)).sum()))/D)
    return float(np.mean(corrs)), float(np.mean(redun))

def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    subs = sorted(subjects_of(s))
    rows = json.load(open(JSON)) if os.path.exists(JSON) else {}
    print(f"Full-cohort (NO-DSA / pure OSFR) | {len(subs)} subjects | resume {len(rows)}\n")
    print(f"{'subj':>5} {'OSFRcos':>9} {'OSFRacorr':>10} {'OSFRredun':>10} {'EEGcos':>9} {'EEGacorr':>10} {'EEGredun':>10}")
    for i, subj in enumerate(subs):
        if i < START or i >= STOP or str(subj) in rows:
            continue
        te_idx = np.where(s == subj)[0]
        mo = train_fold(Xt, yt, s, subj, C, T, ncl, on_gpu, lam=0.10)
        oc = weight_cos(mo); oac, ored = act_redundancy(mo, Xt, te_idx, on_gpu)
        del mo
        if use_cuda: torch.cuda.empty_cache()
        me = train_fold(Xt, yt, s, subj, C, T, ncl, on_gpu, lam=0.0)
        ec = weight_cos(me); eac, ered = act_redundancy(me, Xt, te_idx, on_gpu)
        del me
        if use_cuda: torch.cuda.empty_cache()
        rows[str(subj)] = dict(osfr_wcos=oc, eeg_wcos=ec, osfr_acorr=oac,
                               eeg_acorr=eac, osfr_aredun=ored, eeg_aredun=ered)
        json.dump(rows, open(JSON, "w"), indent=2)
        print(f"{subj:>5} {oc:>9.4f} {oac:>10.4f} {ored:>10.4f} {ec:>9.4f} {eac:>10.4f} {ered:>10.4f}")
    if rows:
        ow = np.array([r["osfr_wcos"] for r in rows.values()])
        ew = np.array([r["eeg_wcos"] for r in rows.values()])
        oar = np.array([r["osfr_aredun"] for r in rows.values()])
        ear = np.array([r["eeg_aredun"] for r in rows.values()])
        oac = np.array([r["osfr_acorr"] for r in rows.values()])
        eac = np.array([r["eeg_acorr"] for r in rows.values()])
        print("\n" + "="*60)
        print(f"WEIGHT cos (n={len(ow)}): OSFR {ow.mean():.4f}+/-{ow.std():.4f}  EEGNet {ew.mean():.4f}+/-{ew.std():.4f}  max_OSFR={ow.max():.4f}")
        print(f"ACT corr : OSFR {oac.mean():.4f}  EEGNet {eac.mean():.4f}")
        print(f"ACT redun: OSFR {oar.mean():.4f}  EEGNet {ear.mean():.4f}")
        print("="*60)

if __name__ == "__main__":
    main()
