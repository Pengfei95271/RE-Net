"""Activation redundancy with PURE-OSFR model (no DSA). Env: TEST_K=15"""
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
OUT = os.path.join(BASE, "activation_nodsa"); os.makedirs(OUT, exist_ok=True)
TEST_K = int(os.environ.get("TEST_K", "15"))
F1, D = 8, 2

def train_fold(Xt, yt, s, subj, C, T, ncl, on_gpu, lam):
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

def spatial_outputs(model, Xt, idx, on_gpu, batch=128):
    outs = []
    with torch.no_grad():
        for j in range(0, len(idx), batch):
            bx = batch_index(Xt, torch.as_tensor(idx[j:j+batch]), on_gpu)
            z = model.spatial(model.block1(bx.unsqueeze(1)))
            outs.append(z.squeeze(2).cpu())
    return torch.cat(outs, 0)

def output_redundancy(Z):
    corrs, redun = [], []
    for f in range(F1):
        chans = [f*D + j for j in range(D)]
        zf = Z[:, chans, :].numpy()
        cs = []
        for n in range(zf.shape[0]):
            a, b = zf[n, 0], zf[n, 1]
            a = a - a.mean(); b = b - b.mean()
            cs.append(abs(float(a @ b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
        corrs.append(np.mean(cs))
        M = zf.transpose(1, 0, 2).reshape(D, -1); M = M - M.mean(1, keepdims=True)
        cov = (M @ M.T) / (M.shape[1]-1)
        ev = np.clip(np.linalg.eigvalsh(cov).real, 1e-12, None); p = ev/ev.sum()
        eff = float(np.exp(-(p*np.log(p)).sum()))
        redun.append(1.0 - eff/D)
    return float(np.mean(corrs)), float(np.mean(redun))

def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    subs = subjects_of(s)
    rng = np.random.RandomState(SEED)
    test_subs = sorted(rng.choice(subs, size=min(TEST_K, len(subs)), replace=False).tolist())
    print(f"Activation redundancy (NO-DSA / pure OSFR) | {len(test_subs)} subjects\n")
    print(f"{'subj':>5} {'OSFR|corr|':>11} {'EEG|corr|':>11} {'OSFRredun':>10} {'EEGredun':>10}")
    o_corr, e_corr, o_red, e_red = [], [], [], []
    for subj in test_subs:
        te_idx = np.where(s == subj)[0]
        mo = train_fold(Xt, yt, s, subj, C, T, ncl, on_gpu, lam=0.10)
        Zo = spatial_outputs(mo, Xt, te_idx, on_gpu); oc, orr = output_redundancy(Zo)
        del mo
        if use_cuda: torch.cuda.empty_cache()
        me = train_fold(Xt, yt, s, subj, C, T, ncl, on_gpu, lam=0.0)
        Ze = spatial_outputs(me, Xt, te_idx, on_gpu); ec, err = output_redundancy(Ze)
        del me
        if use_cuda: torch.cuda.empty_cache()
        o_corr.append(oc); e_corr.append(ec); o_red.append(orr); e_red.append(err)
        print(f"{subj:>5} {oc:>11.4f} {ec:>11.4f} {orr:>10.4f} {err:>10.4f}")
    from scipy.stats import wilcoxon
    oc, ec = np.array(o_corr), np.array(e_corr)
    orr, err = np.array(o_red), np.array(e_red)
    out = dict(osfr_output_corr_mean=float(oc.mean()), eegnet_output_corr_mean=float(ec.mean()),
               osfr_output_redun_mean=float(orr.mean()), eegnet_output_redun_mean=float(err.mean()),
               n=len(oc))
    try:
        _, p = wilcoxon(oc, ec, alternative="less"); out["wilcoxon_p_osfr_less_corr"] = float(p)
        _, p2 = wilcoxon(orr, err, alternative="less"); out["wilcoxon_p_osfr_less_redun"] = float(p2)
    except Exception:
        pass
    json.dump(dict(summary=out, osfr_corr=o_corr, eegnet_corr=e_corr,
                   osfr_redun=o_red, eegnet_redun=e_red),
              open(os.path.join(OUT, "activation_redundancy.json"), "w"), indent=2)
    print("\n" + "="*60)
    print(f"|corr|   OSFR {oc.mean():.4f}  EEGNet {ec.mean():.4f}  p={out.get('wilcoxon_p_osfr_less_corr','na')}")
    print(f"redun    OSFR {orr.mean():.4f}  EEGNet {err.mean():.4f}  p={out.get('wilcoxon_p_osfr_less_redun','na')}")
    print("="*60)

if __name__ == "__main__":
    main()
