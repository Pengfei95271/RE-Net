"""
Analysis for RE-Net.
  python run_analysis.py ablation full|dsa_only|osfr_only
  python run_analysis.py sensitivity 0.10
  python run_analysis.py complexity
  python run_analysis.py statistical
Env: DATASET=physionet|bci2a|bci2b   SEED=2024
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import (BASE, device, use_cuda, SEED, set_seed,
                    load_data, subjects_of, stratified_val_split, DATASET,
                    to_compute_tensors, batch_index, MATCHED_PROTOCOL, result_dir)

warnings.filterwarnings("ignore")


class DualStateActivation(nn.Module):
    def __init__(self, k): super().__init__(); self.pool = nn.AvgPool2d(k)
    def forward(self, x): return self.pool(F.elu(x)) + torch.log1p(self.pool(x ** 2))

class StandardActivation(nn.Module):
    def __init__(self, k): super().__init__(); self.f = nn.Sequential(nn.ELU(True), nn.AvgPool2d(k))
    def forward(self, x): return self.f(x)


def _build_renet(C, T, use_dsa=True, n_classes=2):
    F1, D, F2, K, p = 8, 2, 16, 64, 0.25
    Act = DualStateActivation if use_dsa else StandardActivation
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, K), padding=(0, K//2), bias=False), nn.BatchNorm2d(F1))
            self.spatial = nn.Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False)
            self.bn1 = nn.BatchNorm2d(F1*D)
            self.act1 = nn.Sequential(Act((1, 4)), nn.Dropout(p))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
                nn.Conv2d(F1*D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2))
            self.act2 = nn.Sequential(Act((1, 8)), nn.Dropout(p))
            with torch.no_grad():
                flat = self.act2(self.block2(self.act1(self.bn1(self.spatial(
                    self.block1(torch.zeros(1, 1, C, T))))))).numel()
            self.head = nn.Linear(flat, n_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out")
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
    return sum(torch.norm(F.normalize(W[f], p=2, dim=-1) @ F.normalize(W[f], p=2, dim=-1).t() - I, p="fro")
               for f in range(8)) / 8


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


def loso_train(model, Xt, yt, s, subj, lam_osfr=0.10, noise=0.03, on_gpu=False):
    tr_all = np.where(s != subj)[0]
    te_idx = torch.as_tensor(np.where(s == subj)[0])
    ti, vi = stratified_val_split(yt[tr_all].cpu().numpy(), 0.15, seed=int(subj))
    tr, val = tr_all[ti], tr_all[vi]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce, scaler = nn.CrossEntropyLoss(), torch.amp.GradScaler("cuda", enabled=use_cuda)
    es, bs = EarlyStopping(20), 64
    val_t = torch.as_tensor(val)
    for ep in range(200):
        model.train()
        for i in torch.randperm(len(tr)).split(bs):
            idx = torch.as_tensor(tr[i.numpy()])
            bx = batch_index(Xt, idx, on_gpu)
            bx = bx + torch.randn_like(bx) * noise
            by = batch_index(yt, idx, on_gpu)
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
                pred = torch.cat([
                    model(batch_index(Xt, val_t[j:j+256], on_gpu)).argmax(1).cpu()
                    for j in range(0, len(val), 256)]).numpy()
            es(accuracy_score(yt[val].cpu().numpy(), pred), model)      # validation
            if es.should_stop: break
    es.restore(model); model.eval()
    with torch.no_grad():
        pred = torch.cat([
            model(batch_index(Xt, te_idx[j:j+256], on_gpu)).argmax(1).cpu()
            for j in range(0, len(te_idx), 256)]).numpy()
    yte = yt[te_idx].cpu().numpy()
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def cmd_ablation():
    variant = sys.argv[2] if len(sys.argv) > 2 else "full"
    assert variant in ("full", "dsa_only", "osfr_only"), "variant must be full|dsa_only|osfr_only"
    use_dsa = variant != "osfr_only"
    lam = 0.0 if variant == "dsa_only" else 0.10
    out, _ = result_dir(f"renet_ablation_{variant}"); os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_ablation_{variant}.json")
    print(f"{'='*50}\nAblation [{variant}] DSA={use_dsa} lambda={lam} | SEED={SEED}\n{'='*50}")
    X, y, s, n_classes = load_data(); C, T = X.shape[1], X.shape[2]; subjects = subjects_of(s)
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for sub in [i for i in subjects if str(i) not in done]:
        set_seed(SEED + sub); t0 = time.time()
        model = _build_renet(C, T, use_dsa, n_classes).to(device)
        acc, f1 = loso_train(model, Xt, yt, s, sub, lam, on_gpu=on_gpu)
        done[str(sub)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        json.dump(done, open(res_file, "w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | {len(done)}/{len(subjects)} {time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache() if use_cuda else None
    accs = [v["acc"] for v in done.values()]
    print(f"\n[{variant}] {len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


def cmd_sensitivity():
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    tag = f"lambda_{lam:.2f}".replace(".", "p")
    out, _ = result_dir(f"sensitivity_{tag}"); os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_{tag}.json")
    print(f"{'='*50}\nSensitivity lambda={lam} | SEED={SEED}\n{'='*50}")
    X, y, s, n_classes = load_data(); C, T = X.shape[1], X.shape[2]; subjects = subjects_of(s)
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for sub in [i for i in subjects if str(i) not in done]:
        set_seed(SEED + sub); t0 = time.time()
        model = _build_renet(C, T, True, n_classes).to(device)
        acc, f1 = loso_train(model, Xt, yt, s, sub, lam, on_gpu=on_gpu)
        done[str(sub)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        json.dump(done, open(res_file, "w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | {len(done)}/{len(subjects)} {time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache() if use_cuda else None
    accs = [v["acc"] for v in done.values()]
    print(f"\nlambda={lam} | {len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


def cmd_complexity():
    try:
        from thop import profile; has_thop = True
    except ImportError:
        has_thop = False; print("thop not installed (pip install thop)")
    sys.path.insert(0, BASE)
    from run_renet import RENet
    from run_baselines import (EEGNet, DeepConvNet, EEGConformer, LMDA,
                               ShallowConvNet, FBCNet, EEGTCNet, ATCNet)
    X, _, _, n_classes = load_data(verbose=False); C, T = X.shape[1], X.shape[2]; del X
    models = {"RE-Net (ours)": RENet(C,T,n_classes), "EEGNet": EEGNet(C,T,n_classes),
              "DeepConvNet": DeepConvNet(C,T,n_classes), "EEG-Conformer": EEGConformer(C,T,n_classes),
              "LMDA-Net": LMDA(C,T,n_classes), "ShallowConvNet": ShallowConvNet(C,T,n_classes),
              "FBCNet": FBCNet(C,T,n_classes), "EEG-TCNet": EEGTCNet(C,T,n_classes),
              "ATCNet": ATCNet(C,T,n_classes)}
    dummy = torch.randn(1, C, T)
    def latency(model, x, dev, n=500):
        model.eval(); x = x.to(next(model.parameters()).device)
        with torch.no_grad():
            for _ in range(50): model(x)
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n): model(x)
            if dev == "cuda": torch.cuda.synchronize()
        return (time.time()-t0)/n*1000
    print(f"\nDataset={DATASET} (C={C}, T={T}, classes={n_classes})")
    print(f"{'Model':16s} | {'Params':>10s} | {'FLOPs':>10s} | {'CPU ms':>8s} | {'GPU ms':>8s}")
    print("-"*66)
    for name, m in models.items():
        p = sum(x.numel() for x in m.parameters() if x.requires_grad); flops = "N/A"
        if has_thop:
            try: macs, _ = profile(m.cpu(), inputs=(dummy.cpu(),), verbose=False); flops = f"{macs/1e6:.2f}M"
            except Exception: flops = "err"
        lat_cpu = latency(m.cpu(), dummy.cpu(), "cpu")
        lat_gpu = latency(m.cuda(), dummy.cuda(), "cuda") if torch.cuda.is_available() else None
        gpu_s = f"{lat_gpu:.3f}" if isinstance(lat_gpu, float) else "N/A"
        print(f"{name:16s} | {p:>10,} | {flops:>10s} | {lat_cpu:>6.3f}ms | {gpu_s:>6s}ms"); m.cpu()


def cmd_statistical():
    from scipy.stats import wilcoxon
    renet_path = os.path.join(result_dir("renet")[0], "loso_renet.json")
    if not os.path.exists(renet_path):
        print("ERROR: RE-Net results not found. Run `python run_renet.py` first."); return
    renet = json.load(open(renet_path))
    baselines = {"EEGNet": "eegnet", "DeepConvNet": "deepconvnet", "EEG-Conformer": "conformer",
                 "LMDA-Net": "lmda", "ShallowConvNet": "shallow", "FBCNet": "fbcnet",
                 "EEG-TCNet": "eegtcnet", "ATCNet": "atcnet",
                 "TS+LR (Riem.)": "tslr", "MDM (Riem.)": "mdm", "CSP+LDA": "csplda"}
    print("=" * 78)
    print(f"  Wilcoxon Signed-Rank (one-sided, H1: RE-Net > Baseline) | DATASET={DATASET}")
    print("=" * 78)
    print(f"{'Comparison':24s} | {'p-value':>12s} | {'Sig':>4s} | {'W/T/L':>10s} | {'Mean Diff':>10s}")
    print("-" * 78)
    for disp, key in baselines.items():
        path = os.path.join(result_dir(key)[0], f"loso_{key}.json")
        if not os.path.exists(path):
            print(f"{'RE-Net vs '+disp:24s} | {'MISSING':>12s} |"); continue
        bl = json.load(open(path))
        subs = sorted(set(renet) & set(bl), key=int)
        if len(subs) < 2:
            print(f"{'RE-Net vs '+disp:24s} | {'<2 subj':>12s} |"); continue
        a = np.array([renet[c]["acc"] for c in subs]); b = np.array([bl[c]["acc"] for c in subs])
        diff = a - b; w, t, l = int((diff>0).sum()), int((diff==0).sum()), int((diff<0).sum())
        if np.allclose(diff, 0): p2 = 1.0
        else:
            try: _, p2 = wilcoxon(a, b, alternative="greater")
            except ValueError: p2 = 1.0
        sig = "***" if p2<0.001 else "**" if p2<0.01 else "*" if p2<0.05 else "ns"
        print(f"{'RE-Net vs '+disp:24s} | {p2:12.3e} | {sig:>4s} | {w:>3d}/{t:>2d}/{l:>2d} | {diff.mean()*100:>+8.2f}%")
    print("=" * 78)
    print(f"  n_subjects(RE-Net)={len(renet)}   * p<0.05, ** p<0.01, *** p<0.001")


COMMANDS = {"ablation": cmd_ablation, "sensitivity": cmd_sensitivity,
            "complexity": cmd_complexity, "statistical": cmd_statistical}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    assert cmd in COMMANDS, f"Usage: python run_analysis.py {{{','.join(COMMANDS)}}}"
    COMMANDS[cmd]()
