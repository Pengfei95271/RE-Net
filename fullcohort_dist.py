"""
Full-cohort weight/activation redundancy distribution (reviewer P1-5, 4.9/5.15).

The current Fig. 4 uses 30 subjects with an unexplained selection. Reviewer asks
for ALL available LOSO subjects (or a pre-defined rule + sensitivity). This
script recomputes, over ALL 109 held-out subjects, the within-branch spatial
weight cosine similarity for RE-Net (OSFR) and the unconstrained backbone, and
also records the activation-level redundancy so both live in one figure.

For speed on a 4 GB GPU we train once per subject per model (single seed, the
main seed), exactly as the analyzability figure did -- but over the full cohort.
Because per-subject training is the slow part, this is the long run
(~all subjects x 2 models). It writes incremental JSON so partial progress is
safe.

Outputs fullcohort_run/fullcohort.json (per-subject weight_cos and act metrics
for both models) and prints summary stats.

Usage:  python fullcohort_dist.py
Env:    DATASET=physionet  START=0  STOP=109
"""
import os, json, warnings
import numpy as np
import torch
import torch.nn.functional as F

import run_renet as R
from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, to_compute_tensors, batch_index)

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "fullcohort_run"); os.makedirs(OUT, exist_ok=True)
JSON = os.path.join(OUT, "fullcohort.json")
START = int(os.environ.get("START", "0"))
STOP = int(os.environ.get("STOP", "1000"))


def weight_cos(model):
    W = model.spatial.weight.view(R.CFG["F1"], R.CFG["D"], -1)
    vals = []
    for f in range(R.CFG["F1"]):
        Wn = F.normalize(W[f], p=2, dim=-1)
        G = (Wn @ Wn.t()).abs()
        D = G.shape[0]
        vals.append(float((G.sum() - G.diag().sum()) / (D * (D - 1) + 1e-9)))
    return float(np.mean(vals))


def act_redundancy(model, Xt, idx, on_gpu):
    F1, D = R.CFG["F1"], R.CFG["D"]
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


def train_one(Xt, yt, tr_idx, lam, C, T, ncl, on_gpu):
    old = R.CFG["lambda_osfr"]; R.CFG["lambda_osfr"] = lam
    try:
        set_seed(SEED); m = R.RENet(C, T, ncl).to(device)
        R.train(m, Xt, yt, tr_idx, SEED, on_gpu)
    finally:
        R.CFG["lambda_osfr"] = old
    m.eval(); return m


def main():
    X, y, s, ncl = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt, on_gpu = to_compute_tensors(X, y)
    subs = sorted(subjects_of(s))
    rows = json.load(open(JSON)) if os.path.exists(JSON) else {}
    print(f"Full-cohort distribution | {len(subs)} subjects | resume from {len(rows)} done\n")
    print(f"{'subj':>5} {'OSFRcos':>9} {'EEGcos':>9} {'OSFRredun':>10} {'EEGredun':>10}")

    for i, subj in enumerate(subs):
        if i < START or i >= STOP: continue
        if str(subj) in rows: continue
        tr_idx = np.where(s != subj)[0]; te_idx = np.where(s == subj)[0]
        mo = train_one(Xt, yt, tr_idx, 0.10, C, T, ncl, on_gpu)
        oc = weight_cos(mo); oac, ored = act_redundancy(mo, Xt, te_idx, on_gpu)
        del mo; torch.cuda.empty_cache() if use_cuda else None
        me = train_one(Xt, yt, tr_idx, 0.0, C, T, ncl, on_gpu)
        ec = weight_cos(me); eac, ered = act_redundancy(me, Xt, te_idx, on_gpu)
        del me; torch.cuda.empty_cache() if use_cuda else None
        rows[str(subj)] = dict(osfr_wcos=oc, eeg_wcos=ec,
                               osfr_acorr=oac, eeg_acorr=eac,
                               osfr_aredun=ored, eeg_aredun=ered)
        json.dump(rows, open(JSON, "w"), indent=2)
        print(f"{subj:>5} {oc:>9.4f} {ec:>9.4f} {ored:>10.4f} {ered:>10.4f}")

    # summary
    ow = np.array([r["osfr_wcos"] for r in rows.values()])
    ew = np.array([r["eeg_wcos"] for r in rows.values()])
    oar = np.array([r["osfr_aredun"] for r in rows.values()])
    ear = np.array([r["eeg_aredun"] for r in rows.values()])
    print("\n" + "="*60)
    print(f"WEIGHT cosine  (n={len(ow)}): OSFR {ow.mean():.4f}+/-{ow.std():.4f}  "
          f"EEGNet {ew.mean():.4f}+/-{ew.std():.4f}  max_OSFR={ow.max():.4f}")
    print(f"ACT redundancy (n={len(oar)}): OSFR {oar.mean():.4f}  EEGNet {ear.mean():.4f}")
    frac = float((ew > 0.3).mean()*100)
    print(f"EEGNet weight-cos > 0.3 in {frac:.1f}% of subjects; "
          f"OSFR max {ow.max():.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
