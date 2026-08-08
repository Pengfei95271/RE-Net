"""
DSA ERD/ERS visualization v2  (reviewer minor #2, corrected)

The v1 script showed monotonically rising power, which does not match ERD.
This version tests the DSA power branch more rigorously and more fairly:

  Fig A  dsa_erd_lateralized.png
     DSA power split by the hemisphere its spatial filter favours
     (contralateral vs ipsilateral to the imagined hand). Genuine ERD should
     show a contralateral power decrease relative to the ipsilateral side.

  Fig B  raw_erd_c3c4.png
     Ground-truth ERD from the raw C3/C4 channels using a standard
     (A - R)/R reference, to confirm the data itself contains ERD and to
     serve as the target the DSA branch is compared against.

  Fig C  dsa_corr_distribution.png
     Distribution, across several held-out subjects, of the correlation
     between the DSA power branch and the raw sensorimotor mu-ERD time
     course. A single subject can be misleading; the distribution is the
     evidence.

Decision rule:
  * If Fig A shows contralateral < ipsilateral (lateralized ERD) AND Fig C is
    centred well above zero -> DSA tracks sensorimotor ERD -> keep DSA.
  * If Fig A shows no lateralization and Fig C straddles zero -> DSA does not
    track ERD; its rising power is discriminative accumulation, not physiology
    -> drop DSA (route A), and report this honestly.

Usage:  python visualize_dsa.py            # uses SUBJECTS below
Env:    DATASET=physionet
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.metrics import accuracy_score

from common import (BASE, device, use_cuda, SEED, set_seed, load_data,
                    subjects_of, stratified_val_split, CFG_DATA)
from run_renet import RENet, osfr_loss, CFG

warnings.filterwarnings("ignore")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
FS = CFG_DATA["resample"]
TMIN, TMAX = CFG_DATA["tmin"], CFG_DATA["tmax"]
SUBJECTS = [int(x) for x in os.environ.get("SUBJECTS", "7,42,88,3,60").split(",")]

# PhysioNet 64-ch (10-10) left/right sensorimotor channel picks.
# moabb returns channels in dataset order; we resolve indices by name if
# available, else fall back to the canonical PhysioNet layout positions.
LEFT_CH  = ["C3", "FC3", "CP3", "C5"]      # left hemisphere (contralateral to RIGHT hand)
RIGHT_CH = ["C4", "FC4", "CP4", "C6"]      # right hemisphere (contralateral to LEFT hand)


def get_channel_names():
    """Best-effort channel names for PhysioNet MI via moabb metadata."""
    try:
        import mne
        from common import _moabb_dataset, DATA_DIR
        mne.set_log_level("CRITICAL")
        mne.set_config("MNE_DATA", DATA_DIR, set_env=True)
        ds = _moabb_dataset(CFG_DATA["dataset"])
        raw = list(list(ds.get_data(subjects=[1])[1].values())[0].values())[0]
        return [c.upper() for c in raw.copy().pick("eeg").ch_names]
    except Exception as e:
        print(f"  (channel names unavailable: {e}); lateralization skipped")
        return None


def train_one(Xt, yt, s, subj, C, T, n_classes):
    set_seed(SEED + subj)
    tr_all = np.where(s != subj)[0]
    ti, vi = stratified_val_split(yt[tr_all].numpy(), 0.15, SEED + subj)
    tr, val = tr_all[ti], tr_all[vi]
    model = RENet(C, T, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    ce = nn.CrossEntropyLoss()
    best, state = -1, None
    for ep in range(80):
        model.train()
        for i in torch.randperm(len(tr)).split(64):
            idx = tr[i.numpy()]
            bx = Xt[idx].to(device) + torch.randn(len(idx), C, T, device=device) * CFG["noise_std"]
            by = yt[idx].to(device)
            opt.zero_grad()
            (ce(model(bx), by) + CFG["lambda_osfr"] * osfr_loss(model)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xt[val].to(device)).argmax(1).cpu().numpy()
            a = accuracy_score(yt[val].numpy(), pv)
            if a > best:
                best, state = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model


def dsa_power(model, X):
    """DSA power branch of act1: (N, F1*D, T')."""
    pool = nn.AvgPool2d((1, 4))
    with torch.no_grad():
        h = model.bn1(model.spatial(model.block1(X.unsqueeze(1))))
        p = torch.log1p(pool(h ** 2))
    return p.squeeze(2).cpu().numpy()


def filter_hemisphere(model, ch_names):
    """For each of the F1*D DSA feature maps, decide whether its spatial filter
    weights the LEFT or RIGHT hemisphere more (sign of hemispheric preference)."""
    W = model.spatial.weight.data.squeeze().cpu().numpy()   # (F1*D, C)
    li = [ch_names.index(c) for c in LEFT_CH if c in ch_names]
    ri = [ch_names.index(c) for c in RIGHT_CH if c in ch_names]
    if not li or not ri:
        return None
    left_energy = np.abs(W[:, li]).mean(1)
    right_energy = np.abs(W[:, ri]).mean(1)
    return np.sign(left_energy - right_energy)   # +1 left-dominant, -1 right-dominant


def erd(sig, tp, base_end=0.75):
    base = sig[..., tp < (tp[0] + (base_end - TMIN))].mean(-1, keepdims=True)
    return (sig - base) / (np.abs(base) + 1e-9) * 100


def raw_bandpower(x, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return np.abs(hilbert(filtfilt(b, a, x, axis=-1), axis=-1)) ** 2


def main():
    X, y, s, n_classes = load_data()
    C, T = X.shape[1], X.shape[2]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    ch = get_channel_names()
    tp = np.linspace(TMIN, TMAX, T // 4)
    tp_raw = np.linspace(TMIN, TMAX, T)

    corr_list = []
    lat_contra, lat_ipsi = [], []
    raw_mu_grand = []

    for subj in SUBJECTS:
        if subj not in subjects_of(s):
            continue
        print(f"[S{subj:03d}] training + analysing...")
        model = train_one(Xt, yt, s, subj, C, T, n_classes)
        te = np.where(s == subj)[0]
        Xte, yte = Xt[te], y[te]
        P = dsa_power(model, Xte.to(device))         # (N, F1*D, T')

        # --- lateralization: contralateral vs ipsilateral DSA power ---
        if ch is not None:
            hemi = filter_hemisphere(model, ch)       # +1 left, -1 right per map
            if hemi is not None:
                for cls, hand in [(0, "left"), (1, "right")]:
                    sel = yte == cls
                    if sel.sum() == 0: continue
                    # contralateral hemisphere: right hemi for left hand, left hemi for right hand
                    contra_sign = -1 if hand == "left" else +1
                    contra = P[sel][:, hemi == contra_sign].mean((0, 1))
                    ipsi   = P[sel][:, hemi == -contra_sign].mean((0, 1))
                    lat_contra.append(erd(contra, tp))
                    lat_ipsi.append(erd(ipsi, tp))

        # --- raw C3/C4 mu ERD as ground truth + correlation with DSA ---
        if ch is not None:
            smc = [ch.index(c) for c in (LEFT_CH + RIGHT_CH) if c in ch]
            if smc:
                bp = raw_bandpower(Xte.numpy()[:, smc], 8, 12).mean((0, 1))   # (T,)
                bp_erd = erd(bp, tp_raw)
                raw_mu_grand.append(bp_erd)
                dsa_mean = erd(P.mean((0, 1)), tp)
                bp_ds = np.interp(tp, tp_raw, bp_erd)
                corr_list.append(np.corrcoef(dsa_mean, bp_ds)[0, 1])

    # ---- Fig A: lateralized DSA ERD ----
    if lat_contra:
        plt.figure(figsize=(8, 4))
        c = np.mean(lat_contra, 0); i = np.mean(lat_ipsi, 0)
        plt.plot(tp, c, color="#c53030", lw=2, label="Contralateral filters")
        plt.plot(tp, i, color="#2b6cb0", lw=2, label="Ipsilateral filters")
        plt.axhline(0, color="k", lw=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("DSA power change vs. baseline (%)")
        plt.title("DSA power, contralateral vs. ipsilateral (grand average)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "dsa_erd_lateralized.png"), dpi=300)
        print("Saved dsa_erd_lateralized.png")
        print(f"  mean contra-minus-ipsi over trial: {np.mean(c - i):+.1f}%  "
              f"(negative = contralateral ERD, the physiological sign)")

    # ---- Fig B: raw mu ERD ground truth ----
    if raw_mu_grand:
        plt.figure(figsize=(8, 4))
        m = np.mean(raw_mu_grand, 0)
        plt.plot(tp_raw, m, color="#2b6cb0", lw=2, label="raw C3/C4 mu (8-12 Hz)")
        plt.axhline(0, color="k", lw=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Power change vs. baseline (%)")
        plt.title("Ground-truth sensorimotor mu ERD/ERS (grand average)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "raw_erd_c3c4.png"), dpi=300)
        print("Saved raw_erd_c3c4.png")

    # ---- Fig C: correlation distribution ----
    if corr_list:
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(corr_list)), corr_list, color="#4a5568")
        plt.axhline(np.mean(corr_list), color="#c53030", ls="--",
                    label=f"mean = {np.mean(corr_list):+.3f}")
        plt.axhline(0, color="k", lw=0.5)
        plt.xticks(range(len(corr_list)), [f"S{sj:03d}" for sj in SUBJECTS[:len(corr_list)]])
        plt.ylabel("corr(DSA power, raw mu-ERD)")
        plt.title("DSA vs. raw mu-ERD correlation per subject")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "dsa_corr_distribution.png"), dpi=300)
        print("Saved dsa_corr_distribution.png")
        print(f"\nCorrelation across subjects: {np.mean(corr_list):+.3f} "
              f"+/- {np.std(corr_list):.3f}  (n={len(corr_list)})")

    print("\nDecision guide:")
    print("  contra-minus-ipsi clearly negative + corr distribution well above 0")
    print("    -> DSA tracks sensorimotor ERD -> KEEP DSA.")
    print("  no lateralization + corr near 0 -> DSA is discriminative accumulation,")
    print("    not ERD -> DROP DSA and say so honestly.")


if __name__ == "__main__":
    main()
