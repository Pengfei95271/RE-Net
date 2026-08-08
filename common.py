"""
Shared utilities: seeding, dataset loading, validation split, GPU residency.

Env vars
--------
DATASET=physionet|bci2a|bci2b|cho2017  (default physionet)
SEED=2024                          per-fold seed is SEED+subject
MNE_DATA=/path                     raw data location
GPU_CACHE=1                        keep whole dataset on GPU (default 1; big speedup)
MATCHED_PROTOCOL=0                 1 = apply RE-Net's noise aug + weight decay to
                                   baselines too (reviewer Major #6)
"""
import os, random
import numpy as np
import torch

BASE     = os.path.dirname(os.path.abspath(__file__))
CACHE    = os.path.join(BASE, "cache")
DATA_DIR = os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "Datasets"))

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

SEED             = int(os.environ.get("SEED", "2024"))
GPU_CACHE        = os.environ.get("GPU_CACHE", "1") == "1"
MATCHED_PROTOCOL = os.environ.get("MATCHED_PROTOCOL", "0") == "1"

# RE-Net's training protocol, reused for baselines when MATCHED_PROTOCOL=1
MATCHED_NOISE_STD   = 0.03
MATCHED_WEIGHT_DECAY = 0.01


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_val_split(y, val_frac=0.15, seed=0):
    """(train_idx, val_idx) stratified by class. Test subject never involved."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y); val_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.append(idx[:n_val])
    val_idx = np.concatenate(val_idx)
    mask = np.ones(len(y), bool); mask[val_idx] = False
    return np.where(mask)[0], val_idx


def to_compute_tensors(X, y):
    """Return (Xt, yt, on_gpu). Keeps the whole dataset resident on the GPU when
    it fits -- this removes the per-batch host->device copy that dominates
    runtime for these small models. Numerically identical to per-batch .to()."""
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    if use_cuda and GPU_CACHE:
        need_gb = (Xt.numel() * 4 + yt.numel() * 8) / 1e9
        try:
            free, _ = torch.cuda.mem_get_info()
            if free / 1e9 > need_gb * 1.6:      # leave headroom for activations
                Xt = Xt.to(device); yt = yt.to(device)
                print(f"  [GPU_CACHE] dataset resident on GPU ({need_gb:.2f} GB)")
                return Xt, yt, True
            print(f"  [GPU_CACHE] skipped: needs {need_gb:.2f} GB, free {free/1e9:.2f} GB")
        except Exception as e:
            print(f"  [GPU_CACHE] skipped ({e})")
    if use_cuda:
        Xt = Xt.pin_memory(); yt = yt.pin_memory()
    return Xt, yt, False


def batch_index(t, idx, on_gpu):
    """Index a (possibly GPU-resident) tensor with a CPU index tensor."""
    if on_gpu:
        return t[idx.to(t.device)]
    return t[idx].to(device)


DATASETS = {
    "physionet": dict(events=["left_hand", "right_hand"], n_classes=2,
                      dataset="PhysionetMI", subjects=list(range(1, 110)),
                      cache="physionetmi_casa_preprocessed.npz",
                      fmin=4, fmax=40, tmin=0.5, tmax=3.5, resample=128),
    "bci2a":     dict(events=["left_hand", "right_hand", "feet", "tongue"], n_classes=4,
                      dataset="BNCI2014_001", subjects=list(range(1, 10)),
                      cache="bci2a_preprocessed.npz",
                      fmin=4, fmax=40, tmin=2.0, tmax=6.0, resample=128),
    "bci2b":     dict(events=["left_hand", "right_hand"], n_classes=2,
                      dataset="BNCI2014_004", subjects=list(range(1, 10)),
                      cache="bci2b_preprocessed.npz",
                      fmin=4, fmax=40, tmin=0.5, tmax=3.5, resample=128),
    "cho2017":   dict(events=["left_hand", "right_hand"], n_classes=2,
                      dataset="Cho2017", subjects=list(range(1, 53)),
                      cache="cho2017_preprocessed.npz",
                      fmin=4, fmax=40, tmin=0.5, tmax=3.5, resample=128),
}
DATASET = os.environ.get("DATASET", "physionet").lower()
assert DATASET in DATASETS, f"DATASET must be one of {list(DATASETS)}"
CFG_DATA = DATASETS[DATASET]


def _moabb_dataset(name):
    import moabb.datasets as md
    if hasattr(md, name):
        return getattr(md, name)()
    return getattr(md, name.replace("_", ""))()


def load_data(verbose=True):
    cache = os.path.join(CACHE, CFG_DATA["cache"])
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y, s = d["X"], d["y"], d["s"]
    else:
        import mne, moabb
        from moabb.paradigms import MotorImagery
        from sklearn.preprocessing import LabelEncoder
        mne.set_log_level("CRITICAL"); moabb.set_log_level("CRITICAL")
        mne.set_config("MNE_DATA", DATA_DIR, set_env=True)
        par = MotorImagery(events=CFG_DATA["events"], n_classes=CFG_DATA["n_classes"],
                           fmin=CFG_DATA["fmin"], fmax=CFG_DATA["fmax"],
                           tmin=CFG_DATA["tmin"], tmax=CFG_DATA["tmax"],
                           resample=CFG_DATA["resample"])
        X, y, meta = par.get_data(dataset=_moabb_dataset(CFG_DATA["dataset"]),
                                  subjects=CFG_DATA["subjects"])
        s = meta["subject"].values.astype(int)
        y = LabelEncoder().fit_transform(y)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(cache, X=X.astype(np.float32),
                            y=y.astype(np.int64), s=s.astype(np.int64))
    X = ((X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
    y = y.astype(np.int64); s = s.astype(np.int64)
    n_classes = int(len(np.unique(y)))
    if verbose:
        print(f"[{DATASET}] Data: {X.shape[0]} trials, {X.shape[1]}ch, "
              f"{X.shape[2]}tp, {n_classes} classes, {len(np.unique(s))} subjects")
    return X, y, s, n_classes


def subjects_of(s):
    return sorted(int(v) for v in np.unique(s))


def result_dir(tag):
    """Results go to <tag>_run for physionet. Non-default datasets get a
    dataset suffix (<tag>_bci2a_run), MATCHED_PROTOCOL adds _matched, and a
    non-default SEED adds _seed<N>. This guarantees PhysioNet, 2a and 2b
    results never overwrite one another."""
    name = tag
    if DATASET != "physionet":
        name = f"{name}_{DATASET}"
    if MATCHED_PROTOCOL:
        name = f"{name}_matched"
    if SEED != 2024:
        name = f"{name}_seed{SEED}"
    return os.path.join(BASE, f"{name}_run", "results"), name
