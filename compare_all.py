"""
Pairwise comparison across finished runs for ONE dataset.

Usage:  python compare_all.py
        DATASET=bci2a python compare_all.py
Env:    DATASET=physionet|bci2a|bci2b  (default physionet)

Only runs belonging to the selected dataset are shown, so PhysioNet and 2a
results never appear mixed in the same table.
"""
import os, json, glob
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ.get("DATASET", "physionet").lower()
OTHER = [d for d in ("bci2a", "bci2b") if d != DATASET]


def folder_dataset(folder):
    """Infer which dataset a *_run folder belongs to from its name."""
    for ds in ("bci2a", "bci2b"):
        if f"_{ds}" in folder:
            return ds
    return "physionet"


def strip_suffix(folder):
    name = folder
    for ds in ("bci2a", "bci2b"):
        name = name.replace(f"_{ds}", "")
    return name


def load_all():
    runs = {}
    for path in glob.glob(os.path.join(BASE, "*_run", "results", "loso_*.json")):
        folder = path.split(os.sep)[-3].replace("_run", "")
        if folder_dataset(folder) != DATASET:
            continue
        base = strip_suffix(folder)
        tag = os.path.basename(path)[len("loso_"):-len(".json")]
        name = tag if base == tag else base
        try:
            runs[name] = json.load(open(path))
        except Exception:
            pass
    return runs


def rank_biserial(d):
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    Rp, Rm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return (Rp - Rm) / (len(nz) * (len(nz) + 1) / 2)


def main():
    runs = load_all()
    if not runs:
        print(f"No results found for DATASET={DATASET}."); return

    print("=" * 72)
    print(f"  Mean accuracy per run  [DATASET={DATASET}]")
    print("=" * 72)
    print(f"{'Run':24s} | {'n':>4s} | {'Acc (%)':>16s} | {'F1':>7s} | {'kappa':>6s}")
    print("-" * 72)
    for name in sorted(runs, key=lambda k: -np.mean([v["acc"] for v in runs[k].values()])):
        r = runs[name]
        a = np.array([v["acc"] for v in r.values()]) * 100
        f = np.array([v.get("f1", np.nan) for v in r.values()])
        k = np.array([v.get("kappa", np.nan) for v in r.values()])
        kap = f"{np.nanmean(k):.3f}" if not np.all(np.isnan(k)) else "  -  "
        print(f"{name:24s} | {len(a):>4d} | {a.mean():>7.2f} +/- {a.std():5.2f} | "
              f"{np.nanmean(f):>7.4f} | {kap:>6s}")

    ref = "renet" if "renet" in runs else ("ablation:full" if "ablation:full" in runs else None)
    if ref is None:
        return
    print("\n" + "=" * 72)
    print(f"  Wilcoxon vs {ref}  (one-sided H1: {ref} > other)")
    print("=" * 72)
    print(f"{'Comparison':24s} | {'n':>4s} | {'delta':>8s} | {'p':>11s} | {'sig':>4s} | {'W/T/L':>10s} | {'r':>6s}")
    print("-" * 72)
    R = runs[ref]
    for name in sorted(runs):
        if name == ref:
            continue
        B = runs[name]
        subs = sorted(set(R) & set(B), key=int)
        if len(subs) < 3:
            continue
        a = np.array([R[s]["acc"] for s in subs])
        b = np.array([B[s]["acc"] for s in subs])
        d = a - b
        w, t, l = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
        if np.allclose(d, 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(a, b, alternative="greater")
            except ValueError:
                p = 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{name:24s} | {len(subs):>4d} | {d.mean()*100:>+7.2f}% | {p:>11.3e} | {sig:>4s} | "
              f"{w:>3d}/{t:>2d}/{l:>2d} | {rank_biserial(d):>+6.3f}")

    print("\nNote: With 4+ comparisons use a Bonferroni threshold (0.05/k).")


if __name__ == "__main__":
    main()
