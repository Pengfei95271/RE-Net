"""
Aggregate LOSO results across seeds (reviewer Major #4).

Usage:
  python aggregate.py renet eegnet deepconvnet shallow fbcnet
  python aggregate.py --matched eegnet deepconvnet

Reads every <tag>_run and <tag>_seed*_run folder it can find, reports
per-seed means and the across-seed mean +/- std, then runs a Wilcoxon test
against RE-Net on the seed-averaged per-subject accuracies.
"""
import os, sys, json, glob
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def collect(tag, matched=False):
    """-> {seed: {subject: {'acc':..,'f1':..}}}"""
    base = f"{tag}_matched" if matched else tag
    out = {}
    default = os.path.join(BASE, f"{base}_run", "results", f"loso_{tag}.json")
    if os.path.exists(default):
        out[2024] = json.load(open(default))
    for path in sorted(glob.glob(os.path.join(BASE, f"{base}_seed*_run", "results", f"loso_{tag}.json"))):
        folder = path.split(os.sep)[-3]
        try:
            seed = int(folder.split("_seed")[1].split("_run")[0])
        except (IndexError, ValueError):
            continue
        out[seed] = json.load(open(path))
    return out


def seed_average(runs, metric="acc"):
    """-> (subjects, mean_over_seeds_per_subject, per_seed_means)"""
    if not runs:
        return [], np.array([]), {}
    subs = sorted(set.intersection(*[set(r) for r in runs.values()]), key=int)
    per_seed_mean, mat = {}, []
    for seed, r in sorted(runs.items()):
        v = np.array([r[s][metric] for s in subs])
        per_seed_mean[seed] = v.mean()
        mat.append(v)
    return subs, np.mean(mat, axis=0), per_seed_mean


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    matched = "--matched" in sys.argv
    if not args:
        print(__doc__); return

    print("=" * 74)
    print(f"  Multi-seed aggregation{'  [MATCHED_PROTOCOL]' if matched else ''}")
    print("=" * 74)
    print(f"{'Model':16s} | {'seeds':>5s} | {'per-seed means':<30s} | {'mean +/- std':>14s}")
    print("-" * 74)

    table = {}
    for tag in args:
        runs = collect(tag, matched)
        subs, avg, per_seed = seed_average(runs)
        if not runs:
            print(f"{tag:16s} | {'-':>5s} | {'MISSING':<30s} |")
            continue
        means = np.array(list(per_seed.values())) * 100
        shown = " ".join(f"{m:.2f}" for m in means[:5])
        print(f"{tag:16s} | {len(runs):>5d} | {shown:<30s} | "
              f"{means.mean():>7.2f} +/- {means.std():.2f}")
        table[tag] = (subs, avg)

    if "renet" in table and len(table) > 1:
        from scipy.stats import wilcoxon
        print("\n" + "=" * 74)
        print("  Wilcoxon on seed-averaged per-subject accuracy (H1: RE-Net > baseline)")
        print("=" * 74)
        rs, ra = table["renet"]
        for tag, (bs, ba) in table.items():
            if tag == "renet":
                continue
            common = sorted(set(rs) & set(bs), key=int)
            a = np.array([ra[rs.index(c)] for c in common])
            b = np.array([ba[bs.index(c)] for c in common])
            d = a - b
            if np.allclose(d, 0):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(a, b, alternative="greater")
                except ValueError:
                    p = 1.0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  vs {tag:14s} n={len(common):3d}  delta={d.mean()*100:+6.2f}%  "
                  f"p={p:.3e} {sig}")


if __name__ == "__main__":
    main()
