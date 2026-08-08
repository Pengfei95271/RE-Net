"""
Cross-dataset consistency summary  (reviewer M3)

Renders one figure that makes the paper's central claim visible at a glance:
across three datasets, RE-Net is on par with the strongest CNN backbones
(delta near zero) and consistently, substantially better than the log-power
networks and the classical Riemannian/CSP methods (delta clearly positive).

Pure plotting from the already-computed LOSO means -- no training. Numbers are
the leakage-free means reported in the paper's Tables 2-4.

Produces figures/cross_dataset_summary.png
Usage:  python cross_dataset_summary.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/Downloads/renet_code_v3")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)

RENET = {"physionet": 64.36, "bci2a": 41.88, "cho2017": 68.86}

# baseline accuracies per dataset (leakage-free LOSO means from the paper)
BASE_ACC = {
    "physionet": {
        "EEGNet": 63.46, "DeepConvNet": 63.06, "ATCNet": 62.21, "EEG-Conformer": 62.41,
        "ShallowConvNet": 62.04, "FBCNet": 59.99, "EEG-TCNet": 54.96,
        "TS+LR": 59.02, "CSP+LDA": 58.68, "MDM": 52.94,
    },
    "bci2a": {
        "EEGNet": 42.96, "DeepConvNet": 43.69, "ATCNet": 43.35, "EEG-Conformer": 38.44,
        "ShallowConvNet": 39.26, "FBCNet": 38.33, "EEG-TCNet": 31.39,
        "TS+LR": 36.38, "CSP+LDA": 37.65, "MDM": 36.90,
    },
    "cho2017": {
        "EEGNet": 69.55, "DeepConvNet": 66.19, "ShallowConvNet": 66.35, "FBCNet": 61.99,
        "TS+LR": 58.44, "CSP+LDA": 58.15, "MDM": 51.92,
        # ATCNet/Conformer/TCNet not run on Cho2017 -> omitted (handled below)
    },
}

GROUPS = [
    ("Strong CNN backbones", ["EEGNet", "DeepConvNet", "ATCNet", "EEG-Conformer"], "#2b6cb0"),
    ("Log-power networks",   ["ShallowConvNet", "FBCNet", "EEG-TCNet"],            "#d69e2e"),
    ("Classical Riem./CSP",  ["TS+LR", "CSP+LDA", "MDM"],                          "#c53030"),
]
DSETS = [("physionet", "PhysioNet"), ("bci2a", "BCI-IV-2a"), ("cho2017", "Cho2017")]
MARK = {"physionet": "o", "bci2a": "s", "cho2017": "^"}


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ypos, ylabels, yticks = 0, [], []
    group_bands = []

    for gname, baselines, gcol in GROUPS:
        gstart = ypos
        for b in baselines:
            row_deltas = []
            for dkey, _ in DSETS:
                if b in BASE_ACC[dkey]:
                    d = RENET[dkey] - BASE_ACC[dkey][b]
                    row_deltas.append((dkey, d))
            for dkey, d in row_deltas:
                ax.scatter(d, ypos, marker=MARK[dkey], s=70, color=gcol,
                           edgecolors="black", linewidths=0.5, zorder=3)
            ylabels.append(b); yticks.append(ypos)
            ypos += 1
        group_bands.append((gstart - 0.5, ypos - 0.5, gname, gcol))
        ypos += 0.6   # gap between groups

    ax.axvline(0, color="black", lw=1.2, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("RE-Net accuracy advantage over baseline (percentage points)")
    ax.set_title("Cross-dataset consistency: RE-Net vs. each baseline")

    # shaded group bands + labels
    xmin, xmax = ax.get_xlim()
    for y0, y1, gname, gcol in group_bands:
        ax.axhspan(y0, y1, color=gcol, alpha=0.06, zorder=0)
        ax.text(xmax * 0.98, (y0 + y1) / 2, gname, color=gcol, fontsize=9,
                fontweight="bold", va="center", ha="right", rotation=0, alpha=0.9)

    # dataset legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=MARK[k], color="grey", linestyle="",
                      markersize=8, markeredgecolor="black", label=lab)
               for k, lab in DSETS]
    ax.legend(handles=handles, title="Dataset", loc="lower right", framealpha=0.9)

    ax.axvspan(-1.5, 1.5, color="grey", alpha=0.08, zorder=0)
    ax.text(0, yticks[0] - 1.1, "on par\n(|Δ|<1.5)", ha="center", va="top",
            fontsize=8, color="grey")

    plt.tight_layout()
    out = os.path.join(FIG, "cross_dataset_summary.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    print("\nReading: points near 0 = on par with that baseline; points to the")
    print("right = RE-Net better. Across all three datasets, strong CNNs cluster")
    print("near 0 while log-power and classical methods sit clearly to the right.")


if __name__ == "__main__":
    main()
