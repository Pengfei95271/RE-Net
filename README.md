# RE-Net: A Parameter-Free Stiefel-Manifold Orthogonality Prior for Motor-Imagery Decoding

Code to reproduce the experiments in:

> **RE-Net: A Parameter-Free Stiefel-Manifold Orthogonality Prior for
> Non-Redundant, Interpretable Motor-Imagery Decoding with EEGNet.**
> Pengfei Cao et al. *Biomedical Signal Processing and Control* (under review).

RE-Net augments the EEGNet backbone with a single parameter-free module,
**Orthogonal Spatial-Filter Regularization (OSFR)**: a Frobenius-norm penalty
`‖ W̄ W̄ᵀ − I ‖_F` on the row-normalized depthwise spatial-convolution weights of
each frequency band, drawing them toward the Stiefel manifold of orthonormal
frames. OSFR yields non-redundant, individually analyzable spatial filters at
**no added parameters and no inference-time cost** — at test time RE-Net is
exactly EEGNet.

---

## Installation

```bash
git clone https://github.com/Pengfei95271/RE-Net.git
cd RE-Net
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.x. A CUDA GPU is recommended
(experiments were run on an RTX 3050, 4 GB).

## Data

The three motor-imagery datasets are **public** and are downloaded on first use
through MOABB / MNE. They are **not** redistributed here.

| Dataset       | Subjects | Channels | Classes | Source                        |
|---------------|----------|----------|---------|-------------------------------|
| PhysioNet MI  | 109      | 64       | 2       | PhysioNet EEGMMIDB            |
| BCI-IV-2a     | 9        | 22       | 4       | MOABB `BNCI2014001`           |
| Cho2017       | 52       | 64       | 2       | MOABB `Cho2017`               |

To pre-download all data:

```bash
python download.py
```

By default data is cached under `~/Datasets` (override with `MNE_DATA=/path`).

## Reproducing the main results

All experiments use **leakage-free leave-one-subject-out (LOSO)**: model
selection (early stopping) is performed on a validation split drawn only from
the training subjects, never on the test subject.

```bash
# --- PhysioNet MI (default dataset) ---
python run_renet.py                       # RE-Net (EEGNet + OSFR)
python run_baselines.py eegnet            # neural baselines
python run_baselines.py deepconvnet
python run_baselines.py shallow
python run_baselines.py atcnet
python run_baselines.py conformer
python run_baselines.py lmda
python run_baselines.py fbcnet
python run_baselines.py eegtcnet
python run_riemannian.py tslr             # classical Riemannian / CSP
python run_riemannian.py csplda
python run_riemannian.py mdm

# --- other datasets: set DATASET ---
DATASET=bci2a   python run_renet.py
DATASET=cho2017 python run_renet.py

# --- aggregate into the paper tables ---
python aggregate.py
```

### Key environment variables (see `common.py`)

| Variable            | Default     | Meaning                                              |
|---------------------|-------------|------------------------------------------------------|
| `DATASET`           | `physionet` | `physionet` \| `bci2a` \| `cho2017`                  |
| `SEED`              | `2024`      | per-fold seed is `SEED + subject`                    |
| `MNE_DATA`          | `~/Datasets`| raw-data location                                    |
| `MATCHED_PROTOCOL`  | `0`         | `1` = apply RE-Net's augmentation + weight decay to baselines (matched-protocol control) |

### Ablation and sensitivity

```bash
python run_analysis.py ablation osfr_only     # RE-Net vs EEGNet (OSFR on/off)
python run_analysis.py lambda                 # λ sensitivity sweep
```

### Multi-seed robustness (PhysioNet)

```bash
for s in 2024 2025 2026; do
  SEED=$s python run_renet.py
  SEED=$s python run_baselines.py eegnet
done
```

## Reproducing the figures

```bash
python osfr_dimensions.py            # orthogonality vs D  (Table: tab:osfr-D)
python analyzability_distribution.py # analyzability distribution figure
python cross_dataset_summary.py      # cross-dataset consistency figure
python make_missing_figs.py          # per-subject scatters + OSFR orthogonality/comparison
```

Figures are written to `figures/`. Reference copies of the published figures are
included there.

## Repository layout

```
.
├── common.py                 # data loading, seeding, val split, GPU residency
├── run_renet.py              # RE-Net (EEGNet + OSFR) LOSO
├── run_baselines.py          # neural-network baselines
├── run_riemannian.py         # classical Riemannian / CSP baselines
├── run_analysis.py           # ablation + λ sensitivity
├── aggregate.py              # collect results into tables
├── compare_all.py            # cross-model comparison
├── download.py               # pre-download datasets
├── *.py (paper figures)      # analyzability_distribution, cross_dataset_summary, ...
├── figures/                  # published figures (reference copies)
└── exploratory/              # exploratory experiments NOT reported in the paper
```

### `exploratory/` — negative and exploratory results

In the spirit of the paper's openness about what OSFR does **not** deliver, this
directory retains experiments that were run but **not included** in the
manuscript, because they were negative or inconclusive: low-data régime,
channel-dropout robustness, per-filter attribution, cross-subject filter
consistency, OSFR-vs-FBCSP subspace alignment, and DSA / topographic
visualizations from an earlier two-module design. They are provided for
transparency and are **not** needed to reproduce any paper result.

## Citation

```bibtex
@article{cao_renet,
  title   = {RE-Net: A Parameter-Free Stiefel-Manifold Orthogonality Prior
             for Non-Redundant, Interpretable Motor-Imagery Decoding with EEGNet},
  author  = {Cao, Pengfei and others},
  journal = {Biomedical Signal Processing and Control},
  note    = {under review}
}
```

## License

Released under the MIT License (see `LICENSE`).
