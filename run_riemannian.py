"""
Classical Riemannian / CSP baselines under the SAME LOSO protocol (no DL).

  tslr   : Covariances -> Tangent Space (affine-invariant) -> Logistic Regression
  mdm    : Covariances -> Minimum Distance to Riemannian Mean
  csplda : Covariances -> CSP (log-variance) -> LDA

Usage:  python run_riemannian.py tslr | mdm | csplda
Env:    DATASET=physionet|bci2a|bci2b   SEED=2024
"""
import os, sys, json, time, warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import CSP

from common import BASE, SEED, set_seed, load_data, subjects_of, result_dir

warnings.filterwarnings("ignore")


def make_model(name):
    if name == "tslr":
        return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                             LogisticRegression(max_iter=2000, C=1.0))
    if name == "mdm":
        return make_pipeline(Covariances("oas"), MDM(metric="riemann"))
    if name == "csplda":
        return make_pipeline(Covariances("oas"), CSP(nfilter=6, log=True),
                             LinearDiscriminantAnalysis())
    raise ValueError(name)


def run(name):
    set_seed(SEED)
    out, folder = result_dir(name)
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(out, f"loso_{name}.json")

    print("=" * 50)
    print(f"{name.upper()} (classical) LOSO | SEED={SEED}")
    print("=" * 50)

    X, y, s, n_classes = load_data()
    subjects = subjects_of(s)
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}

    for sub in [sb for sb in subjects if str(sb) not in done]:
        t0 = time.time()
        tr, te = s != sub, s == sub
        clf = make_model(name)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = accuracy_score(y[te], pred)
        f1  = f1_score(y[te], pred, average="macro")
        kappa = cohen_kappa_score(y[te], pred)
        done[str(sub)] = {"acc": round(acc, 4), "f1": round(f1, 4), "kappa": round(kappa, 4)}
        json.dump(done, open(res_file, "w"), indent=2)
        print(f"S{sub:03d} | Acc:{acc:.2%} F1:{f1:.4f} | "
              f"{len(done)}/{len(subjects)} {time.time()-t0:.0f}s")

    accs = [v["acc"] for v in done.values()]
    print(f"\n{len(accs)} subjects: {np.mean(accs):.2%} +/- {np.std(accs):.2%}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tslr"
    assert name in ("tslr", "mdm", "csplda"), "Choose: tslr | mdm | csplda"
    run(name)
