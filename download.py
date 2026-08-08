"""Download raw data for the selected dataset. DATASET=physionet|bci2a|bci2b python download.py"""
import os, warnings
import mne, moabb
from common import DATASET, CFG_DATA, DATA_DIR, _moabb_dataset
warnings.filterwarnings("ignore")
mne.set_log_level("CRITICAL"); moabb.set_log_level("CRITICAL")
os.makedirs(DATA_DIR, exist_ok=True); mne.set_config("MNE_DATA", DATA_DIR, set_env=True)
if __name__ == "__main__":
    ds = _moabb_dataset(CFG_DATA["dataset"]); subjects = CFG_DATA["subjects"]
    print(f"Downloading {DATASET} ({CFG_DATA['dataset']}) to {DATA_DIR} ...")
    try:
        ds.download(subject_list=subjects, path=DATA_DIR, verbose=False); print("Done.")
    except Exception:
        for sid in subjects:
            try: ds.download(subject_list=[sid], path=DATA_DIR, verbose=False); print(f"  S{sid:03d} OK")
            except Exception as e: print(f"  S{sid:03d} FAILED: {e}")
