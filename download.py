import os, warnings
import mne, moabb
from moabb.datasets import PhysionetMI

warnings.filterwarnings("ignore")
mne.set_log_level("CRITICAL")
moabb.set_log_level("CRITICAL")

DATA_PATH = os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "Datasets"))
os.makedirs(DATA_PATH, exist_ok=True)
mne.set_config("MNE_DATA", DATA_PATH, set_env=True)

if __name__ == "__main__":
    ds = PhysionetMI()
    print(f"Downloading to {DATA_PATH}...")
    try:
        ds.download(subject_list=list(range(1, 110)), path=DATA_PATH, verbose=False)
        print("Done.")
    except Exception:
        for sid in range(1, 110):
            try:
                ds.download(subject_list=[sid], path=DATA_PATH, verbose=False)
                print(f"  S{sid:03d} OK")
            except Exception as e:
                print(f"  S{sid:03d} FAILED: {e}")
