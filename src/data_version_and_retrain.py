# scripts/data_version_and_retrain.py
import os
import re
import shutil
import pandas as pd
import subprocess
from datetime import datetime

# Paths (tweak if your paths differ)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed_data")
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new_data", "new_student_data.csv")
BASE_FILE_NAME = "cleaned_student_dataset.csv"  # original file you currently have
BASE_FILE_PATH = os.path.join(PREPROCESSED_DIR, BASE_FILE_NAME)

def get_next_version(preprocessed_dir, base_name="cleaned_student_dataset"):
    files = [f for f in os.listdir(preprocessed_dir) if re.match(rf"{re.escape(base_name)}_v\d+\.csv", f)]
    if not files:
        return 1  # v1 is implicit as base file; next will be v2
    versions = [int(re.search(r"_v(\d+)\.csv", f).group(1)) for f in files]
    return max(versions) + 1

def ensure_paths():
    if not os.path.exists(PREPROCESSED_DIR):
        raise FileNotFoundError(f"Preprocessed dir not found: {PREPROCESSED_DIR}")
    if not os.path.exists(BASE_FILE_PATH):
        raise FileNotFoundError(f"Base dataset not found: {BASE_FILE_PATH}")
    if not os.path.exists(NEW_DATA_PATH):
        raise FileNotFoundError(f"No new data found at: {NEW_DATA_PATH}")

def create_versioned_dataset():
    ensure_paths()
    print("Loading files...")
    df_base = pd.read_csv(BASE_FILE_PATH)
    df_new = pd.read_csv(NEW_DATA_PATH)

    print(f"Base rows: {len(df_base)}, New rows: {len(df_new)}")

    df_combined = pd.concat([df_base, df_new], ignore_index=True)

    # Save versioned dataset
    v = get_next_version(PREPROCESSED_DIR, base_name="cleaned_student_dataset")
    versioned_filename = f"cleaned_student_dataset_v{v}.csv"
    versioned_path = os.path.join(PREPROCESSED_DIR, versioned_filename)
    df_combined.to_csv(versioned_path, index=False)
    print(f"Saved versioned dataset: {versioned_path}")

    # ALSO save a fixed "latest" dataset for model_training.py
    latest_path = os.path.join(PREPROCESSED_DIR, "cleaned_student_dataset_latest.csv")
    df_combined.to_csv(latest_path, index=False)
    print(f"Updated latest dataset for training: {latest_path}")

    return versioned_path  # keep returning versioned path for your existing logic

def run_preprocessing_and_training(versioned_csv_path):
    # We'll call data_preprocessing.py with env override so no YAML change required
    env = os.environ.copy()
    env["RAW_DATA_PATH_OVERRIDE"] = versioned_csv_path

    # call data_preprocessing.py (same as in your Docker: python src/data_preprocessing.py)
    print("Running preprocessing...")
    p1 = subprocess.run(["python", "src/data_preprocessing.py"], env=env, check=True)
    if p1.returncode != 0:
        raise RuntimeError("data_preprocessing.py failed (see logs)")

    # call model_training.py
    print("Running training...")
    p2 = subprocess.run(["python", "src/model_training.py"], env=env, check=True)
    if p2.returncode != 0:
        raise RuntimeError("model_training.py failed (see logs)")

    print("Preprocessing and training finished successfully.")

def main():
    versioned = create_versioned_dataset()
    run_preprocessing_and_training(versioned)
    # optional: move the processed new data to archive or rename new_student_data.csv
    # shutil.move(NEW_DATA_PATH, NEW_DATA_PATH + ".processed")

if __name__ == "__main__":
    main()
