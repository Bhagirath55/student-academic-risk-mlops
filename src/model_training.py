import os
import re
import json
import logging
import subprocess
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# TensorFlow/Keras commented out in your original file (kept commented)
# ...
# MLflow support
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# -------------------- Loading Config -------------------- #
with open("config/training_config.yaml") as f:
    config = yaml.safe_load(f)

ARTIFACTS_DIR = config['artifacts']['dir']
PROCESSED_DIR = config['artifacts']['processed_dir']
MODELS_DIR = config['artifacts']['models_dir']
METRICS_JSON = config['artifacts']['metrics_file']
TRAIN_LOG = config['artifacts']['log_file']
# Use env variable override if available (set by data_version_and_retrain.py)
DATA_PATH = os.environ.get("RAW_DATA_PATH_OVERRIDE", config['artifacts']['data_path'])

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------- Logging -------------------- #
logging.basicConfig(
    filename=TRAIN_LOG,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================
# 🧩 Data-change detection and retraining control
# ===============================================================
TIMESTAMP_FILE = os.path.join(MODELS_DIR, "data_timestamp.txt")

def data_changed():
    """Detect if new data arrived by comparing modification timestamps."""
    if not os.path.exists(DATA_PATH):
        logger.warning("⚠️ Data file not found for data_changed check: %s", DATA_PATH)
        return False

    new_data_time = datetime.fromtimestamp(os.path.getmtime(DATA_PATH))
    if not os.path.exists(TIMESTAMP_FILE):
        # No timestamp recorded -> consider as changed (first run)
        logger.info("No timestamp file found, treating data as changed.")
        return True

    with open(TIMESTAMP_FILE, "r") as f:
        last_time_str = f.read().strip()

    try:
        last_time = datetime.fromisoformat(last_time_str)
        changed = new_data_time > last_time
        logger.debug("Data changed? %s (new: %s, last: %s)", changed, new_data_time, last_time)
        return changed
    except Exception:
        logger.exception("Failed to parse timestamp file; treating data as changed.")
        return True

def update_data_timestamp():
    """Save latest data modification time after training."""
    if os.path.exists(DATA_PATH):
        ts = datetime.fromtimestamp(os.path.getmtime(DATA_PATH)).isoformat()
        with open(TIMESTAMP_FILE, "w") as f:
            f.write(ts)

# -------------------- Utility Functions -------------------- #
def load_processed_data():
    def _try_load(path):
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        return None

    X_train = _try_load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = _try_load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = _try_load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = _try_load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    if X_train is None or X_test is None or y_train is None or y_test is None:
        logger.error("Processed data missing! Run data_preprocessing.py first.")
        raise FileNotFoundError("Processed datasets not found. Ensure data_preprocessing.py has run and created X_train.npy etc.")

    return X_train, X_test, y_train, y_test

def run_preprocessing(data_path_override=None):
    """Run data_preprocessing.py optionally passing RAW_DATA_PATH_OVERRIDE env var.
       Returns True on success, raises on failure."""
    env = os.environ.copy()
    if data_path_override:
        env["RAW_DATA_PATH_OVERRIDE"] = data_path_override
        logger.info("Running preprocessing with RAW_DATA_PATH_OVERRIDE=%s", data_path_override)
    else:
        logger.info("Running preprocessing with default raw data path from config.")

    # Use subprocess.run to catch return code and logs
    p = subprocess.run(["python", "src/data_preprocessing.py"], env=env)
    if p.returncode != 0:
        logger.error("data_preprocessing.py failed with returncode %s", p.returncode)
        raise RuntimeError("data_preprocessing.py failed (check preprocessing logs).")
    logger.info("data_preprocessing.py finished successfully")
    return True

def eval_and_log(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

# -------------------- Model Versioning -------------------- #
def get_next_model_version(model_dir, base_name="best_model"):
    os.makedirs(model_dir, exist_ok=True)
    existing = [f for f in os.listdir(model_dir) if re.match(rf"{base_name}_v\d+\.pkl", f)]
    if not existing:
        return 1
    versions = [int(re.search(r"v(\d+)", f).group(1)) for f in existing]
    return max(versions) + 1

# -------------------- Main -------------------- #
def main():
    logger.info("🚀 Starting training pipeline")
    print("🚀 Starting training pipeline")

    # Determine whether to re-run preprocessing:
    processed_files_present = all(os.path.exists(os.path.join(PROCESSED_DIR, fn)) for fn in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"])
    logger.debug("Processed files present: %s", processed_files_present)

    # If data changed OR processed artifacts are missing -> run preprocessing
    if data_changed() or not processed_files_present:
        if data_changed():
            logger.info("📊 New or changed data detected. Will run preprocessing before training.")
            print("📊 New or changed data detected. Running preprocessing...")
        else:
            logger.info("⚠️ Processed artifacts missing. Running preprocessing before training.")
            print("⚠️ Processed artifacts missing. Running preprocessing...")

        # Run preprocessing; set RAW_DATA_PATH_OVERRIDE so preprocessing reads the right CSV
        try:
            run_preprocessing(data_path_override=DATA_PATH)
        except Exception as e:
            logger.exception("Preprocessing failed; aborting training.")
            raise

    else:
        logger.info("ℹ️ No data change detected and processed artifacts present. Skipping preprocessing.")
        print("ℹ️ No data change detected and processed artifacts present. Skipping preprocessing.")

    # Load processed data (X_train.npy etc.)
    X_train, X_test, y_train, y_test = load_processed_data()

    # -------------------- Loading preprocessor -------------------- #
    preprocessor_path = os.path.join(PROCESSED_DIR, "preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None

    if preprocessor and not isinstance(X_train, pd.DataFrame):
        if X_train.shape[1] == len(preprocessor.feature_names_in_):
            X_train = pd.DataFrame(X_train, columns=preprocessor.feature_names_in_)
            X_test = pd.DataFrame(X_test, columns=preprocessor.feature_names_in_)

    # -------------------- Candidate Models -------------------- #
    # input_dim = X_train.shape[1]
    # output_dim = len(np.unique(y_train))

    # handle possible class imbalance safe-guard for bincount
    try:
        pos_counts = np.bincount(y_train)
        scale_pos_weight = float(pos_counts[0] / pos_counts[1]) if len(pos_counts) > 1 and pos_counts[1] > 0 else 1.0
    except Exception:
        scale_pos_weight = 1.0

    candidates = {
        "RandomForest": RandomForestClassifier(
            **config['models']['RandomForest'],
            verbose=2,
        ),
        "XGBoost": XGBClassifier(
            **config['models']['XGBoost'],
            verbose=2,
            scale_pos_weight=scale_pos_weight
        )
    }

    cv = StratifiedKFold(n_splits=max(2, config['training'].get('cv_splits', 2)), shuffle=True, random_state=config['training']['random_state'])
    best_model, best_score, best_name = None, -1, None
    all_results = {}

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Student_Academic_Risk_Classification")
        mlflow.start_run()
        mlflow.set_tag("stage", "training")

    for name, model in candidates.items():
        logger.info(f"Training model: {name}")
        grid = GridSearchCV(model, {}, cv=cv, scoring='accuracy', n_jobs=1, verbose=2, error_score='raise')
        grid.fit(X_train, y_train)

        chosen = grid.best_estimator_
        score = grid.best_score_
        logger.info(f"{name} - Best CV Accuracy: {score:.4f}")

        test_metrics = eval_and_log(chosen, X_test, y_test)
        logger.info(f"{name} Test Accuracy: {test_metrics['accuracy']:.4f}")

        all_results[name] = {
            "cv_score": float(score),
            "test_metrics": test_metrics,
            "best_params": {}
        }

        if test_metrics['accuracy'] > best_score:
            best_score, best_model, best_name = test_metrics['accuracy'], chosen, name

        if MLFLOW_AVAILABLE:
            mlflow.log_param(f"{name}_best_params", {})
            mlflow.log_metric(f"{name}_cv_accuracy", float(score))
            mlflow.log_metric(f"{name}_test_accuracy", float(test_metrics['accuracy']))

    # -------------------- Saving Best Model with Version -------------------- #
    if best_model:
        version = get_next_model_version(MODELS_DIR)
        versioned_model_path = os.path.join(MODELS_DIR, f"best_model_v{version}.pkl")
        latest_model_path = os.path.join(MODELS_DIR, "best_model.pkl")

        if preprocessor:
            final_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', best_model)])
            joblib.dump(final_pipeline, versioned_model_path)
            joblib.dump(final_pipeline, latest_model_path)
        else:
            joblib.dump(best_model, versioned_model_path)
            joblib.dump(best_model, latest_model_path)

        with open(METRICS_JSON, "w") as f:
            json.dump(all_results, f, indent=4)

        update_data_timestamp()  # ✅ update timestamp only when retraining completes

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(versioned_model_path)
            mlflow.log_artifact(METRICS_JSON)
            mlflow.end_run()

        logger.info(f"Best model saved: {best_name} with accuracy {best_score:.4f} (version {version})")
        print(f"✅ Best model: {best_name}, Accuracy: {best_score:.4f}, Version: {version}")
    else:
        logger.warning("No best model found - training loop did not produce any model.")
        print("⚠️ No best model found - nothing saved.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed")
        raise e
