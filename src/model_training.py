import os
import re
import json
import logging
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

# TensorFlow/Keras for ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

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

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------- Logging -------------------- #
logging.basicConfig(
    filename=TRAIN_LOG,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

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
        raise FileNotFoundError("Processed datasets not found.")

    return X_train, X_test, y_train, y_test

def build_ann_from_config(cfg, input_dim=None, output_dim=None):
    model = Sequential()
    layers = cfg['layers']

    # Input layer
    model.add(Dense(layers[0]['units'], activation=layers[0]['activation'], input_dim=input_dim))
    if 'dropout' in layers[0] and layers[0]['dropout'] > 0:
        model.add(Dropout(layers[0]['dropout']))

    # Hidden layers
    for layer in layers[1:]:
        model.add(Dense(layer['units'], activation=layer['activation']))
        if 'dropout' in layer and layer['dropout'] > 0:
            model.add(Dropout(layer['dropout']))

    # Output layer
    model.add(Dense(output_dim, activation=cfg['output_activation']))

    # Optimizer
    opt_cfg = cfg['optimizer']
    if opt_cfg['type'].lower() == 'adam':
        optimizer = Adam(learning_rate=opt_cfg['learning_rate'])
    else:
        optimizer = opt_cfg['type']

    model.compile(optimizer=optimizer, loss=cfg['loss'], metrics=cfg['metrics'])
    return model

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
    logger.info("Starting training pipeline")
    X_train, X_test, y_train, y_test = load_processed_data()

    # -------------------- Loading preprocessor -------------------- #
    preprocessor_path = os.path.join(PROCESSED_DIR, "preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None

    if preprocessor and not isinstance(X_train, pd.DataFrame):
        if X_train.shape[1] == len(preprocessor.feature_names_in_):
            X_train = pd.DataFrame(X_train, columns=preprocessor.feature_names_in_)
            X_test = pd.DataFrame(X_test, columns=preprocessor.feature_names_in_)

    # -------------------- Candidate Models -------------------- #
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    candidates = {
        "RandomForest": RandomForestClassifier(
            **config['models']['RandomForest'],
            verbose=2,
        ),
        "XGBoost": XGBClassifier(
            **config['models']['XGBoost'],
            verbose=2,
            scale_pos_weight=np.bincount(y_train)[0] / np.bincount(y_train)[1]
        ),
        "ANN": KerasClassifier(
            model=build_ann_from_config,
            model__cfg=config['models']['ANN'],
            model__input_dim=input_dim,
            model__output_dim=output_dim,
            verbose=2
        )
    }

    # -------------------- Cross-Validation and Training -------------------- #
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=config['training']['random_state'])
    best_model, best_score, best_name = None, -1, None
    all_results = {}

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Student_Academic_Risk_Classification")
        mlflow.start_run()
        mlflow.set_tag("stage", "training")

    # -------------------- Training Loop -------------------- #
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
            "best_params": {}  # fixed hyperparameters from config
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

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(versioned_model_path)
            mlflow.log_artifact(METRICS_JSON)
            mlflow.end_run()

        logger.info(f"Best model saved: {best_name} with accuracy {best_score:.4f} (version {version})")
        print(f"✅ Best model: {best_name}, Accuracy: {best_score:.4f}, Version: {version}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed")
        raise e
