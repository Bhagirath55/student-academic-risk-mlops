import os
import pandas as pd
import numpy as np
import yaml
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------  Loading Config -------------------- #
with open("config/preprocessing_config.yaml", "r") as file:
    config = yaml.safe_load(file)

raw_data_path = os.getenv("RAW_DATA_PATH_OVERRIDE",config["paths"]["raw_data"])
print("Using Dataset:", raw_data_path)
processed_dir = config["paths"]["processed_dir"]
log_file = config["paths"]["log_file"]
preprocessor_file = config["paths"]["preprocessor_file"]

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# -------------------- Setup Logging -------------------- #
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(" Started Data Preprocessing")

# -------------------- Loading Dataset -------------------- #
data = pd.read_csv(raw_data_path)
logging.info(f"Loaded dataset with shape: {data.shape}")

# --------------------  Splitting X and y -------------------- #
target_col = config["columns"]["target"]
drop_cols = config["columns"]["drop"]
categorical_cols = config["columns"]["categorical"]

X = data.drop(drop_cols, axis=1)
y = data[target_col]

# Encoding target
le_target = LabelEncoder()
y = le_target.fit_transform(y)
logging.info("Encoded target column")

# Converting categorical cols to string
for col in categorical_cols:
    X[col] = X[col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
logging.info(f"Numeric features: {numeric_features}")
logging.info(f"Categorical features: {categorical_cols}")

# -------------------- Sanity Checks -------------------- #
assert not X.empty, "Dataset is empty!"
assert y.size > 0, "Target column is empty!"
logging.info(f"Data sanity checks passed")

# -------------------- Defining Transformers -------------------- #
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=config["imputation"]["numeric_strategy"])),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=config["imputation"]["categorical_strategy"])),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_cols)
])

# -------------------- Splitting Data -------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["split"]["test_size"],
    random_state=config["split"]["random_state"],
    stratify=y if config["split"]["stratify"] else None
)

logging.info(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")

# -------------------- Fitting and Transforming -------------------- #
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
logging.info("Transformed train and test data")

# -------------------- Saving Processed Files -------------------- #
np.save(f"{processed_dir}/X_train.npy", X_train_processed)
np.save(f"{processed_dir}/X_test.npy", X_test_processed)
np.save(f"{processed_dir}/y_train.npy", y_train)
np.save(f"{processed_dir}/y_test.npy", y_test)

# Saving preprocessor
joblib.dump(preprocessor, preprocessor_file)
logging.info(f"Preprocessor saved: {preprocessor_file}")

# -------------------- Saving Encoded Column Names -------------------- #
encoded_columns = list(
    preprocessor.named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_cols)
)
joblib.dump(encoded_columns, f"{processed_dir}/encoded_columns.pkl")

logging.info(" Data preprocessing completed successfully")
print("Data preprocessing completed successfully ")
