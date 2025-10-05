import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
import numpy as np

# -------------------- Setup -------------------- #
os.makedirs("artifacts/processed", exist_ok=True)

logging.basicConfig(
    filename='artifacts/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

logging.info("Started data preprocessing")

# -------------------- Load data -------------------- #
data = pd.read_csv("data/raw/student_data.csv")
logging.info(f"Dataset shape: {data.shape}")

X = data.drop(['Target', 'id'], axis=1)
y = data['Target']

# Encoding target
le_target = LabelEncoder()
y = le_target.fit_transform(y)
logging.info("Encoded target column")

# -------------------- Categorical columns -------------------- #
categorical_cols = ['Displaced', 'Educational special needs', 'Debtor',
                    'Tuition fees up to date', 'Gender', 'Scholarship holder']

for col in categorical_cols:
    X[col] = X[col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
logging.info(f"Numeric features: {numeric_features}")
logging.info(f"Categorical features: {categorical_cols}")

# -------------------- Preprocessing Pipelines -------------------- #
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer for **all features**
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_cols)
])

# -------------------- Splitting Data -------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logging.info(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")

# -------------------- Fitting preprocessor -------------------- #
preprocessor.fit(X_train)
logging.info("Fitted preprocessor to training data")

# -------------------- Transforming data -------------------- #
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
logging.info("Transformed train and test data")

# -------------------- Saving processed datasets -------------------- #
np.save("artifacts/processed/X_train.npy", X_train_processed)
np.save("artifacts/processed/X_test.npy", X_test_processed)
np.save("artifacts/processed/y_train.npy", y_train)
np.save("artifacts/processed/y_test.npy", y_test)

# Saving preprocessor
joblib.dump(preprocessor, "artifacts/processed/preprocessor.pkl")
logging.info("Saved preprocessor object for future use")

logging.info("Data preprocessing completed successfully")
print("Data preprocessing completed!")
