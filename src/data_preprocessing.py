import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
import numpy as np

# Creating artifacts folder if not exists
os.makedirs("artifacts/processed", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename='artifacts/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

logging.info("Started data preprocessing")

# Loading dataset
data = pd.read_csv("data/raw/student_data.csv")
logging.info(f"Dataset shape: {data.shape}")

# Removing 'id' column and separate target
X = data.drop(['Target', 'id'], axis=1)
y = data['Target']

# Encoding target column
le_target = LabelEncoder()
y = le_target.fit_transform(y)
logging.info("Encoded target column")

# Convert object columns explicitly to string (categorical)
categorical_cols = ['Displaced', 'Educational special needs', 'Debtor',
                    'Tuition fees up to date', 'Gender', 'Scholarship holder']

for col in categorical_cols:
    X[col] = X[col].astype(str)

# Identifying numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

logging.info(f"Numeric features: {numeric_features}")
logging.info(f"Categorical features: {categorical_features}")

# Preprocessing for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ColumnTransformer for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logging.info(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")

# Applying numeric preprocessing
X_train_num = preprocessor.fit_transform(X_train)
X_test_num = preprocessor.transform(X_test)
logging.info("Applied numeric preprocessing")

# Encoding categorical columns separately
for col in categorical_features:
    le_col = LabelEncoder()
    X_train[col] = le_col.fit_transform(X_train[col])
    X_test[col] = le_col.transform(X_test[col])
logging.info("Encoded categorical columns")

# Combining numeric and categorical processed data
X_train_processed = np.hstack([X_train_num, X_train[categorical_features].values])
X_test_processed = np.hstack([X_test_num, X_test[categorical_features].values])
logging.info("Combined numeric and categorical data")

# Saving processed datasets
pd.DataFrame(X_train_processed).to_csv("artifacts/processed/X_train.csv", index=False)
pd.DataFrame(X_test_processed).to_csv("artifacts/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("artifacts/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("artifacts/processed/y_test.csv", index=False)

# Saving preprocessor object for MLOps usage
joblib.dump(preprocessor, "artifacts/processed/preprocessor.pkl")
logging.info("Saved preprocessor object for future use")

logging.info("Data preprocessing completed successfully")
print("Data preprocessing completed!")
