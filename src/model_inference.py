import os
import joblib
import pandas as pd
from datetime import datetime

# -------------------------------------------------------
# Helper Function to Load the Latest Trained Model
# -------------------------------------------------------
def get_latest_model(model_dir="artifacts/models"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No model file found in artifacts/models directory.")
    
    # Sort by modification time and get the latest one
    model_files = sorted(
        model_files,
        key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
        reverse=True
    )
    latest_model = model_files[0]
    latest_model_path = os.path.join(model_dir, latest_model)
    print(f"[INFO] Loaded latest model: {latest_model_path}")
    return joblib.load(latest_model_path)


# -------------------------------------------------------
# Function to Ensure All Required Columns Exist
# -------------------------------------------------------
def validate_input_columns(df_input, required_columns):
    missing_cols = [col for col in required_columns if col not in df_input.columns]
    if missing_cols:
        print(f"[WARNING] Missing columns in input data: {missing_cols}")
        for col in missing_cols:
            df_input[col] = 0  # Default value for missing columns
    df_input = df_input[required_columns]
    return df_input


# -------------------------------------------------------
# Main Inference Function
# -------------------------------------------------------
def main():
    # Load model (auto-detects the latest .pkl)
    pipeline = get_latest_model()

    # Example realistic input (from your dataset)
    df_input = pd.DataFrame([{
        "Application mode": 1,
        "Application order": 1,
        "Course": 9238,
        "Daytime/evening attendance": 1,
        "Previous qualification (grade)": 126,
        "Admission grade": 122.6,
        "Displaced": 0,
        "Debtor": 0,
        "Tuition fees up to date": 1,
        "Gender": 0,
        "Scholarship holder": 1,
        "Age at enrollment": 18,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 14.5,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 7,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 12.4285714285714,
        "Unemployment rate": 11.1,
        "GDP": 2.02
    }])

    # Extract preprocessor columns (from pipeline)
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "transformers_"):
            categorical_cols = []
            numeric_cols = []
            for name, transformer, cols in preprocessor.transformers_:
                if "cat" in name.lower():
                    categorical_cols.extend(cols)
                elif "num" in name.lower():
                    numeric_cols.extend(cols)
            all_required_columns = categorical_cols + numeric_cols
        else:
            all_required_columns = df_input.columns.tolist()
    else:
        all_required_columns = df_input.columns.tolist()

    # Validate input columns
    df_input = validate_input_columns(df_input, all_required_columns)

    # Make prediction
    prediction = pipeline.predict(df_input)[0]

    # Map numeric labels to class names
    class_mapping = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }
    class_name = class_mapping.get(prediction, "Unknown")

    print(f"\n✅ Predicted Academic Risk Level: {prediction} ({class_name})")


# -------------------------------------------------------
# Run Script
# -------------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] Running inference at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
