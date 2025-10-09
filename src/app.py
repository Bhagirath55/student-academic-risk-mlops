from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Load model -------------------- #
model_pipeline = joblib.load("artifacts/models/best_model.pkl")

# -------------------- FastAPI app -------------------- #
app = FastAPI(title="Student Academic Risk Predictor API")

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Input Schema -------------------- #
class InputData(BaseModel):
    data: List[Dict]  # list of dicts, each dict is a row with feature names as keys

# -------------------- Prediction Route -------------------- #
@app.post("/predict")
def predict(input_data: InputData):
    try:
        df = pd.DataFrame(input_data.data)  # each dict becomes a row
        prediction = model_pipeline.predict(df)
        return {"predicted_risk_level": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
