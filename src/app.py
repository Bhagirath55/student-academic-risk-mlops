from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import streamlit as st
import re
import io
from typing import List, Dict
import requests
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Load model -------------------- #
@st.cache_resource
def load_latest_model():
    repo = "Bhagirath55/student-academic-risk-mlops"
    path = "artifacts/models"
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref=dev"
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a valid response
    files = [file["name"] for file in response.json() if file["name"].endswith(".pkl")]
    # Extract version numbers and pick the highest one
    latest_model = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]))[-1]

    raw_url = f"https://raw.githubusercontent.com/{repo}/dev/{path}/{latest_model}"
    model_bytes = requests.get(raw_url).content
    model_pipeline = joblib.load(io.BytesIO(model_bytes))
    return model_pipeline, latest_model

model_pipeline, latest_model = load_latest_model()
#model_pipeline = joblib.load("artifacts/models/best_model.pkl")

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
        prediction = model_pipeline.predict(df)[0]
        return {"predicted_risk_level": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
