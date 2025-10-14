import requests

API_URL = "http://localhost:8000/predict"

def predict_risk(input_features):
    try:
        payload = {"data": [input_features]}  # list of dicts
        response = requests.post(API_URL, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
