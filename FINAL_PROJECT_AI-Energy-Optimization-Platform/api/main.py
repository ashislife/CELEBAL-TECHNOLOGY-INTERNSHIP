from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="AI Energy Optimization API",
    version="1.0"
)

# ===========================
# Load Model
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "energy_forecast_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

# ===========================
# Request Models
# ===========================

class PredictionRequest(BaseModel):
    features: list

class DashboardInput(BaseModel):
    month: int
    day_of_week: int
    is_weekend: int
    temperatureMax: float
    humidity: float
    windSpeed: float
    pressure: float
    lag_1: float
    lag_7: float
    lag_30: float
    rolling_mean_7: float
    rolling_std_7: float

# ===========================
# Home
# ===========================

@app.get("/")
def home():
    return {
        "message": "AI Energy Optimization API is running",
        "model_loaded": True,
        "number_of_features": len(feature_names)
    }

# ===========================
# Original Prediction
# ===========================

@app.post("/predict")
def predict(data: PredictionRequest):

    df = pd.DataFrame(
        [data.features],
        columns=feature_names
    )

    prediction = model.predict(df)[0]

    return {
        "prediction": float(prediction)
    }

# ===========================
# Dashboard Prediction
# ===========================

@app.post("/predict_dashboard")
def predict_dashboard(data: DashboardInput):

    sample = pd.DataFrame(
        [[0]*len(feature_names)],
        columns=feature_names
    )

    sample["month"] = data.month
    sample["day_of_week"] = data.day_of_week
    sample["is_weekend"] = data.is_weekend

    sample["temperatureMax"] = data.temperatureMax
    sample["humidity"] = data.humidity
    sample["windSpeed"] = data.windSpeed
    sample["pressure"] = data.pressure

    sample["lag_1"] = data.lag_1
    sample["lag_7"] = data.lag_7
    sample["lag_30"] = data.lag_30

    sample["rolling_mean_7"] = data.rolling_mean_7
    sample["rolling_std_7"] = data.rolling_std_7

    prediction = model.predict(sample)[0]

    return {
        "predicted_energy_consumption": round(float(prediction),3)
    }