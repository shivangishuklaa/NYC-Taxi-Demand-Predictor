from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# App setup
app = FastAPI(title="NYC Taxi Demand Predictor", version="1.0")

# Load models
with open("models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/kmeans_zones.pkl", "rb") as f:
    kmeans = pickle.load(f)

cluster_centers = kmeans.cluster_centers_  # (40, 2) → [lat, lon]

# Request schema
class PredictRequest(BaseModel):
    cluster_id:  int   = 0
    hour_of_day: int   = 8
    day_of_week: int   = 0
    lag_1:       float = 10.0
    lag_3:       float = 9.0
    lag_6:       float = 8.0
    lag_144:     float = 12.0

#  Helper 
def build_features(r: PredictRequest) -> pd.DataFrame:
    is_weekend = 1 if r.day_of_week >= 5 else 0
    is_rush_am = 1 if 7  <= r.hour_of_day <= 9  else 0
    is_rush_pm = 1 if 17 <= r.hour_of_day <= 19 else 0
    is_night   = 1 if (r.hour_of_day >= 22 or r.hour_of_day <= 5) else 0
    bin_id     = 33 + (r.day_of_week * 144) + (r.hour_of_day * 6)

    center_lat = float(cluster_centers[r.cluster_id][0])
    center_lon = float(cluster_centers[r.cluster_id][1])

    rolling_mean_3  = (r.lag_1 + r.lag_3) / 2
    rolling_mean_12 = (r.lag_1 + r.lag_3 + r.lag_6 + r.lag_144) / 4

    return pd.DataFrame([{
        "cluster_id":       r.cluster_id,
        "bin_id":           bin_id,
        "hour_of_day":      r.hour_of_day,
        "day_of_week":      r.day_of_week,
        "is_weekend":       is_weekend,
        "is_rush_am":       is_rush_am,
        "is_rush_pm":       is_rush_pm,
        "is_night":         is_night,
        "center_lat":       center_lat,
        "center_lon":       center_lon,
        "lag_1":            r.lag_1,
        "lag_3":            r.lag_3,
        "lag_6":            r.lag_6,
        "lag_144":          r.lag_144,
        "rolling_mean_3":   rolling_mean_3,
        "rolling_mean_12":  rolling_mean_12,
    }])

# Routes 

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/zones")
def get_zones():
    """All 40 zone centers"""
    zones = [
        {"zone_id": i, "lat": round(float(cluster_centers[i][0]), 4),
         "lon": round(float(cluster_centers[i][1]), 4)}
        for i in range(len(cluster_centers))
    ]
    return {"zones": zones}

@app.post("/predict")
def predict(req: PredictRequest):
    """Predict demand for a single zone"""
    features   = build_features(req)
    prediction = float(xgb_model.predict(features)[0])
    prediction = max(0, round(prediction, 2))

    if prediction < 5:
        level = "Low"
    elif prediction < 15:
        level = "Medium"
    else:
        level = "High"

    return {
        "zone_id":    req.cluster_id,
        "hour":       req.hour_of_day,
        "day":        req.day_of_week,
        "prediction": prediction,
        "level":      level,
        "lat":        round(float(cluster_centers[req.cluster_id][0]), 4),
        "lon":        round(float(cluster_centers[req.cluster_id][1]), 4),
    }

@app.post("/predict-all")
def predict_all(req: PredictRequest):
    """Predict demand for all 40 zones at given time"""
    results = []
    for i in range(len(cluster_centers)):
        req.cluster_id = i
        features   = build_features(req)
        prediction = max(0, float(xgb_model.predict(features)[0]))
        results.append({
            "zone_id":    i,
            "prediction": round(prediction, 2),
            "lat":        round(float(cluster_centers[i][0]), 4),
            "lon":        round(float(cluster_centers[i][1]), 4),
        })
    results.sort(key=lambda x: x["prediction"], reverse=True)
    return {"hour": req.hour_of_day, "day": req.day_of_week, "zones": results}

#  Static files 
app.mount("/static", StaticFiles(directory="static"), name="static")