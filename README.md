# 🚕 Newyork Taxi Demand Prediction

> Predicting yellow cab pickup demand per 10-minute bin per zone across New York City — trained on Jan 2015, evaluated on Jan 2016.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11-teal?style=flat-square)
![R2](https://img.shields.io/badge/R²-0.965-brightgreen?style=flat-square)

---

## 📌 Problem Statement

NYC taxi operators and dispatchers need to anticipate demand surges before they happen. This project builds a machine learning pipeline that predicts the number of taxi pickups in any of **40 geo-clusters** for any **10-minute time window**, enabling proactive fleet positioning.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | NYC Taxi & Limousine Commission (TLC) |
| Training data | January 2015 yellow cab trips |
| Test data | January 2016 yellow cab trips |
| Raw records | ~12 million trips |
| After cleaning | ~11.2 million trips |

**Download links:**
- [yellow_tripdata_2015-01.csv](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [yellow_tripdata_2016-01.csv](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## 🏗️ Project Pipeline

```
Raw CSV (12M rows)
    ↓
Data Cleaning (Dask — out-of-core processing)
    ↓
Outlier Removal (trip time, speed, distance, fare, coordinates)
    ↓
Geo-clustering (MiniBatchKMeans → 40 NYC zones)
    ↓
Feature Engineering (16 features — temporal + lag + geospatial)
    ↓
Model Training (7 models compared)
    ↓
Best Model: XGBoost
    ↓
Deployment (FastAPI + Interactive Map UI)
```

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `hour_of_day` | 0–23 |
| `day_of_week` | 0=Mon, 6=Sun |
| `is_weekend` | 1 if Sat/Sun |
| `is_rush_am` | 1 if 7–9 AM |
| `is_rush_pm` | 1 if 5–7 PM |
| `is_night` | 1 if 10 PM–5 AM |
| `center_lat/lon` | Zone cluster center coordinates |
| `lag_1` | Pickups 10 min ago |
| `lag_3` | Pickups 30 min ago |
| `lag_6` | Pickups 1 hr ago |
| `lag_144` | Pickups same time yesterday |
| `rolling_mean_3` | Rolling avg last 30 min |
| `rolling_mean_12` | Rolling avg last 2 hrs |

---

## 🤖 Model Comparison

| Rank | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 🥇 1 | **XGBoost** | **11.580** | **7.599** | **0.9652** |
| 🥈 2 | LightGBM | 11.596 | 7.622 | 0.9651 |
| 🥉 3 | Random Forest | 11.625 | 7.600 | 0.9649 |
| 4 | Decision Tree | 12.384 | 8.060 | 0.9602 |
| 5 | Linear Regression | 12.502 | 8.246 | 0.9594 |
| 6 | Ridge Regression | 12.503 | 8.246 | 0.9594 |
| 7 | Baseline (mean) | ~18.2 | — | — |

**XGBoost achieved ~36% RMSE improvement over baseline.**

Key findings:
- `lag_144` (same time yesterday) was the most important feature across all tree models
- Rush-hour bins (7–9 AM, 5–7 PM) had the lowest MAE — strong periodic signal
- Late-night bins (1–4 AM) had the highest MAE — sparse, noisy demand
- Gradient boosting models significantly outperformed linear models → problem is highly non-linear

---

## 🚀 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/nyc-taxi-demand-prediction.git
cd nyc-taxi-demand-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add trained models
Place these files in the `models/` folder:
```
models/
├── xgb_model.pkl
├── lgbm_model.pkl
├── rf_model.pkl
└── kmeans_zones.pkl
```

> To train models from scratch, run the notebook in Google Colab: `notebook/NY_Taxi_Demand_Prediction.ipynb`

### 5. Run the app
```bash
uvicorn app:app --reload
```

Open → `http://127.0.0.1:8000`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| GET | `/zones` | All 40 zone centers (lat/lon) |
| POST | `/predict` | Predict demand for one zone |
| POST | `/predict-all` | Predict demand for all 40 zones |

### Example request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cluster_id": 4,
    "hour_of_day": 8,
    "day_of_week": 0,
    "lag_1": 10,
    "lag_3": 9,
    "lag_6": 8,
    "lag_144": 12
  }'
```

### Example response
```json
{
  "zone_id": 4,
  "hour": 8,
  "day": 0,
  "prediction": 18.45,
  "level": "High",
  "lat": 40.7489,
  "lon": -73.9680
}
```

---

## 📁 Project Structure

```
nyc-taxi-demand-prediction/
├── main.py                  ← FastAPI backend
├── requirements.txt
├── static/
│   └── index.html          ← Frontend UI
├── models/                 ← Trained pkl files (not tracked by git)
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── rf_model.pkl
│   └── kmeans_zones.pkl
├── notebook/
│   └── NY_Taxi_Demand_Prediction.ipynb
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Dask, Pandas, NumPy |
| ML models | XGBoost, LightGBM, Scikit-learn |
| Geo-clustering | MiniBatchKMeans, gpxpy (Haversine) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, Leaflet.js |
| Visualization | Matplotlib, Seaborn, Folium |

---

## 👩‍💻 Author

Made  by **Shivangi Shukla**  
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourusername)
