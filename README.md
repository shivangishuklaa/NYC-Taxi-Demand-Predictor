# 🚕 New York City Taxi Demand Prediction

> Predicting yellow cab pickup demand per 10-minute bin per zone across New York City — trained on Jan 2015, evaluated on Jan 2016.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square)
![R2](https://img.shields.io/badge/R²-0.965-brightgreen?style=flat-square)

---

## 🌐 Live Demo
👉 [nyc-taxi-demand-predictor.streamlit.app](https://nyc-taxi-demand-predictor.onrender.com)

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
Best Model: XGBoost (R² = 0.965)
    ↓
Deployment (Streamlit + Interactive Pydeck Map)
```

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `hour_of_day` | 0–23 |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_rush_am` | 1 if 7–9 AM |
| `is_rush_pm` | 1 if 5–7 PM |
| `is_night` | 1 if 10 PM – 5 AM |
| `center_lat / center_lon` | Zone cluster center coordinates |
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
git clone https://github.com/shivangishuklaa/nyc-taxi-demand-prediction.git
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

> To train models from scratch, run the Colab notebook: `notebook/NY_Taxi_Demand_Prediction.ipynb`

### 5. Run the app
```bash
streamlit run app.py
```

Open → `http://localhost:8501`

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| **Predict this zone** | Predicts taxi pickups for a selected zone, hour, and day |
| **Predict all 40 zones** | Runs predictions for all zones simultaneously |
| **Interactive map** | Pydeck map with color-coded demand circles (green → red) |
| **Top 5 zones** | Bar-ranked list of highest demand zones |
| **Summary stats** | Total pickups, peak zone, and average per zone |
| **Lag inputs** | Manually set recent pickup counts for realistic prediction |

---

## 📁 Project Structure

```
nyc-taxi-demand-prediction/
├── app.py                          ← Streamlit app
├── requirements.txt
├── .gitignore
├── models/                         ← Trained pkl files (not tracked by git)
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
| Geo-clustering | MiniBatchKMeans |
| App framework | Streamlit |
| Map visualization | Pydeck (deck.gl) |
| Notebook | Google Colab |

---

## 👩‍💻 Author

Made by **Shivangi Shukla**  
[GitHub](https://github.com/shivangishuklaa) · [LinkedIn](https://www.linkedin.com/in/shivangi-shukla-data-analyst/)
