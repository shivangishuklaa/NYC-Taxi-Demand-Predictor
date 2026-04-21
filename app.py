import streamlit as st
import pickle
import numpy as np
import pandas as pd
import pydeck as pdk

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Demand Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F4F6F9; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #E8ECF0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #E8ECF0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }

    /* Section headers */
    .section-head {
        font-size: 11px;
        font-weight: 700;
        color: #B0B8C8;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 16px 0 8px 0;
        border-bottom: 1px solid #E8ECF0;
        padding-bottom: 6px;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg,#f0fdf8 0%,#e8faf4 100%);
        border: 1.5px solid #A8DFC9;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .result-num { font-size: 3rem; font-weight: 800; color: #1D9E75; line-height: 1; }
    .result-label { font-size: 11px; color: #7A8499; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    .result-unit  { font-size: 12px; color: #7A8499; margin-top: 4px; }

    /* Badges */
    .badge-low    { background:#EAF3DE; color:#3B6D11; padding:4px 14px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:8px; }
    .badge-medium { background:#FAEEDA; color:#854F0B; padding:4px 14px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:8px; }
    .badge-high   { background:#FCEBEB; color:#A32D2D; padding:4px 14px; border-radius:20px; font-size:13px; font-weight:700; display:inline-block; margin-top:8px; }

    /* Top zone rows */
    .zone-row { display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px solid #F0F3F7; }
    .zone-rank { background:#E1F5EE; color:#0F6E56; font-size:11px; font-weight:700;
                 border-radius:6px; width:22px; height:22px; display:inline-flex;
                 align-items:center; justify-content:center; flex-shrink:0; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/kmeans_zones.pkl", "rb") as f:
        kmeans = pickle.load(f)
    return xgb_model, kmeans.cluster_centers_

xgb_model, cluster_centers = load_models()
N_ZONES = len(cluster_centers)

DAYS      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAY_SHORT = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


# ── Feature builder (same logic as FastAPI) ───────────────────────────────────
def build_features(cluster_id, hour, dow, lag1, lag3, lag6, lag144):
    is_weekend = 1 if dow >= 5 else 0
    is_rush_am = 1 if 7  <= hour <= 9  else 0
    is_rush_pm = 1 if 17 <= hour <= 19 else 0
    is_night   = 1 if (hour >= 22 or hour <= 5) else 0
    bin_id     = 33 + (dow * 144) + (hour * 6)

    center_lat = float(cluster_centers[cluster_id][0])
    center_lon = float(cluster_centers[cluster_id][1])

    rolling_mean_3  = (lag1 + lag3) / 2
    rolling_mean_12 = (lag1 + lag3 + lag6 + lag144) / 4

    return pd.DataFrame([{
        "cluster_id":      cluster_id,
        "bin_id":          bin_id,
        "hour_of_day":     hour,
        "day_of_week":     dow,
        "is_weekend":      is_weekend,
        "is_rush_am":      is_rush_am,
        "is_rush_pm":      is_rush_pm,
        "is_night":        is_night,
        "center_lat":      center_lat,
        "center_lon":      center_lon,
        "lag_1":           lag1,
        "lag_3":           lag3,
        "lag_6":           lag6,
        "lag_144":         lag144,
        "rolling_mean_3":  rolling_mean_3,
        "rolling_mean_12": rolling_mean_12,
    }])


def predict_zone(cluster_id, hour, dow, lag1, lag3, lag6, lag144):
    feat = build_features(cluster_id, hour, dow, lag1, lag3, lag6, lag144)
    pred = max(0.0, float(xgb_model.predict(feat)[0]))
    pred = round(pred, 2)
    level = "Low" if pred < 5 else ("Medium" if pred < 15 else "High")
    return pred, level


def predict_all_zones(hour, dow, lag1, lag3, lag6, lag144):
    results = []
    for i in range(N_ZONES):
        feat = build_features(i, hour, dow, lag1, lag3, lag6, lag144)
        pred = max(0.0, float(xgb_model.predict(feat)[0]))
        results.append({
            "zone_id":    i,
            "prediction": round(pred, 2),
            "lat":        float(cluster_centers[i][0]),
            "lon":        float(cluster_centers[i][1]),
        })
    results.sort(key=lambda x: x["prediction"], reverse=True)
    return results


# ── Helper: time label ────────────────────────────────────────────────────────
def time_label(hour):
    if 7 <= hour <= 9:   return "🌅 AM Rush Hour"
    if 17 <= hour <= 19: return "🌆 PM Rush Hour"
    if hour >= 22 or hour <= 5: return "🌙 Late Night"
    return "☀️ Daytime"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚕 NYC Taxi Demand")
    st.caption("XGBoost · R² 0.965 · 40 Zones")
    st.divider()

    # Time
    st.markdown('<div class="section-head">⏰ Time</div>', unsafe_allow_html=True)
    hour = st.slider("Hour of day", 0, 23, 8, format="%d:00")
    st.caption(time_label(hour))
    dow = st.selectbox("Day of week", range(7), format_func=lambda x: DAYS[x])

    # Zone
    st.markdown('<div class="section-head">📍 Zone</div>', unsafe_allow_html=True)
    cluster_id = st.slider("Cluster ID", 0, N_ZONES - 1, 0)
    clat = round(float(cluster_centers[cluster_id][0]), 4)
    clon = round(float(cluster_centers[cluster_id][1]), 4)
    st.caption(f"Lat {clat} · Lon {clon}")

    # Lag values
    st.markdown('<div class="section-head">📊 Lag Values</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        lag1 = st.number_input("10 min ago",  min_value=0.0, value=10.0, step=1.0, key="l1")
        lag6 = st.number_input("1 hr ago",    min_value=0.0, value=8.0,  step=1.0, key="l6")
    with c2:
        lag3   = st.number_input("30 min ago",  min_value=0.0, value=9.0,  step=1.0, key="l3")
        lag144 = st.number_input("Yesterday",   min_value=0.0, value=12.0, step=1.0, key="l144")

    st.divider()

    # Buttons
    btn_single = st.button("▶ Predict this zone", use_container_width=True, type="primary")
    btn_all    = st.button("🌐 Predict all 40 zones", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## NYC Taxi Demand Predictor")
st.caption(f"Trained on Jan 2015 · Evaluated on Jan 2016 · Showing: **{DAYS[dow]}**, **{hour:02d}:00**")

# ── Session state ─────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = None   # None | "single" | "all"
if "result" not in st.session_state:
    st.session_state.result = None
if "all_results" not in st.session_state:
    st.session_state.all_results = None

if btn_single:
    st.session_state.mode = "single"
    with st.spinner("Predicting..."):
        pred, level = predict_zone(cluster_id, hour, dow, lag1, lag3, lag6, lag144)
    st.session_state.result = dict(pred=pred, level=level, zone=cluster_id,
                                   lat=clat, lon=clon, hour=hour, dow=dow)

if btn_all:
    st.session_state.mode = "all"
    with st.spinner("Predicting all 40 zones..."):
        st.session_state.all_results = predict_all_zones(hour, dow, lag1, lag3, lag6, lag144)


# ── Two-column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="medium")

# ── LEFT: results panel ───────────────────────────────────────────────────────
with col_left:
    if st.session_state.mode == "single" and st.session_state.result:
        r = st.session_state.result
        badge_cls = f"badge-{r['level'].lower()}"
        rush = "AM Rush" if 7 <= r["hour"] <= 9 else ("PM Rush" if 17 <= r["hour"] <= 19 else "—")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Pickups</div>
            <div class="result-num">{r['pred']}</div>
            <div class="result-unit">per 10-min bin · Zone {r['zone']}</div>
            <div class="{badge_cls}">{'🟢' if r['level']=='Low' else '🟡' if r['level']=='Medium' else '🔴'} {r['level']} demand</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Hour",      f"{r['hour']:02d}:00")
        m2.metric("Day",       DAY_SHORT[r["dow"]])
        m3, m4 = st.columns(2)
        m3.metric("Rush hour", rush)
        m4.metric("Demand",    r["level"])

    elif st.session_state.mode == "all" and st.session_state.all_results:
        zones = st.session_state.all_results
        total = sum(z["prediction"] for z in zones)
        avg   = total / len(zones)
        peak  = zones[0]

        st.markdown("#### 📊 Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Pickups", f"{total:.0f}")
        c2.metric("Peak Zone",     f"#{peak['zone_id']}")
        c3.metric("Avg / Zone",    f"{avg:.1f}")

        st.markdown("#### 🏆 Top 5 Zones")
        max_p = peak["prediction"]
        for i, z in enumerate(zones[:5]):
            pct  = int(z["prediction"] / max_p * 100) if max_p > 0 else 0
            st.markdown(f"""
            <div class="zone-row">
                <span class="zone-rank">{i+1}</span>
                <span style="font-size:12px;color:#7A8499;min-width:56px">Zone {z['zone_id']}</span>
                <div style="flex:1;background:#EEF1F6;border-radius:4px;height:6px;overflow:hidden">
                    <div style="width:{pct}%;background:#1D9E75;height:100%;border-radius:4px"></div>
                </div>
                <span style="font-size:12px;font-weight:700;min-width:36px;text-align:right">{z['prediction']}</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👈 Set parameters in the sidebar and click **Predict this zone** or **Predict all 40 zones**.")


# ── RIGHT: map ────────────────────────────────────────────────────────────────
with col_right:
    # Build map data
    if st.session_state.mode == "all" and st.session_state.all_results:
        zones = st.session_state.all_results
        max_p = zones[0]["prediction"] if zones else 1

        map_df = pd.DataFrame([{
            "lat":        z["lat"],
            "lon":        z["lon"],
            "zone_id":    z["zone_id"],
            "prediction": z["prediction"],
            "radius":     int(200 + (z["prediction"] / max_p) * 500),
            # Color: green→red based on demand norm
            "r": int(30  + (z["prediction"] / max_p) * 220),
            "g": int(158 * (1 - z["prediction"] / max_p) + 30),
            "b": 60,
        } for z in zones])

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_fill_color=["r", "g", "b", 200],
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {"html": "<b>Zone {zone_id}</b><br/>Pickups: {prediction}", "style": {"fontSize": "13px"}}

    elif st.session_state.mode == "single" and st.session_state.result:
        r = st.session_state.result
        # All zones in blue, selected in green
        base_df = pd.DataFrame([{
            "lat":     float(cluster_centers[i][0]),
            "lon":     float(cluster_centers[i][1]),
            "zone_id": i,
            "r": 55, "g": 138, "b": 221,
            "radius": 150,
        } for i in range(N_ZONES)])

        highlight_df = pd.DataFrame([{
            "lat":        r["lat"],
            "lon":        r["lon"],
            "zone_id":    r["zone"],
            "prediction": r["pred"],
            "r": 29, "g": 158, "b": 117,
            "radius": 350,
        }])

        layer = [
            pdk.Layer("ScatterplotLayer", data=base_df,
                      get_position=["lon","lat"], get_fill_color=["r","g","b",140],
                      get_radius="radius", pickable=True),
            pdk.Layer("ScatterplotLayer", data=highlight_df,
                      get_position=["lon","lat"], get_fill_color=["r","g","b",240],
                      get_radius="radius", pickable=True),
        ]
        tooltip = {"html": "<b>Zone {zone_id}</b><br/>Pickups: {prediction}", "style": {"fontSize": "13px"}}

    else:
        # Default: all zones in blue
        base_df = pd.DataFrame([{
            "lat":     float(cluster_centers[i][0]),
            "lon":     float(cluster_centers[i][1]),
            "zone_id": i,
            "r": 55, "g": 138, "b": 221,
            "radius": 150,
        } for i in range(N_ZONES)])
        layer   = pdk.Layer("ScatterplotLayer", data=base_df,
                            get_position=["lon","lat"], get_fill_color=["r","g","b",150],
                            get_radius="radius", pickable=True)
        tooltip = {"html": "<b>Zone {zone_id}</b>", "style": {"fontSize": "13px"}}

    view = pdk.ViewState(latitude=40.7346, longitude=-73.9904, zoom=10.5, pitch=0)
    st.pydeck_chart(pdk.Deck(
        layers=[layer] if not isinstance(layer, list) else layer,
        initial_view_state=view,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    ), use_container_width=True, height=520)
