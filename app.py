import streamlit as st
import pickle
import os
import gdown
import numpy as np
import pandas as pd
import pydeck as pdk

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Demand Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; }

[data-testid="stSidebar"] {
    background: #16181f !important;
    border-right: 1px solid #2a2d3a;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e2130 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input {
    background: #1e2130 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}

.sec-label {
    font-size: 10px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #4a5568 !important;
    margin: 18px 0 6px 0; padding-bottom: 6px;
    border-bottom: 1px solid #2a2d3a;
}

.sidebar-logo { display:flex; align-items:center; gap:10px; padding:4px 0 12px 0; }
.sidebar-logo-icon {
    font-size:24px; background:linear-gradient(135deg,#1D9E75,#0d6e50);
    border-radius:10px; width:40px; height:40px;
    display:flex; align-items:center; justify-content:center;
}
.sidebar-logo-text { font-size:15px; font-weight:700; color:#f1f5f9 !important; }
.sidebar-logo-sub  { font-size:11px; color:#64748b !important; margin-top:1px; }

.badge-row { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:4px; }
.pill { font-size:10px; font-weight:700; padding:3px 9px; border-radius:20px; background:#1a2e26; color:#1D9E75; border:1px solid #1D9E75; }
.pill.blue { background:#1a2233; color:#60a5fa; border-color:#60a5fa; }

[data-testid="stSidebar"] .stButton > button {
    border-radius:10px !important; font-weight:700 !important;
    font-size:13px !important; padding:10px 0 !important;
    transition: all .15s !important;
}
.stButton.primary-btn > button {
    background: linear-gradient(135deg,#1D9E75,#0d6e50) !important;
    color:#fff !important; border:none !important;
    box-shadow:0 4px 14px rgba(29,158,117,.35) !important;
}

.main-title {
    font-size: clamp(20px,3vw,28px); font-weight:800;
    color:#f1f5f9; letter-spacing:-.02em; margin:0 0 2px 0;
}
.main-sub { font-size:12px; color:#64748b; margin-bottom:16px; }

.result-card {
    background:linear-gradient(135deg,#0d2e22 0%,#0a1f1a 100%);
    border:1.5px solid #1D9E75; border-radius:16px;
    padding:24px 20px; text-align:center; margin-bottom:14px;
    position:relative; overflow:hidden;
}
.result-card::before {
    content:''; position:absolute; top:-30px; right:-30px;
    width:100px; height:100px; background:rgba(29,158,117,.1); border-radius:50%;
}
.result-label { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:#4ade80; margin-bottom:6px; }
.result-num   { font-family:'DM Mono',monospace; font-size:clamp(2.2rem,5vw,3.4rem); font-weight:500; color:#1D9E75; line-height:1; margin-bottom:4px; }
.result-unit  { font-size:12px; color:#64748b; }

.badge-low    { background:#0d2a0a; color:#4ade80; padding:5px 16px; border-radius:20px; font-size:12px; font-weight:700; display:inline-flex; align-items:center; gap:6px; margin-top:10px; border:1px solid #166534; }
.badge-medium { background:#2a1a0a; color:#fbbf24; padding:5px 16px; border-radius:20px; font-size:12px; font-weight:700; display:inline-flex; align-items:center; gap:6px; margin-top:10px; border:1px solid #92400e; }
.badge-high   { background:#2a0a0a; color:#f87171; padding:5px 16px; border-radius:20px; font-size:12px; font-weight:700; display:inline-flex; align-items:center; gap:6px; margin-top:10px; border:1px solid #991b1b; }

[data-testid="stMetric"] {
    background:#16181f !important; border:1px solid #2a2d3a !important;
    border-radius:12px !important; padding:14px 16px !important;
}
[data-testid="stMetricLabel"] { color:#64748b !important; font-size:11px !important; }
[data-testid="stMetricValue"] { color:#f1f5f9 !important; font-size:1.3rem !important; font-weight:700 !important; }

.zone-row { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #1e2130; }
.zone-rank {
    background:#0d2e22; color:#1D9E75; font-size:11px; font-weight:700;
    border-radius:7px; width:24px; height:24px;
    display:inline-flex; align-items:center; justify-content:center;
    flex-shrink:0; border:1px solid #1D9E75;
}
.section-heading { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#475569; margin:12px 0 8px 0; }

.info-box {
    background:#16181f; border:1px solid #2a2d3a; border-radius:12px;
    padding:28px 20px; text-align:center; color:#64748b; font-size:13px; line-height:1.8;
}
.info-box span { font-size:32px; display:block; margin-bottom:10px; }

.time-tag { font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; display:inline-block; margin-top:4px; }
.time-rush  { background:#2a1a0a; color:#fbbf24; border:1px solid #92400e; }
.time-night { background:#0f1626; color:#818cf8; border:1px solid #3730a3; }
.time-day   { background:#0d2e22; color:#4ade80; border:1px solid #166534; }

.zone-coord { font-family:'DM Mono',monospace; font-size:11px; color:#475569; text-align:center; margin-top:2px; }

#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }

@media (max-width:768px) {
    .main-title { font-size:18px; }
    .result-num { font-size:2rem; }
    [data-testid="stMetricValue"] { font-size:1rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (auto-download from Google Drive if not present) ────────────


GDRIVE_FILES = {
    "models/xgb_model.pkl":    "1ppSv3GWX1Ha2WMi5mXmn7Xnx_qufogUc",
    "models/kmeans_zones.pkl": "18vYxo6eCoi7YChqWX-ZXxkWE8KoX8Wbo",
}

def download_models(timeout=60):
    """Download model files from Google Drive if not already present.
    
    Args:
        timeout: Download timeout in seconds (default: 60)
    """
    os.makedirs("models", exist_ok=True)
    for local_path, file_id in GDRIVE_FILES.items():
        if not os.path.exists(local_path):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                with st.spinner(f"⬇️ Downloading {os.path.basename(local_path)}..."):
                    gdown.download(
                        url, 
                        local_path, 
                        quiet=False,
                        timeout=timeout
                    )
                st.success(f"✅ Downloaded {os.path.basename(local_path)}")
            except Exception as e:
                st.error(f"❌ Failed to download {os.path.basename(local_path)}: {str(e)}")
                st.error("Please check your internet connection and try refreshing the page.")
                st.stop()

@st.cache_resource
def load_models():
    """Load models from pickle files."""
    try:
        download_models()
        with open("models/xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        with open("models/kmeans_zones.pkl", "rb") as f:
            kmeans = pickle.load(f)
        return xgb_model, kmeans.cluster_centers_
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Initialize session state
if "mode" not in st.session_state:
    st.session_state.mode = None
if "result" not in st.session_state:
    st.session_state.result = None
if "all_results" not in st.session_state:
    st.session_state.all_results = None

xgb_model, cluster_centers = load_models()
N_ZONES   = len(cluster_centers)
DAYS      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAY_SHORT = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(cluster_id, hour, dow, lag1, lag3, lag6, lag144):
    is_weekend      = 1 if dow >= 5 else 0
    is_rush_am      = 1 if 7 <= hour <= 9 else 0
    is_rush_pm      = 1 if 17 <= hour <= 19 else 0
    is_night        = 1 if hour < 6 or hour >= 22 else 0
    is_business_day = 1 if dow < 5 else 0
    hour_sin        = np.sin(2 * np.pi * hour / 24)
    hour_cos        = np.cos(2 * np.pi * hour / 24)
    dow_sin         = np.sin(2 * np.pi * dow / 7)
    dow_cos         = np.cos(2 * np.pi * dow / 7)

    return np.array([
        cluster_id,      lag1,       lag3,           lag6,        lag144,
        is_weekend,      is_rush_am, is_rush_pm,     is_night,    is_business_day,
        hour_sin,        hour_cos,   dow_sin,        dow_cos,     hour, dow
    ]).reshape(1, -1)

def predict_zone(cluster_id, hour, dow, lag1, lag3, lag6, lag144):
    X = build_features(cluster_id, hour, dow, lag1, lag3, lag6, lag144)
    pred = max(0, round(xgb_model.predict(X)[0]))
    if pred < 5:
        level = "Low"
    elif pred < 15:
        level = "Medium"
    else:
        level = "High"
    return pred, level

def predict_all_zones(hour, dow, lag1, lag3, lag6, lag144):
    results = []
    for zone_id in range(N_ZONES):
        X = build_features(zone_id, hour, dow, lag1, lag3, lag6, lag144)
        pred = max(0, round(xgb_model.predict(X)[0]))
        results.append({
            "zone_id": zone_id,
            "prediction": pred,
            "lat": float(cluster_centers[zone_id][0]),
            "lon": float(cluster_centers[zone_id][1]),
        })
    return sorted(results, key=lambda x: x["prediction"], reverse=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-label">⏰ Time</div>', unsafe_allow_html=True)
    hour = st.slider("Hour of Day", 0, 23, 9, label_visibility="collapsed")
    dow = st.slider("Day of Week", 0, 6, 0, format_option=lambda x: DAYS[x], label_visibility="collapsed")

    st.markdown('<div class="sec-label">📍 Zone</div>', unsafe_allow_html=True)
    cluster_id = st.slider("Cluster ID", 0, N_ZONES - 1, 0, label_visibility="collapsed")
    clat = round(float(cluster_centers[cluster_id][0]), 4)
    clon = round(float(cluster_centers[cluster_id][1]), 4)
    st.markdown(f'<div class="zone-coord">Zone {cluster_id} &nbsp;·&nbsp; {clat}, {clon}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">📊 Lag Values (pickups)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        lag1 = st.number_input("10 min ago", min_value=0.0, value=10.0, step=1.0, key="l1")
        lag6 = st.number_input("1 hr ago",   min_value=0.0, value=8.0,  step=1.0, key="l6")
    with c2:
        lag3   = st.number_input("30 min ago", min_value=0.0, value=9.0,  step=1.0, key="l3")
        lag144 = st.number_input("Yesterday",  min_value=0.0, value=12.0, step=1.0, key="l144")

    st.divider()
    btn_single = st.button("▶  Predict this zone",    use_container_width=True, type="primary")
    btn_all    = st.button("🌐  Predict all 40 zones", use_container_width=True)


# ── Trigger ───────────────────────────────────────────────────────────────────
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


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🚕 NYC Taxi Demand Predictor</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="main-sub">Trained Jan 2015 · Evaluated Jan 2016 · '
    f'<b style="color:#94a3b8">{DAYS[dow]}</b> · <b style="color:#94a3b8">{hour:02d}:00</b></p>',
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1, 2], gap="medium")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with col_left:
    if st.session_state.mode == "single" and st.session_state.result:
        r         = st.session_state.result
        badge_cls = f"badge-{r['level'].lower()}"
        dot       = "🟢" if r["level"] == "Low" else ("🟡" if r["level"] == "Medium" else "🔴")
        rush      = "AM Rush" if 7 <= r["hour"] <= 9 else ("PM Rush" if 17 <= r["hour"] <= 19 else "—")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Pickups</div>
            <div class="result-num">{r['pred']}</div>
            <div class="result-unit">per 10-min bin &nbsp;·&nbsp; Zone {r['zone']}</div>
            <div class="{badge_cls}">{dot} {r['level']} Demand</div>
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

        st.markdown('<div class="section-heading">📊 Summary</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total",    f"{total:.0f}")
        c2.metric("Peak",     f"Zone {peak['zone_id']}")
        c3.metric("Avg/Zone", f"{avg:.1f}")

        st.markdown('<div class="section-heading" style="margin-top:16px">🏆 Top 5 Zones</div>', unsafe_allow_html=True)
        max_p = peak["prediction"]
        for i, z in enumerate(zones[:5]):
            pct = int(z["prediction"] / max_p * 100) if max_p > 0 else 0
            st.markdown(f"""
            <div class="zone-row">
                <span class="zone-rank">{i+1}</span>
                <span style="font-size:12px;color:#94a3b8;min-width:60px;font-weight:600">Zone {z['zone_id']}</span>
                <div style="flex:1;background:#1e2130;border-radius:4px;height:5px;overflow:hidden">
                    <div style="width:{pct}%;background:linear-gradient(90deg,#1D9E75,#4ade80);height:100%;border-radius:4px"></div>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:12px;color:#f1f5f9;min-width:38px;text-align:right">{z['prediction']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <span>🗺️</span>
            Set your parameters in the sidebar,<br>
            then click <b>Predict this zone</b><br>or <b>Predict all 40 zones</b>.
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT: Map ────────────────────────────────────────────────────────────────
with col_right:
    if st.session_state.mode == "all" and st.session_state.all_results:
        zones = st.session_state.all_results
        max_p = zones[0]["prediction"] if zones else 1
        map_df = pd.DataFrame([{
            "lat": z["lat"], "lon": z["lon"],
            "zone_id": z["zone_id"],
            "prediction": z["prediction"],
            "label": f"Zone {z['zone_id']} — {z['prediction']} pickups",
            "radius": int(150 + (z["prediction"] / max_p) * 420),
            "r": int(29  + (z["prediction"] / max_p) * 220),
            "g": int(158 * (1 - z["prediction"] / max_p) + 30),
            "b": 80,
        } for z in zones])
        layer   = pdk.Layer("ScatterplotLayer", data=map_df,
                            get_position=["lon","lat"],
                            get_fill_color=["r","g","b", 210],
                            get_radius="radius", pickable=True, auto_highlight=True)
        tooltip = {"html": "<b>{label}</b>",
                   "style": {"background":"#16181f","color":"#f1f5f9",
                             "fontSize":"13px","borderRadius":"8px","padding":"6px 10px"}}

    elif st.session_state.mode == "single" and st.session_state.result:
        r = st.session_state.result
        base_df = pd.DataFrame([{
            "lat": float(cluster_centers[i][0]), "lon": float(cluster_centers[i][1]),
            "zone_id": i, "label": f"Zone {i}",
            "r":55,"g":100,"b":200,"radius":130,
        } for i in range(N_ZONES)])
        hi_df = pd.DataFrame([{
            "lat": r["lat"], "lon": r["lon"], "zone_id": r["zone"],
            "label": f"Zone {r['zone']} — {r['pred']} pickups",
            "r":29,"g":200,"b":117,"radius":360,
        }])
        layer = [
            pdk.Layer("ScatterplotLayer", data=base_df,
                      get_position=["lon","lat"], get_fill_color=["r","g","b",120],
                      get_radius="radius", pickable=True),
            pdk.Layer("ScatterplotLayer", data=hi_df,
                      get_position=["lon","lat"], get_fill_color=["r","g","b",240],
                      get_radius="radius", pickable=True),
        ]
        tooltip = {"html": "<b>{label}</b>",
                   "style": {"background":"#16181f","color":"#f1f5f9",
                             "fontSize":"13px","borderRadius":"8px","padding":"6px 10px"}}
    else:
        base_df = pd.DataFrame([{
            "lat": float(cluster_centers[i][0]), "lon": float(cluster_centers[i][1]),
            "zone_id": i, "label": f"Zone {i}",
            "r":55,"g":100,"b":200,"radius":130,
        } for i in range(N_ZONES)])
        layer   = pdk.Layer("ScatterplotLayer", data=base_df,
                            get_position=["lon","lat"], get_fill_color=["r","g","b",150],
                            get_radius="radius", pickable=True)
        tooltip = {"html": "<b>{label}</b>",
                   "style": {"background":"#16181f","color":"#f1f5f9",
                             "fontSize":"13px","borderRadius":"8px","padding":"6px 10px"}}

    # ── Proper NYC zoom ──
    view = pdk.ViewState(latitude=40.7128, longitude=-73.9960, zoom=11, pitch=0, bearing=0)

    st.pydeck_chart(pdk.Deck(
        layers=[layer] if not isinstance(layer, list) else layer,
        initial_view_state=view,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    ), use_container_width=True, height=540)
