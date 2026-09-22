import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# --- HIDDEN DOM WATERMARK (Inspect Element -> HTML source-la mattum irukum) ---
st.markdown("<!-- WATERMARK: DEVELOPED_AND_OWNED_BY_KRISH_2026 -->", unsafe_allow_html=True)

st.set_page_config(page_title="ER-Pulse | Smart Triage", page_icon="🚑", layout="wide")

st.title("🚑 ER-Pulse: Real-GPS Dynamic Emergency Triage")
st.caption("Smart Urban Corridor Routing, Live Bed Lock & Pre-Alert Engine")
st.divider()

# Real Tirunelveli Hospital GPS Coordinates
if "tvl_hospitals" not in st.session_state:
    st.session_state.tvl_hospitals = pd.DataFrame([
        {"id": "H1", "name": "TVMC (High Grounds Govt Hospital)", "lat": 8.7139, "lon": 77.7275, "beds_free": 12, "icu_free": 4, "specialties": "Trauma,Burns,General,Cardiac"},
        {"id": "H2", "name": "Shifa Multi-Specialty Hospital", "lat": 8.7280, "lon": 77.7120, "beds_free": 8, "icu_free": 3, "specialties": "Cardiology,Oncology,Ortho"},
        {"id": "H3", "name": "Galaxy Hospitals (24/7 ER)", "lat": 8.7350, "lon": 77.7010, "beds_free": 6, "icu_free": 2, "specialties": "Cardiology,Trauma,Emergency"},
        {"id": "H4", "name": "CSI Mission Hospital", "lat": 8.7090, "lon": 77.7400, "beds_free": 5, "icu_free": 1, "specialties": "General,Trauma"},
        {"id": "H5", "name": "Rosemary Mission Hospital", "lat": 8.7240, "lon": 77.7180, "beds_free": 4, "icu_free": 1, "specialties": "General,Emergency"}
    ])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

st.sidebar.header("🚨 Live Emergency Dispatch (GPS)")
patient_id = st.sidebar.text_input("Patient ID", value="ER-CASE-904")
symptom_cat = st.sidebar.selectbox("Emergency Category", [
    "Acute Cardiac / Chest Pain", 
    "High-Velocity Bypass Trauma / Accident", 
    "Severe Burns / Critical Respiratory", 
    "General Emergency / Stroke"
])

amb_lat = st.sidebar.slider("Ambulance Latitude", 8.7050, 8.7400, 8.7200, format="%.4f")
amb_lon = st.sidebar.slider("Ambulance Longitude", 77.6950, 77.7450, 77.7150, format="%.4f")

req_spec = "Cardiology" if "Cardiac" in symptom_cat or "Stroke" in symptom_cat else ("Trauma" if "Trauma" in symptom_cat or "Accident" in symptom_cat else "General")

h_df = st.session_state.tvl_hospitals.copy()
h_df["distance_km"] = haversine(amb_lat, amb_lon, h_df["lat"], h_df["lon"])
h_df["eta_mins"] = (h_df["distance_km"] * 4.5).round(1)
h_df["match_score"] = h_df["specialties"].apply(lambda s: 1.0 if req_spec in s else 0.4)
h_df["urgency_rank_score"] = (h_df["eta_mins"] * 1.5) - (h_df["beds_free"] * 0.9) - (h_df["match_score"] * 6.0)

best_hosp = h_df.sort_values(by="urgency_rank_score").iloc[0]

col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    st.subheader("🎯 AI Optimal Triage Decision")
    st.error(f"**Recommended Destination**: {best_hosp['name']}")
    st.caption(f"Required Specialty Filter: **{req_spec}**")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Live ETA", f"{best_hosp['eta_mins']} mins")
    m2.metric("Free ER Beds", f"{best_hosp['beds_free']}")
    m3.metric("ICU Buffer", f"{best_hosp['icu_free']}")
    
    st.success(f"⚡ **Auto-Pre-Alert Triggered**: *Sent to {best_hosp['name']} ER team (Bay reserved for {patient_id}).*")
    
    if st.button("Confirm Handover & Lock Bed", type="primary"):
        idx = st.session_state.tvl_hospitals[st.session_state.tvl_hospitals["id"] == best_hosp["id"]].index[0]
        st.session_state.tvl_hospitals.loc[idx, "beds_free"] = max(0, st.session_state.tvl_hospitals.loc[idx, "beds_free"] - 1)
        st.toast(f"Bed locked at {best_hosp['name']}!", icon="✅")
        st.rerun()
        
    st.divider()
    if st.button(f"Simulate Sudden ER Jam at {best_hosp['id']}", type="secondary"):
        idx = st.session_state.tvl_hospitals[st.session_state.tvl_hospitals["id"] == best_hosp["id"]].index[0]
        st.session_state.tvl_hospitals.loc[idx, "beds_free"] = 0
        st.warning("Capacity dropped to 0! Watch map & score update.")
        st.rerun()

with col_right:
    st.subheader("🗺️ Real OpenStreetMap (Live Corridor View)")
    
    m = folium.Map(location=[8.7200, 77.7180], zoom_start=13, tiles="OpenStreetMap")
    
    for _, row in h_df.iterrows():
        is_best = (row["id"] == best_hosp["id"])
        color = "green" if is_best else "blue"
        popup_html = f"<b>{row['name']}</b><br>Free Beds: {row['beds_free']}<br>ETA: {row['eta_mins']}m"
        folium.Marker(location=[row["lat"], row["lon"]], popup=popup_html, tooltip=row["name"], icon=folium.Icon(color=color, icon="plus-sign" if is_best else "info-sign")).add_to(m)
        
    folium.Marker(location=[amb_lat, amb_lon], popup=f"<b>Ambulance ({patient_id})</b>", tooltip="Ambulance Live GPS", icon=folium.Icon(color="red", icon="ambulance", prefix="fa")).add_to(m)
    folium.PolyLine(locations=[[amb_lat, amb_lon], [best_hosp["lat"], best_hosp["lon"]]], color="red", weight=3, dash_array="5, 5", tooltip=f"AI Route to {best_hosp['name']}").add_to(m)
    st_folium(m, height=400, use_container_width=True)
    st.subheader("🏥 Live Corridor Hospital Status Grid")
    st.dataframe(st.session_state.tvl_hospitals[["id", "name", "beds_free", "icu_free", "specialties"]], use_container_width=True, hide_index=True)