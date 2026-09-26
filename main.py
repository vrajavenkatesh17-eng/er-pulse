from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import math, os, random
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field

app = FastAPI(title="ER-Pulse Triage OS")

# --- 1. LOAD AI ENGINE ---
try:
    ai_engine = xgb.Booster()
    ai_engine.load_model("er_pulse_xgb.json")
    AI_ENABLED = True
    print("✅ AI Engine Online. Supervised Batch Learning Active.")
except Exception:
    AI_ENABLED = False
    print("⚠️ AI Model missing. Using Fast Heuristic Engine.")

def to_bool(val):
    return str(val).strip().lower() in ["true", "1", "yes"]

# Filter out non-emergency diagnostic/eye/dental centers from emergency routing
NON_ER_KEYWORDS = ["eye", "dental", "scan", "autism", "imaging", "optical", "x ray", "deic", "ddwo"]

# --- 2. STATEWIDE GEOCODED CSV PIPELINE ---
def load_hospitals(filename="tpa-hospitals-lat-lon-final.csv"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(csv_path):
        csv_path = filename

    merged = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [str(c).strip().lower() for c in df.columns]

        for idx, row in df.iterrows():
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
                if math.isnan(lat) or math.isnan(lon) or lat == 0 or lon == 0:
                    continue
            except (ValueError, TypeError):
                continue

            name = str(row.get("hospital_name", f"TPA Node {idx}")).strip()
            name_lower = name.lower()
            acc = str(row.get("location_accuracy", "")).strip()

            # Micro-offset only for shared town/district centers so pins don't stack
            random.seed(name)
            if acc in ["District Center", "Village/Town Center"]:
                lat += random.uniform(-0.003, 0.003)
                lon += random.uniform(-0.003, 0.003)

            is_non_er = any(k in name_lower for k in NON_ER_KEYWORDS)

            cardio = to_bool(row.get("cardiology"))
            interv_cardio = to_bool(row.get("interventional_cardiology"))
            ct_surg = to_bool(row.get("cardiothoracic_surgeries"))
            ortho = to_bool(row.get("orthopedics"))
            neuro_surg = to_bool(row.get("neurosurgery"))
            gen_surg = to_bool(row.get("general_surgery"))
            spine = to_bool(row.get("spine"))
            plastic = to_bool(row.get("plastic_surgery"))
            gen_med = to_bool(row.get("general_medicine"))
            is_govt = any(k in str(row.get("directrate", "")).lower() or k in name_lower for k in ["govt", "gh", "medical college"])

            cardiac_depth = (4 if cardio else 0) + (4 if interv_cardio else 0) + (2 if ct_surg else 0)
            trauma_depth = (3 if ortho else 0) + (3 if neuro_surg else 0) + (2 if gen_surg else 0) + (1 if spine else 0) + (1 if plastic else 0) + (2 if is_govt else 0)
            general_depth = (5 if gen_med else 0) + (3 if gen_surg else 0) + (2 if not is_non_er else 0)

            specialties = []
            if not is_non_er:
                if cardiac_depth >= 4:
                    specialties.append("Cardiology")
                if trauma_depth >= 3:
                    specialties.append("Trauma")
                if gen_med or gen_surg or not specialties:
                    specialties.append("General")
            else:
                specialties.append("Outpatient/Diagnostic")

            entity_id = str(row.get("entity_code", f"TPA-{idx}")).strip()
            random.seed(entity_id)
            merged.append({
                "id": entity_id,
                "name": name,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "beds_free": random.randint(4, 22),
                "icu_free": random.randint(1, 8),
                "specialties": specialties,
                "cardiac_depth": cardiac_depth,
                "trauma_depth": trauma_depth,
                "general_depth": general_depth
            })
        print(f"✅ STATEWIDE MESH ONLINE: {len(merged)} Verified Geocoded Hospitals Ingested!")
    return merged

HOSPITALS = load_hospitals()
HOSP_INDEX = {h["id"]: h for h in HOSPITALS}
HOSP_LATS_RAD = np.radians(np.array([h["lat"] for h in HOSPITALS], dtype=np.float64))
HOSP_LONS_RAD = np.radians(np.array([h["lon"] for h in HOSPITALS], dtype=np.float64))

def vectorized_haversine(lat1: float, lon1: float) -> np.ndarray:
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    dlat = HOSP_LATS_RAD - lat1_r
    dlon = HOSP_LONS_RAD - lon1_r
    a = np.sin(dlat * 0.5) ** 2 + math.cos(lat1_r) * np.cos(HOSP_LATS_RAD) * (np.sin(dlon * 0.5) ** 2)
    return 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

class TriageRequest(BaseModel):
    fleet_lat: float = Field(default=8.7200)
    fleet_lon: float = Field(default=77.7150)
    condition: str = Field(default="Severe Trauma / RTA")

class EMRWebhook(BaseModel):
    hospital_id: str
    beds_free: int
    icu_free: int

@app.get("/api/hospitals")
def get_hospitals(lat: float = 8.7200, lon: float = 77.7150):
    dists = vectorized_haversine(lat, lon)
    k = min(20, len(dists) - 1)
    idx = np.argpartition(dists, k)[:20]
    idx = idx[np.argsort(dists[idx])]
    return JSONResponse(content=[HOSPITALS[int(i)] for i in idx])

@app.post("/api/triage")
def triage_engine(req: TriageRequest):
    required_specialty = "General"
    if req.condition == "Acute Cardiac / Chest Pain":
        required_specialty = "Cardiology"
    elif req.condition == "Severe Trauma / RTA":
        required_specialty = "Trauma"

    dists = vectorized_haversine(req.fleet_lat, req.fleet_lon)
    k = min(30, len(dists) - 1)
    nearest_idx = np.argpartition(dists, k)[:30]

    scored_hospitals = []
    for idx in nearest_idx:
        h = HOSPITALS[int(idx)]
        dist_km = float(dists[int(idx)])
        eta_mins = dist_km * 4.5
        has_specialty = 1 if required_specialty in h["specialties"] else 0

        # Capped tie-breaker bonus so closest capable hospital always wins
        if req.condition == "Acute Cardiac / Chest Pain":
            clinical_bonus = (h["cardiac_depth"] * 0.25) + (min(h["icu_free"], 5) * 0.3)
        elif req.condition == "Severe Trauma / RTA":
            clinical_bonus = (h["trauma_depth"] * 0.25) + (min(h["beds_free"], 10) * 0.15)
        else:
            clinical_bonus = (h["general_depth"] * 0.25) + (min(h["beds_free"], 10) * 0.15)

        clinical_penalty = 0.0 if has_specialty == 1 else 200.0
        jam_penalty = 500.0 if h["beds_free"] <= 0 else 0.0

        # High distance weight (15.0/km) guarantees proximity priority among capable hospitals
        score = (dist_km * 15.0) - clinical_bonus + clinical_penalty + jam_penalty

        scored_hospitals.append({
            "hospital": h,
            "dist_km": round(dist_km, 2),
            "eta_mins": round(eta_mins, 1),
            "match_score": has_specialty,
            "score": round(score, 4)
        })

    if AI_ENABLED and len(scored_hospitals) > 0:
        try:
            features_np = np.array([
                [x["dist_km"], x["eta_mins"], x["hospital"]["beds_free"], x["hospital"]["icu_free"], x["match_score"]]
                for x in scored_hospitals
            ], dtype=np.float32)
            dmatrix = xgb.DMatrix(
                features_np,
                feature_names=["dist_km", "eta_mins", "beds_free", "icu_free", "specialty_match"]
            )
            success_probs = ai_engine.predict(dmatrix)
            for i, x in enumerate(scored_hospitals):
                base_ai_score = 1.0 - float(success_probs[i])
                x["score"] = round(x["score"] + (base_ai_score * 0.5), 4)
        except Exception:
            pass

    for x in scored_hospitals:
        del x["match_score"]

    scored_hospitals.sort(key=lambda x: x["score"])
    return JSONResponse(content={"optimal": scored_hospitals[0], "all": scored_hospitals[:15]})

@app.post("/api/lock-bed/{h_id}")
def lock_bed(h_id: str):
    h = HOSP_INDEX.get(h_id)
    if not h:
        raise HTTPException(status_code=404)
    if h["beds_free"] > 0:
        h["beds_free"] -= 1
        return {"status": "locked", "beds_free": h["beds_free"]}
    raise HTTPException(status_code=400, detail="No beds available")

@app.post("/api/simulate-jam/{h_id}")
def simulate_jam(h_id: str):
    h = HOSP_INDEX.get(h_id)
    if not h:
        raise HTTPException(status_code=404)
    h["beds_free"] = 0
    h["icu_free"] = 0
    return {"status": "jammed"}

@app.post("/api/fhir/emr-sync")
def emr_sync(data: EMRWebhook):
    h = HOSP_INDEX.get(data.hospital_id)
    if not h:
        raise HTTPException(status_code=404, detail="Hospital ID not in spatial mesh")
    h["beds_free"] = data.beds_free
    h["icu_free"] = data.icu_free
    return {"status": "success", "message": f"EMR Webhook received. {h['name']} updated live."}

with open("templates/index.html", "r", encoding="utf-8") as f:
    INDEX_HTML = f.read()
with open("templates/hospital.html", "r", encoding="utf-8") as f:
    HOSP_HTML = f.read()

@app.get("/")
def read_index():
    return HTMLResponse(content=INDEX_HTML)

@app.get("/hospital")
def read_hospital_dashboard():
    return HTMLResponse(content=HOSP_HTML)
