from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import os, json

app = FastAPI(title="ER-Pulse Enterprise Health Core API", version="2.6.0")

HOSPITALS = [
    {"id": "H1", "name": "TVMC (High Grounds Govt Hospital)", "lat": 8.7139, "lon": 77.7275, "beds_free": 12, "icu_free": 4, "specialties": "Trauma,Burns,General,Cardiac"},
    {"id": "H2", "name": "Shifa Multi-Specialty Hospital", "lat": 8.7280, "lon": 77.7120, "beds_free": 8, "icu_free": 3, "specialties": "Cardiology,Oncology,Ortho"},
    {"id": "H3", "name": "Galaxy Hospitals (24/7 ER)", "lat": 8.7350, "lon": 77.7010, "beds_free": 6, "icu_free": 2, "specialties": "Cardiology,Trauma,Emergency"},
    {"id": "H4", "name": "CSI Mission Hospital", "lat": 8.7090, "lon": 77.7400, "beds_free": 5, "icu_free": 1, "specialties": "General,Trauma"},
    {"id": "H5", "name": "Rosemary Mission Hospital", "lat": 8.7240, "lon": 77.7180, "beds_free": 4, "icu_free": 1, "specialties": "General,Emergency"}
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)

class TriagePayload(BaseModel):
    patient_id: str
    symptom_cat: str
    amb_lat: float
    amb_lon: float

@app.get("/api/hospitals")
def get_hospitals():
    return {"status": "ok", "count": len(HOSPITALS), "hospitals": HOSPITALS}

@app.post("/api/triage")
def evaluate_triage(req: TriagePayload):
    req_spec = "Cardiology" if "Cardiac" in req.symptom_cat or "Stroke" in req.symptom_cat else ("Trauma" if "Trauma" in req.symptom_cat or "Accident" in req.symptom_cat else "General")
    scored = []
    for h in HOSPITALS:
        dist = haversine(req.amb_lat, req.amb_lon, h["lat"], h["lon"])
        eta = round(dist * 4.5, 1)
        match = 1.0 if req_spec in h["specialties"] else 0.4
        score = round((eta * 1.5) - (h["beds_free"] * 0.9) - (match * 6.0), 2)
        scored.append({**h, "distance_km": round(dist, 2), "eta_mins": eta, "match_score": match, "urgency_score": score})
    scored.sort(key=lambda x: x["urgency_score"])
    return {
        "status": "computed",
        "recommended": scored[0],
        "ranked_ledger": scored,
        "required_specialty": req_spec
    }

@app.post("/api/lock-bed/{h_id}")
def lock_bed(h_id: str):
    for h in HOSPITALS:
        if h["id"] == h_id:
            h["beds_free"] = max(0, h["beds_free"] - 1)
            return {"status": "locked", "hospital": h}
    raise HTTPException(status_code=404, detail="Hospital ID not found")

@app.post("/api/simulate-jam/{h_id}")
def simulate_jam(h_id: str):
    for h in HOSPITALS:
        if h["id"] == h_id:
            h["beds_free"] = 0
            return {"status": "jam_injected", "hospital": h}
    raise HTTPException(status_code=404, detail="Hospital ID not found")

@app.get("/api/verify-watermark")
def verify_ownership():
    path = ".krish_watermark.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {"verified": True, "data": json.load(f)}
    return {"verified": False}

@app.get("/", response_class=HTMLResponse)
def serve_portal():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
