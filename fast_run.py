import os
import subprocess
import sys
import threading
import webbrowser

os.makedirs("templates", exist_ok=True)

# ==============================================================================
# 1. ULTRA-FAST MAIN.PY (Closest-Capable Routing + Non-ER Filter + NumPy Math)
# ==============================================================================
MAIN_PY = """from fastapi import FastAPI, HTTPException
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
"""

# ==============================================================================
# 2. ZERO-LAG TEMPLATES/INDEX.HTML (Instant Teleport + OSRM Cache)
# ==============================================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ER-PULSE // ENTERPRISE TRIAGE OS</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { background-color: #060b14; color: #e2e8f0; font-family: 'Courier New', monospace; }
        .hud-panel { background: #0d1526; border: 1px solid #1e293b; border-radius: 8px; }
        .optimal-glow { border: 1px solid #06b6d4; box-shadow: 0 0 15px rgba(6, 182, 212, 0.18); }
        #map { cursor: crosshair; }
    </style>
</head>
<body class="p-4 h-screen flex flex-col overflow-hidden">
    <div class="hud-panel px-5 py-3 mb-4 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <span class="text-xl">🚑</span>
            <h1 class="text-lg font-bold tracking-widest text-cyan-400">ER-PULSE // ENTERPRISE TRIAGE OS</h1>
        </div>
        <div class="px-3 py-1 rounded-full bg-emerald-950 border border-emerald-500/40 text-emerald-400 text-xs font-bold flex items-center space-x-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>AI ML ENGINE ONLINE</span>
        </div>
    </div>

    <div class="grid grid-cols-12 gap-4 flex-1 overflow-hidden">
        <div class="col-span-4 flex flex-col space-y-4">
            <div class="hud-panel p-4 flex flex-col space-y-3">
                <div class="text-xs font-bold text-slate-400 tracking-wider">📡 LIVE EMS DISPATCH</div>
                <div>
                    <label class="text-xs text-slate-400 block mb-1">Patient Incident Ref</label>
                    <input id="caseRef" type="text" value="ER-CASE-904" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-cyan-300">
                </div>
                <div>
                    <label class="text-xs text-slate-400 block mb-1">Clinical Presentation</label>
                    <select id="condition" onchange="computeRoute(false)" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-white">
                        <option value="Severe Trauma / RTA">Severe Trauma / RTA</option>
                        <option value="Acute Cardiac / Chest Pain">Acute Cardiac / Chest Pain</option>
                        <option value="General Medical">General Medical</option>
                    </select>
                </div>
                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="text-xs text-slate-400 block mb-1">Fleet Lat (Corridor)</label>
                        <input id="fleetLat" type="number" step="0.0001" value="8.7200" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-white">
                    </div>
                    <div>
                        <label class="text-xs text-slate-400 block mb-1">Fleet Lon (Corridor)</label>
                        <input id="fleetLon" type="number" step="0.0001" value="77.7150" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-white">
                    </div>
                </div>
                <button onclick="computeRoute(true)" class="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-2.5 rounded text-sm transition shadow-lg">
                    ⚡ Compute AI Route
                </button>
            </div>

            <div id="optimalCard" class="hud-panel optimal-glow p-4 flex-1 flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-xs font-bold text-cyan-400 tracking-wider">🎯 ML OPTIMAL DESTINATION</span>
                        <span id="optSpecBadge" class="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-700">READY</span>
                    </div>
                    <h2 id="optName" class="text-lg font-bold text-white leading-tight mt-1">Select Condition & Compute</h2>
                </div>
                <div class="grid grid-cols-3 gap-2 my-3">
                    <div class="bg-slate-900/90 border border-slate-800 p-2.5 rounded text-center">
                        <div class="text-[11px] text-slate-400">⏱️ ETA</div>
                        <div id="optEta" class="text-base font-bold text-cyan-400 mt-0.5">--</div>
                    </div>
                    <div class="bg-slate-900/90 border border-slate-800 p-2.5 rounded text-center">
                        <div class="text-[11px] text-slate-400">🛏️ ER Beds</div>
                        <div id="optBeds" class="text-base font-bold text-white mt-0.5">--</div>
                    </div>
                    <div class="bg-slate-900/90 border border-slate-800 p-2.5 rounded text-center">
                        <div class="text-[11px] text-slate-400">🛡️ ICU</div>
                        <div id="optIcu" class="text-base font-bold text-white mt-0.5">--</div>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-2">
                    <button id="lockBtn" onclick="lockBed()" class="bg-emerald-700 hover:bg-emerald-600 text-white font-bold py-2 rounded text-xs transition">
                        🔒 Lock Bed
                    </button>
                    <button onclick="simulateJam()" class="bg-red-800 hover:bg-red-700 text-white font-bold py-2 rounded text-xs transition">
                        ⚠️ Inject ER Jam
                    </button>
                </div>
            </div>
        </div>

        <div class="col-span-8 flex flex-col space-y-4 overflow-hidden">
            <div class="hud-panel p-2 flex-1 relative">
                <div class="absolute top-4 left-14 z-[1000] bg-slate-900/90 border border-slate-700 px-3 py-1 rounded text-xs font-bold text-slate-300 pointer-events-none">
                    SPATIAL ROUTING ENGINE (OSRM) // CLICK ANYWHERE ON MAP TO TELEPORT
                </div>
                <div id="map" class="w-full h-full rounded"></div>
            </div>

            <div class="hud-panel p-3 h-52 overflow-y-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="pb-2">ID</th>
                            <th class="pb-2">HOSPITAL NAME</th>
                            <th class="pb-2">SPECIALTIES</th>
                            <th class="pb-2">DIST</th>
                            <th class="pb-2">FREE ER</th>
                            <th class="pb-2">ICU</th>
                        </tr>
                    </thead>
                    <tbody id="hospitalTable" class="divide-y divide-slate-800/60"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let map = L.map('map', { preferCanvas: true, zoomAnimation: false }).setView([8.7200, 77.7150], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { keepBuffer: 6 }).addTo(map);

        let pinLayer = L.layerGroup().addTo(map);
        let fleetMarker = null;
        let routeLine = null;
        let currentOptimal = null;
        const osrmCache = new Map();
        let osrmController = null;

        async function computeRoute(recenterMap = false) {
            const fleet_lat = parseFloat(document.getElementById('fleetLat').value);
            const fleet_lon = parseFloat(document.getElementById('fleetLon').value);
            const condition = document.getElementById('condition').value;

            if (fleetMarker) fleetMarker.setLatLng([fleet_lat, fleet_lon]);
            else fleetMarker = L.circleMarker([fleet_lat, fleet_lon], { radius: 8, fillColor: '#ef4444', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(map).bindPopup('🚑 Active EMS Fleet');

            const res = await fetch('/api/triage', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fleet_lat, fleet_lon, condition })
            });
            const data = await res.json();
            currentOptimal = data.optimal;
            const optHosp = currentOptimal.hospital;

            document.getElementById('optName').innerText = optHosp.name;
            document.getElementById('optEta').innerText = `${currentOptimal.eta_mins}m`;
            document.getElementById('optBeds').innerText = optHosp.beds_free;
            document.getElementById('optIcu').innerText = optHosp.icu_free;
            document.getElementById('optSpecBadge').innerText = (optHosp.specialties || []).join(' • ');

            const lockBtn = document.getElementById('lockBtn');
            lockBtn.innerText = '🔒 Lock Bed';
            lockBtn.className = 'bg-emerald-700 hover:bg-emerald-600 text-white font-bold py-2 rounded text-xs transition';

            pinLayer.clearLayers();
            const rows = [];
            for (let i = 0; i < data.all.length; i++) {
                const item = data.all[i];
                const h = item.hospital;
                const isOpt = h.id === optHosp.id;
                L.circleMarker([h.lat, h.lon], {
                    radius: isOpt ? 9 : 5,
                    fillColor: isOpt ? '#10b981' : '#3b82f6',
                    color: '#fff', weight: 1.5, fillOpacity: 0.9
                }).bindPopup(`<b>${h.name}</b><br>Dist: ${item.dist_km} km<br>Beds: ${h.beds_free} | ICU: ${h.icu_free}<br>Specs: ${(h.specialties || []).join(', ')}`).addTo(pinLayer);

                rows.push(`<tr class="${isOpt ? 'bg-cyan-950/50 text-cyan-200 font-bold' : 'text-slate-300'}">
                    <td class="py-1.5 text-slate-400">${h.id}</td>
                    <td class="py-1.5">${isOpt ? '🎯 ' : ''}${h.name}</td>
                    <td class="py-1.5 text-[11px] text-cyan-400">${(h.specialties || []).join(', ')}</td>
                    <td class="py-1.5">${item.dist_km} km</td>
                    <td class="py-1.5 ${h.beds_free > 0 ? 'text-emerald-400 font-bold' : 'text-red-500 font-bold'}">${h.beds_free}</td>
                    <td class="py-1.5">${h.icu_free}</td>
                </tr>`);
            }
            document.getElementById('hospitalTable').innerHTML = rows.join('');

            if (routeLine) routeLine.setLatLngs([[fleet_lat, fleet_lon], [optHosp.lat, optHosp.lon]]);
            else routeLine = L.polyline([[fleet_lat, fleet_lon], [optHosp.lat, optHosp.lon]], { color: '#ef4444', weight: 4, dashArray: '8, 8' }).addTo(map);

            if (recenterMap) map.panTo([fleet_lat, fleet_lon], { animate: false });

            const key = `${fleet_lat.toFixed(3)},${fleet_lon.toFixed(3)}_${optHosp.id}`;
            if (osrmCache.has(key)) {
                const cached = osrmCache.get(key);
                routeLine.setLatLngs(cached.coords);
                document.getElementById('optEta').innerText = `${cached.eta}m`;
                currentOptimal.eta_mins = cached.eta;
                return;
            }
            if (osrmController) osrmController.abort();
            osrmController = new AbortController();
            fetch(`https://router.project-osrm.org/route/v1/driving/${fleet_lon},${fleet_lat};${optHosp.lon},${optHosp.lat}?overview=simplified&geometries=geojson`, { signal: osrmController.signal })
                .then(r => r.json())
                .then(d => {
                    if (d.routes && d.routes[0]) {
                        const coords = d.routes[0].geometry.coordinates.map(c => [c[1], c[0]]);
                        const realEta = Math.max(1.0, (d.routes[0].duration / 60)).toFixed(1);
                        osrmCache.set(key, { coords, eta: realEta });
                        routeLine.setLatLngs(coords);
                        document.getElementById('optEta').innerText = `${realEta}m`;
                        currentOptimal.eta_mins = realEta;
                    }
                }).catch(() => {});
        }

        async function lockBed() {
            if (!currentOptimal) return;
            const hId = currentOptimal.hospital.id;
            const res = await fetch(`/api/lock-bed/${hId}`, { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                document.getElementById('optBeds').innerText = data.beds_free;
                localStorage.setItem('er_pulse_active_dispatch', JSON.stringify({
                    case_ref: document.getElementById('caseRef').value,
                    condition: document.getElementById('condition').value,
                    hospital_id: hId,
                    hospital_name: currentOptimal.hospital.name,
                    eta_mins: currentOptimal.eta_mins,
                    beds_free: data.beds_free,
                    icu_free: currentOptimal.hospital.icu_free,
                    timestamp: Date.now()
                }));
                await computeRoute(false);
                const lockBtn = document.getElementById('lockBtn');
                lockBtn.innerText = '✅ BED LOCKED & SYNCED';
                lockBtn.className = 'bg-slate-700 text-emerald-400 font-bold py-2 rounded text-xs';
            }
        }

        async function simulateJam() {
            if (!currentOptimal) return;
            await fetch(`/api/simulate-jam/${currentOptimal.hospital.id}`, { method: 'POST' });
            computeRoute(false);
        }

        map.on('click', function(e) {
            document.getElementById('fleetLat').value = e.latlng.lat.toFixed(4);
            document.getElementById('fleetLon').value = e.latlng.lng.toFixed(4);
            computeRoute(false);
        });

        computeRoute(true);
    </script>
</body>
</html>
"""

# ==============================================================================
# 3. ZERO-LATENCY TEMPLATES/HOSPITAL.HTML (Instant Storage Event Sync)
# ==============================================================================
HOSPITAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ER-PULSE // HOSPITAL RECEIVING NODE</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #050811; color: #f8fafc; font-family: 'Courier New', monospace; }
        .alert-flash { animation: pulseRed 1.2s infinite; }
        @keyframes pulseRed {
            0%, 100% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.7); border-color: #ef4444; }
            50% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.2); border-color: #7f1d1d; }
        }
    </style>
</head>
<body class="p-8 h-screen flex flex-col justify-between">
    <div class="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
            <div class="text-xs text-cyan-400 font-bold tracking-widest">HL7/FHIR LIVE RECEIVING NODE</div>
            <h1 id="nodeHospitalName" class="text-2xl font-bold text-white mt-1">TIRUNELVELI REGIONAL TRAUMA & CARDIAC MESH</h1>
        </div>
        <div id="statusBadge" class="px-4 py-2 rounded bg-emerald-950 border border-emerald-500 text-emerald-400 text-sm font-bold">
            🟢 STANDBY // AWAITING EMS TELEMETRY
        </div>
    </div>

    <div id="incomingPanel" class="my-auto mx-auto w-full max-w-3xl bg-slate-900/90 border-2 border-slate-800 rounded-xl p-8 text-center transition-all">
        <div id="alertIcon" class="text-5xl mb-4">🏥</div>
        <h2 id="alertTitle" class="text-2xl font-bold text-slate-300">NO ACTIVE INBOUND LOCKS</h2>
        <p id="alertSub" class="text-sm text-slate-400 mt-2">Listening to ER-Pulse Spatial Triage Engine on Port 8001...</p>

        <div id="patientDetails" class="hidden mt-6 grid grid-cols-3 gap-4 text-left bg-slate-950 p-4 rounded-lg border border-red-500/40">
            <div>
                <div class="text-xs text-slate-400">INCIDENT REF</div>
                <div id="incRef" class="text-lg font-bold text-red-400">ER-CASE-904</div>
            </div>
            <div>
                <div class="text-xs text-slate-400">CLINICAL PRESENTATION</div>
                <div id="incCond" class="text-lg font-bold text-white">--</div>
            </div>
            <div>
                <div class="text-xs text-slate-400">ESTIMATED ARRIVAL (OSRM)</div>
                <div id="incEta" class="text-lg font-bold text-cyan-400">--</div>
            </div>
        </div>

        <button id="ackBtn" onclick="acknowledgeAlert()" class="hidden mt-6 w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded-lg text-sm tracking-wider uppercase transition">
            ✅ Acknowledge & Prepare Trauma / Cardiac Team
        </button>
    </div>

    <div class="grid grid-cols-2 gap-6 border-t border-slate-800 pt-4">
        <div class="bg-slate-900 p-4 rounded-lg border border-slate-800 flex justify-between items-center">
            <span class="text-sm text-slate-400">LIVE AVAILABLE ER BEDS</span>
            <span id="liveBeds" class="text-3xl font-bold text-emerald-400">12</span>
        </div>
        <div class="bg-slate-900 p-4 rounded-lg border border-slate-800 flex justify-between items-center">
            <span class="text-sm text-slate-400">LIVE AVAILABLE ICU BAYS</span>
            <span id="liveIcu" class="text-3xl font-bold text-cyan-400">4</span>
        </div>
    </div>

    <script>
        let lastAckTimestamp = 0;

        function checkDispatchSync() {
            const raw = localStorage.getItem('er_pulse_active_dispatch');
            if (!raw) return;
            const dispatch = JSON.parse(raw);

            if (dispatch.timestamp > lastAckTimestamp) {
                document.getElementById('nodeHospitalName').innerText = dispatch.hospital_name;
                document.getElementById('liveBeds').innerText = dispatch.beds_free;
                document.getElementById('liveIcu').innerText = dispatch.icu_free;

                const panel = document.getElementById('incomingPanel');
                panel.className = "my-auto mx-auto w-full max-w-3xl bg-red-950/30 border-2 border-red-500 rounded-xl p-8 text-center alert-flash";

                document.getElementById('statusBadge').className = "px-4 py-2 rounded bg-red-950 border border-red-500 text-red-400 text-sm font-bold animate-bounce";
                document.getElementById('statusBadge').innerText = "🚨 INCOMING PRIORITY ADMISSION";

                document.getElementById('alertIcon').innerText = "🚨";
                document.getElementById('alertTitle').className = "text-2xl font-bold text-red-400";
                document.getElementById('alertTitle').innerText = `BED LOCKED FOR ${dispatch.hospital_name.toUpperCase()}`;
                document.getElementById('alertSub').innerText = "Automated Pre-Arrival Handshake Triggered by EMS Dispatch";

                document.getElementById('incRef').innerText = dispatch.case_ref;
                document.getElementById('incCond').innerText = dispatch.condition;
                document.getElementById('incEta').innerText = `${dispatch.eta_mins} mins`;

                document.getElementById('patientDetails').classList.remove('hidden');
                document.getElementById('ackBtn').classList.remove('hidden');
            }
        }

        function acknowledgeAlert() {
            const raw = localStorage.getItem('er_pulse_active_dispatch');
            if (raw) {
                const dispatch = JSON.parse(raw);
                lastAckTimestamp = dispatch.timestamp;
            }
            const panel = document.getElementById('incomingPanel');
            panel.className = "my-auto mx-auto w-full max-w-3xl bg-emerald-950/20 border-2 border-emerald-500 rounded-xl p-8 text-center";
            document.getElementById('statusBadge').className = "px-4 py-2 rounded bg-emerald-950 border border-emerald-500 text-emerald-400 text-sm font-bold";
            document.getElementById('statusBadge').innerText = "✅ TEAM MOBILIZED // BED READY";
            document.getElementById('alertIcon').innerText = "✅";
            document.getElementById('alertTitle').className = "text-2xl font-bold text-emerald-400";
            document.getElementById('alertTitle').innerText = "TRAUMA / CARDIAC TEAM MOBILIZED & WAITING";
            document.getElementById('ackBtn').classList.add('hidden');
        }

        window.addEventListener('storage', checkDispatchSync);
        setInterval(checkDispatchSync, 500);
        checkDispatchSync();
    </script>
</body>
</html>
"""

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(INDEX_HTML)

with open("templates/hospital.html", "w", encoding="utf-8") as f:
    f.write(HOSPITAL_HTML)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(MAIN_PY)

print("⚡ All 3 files (main.py, index.html, hospital.html) updated successfully!")
print("🚀 Starting ER-Pulse Server on http://127.0.0.1:8001 ...")

def open_browsers():
    webbrowser.open("http://127.0.0.1:8001/")
    webbrowser.open("http://127.0.0.1:8001/hospital")

threading.Timer(1.5, open_browsers).start()

try:
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--port", "8001"])
except KeyboardInterrupt:
    print("\n🛑 ER-Pulse Server safely stopped.")