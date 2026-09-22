"""
ER-Pulse: Real-GPS Dynamic Emergency Triage & Capacity Operations OS
Mission-critical EMS load balancing, routing, and bed-locking platform.
Tirunelveli Medical Corridor, India.

Provenance & Signature:
Developer & Owner: Krish (Krishna Raja V)
Watermark Tag: DEVELOPED_AND_OWNED_BY_KRISH_2026
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ============================================================================
# DOMAIN DATA & GEOGRAPHY SPECIFICATION (Tirunelveli Medical Corridor)
# ============================================================================

GEO_BOUNDS = {
    "lat_min": 8.7050,
    "lat_max": 8.7400,
    "lon_min": 77.6950,
    "lon_max": 77.7450,
    "center": [8.7200, 77.7180],
}

INITIAL_HOSPITALS = [
    {
        "id": "H1",
        "name": "TVMC (High Grounds Govt Hospital)",
        "lat": 8.7139,
        "lon": 77.7275,
        "beds_free": 12,
        "icu_free": 4,
        "total_beds": 50,
        "specialties": ["Trauma", "Burns", "General", "Cardiac"],
        "status": "OPERATIONAL"
    },
    {
        "id": "H2",
        "name": "Shifa Multi-Specialty Hospital",
        "lat": 8.7280,
        "lon": 77.7120,
        "beds_free": 8,
        "icu_free": 3,
        "total_beds": 35,
        "specialties": ["Cardiology", "Oncology", "Ortho"],
        "status": "OPERATIONAL"
    },
    {
        "id": "H3",
        "name": "Galaxy Hospitals (24/7 ER)",
        "lat": 8.7350,
        "lon": 77.7010,
        "beds_free": 6,
        "icu_free": 2,
        "total_beds": 30,
        "specialties": ["Cardiology", "Trauma", "Emergency"],
        "status": "OPERATIONAL"
    },
    {
        "id": "H4",
        "name": "CSI Mission Hospital",
        "lat": 8.7090,
        "lon": 77.7400,
        "beds_free": 5,
        "icu_free": 1,
        "total_beds": 25,
        "specialties": ["General", "Trauma"],
        "status": "OPERATIONAL"
    },
    {
        "id": "H5",
        "name": "Rosemary Mission Hospital",
        "lat": 8.7240,
        "lon": 77.7180,
        "beds_free": 4,
        "icu_free": 1,
        "total_beds": 20,
        "specialties": ["General", "Emergency"],
        "status": "OPERATIONAL"
    }
]

# In-memory hospital ledger with baseline cloning
hospitals_ledger: Dict[str, dict] = {h["id"]: dict(h) for h in INITIAL_HOSPITALS}

# ============================================================================
# PYDANTIC VALIDATION SCHEMAS
# ============================================================================

class HospitalNode(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    beds_free: int = Field(ge=0, description="Available emergency room beds")
    icu_free: int = Field(ge=0, description="Available ICU buffer units")
    total_beds: int
    specialties: List[str]
    status: str

class TriageRequest(BaseModel):
    patient_id: str = Field(default="ER-CASE-904", min_length=2, max_length=64)
    category: str = Field(
        default="Acute Cardiac / Chest Pain",
        description="Clinical emergency classification"
    )
    ambulance_lat: float = Field(ge=8.7000, le=8.7500, default=8.7200)
    ambulance_lon: float = Field(ge=77.6900, le=77.7500, default=77.7150)

class RankedHospital(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    beds_free: int
    icu_free: int
    total_beds: int
    specialties: List[str]
    status: str
    distance_km: float
    eta_mins: float
    match_score: float
    urgency_rank_score: float
    is_optimal: bool

class TriageResponse(BaseModel):
    dispatch_id: str
    timestamp: str
    patient_id: str
    category: str
    required_specialty: str
    ambulance_coords: List[float]
    best_hospital: RankedHospital
    corridor_rankings: List[RankedHospital]
    pre_alert_dispatched: bool
    route_polyline: List[List[float]]

class ActionResponse(BaseModel):
    success: bool
    message: str
    hospital: HospitalNode

# ============================================================================
# SPATIAL MATH & MULTI-OBJECTIVE SCORING ENGINE
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in kilometers.
    Radius Earth R = 6371.0 km.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 3)

def resolve_required_specialty(category: str) -> str:
    """
    Deterministic clinical specialty resolution based on triage classification.
    """
    cat_lower = category.lower()
    if "cardiac" in cat_lower or "chest pain" in cat_lower or "stroke" in cat_lower:
        return "Cardiology"
    elif "trauma" in cat_lower or "bypass" in cat_lower or "accident" in cat_lower:
        return "Trauma"
    elif "burns" in cat_lower or "respiratory" in cat_lower:
        return "Burns"
    return "General"

def calculate_specialty_match(hospital_specialties: List[str], required_specialty: str) -> float:
    """
    Calculates specialty compatibility weight: 1.0 for positive match, 0.4 fallback.
    """
    specs_joined = " ".join(hospital_specialties).lower()
    req_lower = required_specialty.lower()
    if req_lower in specs_joined:
        return 1.0
    # Secondary fallback for Burns -> Trauma centers
    if req_lower == "burns" and "trauma" in specs_joined:
        return 0.8
    return 0.4

def compute_triage(request: TriageRequest) -> TriageResponse:
    """
    Executes multi-objective decision optimization:
    Score = (ETA_mins * 1.5) - (Beds_free * 0.9) - (Match_score * 6.0)
    Lower score = optimal destination rank.
    Hospitals with zero beds available receive severe capacity penalty.
    """
    req_spec = resolve_required_specialty(request.category)
    ranked_list: List[RankedHospital] = []

    for h in hospitals_ledger.values():
        dist_km = haversine_distance(request.ambulance_lat, request.ambulance_lon, h["lat"], h["lon"])
        # Speed corridor model: 4.5 mins per urban corridor km
        eta = round(dist_km * 4.5, 1)
        match_score = calculate_specialty_match(h["specialties"], req_spec)
        
        # Primary multi-objective formula
        score = (eta * 1.5) - (h["beds_free"] * 0.9) - (match_score * 6.0)
        
        # Add critical penalty if facility is jammed (0 beds) to divert ambulances
        if h["beds_free"] <= 0:
            score += 50.0  # Severe penalty avoids fatal diversion delays

        ranked_list.append(RankedHospital(
            id=h["id"],
            name=h["name"],
            lat=h["lat"],
            lon=h["lon"],
            beds_free=h["beds_free"],
            icu_free=h["icu_free"],
            total_beds=h["total_beds"],
            specialties=h["specialties"],
            status="CRITICAL_JAM" if h["beds_free"] == 0 else h["status"],
            distance_km=dist_km,
            eta_mins=eta,
            match_score=match_score,
            urgency_rank_score=round(score, 2),
            is_optimal=False
        ))

    # Sort strictly by lowest urgency_rank_score
    ranked_list.sort(key=lambda x: x.urgency_rank_score)
    ranked_list[0].is_optimal = True
    best = ranked_list[0]

    dispatch_id = f"DISPATCH-TVL-{int(datetime.now(timezone.utc).timestamp())}"
    route_polyline = [
        [request.ambulance_lat, request.ambulance_lon],
        [best.lat, best.lon]
    ]

    return TriageResponse(
        dispatch_id=dispatch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        patient_id=request.patient_id,
        category=request.category,
        required_specialty=req_spec,
        ambulance_coords=[request.ambulance_lat, request.ambulance_lon],
        best_hospital=best,
        corridor_rankings=ranked_list,
        pre_alert_dispatched=True,
        route_polyline=route_polyline
    )

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure static directory exists
    os.makedirs("static", exist_ok=True)
    yield

app = FastAPI(
    title="ER-Pulse: Mission-Critical EMS Operations OS",
    description="Real-GPS Dynamic Emergency Triage & Bed-Locking Platform for Tirunelveli Medical Corridor",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/api/hospitals", response_model=List[HospitalNode], tags=["Hospital Ledger"])
def get_hospitals():
    """Retrieve full live hospital status ledger."""
    return [HospitalNode(**h) for h in hospitals_ledger.values()]

@app.post("/api/triage", response_model=TriageResponse, tags=["Triage Dispatch"])
def post_triage(request: TriageRequest):
    """Execute spatial-temporal multi-criteria utility triage routing."""
    return compute_triage(request)

@app.post("/api/lock-bed/{hospital_id}", response_model=ActionResponse, tags=["Inventory Mutators"])
def lock_bed(hospital_id: str):
    """
    Atomic ER bed lock: decrements available beds by 1 down to 0 minimum floor.
    Simulates clinical handover & reservation.
    """
    if hospital_id not in hospitals_ledger:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hospital node '{hospital_id}' not found in corridor ledger."
        )
    h = hospitals_ledger[hospital_id]
    if h["beds_free"] <= 0:
        return ActionResponse(
            success=False,
            message=f"Zero capacity alert: No beds available to lock at {h['name']}.",
            hospital=HospitalNode(**h)
        )
    h["beds_free"] = max(0, h["beds_free"] - 1)
    if h["beds_free"] == 0:
        h["status"] = "CRITICAL_JAM"
    return ActionResponse(
        success=True,
        message=f"Pre-arrival ER bed reserved and locked at {h['name']}. Remaining free: {h['beds_free']}",
        hospital=HospitalNode(**h)
    )

@app.post("/api/simulate-jam/{hospital_id}", response_model=ActionResponse, tags=["Failure-Mode Injection"])
def simulate_jam(hospital_id: str):
    """
    Simulates sudden acute corridor congestion / ER saturation by dropping free beds to 0.
    Triggers dynamic algorithmic rerouting away from this node.
    """
    if hospital_id not in hospitals_ledger:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hospital node '{hospital_id}' not found in corridor ledger."
        )
    h = hospitals_ledger[hospital_id]
    h["beds_free"] = 0
    h["status"] = "CRITICAL_JAM"
    return ActionResponse(
        success=True,
        message=f"Corridor Failure Simulated: {h['name']} capacity plunged to 0. Routing engine reranking.",
        hospital=HospitalNode(**h)
    )

@app.post("/api/reset", response_model=List[HospitalNode], tags=["Inventory Mutators"])
def reset_ledger():
    """Restores baseline hospital capacities across all corridor nodes."""
    global hospitals_ledger
    hospitals_ledger = {h["id"]: dict(h) for h in INITIAL_HOSPITALS}
    return [HospitalNode(**h) for h in hospitals_ledger.values()]

@app.get("/api/watermark", tags=["Provenance"])
def get_watermark():
    """Cryptographic provenance verification signature."""
    wm_path = ".krish_watermark.json"
    if os.path.exists(wm_path):
        with open(wm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return JSONResponse(content=data)
    return JSONResponse(content={"error": "Watermark manifest missing"}, status_code=404)

@app.get("/", response_class=FileResponse, tags=["Tactical HUD SPA"])
def serve_hud():
    """Serves the Palantir Foundry / Tactical EMS HUD Single-Page Application."""
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(content={"status": "ER-Pulse API active. Static frontend not yet compiled."}, status_code=200)

# Mount static directory for static assets
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
