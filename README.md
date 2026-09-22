# 🚑 ER-Pulse: Enterprise Real-GPS Dynamic Emergency Triage & Capacity Operations OS

An enterprise-grade Emergency Medical Services (EMS) computer-aided dispatch and hospital capacity load balancing platform engineered to eliminate golden-hour delays and blind emergency room rejections.

---

## 🌐 Geography Bounds: Tirunelveli Medical Corridor, India
- **Latitude Span**: `8.7050` to `8.7400` N
- **Longitude Span**: `77.6950` to `77.7450` E
- **Corridor Hospital Nodes**:
  - `H1`: TVMC (High Grounds Govt Hospital) [8.7139, 77.7275 | 12 ER Beds, 4 ICU | Trauma, Burns, General, Cardiac]
  - `H2`: Shifa Multi-Specialty Hospital [8.7280, 77.7120 | 8 ER Beds, 3 ICU | Cardiology, Oncology, Ortho]
  - `H3`: Galaxy Hospitals (24/7 ER) [8.7350, 77.7010 | 6 ER Beds, 2 ICU | Cardiology, Trauma, Emergency]
  - `H4`: CSI Mission Hospital [8.7090, 77.7400 | 5 ER Beds, 1 ICU | General, Trauma]
  - `H5`: Rosemary Mission Hospital [8.7240, 77.7180 | 4 ER Beds, 1 ICU | General, Emergency]

---

## 📐 Multi-Objective Scoring & Spatial Formula

### 1. Great-Circle Distance (Haversine Formula)
$$\text{distance\_km} = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
*(where $R = 6371.0\text{ km}$)*

### 2. Urban Corridor Speed Scaling Factor
$$\text{ETA}_{\text{mins}} = \text{distance\_km} \times 4.5\text{ mins}$$

### 3. Utility Score (Lower = Optimal Destination)
$$\text{Score} = (\text{ETA}_{\text{mins}} \times 1.5) - (\text{Beds}_{\text{free}} \times 0.9) - (\text{Match}_{\text{score}} \times 6.0)$$

---

## 🛠️ Architecture & Tech Stack

- **Backend**: FastAPI (Python asynchronous REST engine), Uvicorn ASGI, Pydantic v2 data validation schemas.
- **Frontend**: Single-Page Application (HTML5 + Tailwind CSS + Leaflet.js real OpenStreetMap dark canvas).
- **Design System**: Palantir Foundry / Tactical EMS HUD (Obsidian slate-950 background, high-density monospace telemetry, instant client-server reactivity).
- **Provenance**: Cryptographic SHA-256 fingerprinting in `.krish_watermark.json`, DOM comments, and git author provenance (`Krish`).

---

## 🚀 Quickstart & Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access Tactical Command Center
Navigate to `http://localhost:8000` in any modern web browser.

---

## 📡 API Reference

- `GET /` - Tactical EMS HUD Single-Page Application.
- `GET /api/hospitals` - Returns full live hospital capacity ledger.
- `POST /api/triage` - Dispatches multi-objective spatial calculation and route vectors.
- `POST /api/lock-bed/{hospital_id}` - Atomically reserves and locks 1 ER bed at target facility.
- `POST /api/simulate-jam/{hospital_id}` - Simulates sudden ER saturation (capacity drops to 0) and triggers rerouting.
- `POST /api/reset` - Restores corridor baseline capacity.
- `GET /api/watermark` - Returns cryptographic tamper-check signatures and developer provenance.
