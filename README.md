# Infilect AI Pipeline

AI-powered retail shelf analysis pipeline — detects products, groups them by brand, and returns annotated visualizations.

## Architecture

Three independent microservices, orchestrated by a Flask frontend:

```
Browser → Flask Orchestrator (5000)
               ↓ POST /detect
          Detector Service (5001) ← YOLO11s (SKU-110K weights)
               ↓ raw boxes
          filtering.py (in-process, rate-card removal)
               ↓ filtered boxes
          Grouping Service (5002) ← DINOv2 + DBSCAN
               ↓ group_ids
          Visualization → /outputs/ → JSON → Browser
```

See [`ARCHITECTURE.md`](../ARCHITECTURE.md) and [`API_SPEC.md`](../API_SPEC.md) for full design detail.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Docker Desktop | 4.x+ |
| docker-compose | v2 (bundled with Docker Desktop) |

---

## Setup

### 1. Place the model weights

```
detector_service/weights/best.pt
```

The weights file (`sku110k-yolo11-s640.pt`) must be copied/renamed to `best.pt` at the path above **before** running. It is not included in the repo (listed in `.gitignore`).

### 2. Copy the environment file (optional)

```bash
cp .env.example .env
```

Edit `.env` to tune thresholds or timeouts. All values have sensible defaults.

---

## Running with Docker Compose (recommended)

```bash
# Build images and start all 3 services (auto-detects GPU if available, else runs on CPU)
docker compose up --build

# Or in detached mode
docker compose up --build -d
```

Services start in dependency order (detector + grouping must be healthy before Flask starts). The first run downloads DINOv2 weights (~80MB) for the grouping service.

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Smoke test

```bash
# Health checks
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health

# End-to-end test (replace with any sample image)
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@sample_images/128008.jpg" | python -m json.tool
```

### Stop

```bash
docker compose down
```

---

## Running locally (without Docker)

Requires three terminal sessions.

### Terminal 1 — Detector Service

```bash
cd detector_service
pip install -r requirements.txt
DETECTOR_WEIGHTS_PATH=weights/best.pt python app.py
```

### Terminal 2 — Grouping Service

```bash
cd grouping_service
pip install -r requirements.txt
python app.py
```

### Terminal 3 — Flask Orchestrator

```bash
cd flask_app
pip install -r requirements.txt
DETECTOR_URL=http://localhost:5001 GROUPING_URL=http://localhost:5002 python app.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## Batch test (all 20 sample images)

With the stack running:

```bash
python scripts/run_all_samples.py
```

Prints a summary table and saves `scripts/batch_results.csv`.

---

## Project structure

```
infilect_pipeline/
├── docker-compose.yml
├── .env.example
├── README.md
├── flask_app/
│   ├── app.py              ← pipeline orchestrator
│   ├── filtering.py        ← rate-card filtering (Stages A/B/C)
│   ├── templates/index.html
│   ├── static/style.css
│   ├── requirements.txt
│   └── Dockerfile
├── detector_service/
│   ├── app.py              ← YOLO11-s640 SKU-110K detector
│   ├── weights/best.pt     ← place your weights here (not in git)
│   ├── requirements.txt
│   └── Dockerfile
├── grouping_service/
│   ├── app.py              ← DINOv2 crop-embed + DBSCAN
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   └── run_all_samples.py
├── sample_images/          ← 20 provided shelf images
├── outputs/                ← saved visualizations (runtime, gitignored)
└── docs/
    ├── ARCHITECTURE.md
    ├── API_SPEC.md
    └── WRITEUP.md
```

---

## API

See [`API_SPEC.md`](../API_SPEC.md) for full contract. Quick reference:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Upload UI |
| `/health` | GET | `{"status":"ok"}` |
| `/api/analyze` | POST | Full pipeline — returns JSON + viz URL |
| `/outputs/<file>` | GET | Serves saved visualization image |

---

## Configuration

All thresholds are env-configurable — see `.env.example` for the full list. Key ones:

| Variable | Default | Effect |
|---|---|---|
| `DEVICE` | `auto` | Auto-detects GPU if available, else falls back to CPU |
| `DET_CONF_THRESHOLD` | `0.25` | YOLO detection confidence cutoff |
| `DBSCAN_EPS` | `0.45` | Brand-cluster neighbourhood radius |
| `RATECARD_MAX_ASPECT` | `2.5` | Rate-card geometry filter |
| `RATECARD_DEBUG` | `false` | Attach `filter_reason` to dropped boxes |
