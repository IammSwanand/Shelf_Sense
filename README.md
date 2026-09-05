# Infilect AI Pipeline

AI-powered retail shelf analysis pipeline — detects products, groups them by brand, and returns annotated visualizations.

> 📘 **Evaluator Quick Start:** Check out [`docs/EVALUATOR_GUIDE.md`](docs/EVALUATOR_GUIDE.md) for a comprehensive step-by-step evaluator walkthrough and benchmarking guide.

---

## Architecture

Three independent microservices, orchestrated by a Flask frontend:

```
Browser → Flask Orchestrator (5000)
               ↓ POST /detect
          Detector Service (5001) ← YOLO11s (SKU-110K weights)
               ↓ raw boxes
          filtering.py (in-process, rate-card removal)
               ↓ filtered boxes
          Grouping Service (5002) ← DINOv2 + Multimodal Agglomerative Clustering
               ↓ group_ids
          Visualization → /outputs/ → JSON → Browser
```

See [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`API_SPEC.md`](../API_SPEC.md), and [`WRITEUP.md`](docs/WRITEUP.md) for full design detail.

---

## Prerequisites

| Requirement | Version / Details |
|---|---|
| Docker Desktop | 4.x+ (includes docker-compose v2) |
| Internet Access | Required on initial start to download DINOv2 weights (~80 MB) |
| Python | 3.10+ *(Optional, for standalone benchmark scripts)* |

---

## Setup

### 1. Place the model weights

```
detector_service/weights/sku110k-yolo11-s640.pt
```

The weights file `sku110k-yolo11-s640.pt` is located in `detector_service/weights/`.

### 2. Copy the environment file (optional)

```bash
cp .env.example .env
```

Edit `.env` to tune thresholds or timeouts. All values have sensible defaults.

---

## Running with Docker Compose (Recommended)

### Option A: Standard CPU Mode (Default — Lightweight ~1.5 GB Build)
By default, Docker builds use official PyTorch CPU-only wheels for minimal image size and fast startup on any machine:

```bash
# Build images and start all 3 services in detached mode
docker compose up --build -d
```

### Option B: GPU / CUDA Mode (Optional)
If running on a machine with NVIDIA GPU and NVIDIA Container Toolkit:

```bash
# CUDA 12.6 is preferred & recommended:
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 docker compose up --build -d
```
*(Note: Evaluators can also pass other CUDA versions such as `cu124`, `cu121`, or `cu118` if required by their driver).*

---

### Verify and Test

Services start in dependency order (detector + grouping must be healthy before Flask starts). The first run downloads DINOv2 weights (~80MB) for the grouping service.

Open **[http://localhost:5000](http://localhost:5000)** in your browser to interact with the dashboard.

#### Smoke test endpoints:
```bash
# Health checks
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health

# End-to-end test (replace with any sample image)
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@sample_images/128008.jpg" | python -m json.tool
```

#### Stop services:
```bash
docker compose down
```

---

## Running locally (without Docker)

Install all requirements once in the root directory:
```bash
pip install -r requirements.txt
```

Then open three terminal sessions:

### Terminal 1 — Detector Service (Port 5001)
```bash
python detector_service/app.py
```

### Terminal 2 — Grouping Service (Port 5002)
```bash
python grouping_service/app.py
```

### Terminal 3 — Flask Orchestrator (Port 5000)
```bash
python flask_app/app.py
```

Open **[http://localhost:5000](http://localhost:5000)** in your browser.

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
├── .dockerignore
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
│   ├── weights/sku110k-yolo11-s640.pt ← SKU-110K detector weights
│   ├── requirements.txt
│   └── Dockerfile
├── grouping_service/
│   ├── app.py              ← DINOv2 crop-embed + Multimodal Agglomerative Linkage
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   └── run_all_samples.py
├── sample_images/          ← 20 provided shelf images
├── outputs/                ← saved visualizations (runtime)
└── docs/
    ├── EVALUATOR_GUIDE.md   ← Evaluator setup and execution manual
    ├── ARCHITECTURE.md
    ├── API_SPEC.md
    └── WRITEUP.md
```

---

## API

See [`API_SPEC.md`](../API_SPEC.md) and [`docs/EVALUATOR_GUIDE.md`](docs/EVALUATOR_GUIDE.md) for full contract. Quick reference:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive Web Dashboard |
| `/health` | GET | `{"status":"ok"}` |
| `/api/analyze` | POST | Full pipeline — returns JSON + viz URL |
| `/outputs/<file>` | GET | Serves saved visualization image |

---

## Configuration

All thresholds are env-configurable — see `.env.example` for the full list. Key ones:

| Variable | Default | Effect |
|---|---|---|
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | PyTorch wheel index (`cu126` for CUDA GPU) |
| `DEVICE` | `auto` | Auto-detects GPU if available, else falls back to CPU |
| `DET_CONF_THRESHOLD` | `0.65` | YOLO detection confidence cutoff |
| `DET_IOU_THRESHOLD` | `0.70` | YOLO NMS IoU cutoff |
| `CLUSTER_THRESHOLD` | `auto` | Self-calibrating distance threshold for Agglomerative Linkage |
| `DINO_WEIGHT` | `0.70` | Weight for DINOv2 visual similarity |
| `COLOR_WEIGHT` | `0.30` | Weight for 3D HSV color similarity |
| `RATECARD_MAX_ASPECT`| `2.5` | Rate-card geometry aspect ratio filter |
