# Infilect Retail Shelf Analysis Pipeline

> **Tip**: Press `Ctrl + Shift + V` (or `Cmd + Shift + V` on macOS) to open the rendered Markdown Preview in your IDE.

> **Documentation Index**:
> - **Setup & Execution Guide**: [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) (Docker CPU/GPU modes, local setup, resource requirements)
> - **API Specification**: [`docs/API_SPEC.md`](docs/API_SPEC.md) (Endpoint contracts, request/response JSON schemas)
> - **System Architecture**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (Microservice decomposition, data flow)
> - **Technical Write-Up**: [`docs/WRITEUP.md`](docs/WRITEUP.md) (Design decisions, filtering strategy, benchmarks, scaling)

---

## Overview

A scalable microservice pipeline for automated retail shelf analysis:
- **Product Detection**: Localizes all product instances on dense retail shelves using YOLO11s trained on SKU-110K.
- **Rate-Card Filtering**: Automatically removes non-product clutter (shelf-edge price tags, rate cards) using an in-process heuristic filter.
- **Brand Grouping**: Clusters products into brand families using a multimodal fusion of Dual-Scale DINOv2 vision embeddings and 3D HSV Color Histograms with Agglomerative Clustering.
- **Interactive UI & Analytics**: Serves annotated visualizations, brand share-of-shelf percentage metrics, and JSON results.

---

## System Architecture

![Infilect System Architecture](docs/assets/architecture_diagram.png)

---

## Quick Start

### 1. Model Weights
Verify that `sku110k-yolo11-s640.pt` exists in `detector_service/weights/` (downloaded automatically on first startup from Hugging Face: [`chistopat/sku110k-yolo11-object-detector`](https://huggingface.co/chistopat/sku110k-yolo11-object-detector) if missing).

### 2. Run with Docker Compose (Recommended)

```bash
# Start all microservices (CPU Mode)
docker compose up --build -d
```

*(For GPU acceleration with CUDA 12.6, run: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 docker compose up --build -d`)*

### 3. Alternative: Run Locally (Without Docker)

```bash
# 1. Create and activate an isolated virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the three services (in separate terminals with .venv activated):
python detector_service/app.py    # Port 5001
python grouping_service/app.py    # Port 5002
python flask_app/app.py           # Port 5000
```

### 4. Open Web Dashboard
Navigate to **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## Core API Endpoint

### `POST /api/analyze`
Accepts a shelf image and returns detections, brand groups, and visualization URL.

```bash
curl -X POST -F "image=@test_images/dense_61.jpg" http://localhost:5000/api/analyze
```

**Example Response:**
```json
{
  "image_id": "f71d7060ad03",
  "width": 1080,
  "height": 1920,
  "num_products": 31,
  "num_groups": 11,
  "num_filtered_ratecards": 11,
  "model_used": "sku110k-yolo11-s640",
  "visualization_url": "/outputs/f71d7060ad03_viz.jpg",
  "latency_ms": 3025.0,
  "detections": [
    {
      "id": 0,
      "box": [349.0, 1331.9, 507.8, 1429.0],
      "det_score": 0.82,
      "group_id": 10
    }
  ]
}
```

---

## Project Structure

```
infilect_pipeline/
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── README.md
├── flask_app/              <- Web UI, API orchestrator, rate-card filtering
├── detector_service/       <- YOLO11s SKU-110K detection microservice
├── grouping_service/       <- DINOv2 + HSV Multimodal clustering microservice
├── test_images/            <- Retail shelf test images
├── outputs/                <- Saved visual output images
└── docs/
    ├── SETUP_GUIDE.md      <- Setup and execution manual
    ├── API_SPEC.md         <- Complete API specification
    ├── ARCHITECTURE.md     <- End-to-end architecture breakdown
    └── WRITEUP.md          <- Technical write-up and evaluation
```
