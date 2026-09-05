# Infilect AI Pipeline — Evaluator Setup & Execution Guide

This guide is prepared for technical evaluators to set up, test, and benchmark the **Infilect Retail Shelf Analyzer Pipeline**.

---

## 1. Executive Summary & Architecture

The solution implements a fully modular, horizontally scalable microservice architecture containing three independent services orchestrated via Docker Compose:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                   FLASK ORCHESTRATOR & UI                   │
                  │              (Port 5000: Web Dashboard + API)               │
                  └─────────────────────────────────────────────────────────────┘
                                   │                          │
        (1) POST /detect           │                          │ (3) Filter & Assemble
        (Raw image via HTTP)       ▼                          ▼     (Outlines, Share-of-Shelf)
                    ┌─────────────────────────┐     ┌─────────────────────────┐
                    │    DETECTOR SERVICE     │     │    GROUPING SERVICE     │
                    │       (Port 5001)       │     │       (Port 5002)       │
                    │   YOLO11s-640 (SKU-110K)│     │  Dual-Scale DINOv2 ViT  │
                    │  Product Localization   │     │  + 3D HSV Color Fusion  │
                    └─────────────────────────┘     └─────────────────────────┘
                                   │                              ▲
                                   └─────── (2) POST /group ──────┘
                                          (Filtered Bounding Boxes)
```

| Service | Container Name | Port | Key Technologies |
|:---|:---|:---:|:---|
| **Flask Orchestrator** | `infilect_flask` | `5000` | Flask, Pillow, OpenCV, Interactive Dashboard UI |
| **Detector Service** | `infilect_detector` | `5001` | YOLO11s-640 (SKU-110K weights), Gunicorn |
| **Grouping Service** | `infilect_grouping` | `5002` | DINOv2 ViT-S/14, HSV Histograms, Agglomerative Linkage |

---

## 2. Prerequisites & Setup

### Requirements
- **Docker Desktop** (v4.x+ with Docker Compose v2)
- **Model Weights**: Verify that `best.pt` exists in `detector_service/weights/best.pt`.
- **Internet Connection**: An active internet connection is required during first startup so `grouping_service` can automatically download the official DINOv2 model weights (~80 MB) from Hugging Face.
- *(Optional)* Python 3.10+ if running standalone benchmark scripts outside Docker.

---

## 3. Launching the Pipeline

### Option A: Standard CPU Mode (Default & Recommended)
By default, Docker builds use official PyTorch CPU-only wheels. This keeps container image sizes minimal (**~1.2 – 1.8 GB** per image instead of >10 GB) and starts up quickly on any standard laptop or workstation without requiring GPU drivers:

```bash
# Clone/Navigate to the project root directory
cd infilect_pipeline

# Build and start all services in detached mode
docker compose up --build -d
```

---

### Option B: GPU / CUDA Acceleration Mode
If you have an NVIDIA GPU with the NVIDIA Container Toolkit installed, you can build with CUDA-accelerated PyTorch:

> **CUDA Compatibility Note:**
> - **`cu126` (CUDA 12.6) is preferred and tested for highest throughput.**
> - Evaluators can also specify other CUDA wheel versions depending on their driver (e.g. `cu124`, `cu121`, `cu118`).

#### Running with CUDA via CLI:

**Linux / macOS (Bash):**
```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 docker compose up --build -d
```

**Windows (PowerShell):**
```powershell
$env:TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"; docker compose up --build -d
```

**Alternative (via `.env` file):**
Uncomment and set `TORCH_INDEX_URL` in `.env`:
```ini
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
DEVICE=cuda
```
Then run `docker compose up --build -d`.

---

## 4. Verification & Testing

### 1. Health Checks
Verify that all microservices have initialized and report healthy:

```bash
curl http://localhost:5000/health
# {"status":"ok","services":{"detector":"ok","grouping":"ok"}}

curl http://localhost:5001/health
# {"device":"cpu","model":"sku110k-yolo11-s640","status":"ok"}

curl http://localhost:5002/health
# {"device":"cpu","dinov2_model":"dinov2_vits14","status":"ok"}
```

---

### 2. Interactive Web Dashboard
Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

**Features of the UI:**
1. **Drag & Drop Upload:** Upload any retail shelf image (`.jpg`, `.png`, `.webp`).
2. **Stage Progression Indicator:** Visual real-time indicator for `Detection` ➔ `Filtering` ➔ `Grouping` ➔ `Visualization`.
3. **Annotated Output Visualization:** High-contrast color-coded bounding boxes per brand cluster.
4. **Brand Group Breakdown Cards:** Share-of-shelf percentage metrics, item counts, and color swatches.
5. **Full-Resolution Lightbox:** Click on the visualizer or "Expand View" for detailed inspection.
6. **Raw JSON Inspector & Copy Tool:** Inspect complete bounding box coordinates and cluster IDs.

---

### 3. CLI Single-Image Test
Send a sample image directly to the REST API:

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@sample_images/128008.jpg" | python -m json.tool
```

---

### 4. Automated 20-Sample Benchmark Script
To run an automated end-to-end evaluation across all 20 provided shelf samples:

```bash
# In your host environment with requests & PIL installed:
python scripts/run_all_samples.py
```
This prints latency, product counts, and group breakdowns in tabular format and generates a detailed CSV summary report at `scripts/batch_results.csv`.

---

## 5. Alternative: Running Locally without Docker

If you prefer to run the Python processes directly on your host machine (or in a Python virtual environment):

```bash
# 1. Install all dependencies from the root directory
pip install -r requirements.txt
```

Then open three terminal sessions:

- **Terminal 1 (Detector Service - Port 5001):**
  ```bash
  python detector_service/app.py
  ```
- **Terminal 2 (Grouping Service - Port 5002):**
  ```bash
  python grouping_service/app.py
  ```
- **Terminal 3 (Flask Orchestrator - Port 5000):**
  ```bash
  python flask_app/app.py
  ```

Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 6. API Contract & JSON Specification

### `POST /api/analyze`
- **Request:** `multipart/form-data` with field `image` containing the binary image file.
- **Response Schema:**

```json
{
  "status": "success",
  "filename": "128008.jpg",
  "num_products": 22,
  "num_groups": 6,
  "num_filtered_ratecards": 2,
  "latency_ms": 348.5,
  "model_used": "sku110k-yolo11-s640",
  "visualization_url": "/outputs/viz_128008_1725528000.jpg",
  "detections": [
    {
      "box_id": 0,
      "group_id": 1,
      "confidence": 0.884,
      "bbox": {
        "x1": 142.0,
        "y1": 310.0,
        "x2": 265.0,
        "y2": 580.0,
        "width": 123.0,
        "height": 270.0
      }
    }
  ]
}
```

---

## 7. Stopping the Services

To shut down and clean up containers:
```bash
docker compose down
```

To remove shared volumes as well:
```bash
docker compose down -v
```
