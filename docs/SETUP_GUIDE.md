# Setup & Execution Guide

> **Tip**: Press `Ctrl + Shift + V` (or `Cmd + Shift + V` on macOS) to open the rendered Markdown Preview in your IDE.

This guide provides simple step-by-step instructions to set up and run the **Infilect Retail Shelf Analyzer Pipeline** using either Docker or direct Terminal execution.

---

## Prerequisites

- **Model Weights**: Ensure `sku110k-yolo11-s640.pt` exists in `detector_service/weights/sku110k-yolo11-s640.pt` (automatically fetched on startup from Hugging Face [`chistopat/sku110k-yolo11-object-detector`](https://huggingface.co/chistopat/sku110k-yolo11-object-detector) if not present).
- **Internet Connection**: Required during initial setup to download packages/models (DINOv2 weights ~80 MB are downloaded automatically on first run).

---

## Approach 1: Docker Compose Mode (Recommended)

Runs all three microservices in isolated, reproducible Docker containers.

- **Storage Needed**: ~3.5 GB (CPU) / ~10-12 GB (GPU with CUDA runtime)
- **Setup Time**: ~2-4 minutes (builds images & starts all services in background)

Navigate to the project directory:
```bash
cd infilect_pipeline
```

### 1.1 CPU Mode (Default)
Builds lightweight containers using PyTorch CPU wheels. Works on any standard computer without GPU setup:

```bash
docker compose up --build -d
```

### 1.2 GPU Mode (CUDA Acceleration)
Utilizes your NVIDIA GPU for 8-10x faster inference (~300ms vs ~2.5s per shelf):

**Linux / macOS (Bash):**
```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 docker compose up --build -d
```

**Windows (PowerShell):**
```powershell
$env:TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"; docker compose up --build -d
```

*To stop Docker containers:*
```bash
docker compose down
```

---

## Approach 2: Terminal / Native Python Mode

Runs the three microservices directly on your host machine without Docker.

- **Storage Needed**: ~2.5 GB (CPU venv) / ~5.5 GB (GPU venv with CUDA wheels)
- **Setup Time**: ~1-2 minutes (pip package installation)

### 2.1 Virtual Environment Setup (Recommended)

Always create and activate an isolated virtual environment before installing packages:

```bash
cd infilect_pipeline
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell / Command Prompt):
.venv\Scripts\activate

# On Linux / macOS (Bash / Zsh):
source .venv/bin/activate
```

#### Install Dependencies:

**Option A: CPU Mode (Default)**
```bash
pip install -r requirements.txt
```

**Option B: GPU / CUDA Mode (Optional)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 2.2 Launching the Services

Open **3 separate terminal windows** (with your virtual environment activated) and run:

**Terminal 1: Detector Service (Port 5001):**
```bash
python detector_service/app.py
```

**Terminal 2: Grouping Service (Port 5002):**
```bash
python grouping_service/app.py
```

**Terminal 3: Flask Orchestrator & UI (Port 5000):**
```bash
python flask_app/app.py
```

---

## Verification & Testing

### 1. Interactive Web Dashboard
Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

Upload any shelf photograph to view real-time stage progress, color-coded product grouping visualization, brand share-of-shelf breakdown, and raw JSON.

### 2. CLI Single-Image Test
Send a sample image directly to the API endpoint:

```bash
curl -X POST -F "image=@../sample_images/128008.jpg" http://localhost:5000/api/analyze
```

### 3. Health Checks
Verify all 3 services report healthy:

```bash
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
```
