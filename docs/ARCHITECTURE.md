# ShelfSense - End-to-End System Architecture

> **Tip**: Press `Ctrl + Shift + V` (or `Cmd + Shift + V` on macOS) to open the rendered Markdown Preview in your IDE.

## 1. High-Level Architecture

The system employs a decoupled, containerized microservice architecture:

![System Architecture Diagram](assets/architecture_diagram.png)

### 1.1 End-to-End Execution Flow

![High-Level Execution Flow](assets/pipeline_flow_diagram.png)

---

## 2. Components

### 2.1 Flask Orchestrator (`flask_app/`)
- Serves an interactive web UI (upload button, progress stages, brand legend, share-of-shelf analytics, and lightbox viewer).
- `POST /api/analyze`: accepts an image (multipart file or JSON base64),
  drives downstream steps, assembles final response, saves
  visualization to disk, and serves it back over `/outputs/<file>` - **never**
  a raw filesystem path shared with the client.
- Stateless - can be scaled horizontally behind a reverse proxy or load balancer.

### 2.2 Detector Service (`detector_service/`)
- Wraps **YOLO11s-640 trained on SKU-110K** ([`chistopat/sku110k-yolo11-object-detector`](https://huggingface.co/chistopat/sku110k-yolo11-object-detector)).
- Single responsibility: image in -> raw bounding boxes + confidence out.
- Automatic weight resolution: loads `sku110k-yolo11-s640.pt` from `weights/` directory (with automatic download fallback from Hugging Face on startup if missing).

### 2.3 Post-Processing / Filtering Module (`flask_app/filtering.py`)
**Purpose:** remove non-product detections (rate cards, price tags, shelf
edge labels, promotional strips) from detector's raw output before
grouping.

### 2.4 Grouping Service (`grouping_service/`)
- Input: the image + **filtered** detections only.
- Extracts a **Dual-Scale Multimodal Representation**:
  - **Label ROI Crop** (aspect-ratio adaptive, extracts typography, logo, and artwork while omitting generic dark glass/neck/cap).
  - **Full Silhouette Crop** (captures overall packaging form factor & geometry).
  - **3D HSV Color Histogram** (512-dim, captures distinctive brand color palette).
- **Multimodal Distance Fusion**:
  $$D_{\text{fused}} = (0.45 \cdot D_{\text{label\_dino}}) + (0.35 \cdot D_{\text{full\_dino}}) + (0.20 \cdot D_{\text{color\_hsv}})$$
- **Dynamic Image-Adaptive Auto-Thresholding**: evaluates 85th percentile intra-cluster 1-NN distances and Otsu valley separation, self-calibrating within safe bounds $[0.220, 0.348]$.
- **Agglomerative Clustering** (Average Linkage, precomputed distance matrix): cleanly separates subtle bottle variants while grouping shadowed flexible pouches, with natural singleton handling.
- Returns a `group_id` per input box, 1:1 aligned with the input list.

### 2.5 Result Assembly + Visualization & Dashboard
- Draws crisp, high-contrast bounding box outlines (16-color palette mod-indexed by `group_id`) with transparent product interiors so products and price tags remain 100% visible.
- Serves an interactive web dashboard with live **Brand Group Color Legend & Share-of-Shelf Breakdown** and a **Responsive Fullscreen Lightbox Viewer**.

---

## 3. Scalability & Production Readiness

- **Independent Microservices:** Decoupled container architecture (`detector`, `grouping`, `flask`) communicating via HTTP REST APIs.
- **Horizontal Scaling:** Supports `docker compose up --scale detector=3 --scale grouping=2` with automatic round-robin DNS load balancing.
- **Multi-Worker Concurrency:** Gunicorn WSGI servers with concurrent worker threads.
- **Stateless Design:** No per-request filesystem or session coupling.
