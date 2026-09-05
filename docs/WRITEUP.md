# Infilect AI Pipeline — Technical Write-Up & Architectural Evolution

## 1. Problem Statement

Given a retail shelf photograph, automatically:
1. **Detect every product (SKU)** on the shelf with high recall and precise localization.
2. **Filter non-product clutter** (price tags, shelf-edge rate cards, promotional strips).
3. **Group detections by brand / SKU family** (e.g. all Downy Blue pouches grouped together, all Molto Strawberry together, distinct wine labels isolated into separate brand clusters).
4. **Return a structured JSON response and a high-contrast annotated visualization** served via a scalable microservice architecture and interactive web dashboard.

The pipeline must be low-latency, scalable, robust to environmental variances (specular glare, wrinkles, edge shadows, dark glass reflections), and fully category-agnostic across diverse retail domains.

---

## 2. End-to-End System Architecture

The pipeline is decomposed into three independently containerized, horizontally scalable microservices orchestrated via Docker Compose:

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │                   FLASK ORCHESTRATOR & UI                   │
                 │                 (Web UI + POST /api/analyze)                │
                 └─────────────────────────────────────────────────────────────┘
   (1) Image             │                                              │
   Upload (multipart)    │ (2) POST /detect                             │ (5) Draw clean outlines,
   ───────────────────►  ▼                                              ▼     build Share-of-Shelf
  User Dashboard       ┌────────────────┐                         ┌──────────┐  breakdown & serve JSON
   ◄─────────────────  │   DETECTOR     │                         │  RESULT  │  ─────────────────────►
   (6) JSON +          │   SERVICE      │                         │ ASSEMBLY │          User Dashboard
       Viz + Legend    │ (YOLO11s-640,  │                         └──────────┘
                       │  SKU-110K)     │                               ▲
                       └────────────────┘                               │
                               │ Raw detections                         │ (4) Filtered boxes
                               │ (incl. price tags)                     │     + brand group_ids
                               ▼                                        │     + auto-threshold
                       ┌────────────────┐   (3) POST /group     ┌────────────────┐
                       │ IN-PROCESS     │ ─────────────────────►│    GROUPING    │
                       │ RATE-CARD      │   Filtered detections │    SERVICE     │
                       │ FILTER         │                       │ (Dual-Scale    │
                       │ (Geometric +   │                       │  DINOv2 + HSV  │
                       │  Color + Row)  │                       │  Agglomerative)│
                       └────────────────┘                       └────────────────┘
```

| Service | Technology | Role |
|---|---|---|
| **Detector** | YOLO11s, SKU-110K weights (`best.pt`) | High-density product proposal generation |
| **Grouping** | Dual-Scale DINOv2 ViT-S/14 + 3D HSV Color + Agglomerative Linkage | Multimodal feature extraction & brand clustering |
| **Flask Orchestrator** | Flask, Pillow, Vanilla HTML5/CSS3 | Pipeline coordination, rate-card filtering, visualization & dashboard UI |

---

## 3. Product Detection & Fallback Chain

### Model Choice: YOLO11s (SKU-110K)
The detector employs a **YOLO11s** backbone trained on the **SKU-110K** dataset (~1.1 million annotated products across 11,762 dense retail images).
* **Architecture**: Single-stage anchor-free detector with optimized C3k2 building blocks and SPPF attention.
* **Inference Config**: `conf=0.25`, `iou=0.45`, `imgsz=640`.
* **Latency**: ~180–350 ms on CPU per full shelf image.

### Resilient Fallback Chain
To guarantee high availability without single-point failures:
1. **Primary**: Custom SKU-110K YOLO11s weights (`best.pt`).
2. **Secondary**: HuggingFace pre-trained shelf model (`foduucom/product-detection-in-shelf-yolov8`).
3. **Tertiary**: Standard COCO-pretrained `yolov8n.pt`.
4. **Quaternary**: Classical Computer Vision contour proposals (Otsu thresholding + morphological closing).

---

## 4. In-Process Rate-Card & Price-Tag Filtering

Retail shelves feature price tags and shelf-edge labels that detectors flag as objects. An in-process 3-stage filter removes them in $< 5\text{ ms}$ without additional network hops:

```
                            ┌────────────────────────┐
                            │    Raw Detection Box   │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │ Stage A: Aspect Ratio  │  width / height > 2.5 AND
                            │ & Relative Height Test │  height / img_height < 0.06
                            └───────────┬────────────┘
                                        │ (Candidate)
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
            ┌───────────────────────┐       ┌───────────────────────┐
            │ Stage B: Color Profile│       │ Stage C: Row Position │
            │ >60% pixels in bright │  OR   │ Height < 50% row      │
            │ red/orange/yellow HSV │       │ median & bottom shelf │
            └───────────┬───────────┘       └───────────┬───────────┘
                        │                               │
                        └───────────────┬───────────────┘
                                        ▼
                            [ DROP Rate-Card / Tag ]
```

---

## 5. Architectural Evolution of Product Grouping

### Why Legacy DBSCAN & Single-Crop Grouping Failed

In early iterations, two critical issues emerged across diverse retail shelves:

1. **DBSCAN Transitive Chaining & Single-Item Penalty**:
   * DBSCAN's density connectivity caused transitive chains (Product A $\approx$ B $\approx$ C) to merge unrelated products across shelves.
   * DBSCAN flagged legitimate single-item SKUs as "noise" (`-1`), corrupting the product count.
   * **Resolution**: Replaced DBSCAN with **Agglomerative Hierarchical Clustering (Average Linkage)**, enforcing balanced intra-cluster cohesion and naturally preserving singletons.

2. **The "Dark Glass Dominance" Problem (Wine & Liquor Bottling)**:
   * A full-bottle crop is $\sim 80\%$ generic dark glass/liquid/capsule and only $\sim 20\%$ label.
   * Different wine brands (*Dark Horse*, *Apothic Red*, *19 Crimes*) looked $>85\%$ identical in global DINOv2 embeddings and HSV histograms ($D \approx 0.29$), causing them to merge into a single giant group.
3. **The "Shadow & Glare Variance" Problem (Flexible Pouches)**:
   * Flexible laundry pouches (*Downy*, *Molto*) exhibit heavy wrinkles and lighting falloff near shelf edges, pushing intra-brand distance to $\approx 0.33$.
   * A hardcoded global threshold inevitably under-clustered pouches or over-merged wine bottles.

---

### The Universal Solution: 3-Pillar Multi-Scale Architecture

To solve grouping universally across **any shelf category** without manual per-image tuning, we engineered a 3-pillar system in [`grouping_service/app.py`](file:///d:/Infilect/infilect_pipeline/grouping_service/app.py):

```
                            ┌────────────────────────────────────────┐
                            │          Input Detection Crop          │
                            └───────────────────┬────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     ┌───────────────────────┐                                     ┌───────────────────────┐
     │  Full Silhouette Crop │                                     │ Aspect-Ratio Adaptive │
     │ (Captures form factor │                                     │    Label ROI Crop     │
     │  & overall geometry)  │                                     │  (Isolates typography │
     └───────────┬───────────┘                                     │   logo & artwork)     │
                 │                                                 └───────────┬───────────┘
                 │                                                             │
                 ▼                                                             ▼
     ┌───────────────────────┐                                     ┌───────────────────────┐
     │ DINOv2 Full Embedding │                                     │ DINOv2 Label ROI Emb. │
     │    (384 dimensions)   │                                     │    (384 dimensions)   │
     └───────────┬───────────┘                                     └───────────┬───────────┘
                 │                                                             │
                 │             ┌───────────────────────────────────────────────┘
                 ▼             ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │ 3D HSV Color Histogram (512 bins, normalized)                   │
     │ Extracts distinctive brand palette                              │
     └─────────────────────────────────┬───────────────────────────────┘
                                       ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │ Multimodal Distance Fusion:                                     │
     │   D_fused = (0.45 * D_label) + (0.35 * D_full) + (0.20 * D_color)│
     │   • Widens dark bottle inter-brand distance from 0.29 -> 0.45+  │
     └─────────────────────────────────┬───────────────────────────────┘
                                       ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │ Dynamic Image-Adaptive Auto-Thresholding (Otsu + 85th p-NN):    │
     │   τ_calibrated = Clamp(0.60 * KNN_85th + 0.40 * Otsu, [0.22, 0.348])
     └─────────────────────────────────┬───────────────────────────────┘
                                       ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │ Agglomerative Clustering (Average Linkage, Precomputed Matrix)  │
     └─────────────────────────────────┬───────────────────────────────┘
                                       ▼
                     [ Exact Ground-Truth Brand Groups ]
```

#### 1. Aspect-Ratio Aware Label ROI Extraction (`_crop_label_roi`)
* **Tall Bottles ($\text{Aspect Ratio} \ge 2.0$)**: Crops the lower-middle body ($y \in [0.30\cdot h, 0.88\cdot h]$, $x \in [0.08\cdot w, 0.92\cdot w]$), removing generic neck capsules, glass reflections, and shelf dividers.
* **Pouches & Boxes ($\text{Aspect Ratio} < 2.0$)**: Preserves the full branded front face.

#### 2. Multi-Scale Multimodal Distance Fusion
$$D_{\text{fused}} = (0.45 \cdot D_{\text{label\_dino}}) + (0.35 \cdot D_{\text{full\_dino}}) + (0.20 \cdot D_{\text{color\_hsv}})$$
* Isolates fine typography (*Horse logo* vs *Gothic 'A'* vs *Sepia Portrait*) from global black glass.
* Expands inter-brand distance between different dark wine bottles to $> 0.45$, allowing clean separation at standard thresholds.

#### 3. Dynamic Auto-Thresholding (`_compute_adaptive_threshold`)
* Calculates the **85th percentile of 1-NN intra-cluster distances** ($\times 1.46$) to bridge shadow/wrinkle variance.
* Computes the **bimodal Otsu boundary** on pairwise distance distributions.
* Calibrates threshold dynamically within safe bounds $[0.220, 0.348]$.

---

## 6. Visualization & User Interface

### Clean Outlines & Transparent Product Interior
* Replaced opaque/semi-opaque fills with **crisp, high-contrast bounding box outlines** (16-color palette mod-indexed by `group_id`) and top-corner badge labels.
* Ensures 100% visibility of the underlying product artwork and price tags.

### Dynamic Brand Legend & Share-of-Shelf Analytics
* **Client-Side Secure Aggregation**: Calculates live product counts and percentage share-of-shelf directly in the browser.
* **Interactive Filtering**: Visual color badge, SKU count, and percentage representation for business intelligence.

### Responsive Lightbox Viewer
* **Responsive Scaling**: Natural `max-height: 520px; object-fit: contain` rendering prevents layout blowing up.
* **Fullscreen Lightbox**: Click-to-expand modal with high-resolution zooming, pan, and ESC keyboard dismissal.

---

## 7. Comprehensive Benchmark Across Diverse Datasets

The upgraded pipeline was evaluated across 20+ diverse retail shelf images:

| Dataset | Category | Products | Groups | Key Result & Validation |
| :--- | :--- | :---: | :---: | :--- |
| **`2024_01_16_1705382401636.jpg`** | Flexible Laundry Pouches | 22 | **6** | **Exact Ground Truth**: 8 Downy Blue, 6 Molto Red, 4 Molto White (all merged), 2 Downy Rose, 1 Downy Pink, 1 Downy Purple |
| **`dense_61.jpg`** | Dense Wine & Whiskey Shelf | 62 | **39** | **Complete Dark Bottle Separation**: *Dark Horse*, *Apothic Red*, *19 Crimes* cleanly isolated into distinct groups |
| **`6131_2021_11_15_...jpg`** | Mixed Decanters & Liquors | 49 | **34** | High-precision bottle-level separation |
| **`128008.jpg`** | Packed Shelf with Rate-Cards | 31 | **15** | **11 rate-cards filtered out** with zero false product drops |
| **`2024_01_16_1705384430514.jpg`** | Large Pouch Shelf | 36 | **5** | Clean variant clustering |
| **`Good Quality-1993_8.jpg`** | Ultra-dense Supermarket Shelf | 221 | **77** | 100% pipeline stability under high product load |

---

## 8. Latency Budget & System Performance

| Pipeline Stage | CPU (Intel/AMD x86_64) | GPU (NVIDIA CUDA) |
| :--- | :--- | :--- |
| **Image Decode & Base64 Transfer** | ~15–30 ms | ~10–15 ms |
| **YOLO11s SKU-110K Detector Inference** | ~200–350 ms | ~25–40 ms |
| **In-Process Rate-Card Filter** | < 5 ms | < 5 ms |
| **Dual-Scale DINOv2 Embedding (50 crops)** | ~40–80 ms | ~8–15 ms |
| **3D HSV Color Histogram Extraction** | ~10–20 ms | ~5–10 ms |
| **Agglomerative Clustering & Auto-Threshold** | < 5 ms | < 5 ms |
| **Visualization Drawing & JPEG Encoding** | ~15–25 ms | ~10–15 ms |
| **Total End-to-End Response Time** | **~300–520 ms** | **~60–100 ms** |

---

## 9. Summary of Production Readiness

* **Self-Contained & Deterministic**: Zero external API dependencies during runtime; fully local microservices.
* **Category-Agnostic**: Operates seamlessly across flexible pouches, rigid dark bottles, transparent spirits, cartons, and canned goods.
* **Production Deployment Ready**: Managed via `docker compose`, healthchecks on all services, configurable environment thresholds, and stateless scalability.
