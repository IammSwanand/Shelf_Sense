# Infilect AI Pipeline — Technical Write-Up

## 1. Problem Statement

Given a retail shelf photograph, automatically:
1. Detect every product (SKU) on the shelf
2. Group detections by brand (e.g. all Huggies products get the same group id)
3. Return a JSON response and a colour-coded annotated image

The pipeline must be low-latency, scalable, and robust to clutter (price tags, rate cards).

---

## 2. Solution Overview

The pipeline is decomposed into three independently scalable microservices:

| Service | Technology | Role |
|---|---|---|
| **Detector** | YOLO11s, SKU-110K weights | Finds every product-like region |
| **Grouping** | DINOv2 ViT-S/14 + DBSCAN | Clusters crops by brand visual identity |
| **Flask** | Flask + PIL | Orchestrates, filters, visualizes, serves UI |

An additional in-process filtering module (`filtering.py`) removes price tags and rate cards between detection and grouping.

---

## 3. Detection

### Model choice: YOLO11s (SKU-110K)

The detector is a **YOLO11s** model trained on the **SKU-110K** dataset — a large-scale, densely-packed retail shelf dataset with ~1.1 million annotated products across 11,762 images. This makes it an ideal pre-trained backbone for this task without requiring any custom labeling.

**Why YOLO11s specifically?**
- Single-stage detector: fast CPU inference (~200–400 ms at 640px)
- Small model (~57MB weights) — meets the "runs without a lot of compute" requirement
- SKU-110K training gives it retail-specific priors (packed shelves, varying scales)

**Inference config used:**
- `conf=0.25` (raw output, includes rate cards — filtered downstream)
- `iou=0.45` (NMS threshold)
- `imgsz=640` (matches training resolution)

### Fallback chain

To ensure the pipeline never crashes due to a missing weights file:

| Level | Model | Trigger |
|---|---|---|
| 1 | Custom SKU-110K YOLO11s (`best.pt`) | Primary |
| 2 | `foduucom/product-detection-in-shelf-yolov8` (HuggingFace) | `best.pt` missing |
| 3 | Generic `yolov8n.pt` (COCO pretrained) | HF download fails |
| 4 | Classical CV contour proposals | All YOLO models fail |

The classical CV fallback (OpenCV adaptive threshold → morphological close → contour extraction) requires zero downloads and always produces proposals, though with lower precision.

---

## 4. Rate-Card / Price-Tag Filtering

Retail shelf images contain price tags, rate cards, and shelf-edge labels that the detector flags as objects. These are not products and must be removed before grouping to avoid polluting brand clusters.

### Three-stage filtering pipeline

**Stage A — Geometric heuristics** (always runs, ~free)

Price tags are visually flat and short. A detection is a "geometric candidate" if:
- `width / height > 2.5` (wide relative to height)
- `height / image_height < 0.06` (absolutely short)

Real products (bottles, boxes, cartons) rarely satisfy both conditions simultaneously.

**Stage B — Colour heuristics** (cheap)

Retail price tags are almost universally printed in attention-grabbing colours (orange, red, yellow). For each flagged crop:
- Convert to HSV
- Measure fraction of pixels in configurable orange/red/yellow HSV ranges
- Flag if fraction > 0.60

**Stage C — Row-relative position** (structural)

Price strips run along the bottom lip of each shelf row. We exploit this:
1. Cluster all detections into horizontal rows by y-center (gap-based binning)
2. Compute median height per row
3. Flag boxes with height < 50% of row median AND positioned at the row's bottom edge

**Combination rule:**
```
DROP if: Stage_A AND (Stage_B OR Stage_C)
```

Requiring the geometric flag *plus* a corroborating signal minimises false positives on real products that happen to be small or brightly coloured (e.g. a small orange juice carton).

### Why not a learned classifier?

Heuristics were chosen for v1 because:
- Zero training data required
- All three stages run in < 5ms total on a full shelf's worth of boxes
- Thresholds are fully configurable via environment variables for calibration

An obvious next step would be a lightweight binary classifier (product vs. tag) trained on crops from the detector's own output — logistic regression over the same HSV + edge features, or a MobileNetV3 crop classifier. This would replace only the decision function in `filtering.py` without changing any other component's contract.

---

## 5. Product Grouping

### Approach: DINOv2 embeddings + DBSCAN

**Why embeddings instead of hand-crafted features?**

An earlier iteration used HSV colour histograms + edge-orientation histograms (HOG). This grouped products primarily by colour, causing false groupings (e.g. a red Coca-Cola can and a red Pringles tube in the same cluster).

**DINOv2** (Meta AI, ICLR 2024) is a Vision Transformer pre-trained with self-supervised DINO objectives on a 142M-image curated dataset. Its representations encode fine-grained visual structure — logo shape, label typography, packaging geometry — making it far more discriminative for brand identity than colour alone.

**Pipeline per request:**
1. For each filtered detection box: crop from full image (+ 5px padding) → resize to 224×224
2. Run `facebook/dinov2-small` (ViT-S/14) → 384-dim CLS token embedding
3. L2-normalise all embeddings
4. DBSCAN (cosine distance, `eps=0.30`, `min_samples=1`) → cluster labels

**Why DBSCAN over KMeans?**
- Does not require specifying the number of brands in advance (unknown per image)
- Naturally handles outliers (singleton products) without forcing them into a wrong cluster
- Cosine distance over normalised embeddings is directly interpretable as angular similarity

**Why DINOv2-small?**
- ~80MB weights (downloaded from HuggingFace once)
- CPU inference: ~15–30ms per batch of ~50 crops
- No fine-tuning or labelled data needed

### Alternative considered: CLIP

CLIP (OpenAI) is also a strong option — its image encoder produces semantically rich embeddings. However:
- CLIP's ViT-B/32 is ~330MB vs DINOv2-small at ~80MB
- CLIP is optimised for image-text matching; DINOv2 is self-supervised specifically for visual similarity — better suited for grouping by appearance

---

## 6. Visualization

For each detection:
- One colour from a 16-colour fixed palette, mod-indexed by `group_id`
- 2px coloured border drawn around each box
- Label `G<id>` on a coloured background in the top-left corner
- Saved as JPEG to `outputs/<image_id>_viz.jpg`
- Served over `/outputs/<filename>` (never a raw filesystem path — per the assignment's explicit requirement)

---

## 7. Scalability

| Design decision | Scalability implication |
|---|---|
| Three separate Docker containers | Each service scaled independently |
| Stateless services (no per-request state) | Any replica can serve any request |
| Model loaded once at process start | Per-request latency = inference only |
| Gunicorn multi-worker + multi-thread | One container handles concurrent requests |
| `docker-compose --scale detector=N` | Horizontal detector scaling with no code change |

**Future options documented:**
- **Redis caching**: identical images (same hash) skip inference entirely
- **Async job queue**: high-burst traffic served via Celery + Redis, browser polls `/api/result/<id>`
- **GPU**: same weights run unchanged on CUDA — no re-export or retraining needed

---

## 8. Latency Budget (CPU, single image)

| Stage | Approx. time |
|---|---|
| Image decode + base64 overhead | ~10–30 ms |
| Detector inference (YOLO11s-640, CPU) | ~200–400 ms |
| Rate-card filtering (heuristics, ~50 boxes) | < 5 ms |
| DINOv2 embedding (batch of ~50 crops) | ~20–50 ms |
| DBSCAN clustering | < 5 ms |
| Visualization draw + JPEG encode | ~10–20 ms |
| **Total** | **~250–510 ms** |

---

## 9. Batch Results (20 Sample Images)

*(Fill in after running `python scripts/run_all_samples.py`)*

| Image | num_products | num_groups | num_filtered_ratecards | latency_ms | model_used |
|---|---|---|---|---|---|
| … | | | | | |

---

## 10. Open Questions / Possible Improvements

- **DBSCAN `eps` tuning**: the default 0.30 is a starting point. Plotting the cosine distance distribution over the 20 sample images would give a principled `eps` value.
- **Rate-card threshold calibration**: Stage A/B/C thresholds were set conservatively. Running the pipeline with `RATECARD_DEBUG=true` and visualising dropped vs. kept boxes on all 20 images would allow fine-tuning without code changes.
- **Multi-scale detection**: very small products at shelf edges may be missed at 640px input. Running the detector at 1280px on high-res images is a straightforward upgrade (at ~4× latency cost).
- **Brand name extraction**: adding an OCR pass (PaddleOCR, EasyOCR) on each crop to read brand text could improve grouping accuracy for text-heavy packaging — potentially combining embedding similarity + text match.
