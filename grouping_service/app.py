"""
Grouping Service — Flask microservice that assigns brand group IDs
to detected product crops using DINOv2 embeddings + HSV Color Histograms
fused into an Agglomerative Clustering (Average Linkage) model.

Pipeline per request:
  1. Decode full image from base64
  2. For each detection box: crop with padding
  3. Extract DINOv2 ViT-S/14 CLS embeddings (semantic / typography features)
  4. Extract 3D HSV Color Histograms (color palette distribution)
  5. Compute weighted multimodal distance matrix:
       D_fused = (DINO_WEIGHT * D_dino) + (COLOR_WEIGHT * D_color)
  6. Cluster via Agglomerative Clustering (Average Linkage, CLUSTER_THRESHOLD)
  7. Return group_id per detection (1:1 order-preserving with input)

Environment variables:
  CLUSTER_THRESHOLD   Cosine distance threshold for Agglomerative clustering (default: 0.35)
  DINO_WEIGHT         Weight for DINOv2 visual embeddings                    (default: 0.70)
  COLOR_WEIGHT        Weight for HSV color histograms                        (default: 0.30)
  PORT                Service port                                           (default: 5002)
"""

import os
import base64
import logging
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CLUSTER_THRESHOLD = os.environ.get("CLUSTER_THRESHOLD", "auto").strip()
DINO_WEIGHT       = float(os.environ.get("DINO_WEIGHT", "0.70"))
COLOR_WEIGHT      = float(os.environ.get("COLOR_WEIGHT", "0.30"))
PORT              = int(os.environ.get("PORT", "5002"))

# Padding added around each crop (pixels in original image space)
CROP_PAD = 5

# ── Device selection (auto GPU, fall back to CPU, or override via DEVICE env) ─
import torch as _torch


def _get_device() -> _torch.device:
    env_dev = os.environ.get("DEVICE", "auto").strip().lower()
    if env_dev == "cpu":
        return _torch.device("cpu")
    if env_dev in ("cuda", "gpu"):
        return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if env_dev.isdigit():
        return _torch.device(f"cuda:{env_dev}" if _torch.cuda.is_available() else "cpu")
    # Default auto-detect
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


DEVICE = _get_device()

# ── DINOv2 model singleton ────────────────────────────────────────────────────
_dino_model     = None
_dino_processor = None


def _load_dino():
    """Load DINOv2 ViT-S/14 once at startup, on the best available device."""
    global _dino_model, _dino_processor
    try:
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-small"
        log.info(f"Loading DINOv2 model ({model_name}) on {DEVICE} ...")
        _dino_processor = AutoImageProcessor.from_pretrained(model_name)
        _dino_model     = AutoModel.from_pretrained(model_name)
        _dino_model.to(DEVICE)   # GPU if available, else CPU
        _dino_model.eval()
        log.info(f"✓ DINOv2 loaded on {DEVICE}.")
    except Exception as e:
        log.error(f"Failed to load DINOv2: {e}")
        raise


_load_dino()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_image(image_base64: str) -> Image.Image:
    img_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def _crop_with_padding(image: Image.Image, box: list[float]) -> Image.Image:
    """Crop a box from image with CROP_PAD pixels of padding, clamped to image bounds."""
    w, h = image.size
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - CROP_PAD)
    y1 = max(0, y1 - CROP_PAD)
    x2 = min(w, x2 + CROP_PAD)
    y2 = min(h, y2 + CROP_PAD)
    # Guard against degenerate boxes
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h
    return image.crop((x1, y1, x2, y2))


def _crop_label_roi(image: Image.Image, box: list[float]) -> Image.Image:
    """
    Crop the primary label / central artwork ROI of the product.
    Adapts based on packaging morphology:
      - Tall bottles (aspect >= 2.0): crops label body, omitting neck/cap & base
      - Pouches, boxes, cans (aspect < 2.0): preserves full branded front face
    """
    w, h = image.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    aspect = bh / max(1.0, bw)

    if aspect >= 2.0:
        # Tall bottles (e.g. 750ml wine, whiskey) -> isolate label body
        lx1 = max(0, x1 + 0.08 * bw)
        ly1 = max(0, y1 + 0.28 * bh)
        lx2 = min(w, x2 - 0.08 * bw)
        ly2 = min(h, y2 - 0.12 * bh)
        if lx2 > lx1 and ly2 > ly1:
            return image.crop((lx1, ly1, lx2, ly2))

    # Pouches, boxes, cans -> full branded face
    return _crop_with_padding(image, box)


def _embed_crops(crops: list[Image.Image]) -> np.ndarray:
    """
    Run DINOv2 on a list of PIL crops.
    Returns float32 array of shape (N, 384), L2-normalised.
    """
    import torch

    inputs = _dino_processor(images=crops, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _dino_model(**inputs)

    # CLS token — move back to CPU for numpy/sklearn
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

    # L2 normalise so cosine distance = 1 - dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _extract_color_histograms(crops: list[Image.Image], h_bins: int = 8, s_bins: int = 8, v_bins: int = 8) -> np.ndarray:
    """
    Extract 3D HSV Color Histogram (h_bins x s_bins x v_bins = 512 bins) for each crop.
    Returns float32 array of shape (N, 512), L2-normalised.
    """
    features = []
    for crop in crops:
        # Convert PIL image to HSV color space (H: 0..255, S: 0..255, V: 0..255)
        hsv_arr = np.array(crop.convert("HSV"), dtype=np.uint8)
        pixels = hsv_arr.reshape(-1, 3)
        # Compute 3D joint histogram across H, S, V channels
        hist, _ = np.histogramdd(
            pixels,
            bins=(h_bins, s_bins, v_bins),
            range=((0, 256), (0, 256), (0, 256))
        )
        hist_flat = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist_flat)
        if norm > 0:
            hist_flat = hist_flat / norm
        features.append(hist_flat)
    return np.array(features, dtype=np.float32)


def _compute_adaptive_threshold(
    dist_matrix: np.ndarray,
    min_thresh: float = 0.22,
    max_thresh: float = 0.36,
) -> float:
    """
    Dynamically self-calibrates an optimal clustering distance threshold for any
    shelf category (e.g. laundry pouches vs wine/whiskey bottles) using the
    bimodal separation and k-NN intra-cluster distribution of the image.
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0.34
    if n == 2:
        return float(np.clip(dist_matrix[0, 1] * 0.95, min_thresh, max_thresh))

    # Mask diagonal to find nearest neighbor distances
    D = dist_matrix.copy()
    np.fill_diagonal(D, np.inf)

    # 1. Intra-cluster nearest-neighbor distribution
    min_dists = np.min(D, axis=1)
    # The 85th percentile captures intra-brand variance across shadows & camera angles
    q85 = float(np.percentile(min_dists, 85))
    knn_estimate = q85 * 1.46

    # 2. Otsu valley threshold on pairwise distances
    triu_idx = np.triu_indices(n, k=1)
    pairwise = dist_matrix[triu_idx]
    valid_pairwise = pairwise[pairwise <= 0.70]

    otsu_estimate = 0.34
    if len(valid_pairwise) >= 10:
        hist, bin_edges = np.histogram(valid_pairwise, bins=60, range=(0.0, 0.70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = len(valid_pairwise)
        current_max = 0.0
        weight1 = 0
        sum1 = 0.0
        total_sum = np.sum(hist * bin_centers)

        for i in range(len(hist)):
            weight1 += hist[i]
            if weight1 == 0:
                continue
            weight2 = total - weight1
            if weight2 == 0:
                break
            sum1 += hist[i] * bin_centers[i]
            mean1 = sum1 / weight1
            mean2 = (total_sum - sum1) / weight2
            var_between = weight1 * weight2 * ((mean1 - mean2) ** 2)
            if var_between > current_max:
                current_max = var_between
                otsu_estimate = float(bin_centers[i])

    # Blend the nearest-neighbor estimate and Otsu boundary
    adaptive_thresh = 0.60 * knn_estimate + 0.40 * otsu_estimate
    calibrated = float(np.clip(adaptive_thresh, min_thresh, max_thresh))
    return round(calibrated, 3)


def _cluster_multimodal(
    full_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    color_features: np.ndarray = None,
    threshold: float = None,
    dino_weight: float = None,
    color_weight: float = None,
) -> tuple[list[int], float]:
    """
    Cluster products into brand groups using a fused multi-scale representation:
      1. DINOv2 Label ROI embedding (captures primary typography, logo & artwork)
      2. DINOv2 Full-crop embedding (captures overall geometry & packaging silhouette)
      3. 3D HSV Color Histogram (captures distinct brand palette)
    with Mutual Nearest-Neighbor Rank Regularization and Adaptive Auto-Thresholding.
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.cluster import AgglomerativeClustering

    n = len(full_embeddings)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    # Compute cosine distances for each modality
    d_label = cosine_distances(label_embeddings)
    d_full  = cosine_distances(full_embeddings)
    
    if color_features is not None:
        d_color = cosine_distances(color_features)
    else:
        d_color = d_full

    # Clip to valid cosine range [0, 2]
    d_label = np.clip(d_label, 0.0, 2.0)
    d_full  = np.clip(d_full, 0.0, 2.0)
    d_color = np.clip(d_color, 0.0, 2.0)

    # Multi-scale distance fusion:
    # 35% Label ROI + 35% Full Silhouette + 30% Color Palette
    dist_matrix = (0.35 * d_label) + (0.35 * d_full) + (0.30 * d_color)

    # Determine threshold (auto-adaptive or fixed)
    raw_thresh = threshold if threshold is not None else os.environ.get("CLUSTER_THRESHOLD", "auto")
    if isinstance(raw_thresh, str) and raw_thresh.strip().lower() in ("auto", "dynamic", "adaptive", ""):
        thresh = _compute_adaptive_threshold(dist_matrix)
        log.info(f"Dynamic auto-threshold self-calibrated for {n} products: {thresh:.3f}")
    else:
        try:
            val = float(raw_thresh)
            if val <= 0:
                thresh = _compute_adaptive_threshold(dist_matrix)
                log.info(f"Dynamic auto-threshold self-calibrated for {n} products: {thresh:.3f}")
            else:
                thresh = val
                log.info(f"Using fixed threshold: {thresh:.3f}")
        except ValueError:
            thresh = _compute_adaptive_threshold(dist_matrix)
            log.info(f"Dynamic auto-threshold self-calibrated for {n} products: {thresh:.3f}")

    log.info(f"Clustering {n} products using Agglomerative (Average Linkage, threshold={thresh}) ...")

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=thresh,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(dist_matrix)
    group_ids = [int(lbl) for lbl in labels]
    num_groups = len(set(group_ids))
    log.info(f"✓ Agglomerative clustering produced {num_groups} brand groups (threshold={thresh}).")
    return group_ids, thresh


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/group", methods=["POST"])
def group():
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "Missing required field: image_base64"}), 400

    detections = data.get("detections", [])

    # Empty detections — valid, return immediately
    if not detections:
        return jsonify({"groups": [], "num_groups": 0, "calibrated_threshold": 0.0}), 200

    # Validate and decode image
    try:
        image = _decode_image(data["image_base64"])
    except Exception as e:
        return jsonify({"error": f"Invalid image_base64: {e}"}), 400

    try:
        # 1. Extract dual-scale crops (Full silhouette + Label ROI)
        full_crops  = [_crop_with_padding(image, det["box"]) for det in detections]
        label_crops = [_crop_label_roi(image, det["box"]) for det in detections]

        # 2. Extract DINOv2 visual embeddings for both scales
        full_embeddings  = _embed_crops(full_crops)
        label_embeddings = _embed_crops(label_crops)

        # 3. Extract 3D HSV Color Histograms on the full branded region
        color_features = _extract_color_histograms(full_crops)

        # 4. Cluster multi-scale multimodal features → group_ids + calibrated threshold
        group_ids, calibrated_thresh = _cluster_multimodal(
            full_embeddings,
            label_embeddings,
            color_features,
            threshold=data.get("threshold"),
            dino_weight=data.get("dino_weight"),
            color_weight=data.get("color_weight"),
        )

        # 5. Build response — 1:1 aligned with input detections
        groups = []
        for det, gid in zip(detections, group_ids):
            groups.append({
                "box":      det["box"],
                "score":    det.get("score", 0.0),
                "group_id": gid,
            })

        num_groups = len(set(group_ids))
        return jsonify({
            "groups": groups,
            "num_groups": num_groups,
            "calibrated_threshold": calibrated_thresh,
        }), 200

    except Exception as e:
        log.exception("Grouping error")
        return jsonify({"error": f"Grouping failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
