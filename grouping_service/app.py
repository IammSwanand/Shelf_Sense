"""
Grouping Service — Flask microservice that assigns brand group IDs
to detected product crops using DINOv2 embeddings + DBSCAN clustering.

Pipeline per request:
  1. Decode full image from base64
  2. For each detection box: crop → resize 224×224 → DINOv2 CLS embedding
  3. L2-normalise all embeddings → cosine-distance DBSCAN
  4. Return group_id per detection (1:1 order-preserving with input)

Environment variables:
  DBSCAN_EPS          Cosine-distance neighbourhood radius  (default: 0.30)
  DBSCAN_MIN_SAMPLES  Min samples per cluster               (default: 1)
  PORT                Service port                          (default: 5002)
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
DBSCAN_EPS         = float(os.environ.get("DBSCAN_EPS", "0.28"))  # API_SPEC.md §3
DBSCAN_MIN_SAMPLES = int(os.environ.get("DBSCAN_MIN_SAMPLES", "1"))
PORT               = int(os.environ.get("PORT", "5002"))

# Padding added around each crop (pixels in original image space)
CROP_PAD = 5

# ── DINOv2 model singleton ────────────────────────────────────────────────────
_dino_model     = None
_dino_processor = None


def _load_dino():
    """Load DINOv2 ViT-S/14 once at startup."""
    global _dino_model, _dino_processor
    try:
        from transformers import AutoImageProcessor, AutoModel
        import torch

        model_name = "facebook/dinov2-small"
        log.info(f"Loading DINOv2 model ({model_name}) ...")
        _dino_processor = AutoImageProcessor.from_pretrained(model_name)
        _dino_model     = AutoModel.from_pretrained(model_name)
        _dino_model.eval()
        log.info("✓ DINOv2 loaded successfully.")
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


def _embed_crops(crops: list[Image.Image]) -> np.ndarray:
    """
    Run DINOv2 on a list of PIL crops.
    Returns float32 array of shape (N, 384), L2-normalised.
    """
    import torch

    inputs = _dino_processor(images=crops, return_tensors="pt")
    with torch.no_grad():
        outputs = _dino_model(**inputs)

    # CLS token embedding — shape (N, 384)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)

    # L2 normalise so cosine distance = 1 - dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _cluster(embeddings: np.ndarray) -> list[int]:
    """
    DBSCAN over cosine distance matrix.
    Noise points (label -1) are each assigned a unique singleton group id.
    Returns list of int group_ids, length == len(embeddings).
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_distances

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    dist_matrix = cosine_distances(embeddings)
    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="precomputed",
    ).fit_predict(dist_matrix)

    # Remap noise (-1) to unique singleton ids beyond the max cluster id
    max_label = int(labels.max()) if labels.max() >= 0 else -1
    next_id = max_label + 1
    group_ids = []
    for lbl in labels:
        if lbl == -1:
            group_ids.append(next_id)
            next_id += 1
        else:
            group_ids.append(int(lbl))
    return group_ids


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
        return jsonify({"groups": [], "num_groups": 0}), 200

    # Validate and decode image
    try:
        image = _decode_image(data["image_base64"])
    except Exception as e:
        return jsonify({"error": f"Invalid image_base64: {e}"}), 400

    try:
        # 1. Crop each detected region
        crops = [_crop_with_padding(image, det["box"]) for det in detections]

        # 2. Embed all crops in one batch
        embeddings = _embed_crops(crops)

        # 3. Cluster embeddings → group_ids
        group_ids = _cluster(embeddings)

        # 4. Build response — 1:1 aligned with input detections
        groups = []
        for det, gid in zip(detections, group_ids):
            groups.append({
                "box":      det["box"],
                "score":    det.get("score", 0.0),
                "group_id": gid,
            })

        num_groups = len(set(group_ids))
        return jsonify({"groups": groups, "num_groups": num_groups}), 200

    except Exception as e:
        log.exception("Grouping error")
        return jsonify({"error": f"Grouping failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
