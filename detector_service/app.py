"""
Detector Service — Flask microservice wrapping YOLO11-s640 (SKU-110K weights).

Environment variables (all optional, have defaults):
  DETECTOR_WEIGHTS_PATH   Path to YOLO weights    (default: /app/weights/best.pt)
  DET_CONF_THRESHOLD      YOLO conf threshold     (default: 0.25)
  DET_IOU_THRESHOLD       YOLO NMS IoU threshold  (default: 0.45)
  DET_IMG_SIZE            Inference resolution    (default: 640)
  PORT                    Service port            (default: 5001)
"""

import os
import base64
import logging
from io import BytesIO

from flask import Flask, request, jsonify
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config from environment ───────────────────────────────────────────────────
CONF_THRESHOLD  = float(os.environ.get("DET_CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD   = float(os.environ.get("DET_IOU_THRESHOLD", "0.45"))
IMG_SIZE        = int(os.environ.get("DET_IMG_SIZE", "640"))
PORT            = int(os.environ.get("PORT", "5001"))

# ── Device selection (auto GPU, fall back to CPU, or override via DEVICE env) ─
import torch as _torch


def _get_device():
    env_dev = os.environ.get("DEVICE", "auto").strip().lower()
    if env_dev == "cpu":
        return "cpu"
    if env_dev in ("cuda", "gpu"):
        return 0 if _torch.cuda.is_available() else "cpu"
    if env_dev.isdigit():
        return int(env_dev) if _torch.cuda.is_available() else "cpu"
    # Default auto-detect
    return 0 if _torch.cuda.is_available() else "cpu"


DEVICE = _get_device()

# ── Model singleton (loaded once at startup) ──────────────────────────────────
_model      = None   # YOLO model object
_model_used = "sku110k-yolo11-s640"


def _find_weights_path() -> str:
    """Find the YOLO11-s640 weight file across common locations."""
    candidate_paths = [
        os.environ.get("DETECTOR_WEIGHTS_PATH"),
        "/app/weights/best.pt",
        "/app/weights/sku110k-yolo11-s640.pt",
        os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
        os.path.join(os.path.dirname(__file__), "weights", "sku110k-yolo11-s640.pt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "sku110k-yolo11-s640.pt"),
        os.path.join(os.path.dirname(__file__), "..", "sku110k-yolo11-s640.pt"),
        "sku110k-yolo11-s640.pt",
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return os.environ.get("DETECTOR_WEIGHTS_PATH", "/app/weights/best.pt")


def _load_model():
    """Load YOLO11-s640 SKU-110K model."""
    global _model
    from ultralytics import YOLO

    weights_path = _find_weights_path()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"YOLO11-s640 weights not found at '{weights_path}'. "
            f"Please ensure the weights file is present."
        )

    log.info(f"Loading YOLO11-s640 weights from {weights_path} on {DEVICE} ...")
    _model = YOLO(weights_path)
    log.info("YOLO11-s640 model loaded successfully.")


def _run_yolo(image_base64: str) -> tuple[list[dict], int, int]:
    """Run YOLO inference; returns (detections, width, height)."""
    img_bytes = base64.b64decode(image_base64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    width, height = pil_img.size

    results = _model.predict(
        source=pil_img,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )
    result = results[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        detections.append({
            "box":   [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "score": round(score, 4),
        })
    return detections, width, height


# Load at import time (gunicorn worker startup)
try:
    _load_model()
except Exception as err:
    log.error(f"Failed to load YOLO11-s640 model: {err}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    # API_SPEC.md §2: exactly {"status": "ok", "model_loaded": true}
    return jsonify({
        "status":       "ok",
        "model_loaded": _model is not None,
    }), 200


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "Missing required field: image_base64"}), 400

    if _model is None:
        return jsonify({"error": "Model not loaded. Please check weights file."}), 503

    image_base64 = data["image_base64"]

    # Validate the image before inference
    try:
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        width, height = pil_img.size
    except Exception as e:
        return jsonify({"error": f"Invalid image_base64: {e}"}), 400

    try:
        detections, width, height = _run_yolo(image_base64)
        return jsonify({
            "width":      width,
            "height":     height,
            "detections": detections,
            "model_used": _model_used,
        }), 200

    except Exception as e:
        log.exception("Inference error")
        return jsonify({"error": f"Inference failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
