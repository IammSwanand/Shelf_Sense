"""
Detector Service — Flask microservice wrapping YOLO11s (SKU-110K weights)
with a 4-level fallback chain so the pipeline never hard-crashes.

Fallback order:
  1. Custom SKU-110K YOLO11s weights (best.pt)  → model_used: "custom-sku110k-yolo11s"
  2. HuggingFace shelf-tuned YOLOv8             → model_used: "shelf-yolov8"
  3. Generic COCO YOLOv8n                       → model_used: "generic-yolov8"
  4. Classical CV contour proposals              → model_used: "classical-cv"

Environment variables (all optional, have defaults):
  DETECTOR_WEIGHTS_PATH   Path to custom weights  (default: /app/weights/best.pt)
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
WEIGHTS_PATH    = os.environ.get("DETECTOR_WEIGHTS_PATH", "/app/weights/best.pt")
CONF_THRESHOLD  = float(os.environ.get("DET_CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD   = float(os.environ.get("DET_IOU_THRESHOLD", "0.45"))
IMG_SIZE        = int(os.environ.get("DET_IMG_SIZE", "640"))
PORT            = int(os.environ.get("PORT", "5001"))

# ── Model singleton (loaded once at startup) ──────────────────────────────────
_model      = None   # YOLO model object, or None for classical-cv
_model_used = None   # string label for the API response


def _try_load_custom() -> bool:
    """Level 1: custom SKU-110K YOLO11s weights."""
    global _model, _model_used
    try:
        if not os.path.exists(WEIGHTS_PATH):
            log.warning(f"Custom weights not found at {WEIGHTS_PATH}.")
            return False
        from ultralytics import YOLO
        log.info(f"Loading custom weights from {WEIGHTS_PATH} ...")
        _model = YOLO(WEIGHTS_PATH)
        _model_used = "custom-sku110k-yolov8s"  # API_SPEC.md §2 mandated string
        log.info("✓ Custom SKU-110K YOLO11s loaded.")
        return True
    except Exception as e:
        log.warning(f"Custom weights failed: {e}")
        return False


def _try_load_hf_shelf() -> bool:
    """Level 2: HuggingFace shelf-tuned YOLOv8."""
    global _model, _model_used
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        log.info("Downloading shelf YOLOv8 from HuggingFace ...")
        hf_path = hf_hub_download(
            repo_id="foduucom/product-detection-in-shelf-yolov8",
            filename="best.pt",
        )
        _model = YOLO(hf_path)
        _model_used = "shelf-yolov8"
        log.info("✓ HuggingFace shelf YOLOv8 loaded.")
        return True
    except Exception as e:
        log.warning(f"HuggingFace shelf model failed: {e}")
        return False


def _try_load_generic_yolo() -> bool:
    """Level 3: Generic COCO YOLOv8n."""
    global _model, _model_used
    try:
        from ultralytics import YOLO
        log.info("Loading generic YOLOv8n (COCO pretrained) ...")
        _model = YOLO("yolov8n.pt")
        _model_used = "generic-yolov8"
        log.info("✓ Generic YOLOv8n loaded.")
        return True
    except Exception as e:
        log.warning(f"Generic YOLOv8n failed: {e}")
        return False


def _load_model():
    """Walk the fallback chain until one level succeeds."""
    global _model, _model_used
    if _try_load_custom():
        return
    if _try_load_hf_shelf():
        return
    if _try_load_generic_yolo():
        return
    # Level 4 — classical CV always works (no weights needed)
    _model = None
    _model_used = "classical-cv"
    log.info("✓ Using classical CV contour detector (final fallback).")


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
_load_model()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    # API_SPEC.md §2: exactly {"status": "ok", "model_loaded": true}
    return jsonify({
        "status":       "ok",
        "model_loaded": _model_used is not None,
    }), 200


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "Missing required field: image_base64"}), 400

    image_base64 = data["image_base64"]

    # Validate the image before inference
    try:
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        width, height = pil_img.size
    except Exception as e:
        return jsonify({"error": f"Invalid image_base64: {e}"}), 400

    try:
        if _model_used == "classical-cv":
            import classical_detector
            detections, width, height = classical_detector.detect(
                image_base64, conf_threshold=CONF_THRESHOLD
            )
        else:
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
