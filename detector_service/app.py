"""
Detector Service: ShelfSense microservice wrapping YOLO11-s640 (SKU-110K weights).
Accepts an image and returns product bounding boxes with confidence scores.
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

# Configuration
CONF_THRESHOLD  = float(os.environ.get("DET_CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD   = float(os.environ.get("DET_IOU_THRESHOLD", "0.45"))
IMG_SIZE        = int(os.environ.get("DET_IMG_SIZE", "640"))
PORT            = int(os.environ.get("PORT", "5001"))

# Device selection (CUDA GPU if available, else CPU)
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

# Model singleton (loaded once at startup)
_model      = None
_model_used = "sku110k-yolo11-s640"


def _find_weights_path() -> str:
    """Find the YOLO11-s640 weight file across local container/service locations."""
    candidate_paths = [
        os.environ.get("DETECTOR_WEIGHTS_PATH"),
        "/app/weights/sku110k-yolo11-s640.pt",
        os.path.join(os.path.dirname(__file__), "weights", "sku110k-yolo11-s640.pt"),
        os.path.join(os.path.dirname(__file__), "..", "sku110k-yolo11-s640.pt"),
        "/app/weights/best.pt",
        os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    ]
    for path in candidate_paths:
        if path and os.path.exists(path) and os.path.getsize(path) > 1000:
            return os.path.abspath(path)
    return os.environ.get("DETECTOR_WEIGHTS_PATH", "/app/weights/sku110k-yolo11-s640.pt")


def _download_weights_if_needed(target_path: str) -> str:
    """
    If weights already exist on disk, returns immediately (0 network overhead).
    If missing, automatically downloads from Hugging Face / configured URL.
    """
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return target_path

    default_url = "https://huggingface.co/chistopat/sku110k-yolo11-object-detector/resolve/main/weights/sku110k-yolo11-s640.pt"
    download_url = os.environ.get("WEIGHTS_DOWNLOAD_URL", "").strip() or default_url

    log.info(f"Weights file not found at '{target_path}'. Auto-downloading from Hugging Face ({download_url}) ...")
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_path = f"{target_path}.tmp"
    
    import urllib.request
    try:
        urllib.request.urlretrieve(download_url, temp_path)
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
            os.replace(temp_path, target_path)
            log.info(f"Successfully downloaded and verified weights at '{target_path}' ({os.path.getsize(target_path)} bytes).")
        else:
            raise RuntimeError("Downloaded weights file is empty or corrupted.")
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        log.error(f"Failed to auto-download weights from {download_url}: {e}")
        raise

    return target_path


def _load_model():
    """Load YOLO11-s640 SKU-110K model."""
    global _model
    from ultralytics import YOLO

    weights_path = _find_weights_path()
    
    # Only triggers download if file does not exist locally
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) <= 1000:
        weights_path = _download_weights_if_needed(weights_path)

    if not os.path.exists(weights_path) or os.path.getsize(weights_path) <= 1000:
        raise FileNotFoundError(
            f"YOLO11-s640 weights not found at '{weights_path}'. "
            f"Please ensure 'sku110k-yolo11-s640.pt' is present in 'detector_service/weights/' or set WEIGHTS_DOWNLOAD_URL."
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


#  Endpoints 
@app.route("/health", methods=["GET"])
def health():
    # API_SPEC.md 2: exactly {"status": "ok", "model_loaded": true}
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
