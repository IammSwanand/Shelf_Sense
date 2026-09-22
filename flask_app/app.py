"""
Flask Orchestrator: Main entrypoint for the ShelfSense Retail Shelf AI pipeline.

Routes:
  GET  /               HTML upload UI
  GET  /health         {"status": "ok"}
  POST /api/analyze    Full pipeline: detect, filter, group, visualize
  GET  /outputs/<fn>   Serve saved visualization images
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import base64
import hashlib
import json
import logging
import time
import datetime
from io import BytesIO
from pathlib import Path

import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw, ImageFont

from filtering import remove_rate_cards

# Create a global session to reuse TCP connections and bypass Windows proxy autodiscovery (WPAD) delays
session = requests.Session()
session.trust_env = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Configuration
DETECTOR_URL       = os.environ.get("DETECTOR_URL",       "http://127.0.0.1:5001")
GROUPING_URL       = os.environ.get("GROUPING_URL",       "http://127.0.0.1:5002")
DOWNSTREAM_TIMEOUT = int(os.environ.get("DOWNSTREAM_TIMEOUT", "60"))
PORT               = int(os.environ.get("PORT", "5000"))
OUTPUTS_DIR        = Path(os.environ.get("OUTPUTS_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs")))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# High-contrast color palette for brand group labels
_PALETTE = [
    "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16", 
    "#F97316", "#6366F1", "#14B8A6", "#D946EF", "#3B82F6", "#E11D48", "#EAB308", "#64748B",
]

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _image_from_request() -> tuple[Image.Image, str, str]:
    if request.content_type and "multipart/form-data" in request.content_type:
        f = request.files.get("image")
        if not f:
            return None, "", "No 'image' field in multipart form"
        raw = f.read()
        if not raw:
            return None, "", "Empty image file"
        image_base64 = base64.b64encode(raw).decode()
    else:
        data = request.get_json(silent=True) or {}
        image_base64 = data.get("image_base64", "")
        if not image_base64:
            return None, "", "No image provided (send multipart 'image' or JSON 'image_base64')"
        raw = base64.b64decode(image_base64)

    try:
        pil_img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        return None, "", f"Cannot decode image: {e}"

    return pil_img, image_base64, ""

def _call_detector(image_base64: str) -> tuple[dict, str]:
    try:
        resp = session.post(
            f"{DETECTOR_URL}/detect",
            json={"image_base64": image_base64},
            timeout=DOWNSTREAM_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}, f"detector returned {resp.status_code}: {resp.text[:200]}"
        return resp.json(), ""
    except requests.exceptions.RequestException as e:
        return {}, f"detector unreachable: {e}"

def _call_grouping(image_base64: str, detections: list[dict]) -> tuple[dict, str]:
    try:
        resp = session.post(
            f"{GROUPING_URL}/group",
            json={"image_base64": image_base64, "detections": detections},
            timeout=DOWNSTREAM_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}, f"grouping returned {resp.status_code}: {resp.text[:200]}"
        return resp.json(), ""
    except requests.exceptions.RequestException as e:
        return {}, f"grouping unreachable: {e}"

def _draw_visualization(image: Image.Image, groups: list[dict], image_id: str) -> str:
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    stroke = max(3, img_w // 350)
    font_size = max(18, img_w // 48)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()

    for item in groups:
        gid = item.get("group_id", 0)
        box = item["box"]
        color_hex = _PALETTE[gid % len(_PALETTE)]
        color_rgb = _hex_to_rgb(color_hex)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=stroke)
        label = f"Group {gid}"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            lw, lh = int(len(label) * font_size * 0.6), font_size

        pad_x = max(6, font_size // 4)
        pad_y = max(4, font_size // 5)
        badge_top = y1 - lh - (pad_y * 2)
        badge_bottom = y1

        if badge_top < 0:
            badge_top = y1
            badge_bottom = y1 + lh + (pad_y * 2)
            text_y = badge_top + pad_y
        else:
            text_y = badge_top + pad_y

        draw.rectangle([x1, badge_top, x1 + lw + (pad_x * 2), badge_bottom], fill=color_rgb)
        draw.text((x1 + pad_x, text_y), label, fill=(255, 255, 255), font=font)

    filename = f"{image_id}_viz.jpg"
    save_path = OUTPUTS_DIR / filename
    image.save(save_path, "JPEG", quality=92)
    log.info(f"Visualization saved: {save_path}")
    return filename

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(str(OUTPUTS_DIR), filename)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    t_start = time.perf_counter()

    pil_img, image_base64, err = _image_from_request()
    if err:
        return jsonify({"error": err}), 400

    width, height = pil_img.size
    base_hash = hashlib.md5(image_base64[:512].encode()).hexdigest()[:8]
    image_id = f"{base_hash}_{int(time.time())}"

    t_det_start = time.perf_counter()
    det_resp, err = _call_detector(image_base64)
    t_det_ms = round((time.perf_counter() - t_det_start) * 1000, 1)
    if err:
        return jsonify({"error": f"detector service error: {err}"}), 502

    raw_detections = det_resp.get("detections", [])
    model_used     = det_resp.get("model_used", "unknown")

    t_filt_start = time.perf_counter()
    kept, dropped = remove_rate_cards(pil_img, raw_detections)
    t_filt_ms = round((time.perf_counter() - t_filt_start) * 1000, 1)
    num_filtered  = len(dropped)

    t_grp_start = time.perf_counter()
    if kept:
        grp_resp, err = _call_grouping(image_base64, kept)
        if err:
            return jsonify({"error": f"grouping service error: {err}"}), 502
        groups      = grp_resp.get("groups", [])
        num_groups  = grp_resp.get("num_groups", 0)
        calibrated_threshold = grp_resp.get("calibrated_threshold", 0.0)
    else:
        grp_resp = {}
        groups, num_groups, calibrated_threshold = [], 0, 0.0
    t_grp_ms = round((time.perf_counter() - t_grp_start) * 1000, 1)

    t_viz_start = time.perf_counter()
    viz_img  = pil_img.copy()
    viz_file = _draw_visualization(viz_img, groups, image_id)
    t_viz_ms = round((time.perf_counter() - t_viz_start) * 1000, 1)

    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    detections_out = [
        {
            "id":        i,
            "box":       g["box"],
            "det_score": g.get("score", 0.0),
            "group_id":  g["group_id"],
        }
        for i, g in enumerate(groups)
    ]

    response_data = {
        "image_id":               image_id,
        "width":                  width,
        "height":                 height,
        "detections":             detections_out,
        "num_products":           len(detections_out),
        "num_groups":             num_groups,
        "num_filtered_ratecards": num_filtered,
        "model_used":             model_used,
        "visualization_url":      f"/outputs/{viz_file}",
        "latency_ms":             latency_ms,
        "timing_breakdown_ms": {
            "detector": t_det_ms,
            "filtering": t_filt_ms,
            "grouping": t_grp_ms,
            "visualization": t_viz_ms,
            "orchestrator_overhead": round(latency_ms - (t_det_ms + t_filt_ms + t_grp_ms + t_viz_ms), 1),
            "detector_internal": det_resp.get("timing_internal", {}),
            "grouping_internal": grp_resp.get("timing_internal", {}),
        },
        "config": {
            "clustering_method": "Agglomerative Clustering (Average Linkage)",
            "calibrated_threshold": calibrated_threshold,
        },
        "timestamp":              datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    metadata_path = OUTPUTS_DIR / f"{image_id}_metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(response_data, f, indent=2)
        log.info(f"Metadata saved: {metadata_path}")
    except Exception as e:
        log.error(f"Failed to save metadata: {e}")

    return jsonify(response_data), 200

if __name__ == "__main__":
    from waitress import serve
    log.info("Starting Waitress production server on port 5000...")
    serve(app, host="0.0.0.0", port=5000)
