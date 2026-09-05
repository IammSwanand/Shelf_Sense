"""
Flask Orchestrator  main entrypoint for the Infilect AI pipeline.

Routes:
  GET  /               HTML upload UI
  GET  /health         {"status": "ok"}
  POST /api/analyze    full pipeline: detect  filter  group  visualize
  GET  /outputs/<fn>   serve saved visualization images

Environment variables:
  DETECTOR_URL        URL of the detector service  (default: http://localhost:5001)
  GROUPING_URL        URL of the grouping service  (default: http://localhost:5002)
  DOWNSTREAM_TIMEOUT  Seconds before upstream call times out (default: 60)
  PORT                Port to listen on (default: 5000)
  OUTPUTS_DIR         Directory to save visualizations (default: /app/outputs)
"""

import os
import base64
import hashlib
import logging
import time
from io import BytesIO
from pathlib import Path

import requests as http
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw, ImageFont

from filtering import remove_rate_cards

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

#  Config 
DETECTOR_URL       = os.environ.get("DETECTOR_URL",       "http://localhost:5001")
GROUPING_URL       = os.environ.get("GROUPING_URL",       "http://localhost:5002")
DOWNSTREAM_TIMEOUT = int(os.environ.get("DOWNSTREAM_TIMEOUT", "60"))
PORT               = int(os.environ.get("PORT", "5000"))
OUTPUTS_DIR        = Path(os.environ.get("OUTPUTS_DIR",  "/app/outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

#  Colour palette (high-contrast vibrant colors per brand group) 
_PALETTE = [
    "#2563EB",  # Royal Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
    "#84CC16",  # Lime
    "#F97316",  # Orange
    "#6366F1",  # Indigo
    "#14B8A6",  # Teal
    "#D946EF",  # Fuchsia
    "#3B82F6",  # Sky Blue
    "#E11D48",  # Rose
    "#EAB308",  # Yellow
    "#64748B",  # Slate
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


#  Helpers 
def _image_from_request() -> tuple[Image.Image, str, str]:
    """
    Extract the image from multipart or JSON body.
    Returns (pil_image, image_base64, error_message).
    error_message is "" on success.
    """
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
    """POST to detector service. Returns (response_dict, error_string)."""
    try:
        resp = http.post(
            f"{DETECTOR_URL}/detect",
            json={"image_base64": image_base64},
            timeout=DOWNSTREAM_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}, f"detector returned {resp.status_code}: {resp.text[:200]}"
        return resp.json(), ""
    except http.exceptions.RequestException as e:
        return {}, f"detector unreachable: {e}"


def _call_grouping(image_base64: str, detections: list[dict]) -> tuple[dict, str]:
    """POST to grouping service. Returns (response_dict, error_string)."""
    try:
        resp = http.post(
            f"{GROUPING_URL}/group",
            json={"image_base64": image_base64, "detections": detections},
            timeout=DOWNSTREAM_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}, f"grouping returned {resp.status_code}: {resp.text[:200]}"
        return resp.json(), ""
    except http.exceptions.RequestException as e:
        return {}, f"grouping unreachable: {e}"


def _draw_visualization(
    image: Image.Image,
    groups: list[dict],
    image_id: str,
) -> str:
    """
    Draw clean, high-contrast bounding box outlines (one colour per brand group)
    without opaque interior fills so products remain completely visible.
    Saves the result as JPEG to OUTPUTS_DIR.
    """
    draw = ImageDraw.Draw(image)

    # Adaptive stroke & font sizing based on image resolution
    img_w, img_h = image.size
    stroke = max(2, img_w // 400)
    font_size = max(13, img_w // 75)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    for item in groups:
        gid = item.get("group_id", 0)
        box = item["box"]
        color_hex = _PALETTE[gid % len(_PALETTE)]
        color_rgb = _hex_to_rgb(color_hex)

        x1, y1, x2, y2 = [int(round(v)) for v in box]

        # Draw clean, crisp outline (NO interior fill so product is 100% visible)
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=stroke)

        label = f"Group {gid}"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            lw, lh = len(label) * 8, font_size

        pad = 3
        badge_top = y1 - lh - (pad * 2)
        badge_bottom = y1

        # Keep badge within image bounds
        if badge_top < 0:
            badge_top = y1
            badge_bottom = y1 + lh + (pad * 2)
            text_y = badge_top + pad
        else:
            text_y = badge_top + pad

        draw.rectangle([x1, badge_top, x1 + lw + (pad * 2), badge_bottom], fill=color_rgb)
        draw.text((x1 + pad, text_y), label, fill=(255, 255, 255), font=font)

    filename = f"{image_id}_viz.jpg"
    save_path = OUTPUTS_DIR / filename
    image.save(save_path, "JPEG", quality=92)
    log.info(f"Visualization saved: {save_path}")
    return filename


#  Routes 
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

    #  1. Extract image 
    pil_img, image_base64, err = _image_from_request()
    if err:
        return jsonify({"error": err}), 400

    width, height = pil_img.size
    image_id = hashlib.md5(image_base64[:512].encode()).hexdigest()[:12]

    #  2. Call detector 
    det_resp, err = _call_detector(image_base64)
    if err:
        return jsonify({"error": f"detector service error: {err}"}), 502

    raw_detections = det_resp.get("detections", [])
    model_used     = det_resp.get("model_used", "unknown")

    #  3. Filter rate cards (in-process, no network hop) 
    kept, dropped = remove_rate_cards(pil_img, raw_detections)
    num_filtered  = len(dropped)

    #  4. Call grouping (on filtered detections only) 
    if kept:
        grp_resp, err = _call_grouping(image_base64, kept)
        if err:
            return jsonify({"error": f"grouping service error: {err}"}), 502
        groups      = grp_resp.get("groups", [])
        num_groups  = grp_resp.get("num_groups", 0)
    else:
        groups, num_groups = [], 0

    #  5. Draw and save visualization 
    viz_img  = pil_img.copy()
    viz_file = _draw_visualization(viz_img, groups, image_id)

    #  6. Assemble final response 
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

    return jsonify({
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
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
