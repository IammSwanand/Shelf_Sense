"""
Classical CV contour-based product proposals.
Zero-dependency fallback (only OpenCV + NumPy).
Used as the final level in the detector fallback chain when no YOLO model
is available.
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def _decode_image(image_base64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_base64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)


def detect(image_base64: str, conf_threshold: float = 0.25) -> tuple[list[dict], int, int]:
    """
    Classical contour-based object proposals on a retail shelf image.

    Returns:
        (detections, width, height)
        detections: [{"box": [x1, y1, x2, y2], "score": float}, ...]
        Scores are synthetic (area-rank based) in [conf_threshold, 0.55].
    """
    img_rgb = _decode_image(image_base64)
    h, w = img_rgb.shape[:2]

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Bilateral filter — preserve edges while smoothing noise
    img_blur = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive threshold to segment products from shelf background
    thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10,
    )

    # Morphological close to merge nearby blobs into single products
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    img_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out noise (too small) and full-image boxes (too large)
        if area < img_area * 0.001 or area > img_area * 0.25:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        # Skip very flat boxes (shelf rails) and very tall thin boxes
        if aspect > 4.0 or aspect < 0.1:
            continue
        proposals.append({"box": [float(x), float(y), float(x + bw), float(y + bh)], "_area": area})

    if not proposals:
        return [], w, h

    # Sort largest first; assign synthetic confidence scores
    proposals.sort(key=lambda p: p["_area"], reverse=True)
    n = len(proposals)
    detections = []
    for i, p in enumerate(proposals):
        score = round(max(0.55 - (i / max(n, 1)) * 0.30, conf_threshold), 3)
        detections.append({"box": p["box"], "score": score})

    return detections, w, h
