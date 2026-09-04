"""
Rate-Card / Price-Tag Filtering Module

Removes non-product detections (price tags, rate cards, shelf edge labels,
promotional stickers) from the detector's raw output before grouping.

Three complementary stages run in sequence — cheap enough for the full set of
boxes on a shelf image in < 5 ms total:

  Stage A — Geometric heuristics (always on, ~free)
      Flags boxes that are too flat (wide relative to height) AND too short
      in absolute terms — real products are taller than price tags.

  Stage B — Color heuristics (on by default, cheap)
      Checks the fraction of crop pixels in orange/red/yellow HSV ranges.
      Price tags in retail images are almost always these attention colours.

  Stage C — Row/position clustering (catches edge cases)
      Groups detections into horizontal shelf rows by y-center.
      Within each row flags boxes whose height is far below the row median
      AND whose y-center sits at the bottom edge of the row — i.e. the
      shelf lip where price strips live.

Combination rule (avoids false positives on legitimate small/bright products):
    DROP if: Stage_A AND (Stage_B OR Stage_C)

All thresholds are read from environment variables with sensible defaults,
so they can be calibrated without code changes:
    RATECARD_MAX_ASPECT         (default 2.5)
    RATECARD_MAX_HEIGHT_FRAC    (default 0.06)
    RATECARD_MIN_COLOR_FRACTION (default 0.60)
    RATECARD_ROW_HEIGHT_RATIO   (default 0.50)
    RATECARD_DEBUG              (default false) — attach filter_reason to drops
"""

import os
import numpy as np
from PIL import Image

# ── Threshold configuration ───────────────────────────────────────────────────
MAX_ASPECT         = float(os.environ.get("RATECARD_MAX_ASPECT",          "2.5"))
MAX_HEIGHT_FRAC    = float(os.environ.get("RATECARD_MAX_HEIGHT_FRAC",     "0.06"))
MIN_COLOR_FRAC     = float(os.environ.get("RATECARD_MIN_COLOR_FRACTION",  "0.60"))
ROW_HEIGHT_RATIO   = float(os.environ.get("RATECARD_ROW_HEIGHT_RATIO",    "0.50"))
DEBUG              = os.environ.get("RATECARD_DEBUG", "false").lower() == "true"

# HSV ranges for "tag-like" colours (orange, red, yellow)
# Format: [(h_lo, h_hi, s_lo, v_lo), ...]  — h in [0,180] OpenCV convention
_TAG_HSV_RANGES = [
    (0,   15,  120, 80),   # red-orange low hue
    (165, 180, 120, 80),   # red wrap-around high hue
    (15,  35,  120, 80),   # orange
    (25,  45,  100, 80),   # yellow-orange
    (45,  65,  100, 80),   # yellow
]


# ── Internal helpers ──────────────────────────────────────────────────────────
def _box_wh(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x2 - x1), (y2 - y1)


def _stage_a(box: list[float], img_h: int) -> bool:
    """True = geometrically 'flat and short' → rate-card candidate."""
    bw, bh = _box_wh(box)
    if bh <= 0:
        return False
    aspect = bw / bh
    height_frac = bh / img_h
    return aspect > MAX_ASPECT and height_frac < MAX_HEIGHT_FRAC


def _stage_b(image: Image.Image, box: list[float]) -> bool:
    """True = crop is predominantly tag-coloured (orange/red/yellow)."""
    try:
        import cv2
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image.crop((x1, y1, x2, y2))
        if crop.width < 2 or crop.height < 2:
            return False
        crop_np = np.array(crop.convert("RGB"))
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        total_px = h.size
        tag_px = 0
        for h_lo, h_hi, s_lo, v_lo in _TAG_HSV_RANGES:
            mask = (h >= h_lo) & (h <= h_hi) & (s >= s_lo) & (v >= v_lo)
            tag_px += int(mask.sum())
        return (tag_px / total_px) >= MIN_COLOR_FRAC
    except Exception:
        return False


def _cluster_rows(detections: list[dict], img_h: int) -> dict[int, list[int]]:
    """
    Group detection indices into horizontal shelf rows by y-center.
    Simple gap-based binning: a new row starts when the y-center gap
    exceeds 10 % of image height.
    Returns {row_id: [det_index, ...]}
    """
    if not detections:
        return {}

    indexed = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        y_center = (y1 + y2) / 2.0
        indexed.append((y_center, i))
    indexed.sort(key=lambda t: t[0])

    gap = img_h * 0.10
    rows: dict[int, list[int]] = {}
    row_id = 0
    prev_y = indexed[0][0]
    rows[row_id] = [indexed[0][1]]
    for y_center, idx in indexed[1:]:
        if y_center - prev_y > gap:
            row_id += 1
        rows.setdefault(row_id, []).append(idx)
        prev_y = y_center
    return rows


def _stage_c(det_idx: int, detections: list[dict], rows: dict[int, list[int]], img_h: int) -> bool:
    """
    True = box height is far below its row's median AND sits at the bottom
    edge of the row band — the classic shelf-lip position of a price strip.
    """
    # Find which row this detection belongs to
    containing_row = None
    for row_id, indices in rows.items():
        if det_idx in indices:
            containing_row = row_id
            break
    if containing_row is None:
        return False

    row_indices = rows[containing_row]
    if len(row_indices) < 2:
        return False  # can't compute a meaningful row median

    row_heights = []
    row_y_bottoms = []
    for ri in row_indices:
        bx1, by1, bx2, by2 = detections[ri]["box"]
        row_heights.append(by2 - by1)
        row_y_bottoms.append(by2)

    median_h = float(np.median(row_heights))
    if median_h <= 0:
        return False

    bx1, by1, bx2, by2 = detections[det_idx]["box"]
    box_h = by2 - by1

    # Height is suspiciously short relative to row median
    height_flag = box_h < ROW_HEIGHT_RATIO * median_h

    # y-position sits at or near the bottom edge of the row
    row_bottom = max(row_y_bottoms)
    position_flag = by2 >= (row_bottom - median_h * 0.25)

    return height_flag and position_flag


# ── Public API ────────────────────────────────────────────────────────────────
def remove_rate_cards(
    image: Image.Image,
    detections: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Partition detections into (kept, dropped).

    Parameters
    ----------
    image      : PIL Image (RGB) of the full shelf photo
    detections : list of dicts with at least {"box": [x1,y1,x2,y2], "score": float}

    Returns
    -------
    kept    : detections that are real products (original dicts, unmodified,
              unless DEBUG=true where a filter_reason=None is attached)
    dropped : rate-card detections (original dicts, with filter_reason if DEBUG)

    Combination rule:  DROP if Stage_A AND (Stage_B OR Stage_C)
    """
    if not detections:
        return [], []

    img_w, img_h = image.size
    rows = _cluster_rows(detections, img_h)

    kept, dropped = [], []

    for i, det in enumerate(detections):
        box = det["box"]

        flag_a = _stage_a(box, img_h)

        if not flag_a:
            # No geometric flag → keep without running B or C
            if DEBUG:
                det = {**det, "filter_reason": None}
            kept.append(det)
            continue

        flag_b = _stage_b(image, box)
        flag_c = _stage_c(i, detections, rows, img_h)

        if flag_b or flag_c:
            # Geometric + corroborating signal → drop
            if DEBUG:
                reason_parts = []
                reason_parts.append("aspect")
                if flag_b:
                    reason_parts.append("color")
                if flag_c:
                    reason_parts.append("row_position")
                det = {**det, "filter_reason": "+".join(reason_parts)}
            dropped.append(det)
        else:
            # Geometric flag only — insufficient evidence, keep
            if DEBUG:
                det = {**det, "filter_reason": None}
            kept.append(det)

    return kept, dropped
