r"""
Quick smoke test for filtering.py.
Run from: d:\Infilect\infilect_pipeline\
Usage: .venv\Scripts\python scripts\test_filtering.py
"""
import sys, os, base64, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))

import requests
from PIL import Image, ImageDraw
from filtering import remove_rate_cards

SAMPLE = r"d:\Infilect\infilect_pipeline\sample_images\128008.jpg"
DETECTOR_URL = "http://localhost:5001"
OUT_DIR = r"d:\Infilect\infilect_pipeline\outputs"

os.makedirs(OUT_DIR, exist_ok=True)
os.environ["RATECARD_DEBUG"] = "true"

# 1. Get raw detections from the live detector
print(f"[1] Sending {os.path.basename(SAMPLE)} to detector...")
img_b64 = base64.b64encode(open(SAMPLE, "rb").read()).decode()
resp = requests.post(f"{DETECTOR_URL}/detect", json={"image_base64": img_b64}, timeout=60)
resp.raise_for_status()
data = resp.json()
raw_dets = data["detections"]
print(f"    Raw detections: {len(raw_dets)} (model: {data['model_used']})")

# 2. Run the filter
pil_img = Image.open(SAMPLE).convert("RGB")
kept, dropped = remove_rate_cards(pil_img, raw_dets)
print(f"[2] After filtering: kept={len(kept)}  dropped={len(dropped)}")

# 3. Draw kept=green, dropped=red for visual inspection
canvas = pil_img.copy()
draw = ImageDraw.Draw(canvas)
for det in kept:
    x1,y1,x2,y2 = [int(v) for v in det["box"]]
    draw.rectangle([x1,y1,x2,y2], outline=(0,200,0), width=2)
for det in dropped:
    x1,y1,x2,y2 = [int(v) for v in det["box"]]
    draw.rectangle([x1,y1,x2,y2], outline=(220,0,0), width=2)
    reason = det.get("filter_reason", "?")
    draw.text((x1+2, y1+2), reason or "?", fill=(255,100,100))

out_path = os.path.join(OUT_DIR, "filter_debug_128008.jpg")
canvas.save(out_path, "JPEG", quality=88)
print(f"[3] Saved debug visualization -> {out_path}")

# 4. Print dropped reasons
if dropped:
    from collections import Counter
    reasons = Counter(d.get("filter_reason","unknown") for d in dropped)
    print(f"[4] Drop reasons: {dict(reasons)}")

print("\nFiltering smoke test done.")
