"""
scripts/run_all_samples.py - ShelfSense Batch Processing Script

Batch test: loops over all images in test_images/, POSTs each to
/api/analyze, and prints a summary table to stdout + saves a CSV.

Usage (with the full stack running):
    python scripts/run_all_samples.py

Optional env vars:
    BASE_URL         Flask orchestrator URL  (default: http://localhost:5000)
    TEST_IMAGES_DIR  Path to test images     (default: ./test_images)
    OUTPUT_CSV       CSV output path         (default: ./scripts/batch_results.csv)
"""

import os
import sys
import csv
import time
from pathlib import Path

import requests

ROOT_DIR    = Path(__file__).resolve().parent.parent
BASE_URL    = os.environ.get("BASE_URL",    "http://localhost:5000")
SAMPLES_DIR = Path(os.environ.get("TEST_IMAGES_DIR", os.environ.get("SAMPLES_DIR", ROOT_DIR / "test_images")))
OUTPUT_CSV  = Path(os.environ.get("OUTPUT_CSV",  ROOT_DIR / "scripts" / "batch_results.csv"))

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def analyze_image(image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/analyze",
            files={"image": (image_path.name, f, "image/jpeg")},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def main():
    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        print(f"No images found in {SAMPLES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images in {SAMPLES_DIR}")
    print(f"Target: {BASE_URL}/api/analyze\n")

    # Verify the server is up
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        assert health.json()["status"] == "ok"
        print("[OK] Flask server is up\n")
    except Exception as e:
        print(f"[FAIL] Flask server not reachable at {BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    headers = [
        "image", "num_products", "num_groups",
        "num_filtered_ratecards", "latency_ms", "model_used", "status",
    ]
    rows = []

    col_w = [40, 13, 11, 22, 12, 28, 8]
    header_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    row_fmt    = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(header_fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))

    for img_path in images:
        try:
            t0 = time.perf_counter()
            data = analyze_image(img_path)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            row = {
                "image":                  img_path.name[:38],
                "num_products":           data.get("num_products", "?"),
                "num_groups":             data.get("num_groups", "?"),
                "num_filtered_ratecards": data.get("num_filtered_ratecards", "?"),
                "latency_ms":             data.get("latency_ms", elapsed),
                "model_used":             data.get("model_used", "?")[:26],
                "status":                 "OK",
            }
        except Exception as e:
            row = {
                "image":                  img_path.name[:38],
                "num_products":           "",
                "num_groups":             "",
                "num_filtered_ratecards": "",
                "latency_ms":             "",
                "model_used":             "",
                "status":                 f"ERR: {str(e)[:20]}",
            }

        rows.append(row)
        print(row_fmt.format(*[str(row[h]) for h in headers]))

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Results saved to {OUTPUT_CSV}")

    # Sanity summary
    ok_rows = [r for r in rows if r["status"] == "OK"]
    print(f"\n Sanity checks ({'all OK' if len(ok_rows)==len(rows) else 'some failed'}) ")
    products = [r["num_products"] for r in ok_rows if isinstance(r["num_products"], int)]
    groups   = [r["num_groups"]   for r in ok_rows if isinstance(r["num_groups"], int)]
    if products:
        print(f"  Images with 0 products:               {sum(1 for p in products if p == 0)}")
        print(f"  Images where num_groups == num_products: {sum(1 for p, g in zip(products, groups) if p == g)}")
        print(f"  Avg products per image:                {sum(products)/len(products):.1f}")
        print(f"  Avg groups per image:                  {sum(groups)/len(groups):.1f}")


if __name__ == "__main__":
    main()
