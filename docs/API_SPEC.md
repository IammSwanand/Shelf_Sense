# ShelfSense - API Specification

> **Tip**: Press `Ctrl + Shift + V` (or `Cmd + Shift + V` on macOS) to open the rendered Markdown Preview in your IDE.

This document defines the HTTP endpoints across the three pipeline microservices.

---

## 1. Flask Orchestrator (`Port 5000`)

The main entrypoint coordinating detection, rate-card filtering, grouping, and visualization.

### `POST /api/analyze`
Processes an input image through the complete pipeline.

**Request**
- `multipart/form-data` with form field `image` (JPEG / PNG file), OR
- `application/json`:
```json
{
  "image_base64": "<base64_string>"
}
```

**Response (`200 OK`)**
```json
{
  "image_id": "f71d7060ad03",
  "width": 1080,
  "height": 1920,
  "detections": [
    {
      "id": 0,
      "box": [349.0, 1331.9, 507.8, 1429.0],
      "det_score": 0.82,
      "group_id": 10
    }
  ],
  "num_products": 31,
  "num_groups": 11,
  "num_filtered_ratecards": 11,
  "model_used": "sku110k-yolo11-s640",
  "visualization_url": "/outputs/f71d7060ad03_viz.jpg",
  "latency_ms": 3025.0
}
```

### `GET /outputs/<filename>`
Serves the generated visualization JPEG image.

### `GET /`
Serves the interactive web dashboard for drag-and-drop image uploads, live stage tracking, and brand share-of-shelf analytics.

### Error Codes
| Status | Condition |
|---|---|
| `400 Bad Request` | Missing or unreadable image file/base64 |
| `502 Bad Gateway` | Downstream Detector or Grouping service unreachable |


---

## 2. Detector Service (`Port 5001`)

Wraps the YOLO11s SKU-110K detector to localize all product instances.

### `POST /detect`

**Request**
```json
{
  "image_base64": "<base64_string>"
}
```

**Response (`200 OK`)**
```json
{
  "width": 1080,
  "height": 1920,
  "detections": [
    {
      "box": [349.0, 1331.9, 507.8, 1429.0],
      "score": 0.82
    }
  ],
  "model_used": "sku110k-yolo11-s640"
}
```

---

## 3. Grouping Service (`Port 5002`)

Extracts dual-scale DINOv2 vision embeddings and 3D HSV color features, then clusters products into brand families using Agglomerative Linkage.

### `POST /group`

**Request**
```json
{
  "image_base64": "<base64_string>",
  "detections": [
    {
      "box": [349.0, 1331.9, 507.8, 1429.0],
      "score": 0.82
    }
  ]
}
```

**Response (`200 OK`)**
```json
{
  "num_groups": 11,
  "calibrated_threshold": 0.34,
  "groups": [
    {
      "box": [349.0, 1331.9, 507.8, 1429.0],
      "score": 0.82,
      "group_id": 10
    }
  ]
}
```

---

## 4. Health Checks

Every service exposes a lightweight health endpoint for Docker orchestration:

| Endpoint | Method | Response |
|---|:---:|---|
| `http://localhost:5000/health` | GET | `{"status": "ok"}` |
| `http://localhost:5001/health` | GET | `{"status": "ok"}` |
| `http://localhost:5002/health` | GET | `{"status": "ok"}` |
