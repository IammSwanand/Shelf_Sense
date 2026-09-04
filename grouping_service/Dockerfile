FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# ── PyTorch: choose ONE of the two lines below ───────────────────────────────
# CPU-only (default, smaller image ~2GB):
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
# GPU / CUDA 12.1 (uncomment + comment out the CPU line above for GPU deployments):
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
#     && pip install --no-cache-dir -r requirements.txt

COPY . .

# DINOv2 weights are downloaded from HuggingFace on first request
# and cached in HF_HOME (default ~/.cache/huggingface).
# To pre-cache at build time, uncomment:
# RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
#     AutoImageProcessor.from_pretrained('facebook/dinov2-small'); \
#     AutoModel.from_pretrained('facebook/dinov2-small')"

EXPOSE 5002

CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=180", "--bind=0.0.0.0:5002", "app:app"]
