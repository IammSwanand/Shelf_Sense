FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only wheel — keeps image small (no CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# DINOv2 weights are downloaded from HuggingFace on first request
# and cached in HF_HOME (default ~/.cache/huggingface).
# To pre-cache at build time, uncomment:
# RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
#     AutoImageProcessor.from_pretrained('facebook/dinov2-small'); \
#     AutoModel.from_pretrained('facebook/dinov2-small')"

EXPOSE 5002

CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=180", "--bind=0.0.0.0:5002", "app:app"]
