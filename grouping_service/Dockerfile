FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configurable PyTorch wheel index: defaults to CPU-only (lightweight, ~1.5GB image).
# Set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 for CUDA GPU support.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=180", "--bind=0.0.0.0:5002", "app:app"]
