FROM python:3.11-slim

WORKDIR /app

# System deps required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# weights/best.pt is NOT baked into the image.
# It is bind-mounted read-only at runtime via docker-compose.yml:
#   volumes:
#     - ./detector_service/weights:/app/weights:ro

EXPOSE 5001

CMD ["gunicorn", "--workers=2", "--threads=2", "--timeout=120", "--bind=0.0.0.0:5001", "app:app"]
