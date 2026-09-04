FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# outputs/ is a shared volume mounted by docker-compose — not stored in image
VOLUME ["/app/outputs"]

EXPOSE 5000

CMD ["gunicorn", "--workers=2", "--threads=4", "--timeout=120", "--bind=0.0.0.0:5000", "app:app"]
