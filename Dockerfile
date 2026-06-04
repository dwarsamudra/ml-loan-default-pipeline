# ─────────────────────────────────────────────
# Dockerfile — Loan Default Predictor API
# Author: Anvesh Dubey
# Build:  docker build -t loan-predictor .
# Run:    docker run -p 8000:8000 loan-predictor
# ─────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="Anvesh Dubey"
LABEL description="End-to-End ML Pipeline: Loan Default Prediction API"

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Train model on startup if not already trained
RUN python src/train.py

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
