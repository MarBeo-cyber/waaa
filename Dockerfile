FROM python:3.11-slim

LABEL org.opencontainers.image.title="WAAA"
LABEL org.opencontainers.image.description="Weak Autopoietic Artificial Agent — ML system"
LABEL org.opencontainers.image.source="https://github.com/MarBeo-cyber/waaa"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model persistence directory. main_ml.py reads WAAA_MODEL_DIR;
# scripts/run_ml.sh mounts ./waaa_models here.
RUN mkdir -p /app/waaa_models
ENV WAAA_MODEL_DIR=/app/waaa_models

EXPOSE 5001

# Default: REST API server
CMD ["python", "main_ml.py", "server"]
