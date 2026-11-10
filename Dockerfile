# Corrective RAG (CRAG) Engine Dockerfile
# Supports both Offline Ingestion and Online Question Answering

FROM python:3.11-slim-bookworm

# Python runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for Docling, OpenCV, and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application directories
RUN mkdir -p /app/data /app/storage /app/tmp

# Copy project source
COPY src/ /app/src/
COPY main.py /app/main.py
COPY ingest.py /app/ingest.py
COPY query.py /app/query.py
COPY .env.example /app/.env.example

ENTRYPOINT ["python"]
CMD ["main.py"]
