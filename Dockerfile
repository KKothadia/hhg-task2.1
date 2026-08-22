# ============================================
# HHG Voice RAG — Production Dockerfile
# Hugging Face Spaces Docker Deployment
# ============================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY hhg-task2/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Pre-cache HuggingFace SentenceTransformer model into image with lean memory footprint
ENV HF_HOME=/root/.cache/huggingface \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application source, scripts, and dataset
COPY hhg-task2/src/ ./src/
COPY hhg-task2/scripts/ ./scripts/
COPY hhg-task2/data/ ./data/

# Build multilingual index into image if not provided by git
RUN if [ ! -f data/numpy_store.pkl ]; then python scripts/ingest_multilingual.py; else echo "Index data/numpy_store.pkl already exists, skipping build."; fi

# Create docs directory
RUN mkdir -p /app/docs

# Expose default application ports (8080 for Railway, 7860 for HF Spaces)
EXPOSE 8080 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, httpx; port = os.environ.get('PORT', '8080'); r = httpx.get(f'http://localhost:{port}/health'); assert r.status_code == 200"

# Run the server with single worker and dynamic port binding (default: 8080)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
