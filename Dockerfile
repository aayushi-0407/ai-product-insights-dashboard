FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as a non-root user; give it a
# writable home so huggingface_hub/torch cache dirs don't fail to create.
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image at build time so the container
# never needs network access to it at runtime (retriever.py loads it with
# local_files_only=True).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

COPY static/ static/
COPY data/raw/reviews.jsonl data/raw/reviews.jsonl
COPY data/processed/ data/processed/
COPY data/vector_db/ data/vector_db/

RUN chmod -R 777 /app

EXPOSE 7860

# Render (and HF Spaces) inject $PORT at runtime and require the container to
# bind to it; 7860 is only the local/default fallback. Shell form so $PORT
# expands.
CMD uvicorn ai_product_insights_dashboard.app:app --host 0.0.0.0 --port ${PORT:-7860}
