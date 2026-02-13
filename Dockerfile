FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash git git-lfs wget curl procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (avoids pulling the 2GB CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download multilingual model (~470MB) so first request isn't slow
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "3", "--timeout", "300", "--preload", "app:app"]
