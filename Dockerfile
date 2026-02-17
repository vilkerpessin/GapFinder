FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash git git-lfs wget curl procps \
      build-essential cmake g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (avoids pulling the 2GB CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download embedding model (~470MB) so first request isn't slow
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Pre-download Qwen 2.5-3B quantized GGUF (~2GB) for local LLM inference
RUN mkdir -p /app/models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/app/models')"

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
