FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 from deadsnakes PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev python3-pip \
      bash git git-lfs wget curl procps \
      build-essential cmake g++ && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# PyTorch with CUDA 12.1 support (use python3 -m pip to ensure correct Python)
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Pre-built llama-cpp-python with CUDA 12.1 (avoids compiling from source)
RUN python3 -m pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download embedding model (~470MB) so first request isn't slow
ENV HF_HOME=/app/.cache/huggingface
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Pre-download Qwen 2.5-3B quantized GGUF (~2GB) for local LLM inference
RUN mkdir -p /app/models && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/app/models')"

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
