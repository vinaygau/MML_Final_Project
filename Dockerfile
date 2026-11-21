FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/var/lib/cinecomic/huggingface

WORKDIR /app

# FFmpeg is required by ingestion. The shared libraries support OpenCV and
# InsightFace; compiler tools are needed if pip has to build a source wheel.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-deploy.txt ./

# Install CPU PyTorch wheels explicitly: this deployment image targets a
# single CPU server and should not pull CUDA runtimes.
RUN python -m pip install --upgrade pip \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio \
    && python -m pip install -r requirements-deploy.txt

COPY . .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin cinecomic \
    && mkdir -p /app/data /app/output /app/results /app/.insightface \
        /var/lib/cinecomic/huggingface \
    && chown -R cinecomic:cinecomic /app /var/lib/cinecomic

USER cinecomic

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/healthz', timeout=4)"

# Keep one WSGI process: the pipeline's active-child registry is process-local.
# Threads handle UI polling while a pipeline runs in its child process.
CMD ["waitress-serve", "--listen=0.0.0.0:8501", "--threads=4", "src.web_server:app"]
