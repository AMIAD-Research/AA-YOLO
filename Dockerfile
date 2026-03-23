# ============================================================
# AA-YOLO Dockerfile - Multi-stage build with CUDA support
# ============================================================

# --- Base stage: CUDA runtime ---
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Development stage ---
FROM base AS dev

RUN pip install --no-cache-dir \
    pytest \
    ruff \
    black \
    ipython \
    jupyter

COPY . .

CMD ["bash"]

# --- Production stage ---
FROM base AS prod

COPY . .

# Default: run inference
ENTRYPOINT ["python", "detect.py"]
CMD ["--weights", "best_model_AA_YOLOv7t/irstd1k_best.pt", "--source", "data/datasets/", "--img-size", "640"]
