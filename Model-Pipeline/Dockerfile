# syntax=docker/dockerfile:1.7
# Distillate training image: the model pipeline's single entry point, src.pipeline.run
# (work plan v4 §5.1). The same image runs locally, on Vertex AI (one T4) and in CI.
#
# Build from the repository root (the pipeline reads Data-Pipeline/data/splits, and DVC reads
# need the Git and DVC metadata):
#
#     docker build -f Model-Pipeline/Dockerfile -t distillate-train .
#
# Run (a GPU, the store key for the soft labels, the processed data mounted or read by DVC):
#
#     docker run --gpus all -e GCS_TEACHER_KEY="$(cat key.json)" \
#         -v "$PWD/Data-Pipeline/data/processed:/app/Data-Pipeline/data/processed" \
#         distillate-train --set data.source=official run_name=distilbert-mixed ...
#
# Checks, on CPU and offline (CI runs both after building, model-pipeline-ci.yml):
#
#     docker run --rm --entrypoint python distillate-train -m src.pipeline.container_check
#     docker run --rm --entrypoint python distillate-train -m pytest tests/pipeline -q
ARG BASE_PLATFORM=linux/amd64
FROM --platform=${BASE_PLATFORM} python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface

# git: DVC reads by revision, and each run records its code commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch with the CUDA 12.6 runtime (NVIDIA driver 525 or newer, which Vertex's T4 machines
# have), not the PyPI wheel's CUDA 13 (driver 580 or newer, or training silently falls back
# to CPU). It installs first, so the pinned torch==2.14.1 below is already met.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
COPY Model-Pipeline/requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN pip install --index-url "${TORCH_INDEX}" "torch==$(sed -n 's/^torch==//p' /tmp/requirements-pipeline.txt)" \
    && pip install -r /tmp/requirements-pipeline.txt

# The repository: only what Dockerfile.dockerignore lets in (code, configs, artifacts, tests,
# the id lists, Git and DVC metadata); never data, keys or outputs.
WORKDIR /app
COPY . /app
# /app belongs to root: the system-wide safe.directory lets the non-root user run git (the
# run's code_commit), and DVC's local state (.dvc/tmp) must be writable for load_data(rev=...).
RUN git config --system --add safe.directory /app

RUN useradd --create-home --uid 1000 distillate \
    && mkdir -p /cache/huggingface /app/Model-Pipeline/outputs /app/Model-Pipeline/mlruns \
       /app/Model-Pipeline/reports /app/.dvc/tmp \
    && chown -R distillate /cache /app/Model-Pipeline/outputs /app/Model-Pipeline/mlruns \
       /app/Model-Pipeline/reports /app/.dvc
USER distillate

WORKDIR /app/Model-Pipeline
ENTRYPOINT ["python", "-m", "src.pipeline.run"]
CMD ["--config", "configs/student.yaml"]
