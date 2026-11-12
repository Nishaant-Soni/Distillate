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
# Smoke test on CPU, offline (the tiny model, synthetic data):
#
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

COPY Model-Pipeline/requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN pip install -r /tmp/requirements-pipeline.txt

# The repository (see Dockerfile.dockerignore for what stays out: data, outputs, caches).
WORKDIR /app
COPY . /app
RUN git config --global --add safe.directory /app

RUN useradd --create-home --uid 1000 distillate \
    && mkdir -p /cache/huggingface /app/Model-Pipeline/outputs /app/Model-Pipeline/mlruns \
       /app/Model-Pipeline/reports \
    && chown -R distillate /cache /app/Model-Pipeline/outputs /app/Model-Pipeline/mlruns \
       /app/Model-Pipeline/reports
USER distillate

WORKDIR /app/Model-Pipeline
ENTRYPOINT ["python", "-m", "src.pipeline.run"]
CMD ["--config", "configs/student.yaml"]
