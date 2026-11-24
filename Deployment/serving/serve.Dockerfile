# syntax=docker/dockerfile:1.7
# Distillate serving image (work plan v5 §2.5, §6.1): the FastAPI app with one model package
# baked in. Tag: distillate-images/distillate-api:v<model_version>-<git_sha>.
#
# The build context is Deployment/serving, with the package (downloaded from Artifact Registry
# and checked by deploy.sh) copied to Deployment/serving/package/ first:
#
#     docker build -f serve.Dockerfile --build-arg REGISTRY_VERSION=1 -t distillate-api .
#     docker run --rm -p 8080:8080 distillate-api
#
# The build fails if the package is incomplete, a file's SHA-256 differs from meta.json, or
# the model can't score the reference texts (scorer.py --package).
ARG BASE_PLATFORM=linux/amd64
FROM --platform=${BASE_PLATFORM} python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HOME=/tmp/huggingface

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY app.py scorer.py config.yaml /app/
# The package last, so a new model version rebuilds only this layer and the check.
ARG PACKAGE=package
COPY ${PACKAGE}/ /app/package/
RUN python scorer.py --package /app/package > /dev/null

# The registry version deploy.sh resolved, for tracing an image back to the registry; the
# package's own model_version (what /health and /predict report) comes from its meta.json.
ARG REGISTRY_VERSION=unknown
LABEL org.opencontainers.image.title="distillate-api" \
      ai.distillate.registry-version="${REGISTRY_VERSION}"
ENV DISTILLATE_REGISTRY_VERSION=${REGISTRY_VERSION} \
    DISTILLATE_PACKAGE_DIR=/app/package \
    PORT=8080

RUN useradd --create-home --uid 1000 distillate
USER distillate
EXPOSE 8080

# Cloud Run sets PORT. One worker: Cloud Run scales with instances, and concurrent requests
# share the one ONNX session through FastAPI's thread pool. No access log: each /predict
# writes its own JSON line, with no text in it.
CMD ["sh", "-c", "exec uvicorn app:create_app --factory --host 0.0.0.0 --port \"${PORT}\" --no-access-log --timeout-graceful-shutdown 10"]
