# syntax=docker/dockerfile:1.7
# Distillate data pipeline image: Airflow 3.3.2 on Python 3.12, plus a second,
# isolated Python 3.11 environment at /opt/tfenv for the two TensorFlow tasks.
#
# Pinned to linux/amd64 on purpose. TFDV publishes no Linux aarch64 wheel at any
# version, so on an Apple Silicon machine this image runs under emulation.
# Turn on "Use Rosetta for x86/amd64 emulation" in Docker Desktop settings;
# without it the build falls back to QEMU and is several times slower.
ARG BASE_PLATFORM=linux/amd64
FROM --platform=${BASE_PLATFORM} apache/airflow:3.3.2-python3.12

# Exact patch, so a rebuild months from now resolves the same core pins.
ARG AIRFLOW_VERSION=3.3.2
ARG PYTHON_VERSION=3.12
ARG CONSTRAINT_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt

USER root

# python3.11* : bookworm's system Python, used only for the /opt/tfenv venv.
# build-essential : TFDV pins pyfarmhash<0.4, which ships no Linux wheel and
#                   must be compiled. Kept in the image so later pip installs
#                   inside the container still work.
# git : DVC needs it to resolve the repo root.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
      git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# TFDS cache and the venv are written by the airflow user, not root.
RUN install -d -o airflow -g 0 -m 775 /opt/tfenv /opt/airflow/tfds

USER airflow

# ---------------------------------------------------------------------------
# 1. Main environment (Python 3.12): Airflow + every non-TensorFlow task.
# ---------------------------------------------------------------------------
COPY --chown=airflow:0 requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --constraint "${CONSTRAINT_URL}" -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# 2. TensorFlow environment (Python 3.11) at /opt/tfenv, fully isolated.
#    --system-site-packages is deliberately NOT set: this env must not see
#    Airflow's pandas 3.x / pyarrow 25.x, which TFDV cannot use.
# ---------------------------------------------------------------------------
COPY --chown=airflow:0 requirements-tf.txt /tmp/requirements-tf.txt
RUN python3.11 -m venv /opt/tfenv \
 && /opt/tfenv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && /opt/tfenv/bin/pip install --no-cache-dir -r /tmp/requirements-tf.txt

# ---------------------------------------------------------------------------
# 3. Oct 14 checkpoint, enforced at build time.
#    If TFDV cannot import here, the build fails loudly now rather than at 2am
#    in a DAG run -- that is the signal to switch the validation lane to Great
#    Expectations (risk table, owner Sai).
# ---------------------------------------------------------------------------
RUN /opt/tfenv/bin/python - <<'PY'
import sys
import tensorflow_data_validation as tfdv
import tensorflow_datasets as tfds
import tensorflow as tf
print(f"TFDV checkpoint OK | python={sys.version.split()[0]} "
      f"tfdv={tfdv.__version__} tf={tf.__version__} tfds={tfds.__version__}")
# Exercise the code path schema_stats.py actually uses, not just the import.
import pandas as pd
stats = tfdv.generate_statistics_from_dataframe(
    pd.DataFrame({"text": ["hello", "world"], "toxicity": [0, 1]})
)
schema = tfdv.infer_schema(stats)
assert {f.name for f in schema.feature} == {"text", "toxicity"}, schema
print("TFDV generate_statistics + infer_schema OK")
PY

ENV TFENV_PYTHON=/opt/tfenv/bin/python \
    PYTHONPATH=/opt/airflow/pipeline \
    TFDS_DATA_DIR=/opt/airflow/tfds

WORKDIR /opt/airflow/pipeline
