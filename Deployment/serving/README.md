# Serving (`Deployment/serving/`)

The Distillate API (work plan v5 §2.4): FastAPI over ONNX Runtime, CPU only, with one model
package (§2.5) baked into the image. Cloud Run service `distillate-api`, us-east1.

| Endpoint | Returns |
|---|---|
| `POST /predict` `{"text": "..."}` | `scores`, `flagged`, `flag`, `low_confidence`, `model_version`, `latency_ms` |
| `GET /health` | `{"status": "ok", "model_version": "..."}` |
| `GET /metrics` | Prometheus text with the §2.4 metric names, for local docker-compose. **Off on Cloud Run** (404): nothing scrapes it there, since service metrics come from Cloud Monitoring and model metrics from the batch jobs (`expose_metrics` in config.yaml) |

- `flagged[label]` is `score >= thresholds.json[label]`; `flag` is any label flagged;
  `low_confidence` is any score within `low_confidence_margin` (0.1) of its threshold.
- Scores are the sigmoid of the package's ONNX logits, tokenized exactly as the model pipeline's
  `predict_onnx_logits` does (same pinned `transformers`/`tokenizers`/`onnxruntime`), one text at
  a time. Checked against a real DistilBERT package: max difference 6e-8.
- `model_version` is the package's `meta.json` value (the same string as the prediction files).
  The MLflow registry version deploy.sh resolved is in the image label
  `ai.distillate.registry-version` and the `DISTILLATE_REGISTRY_VERSION` variable.

## Privacy

Comment text is never logged or stored. Each `/predict` writes one JSON line to stdout (Cloud
Logging): `text_sha256`, `char_len`, `scores`, `flagged`, `flag`, `low_confidence`,
`model_version`, `latency_ms`, plus `severity`. Errors log the exception type only. Error
responses never echo the input. There is no access log.

## Input checks

| Request | Answer |
|---|---|
| Content-Type not `application/json` | 415 |
| Body over 64 KiB, or text over 10,000 characters | 413 |
| Not `{"text": "<string>"}`, empty text, invalid Unicode | 422 |
| Over the per-client rate limit | 429 with `Retry-After` |
| Failure on our side | 500 |

Rejections (413/415/422/429) count in `distillate_requests_total` but not in
`distillate_errors_total`, so bad clients don't trip the error-rate alerts. Limits live in
[config.yaml](config.yaml).

## Build and run

deploy.sh copies the downloaded, hash-checked package to `Deployment/serving/package/`
(gitignored), then builds with this folder as the context:

```bash
docker build -f serve.Dockerfile --build-arg REGISTRY_VERSION=<n> -t distillate-api .
docker run --rm -p 8080:8080 distillate-api
curl -s localhost:8080/health
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d '{"text": "thanks for the edit"}'
```

On GCP, `Deployment/deploy/deploy.sh` does this with Cloud Build: it copies this folder and the
checked package to a staging folder and builds from a one-step config with
`-f serve.Dockerfile --build-arg REGISTRY_VERSION=<n>`, tagging
`distillate-images/distillate-api:v<n>-<commit>` (`gcloud builds submit --tag` alone can't pass
`-f` or build args). `.gcloudignore` makes sure `package/` is uploaded even though `.gitignore`
excludes it.

The image build fails if the package is incomplete, any file's SHA-256 differs from
`meta.json`, or the model can't score the reference texts.

Without Docker:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
DISTILLATE_PACKAGE_DIR=<package> .venv/bin/uvicorn app:create_app --factory --port 8080
```

Run-time overrides: `DISTILLATE_PACKAGE_DIR`, `DISTILLATE_RATE_LIMIT_PER_MINUTE` (0 turns the
limit off, for the load test), `DISTILLATE_ORT_THREADS`, `DISTILLATE_EXPOSE_METRICS`.

The rate limit keys on the last `X-Forwarded-For` entry (the one the nearest proxy appended;
entries a caller sends come first and can be forged). If the live service shows that entry is
a Google address rather than the caller, set `rate_limit.trusted_proxy_hops: 1`.

## Smoke-test reference scores

`deploy.sh` makes sure every package has `reference_outputs.json` before the build
(`Deployment/deploy/deploy_tools.py reference`). If the package has none (v1 and v2 don't;
`package.py` doesn't write it yet), deploy_tools computes it with this folder's
`scorer.reference_outputs` on 50 neutral inputs and writes it into the package, so the smoke
test compares the live service with the exact code it runs. The format:

```json
{"model_version": "...", "max_length": 256,
 "items": [{"text": "...", "scores": {"toxicity": 0.01, "...": 0.0}}]}
```

`python scorer.py --package <dir>` is the image build's package check (hashes, labels,
thresholds, and the model scoring the 5 `REFERENCE_TEXTS`, equal to `PARITY_TEXTS` in
`Model-Pipeline/src/student/package.py`). With `--out <file>` it also writes their scores.

## Tests

```bash
.venv/bin/python -m pytest tests -q && .venv/bin/ruff check .
```

The tests build a tiny ONNX model and tokenizer offline, so they need no network or real
package.
