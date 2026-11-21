# Serving (`Deployment/serving/`)

The Distillate API (work plan v5 §2.4): FastAPI over ONNX Runtime, CPU only, with one model
package (§2.5) baked into the image. Cloud Run service `distillate-api`, us-east1.

| Endpoint | Returns |
|---|---|
| `POST /predict` `{"text": "..."}` | `scores`, `flagged`, `flag`, `low_confidence`, `model_version`, `latency_ms` |
| `GET /health` | `{"status": "ok", "model_version": "..."}` |
| `GET /metrics` | Prometheus text with the §2.4 metric names (for local compose; Cloud Run uses its built-in metrics) |

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

For Cloud Build (`gcloud builds submit`), `.gcloudignore` makes sure `package/` is uploaded even
though `.gitignore` excludes it. The Dockerfile isn't named `Dockerfile`, so the build needs
`-f serve.Dockerfile` (a `cloudbuild.yaml` step) rather than `--tag` alone.

The image build fails if the package is incomplete, any file's SHA-256 differs from
`meta.json`, or the model can't score the reference texts.

Without Docker:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
DISTILLATE_PACKAGE_DIR=<package> .venv/bin/uvicorn app:create_app --factory --port 8080
```

Run-time overrides: `DISTILLATE_PACKAGE_DIR`, `DISTILLATE_RATE_LIMIT_PER_MINUTE` (0 turns the
limit off, for the load test), `DISTILLATE_ORT_THREADS`.

## Smoke-test reference scores

`python scorer.py --package <dir> --out reference_outputs.json` writes the expected scores for
the neutral reference texts (equal to `PARITY_TEXTS` in `Model-Pipeline/src/student/package.py`;
a test checks), in the format the smoke test reads:

```json
{"model_version": "...", "max_length": 128,
 "items": [{"text": "...", "scores": {"toxicity": 0.01, "...": 0.0}}]}
```

That's the fallback for packages built before package.py writes `reference_outputs.json` (v1
and v2). `from scorer import reference_outputs` does the same in Python.

## Tests

```bash
.venv/bin/python -m pytest tests -q && .venv/bin/ruff check .
```

The tests build a tiny ONNX model and tokenizer offline, so they need no network or real
package.
