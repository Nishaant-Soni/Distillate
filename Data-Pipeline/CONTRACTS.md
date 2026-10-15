# Distillate: Shared Contracts

Agreed by the team on Oct 13, 2026. Change only by team agreement, and update this file in the same PR as the change.
Everyone codes and tests against these contracts, not against each other's code.


## 1. Repository layout
```
distillate/
├── Data-Pipeline/        # Phase 1 (dags/, scripts/, config/, data/, reports/, tests/)
├── Model-Pipeline/       # Phase 2 (src/teacher, src/student, src/eval, kaggle/, tests/)
├── Deployment/           # Phase 3 (serving/, monitoring/, frontend/, deploy/, dags/)
├── .github/workflows/
└── README.md
```

## 2. Data files

**`data/processed/wikipedia/{train,val,gate,canary_wikipedia}.parquet`**

| Column | Type | Notes |
|---|---|---|
| `id` | string | TFDS id, unique |
| `text` | string | cleaned (NFKC, HTML entities, whitespace) |
| `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack` | int8 | 0 or 1 |
| `char_len` | int32 | |
| `has_identity_term` | bool | from `config/identity_terms.txt` |
| `has_obfuscation` | bool | non-alphabetic share above threshold |
| `sample_weight` | float32 | train only; 1.0 until bias step writes it |

**`data/processed/civil/{h2_eval,canary_civil,bias_eval}.parquet` and `drift/batch_00..19.parquet`**

| Column | Type | Notes |
|---|---|---|
| `id`, `text` | string | |
| `created_date` | timestamp (UTC) | |
| `publication_id`, `article_id` | string | |
| six label columns | int8 | positive at fraction ≥ 0.5 |
| `<label>_frac` for each label | float32 | raw rater fraction |
| identity columns (`female`, `muslim`, `black`, …) | float32 | bias_eval only |

**`data/processed/obfuscation.parquet`:** same columns as Wikipedia, `id` = `<original_id>_obf`.

**Teacher label shards** (`gs://<bucket>/teacher/<prompt_version>/shard_XXXX.parquet`): `id`, `p_toxicity`, `p_severe_toxicity`, `p_obscene`, `p_threat`, `p_insult`, `p_identity_attack` (float32 in [0,1]), `status` (`ok` / `failed`), `prompt_version`, `model_version`. Each shard has a sibling `shard_XXXX.manifest.json` with `row_count`, `ids_sha256`, `prompt_version`, `model_version`.

**Prediction files** (any model, any eval set): `id`, `p_<label>` for all six labels, `model_version`. This is the only thing the evaluation and bias code reads, so it works the same for the TF-IDF baseline, every student, the teacher, and Llama Guard.

## 3. Code conventions
- Every stage is a function `run(cfg: dict) -> dict` (the dict is a JSON summary: rows in, rows out, dropped with reasons, duration) plus a CLI: `python -m scripts.<name> --config config/pipeline.yaml`.
- All tunables live in `config/pipeline.yaml`. Seed is `42` everywhere.
- Logging only through `utils.get_logger(__name__)`. Writes only through `utils.atomic_write(...)`.
- Alerts only through `alerts.send_alert(level, title, details)` where level is `"warn"` or `"fail"`. Until the real one lands, `alerts.py` ships as a no-op that logs.

## 4. Serving API (Phase 3, fixed now so demo, monitoring, and drift work can start early)
- `POST /predict` body `{"text": "..."}` → `{"scores": {label: float}, "flagged": {label: bool}, "low_confidence": bool, "model_version": "...", "latency_ms": float}`
- `GET /health` → `{"status": "ok", "model_version": "..."}`
- `GET /metrics` → Prometheus format with: `distillate_request_latency_seconds` (histogram), `distillate_requests_total`, `distillate_errors_total`, `distillate_flagged_total{label}`, `distillate_score{label}` (histogram), `distillate_low_confidence_total`.
- Text is never logged; logs hold `text_sha256`, `char_len`, and scores only.

## 5. Model artifacts
- MLflow registered model name: `distillate-student`, stages `Staging` and `Production`.
- A model package is a folder with `model.onnx`, `tokenizer/`, `thresholds.json` (per-label), `meta.json` (version, training data hash, alpha, temperature).
