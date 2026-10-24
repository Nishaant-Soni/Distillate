"""Contract-shaped placeholder output, used until the real stage modules land.

Why this exists: the DAG has to run end to end from day 3 (work plan, Oct 14),
long before acquire/preprocess/dedupe/split/bias are written. Every stub writes
files with the exact columns and dtypes from CONTRACTS.md section 2, at tiny row
counts, so downstream tasks and tests have something real to read.

Two rules keep this honest:
  * A stub never writes into a file a real stage has already produced.
  * Every summary records ``"mode": "stub"``, so ``reports/run_summary.json``
    always says which stages are real and which are still placeholders.

Teammates can also import ``wikipedia_frame`` / ``civil_frame`` to get
contract-shaped fixtures for their own unit tests.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from scripts.utils import atomic_path, get_logger, timed

logger = get_logger(__name__)

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
IDENTITY_COLUMNS = ["female", "male", "muslim", "christian", "jewish", "black", "white",
                    "homosexual_gay_or_lesbian", "psychiatric_or_mental_illness"]
_SAMPLE_TEXT = [
    "thanks for the helpful edit, much appreciated",
    "this article needs a citation in the second paragraph",
    "you are completely wrong about this and always have been",
    "please stop reverting my changes without discussion",
    "i disagree with the tone of this section",
]


def _rng(seed: int, salt: str) -> random.Random:
    """Deterministic per-file generator: same seed and name give the same rows."""
    return random.Random(f"{seed}:{salt}")


def wikipedia_frame(n: int, seed: int, prefix: str, with_weight: bool = False):
    """A frame matching the Wikipedia contract (CONTRACTS.md 2.2)."""
    import pandas as pd

    rng = _rng(seed, prefix)
    text = [_SAMPLE_TEXT[rng.randrange(len(_SAMPLE_TEXT))] for _ in range(n)]
    df = pd.DataFrame({
        "id": [f"{prefix}_{i:06d}" for i in range(n)],
        "text": text,
    })
    for label in LABELS:
        # Rare labels stay rare, so stratified-split tests see a realistic shape.
        rate = 0.10 if label == "toxicity" else 0.02
        df[label] = pd.Series([int(rng.random() < rate) for _ in range(n)], dtype="int8")
    df["char_len"] = pd.Series([len(t) for t in text], dtype="int32")
    df["has_identity_term"] = pd.Series([rng.random() < 0.15 for _ in range(n)], dtype="bool")
    df["has_obfuscation"] = pd.Series([rng.random() < 0.05 for _ in range(n)], dtype="bool")
    if with_weight:
        df["sample_weight"] = pd.Series([1.0] * n, dtype="float32")
    return df


def civil_frame(n: int, seed: int, prefix: str, with_identities: bool = False):
    """A frame matching the Civil Comments contract (CONTRACTS.md 2.2)."""
    import pandas as pd

    rng = _rng(seed, prefix)
    text = [_SAMPLE_TEXT[rng.randrange(len(_SAMPLE_TEXT))] for _ in range(n)]
    df = pd.DataFrame({
        "id": [f"{prefix}_{i:06d}" for i in range(n)],
        "text": text,
        # Sorted, so the drift batches are already in created_date order.
        "created_date": pd.date_range("2017-01-01", periods=n, freq="h", tz="UTC"),
        "publication_id": [f"pub_{rng.randrange(5)}" for _ in range(n)],
        "article_id": [f"art_{rng.randrange(50)}" for _ in range(n)],
    })
    for label in LABELS:
        frac = [round(rng.random() if rng.random() < 0.25 else rng.random() * 0.3, 3)
                for _ in range(n)]
        df[f"{label}_frac"] = pd.Series(frac, dtype="float32")
        df[label] = pd.Series([int(f >= 0.5) for f in frac], dtype="int8")
    if with_identities:
        for col in IDENTITY_COLUMNS:
            df[col] = pd.Series([round(rng.random(), 3) for _ in range(n)], dtype="float32")
    return df


def _write(df, target: Path, summary: dict[str, Any]) -> None:
    """Write one parquet file, unless a real stage already wrote it."""
    if target.exists():
        summary.setdefault("skipped_existing", []).append(str(target.name))
        return
    with atomic_path(target) as tmp:
        df.to_parquet(tmp, index=False)
    summary.setdefault("files", {})[str(target.name)] = len(df)


def run(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    """Produce placeholder output for ``stage`` and return the summary dict.

    Signature mirrors the real stages (CONTRACTS.md section 3) apart from the
    ``stage`` argument, which only exists because one function covers them all.
    """
    import pandas as pd

    seed = cfg["seed"]
    n = cfg.get("stubs", {}).get("rows", 200)
    processed = Path(cfg["paths"]["processed_dir"])
    raw = Path(cfg["paths"]["raw_dir"])
    summary: dict[str, Any] = {"task": stage, "mode": "stub", "rows_out": 0, "dropped": {}}

    with timed(summary):
        wiki_raw, civil_raw = raw / "wikipedia", raw / "civil"
        wiki_out, civil_out = processed / "wikipedia", processed / "civil"

        if stage == "acquire_wikipedia":
            for split in ("train", "test"):
                _write(wikipedia_frame(n, seed, f"wiki_{split}"),
                       wiki_raw / f"{split}.parquet", summary)
        elif stage == "acquire_civil":
            for split in ("train", "validation", "test"):
                _write(civil_frame(n, seed, f"civil_{split}"),
                       civil_raw / f"{split}.parquet", summary)
            _write(civil_frame(n, seed, "civil_identities", with_identities=True),
                   civil_raw / "identities_test.parquet", summary)
        elif stage == "preprocess_wikipedia":
            for split in ("train", "test"):
                _write(wikipedia_frame(n, seed, f"wiki_{split}"),
                       wiki_out / f"{split}_clean.parquet", summary)
        elif stage == "preprocess_civil":
            for split in ("train", "validation", "test"):
                _write(civil_frame(n, seed, f"civil_{split}"),
                       civil_out / f"{split}_clean.parquet", summary)
            # subsets.py builds bias_eval from the cleaned Identities test split.
            _write(civil_frame(n, seed, "civil_identities", with_identities=True),
                   civil_out / "identities_test_clean.parquet", summary)
        elif stage == "dedupe_wikipedia":
            _write(wikipedia_frame(n, seed, "wiki_train"),
                   wiki_out / "train_dedup.parquet", summary)
            summary["dropped"] = {"exact_duplicate": 0, "near_duplicate": 0, "gate_overlap": 0}
        elif stage == "split_wikipedia":
            val_n = max(1, int(n * cfg["splits"]["val_fraction"]))
            canary_n = max(1, min(val_n // 2, cfg["splits"]["canary_wikipedia_size"]))
            parts = {
                "train": wikipedia_frame(n - val_n, seed, "wiki_train", with_weight=True),
                "val": wikipedia_frame(val_n, seed, "wiki_val"),
                "gate": wikipedia_frame(n, seed, "wiki_test"),
                "canary_wikipedia": wikipedia_frame(canary_n, seed, "wiki_canary"),
            }
            splits_dir = Path(cfg["paths"]["splits_dir"])
            for name, df in parts.items():
                _write(df, wiki_out / f"{name}.parquet", summary)
                ids = splits_dir / f"{name}_ids.txt"
                if not ids.exists():
                    with atomic_path(ids) as tmp:
                        tmp.write_text("\n".join(df["id"]), encoding="utf-8")
        elif stage == "civil_subsets":
            _write(civil_frame(n, seed, "civil_h2"), civil_out / "h2_eval.parquet", summary)
            _write(civil_frame(n, seed, "civil_canary"),
                   civil_out / "canary_civil.parquet", summary)
            _write(civil_frame(n, seed, "civil_bias", with_identities=True),
                   civil_out / "bias_eval.parquet", summary)
            batches = cfg["splits"]["civil"]["drift_batches"]
            per_batch = max(1, n // batches)
            for b in range(batches):
                _write(civil_frame(per_batch, seed, f"civil_drift_{b:02d}"),
                       civil_out / "drift" / f"batch_{b:02d}.parquet", summary)
        elif stage == "obfuscate":
            df = wikipedia_frame(max(1, n // 4), seed, "wiki_test")
            df["id"] = df["id"] + "_obf"
            _write(df, processed / "obfuscation.parquet", summary)
        elif stage == "schema_stats":
            # A stub cannot produce a real TFDV schema; Sai's --infer mode does
            # that once and commits the result. Record the gap instead of faking it.
            summary["note"] = "no schema written by the stub; run scripts.schema_stats --infer"
        elif stage == "anomalies":
            summary["checks"] = {"run": 0, "warn": 0, "fail": 0}
            summary["note"] = "stub ran no checks"
        elif stage == "bias":
            report = Path(cfg["paths"]["reports_dir"]) / "bias" / "bias_report.md"
            if not report.exists():
                with atomic_path(report) as tmp:
                    tmp.write_text("# Bias report (placeholder)\n\nWritten by scripts/bias.py.\n",
                                   encoding="utf-8")
            train = wiki_out / "train.parquet"
            if train.exists():
                df = pd.read_parquet(train)
                if "sample_weight" not in df.columns:
                    df["sample_weight"] = pd.Series([1.0] * len(df), dtype="float32")
                    with atomic_path(train) as tmp:
                        df.to_parquet(tmp, index=False)
                summary["sample_weight_rows"] = len(df)
        else:
            raise ValueError(f"No stub defined for stage {stage!r}")

    summary["rows_out"] = sum(summary.get("files", {}).values())
    logger.warning("STUB %s wrote %s file(s) of placeholder data",
                   stage, len(summary.get("files", {})))
    # No write_summary here on purpose. One stage can have several stub parts
    # (acquire -> wikipedia + civil), and only the stage is a task. The caller
    # writes one summary per task with these nested under "parts".
    return summary
