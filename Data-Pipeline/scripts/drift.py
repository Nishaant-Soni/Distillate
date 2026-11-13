"""Drift metrics for the Civil drift stream: PSI against a baseline from training data.

The model was trained on Wikipedia train, so that is the baseline: a batch whose inputs look
unlike it is one the model may handle worse. Four features, each compared with PSI:

    input drift (input_drift, before scoring)
        char_len         comment length in characters
        oov_rate         share of a comment's words never seen (twice) in training
        nonalpha_share   share of non-space characters that aren't letters
    output drift (flag_rate_drift, after scoring)
        flag_rate        share of comments the model flags

PSI (population stability index) splits the baseline into equal-count bins (deciles) and
compares the share of the batch in each bin with the baseline's:
    PSI = sum over bins of (batch_share - baseline_share) * ln(batch_share / baseline_share)
0 means no change. Status uses the usual rules of thumb, from config (drift.psi_warn,
drift.psi_alert): ok below 0.1, warn from 0.1, alert from 0.25.

No comment text is stored. The baseline keeps bin edges, bin shares and means, and the
training vocabulary only as truncated BLAKE2b hashes of the words. It lives in
data/processed/monitoring/, which DVC versions with the data.

Used by drift_ingest_dag:
    baseline = drift.load_baseline(cfg)                        # built from train if missing
    drift.input_drift(batch, baseline)                         # in compute_drift
    drift.flag_rate_drift(predictions, reference_rate, cfg)    # in step_score

Build (or rebuild) the baseline on its own:
    python -m scripts.drift --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.utils import (
    PIPELINE_ROOT,
    atomic_write,
    get_logger,
    load_config,
    timed,
    write_summary,
)

logger = get_logger(__name__)
TASK_ID = "drift_baseline"
INPUT_FEATURES = ("char_len", "oov_rate", "nonalpha_share")
WORD = r"[^\W\d_]+"          # a run of letters, in any alphabet
LETTER = r"[^\W\d_]"
BASELINE_VERSION = 1


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


def _hash_word(word: str) -> str:
    """Truncated BLAKE2b, so the stored vocabulary holds no words."""
    return hashlib.blake2b(word.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------- features


def words(texts: pd.Series) -> pd.Series:
    """Each comment's lowercased words (runs of letters); digits and punctuation split words."""
    return texts.fillna("").str.lower().str.findall(WORD)


def char_len(frame: pd.DataFrame) -> pd.Series:
    if "char_len" in frame.columns:
        return frame["char_len"].astype(float)
    return frame["text"].fillna("").str.len().astype(float)


def nonalpha_share(texts: pd.Series) -> pd.Series:
    """Share of non-space characters that aren't letters (the preprocess obfuscation measure)."""
    texts = texts.fillna("")
    nonspace = texts.str.len() - texts.str.count(r"\s")
    letters = texts.str.count(LETTER)
    share = (nonspace - letters) / nonspace.where(nonspace > 0)
    return share.fillna(0.0).astype(float)


def oov_rate(tokens: pd.Series, known: set[str], hashed: bool) -> pd.Series:
    """Share of each comment's words not in the vocabulary; 0 for a comment with no words."""
    exploded = tokens.explode().dropna()
    if exploded.empty:
        return pd.Series(0.0, index=tokens.index)
    unique = pd.Index(exploded.unique())
    lookup = unique.map(_hash_word) if hashed else unique
    in_vocab = pd.Series([w in known for w in lookup], index=unique)
    unknown = (~exploded.map(in_vocab)).groupby(level=0).mean()
    return unknown.reindex(tokens.index, fill_value=0.0).astype(float)


# ---------------------------------------------------------------- PSI


def _bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Share of values in each bin: (-inf, e1], (e1, e2], ..., (ek, inf).

    Bins are closed on the right, so a value tied at an edge (say, many comments with an OOV
    rate of exactly 0) stays apart from values just above it.
    """
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(expected: np.ndarray, actual: np.ndarray, epsilon: float) -> float:
    """Population stability index; empty bins are floored at epsilon so PSI stays finite."""
    e = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    a = np.clip(np.asarray(actual, dtype=float), epsilon, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float | None, warn: float, alert: float) -> str:
    if value is None:
        return "unknown"
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def _worst(statuses: list[str]) -> str:
    order = ["ok", "unknown", "warn", "alert"]
    return max(statuses, key=order.index) if statuses else "unknown"


# ---------------------------------------------------------------- baseline


def _feature_baseline(values: pd.Series, n_bins: int) -> dict[str, Any]:
    """Equal-count bin edges (duplicates merged, so heavy ties still work) and their shares."""
    arr = values.to_numpy(dtype=float)
    quantiles = np.quantile(arr, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = sorted({float(q) for q in quantiles})
    return {
        "edges": edges,
        "shares": _bin_shares(arr, edges).tolist(),
        "mean": float(arr.mean()),
    }


def build_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    """The baseline from training data: per-feature bins and a hashed vocabulary."""
    dcfg = cfg["drift"]
    source = _resolve(dcfg["baseline_source"])
    if not source.exists():
        raise FileNotFoundError(
            f"Baseline source {source.as_posix()} not found; run data_prep_dag (split) first"
        )
    train = pd.read_parquet(source, columns=["text", "char_len"])
    tokens = words(train["text"])
    counts = tokens.explode().dropna().value_counts()
    known = set(counts.index[counts >= dcfg["vocab_min_count"]])

    features = {
        "char_len": char_len(train),
        "oov_rate": oov_rate(tokens, known, hashed=False),
        "nonalpha_share": nonalpha_share(train["text"]),
    }
    return {
        "version": BASELINE_VERSION,
        "source": dcfg["baseline_source"],
        "rows": len(train),
        "vocab_min_count": dcfg["vocab_min_count"],
        "vocab_size": len(known),
        "vocab_hashes": sorted(_hash_word(w) for w in known),
        "features": {name: _feature_baseline(values, dcfg["psi_bins"])
                     for name, values in features.items()},
    }


def save_baseline(baseline: dict[str, Any], cfg: dict[str, Any]) -> Path:
    path = _resolve(cfg["drift"]["baseline_file"])
    atomic_write(path, json.dumps(baseline))
    return path


def load_baseline(cfg: dict[str, Any], build_if_missing: bool = True) -> dict[str, Any]:
    """The saved baseline; built from training data first if it doesn't exist yet."""
    path = _resolve(cfg["drift"]["baseline_file"])
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"No drift baseline at {path.as_posix()}; build it with `python -m scripts.drift`"
            )
        logger.info("No drift baseline yet; building it from %s", cfg["drift"]["baseline_source"])
        save_baseline(build_baseline(cfg), cfg)
    baseline = json.loads(path.read_text(encoding="utf-8"))
    baseline["_vocab"] = set(baseline["vocab_hashes"])
    return baseline


# ---------------------------------------------------------------- drift


def _thresholds(source: dict[str, Any]) -> tuple[float, float, float]:
    return source["psi_warn"], source["psi_alert"], source["psi_epsilon"]


def input_drift(batch: pd.DataFrame, baseline: dict[str, Any],
                cfg: dict[str, Any]) -> dict[str, Any]:
    """PSI of the batch's length, OOV rate and non-alpha share against the baseline."""
    warn, alert, eps = _thresholds(cfg["drift"])
    vocab = baseline.get("_vocab") or set(baseline["vocab_hashes"])
    values = {
        "char_len": char_len(batch),
        "oov_rate": oov_rate(words(batch["text"]), vocab, hashed=True),
        "nonalpha_share": nonalpha_share(batch["text"]),
    }
    features = {}
    for name in INPUT_FEATURES:
        ref = baseline["features"][name]
        arr = values[name].to_numpy(dtype=float)
        value = psi(ref["shares"], _bin_shares(arr, ref["edges"]), eps)
        features[name] = {
            "psi": round(value, 4),
            "status": status(value, warn, alert),
            "batch_mean": round(float(arr.mean()), 4) if len(arr) else None,
            "baseline_mean": round(ref["mean"], 4),
        }
    return {
        "rows": len(batch),
        "features": features,
        "max_psi": max(f["psi"] for f in features.values()),
        "status": _worst([f["status"] for f in features.values()]),
        "thresholds": {"warn": warn, "alert": alert},
    }


def flag_rate_drift(predictions: pd.DataFrame, reference_rate: float | None,
                    cfg: dict[str, Any]) -> dict[str, Any]:
    """PSI of the share of flagged comments against a reference rate (two bins).

    The reference is the flag rate on in-distribution data: in Phase 3, the production
    model on the Wikipedia validation set. Without one, PSI is None and status "unknown".
    """
    warn, alert, eps = _thresholds(cfg["drift"])
    rate = float(predictions["flag"].mean()) if len(predictions) else None
    value = None
    if rate is not None and reference_rate is not None:
        value = psi([1 - reference_rate, reference_rate], [1 - rate, rate], eps)
    return {
        "rows": len(predictions),
        "flag_rate": round(rate, 4) if rate is not None else None,
        "reference_rate": reference_rate,
        "psi": round(value, 4) if value is not None else None,
        "status": status(value, warn, alert),
        "thresholds": {"warn": warn, "alert": alert},
    }


# ---------------------------------------------------------------- stage


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build and save the baseline. Returns the run summary."""
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}}
    with timed(summary):
        baseline = build_baseline(cfg)
        path = save_baseline(baseline, cfg)
        summary.update({
            "rows_in": baseline["rows"],
            "rows_out": 1,
            "vocab_size": baseline["vocab_size"],
            "baseline_file": cfg["drift"]["baseline_file"],
            "feature_means": {k: round(v["mean"], 4) for k, v in baseline["features"].items()},
        })
    logger.info("Drift baseline: %d rows, %d words, written to %s",
                baseline["rows"], baseline["vocab_size"], path.as_posix())
    write_summary(TASK_ID, summary, cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
