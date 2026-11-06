"""Stage ``dedupe``: remove duplicate comments from Wikipedia train.

Why: if the same comment is in training and in the gate (test) set, the model is
graded on something it memorized, and every score looks better than it really is.

Inputs:  data/processed/wikipedia/train_clean.parquet, test_clean.parquet (the gate set)
Outputs: data/processed/wikipedia/train_dedup.parquet
         reports/dedupe/removed_ids.csv  (id, reason, matched_id, jaccard; never text)

Rules:
    - Only TRAIN rows are ever removed. The gate set is never touched.
    - Exact duplicates: same text after lowercasing and removing punctuation.
    - Near duplicates: MinHash + LSH on word n-grams (dedupe.shingle_words), and every
      candidate pair is double-checked with the real Jaccard (>= dedupe.jaccard_threshold)
      before anything is removed.
    - Rows are processed in id order, so "keep the first" keeps the same rows every run.
    - Comments with no words at all (only punctuation or emoji) are never treated as
      duplicates of each other.

Run inside the pipeline:  python -m scripts.stage --stage dedupe
Run on its own:           python -m scripts.dedupe --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasketch import MinHash, MinHashLSH

from scripts.columns import LABELS
from scripts.parquet_io import write_frame
from scripts.utils import atomic_path, get_logger, load_config, timed

logger = get_logger(__name__)

_PUNCTUATION = re.compile(r"[^\w\s]")
REPORT_COLUMNS = ["id", "reason", "matched_id", "jaccard"]


# ---------------------------------------------------------------- text keys


def norm_key(text: str) -> str:
    """Lowercase, punctuation removed, spaces collapsed. Used only for matching."""
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def shingles(text: str, size: int) -> set[str]:
    """Overlapping word n-grams. Comments shorter than size use their whole text."""
    words = norm_key(text).split()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Shared pieces divided by all distinct pieces."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def make_minhash(pieces: set[str], num_perm: int, seed: int) -> MinHash:
    signature = MinHash(num_perm=num_perm, seed=seed)
    signature.update_batch([piece.encode("utf-8") for piece in pieces])
    return signature


def make_index(threshold: float, num_perm: int, bands: int | None) -> MinHashLSH:
    """LSH index. bands=None lets datasketch choose bands from the threshold."""
    if bands is None:
        return MinHashLSH(threshold=threshold, num_perm=num_perm)
    if num_perm % bands:
        raise ValueError(f"minhash_num_perm ({num_perm}) must be divisible by lsh_bands ({bands})")
    return MinHashLSH(num_perm=num_perm, params=(bands, num_perm // bands))


# ---------------------------------------------------------------- exact duplicates


def exact_dedupe(
    train: pd.DataFrame, gate: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Remove train rows whose normalized text equals a gate row or an earlier train row.

    Returns (kept train rows, removal records, label conflicts among train duplicates).
    """
    train = train.sort_values("id", kind="mergesort").reset_index(drop=True)
    keys = train["text"].map(norm_key)

    gate_keys = gate["text"].map(norm_key)
    has_words = gate_keys != ""
    gate_lookup = dict(zip(gate_keys[has_words], gate.loc[has_words, "id"], strict=True))

    records: list[dict[str, Any]] = []
    in_gate = (keys != "") & keys.isin(gate_lookup)
    for idx in train.index[in_gate]:
        records.append(
            {
                "id": train.at[idx, "id"],
                "reason": "exact_vs_gate",
                "matched_id": gate_lookup[keys[idx]],
                "jaccard": 1.0,
            }
        )

    rest = train[~in_gate]
    rest_keys = keys[~in_gate]
    repeated = rest_keys.duplicated(keep="first") & (rest_keys != "")
    first_id = rest.groupby(rest_keys)["id"].transform("first")
    first_labels = rest.groupby(rest_keys)[LABELS].transform("first")
    conflicts = int((repeated & (rest[LABELS] != first_labels).any(axis=1)).sum())
    for idx in rest.index[repeated]:
        records.append(
            {
                "id": rest.at[idx, "id"],
                "reason": "exact_within_train",
                "matched_id": first_id[idx],
                "jaccard": 1.0,
            }
        )

    return rest[~repeated].reset_index(drop=True), records, conflicts


# ---------------------------------------------------------------- near duplicates


def _best_match(
    candidates: list[str], pieces: set[str], texts: dict[str, str], size: int, threshold: float
) -> tuple[str, float] | None:
    """Return (key, jaccard) of the best verified candidate, preferring gate matches."""
    best = None
    for key in candidates:
        score = jaccard(pieces, shingles(texts[key], size))
        if score < threshold:
            continue
        rank = (0 if key.startswith("gate:") else 1, -score, key)
        if best is None or rank < best[0]:
            best = (rank, key, score)
    return None if best is None else (best[1], best[2])


def near_dedupe(
    train: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    threshold: float,
    num_perm: int,
    size: int,
    seed: int,
    bands: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Remove train rows that nearly copy a gate row or an earlier train row.

    Returns (kept train rows, removal records).
    """
    index = make_index(threshold, num_perm, bands)
    texts: dict[str, str] = {}
    for gate_id, text in zip(gate["id"], gate["text"], strict=True):
        if norm_key(text) == "":
            continue
        key = f"gate:{gate_id}"
        index.insert(key, make_minhash(shingles(text, size), num_perm, seed))
        texts[key] = text

    train = train.sort_values("id", kind="mergesort").reset_index(drop=True)
    removed_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for row_id, text in zip(train["id"], train["text"], strict=True):
        if norm_key(text) == "":
            continue
        pieces = shingles(text, size)
        signature = make_minhash(pieces, num_perm, seed)
        match = _best_match(index.query(signature), pieces, texts, size, threshold)
        if match is None:
            key = f"train:{row_id}"
            index.insert(key, signature)
            texts[key] = text
            continue
        key, score = match
        source, matched_id = key.split(":", 1)
        reason = "near_vs_gate" if source == "gate" else "near_within_train"
        records.append(
            {"id": row_id, "reason": reason, "matched_id": matched_id, "jaccard": round(score, 4)}
        )
        removed_ids.add(row_id)

    kept = train[~train["id"].isin(removed_ids)].reset_index(drop=True)
    return kept, records


# ---------------------------------------------------------------- stage entry point


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate Wikipedia train against itself and the gate set. Returns the summary."""
    summary: dict[str, Any] = {"task": "dedupe"}
    with timed(summary):
        settings = cfg["dedupe"]
        wiki_dir = Path(cfg["paths"]["processed_dir"]) / "wikipedia"
        train = pd.read_parquet(wiki_dir / "train_clean.parquet")
        gate = pd.read_parquet(wiki_dir / "test_clean.parquet")
        gate_rows = len(gate)

        after_exact, exact_records, conflicts = exact_dedupe(train, gate)
        after_near, near_records = near_dedupe(
            after_exact,
            gate,
            threshold=settings["jaccard_threshold"],
            num_perm=settings["minhash_num_perm"],
            size=settings["shingle_words"],
            seed=cfg["seed"],
            bands=settings.get("lsh_bands"),
        )
        if len(gate) != gate_rows:
            raise RuntimeError("gate rows must never be removed")

        write_frame(after_near, wiki_dir / "train_dedup.parquet")
        report = pd.DataFrame(exact_records + near_records, columns=REPORT_COLUMNS)
        report_path = Path(cfg["paths"]["reports_dir"]) / "dedupe" / "removed_ids.csv"
        with atomic_path(report_path) as tmp:
            report.to_csv(tmp, index=False)  # ids only, never comment text

        by_reason = {str(k): int(v) for k, v in report["reason"].value_counts().items()}
        summary.update(
            {
                "rows_in": len(train),
                "rows_out": len(after_near),
                "gate_rows": gate_rows,
                # Same keys as the stub summary, so run_summary.json reads the same way.
                "dropped": {
                    "exact_duplicate": by_reason.get("exact_within_train", 0),
                    "near_duplicate": by_reason.get("near_within_train", 0),
                    "gate_overlap": by_reason.get("exact_vs_gate", 0)
                    + by_reason.get("near_vs_gate", 0),
                },
                "removed_by_reason": by_reason,
                "label_conflicts": conflicts,
            }
        )
        logger.info(
            "dedupe: %s in, %s out, removed %s, label conflicts %s",
            len(train),
            len(after_near),
            by_reason,
            conflicts,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicate comments from Wikipedia train.")
    parser.add_argument("--config", default=None, help="path to config/pipeline.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=str))


if __name__ == "__main__":
    main()
