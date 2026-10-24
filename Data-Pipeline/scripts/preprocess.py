"""Stage ``preprocess``: turn raw text and labels into clean, typed files.

Inputs:  data/raw/wikipedia/{train,test}.parquet
         data/raw/civil/{validation,test,identities_test}.parquet
         data/cache/civil/train.parquet
Outputs: data/processed/wikipedia/{train,test}_clean.parquet
         data/processed/civil/{validation,test,identities_test}_clean.parquet
         data/cache/civil/train_clean.parquet
The full 1.8M-row Civil train split stays in the untracked cache (data/cache, not in DVC);
it can be rebuilt from the pinned Kaggle source, and only the sampled subsets are versioned.

What it does, in order:
    1. Labels. Wikipedia: must be 0 or 1, become int8. Civil: the rater fraction is
       kept as <label>_frac (float32) and <label> = fraction >= threshold (int8).
       Labels listed in labels.civil_dropped_labels (sexual_explicit) are dropped.
    2. Civil dates: created_date -> UTC timestamp. format="mixed" because a few
       dates have no fractional seconds.
    3. Text: light cleaning only (HTML entities, Unicode NFKC, whitespace). We do NOT
       lowercase, strip punctuation or fix spellings: CAPS, "!!!" and tricks like
       "1d10t" are signals we want to keep.
    4. Empty comments are dropped and counted. Words like "None" or "NA" are real comments
       and stay. Team rule (validation.max_missing_fraction): if more than that share of a
       file is dropped, the stage fails; otherwise it warns and drops.
    5. Derived columns: char_len, has_identity_term, has_obfuscation.
    6. Identity columns are kept only in the identities file (CONTRACTS.md 2.2).

Run inside the pipeline:  python -m scripts.stage --stage preprocess
Run on its own:           python -m scripts.preprocess --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow.parquet as pq

from scripts.alerts import send_alert
from scripts.columns import (
    CIVIL_CLEAN_COLUMNS,
    CIVIL_IDENTITY_EXTRA_COLUMNS,
    IDENTITY_COLUMNS,
    LABELS,
    WIKI_CLEAN_COLUMNS,
)
from scripts.errors import DataQualityError, LabelError
from scripts.parquet_io import write_chunks, write_frame
from scripts.utils import get_logger, load_config, timed

logger = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
WIKI_SPLITS = ["train", "test"]
CIVIL_SPLITS = ["train", "validation", "test", "identities_test"]


# ---------------------------------------------------------------- text


def clean_text(value: Any) -> str:
    """Light text cleaning, the same as the teacher lane (Model-Pipeline/src/teacher/data.py).

    Order: Unicode NFKC, then HTML entities, then whitespace collapsed, so the teacher
    and the student see identically cleaned text. Keep this function pure: same input,
    same output.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()


def load_identity_pattern(path: Path) -> re.Pattern[str] | None:
    """Compile the identity-term lexicon into one case-insensitive, whole-word regex.

    Blank lines and lines starting with "#" are ignored. Longer terms come first,
    so a phrase wins over a single word inside it. A missing or empty lexicon sends
    a warning and returns None (has_identity_term is then False everywhere).
    """
    if not path.exists():
        send_alert("warn", "Identity lexicon not found", {"path": str(path)})
        return None
    terms = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not terms:
        send_alert("warn", "Identity lexicon is empty", {"path": str(path)})
        return None
    terms.sort(key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE)


def nonalpha_share(text: str) -> float:
    """Share of non-space characters that are not letters ("1d10t" -> 0.6)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(not c.isalpha() for c in chars) / len(chars)


def clean_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the text column and drop comments that end up empty. Returns drop counts."""
    out = df.copy()
    raw = out["text"].fillna("").astype(str)
    empty_in_source = raw.str.strip() == ""
    out["text"] = raw.map(clean_text)
    empty_after_clean = (out["text"] == "") & ~empty_in_source
    dropped = {
        "empty_in_source": int(empty_in_source.sum()),
        "empty_after_clean": int(empty_after_clean.sum()),
    }
    return out[out["text"] != ""].reset_index(drop=True), dropped


def add_text_features(
    df: pd.DataFrame, pattern: re.Pattern[str] | None, obf_threshold: float
) -> pd.DataFrame:
    """Add char_len (int32), has_identity_term (bool) and has_obfuscation (bool)."""
    out = df.copy()
    out["char_len"] = out["text"].str.len().astype("int32")
    if pattern is None:
        out["has_identity_term"] = False
    else:
        out["has_identity_term"] = out["text"].map(lambda t: pattern.search(t) is not None)
    out["has_obfuscation"] = out["text"].map(nonalpha_share) > obf_threshold
    out["has_identity_term"] = out["has_identity_term"].astype(bool)
    out["has_obfuscation"] = out["has_obfuscation"].astype(bool)
    return out


# ---------------------------------------------------------------- labels and fields


def to_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia: every label must be exactly 0 or 1, else the pipeline stops."""
    out = df.copy()
    for label in LABELS:
        values = pd.to_numeric(out[label], errors="coerce")
        bad = values.isna() | ~values.isin([0, 1])
        if bad.any():
            examples = out.loc[bad, "id"].head(3).tolist()
            raise LabelError(
                f"{label}: {int(bad.sum())} rows are missing or not 0/1, e.g. ids {examples}"
            )
        out[label] = values.astype("int8")
    return out


def to_civil_labels(
    df: pd.DataFrame, threshold: float, dropped_labels: list[str]
) -> pd.DataFrame:
    """Civil: keep the rater fraction as <label>_frac, set <label> = fraction >= threshold."""
    out = df.copy()
    for label in LABELS:
        fraction = pd.to_numeric(out[label], errors="coerce")
        bad = fraction.isna() | (fraction < 0) | (fraction > 1)
        if bad.any():
            examples = out.loc[bad, "id"].head(3).tolist()
            raise LabelError(
                f"{label}: {int(bad.sum())} rows are missing or outside 0-1, e.g. ids {examples}"
            )
        out[f"{label}_frac"] = fraction.astype("float32")
        out[label] = (fraction >= threshold).astype("int8")
    return out.drop(columns=dropped_labels, errors="ignore")


def parse_created_date(values: pd.Series) -> pd.Series:
    """Parse dates to UTC. format="mixed" handles dates with and without fractional seconds."""
    return pd.to_datetime(values, format="mixed", utc=True, errors="coerce")


def to_identity_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Identity columns and annotator count -> float32. Blank means "not annotated" -> NaN."""
    out = df.copy()
    for column in [*IDENTITY_COLUMNS, "identity_annotator_count"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    return out


# ---------------------------------------------------------------- per-dataset processing


def process_wiki_frame(
    df: pd.DataFrame, pattern: re.Pattern[str] | None, obf_threshold: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one Wikipedia table into the contract's columns."""
    df = to_binary_labels(df)
    df, dropped = clean_texts(df)
    df = add_text_features(df, pattern, obf_threshold)
    return df[WIKI_CLEAN_COLUMNS], dropped


def process_civil_frame(
    df: pd.DataFrame,
    pattern: re.Pattern[str] | None,
    obf_threshold: float,
    labels_cfg: dict[str, Any],
    with_identities: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one Civil Comments table (or chunk) into the contract's columns."""
    df = to_civil_labels(
        df, labels_cfg["civil_positive_threshold"], labels_cfg["civil_dropped_labels"]
    )
    df["created_date"] = parse_created_date(df["created_date"])
    if with_identities:
        df = to_identity_numbers(df)
    df, dropped = clean_texts(df)
    df = add_text_features(df, pattern, obf_threshold)
    dropped["missing_created_date_kept"] = int(df["created_date"].isna().sum())
    columns = CIVIL_CLEAN_COLUMNS + (CIVIL_IDENTITY_EXTRA_COLUMNS if with_identities else [])
    return df[columns], dropped


def _merge_counts(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def check_drop_rate(
    name: str, rows_in: int, rows_out: int, limit: float, dropped: dict[str, int]
) -> str:
    """Team rule for dropped rows: fail above ``limit``, otherwise warn and drop.

    ``limit`` is validation.max_missing_fraction. Returns "ok" when nothing was dropped
    and "warn" when some rows were dropped within the limit; raises DataQualityError above it.
    """
    removed = rows_in - rows_out
    rate = removed / rows_in if rows_in else 0.0
    if rate > limit:
        raise DataQualityError(
            f"{name}: {removed:,} of {rows_in:,} rows dropped ({rate:.3%}), above the "
            f"{limit:.3%} limit (validation.max_missing_fraction). Reasons: {dropped}"
        )
    if removed:
        send_alert("warn", f"preprocess dropped {removed} rows of {name}", dict(dropped))
        return "warn"
    return "ok"


def civil_paths(cfg: dict[str, Any], split: str) -> tuple[Path, Path]:
    """Raw input and clean output for one Civil split.

    The full train split lives in the untracked cache (too large for DVC); the other
    splits live in data/raw and data/processed.
    """
    if split == "train":
        cache = Path(cfg["paths"]["cache_dir"]) / "civil"
        return cache / "train.parquet", cache / "train_clean.parquet"
    return (
        Path(cfg["paths"]["raw_dir"]) / "civil" / f"{split}.parquet",
        Path(cfg["paths"]["processed_dir"]) / "civil" / f"{split}_clean.parquet",
    )


def _civil_chunks(
    source: Path,
    name: str,
    counts: dict[str, Any],
    batch_size: int,
    pattern: re.Pattern[str] | None,
    obf_threshold: float,
    labels_cfg: dict[str, Any],
    with_identities: bool,
    max_missing: float,
) -> Iterator[pd.DataFrame]:
    """Yield cleaned chunks of one raw Civil file, updating counts as it goes.

    The drop-rate rule is checked after the last chunk, while the output is still being
    written to a temporary file, so a failing file is never saved.
    """
    for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size):
        raw = batch.to_pandas()
        clean, dropped = process_civil_frame(
            raw, pattern, obf_threshold, labels_cfg, with_identities
        )
        counts["rows_in"] += len(raw)
        counts["rows_out"] += len(clean)
        _merge_counts(counts["dropped"], dropped)
        yield clean
    drops = {k: v for k, v in counts["dropped"].items() if k != "missing_created_date_kept"}
    check_drop_rate(name, counts["rows_in"], counts["rows_out"], max_missing, drops)


def _send_date_alerts(files: dict[str, dict[str, Any]]) -> None:
    """Warn when a file has unparseable dates (those rows are kept, not dropped)."""
    for name, info in files.items():
        missing_dates = info["dropped"].get("missing_created_date_kept", 0)
        if missing_dates:
            send_alert(
                "warn",
                f"{missing_dates} unparseable created_date values in {name}",
                info["dropped"],
            )


# ---------------------------------------------------------------- stage entry point


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Clean every raw file and write the processed files. Returns the stage summary."""
    summary: dict[str, Any] = {"task": "preprocess", "files": {}, "dropped": {}}
    with timed(summary):
        raw_dir = Path(cfg["paths"]["raw_dir"])
        processed_dir = Path(cfg["paths"]["processed_dir"])
        settings = cfg["preprocess"]
        labels_cfg = cfg["labels"]
        obf = settings["obfuscation_nonalpha_share"]
        max_missing = cfg["validation"]["max_missing_fraction"]
        pattern = load_identity_pattern(Path(cfg["paths"]["identity_terms"]))

        for split in WIKI_SPLITS:
            name = f"wikipedia/{split}_clean.parquet"
            raw = pd.read_parquet(raw_dir / "wikipedia" / f"{split}.parquet")
            clean, dropped = process_wiki_frame(raw, pattern, obf)
            check_drop_rate(name, len(raw), len(clean), max_missing, dropped)
            write_frame(clean, processed_dir / "wikipedia" / f"{split}_clean.parquet")
            summary["files"][name] = {
                "rows_in": len(raw),
                "rows_out": len(clean),
                "dropped": dropped,
            }

        for split in CIVIL_SPLITS:
            name = f"civil/{split}_clean.parquet"
            source, target = civil_paths(cfg, split)
            counts: dict[str, Any] = {"rows_in": 0, "rows_out": 0, "dropped": {}}
            chunks = _civil_chunks(
                source,
                name,
                counts,
                settings["batch_size"],
                pattern,
                obf,
                labels_cfg,
                with_identities=(split == "identities_test"),
                max_missing=max_missing,
            )
            write_chunks(chunks, target)
            summary["files"][name] = counts

        for info in summary["files"].values():
            _merge_counts(summary["dropped"], info["dropped"])
        summary["rows_in"] = sum(info["rows_in"] for info in summary["files"].values())
        summary["rows_out"] = sum(info["rows_out"] for info in summary["files"].values())
        _send_date_alerts(summary["files"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw data into processed Parquet.")
    parser.add_argument("--config", default=None, help="path to config/pipeline.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=str))


if __name__ == "__main__":
    main()
