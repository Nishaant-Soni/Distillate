"""Raw Wikipedia data for the teacher: loading, light cleaning, checks, manifest.

The source is the Jigsaw Toxic Comment Classification Challenge files on Kaggle, the data
TFDS ``wikipedia_toxicity_subtypes/EnglishSubtypes:0.3.1`` repackaged (TFDS can no longer
download it: its source bucket returns 403). Same rows and ids: train is ``train.csv``
(159,571 rows); test is ``test.csv`` joined with ``test_labels.csv``, keeping the 63,978
rows whose labels are not -1.

Used by ``kaggle/00_prepare_data.ipynb``. The cleaning here must stay identical to the
data pipeline's preprocess step, because the teacher labels the same text the student
is later trained on.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS

_WHITESPACE = re.compile(r"\s+")

# Competition column -> our label name, in LABELS order.
KAGGLE_LABELS = {
    "toxic": "toxicity",
    "severe_toxic": "severe_toxicity",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_hate": "identity_attack",
}
assert tuple(KAGGLE_LABELS.values()) == LABELS


def clean_text(text: str) -> str:
    """Unicode NFKC, then ``html.unescape``, then collapse whitespace runs to one space."""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()


def find_competition_file(competition_dir: str | Path, name: str) -> Path:
    """``<name>`` or ``<name>.zip`` in the competition folder (Kaggle ships both forms)."""
    for candidate in (Path(competition_dir) / name, Path(competition_dir) / f"{name}.zip"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name}[.zip] not found in {competition_dir}")


def find_competition_dir(root: str | Path) -> Path:
    """The folder under ``root`` (e.g. ``/kaggle/input``) that holds ``test_labels.csv``."""
    matches = sorted({p.parent for p in Path(root).rglob("test_labels.csv*")})
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one competition folder under {root}, found {matches}")
    return matches[0]


def _read_csv(path: Path):
    import pandas as pd

    # Text is read as-is: without keep_default_na=False, comments such as "NA" or "null"
    # would become missing values.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_competition_split(competition_dir: str | Path, split: str):
    """The competition rows for ``split`` with columns ``id``, ``comment_text``, label columns."""
    if split == "train":
        return _read_csv(find_competition_file(competition_dir, "train.csv"))
    if split != "test":
        raise ValueError(f"split must be train or test, got {split!r}")

    text = _read_csv(find_competition_file(competition_dir, "test.csv"))
    labels = _read_csv(find_competition_file(competition_dir, "test_labels.csv"))
    df = text.merge(labels, on="id", how="inner", validate="one_to_one")
    if len(df) != len(text) or len(df) != len(labels):
        raise ValueError(f"test.csv ({len(text)}) and test_labels.csv ({len(labels)}) "
                         f"share only {len(df)} ids")
    # -1 marks rows that were never scored; a row is either fully -1 or fully labeled.
    unscored = df[list(KAGGLE_LABELS)].eq("-1")
    all_unscored, any_unscored = unscored.all(axis=1), unscored.any(axis=1)
    if (any_unscored != all_unscored).any():
        raise ValueError("some test rows have -1 for only part of their labels")
    return df[~all_unscored].reset_index(drop=True)


def to_raw_frame(df):
    """Competition rows -> the raw frame: ``id``, cleaned ``text``, six int8 labels."""
    import pandas as pd

    missing = {"id", "comment_text", *KAGGLE_LABELS} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = pd.DataFrame({
        "id": df["id"].astype("string"),
        "text": df["comment_text"].map(clean_text).astype("string"),
    })
    for column, label in KAGGLE_LABELS.items():
        values = df[column]
        bad = ~values.isin(("0", "1"))
        if bad.any():
            raise ValueError(f"label {column!r} has {int(bad.sum())} values that are not 0 or 1")
        out[label] = values.astype("int8")
    return out


def check_frame(df, expected_rows: int) -> dict[str, int]:
    """Step 1 acceptance checks. Raises on a wrong row count or duplicate ids.

    Rows whose text is empty after cleaning are kept and only counted: the teacher marks
    them ``failed`` and the data pipeline drops them anyway.
    """
    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(df)}")
    duplicates = int(df["id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{duplicates} duplicate ids")
    return {"rows": len(df), "empty_text_rows": int((df["text"].str.len() == 0).sum())}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_manifest(files: Mapping[str | Path, Mapping[str, Any]],
                        source: Mapping[str, Any], code_commit: str | None = None) -> str:
    """JSON manifest: where the data came from, and per output file its stats and SHA-256.

    ``files`` maps each Parquet path to its ``check_frame`` result.
    """
    manifest = {
        "source": dict(source),
        "code_commit": code_commit,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {
            Path(path).name: {**dict(stats), "sha256": file_sha256(path)}
            for path, stats in files.items()
        },
    }
    return json.dumps(manifest, indent=2, sort_keys=True)
