"""Raw Wikipedia data for the teacher: light cleaning, TFDS conversion, checks, manifest.

Used by ``kaggle/00_prepare_data.ipynb``. The cleaning here must stay identical to the
data pipeline's preprocess step (CONTRACTS.md section 2), because the teacher labels
the same text the student is later trained on.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Unicode NFKC, then ``html.unescape``, then collapse whitespace runs to one space."""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()


def _as_str(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def records_to_frame(records: Iterable[Mapping[str, Any]]):
    """Turn TFDS examples (``tfds.as_numpy``) into the raw frame: ``id``, ``text``, six labels.

    Text is cleaned; labels must already be 0 or 1 and are cast to int8. Any other TFDS
    feature (for example ``language``) is dropped.
    """
    import numpy as np
    import pandas as pd

    ids, texts = [], []
    labels: dict[str, list[float]] = {name: [] for name in LABELS}
    for rec in records:
        ids.append(_as_str(rec["id"]))
        texts.append(clean_text(_as_str(rec["text"])))
        for name in LABELS:
            labels[name].append(float(rec[name]))

    df = pd.DataFrame({"id": pd.Series(ids, dtype="string"),
                       "text": pd.Series(texts, dtype="string")})
    for name in LABELS:
        values = np.asarray(labels[name], dtype="float64")
        bad = ~np.isin(values, (0.0, 1.0))
        if bad.any():
            raise ValueError(f"label {name!r} has {int(bad.sum())} values that are not 0 or 1")
        df[name] = values.astype("int8")
    return df


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


def build_data_manifest(files: Mapping[str, Mapping[str, Any]], tfds_name: str,
                        code_commit: str | None = None) -> str:
    """JSON manifest: per file its row count, empty-text count, and SHA-256.

    ``files`` maps each Parquet path to its ``check_frame`` result.
    """
    manifest = {
        "tfds_name": tfds_name,
        "code_commit": code_commit,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {
            Path(path).name: {**dict(stats), "sha256": file_sha256(path)}
            for path, stats in files.items()
        },
    }
    return json.dumps(manifest, indent=2, sort_keys=True)
