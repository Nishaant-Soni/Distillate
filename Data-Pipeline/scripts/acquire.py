"""Stage ``acquire``: download the Kaggle source files and write raw Parquet.

The TFDS copies of our datasets now return HTTP 403, so we use the original
Kaggle competition files they were built from (same ids, same row counts):
    Wikipedia Toxicity -> jigsaw-toxic-comment-classification-challenge
    Civil Comments     -> jigsaw-unintended-bias-in-toxicity-classification

Raw layout:
    data/raw/wikipedia/train.parquet, test.parquet
    data/raw/civil/validation.parquet, test.parquet, identities_test.parquet
    data/raw/manifest.json
    data/cache/civil/train.parquet   full 1.8M-row Civil train: untracked cache, not in DVC;
                                     rebuilt from the pinned Kaggle source when missing

What this stage does and doesn't do:
    - Renames Kaggle columns to ours, joins Wikipedia test text with its labels, drops
      unscored test rows (-1), and builds the Identities test split.
    - Keeps ids and text exactly as Kaggle ships them. CSVs are read with
      keep_default_na=False, so comments that are literally "NA" or "null" survive.
    - Turns label, identity and annotator-count columns into float32, like the
      TFDS copies were. All text cleaning happens in preprocess.py.

Run inside the pipeline:  python -m scripts.stage --stage acquire
Run on its own:           python -m scripts.acquire --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from scripts.columns import (
    CIVIL_EXTRA_LABELS,
    CIVIL_RAW_COLUMNS,
    IDENTITY_COLUMNS,
    LABELS,
    WIKI_RAW_COLUMNS,
    WIKI_RENAME,
)
from scripts.errors import (
    DataUnavailableError,
    FileCorruptionError,
    LabelError,
    ManifestMismatchError,
)
from scripts.parquet_io import sha256_file, write_chunks, write_frame
from scripts.utils import atomic_write, get_logger, load_config, timed

logger = get_logger(__name__)

# Name of the toxicity column in each Civil Comments file ("target" in train only).
CIVIL_TOXICITY_COLUMN = {"train": "target", "validation": "toxicity", "test": "toxicity"}

# Raw files too large to version: kept in the untracked cache instead of data/raw.
CACHED_FILES = {"civil/train.parquet"}


def raw_location(cfg: dict[str, Any], name: str) -> Path:
    """Where a raw file lives: the untracked cache for CACHED_FILES, data/raw otherwise."""
    base = cfg["paths"]["cache_dir"] if name in CACHED_FILES else cfg["paths"]["raw_dir"]
    return Path(base) / name


# ---------------------------------------------------------------- reading CSVs


def read_csv_as_text(path: Path, **kwargs: Any):
    """Read a CSV keeping every value as text.

    keep_default_na=False matters: some real comments are literally "NA" or
    "null", and pandas would otherwise turn them into missing values.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def to_float32(df: pd.DataFrame, columns: list[str], *, required: bool, source: str):
    """Convert text columns to float32.

    required=True: an empty or non-numeric value stops the pipeline (labels).
    required=False: empty means "not annotated" and becomes NaN (identity columns).
    """
    out = df.copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce")
        if required:
            bad = values.isna()
            if bad.any():
                examples = out.loc[bad, "id"].head(3).tolist()
                raise LabelError(
                    f"{source}: column {column!r} has {int(bad.sum())} empty or "
                    f"non-numeric values, e.g. ids {examples}"
                )
        out[column] = values.astype("float32")
    return out


# ---------------------------------------------------------------- download


def csv_name_of(file_name: str) -> str:
    """'train.csv.zip' -> 'train.csv'; 'train.csv' -> 'train.csv'."""
    return file_name.removesuffix(".zip")


def kaggle_download(competition: str, file_name: str, folder: Path) -> None:
    """Download one competition file into folder (Kaggle may deliver it zipped)."""
    try:
        # Imported here, not at the top: importing kaggle needs credentials,
        # and the tests must be able to import this module without them.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_file(competition, file_name, path=str(folder), quiet=True)
    except Exception as err:  # the Kaggle client raises many different error types
        raise DataUnavailableError(
            f"Could not download {file_name!r} from Kaggle competition {competition!r}: "
            f"{err}. Check that you accepted the competition rules on kaggle.com, and "
            "that credentials are set: KAGGLE_USERNAME and KAGGLE_KEY (both are in the "
            "kaggle.json that Settings -> API -> Create New Token downloads), or "
            "~/.kaggle/kaggle.json, or an OAuth token in KAGGLE_API_TOKEN."
        ) from err


def _extracted_marker(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.name + ".extracted")


def extract_zip(zip_path: Path, folder: Path) -> None:
    """Check one zip and extract it. A broken zip is deleted, so a rerun downloads it again."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            broken_member = archive.testzip()
            if broken_member is None:
                archive.extractall(folder)
                return
            problem = f"member {broken_member} is corrupted"
    except zipfile.BadZipFile:
        problem = "it is not a valid zip file"
    zip_path.unlink(missing_ok=True)
    raise FileCorruptionError(
        f"{zip_path.name}: {problem}. Deleted it; rerun the stage to download it again."
    )


def extract_zips(folder: Path) -> None:
    """Extract every zip in folder once, including zips found inside zips."""
    while True:
        pending = [z for z in sorted(folder.glob("*.zip")) if not _extracted_marker(z).exists()]
        if not pending:
            return
        for zip_path in pending:
            extract_zip(zip_path, folder)
            _extracted_marker(zip_path).touch()


def fetch_competition(competition: str, files: dict[str, str], folder: Path) -> dict[str, Path]:
    """Make sure each needed file exists as a plain CSV in folder.

    files maps a role ("train", "test", ...) to the Kaggle file name. Only missing
    files are downloaded. Returns role -> CSV path.
    """
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, file_name in files.items():
        csv_path = folder / csv_name_of(file_name)
        if not csv_path.exists():
            if not any(folder.glob(f"{csv_path.name}*.zip")):
                logger.info("Downloading %s from %s", file_name, competition)
                kaggle_download(competition, file_name, folder)
            extract_zips(folder)
        if not csv_path.exists():
            raise DataUnavailableError(
                f"{csv_path.name} not found in {folder} after downloading {file_name}. "
                "Check the file names under datasets in config/pipeline.yaml."
            )
        paths[role] = csv_path
    return paths


# ---------------------------------------------------------------- build raw tables


def build_wiki_train(csv_path: Path) -> pd.DataFrame:
    """Wikipedia train with our column names (expected 159,571 rows)."""
    df = read_csv_as_text(csv_path).rename(columns=WIKI_RENAME)[WIKI_RAW_COLUMNS]
    return to_float32(df, LABELS, required=True, source="wikipedia/train")


def build_wiki_test(text_csv: Path, labels_csv: Path) -> pd.DataFrame:
    """Wikipedia test: join text with labels, drop unscored rows (-1). Expected 63,978 rows."""
    texts = read_csv_as_text(text_csv)
    labels = read_csv_as_text(labels_csv)
    df = texts.merge(labels, on="id", how="inner", validate="one_to_one")
    df = df[df["toxic"] != "-1"]  # values are still text here, so compare to "-1"
    df = df.rename(columns=WIKI_RENAME)[WIKI_RAW_COLUMNS].reset_index(drop=True)
    return to_float32(df, LABELS, required=True, source="wikipedia/test")


def civil_frame(chunk: pd.DataFrame, toxicity_column: str, source: str) -> pd.DataFrame:
    """Rename one chunk of a Civil Comments CSV to our names, keep our columns, set types."""
    renamed = chunk.rename(columns={"comment_text": "text", toxicity_column: "toxicity"})
    df = renamed[CIVIL_RAW_COLUMNS]
    df = to_float32(df, [*LABELS, *CIVIL_EXTRA_LABELS], required=True, source=source)
    return to_float32(
        df, ["identity_annotator_count", *IDENTITY_COLUMNS], required=False, source=source
    )


def civil_source_columns(toxicity_column: str) -> set[str]:
    """Kaggle column names we need to read from one Civil Comments CSV."""
    to_source = {"text": "comment_text", "toxicity": toxicity_column}
    return {to_source.get(column, column) for column in CIVIL_RAW_COLUMNS}


def write_civil_split(csv_path: Path, out_path: Path, split: str, chunksize: int) -> int:
    """Read one Civil Comments CSV in chunks and write it as raw Parquet."""
    toxicity_column = CIVIL_TOXICITY_COLUMN[split]
    wanted = civil_source_columns(toxicity_column)
    chunks = read_csv_as_text(csv_path, usecols=lambda col: col in wanted, chunksize=chunksize)
    frames = (civil_frame(chunk, toxicity_column, f"civil/{split}") for chunk in chunks)
    return write_chunks(frames, out_path)


def build_identities_test(civil_test_path: Path) -> pd.DataFrame:
    """Civil test rows that have identity labels (expected 21,577).

    This is how TFDS built its CivilCommentsIdentities test split.
    """
    df = pd.read_parquet(civil_test_path)
    return df[df["identity_annotator_count"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------- checks and manifest


def expected_rows(cfg: dict[str, Any]) -> dict[str, int]:
    """Expected row count per raw file, from the datasets section of the config."""
    datasets = cfg["datasets"]
    expected = {
        f"wikipedia/{split}.parquet": count
        for split, count in datasets["wikipedia"]["expected_rows"].items()
    }
    expected.update(
        {
            f"civil/{split}.parquet": count
            for split, count in datasets["civil_comments"]["expected_rows"].items()
        }
    )
    expected["civil/identities_test.parquet"] = datasets["civil_identities"]["expected_rows"][
        "test"
    ]
    return expected


def check_rows(name: str, rows: int, expected: int) -> None:
    """Stop the pipeline if a file doesn't have the expected number of rows."""
    if rows != expected:
        raise ManifestMismatchError(f"{name}: expected {expected:,} rows, got {rows:,}")


def verify_existing(cfg: dict[str, Any], manifest: dict[str, Any]) -> bool:
    """True if every file in the manifest exists and still matches it.

    Returns False if a file is missing, so the stage rebuilds the raw data (for example
    on a fresh clone, where the cached Civil train file doesn't exist yet).
    Raises if a file exists but has changed or can't be read.
    """
    for name, info in manifest["files"].items():
        path = raw_location(cfg, name)
        if not path.exists():
            logger.info("%s missing; rebuilding raw data", name)
            return False
        try:
            rows = pq.read_metadata(path).num_rows
        except Exception as err:
            raise FileCorruptionError(f"{path} can't be read; delete it and rerun") from err
        if rows != info["rows"] or sha256_file(path) != info["sha256"]:
            raise ManifestMismatchError(
                f"{path} differs from manifest.json. If that is intended, delete it "
                "and rerun to rebuild it."
            )
    return True


def build_raw_files(cfg: dict[str, Any], kaggle_dir: Path) -> dict[str, Any]:
    """Download whatever is missing and write all six raw files. Returns the manifest."""
    datasets = cfg["datasets"]
    wiki, civil = datasets["wikipedia"], datasets["civil_comments"]
    chunksize = cfg["acquire"]["csv_chunksize"]

    wiki_csv = fetch_competition(
        wiki["kaggle_competition"], wiki["kaggle_files"], kaggle_dir / "wikipedia"
    )
    civil_csv = fetch_competition(
        civil["kaggle_competition"], civil["kaggle_files"], kaggle_dir / "civil"
    )

    rows: dict[str, int] = {}
    rows["wikipedia/train.parquet"] = write_frame(
        build_wiki_train(wiki_csv["train"]), raw_location(cfg, "wikipedia/train.parquet")
    )
    rows["wikipedia/test.parquet"] = write_frame(
        build_wiki_test(wiki_csv["test"], wiki_csv["test_labels"]),
        raw_location(cfg, "wikipedia/test.parquet"),
    )
    for split in ("train", "validation", "test"):
        name = f"civil/{split}.parquet"
        logger.info("Writing %s from %s", raw_location(cfg, name), civil_csv[split].name)
        rows[name] = write_civil_split(civil_csv[split], raw_location(cfg, name), split, chunksize)
    rows["civil/identities_test.parquet"] = write_frame(
        build_identities_test(raw_location(cfg, "civil/test.parquet")),
        raw_location(cfg, "civil/identities_test.parquet"),
    )

    expected = expected_rows(cfg)
    for name, count in rows.items():
        logger.info("%s: %s rows", name, f"{count:,}")
        check_rows(name, count, expected[name])

    sources = {
        f"{dataset}/{role}": {
            "competition": spec["kaggle_competition"],
            "file": spec["kaggle_files"][role],
            "sha256": sha256_file(path),
        }
        for dataset, spec, csv_paths in (("wikipedia", wiki, wiki_csv), ("civil", civil, civil_csv))
        for role, path in csv_paths.items()
    }
    return {
        "sources": sources,
        "files": {
            name: {
                "rows": count,
                "sha256": sha256_file(raw_location(cfg, name)),
                "location": "cache" if name in CACHED_FILES else "raw",
            }
            for name, count in rows.items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- stage entry point


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Download, reshape and save all raw files. Returns the stage summary."""
    summary: dict[str, Any] = {"task": "acquire", "dropped": {}}
    with timed(summary):
        kaggle_dir = Path(cfg["paths"]["kaggle_dir"])
        manifest_path = Path(cfg["paths"]["raw_dir"]) / "manifest.json"

        manifest = None
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if verify_existing(cfg, existing):
                logger.info("Raw data present and matches manifest; skipping download")
                manifest = existing
                summary["skipped"] = True
        if manifest is None:
            manifest = build_raw_files(cfg, kaggle_dir)
            atomic_write(manifest_path, json.dumps(manifest, indent=2))
            summary["skipped"] = False

        summary["files"] = {name: info["rows"] for name, info in manifest["files"].items()}
        summary["rows_out"] = sum(summary["files"].values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle data and write raw Parquet.")
    parser.add_argument("--config", default=None, help="path to config/pipeline.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=str))


if __name__ == "__main__":
    main()
