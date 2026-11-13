"""The evaluation tables from the competition files, where the DVC copy isn't available (Kaggle).

Builds the same tables the data pipeline versions (CONTRACTS.md §2), in its folder layout, so
``src.data.load_data(data_dir=...)`` and ``src.pipeline.run`` read them the same way:

    <out>/wikipedia/gate.parquet    Toxic Comment test rows with labels (gate_ids.txt), 63,978
    <out>/civil/h2_eval.parquet     Civil Comments test rows (h2_eval_ids.txt), 10,000
    <out>/civil/bias_eval.parquet   identity-annotated Civil test rows (bias_eval_ids.txt), 21,577
    <out>/obfuscation.parquet       the data pipeline's own scripts/obfuscate.py on that gate table

Rows are picked by the id lists in git (``Data-Pipeline/data/splits/``); every id must be found.
Text is cleaned as in training (``clean_text``). Civil labels follow the data pipeline:
``<label> = rater fraction >= 0.5`` (int8), the fraction kept as ``<label>_frac``.

``write_gold`` writes the label files the gate and metrics read (``--gold``): ids, labels and
identity columns only, no text.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger

logger = get_logger(__name__)

DATA_PIPELINE = PIPELINE_ROOT.parent / "Data-Pipeline"
CIVIL_POSITIVE = 0.5
# Gold columns kept besides id and the labels: the Civil identity attributes (bias slices).
IDENTITY_COLUMNS = [
    "female", "male", "transgender", "other_gender",
    "homosexual_gay_or_lesbian", "bisexual", "heterosexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]


def read_ids(path: str | Path) -> list[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def take(frame, wanted: list[str], name: str):
    """Rows of ``frame`` for ``wanted`` ids, in that order; every id must be present."""
    frame = frame.drop_duplicates("id").set_index("id")
    missing = [i for i in wanted if i not in frame.index]
    if missing:
        raise ValueError(f"{name}: {len(missing)} ids not in the source, e.g. {missing[:3]}")
    return frame.loc[wanted].reset_index()


def civil_frame(csv_path: str | Path):
    """Civil Comments rows: id, cleaned text, six int8 labels, ``<label>_frac``, identities."""
    import pandas as pd

    from src.teacher.data import clean_text

    raw = pd.read_csv(csv_path, dtype={"id": str, "comment_text": str}, keep_default_na=False,
                      low_memory=False)
    if "toxicity" not in raw.columns and "target" in raw.columns:
        raw = raw.rename(columns={"target": "toxicity"})
    missing = [c for c in ("id", "comment_text", *LABELS) if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    out = pd.DataFrame({"id": raw["id"].astype(str).astype("string"),
                        "text": raw["comment_text"].map(clean_text).astype("string")})
    for label in LABELS:
        frac = pd.to_numeric(raw[label], errors="coerce").astype("float32")
        out[label] = (frac >= CIVIL_POSITIVE).astype("int8")
        out[f"{label}_frac"] = frac
    for column in IDENTITY_COLUMNS:
        if column in raw.columns:
            out[column] = pd.to_numeric(raw[column], errors="coerce").astype("float32")
    return out


def obfuscation_table(gate, data_pipeline: Path = DATA_PIPELINE):
    """The data pipeline's obfuscation set, by running its own script on ``gate``."""
    import pandas as pd

    sys.path.insert(0, str(data_pipeline))
    try:
        from scripts import obfuscate
        from scripts.utils import load_config as load_pipeline_config
    finally:
        sys.path.remove(str(data_pipeline))
    cfg = load_pipeline_config(data_pipeline / "config" / "pipeline.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp) / "processed"
        (processed / "wikipedia").mkdir(parents=True)
        gate.to_parquet(processed / "wikipedia" / "gate.parquet", index=False)
        cfg["paths"]["processed_dir"] = str(processed)
        cfg["paths"]["summaries_dir"] = str(Path(tmp) / "summaries")
        summary = obfuscate.run(cfg)
        table = pd.read_parquet(processed / "obfuscation.parquet")
    logger.info("obfuscation: %d rows %s", len(table), summary.get("techniques"))
    return table, summary


def build_eval_tables(wiki_dir: str | Path, civil_csv: str | Path, out: str | Path,
                      splits_dir: str | Path = DATA_PIPELINE / "data" / "splits",
                      data_pipeline: Path = DATA_PIPELINE) -> dict[str, Any]:
    """Write the four evaluation tables under ``out`` (the data pipeline's layout)."""
    import pandas as pd

    from src.teacher.data import read_competition_split, to_raw_frame

    out, splits_dir = Path(out), Path(splits_dir)
    wiki = to_raw_frame(read_competition_split(wiki_dir, "test"))
    gate = take(wiki, read_ids(splits_dir / "gate_ids.txt"), "gate")
    gate["char_len"] = gate["text"].str.len().astype("int32")
    civil = civil_frame(civil_csv)
    h2 = take(civil.drop(columns=[c for c in IDENTITY_COLUMNS if c in civil.columns]),
              read_ids(splits_dir / "h2_eval_ids.txt"), "h2_eval")
    bias = take(civil, read_ids(splits_dir / "bias_eval_ids.txt"), "bias_eval")
    absent = [c for c in IDENTITY_COLUMNS if c not in bias.columns]
    if absent:
        raise ValueError(f"bias_eval: identity columns missing from {civil_csv}: {absent}")
    obfuscation, obf_summary = obfuscation_table(gate, data_pipeline)
    tables = {"wikipedia/gate.parquet": gate, "civil/h2_eval.parquet": h2,
              "civil/bias_eval.parquet": bias, "obfuscation.parquet": obfuscation}
    for rel, frame in tables.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out / rel, index=False)
    counts = {rel: len(frame) for rel, frame in tables.items()}
    logger.info("evaluation tables: %s", counts)
    return {"rows": counts, "obfuscation": {k: obf_summary.get(k) for k in (
        "rows_out", "techniques", "dropped")}, "positives": {
        rel: {label: int(pd.Series(frame[label]).sum()) for label in LABELS}
        for rel, frame in tables.items()}}


def write_gold(tables: Mapping[str, Any], gold_dir: str | Path) -> dict[str, int]:
    """``<gold_dir>/<stem>.parquet``: id, the six labels, identity columns if any. No text."""
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for stem, frame in tables.items():
        keep = ["id", *LABELS, *[c for c in IDENTITY_COLUMNS if c in frame.columns]]
        gold = frame[keep].copy()
        gold["id"] = gold["id"].astype(str)
        gold.to_parquet(gold_dir / f"{stem}.parquet", index=False)
        counts[stem] = len(gold)
    return counts
