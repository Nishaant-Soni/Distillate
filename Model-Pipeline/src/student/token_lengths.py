"""How many training comments the student's tokenizer cuts at each candidate max length.

Decides ``model.max_length`` (128 unless the truncation share at 128 is large; Team Work
Plan §4). Writes counts and percentiles only, never text, to
``artifacts/student/token_lengths_<model>.json``.

    python -m src.student.token_lengths --set paths.raw_train=/kaggle/working/wiki_train_raw.parquet
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.student.config import load_config
from src.teacher.config import PIPELINE_ROOT, get_logger

logger = get_logger(__name__)

CANDIDATES = (64, 128, 256, 512)


def length_stats(lengths: Sequence[int], candidates: Sequence[int] = CANDIDATES) -> dict[str, Any]:
    """Percentiles and, per candidate length, the share of rows longer than it."""
    import numpy as np

    n = np.asarray(lengths)
    return {
        "rows": int(len(n)),
        "percentiles": {str(q): int(np.percentile(n, q)) for q in (50, 75, 90, 95, 99)},
        "max": int(n.max()) if len(n) else 0,
        "truncated_share": {str(c): round(float((n > c).mean()), 5) for c in candidates},
    }


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from src.student.data import read_rows
    from src.student.model import build

    rows = read_rows(cfg["paths"]["raw_train"])
    _, tokenizer, model_id = build(cfg)
    lengths = [len(ids) for ids in tokenizer(rows["text"].tolist())["input_ids"]]
    result = {"model_id": model_id, "source": Path(cfg["paths"]["raw_train"]).name,
              **length_stats(lengths)}
    out = PIPELINE_ROOT / "artifacts" / "student" / f"token_lengths_{cfg['model']['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("token lengths: %s", json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncation share per max length.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
