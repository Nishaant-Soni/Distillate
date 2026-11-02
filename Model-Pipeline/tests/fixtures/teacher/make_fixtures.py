"""Write the synthetic teacher-shard fixtures in this folder (see README.md).

Uses the production shard code (``shard_frame``, ``build_manifest``, ``write_shard``), so
the fixtures have exactly the format of real shards. Deterministic: rerunning rewrites the
same files, and ``tests/teacher/test_fixtures.py`` checks that they are up to date.

    python tests/fixtures/teacher/make_fixtures.py     # from Model-Pipeline/
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))   # Model-Pipeline/, when run as a script

ROWS, SHARD_SIZE, SEED = 25, 10, 42
PROMPT_VERSION = "p2-zs-v2b"
MODEL_VERSION = "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
CREATED_AT = "2026-10-20T00:00:00+00:00"   # fixed, so the files don't change on rerun
BROKEN_SHARD, BROKEN_ROW_COUNT = "shard_0001", 11


def fixture_ids() -> list[str]:
    """16-hex-digit ids like the Jigsaw ones, clearly synthetic (they start with ``f1x``)."""
    return [f"f1x{i:013x}" for i in range(ROWS)]


def scores(ids):
    """Teacher-shard rows: ``p`` in (0, 1) and the matching ``logit``, all ``ok``."""
    import numpy as np
    import pandas as pd

    from src.teacher.config import LABELS

    rng = np.random.default_rng(SEED)
    pred = pd.DataFrame({"id": ids, "status": "ok"})
    for label in LABELS:
        logit = rng.normal(-3.0, 3.0, len(ids))
        pred[f"p_{label}"] = 1 / (1 + np.exp(-logit))
        pred[f"logit_{label}"] = logit
    return pred


def write(root: Path = HERE) -> None:
    import json

    import pandas as pd

    from src.teacher.shards import build_manifest, plan_shards, shard_frame, shard_name
    from src.teacher.storage import LocalStorage, shard_paths, write_shard

    ids = fixture_ids()
    storage, pred = LocalStorage(root), scores(ids).set_index("id")
    for k, shard_ids in enumerate(plan_shards(ids, SHARD_SIZE)):
        frame = shard_frame(pred.loc[shard_ids].reset_index(), PROMPT_VERSION, MODEL_VERSION)
        manifest = build_manifest(frame, PROMPT_VERSION, MODEL_VERSION)
        manifest["created_at"] = CREATED_AT
        for run in ("valid", "broken"):
            write_shard(storage, f"{run}/", shard_name(k), frame, manifest)
    # The broken run: one manifest whose row count disagrees with its Parquet.
    path = root / shard_paths("broken/", BROKEN_SHARD)[1]
    manifest = json.loads(path.read_text())
    manifest["row_count"] = BROKEN_ROW_COUNT
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    pd.DataFrame({"id": ids}).to_csv(root / "source_ids.csv", index=False)


if __name__ == "__main__":
    write()
    print("wrote fixtures under", HERE)
