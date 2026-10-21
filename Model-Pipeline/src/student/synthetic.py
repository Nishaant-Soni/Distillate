"""Synthetic, contract-shaped inputs for the student: raw train rows and a soft-label store.

No real comment text: each row is neutral filler words plus marker words (``marktoxicity``, ...)
that make its labels learnable. The soft-label shards follow the spec exactly (§5 columns,
manifests with ``ids_sha256`` and ``stack_sha256`` of the committed stack file), so the
student's validation and join run unchanged on them.

    python -m src.student.synthetic --out outputs/synthetic       # then configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger

logger = get_logger(__name__)

FILLER = ("the", "page", "edit", "article", "source", "talk", "thanks", "please", "see",
          "section", "image", "user", "revert", "link", "fact", "wiki", "change", "note")
MARKERS = {label: "mark" + label.replace("_", "") for label in LABELS}   # one BERT token each
VOCAB = (*FILLER, *MARKERS.values())
# Rough Wikipedia prevalences, raised so every label has positives in a few hundred rows.
PREVALENCE = {"toxicity": 0.25, "severe_toxicity": 0.06, "obscene": 0.15, "threat": 0.05,
              "insult": 0.15, "identity_attack": 0.06}
SOFT_DEFAULTS = {
    "prefix": "soft_labels/stack-v1/p2-zs-v2b/",
    "prompt_version": "p2-zs-v2b",
    "model_version": "stack-v1+Qwen/Qwen3-4B-Instruct-2507@"
                     "cdbee75f17c01a7cc42f958dc650907174af0554+tfidf-lr-v1",
    "shard_size": 5000,
}
STACK_FILE = PIPELINE_ROOT / "artifacts" / "stack" / "stack_p2-zs-v2b.json"


def make_rows(n: int = 600, seed: int = 42):
    """``id`` (16 hex chars, like Kaggle's), ``text``, six int8 labels."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    ids = sorted({f"{x:016x}" for x in rng.integers(0, 2**63, size=n * 2)})[:n]
    labels = {label: (rng.random(n) < PREVALENCE[label]).astype("int8") for label in LABELS}
    labels["toxicity"] |= labels["severe_toxicity"]   # severe implies toxic, as in the data
    texts = []
    for i in range(n):
        words = list(rng.choice(FILLER, size=int(rng.integers(4, 20))))
        for label in LABELS:
            if labels[label][i] and rng.random() < 0.9:   # 10% of positives carry no marker
                words.insert(int(rng.integers(0, len(words) + 1)), MARKERS[label])
        texts.append(" ".join(words))
    df = pd.DataFrame({"id": pd.Series(ids, dtype="string"),
                       "text": pd.Series(texts, dtype="string")})
    for label in LABELS:
        df[label] = labels[label]
    return df


def make_soft_labels(rows, seed: int = 42, failed: int = 0):
    """A teacher that sees the gold label through noise, in the §5 shard columns."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 1)
    out = pd.DataFrame({"id": rows["id"].astype("string")})
    for label in LABELS:
        z = np.where(rows[label] == 1, 3.0, -4.0) + rng.normal(0, 1.5, len(rows))
        out[f"p_{label}"] = (1 / (1 + np.exp(-z))).astype("float32")
    for label in LABELS:
        out[f"logit_{label}"] = np.log(out[f"p_{label}"] / (1 - out[f"p_{label}"])).astype(
            "float32")
    out["status"] = pd.Series(["ok"] * len(out), dtype="string")
    if failed:
        idx = rng.choice(len(out), size=failed, replace=False)
        out.loc[idx, "status"] = "failed"
        score_columns = [c for c in out.columns if c.startswith(("p_", "logit_"))]
        out.loc[idx, score_columns] = np.float32("nan")
    out["prompt_version"] = pd.Series([SOFT_DEFAULTS["prompt_version"]] * len(out),
                                      dtype="string")
    out["model_version"] = pd.Series([SOFT_DEFAULTS["model_version"]] * len(out),
                                     dtype="string")
    return out


def write_store(soft, root: str | Path, soft_cfg: Mapping[str, Any] | None = None,
                stack_file: str | Path = STACK_FILE, stack_sha256: str | None = None) -> str:
    """Write ``soft`` as shards under ``<root>/<prefix>`` per the shard plan; return the prefix."""
    from src.student.data import file_sha256
    from src.teacher.shards import ids_sha256, plan_shards, shard_name
    from src.teacher.storage import LocalStorage, write_shard

    sl = {**SOFT_DEFAULTS, **(soft_cfg or {})}
    storage = LocalStorage(root)
    stack_sha = stack_sha256 or file_sha256(stack_file)
    by_id = soft.set_index("id", drop=False)
    for k, ids in enumerate(plan_shards(soft["id"].astype(str).tolist(), int(sl["shard_size"]))):
        frame = by_id.loc[ids].reset_index(drop=True)
        manifest = {"row_count": len(frame), "ids_sha256": ids_sha256(ids),
                    "prompt_version": sl["prompt_version"], "model_version": sl["model_version"],
                    "stack_version": "stack-v1", "stack_sha256": stack_sha,
                    "stack_file": "artifacts/stack/stack_p2-zs-v2b.json", "synthetic": True}
        write_shard(storage, sl["prefix"], shard_name(k), frame, manifest)
    return sl["prefix"]


def build(out: str | Path, n: int = 600, seed: int = 42, failed: int = 0) -> dict[str, Any]:
    """``<out>/wiki_train_raw.parquet`` and the store under ``<out>/store``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = make_rows(n, seed)
    rows.to_parquet(out / "wiki_train_raw.parquet", index=False)
    prefix = write_store(make_soft_labels(rows, seed, failed), out / "store")
    summary = {"rows": n, "failed": failed, "raw_train": str(out / "wiki_train_raw.parquet"),
               "store": str(out / "store"), "prefix": prefix}
    logger.info("synthetic data: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic student inputs (no real text).")
    parser.add_argument("--out", default=str(PIPELINE_ROOT / "outputs" / "synthetic"))
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--failed", type=int, default=0, help="rows with status failed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.out, args.rows, args.seed, args.failed)


if __name__ == "__main__":
    main()
