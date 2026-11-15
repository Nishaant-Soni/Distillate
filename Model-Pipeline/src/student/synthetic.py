"""Synthetic, contract-shaped inputs for the student: raw train rows and a soft-label store.

No real comment text: each row is neutral filler words plus marker words (``marktoxicity``, ...)
that make its labels learnable. The soft-label shards follow the spec exactly (§5 columns,
manifests with ``ids_sha256`` and ``stack_sha256`` of the committed stack file), so the
student's validation and join run unchanged on them. The two attribution arms are written
too: calibrated-Qwen shards (``calibration_sha256`` of the committed calibration) and a
TF-IDF out-of-fold file with its manifest.

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
CALIBRATION_FILE = PIPELINE_ROOT / "artifacts" / "pilot" / "calibration_p2-zs-v2b.json"
CALQWEN = {"prefix": "soft_labels/calqwen-v1/p2-zs-v2b/",
           "model_version": "calqwen-v1+Qwen/Qwen3-4B-Instruct-2507@"
                            "cdbee75f17c01a7cc42f958dc650907174af0554"}
OOF_PATH = "tfidf/tfidf-lr-v1-oof/tfidf_oof.parquet"
OOF_MODEL_VERSION = "tfidf-lr-v1"


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


def make_soft_labels(rows, seed: int = 42, failed: int = 0, model_version: str | None = None):
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
    out["model_version"] = pd.Series([model_version or SOFT_DEFAULTS["model_version"]]
                                     * len(out), dtype="string")
    return out


def write_store(soft, root: str | Path, soft_cfg: Mapping[str, Any] | None = None,
                stack_file: str | Path = STACK_FILE, stack_sha256: str | None = None,
                arm: str = "stack") -> str:
    """Write ``soft`` as shards under ``<root>/<prefix>`` per the shard plan; return the prefix.

    ``arm="calqwen"`` writes calibrated-Qwen manifests (``calibration_sha256`` of the
    committed calibration) under the calqwen prefix instead of the stack's.
    """
    from src.student.data import file_sha256
    from src.teacher.shards import ids_sha256, plan_shards, shard_name
    from src.teacher.storage import LocalStorage, write_shard

    sl = {**SOFT_DEFAULTS, **(CALQWEN if arm == "calqwen" else {}), **(soft_cfg or {})}
    storage = LocalStorage(root)
    if arm == "calqwen":
        lineage = {"calqwen_version": "calqwen-v1", "calibration_version": "platt-v1",
                   "calibration_file": "artifacts/pilot/calibration_p2-zs-v2b.json",
                   "calibration_sha256": file_sha256(CALIBRATION_FILE)}
    else:
        lineage = {"stack_version": "stack-v1",
                   "stack_sha256": stack_sha256 or file_sha256(stack_file),
                   "stack_file": "artifacts/stack/stack_p2-zs-v2b.json"}
    by_id = soft.set_index("id", drop=False)
    for k, ids in enumerate(plan_shards(soft["id"].astype(str).tolist(), int(sl["shard_size"]))):
        frame = by_id.loc[ids].reset_index(drop=True)
        manifest = {"row_count": len(frame), "ids_sha256": ids_sha256(ids),
                    "prompt_version": sl["prompt_version"], "model_version": sl["model_version"],
                    **lineage, "synthetic": True}
        write_shard(storage, sl["prefix"], shard_name(k), frame, manifest)
    return sl["prefix"]


def write_oof(rows, root: str | Path, manifest_path: str | Path, seed: int = 42) -> str:
    """A TF-IDF-like out-of-fold file at ``<root>/<OOF_PATH>`` plus its manifest."""
    import json

    import numpy as np
    import pandas as pd

    from src.student.data import file_sha256

    soft = make_soft_labels(rows, seed + 7)
    out = pd.DataFrame({"id": rows["id"].astype("string"),
                        "fold": (np.arange(len(rows)) % 5).astype("int8")})
    for label in LABELS:
        out[f"p_{label}"] = soft[f"p_{label}"]
    out["model_version"] = pd.Series([OOF_MODEL_VERSION] * len(out), dtype="string")
    target = Path(root) / OOF_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    manifest = {"file": target.name, "sha256": file_sha256(target), "row_count": len(out),
                "folds": 5, "seed": seed, "baseline_version": OOF_MODEL_VERSION,
                "synthetic": True}
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return OOF_PATH


def build(out: str | Path, n: int = 600, seed: int = 42, failed: int = 0) -> dict[str, Any]:
    """``<out>/wiki_train_raw.parquet`` and the store under ``<out>/store``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = make_rows(n, seed)
    rows.to_parquet(out / "wiki_train_raw.parquet", index=False)
    prefix = write_store(make_soft_labels(rows, seed, failed), out / "store")
    write_store(make_soft_labels(rows, seed + 3, model_version=CALQWEN["model_version"]),
                out / "store", arm="calqwen")
    write_oof(rows, out / "store", out / "tfidf_oof.manifest.json", seed)
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
