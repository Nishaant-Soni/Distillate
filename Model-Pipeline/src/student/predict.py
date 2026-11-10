"""Prediction files (CONTRACTS.md §2.2): ``id``, six ``p_<label>`` columns, ``model_version``.

This is the only thing the evaluation and bias code reads, so every student evaluation
writes one. No text goes into the file.

    python -m src.student.predict --model-dir outputs/student/<run>/model \
        --input data/some_split.parquet --output outputs/student/<run>/predictions/x.parquet
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.teacher.config import LABELS, get_logger

logger = get_logger(__name__)

PREDICTION_COLUMNS = ["id", *[f"p_{label}" for label in LABELS], "model_version"]


def predict_logits(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int,
                   device, fp16: bool = False):
    """``(rows, 6)`` float32 numpy logits, in input order."""
    import numpy as np
    import torch

    model.eval()
    out = []
    use_amp = fp16 and getattr(device, "type", str(device)) == "cuda"
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[start:start + batch_size]), truncation=True,
                              max_length=max_length, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), dtype="float32")


def prediction_frame(ids: Sequence[str], logits, model_version: str):
    """The §2.2 prediction file from logits."""
    import numpy as np
    import pandas as pd

    p = 1 / (1 + np.exp(-np.asarray(logits, dtype="float64")))
    frame = pd.DataFrame({"id": pd.Series(list(ids), dtype="string")})
    for k, label in enumerate(LABELS):
        frame[f"p_{label}"] = p[:, k].astype("float32")
    frame["model_version"] = pd.Series([model_version] * len(frame), dtype="string")
    return frame


def check_prediction_frame(frame) -> None:
    """Raise unless ``frame`` is a valid §2.2 prediction file."""
    import numpy as np

    if list(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"columns {list(frame.columns)}, expected {PREDICTION_COLUMNS}")
    if frame["id"].duplicated().any():
        raise ValueError("duplicate ids")
    p = frame[PREDICTION_COLUMNS[1:-1]].to_numpy(dtype="float64")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("probabilities outside [0, 1] or NaN")
    if frame["model_version"].nunique() != 1:
        raise ValueError("model_version must be one value per file")


def main() -> None:
    import pandas as pd
    import torch

    from src.student.model import load_trained

    parser = argparse.ArgumentParser(description="Write a prediction file with a trained student.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="Parquet with id and text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    meta = json.loads((Path(args.model_dir) / "meta.json").read_text())
    model, tokenizer = load_trained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = pd.read_parquet(args.input, columns=["id", "text"])
    logits = predict_logits(model, tokenizer, rows["text"].fillna("").tolist(),
                            meta["max_length"], args.batch_size, device)
    frame = prediction_frame(rows["id"].astype(str), logits, meta["model_version"])
    check_prediction_frame(frame)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    logger.info("wrote %d predictions to %s", len(frame), args.output)


if __name__ == "__main__":
    main()
