"""TF-IDF + logistic regression baseline (``tfidf-lr-v1``), the bar the teacher must beat.

Word 1-2-grams and character 3-5-grams (sublinear TF), one logistic regression per label.
Trained on raw train minus the pilot and few-shot rows, so the pilot is a clean holdout;
predicts the pilot in the prediction-file format (``id``, ``p_<label>``, ``model_version``).

    python -m src.teacher.baseline_tfidf   # writes pilot_tfidf.parquet and metrics_tfidf.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger, load_config

logger = get_logger(__name__)


def train_predict(train, predict, params: Mapping[str, Any], seed: int):
    """Fit on ``train`` (``text``, labels); predict ``predict`` (``id``, ``text``)."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizers = [
        TfidfVectorizer(analyzer="word", ngram_range=tuple(params["word_ngram_range"]),
                        max_features=params["max_features"], sublinear_tf=True,
                        strip_accents="unicode", dtype=np.float32),
        TfidfVectorizer(analyzer="char", ngram_range=tuple(params["char_ngram_range"]),
                        max_features=params["max_features"], sublinear_tf=True,
                        strip_accents="unicode", dtype=np.float32),
    ]
    x_train = hstack([v.fit_transform(train["text"]) for v in vectorizers]).tocsr()
    x_pred = hstack([v.transform(predict["text"]) for v in vectorizers]).tocsr()

    out = pd.DataFrame({"id": predict["id"].astype("string").to_numpy()})
    for label in LABELS:
        model = LogisticRegression(C=params["C"], solver="liblinear", max_iter=1000,
                                   random_state=seed)
        model.fit(x_train, train[label].to_numpy())
        out[f"p_{label}"] = model.predict_proba(x_pred)[:, 1].astype("float32")
    out["model_version"] = pd.Series([params["model_version"]] * len(out), dtype="string")
    return out


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Train on train minus pilot/few-shot rows, predict the pilot, save predictions and metrics."""
    import pandas as pd

    from src.teacher.metrics import evaluate

    data_dir, pilot_dir = Path(cfg["paths"]["data_dir"]), Path(cfg["paths"]["pilot_dir"])
    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    pilot_ids = pd.read_parquet(pilot_dir / "pilot_ids.parquet")
    fewshot = json.loads((pilot_dir / "fewshot_ids.json").read_text())["ids"]
    held_out = set(pilot_ids["id"]) | {i for ids in fewshot.values() for i in ids}

    fit_rows = train[~train["id"].isin(held_out)]
    pilot_rows = train[train["id"].isin(set(pilot_ids["id"]))]
    start = time.perf_counter()
    pred = train_predict(fit_rows, pilot_rows, cfg["baseline"], cfg["seed"])
    seconds = round(time.perf_counter() - start, 1)

    table = evaluate(pred, pilot_rows, pilot_ids, cfg["metrics"]["ece_bins"])
    pred.to_parquet(pilot_dir / "pilot_tfidf.parquet", index=False)
    result = {"model_version": cfg["baseline"]["model_version"], "train_rows": len(fit_rows),
              "pilot_rows": len(pilot_rows), "fit_seconds": seconds,
              "metrics": json.loads(table.round(5).to_json(orient="index"))}
    (pilot_dir / "metrics_tfidf.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info("TF-IDF on pilot (weighted):\n%s", table.round(4).to_string())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + logistic regression pilot baseline.")
    parser.add_argument("--config", default=None)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
