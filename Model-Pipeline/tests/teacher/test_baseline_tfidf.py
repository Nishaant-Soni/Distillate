import numpy as np
import pandas as pd

from src.teacher.baseline_tfidf import train_predict
from src.teacher.config import LABELS

PARAMS = {"model_version": "tfidf-lr-v1", "word_ngram_range": [1, 2], "char_ngram_range": [3, 5],
          "max_features": 5000, "C": 4.0}


def corpus(n, seed):
    rng = np.random.default_rng(seed)
    bad = rng.random(n) < 0.3
    text = np.where(bad, "you are a stupid idiot", "thanks for the helpful edit")
    df = pd.DataFrame({"id": [f"{seed}-{i}" for i in range(n)], "text": text})
    for label in LABELS:
        df[label] = bad.astype("int8")
    return df


def test_prediction_file_format_and_signal():
    pred = train_predict(corpus(300, 0), corpus(50, 1), PARAMS, seed=42)
    assert list(pred.columns) == ["id", *[f"p_{label}" for label in LABELS], "model_version"]
    assert (pred["model_version"] == "tfidf-lr-v1").all()
    assert all(pred[f"p_{label}"].dtype == np.float32 for label in LABELS)
    test = corpus(50, 1)
    bad = test["toxicity"] == 1
    assert pred.loc[bad, "p_toxicity"].min() > pred.loc[~bad, "p_toxicity"].max()
