import json

import numpy as np
import pandas as pd
import pytest

from src.sensitivity.obfuscation import substitute
from src.sensitivity.scorer import Scorer, load_scorer
from src.sensitivity.sweeps import (
    assign_bands,
    band_names,
    label_table,
    load_thresholds,
    obfuscation_sweep,
    token_lengths,
    truncation_sweep,
)
from src.teacher.config import LABELS
from tests.sensitivity.conftest import WORDS

TABLE = {"a": "4", "e": "3", "i": "1", "l": "1", "o": "0", "s": "5", "t": "7"}
THRESHOLDS = dict.fromkeys(LABELS, 0.5)


def validation(n=120, seed=0):
    """Made-up validation rows of 2 to 40 words from the tiny vocabulary; every label mixed."""
    rng = np.random.default_rng(seed)
    texts = [" ".join(rng.choice(WORDS, rng.integers(2, 41))) for _ in range(n)]
    frame = pd.DataFrame({"id": [f"v{i:04d}" for i in range(n)], "text": texts})
    for k, label in enumerate(LABELS):
        frame[label] = (np.arange(n) % (k + 2) == 0).astype("int8")
    return frame


def test_label_table_ranks_on_logits():
    truth = pd.DataFrame({label: [0, 0, 1, 1] for label in LABELS})
    # 30 and 40 both round to p == 1.0 in float32, so only the logits order them.
    logits = np.tile(np.array([[-2.0], [35.0], [30.0], [40.0]]), (1, len(LABELS)))
    table = label_table(truth, logits, THRESHOLDS)
    assert table.loc["toxicity", "auc"] == pytest.approx(0.75)
    assert table.loc["toxicity", "recall"] == 1.0
    assert table.loc["toxicity", "flag_rate"] == 0.75
    assert list(table.index) == [*LABELS, "mean"]


def test_label_table_mean_skips_one_class_labels():
    truth = pd.DataFrame({label: [0, 1, 0, 1] for label in LABELS}).assign(threat=0)
    logits = np.tile(np.array([[-1.0], [1.0], [-1.0], [1.0]]), (1, len(LABELS)))
    table = label_table(truth, logits, THRESHOLDS)
    assert np.isnan(table.loc["threat", "auc"]) and np.isnan(table.loc["threat", "recall"])
    assert table.loc["mean", "auc"] == 1.0


def test_thresholds(tmp_path):
    assert load_thresholds(None, 0.4) == dict.fromkeys(LABELS, 0.4)
    assert load_thresholds(tmp_path, 0.5) == THRESHOLDS
    (tmp_path / "thresholds.json").write_text(json.dumps({**THRESHOLDS, "threat": 0.2}))
    assert load_thresholds(tmp_path, 0.5)["threat"] == 0.2
    (tmp_path / "thresholds.json").write_text(json.dumps({"toxicity": 0.3}))
    with pytest.raises(ValueError, match="no threshold"):
        load_thresholds(tmp_path, 0.5)


def test_bands():
    assert band_names([64, 128, 256]) == ["<=64", "65-128", "129-256", ">256"]
    np.testing.assert_array_equal(assign_bands(np.array([1, 64, 65, 128, 129, 256, 257]),
                                               [64, 128, 256]),
                                  ["<=64", "<=64", "65-128", "65-128", "129-256", "129-256",
                                   ">256"])


def test_truncation_sweep(student_dir):
    scorer = load_scorer(student_dir)
    frame = validation()
    lengths = token_lengths(scorer, frame["text"])
    assert lengths.min() >= 4 and lengths.max() == 42       # 2 to 40 words plus [CLS], [SEP]
    out = truncation_sweep(scorer, frame, [8, 16, 64], [8, 16], THRESHOLDS)
    overall = out["overall"].set_index(["max_length", "label"])
    assert len(overall) == 3 * (len(LABELS) + 1)
    shares = overall.xs("mean", level="label")["truncated_share"]
    assert shares[8] > shares[16] > shares[64] == 0.0
    # Rescoring at max_length 64 truncates nothing: the metrics equal the untruncated ones.
    full = label_table(frame, scorer.logits(frame["text"], max_length=512), THRESHOLDS)
    np.testing.assert_allclose(overall.loc[64].loc[full.index, "auc"], full["auc"])
    # The bands hold every row once per max_length.
    bands = out["bands"]
    counts = bands[bands["label"] == "mean"].groupby("max_length")["n"].sum()
    assert (counts == len(frame)).all()
    expected = pd.Series(assign_bands(lengths, [8, 16])).value_counts()
    got = bands[(bands["label"] == "mean") & (bands["max_length"] == 8)].set_index("band")["n"]
    assert got.sort_index().to_dict() == expected.sort_index().to_dict()


class CountingScorer(Scorer):
    """Logits from the share of digits in the text: obfuscation pushes them all up or down."""

    def __init__(self, sign=-1.0):
        super().__init__(tokenizer=None, max_length=512, model_version="fake")
        self.sign = sign

    def logits(self, texts, max_length=None):
        digits = np.array([sum(c.isdigit() for c in t) / max(len(t), 1) for t in texts])
        base = np.array([3.0 if "idiot" in t or "1d" in t else -3.0 for t in texts])
        return np.tile((base + self.sign * 20 * digits)[:, None], (1, len(LABELS))).astype(
            "float32")


def test_obfuscation_sweep():
    frame = pd.DataFrame({"id": ["a", "b", "c", "d"],
                          "text": ["you idiot", "thanks for the edit", "idiot idiot", "talk"]})
    for label in LABELS:
        frame[label] = [1, 0, 1, 0]
    out = obfuscation_sweep(CountingScorer(), frame, [0.0, 0.5, 1.0], TABLE, 42, THRESHOLDS)
    mean = out[out["label"] == "mean"].set_index("rate")
    assert mean.loc[0.0, "chars_changed"] == 0.0
    assert mean.loc[0.0, "chars_changed"] < mean.loc[0.5, "chars_changed"] < \
        mean.loc[1.0, "chars_changed"]
    assert mean.loc[0.0, "missed"] == 0.0 and mean.loc[0.0, "recall"] == 1.0
    # Fully obfuscated, the digits drag the toxic scores under the threshold: all missed.
    assert mean.loc[1.0, "missed"] == 1.0 and mean.loc[1.0, "recall"] == 0.0
    assert len(out) == 3 * (len(LABELS) + 1)


def test_obfuscation_sweep_scores_the_substituted_text():
    seen = []

    class Recording(CountingScorer):
        def logits(self, texts, max_length=None):
            seen.append(list(texts))
            return super().logits(texts, max_length)

    frame = validation(n=10)
    obfuscation_sweep(Recording(), frame, [0.5], TABLE, 42, THRESHOLDS)
    expected = [substitute(t, 0.5, i, 42, TABLE) for t, i in zip(frame["text"], frame["id"],
                                                                 strict=True)]
    assert seen[-1] == expected                    # seen[0] is the clean pass
