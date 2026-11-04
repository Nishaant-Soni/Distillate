import numpy as np
import pandas as pd
import pytest

from src.sensitivity.attributions import (
    MASK,
    Attribution,
    document_frequency,
    examples,
    explain,
    masked_example,
    merge_words,
    normalize,
    select_examples,
    shap_analysis,
    token_table,
)
from src.sensitivity.scorer import load_scorer
from src.teacher.config import LABELS
from tests.sensitivity.test_sweeps import validation

THRESHOLDS = dict.fromkeys(LABELS, 0.5)
SHAP_CFG = {"per_group": 3, "max_evals": 100, "batch_size": 50, "top_k": 5,
            "min_occurrences": 1, "min_doc_freq": 2, "examples_per_group": 2,
            "visible_words": 2}


def attribution(words, toxicity):
    """Values only toward toxicity; the other labels get 0."""
    values = np.zeros((len(words), len(LABELS)))
    values[:, 0] = toxicity
    return Attribution(tuple(words), values)


def test_normalize_and_document_frequency():
    assert normalize("  Idiot!! ") == "idiot" and normalize("!!") == "!!"
    freq = document_frequency(["You idiot!", "idiot idiot", "thanks"])
    assert freq["idiot"] == 2 and freq["you"] == 1 and freq["thanks"] == 1


def test_merge_words_joins_pieces_and_sums_values():
    pieces = ["", "you ", "idiot", "ic ", "!", ""]
    values = np.arange(12, dtype=float).reshape(6, 2)
    merged = merge_words(pieces, values)
    assert merged.words == ("you", "idiotic", "!")
    np.testing.assert_array_equal(merged.values, [[2, 3], [10, 12], [8, 9]])


def test_select_examples_groups_and_caps():
    frame = pd.DataFrame({"id": [f"r{i}" for i in range(8)]})
    for label in LABELS:
        frame[label] = [1, 1, 1, 1, 0, 0, 0, 0]
    p = np.tile(np.array([[0.9], [0.8], [0.7], [0.2], [0.6], [0.1], [0.1], [0.1]]),
                (1, len(LABELS)))
    selection = select_examples(frame, p, THRESHOLDS, per_group=2, seed=42)
    tox = selection[selection["label"] == "toxicity"].groupby("group")["id"].apply(set)
    assert tox["caught"] <= {"r0", "r1", "r2"} and len(tox["caught"]) == 2
    assert tox["missed"] == {"r3"} and tox["false_alarm"] == {"r4"}
    assert selection.equals(select_examples(frame, p, THRESHOLDS, per_group=2, seed=42))


def test_token_table_ranks_and_filters():
    selection = pd.DataFrame({"id": ["a", "b"], "label": "toxicity", "group": "caught",
                              "p": 0.9, "y": 1})
    attributions = {"a": attribution(["you", "Idiot", "zzname"], [0.1, 2.0, 5.0]),
                    "b": attribution(["idiot!", "you"], [1.0, 0.3])}
    freq = {"you": 10, "idiot": 10, "zzname": 1}
    table = token_table(selection, attributions, freq, min_doc_freq=5, min_occurrences=2,
                        top_k=5)
    assert table["word"].tolist() == ["idiot", "you"]          # zzname: too rare to show
    assert table.loc[0, "mean_value"] == pytest.approx(1.5)
    assert table.loc[0, "occurrences"] == 2 and table.loc[0, "comments"] == 2


def test_masked_example_shows_only_top_common_words():
    words = ["you", "are", "a", "zzname", "the", "idiot", "and", "stupid"]
    a = attribution(words, [0.1, -0.2, 0.0, 9.0, 0.0, 2.0, -1.0, 0.5])
    freq = dict.fromkeys(["you", "are", "a", "the", "idiot", "and", "stupid"], 10)
    line = masked_example(a, "toxicity", visible=2, doc_freq=freq, min_doc_freq=5)
    assert line == f"{MASK} **idiot** (+2.00) {MASK} **stupid** (+0.50)"
    assert "zzname" not in line


def test_masked_example_shows_at_most_a_third_of_the_words():
    freq = dict.fromkeys(["you", "are", "an", "idiot", "so", "stupid"], 10)
    six = attribution(["you", "are", "an", "idiot", "so", "stupid"], [1, 2, 3, 4, 5, 6])
    shown = [w for w in masked_example(six, "toxicity", 4, freq, 5).split() if w[:2] == "**"]
    assert shown == ["**so**", "**stupid**"]                  # 6 words: 2 visible, not 4
    two = attribution(["you", "idiot"], [1, 2])
    assert masked_example(two, "toxicity", 4, freq, 5) == MASK  # under 3 words: none


def test_explain_on_the_tiny_student(student_dir):
    scorer = load_scorer(student_dir)
    first = explain(scorer, ["you are an idiotic", "thanks"], max_evals=100, batch_size=50)
    assert first[0].words == ("you", "are", "an", "idiotic")
    assert first[0].values.shape == (4, len(LABELS)) and first[1].values.shape == (1, 6)
    again = explain(scorer, ["you are an idiotic", "thanks"], max_evals=100, batch_size=50)
    np.testing.assert_array_equal(first[0].values, again[0].values)


def test_shap_analysis_never_reports_rare_words(student_dir):
    scorer = load_scorer(student_dir)
    frame = validation(n=40)
    frame.loc[0, "text"] = "zzsecretname " + frame.loc[0, "text"]
    p = scorer.probabilities(frame["text"])
    # Make the threshold split rows into every group.
    thresholds = {label: float(np.median(p[:, k])) for k, label in enumerate(LABELS)}
    out = shap_analysis(scorer, frame, p, thresholds, SHAP_CFG, seed=42)
    assert set(out["selection"]["group"]) == {"caught", "missed", "false_alarm"}
    assert "zzsecretname" not in set(out["tokens"]["word"])
    assert not out["examples"]["masked"].str.contains("zzsecretname").any()
    freq = document_frequency(frame["text"])
    for line in out["examples"]["masked"]:
        shown = [w.strip("*") for w in line.split() if w.startswith("**")]
        assert all(freq[w] >= SHAP_CFG["min_doc_freq"] for w in shown)
        assert len(shown) <= SHAP_CFG["visible_words"]


def test_examples_take_the_first_rows_per_group():
    selection = pd.DataFrame({"id": ["a", "b", "c"], "label": "toxicity",
                              "group": ["caught", "caught", "missed"], "p": [0.9, 0.8, 0.1],
                              "y": 1})
    attributions = {i: attribution(["you", "are", "idiot"], [0.0, 0.0, 1.0]) for i in "abc"}
    out = examples(selection, attributions, per_group=1, visible=1,
                   doc_freq={"you": 9, "are": 9, "idiot": 9}, min_doc_freq=1)
    assert out["group"].tolist() == ["caught", "missed"]
    assert out["correct"].tolist() == [True, False]
    assert out.loc[0, "masked"] == f"{MASK} **idiot** (+1.00)"
