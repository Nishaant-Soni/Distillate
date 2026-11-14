import numpy as np
import pandas as pd
import pytest

from src.sensitivity.attributions import (
    MASK,
    Attribution,
    display,
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
    # "you" and "stupid" are the first and last words, so hidden; zzname is too rare.
    assert line == f"{MASK} **idiot** (+2.00) {MASK}"
    assert "zzname" not in line


def test_masked_example_shows_at_most_a_third_of_the_words():
    freq = dict.fromkeys(["you", "are", "an", "idiot", "so", "stupid"], 10)
    six = attribution(["you", "are", "an", "idiot", "so", "stupid"], [1, 2, 3, 4, 5, 6])
    shown = [w for w in masked_example(six, "toxicity", 4, freq, 5).split() if w[:2] == "**"]
    assert shown == ["**an**", "**so**"]          # 6 words: 2 visible; "idiot" is next to "so"
    two = attribution(["you", "idiot"], [1, 2])
    assert masked_example(two, "toxicity", 4, freq, 5) == MASK  # under 3 words: none


def test_masked_example_never_shows_adjacent_words_or_the_ends():
    words = [f"w{i}" for i in range(30)]
    freq = dict.fromkeys(words, 10)
    line = masked_example(attribution(words, np.arange(30, 0, -1.0)), "toxicity", 10, freq, 5)
    parts = line.split(" ")
    assert parts[0] == MASK and parts[-1] == MASK
    tokens = [t for t in parts if not t.startswith("(")]
    assert all(MASK in (a, b) for a, b in zip(tokens, tokens[1:], strict=False))
    assert [t.strip("*") for t in tokens if t != MASK] == [f"w{i}" for i in range(1, 20, 2)]


def test_slurs_are_shown_with_inner_letters_masked():
    import hashlib

    from src.sensitivity import attributions as mod

    word = "zzslurword"
    mod.MASKED_TERMS = mod.MASKED_TERMS | {hashlib.sha256(word.encode()).hexdigest()}
    try:
        assert display("ZZslurword!") == "z********d" and display("idiot") == "idiot"
        selection = pd.DataFrame({"id": ["a"], "label": "toxicity", "group": "caught",
                                  "p": 0.9, "y": 1})
        a = attribution(["you", word, "are", "so", "dumb"], [0.0, 3.0, 0.0, 0.0, 0.0])
        freq = dict.fromkeys(["you", word, "are", "so", "dumb"], 10)
        table = token_table(selection, {"a": a}, freq, min_doc_freq=5, min_occurrences=1,
                            top_k=5)
        assert table["word"].tolist() == ["z********d"]
        line = masked_example(a, "toxicity", 2, freq, 5)
        assert line == f"{MASK} **z********d** (+3.00) {MASK}"
    finally:
        mod.MASKED_TERMS = mod.MASKED_TERMS - {hashlib.sha256(word.encode()).hexdigest()}


def ngrams(text, n=3):
    words = [normalize(w) for w in str(text).split()]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def visible_ngrams(line, n=3):
    """Runs of n words with no mask between them, as a masked line shows them."""
    tokens = [t.strip("*") for t in line.split() if not t.startswith("(")]
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
            if MASK not in tokens[i:i + n]}


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
    # A unique rare word in every row, so whichever comments are explained carry one.
    frame["text"] = [f"zzsecret{i} {t} zzsecretend{i}" for i, t in enumerate(frame["text"])]
    p = scorer.probabilities(frame["text"])
    # Make the threshold split rows into every group.
    thresholds = {label: float(np.median(p[:, k])) for k, label in enumerate(LABELS)}
    out = shap_analysis(scorer, frame, p, thresholds, SHAP_CFG, seed=42)
    assert set(out["selection"]["group"]) == {"caught", "missed", "false_alarm"}
    assert len(out["examples"]) > 0
    assert not out["tokens"]["word"].str.contains("zzsecret").any()
    assert not out["examples"]["masked"].str.contains("zzsecret").any()
    freq = document_frequency(frame["text"])
    comment_ngrams = set().union(*(ngrams(t) for t in frame["text"]))
    for line in out["examples"]["masked"]:
        shown = [w.strip("*") for w in line.split() if w.startswith("**")]
        assert all(freq[w] >= SHAP_CFG["min_doc_freq"] for w in shown)
        assert len(shown) <= SHAP_CFG["visible_words"]
        assert not visible_ngrams(line, 2) and not visible_ngrams(line) & comment_ngrams


def test_examples_take_the_first_rows_per_group():
    selection = pd.DataFrame({"id": ["a", "b", "c"], "label": "toxicity",
                              "group": ["caught", "caught", "missed"], "p": [0.9, 0.8, 0.1],
                              "y": 1})
    attributions = {i: attribution(["you", "idiot", "are"], [0.0, 1.0, 0.0]) for i in "abc"}
    out = examples(selection, attributions, per_group=1, visible=1,
                   doc_freq={"you": 9, "are": 9, "idiot": 9}, min_doc_freq=1)
    assert out["group"].tolist() == ["caught", "missed"]
    assert out["correct"].tolist() == [True, False]
    assert out.loc[0, "masked"] == f"{MASK} **idiot** (+1.00) {MASK}"
