"""Tests for scripts/dedupe.py, using tiny made-up comments."""

import pandas as pd
import pytest

from scripts.columns import LABELS
from scripts.dedupe import exact_dedupe, jaccard, make_index, near_dedupe, norm_key, shingles

LONG = " ".join(f"word{i}" for i in range(60))  # long comment for near-duplicate tests


def frame(rows):
    """rows: list of (id, text) or (id, text, toxicity)."""
    data = []
    for row in rows:
        row_id, text, toxic = (*row, 0) if len(row) == 2 else row
        labels = {label: 0 for label in LABELS}
        data.append({"id": row_id, "text": text, **labels, "toxicity": toxic})
    return pd.DataFrame(data, columns=["id", "text", *LABELS])


def near(train, gate):
    return near_dedupe(train, gate, threshold=0.9, num_perm=128, size=3, seed=42)


def test_norm_key_ignores_case_and_punctuation():
    assert norm_key("Hello, World!") == norm_key("hello world")


def test_short_comments_compare_whole_text():
    assert jaccard(shingles("thanks", 3), shingles("no thanks", 3)) == 0.0


def test_exact_duplicates_removed_first_id_kept():
    kept, records, _ = exact_dedupe(
        frame([("b", "Hello world"), ("a", "hello, WORLD!")]), frame([])
    )
    assert kept["id"].tolist() == ["a"]
    assert records[0]["reason"] == "exact_within_train"


def test_train_row_copying_gate_removed_and_gate_untouched():
    gate = frame([("g1", "see you tomorrow")])
    kept, records, _ = exact_dedupe(frame([("t1", "See you tomorrow!"), ("t2", "different")]), gate)
    assert kept["id"].tolist() == ["t2"]
    assert records[0] == {
        "id": "t1",
        "reason": "exact_vs_gate",
        "matched_id": "g1",
        "jaccard": 1.0,
    }
    assert gate["id"].tolist() == ["g1"]


def test_label_conflicts_counted():
    _, _, conflicts = exact_dedupe(
        frame([("a", "same text", 0), ("b", "same text", 1)]), frame([])
    )
    assert conflicts == 1


def test_punctuation_only_comments_not_treated_as_duplicates():
    kept, records, _ = exact_dedupe(frame([("a", "!!!"), ("b", "???")]), frame([]))
    assert len(kept) == 2 and records == []


def test_near_duplicate_within_train_removed():
    kept, records = near(frame([("a", LONG), ("b", LONG + " thanks")]), frame([]))
    assert kept["id"].tolist() == ["a"]
    assert records[0]["reason"] == "near_within_train"


def test_near_duplicate_of_gate_removed():
    kept, records = near(frame([("t1", LONG + " thanks")]), frame([("g1", LONG)]))
    assert kept.empty
    assert records[0]["matched_id"] == "g1"


def test_distinct_comments_kept():
    train = frame([("a", LONG), ("b", " ".join(f"other{i}" for i in range(60))), ("c", "thanks")])
    kept, records = near(train, frame([("g", "no thanks")]))
    assert len(kept) == 3 and records == []


def test_same_result_every_run():
    train = frame([("c", LONG), ("a", LONG + " x"), ("b", "unique words here")])
    first, _ = near(train, frame([]))
    second, _ = near(train.sample(frac=1, random_state=1), frame([]))
    assert first["id"].tolist() == second["id"].tolist()


def test_lsh_bands_must_divide_num_perm():
    with pytest.raises(ValueError):
        make_index(0.9, 128, bands=3)
