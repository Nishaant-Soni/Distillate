"""Tests for scripts/preprocess.py, using tiny made-up comments."""

import pandas as pd
import pytest

from scripts.columns import (
    CIVIL_CLEAN_COLUMNS,
    CIVIL_IDENTITY_EXTRA_COLUMNS,
    IDENTITY_COLUMNS,
    LABELS,
    WIKI_CLEAN_COLUMNS,
)
from scripts.errors import LabelError
from scripts.preprocess import (
    add_text_features,
    clean_text,
    clean_texts,
    load_identity_pattern,
    nonalpha_share,
    parse_created_date,
    process_civil_frame,
    process_wiki_frame,
    to_binary_labels,
    to_civil_labels,
    to_identity_numbers,
)

LABELS_CFG = {"civil_positive_threshold": 0.5, "civil_dropped_labels": ["sexual_explicit"]}


def wiki_rows(**labels):
    base = {"id": ["a", "b"], "text": ["hello", "world"]}
    base.update({label: labels.get(label, [0.0, 1.0]) for label in LABELS})
    return pd.DataFrame(base)


def civil_rows(texts, dates, **extra):
    n = len(texts)
    data = {
        "id": [str(i) for i in range(n)],
        "text": texts,
        "created_date": dates,
        "publication_id": ["1"] * n,
        "article_id": ["2"] * n,
        "sexual_explicit": [0.0] * n,
        **{label: [0.0] * n for label in LABELS},
    }
    data.update(extra)
    return pd.DataFrame(data)


# ---- text cleaning


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a &amp; b", "a & b"),
        ("a &amp;amp; b", "a & b"),
        ("  lots   of\n\tspace ", "lots of space"),
        ("ｆｕｌｌｗｉｄｔｈ", "fullwidth"),
        ("you 1d10t!!!", "you 1d10t!!!"),
        ("STOP SHOUTING", "STOP SHOUTING"),
        (None, ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_empty_comments_dropped_but_na_and_null_kept():
    df = pd.DataFrame({"id": list("abcde"), "text": ["NA", "null", "", "   ", "&nbsp;"]})
    out, dropped = clean_texts(df)
    assert out["text"].tolist() == ["NA", "null"]
    assert dropped == {"empty_in_source": 2, "empty_after_clean": 1}


# ---- labels


def test_wiki_labels_become_int8():
    out = to_binary_labels(wiki_rows())
    assert all(out[label].dtype == "int8" for label in LABELS)
    assert out["toxicity"].tolist() == [0, 1]


def test_missing_wiki_label_rejected():
    with pytest.raises(LabelError):
        to_binary_labels(wiki_rows(threat=[0.0, float("nan")]))


def test_non_binary_wiki_label_rejected():
    with pytest.raises(LabelError):
        to_binary_labels(wiki_rows(insult=[0.0, 2.0]))


def test_civil_threshold_boundary_and_frac_kept():
    df = pd.DataFrame(
        {"id": ["a", "b"], "sexual_explicit": [0.0, 0.0], **{lab: [0.5, 0.4999] for lab in LABELS}}
    )
    out = to_civil_labels(df, 0.5, ["sexual_explicit"])
    assert out["toxicity"].tolist() == [1, 0]
    assert out["toxicity_frac"].dtype == "float32"
    assert "sexual_explicit" not in out.columns


def test_civil_label_out_of_range_rejected():
    df = pd.DataFrame({"id": ["a"], **{label: [1.5] for label in LABELS}})
    with pytest.raises(LabelError):
        to_civil_labels(df, 0.5, [])


# ---- civil fields


def test_dates_with_and_without_fractional_seconds_parse():
    out = parse_created_date(
        pd.Series(["2017-01-14 06:43:20.123456+00", "2017-01-14 06:43:20+00"])
    )
    assert out.notna().all()
    assert str(out.dt.tz) == "UTC"


def test_blank_identity_values_become_nan():
    columns = [*IDENTITY_COLUMNS, "identity_annotator_count"]
    out = to_identity_numbers(pd.DataFrame({column: ["", "0.5"] for column in columns}))
    assert out["muslim"].isna().tolist() == [True, False]
    assert out["muslim"].dtype == "float32"


# ---- derived columns


def test_identity_terms_match_whole_words_only(tmp_path):
    lexicon = tmp_path / "terms.txt"
    lexicon.write_text("# comment line\nmuslim\ngay\n\n", encoding="utf-8")
    pattern = load_identity_pattern(lexicon)
    df = pd.DataFrame({"text": ["A Muslim neighbor", "gayety", "nothing here"]})
    out = add_text_features(df, pattern, obf_threshold=0.3)
    assert out["has_identity_term"].tolist() == [True, False, False]


def test_missing_lexicon_gives_none(tmp_path):
    assert load_identity_pattern(tmp_path / "missing.txt") is None


def test_char_len_and_obfuscation_flag():
    out = add_text_features(pd.DataFrame({"text": ["hello there", "1d10t"]}), None, 0.3)
    assert out["char_len"].tolist() == [11, 5]
    assert out["char_len"].dtype == "int32"
    assert out["has_obfuscation"].tolist() == [False, True]
    assert nonalpha_share("") == 0.0


# ---- whole tables match the contract


def test_wiki_frame_matches_contract_columns():
    out, _ = process_wiki_frame(wiki_rows(), None, 0.3)
    assert list(out.columns) == WIKI_CLEAN_COLUMNS


def test_civil_frame_identity_columns_only_in_identities_file():
    dates = ["2017-01-14 06:43:20.1+00", "2017-01-14 06:43:20+00"]
    identities = {column: ["", "1"] for column in CIVIL_IDENTITY_EXTRA_COLUMNS}
    df = civil_rows(["NA", "hi"], dates, **identities)
    plain, dropped = process_civil_frame(df, None, 0.3, LABELS_CFG, with_identities=False)
    with_ids, _ = process_civil_frame(df, None, 0.3, LABELS_CFG, with_identities=True)
    assert list(plain.columns) == CIVIL_CLEAN_COLUMNS
    assert list(with_ids.columns) == CIVIL_CLEAN_COLUMNS + CIVIL_IDENTITY_EXTRA_COLUMNS
    assert plain["text"].tolist() == ["NA", "hi"]
    assert dropped["missing_created_date_kept"] == 0


def test_missing_text_dropped():
    df = pd.DataFrame({"id": ["a", "b"], "text": [None, "hello"]})
    out, dropped = clean_texts(df)
    assert out["id"].tolist() == ["b"]
    assert dropped["empty_in_source"] == 1
