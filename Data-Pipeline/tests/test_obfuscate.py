"""Tests for scripts/obfuscate.py: the disguised-toxic-comment evaluation set."""

import random
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from scripts import obfuscate  # noqa: E402
from scripts.stubs import LABELS, wikipedia_frame  # noqa: E402
from scripts.utils import load_config  # noqa: E402

N_GATE = 2000
TARGET = 100
TOXIC_TEXT = "you are a complete idiot and your edits are worthless"


def _gate(n=N_GATE):
    df = wikipedia_frame(n, 42, "wiki_test")
    df.loc[df["toxicity"] == 1, "text"] = TOXIC_TEXT
    return df


def _write_gate(cfg, df):
    wiki = Path(cfg["paths"]["processed_dir"]) / "wikipedia"
    wiki.mkdir(parents=True, exist_ok=True)
    df.to_parquet(wiki / "gate.parquet", index=False)


@pytest.fixture
def cfg(tmp_path):
    """Config pointed at a temp dir, with a contract-shaped gate set and a tiny lexicon."""
    cfg = load_config()
    for key in ("processed_dir", "summaries_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    lexicon = tmp_path / "identity_terms.txt"
    lexicon.write_text("# test lexicon\nidiot\n", encoding="utf-8")
    cfg["paths"]["identity_terms"] = str(lexicon)
    cfg["splits"]["obfuscation_size"] = TARGET
    _write_gate(cfg, _gate())
    return cfg


def _out(cfg):
    return pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "obfuscation.parquet")


def test_writes_the_configured_number_of_rows(cfg):
    obfuscate.run(cfg)
    assert len(_out(cfg)) == TARGET


def test_ids_are_original_gate_ids_with_obf_suffix(cfg):
    obfuscate.run(cfg)
    out = _out(cfg)
    assert out["id"].str.endswith("_obf").all()
    toxic_ids = set(_gate().query("toxicity == 1")["id"])
    assert set(out["id"].str.removesuffix("_obf")) <= toxic_ids
    assert out["id"].is_unique


def test_text_changes_and_labels_do_not(cfg):
    obfuscate.run(cfg)
    out = _out(cfg).assign(orig=lambda d: d["id"].str.removesuffix("_obf"))
    gate = _gate().set_index("id")
    for _, row in out.iterrows():
        source = gate.loc[row["orig"]]
        assert row["text"] != source["text"], row["id"]
        for label in LABELS:
            assert row[label] == source[label], (row["id"], label)


def test_output_is_deterministic(cfg):
    obfuscate.run(cfg)
    first = _out(cfg)
    obfuscate.run(cfg)
    pd.testing.assert_frame_equal(_out(cfg), first)


def test_input_row_order_does_not_change_the_output(cfg):
    obfuscate.run(cfg)
    first = _out(cfg)
    _write_gate(cfg, _gate().sample(frac=1, random_state=3))
    obfuscate.run(cfg)
    pd.testing.assert_frame_equal(_out(cfg), first)


def test_all_three_techniques_are_used(cfg):
    summary = obfuscate.run(cfg)
    assert set(summary["techniques"]) == {"substitution", "spacing", "homoglyph"}
    assert sum(summary["techniques"].values()) == TARGET


def test_contract_columns_and_types_are_kept(cfg):
    obfuscate.run(cfg)
    out, gate = _out(cfg), _gate()
    assert list(out.columns) == list(gate.columns)
    for col in gate.columns:
        assert str(out[col].dtype) == str(gate[col].dtype), col


def test_derived_columns_are_recomputed_on_the_new_text(cfg):
    obfuscate.run(cfg)
    out = _out(cfg)
    assert (out["char_len"] == out["text"].str.len()).all()
    threshold = cfg["preprocess"]["obfuscation_nonalpha_share"]
    expected = out["text"].map(obfuscate.nonalpha_share) > threshold
    assert (out["has_obfuscation"] == expected).all()
    # The lexicon term survives only where the trick left the word intact.
    intact = out["text"].str.lower().str.contains(r"\bidiot\b")
    assert (out["has_identity_term"] == intact).all()


def test_fewer_toxic_comments_than_target_uses_all_of_them(cfg):
    small = _gate(200)
    _write_gate(cfg, small)
    cfg["splits"]["obfuscation_size"] = 10_000
    summary = obfuscate.run(cfg)
    assert summary["rows_out"] == int((small["toxicity"] == 1).sum())


def test_missing_gate_gives_a_clear_error(cfg):
    (Path(cfg["paths"]["processed_dir"]) / "wikipedia" / "gate.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="gate.parquet"):
        obfuscate.run(cfg)


def test_summary_records_counts(cfg):
    summary = obfuscate.run(cfg)
    assert summary["rows_in"]["gate"] == N_GATE
    assert summary["rows_out"] == TARGET
    assert (Path(cfg["paths"]["summaries_dir"]) / "obfuscate.json").exists()


# Unit tests for the individual tricks.

def test_substitution_swaps_letters_for_digits():
    out = obfuscate.substitute("idiot " * 20, random.Random(0))
    assert any(ch.isdigit() for ch in out)
    assert len(out) == len("idiot " * 20)


def test_spacing_splits_words_into_letters():
    out = obfuscate.space_out(" ".join(["idiot"] * 20), random.Random(0))
    assert "i d i o t" in out


def test_homoglyphs_look_the_same_but_are_not_ascii():
    text = "idiot " * 20
    out = obfuscate.homoglyph(text, random.Random(0))
    assert out != text
    assert not out.isascii()
    back = {v: k for k, v in obfuscate.HOMOGLYPHS.items()}
    assert "".join(back.get(ch, ch) for ch in out) == text


def test_text_no_trick_can_change_is_reported_unchanged():
    assert obfuscate.obfuscate_text("!!", "some_id", 42) == ("!!", "none")
