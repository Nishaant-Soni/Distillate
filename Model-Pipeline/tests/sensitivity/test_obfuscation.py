import pytest

from src.sensitivity.obfuscation import changed_share, substitute, substitution_table
from src.teacher.config import PIPELINE_ROOT

DATA_PIPELINE = PIPELINE_ROOT.parent / "Data-Pipeline"
TABLE = {"a": "4", "e": "3", "i": "1", "l": "1", "o": "0", "s": "5", "t": "7"}
TEXT = "This is a Test of the obfuscation sweep, Idiot."


def test_the_table_is_the_data_pipelines():
    assert substitution_table(DATA_PIPELINE) == TABLE


def test_missing_data_pipeline(tmp_path):
    with pytest.raises(FileNotFoundError, match="obfuscate.py"):
        substitution_table(tmp_path)


def test_rate_zero_and_one():
    assert substitute(TEXT, 0.0, "id1", 42, TABLE) == TEXT
    full = substitute(TEXT, 1.0, "id1", 42, TABLE)
    assert full == "7h15 15 4 7357 0f 7h3 0bfu5c4710n 5w33p, 1d107."
    assert len(full) == len(TEXT)


def test_rates_are_nested_and_seeded_per_id():
    def changed(rate, key="id1"):
        out = substitute(TEXT, rate, key, 42, TABLE)
        return {i for i, (a, b) in enumerate(zip(TEXT, out, strict=True)) if a != b}

    rates = [0.1, 0.25, 0.5, 0.75, 1.0]
    sets = [changed(rate) for rate in rates]
    assert all(small <= large for small, large in zip(sets, sets[1:], strict=False))
    assert len(sets[0]) < len(sets[-1])
    assert changed(0.5) == changed(0.5)                       # same id, same result
    assert changed(0.5, "id1") != changed(0.5, "id2")          # another id, other draws


def test_bad_rate():
    with pytest.raises(ValueError, match="rate"):
        substitute(TEXT, 1.5, "id1", 42, TABLE)


def test_changed_share():
    assert changed_share("", "") == 0.0
    assert changed_share("idiot", "1d10t") == pytest.approx(0.6)
