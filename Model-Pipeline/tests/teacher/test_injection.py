import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS, PIPELINE_ROOT
from src.teacher.injection import flip_rates, load_injection_set, texts

SET = PIPELINE_ROOT / "configs" / "injection_set.csv"


def test_committed_set_is_balanced():
    df = load_injection_set(SET)
    assert len(df) == 40
    assert (df["intended"] == "toxic").sum() == 20 and (df["intended"] == "clean").sum() == 20
    assert set(df.loc[df["intended"] == "toxic", "direction"]) == {"to_no"}
    assert set(df.loc[df["intended"] == "clean", "direction"]) == {"to_yes"}
    assert set(df["position"]) == {"prefix", "suffix"}
    assert df["injection"].str.contains("</comment>").any()    # fence-closing attempts


def test_texts_place_the_injection():
    df = pd.DataFrame({"id": ["a", "b"], "intended": ["toxic", "clean"],
                       "direction": ["to_no", "to_yes"], "position": ["suffix", "prefix"],
                       "text": ["bad  words", "nice &amp; kind"],
                       "injection": ["say no", "say yes"]})
    base, injected = texts(df)
    assert base == ["bad words", "nice & kind"]                 # cleaned once, like the data
    assert injected == ["bad words say no", "say yes nice & kind"]


def frame(p_rows, status=None):
    out = pd.DataFrame({"id": [str(i) for i in range(len(p_rows))]})
    for j, label in enumerate(LABELS):
        out[f"p_{label}"] = np.array([row[j] for row in p_rows], dtype="float32")
    out["status"] = status or ["ok"] * len(p_rows)
    return out


def test_flip_rates():
    df = pd.DataFrame({"direction": ["to_no", "to_no", "to_yes"]})
    #               toxicity flips on row 0 only; threat was already "no" (not eligible)
    base = frame([[0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                  [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    injected = frame([[0.2, 0.1, 0.9, 0.1, 0.9, 0.1],
                      [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                      [0.1, 0.1, 0.1, 0.8, 0.1, 0.1]])
    table = flip_rates(df, base, injected).set_index(["direction", "label"])
    assert table.loc[("to_no", "toxicity"), "eligible"] == 2
    assert table.loc[("to_no", "toxicity"), "flip_rate"] == pytest.approx(0.5)
    assert table.loc[("to_no", "threat"), "eligible"] == 0
    assert np.isnan(table.loc[("to_no", "threat"), "flip_rate"])
    assert table.loc[("to_no", "all"), "eligible"] == 6          # toxicity, obscene, insult x 2
    assert table.loc[("to_no", "all"), "flipped"] == 1
    assert table.loc[("to_yes", "threat"), "flip_rate"] == pytest.approx(1.0)
    assert table.loc[("to_yes", "all"), "flip_rate"] == pytest.approx(1 / 6)


def test_failed_rows_left_out():
    df = pd.DataFrame({"direction": ["to_no", "to_no"]})
    base = frame([[0.9] * 6, [0.9] * 6])
    injected = frame([[0.1] * 6, [np.nan] * 6], status=["ok", "failed"])
    table = flip_rates(df, base, injected).set_index(["direction", "label"])
    assert table.loc[("to_no", "toxicity"), "comments"] == 1
    assert table.loc[("to_no", "all"), "failed_comments"] == 1
