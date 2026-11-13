"""Building the evaluation tables from competition-format files, by the id lists."""

import numpy as np
import pandas as pd
import pytest

from src.student.eval_sets import IDENTITY_COLUMNS, build_eval_tables, civil_frame, write_gold
from src.teacher.config import LABELS
from src.teacher.data import KAGGLE_LABELS

WORDS = ["you", "are", "an", "idiot", "please", "stop", "editing", "this", "article", "talk"]


def _texts(rng, n):
    return [" ".join(rng.choice(WORDS, size=8)) + " &amp; more" for _ in range(n)]


@pytest.fixture
def competitions(tmp_path):
    rng = np.random.default_rng(0)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    ids = [f"{i:016x}" for i in range(60)]
    pd.DataFrame({"id": ids, "comment_text": _texts(rng, 60)}).to_csv(wiki / "test.csv",
                                                                      index=False)
    labels = pd.DataFrame({"id": ids, **{k: rng.integers(0, 2, 60) for k in KAGGLE_LABELS}})
    labels.loc[:9, list(KAGGLE_LABELS)] = -1                  # never scored
    labels.to_csv(wiki / "test_labels.csv", index=False)
    civil_ids = [str(7_000_000 + i) for i in range(80)]
    civil = pd.DataFrame({"id": civil_ids, "comment_text": _texts(rng, 80),
                          **{k: rng.random(80).round(3) for k in LABELS},
                          **{c: rng.choice([0.0, 0.7], 80) for c in IDENTITY_COLUMNS}})
    civil.loc[0, "toxicity"] = 0.5                          # exactly at the cut: positive
    civil.to_csv(tmp_path / "test_private_expanded.csv", index=False)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "gate_ids.txt").write_text("\n".join(reversed(ids[10:])) + "\n")
    (splits / "h2_eval_ids.txt").write_text("\n".join(civil_ids[:30]) + "\n")
    (splits / "bias_eval_ids.txt").write_text("\n".join(civil_ids[30:]) + "\n")
    return tmp_path, wiki, splits


def test_tables_follow_the_id_lists_and_the_contract(competitions):
    root, wiki, splits = competitions
    out = root / "processed"
    info = build_eval_tables(wiki, root / "test_private_expanded.csv", out, splits)
    gate = pd.read_parquet(out / "wikipedia" / "gate.parquet")
    assert gate["id"].tolist() == (splits / "gate_ids.txt").read_text().split()
    assert "&amp;" not in gate["text"].iloc[0] and gate["toxicity"].dtype == "int8"
    bias = pd.read_parquet(out / "civil" / "bias_eval.parquet")
    h2 = pd.read_parquet(out / "civil" / "h2_eval.parquet")
    assert len(bias) == 50 and len(h2) == 30 and set(IDENTITY_COLUMNS) <= set(bias.columns)
    assert not set(IDENTITY_COLUMNS) & set(h2.columns)
    obf = pd.read_parquet(out / "obfuscation.parquet")
    assert len(obf) == info["rows"]["obfuscation.parquet"] > 0
    assert obf["id"].str.endswith("_obf").all() and (obf["toxicity"] == 1).all()
    originals = gate.set_index("id").loc[obf["id"].str[:-4], "text"].tolist()
    assert all(a != b for a, b in zip(originals, obf["text"], strict=True))


def test_civil_labels_cut_at_half_and_keep_the_fraction(competitions):
    root, _, _ = competitions
    civil = civil_frame(root / "test_private_expanded.csv")
    assert civil.loc[0, "toxicity"] == 1 and civil.loc[0, "toxicity_frac"] == pytest.approx(0.5)
    assert ((civil["insult_frac"] >= 0.5).astype("int8") == civil["insult"]).all()


def test_missing_ids_fail(competitions):
    root, wiki, splits = competitions
    (splits / "h2_eval_ids.txt").write_text("999\n")
    with pytest.raises(ValueError, match="h2_eval: 1 ids"):
        build_eval_tables(wiki, root / "test_private_expanded.csv", root / "out", splits)


def test_gold_files_have_no_text(competitions, tmp_path):
    root, _, _ = competitions
    civil = civil_frame(root / "test_private_expanded.csv")
    write_gold({"bias_eval": civil}, tmp_path / "gold")
    gold = pd.read_parquet(tmp_path / "gold" / "bias_eval.parquet")
    assert "text" not in gold.columns and "toxicity_frac" not in gold.columns
    assert list(gold.columns[:7]) == ["id", *LABELS]
