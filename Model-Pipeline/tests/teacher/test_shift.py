import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS, load_config
from src.teacher.sampling import enriched_sample
from src.teacher.shift import load_civil, shift_rows, summary


def civil_file(tmp_path, n=12):
    raw = pd.DataFrame({"text": [f"comment  {i} &amp; more" for i in range(n)]})
    raw.loc[3, "text"] = None
    for label in LABELS:
        raw[label] = np.float32(0.0)
    raw.loc[1, "toxicity"] = 0.5          # exactly at the threshold: positive
    raw.loc[2, "toxicity"] = 0.4999       # just below: negative
    raw.loc[4, "threat"] = 0.8
    raw["sexual_explicit"] = np.float32(0.0)
    path = tmp_path / "validation.parquet"
    raw.to_parquet(path)
    return path


def test_load_civil_ids_labels_and_cleaning(tmp_path):
    civil = load_civil(civil_file(tmp_path), "validation", 0.5)
    assert list(civil.columns) == ["id", "text", *LABELS]
    assert civil["id"].tolist()[:2] == ["civil-validation-00", "civil-validation-01"]
    assert civil.loc[0, "text"] == "comment 0 & more"      # NFKC, unescape, whitespace
    assert civil.loc[3, "text"] == ""                       # missing text becomes empty
    assert civil["toxicity"].tolist()[:3] == [0, 1, 0]
    assert civil.loc[4, "threat"] == 1 and civil["threat"].sum() == 1


def test_ids_sort_in_file_order(tmp_path):
    civil = load_civil(civil_file(tmp_path, n=120), "validation", 0.5)
    assert civil["id"].tolist() == sorted(civil["id"])    # zero-padded, so sampling is stable


def test_load_civil_requires_label_columns(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"text": ["a"]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing"):
        load_civil(path, "validation", 0.5)


def test_shift_rows_order_and_missing(tmp_path):
    civil = load_civil(civil_file(tmp_path), "validation", 0.5)
    ids = pd.DataFrame({"id": ["civil-validation-04", "civil-validation-01"]})
    assert shift_rows(civil, ids)["id"].tolist() == ids["id"].tolist()
    with pytest.raises(ValueError):
        shift_rows(civil, pd.DataFrame({"id": ["civil-validation-99"]}))


def test_weighted_prevalence_estimates_the_split():
    rng = np.random.default_rng(1)
    n = 20000
    civil = pd.DataFrame({"id": [f"civil-validation-{i:05d}" for i in range(n)],
                          "text": ""})
    for label in LABELS:
        civil[label] = (rng.random(n) < 0.01).astype("int8")
    sample = enriched_sample(civil, 2000, {"threat": 150, "obscene": 150}, seed=42)
    info = summary(civil, sample)
    assert info["labels"]["threat"]["positives"] >= 150
    for label in LABELS:
        v = info["labels"][label]
        assert v["weighted_prevalence"] == pytest.approx(v["true_prevalence"], abs=0.006)


def test_config_has_shift_settings():
    cfg = load_config()
    s = cfg["shift"]
    assert len(s["revision"]) == 40 and s["split"] in s["file"]
    assert s["prompt_version"] == "p2-zs-v1" and cfg["stack"]["version"] == "stack-v1"
    assert sum(s["enrich"].values()) < s["size"]
