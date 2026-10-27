"""Tests for scripts/schema_stats.py. TFDV only exists in /opt/tfenv, so these skip elsewhere.

Run inside the image, from Data-Pipeline:
    docker compose exec airflow-scheduler bash -c \
      "cd /opt/airflow/repo/Data-Pipeline &&
       /opt/tfenv/bin/python -m pytest tests/test_schema_stats.py"
"""

from pathlib import Path

import pytest

pytest.importorskip("tensorflow_data_validation")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from scripts import schema_stats  # noqa: E402
from scripts.stubs import IDENTITY_COLUMNS, LABELS, civil_frame, wikipedia_frame  # noqa: E402
from scripts.utils import load_config  # noqa: E402

N = 200
SAMPLE = 120  # smaller than N, so the sampled Civil train files are really sampled


def _with_derived(df):
    """civil_frame lacks the derived columns every processed file carries."""
    df["char_len"] = df["text"].str.len().astype("int32")
    df["has_identity_term"] = False
    df["has_obfuscation"] = False
    return df


def _civil(prefix, identities=False):
    df = _with_derived(civil_frame(N, 42, prefix, with_identities=identities))
    if identities:
        df["identity_annotator_count"] = np.float32(4.0)
    return df


def _raw_wiki(prefix):
    df = wikipedia_frame(N, 42, prefix)[["id", "text", *LABELS]]
    return df.astype({label: "float32" for label in LABELS})


def _raw_civil(prefix, annotated_share=1.0):
    """Raw Civil: float rater fractions, string dates, identity scores often missing."""
    df = civil_frame(N, 42, prefix, with_identities=True)
    for label in LABELS:
        df[label] = df.pop(f"{label}_frac")
    df["created_date"] = df["created_date"].dt.strftime("%Y-%m-%d %H:%M:%S.%f+00")
    df["sexual_explicit"] = np.float32(0.0)
    annotated = np.arange(N) < int(N * annotated_share)
    df["identity_annotator_count"] = np.where(annotated, 4.0, 0.0).astype("float32")
    for col in IDENTITY_COLUMNS:
        df.loc[~annotated, col] = np.nan
    return df


def _write(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def cfg(tmp_path):
    """Every file schema_stats validates, contract-shaped, plus inferred schemas."""
    cfg = load_config()
    for key in ("raw_dir", "processed_dir", "cache_dir", "schema_dir", "reports_dir",
                "summaries_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["splits"]["civil"]["drift_batches"] = 2
    cfg["validation"]["raw_civil_stats_sample"] = SAMPLE

    raw, proc, cache = (Path(cfg["paths"][k]) for k in ("raw_dir", "processed_dir", "cache_dir"))
    _write(raw / "wikipedia/train.parquet", _raw_wiki("wiki_train"))
    _write(raw / "wikipedia/test.parquet", _raw_wiki("wiki_test"))
    _write(cache / "civil/train.parquet", _raw_civil("civil_train", annotated_share=0.2))
    _write(raw / "civil/validation.parquet", _raw_civil("civil_val", annotated_share=0.2))
    _write(raw / "civil/test.parquet", _raw_civil("civil_test", annotated_share=0.2))
    _write(raw / "civil/identities_test.parquet", _raw_civil("civil_ident"))

    for name in ("train_clean", "test_clean", "train_dedup", "val", "gate", "canary_wikipedia"):
        _write(proc / f"wikipedia/{name}.parquet", wikipedia_frame(N, 42, f"wiki_{name}"))
    _write(proc / "wikipedia/train.parquet", wikipedia_frame(N, 42, "wiki_tr", with_weight=True))
    _write(proc / "obfuscation.parquet", wikipedia_frame(N, 42, "wiki_obf"))
    _write(cache / "civil/train_clean.parquet", _civil("civil_train"))
    for name in ("validation_clean", "test_clean", "h2_eval", "canary_civil"):
        _write(proc / f"civil/{name}.parquet", _civil(f"civil_{name}"))
    for b in range(2):
        _write(proc / f"civil/drift/batch_{b:02d}.parquet", _civil(f"civil_drift_{b}"))
    for name in ("identities_test_clean", "bias_eval"):
        _write(proc / f"civil/{name}.parquet", _civil(f"civil_{name}", identities=True))

    schema_stats.infer(cfg)
    return cfg


def _proc(cfg, rel):
    return Path(cfg["paths"]["processed_dir"]) / rel


def _anomalies(cfg, target):
    [result] = schema_stats.validate(cfg, only=[target])["results"]
    return result["anomalies"]


def test_infer_writes_both_schemas(cfg):
    schema_dir = Path(cfg["paths"]["schema_dir"])
    assert (schema_dir / "schema.pbtxt").exists()
    assert (schema_dir / "raw_schema.pbtxt").exists()


def test_every_clean_file_matches_its_schema(cfg):
    summary = schema_stats.run(cfg)
    assert summary["files_with_anomalies"] == []
    assert summary["files_checked"] == len(schema_stats.targets(cfg))
    assert (Path(cfg["paths"]["reports_dir"]) / "schema/validation.json").exists()


def test_civil_train_files_are_validated_on_a_sample(cfg):
    summary = schema_stats.run(cfg)
    assert summary["rows_checked"]["cache/civil/train"] == SAMPLE
    assert summary["rows_checked"]["cache/civil/train_clean"] == SAMPLE
    assert summary["rows_checked"]["processed/civil/h2_eval"] == N


def test_statistics_are_saved_for_every_file(cfg):
    """One loadable text proto per target, so two runs can be compared later."""
    import tensorflow_data_validation as tfdv

    summary = schema_stats.run(cfg)
    stats_dir = Path(cfg["paths"]["reports_dir"]) / "schema" / "stats"
    for target in schema_stats.targets(cfg):
        path = stats_dir / f"{target['name']}.pbtxt"
        assert path.exists(), target["name"]
        stats = tfdv.load_stats_text(str(path))
        assert stats.datasets[0].num_examples == summary["rows_checked"][target["name"]]


def test_saved_statistics_contain_no_comment_text(cfg):
    """Kaggle rules: comment text never goes into the repository (work plan v3, 2.2)."""
    schema_stats.run(cfg)
    comments = set(pd.read_parquet(_proc(cfg, "wikipedia/val.parquet"))["text"])
    comments |= set(pd.read_parquet(_proc(cfg, "civil/h2_eval.parquet"))["text"])
    stats_dir = Path(cfg["paths"]["reports_dir"]) / "schema" / "stats"
    for path in stats_dir.rglob("*.pbtxt"):
        content = path.read_text(encoding="utf-8")
        leaked = [c for c in comments if c in content]
        assert not leaked, f"{path.name} contains comment text: {leaked[:2]}"


def test_raw_identity_scores_may_be_missing(cfg):
    """Most raw Civil rows were never annotated; that alone is not an anomaly."""
    assert _anomalies(cfg, "raw/civil/test") == {}


def test_label_outside_zero_one_is_an_anomaly(cfg):
    df = pd.read_parquet(_proc(cfg, "wikipedia/val.parquet"))
    df.loc[0, "insult"] = 2
    _write(_proc(cfg, "wikipedia/val.parquet"), df)
    assert "insult" in _anomalies(cfg, "processed/wikipedia/val")


def test_missing_text_is_an_anomaly(cfg):
    df = pd.read_parquet(_proc(cfg, "wikipedia/gate.parquet"))
    df.loc[0, "text"] = None
    _write(_proc(cfg, "wikipedia/gate.parquet"), df)
    assert "text" in _anomalies(cfg, "processed/wikipedia/gate")


def test_missing_required_civil_column_is_an_anomaly(cfg):
    df = pd.read_parquet(_proc(cfg, "civil/h2_eval.parquet")).drop(columns="created_date")
    _write(_proc(cfg, "civil/h2_eval.parquet"), df)
    assert "created_date" in _anomalies(cfg, "processed/civil/h2_eval")


def test_unexpected_column_is_an_anomaly(cfg):
    df = pd.read_parquet(_proc(cfg, "wikipedia/canary_wikipedia.parquet"))
    df["surprise"] = 1
    _write(_proc(cfg, "wikipedia/canary_wikipedia.parquet"), df)
    assert "surprise" in _anomalies(cfg, "processed/wikipedia/canary_wikipedia")


def test_any_anomaly_fails_the_stage_after_writing_the_report(cfg):
    df = pd.read_parquet(_proc(cfg, "wikipedia/val.parquet"))
    df.loc[0, "toxicity"] = 7
    _write(_proc(cfg, "wikipedia/val.parquet"), df)
    with pytest.raises(schema_stats.SchemaAnomalyError, match="processed/wikipedia/val"):
        schema_stats.run(cfg)
    assert (Path(cfg["paths"]["reports_dir"]) / "schema/validation.json").exists()


def test_missing_file_is_reported_not_skipped(cfg):
    _proc(cfg, "obfuscation.parquet").unlink()
    assert "(file)" in _anomalies(cfg, "processed/obfuscation")


def test_missing_schema_gives_a_clear_error(cfg):
    (Path(cfg["paths"]["schema_dir"]) / "schema.pbtxt").unlink()
    with pytest.raises(FileNotFoundError, match="--infer"):
        schema_stats.run(cfg)
