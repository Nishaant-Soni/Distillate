import copy
import json

import pytest

pytest.importorskip("matplotlib")

from src.sensitivity.run import DEFAULT_CONFIG, headline, run  # noqa: E402
from src.teacher.config import PIPELINE_ROOT, load_config  # noqa: E402
from tests.sensitivity.test_sweeps import validation  # noqa: E402

SECRET = "zzsecretname"


def config(tmp_path, student_dir, mlflow=False, study=None):
    """configs/sensitivity.yaml with every input and output under ``tmp_path``."""
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    frame = validation(n=60)
    frame.loc[0, "text"] = f"{SECRET} {frame.loc[0, 'text']}"
    frame.to_parquet(tmp_path / "val.parquet", index=False)
    cfg["model"]["dir"] = str(student_dir)
    cfg["mlflow"]["enabled"] = mlflow
    cfg["truncation"]["max_lengths"] = [8, 16, 64]
    cfg["truncation"]["band_edges"] = [8, 16]
    cfg["obfuscation"]["rates"] = [0.0, 0.5, 1.0]
    cfg["shap"] |= {"per_group": 2, "max_evals": 60, "min_doc_freq": 2, "min_occurrences": 1}
    if study:
        cfg["optuna"] |= {"storage": study, "study_name": "sweep"}
    cfg["paths"] |= {
        "val": str(tmp_path / "val.parquet"),
        "data_pipeline": str(PIPELINE_ROOT.parent / "Data-Pipeline"),
        "report": str(tmp_path / "reports" / "sensitivity.md"),
        "report_dir": str(tmp_path / "reports" / "sensitivity"),
        "findings": str(tmp_path / "reports" / "sensitivity_findings.md"),
        "summary": str(tmp_path / "reports" / "summaries" / "sensitivity.json"),
        "out_dir": str(tmp_path / "outputs"),
        "mlflow_db": str(tmp_path / "mlruns" / "mlflow.db"),
        "mlflow_artifacts": str(tmp_path / "mlruns" / "artifacts"),
    }
    return cfg, frame


def committed_outputs(tmp_path):
    """Everything the run writes under reports/: what would be committed."""
    return {p: p.read_bytes() for p in (tmp_path / "reports").rglob("*") if p.is_file()}


def test_run_writes_the_report_without_text_or_ids(tmp_path, student_dir):
    cfg, frame = config(tmp_path, student_dir)
    summary = run(cfg, code_commit="abc123")
    report = (tmp_path / "reports" / "sensitivity.md").read_text()
    for heading in ("## Setup", "## 1. Hyperparameter importance", "## 2. Token attributions",
                    "## 3. Input length", "## 4. Obfuscation strength"):
        assert heading in report
    assert "Not run: optuna.storage and optuna.study_name are not set" in report
    assert "`abc123`" in report and "tiny-student-test" in report
    for name in ("truncation", "length_bands", "obfuscation", "words_caught"):
        assert (tmp_path / "reports" / "sensitivity" / f"{name}.png").exists()
        assert f"(sensitivity/{name}.png)" in report

    # No comment text, no ids, in anything under reports/.
    outputs = committed_outputs(tmp_path)
    for content in outputs.values():
        assert SECRET.encode() not in content
        assert not any(i.encode() in content for i in frame["id"])
    longest = max(frame["text"], key=len)
    assert longest.encode() not in b"".join(outputs.values())
    # The ids stay in the gitignored scratch folder.
    assert (tmp_path / "outputs" / "shap_selection.parquet").exists()

    saved = json.loads((tmp_path / "reports" / "summaries" / "sensitivity.json").read_text())
    assert saved == summary and summary["rows_in"] == 60
    assert set(summary["headline"]) >= {"truncation_64_mean_auc", "obfuscation_0.5_missed"}
    assert "mlflow_run_id" not in summary


def test_findings_are_included(tmp_path, student_dir):
    cfg, _ = config(tmp_path, student_dir)
    findings = tmp_path / "reports" / "sensitivity_findings.md"
    findings.parent.mkdir(parents=True)
    findings.write_text("Truncation at 64 costs little.\n")
    run(cfg)
    report = (tmp_path / "reports" / "sensitivity.md").read_text()
    assert "## Findings\n\nTruncation at 64 costs little." in report


def test_sample_and_missing_model(tmp_path, student_dir):
    cfg, _ = config(tmp_path, student_dir)
    cfg["data"]["sample"] = 25
    assert run(cfg)["rows_in"] == 25
    cfg["model"]["dir"] = None
    with pytest.raises(ValueError, match="model.dir"):
        run(cfg)


def test_with_the_study_and_mlflow(tmp_path, student_dir):
    pytest.importorskip("mlflow")
    optuna = pytest.importorskip("optuna")
    from tests.sensitivity.test_hyperparameters import make_study

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    make_study(storage=storage, name="sweep")
    cfg, _ = config(tmp_path, student_dir, mlflow=True, study=storage)
    summary = run(cfg)
    report = (tmp_path / "reports" / "sensitivity.md").read_text()
    assert "Study `sweep`: 40 completed trials" in report and "| alpha |" in report
    assert (tmp_path / "reports" / "sensitivity" / "importance.png").exists()

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    logged = mlflow.get_run(summary["mlflow_run_id"])
    assert logged.data.tags["model_version"] == "tiny-student-test"
    assert logged.data.metrics["truncation_64_mean_auc"] == summary["headline"][
        "truncation_64_mean_auc"]
    names = {a.path for a in mlflow.MlflowClient().list_artifacts(logged.info.run_id,
                                                                   "sensitivity")}
    assert "sensitivity/sensitivity.md" in names
    assert "sensitivity/shap_selection.parquet" not in names


def test_headline_leaves_out_nan():
    import pandas as pd

    overall = pd.DataFrame({"label": ["mean"], "max_length": [64], "auc": [float("nan")]})
    obf = pd.DataFrame({"label": ["mean"], "rate": [0.5], "auc": [0.9], "missed": [0.1]})
    assert headline(overall, obf) == {"obfuscation_0.5_mean_auc": 0.9,
                                      "obfuscation_0.5_missed": 0.1}
