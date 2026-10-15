import json

import pytest

from scripts import utils


def test_load_config_resolves_paths_against_pipeline_root():
    cfg = utils.load_config()
    assert cfg["seed"] == 42
    assert cfg["paths"]["raw_dir"].startswith(str(utils.PIPELINE_ROOT))
    assert cfg["labels"]["names"] == [
        "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack",
    ]


def test_load_config_dvc_push_env_override(monkeypatch):
    monkeypatch.setenv("DVC_PUSH", "false")
    assert utils.load_config()["dvc"]["push"] is False
    monkeypatch.setenv("DVC_PUSH", "true")
    assert utils.load_config()["dvc"]["push"] is True


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_config(tmp_path / "nope.yaml")


def test_atomic_write_text_and_bytes(tmp_path):
    target = tmp_path / "sub" / "out.txt"
    utils.atomic_write(target, "hello")
    assert target.read_text() == "hello"
    utils.atomic_write(target, b"bytes")
    assert target.read_bytes() == b"bytes"


def test_atomic_path_keeps_old_file_on_crash(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    with pytest.raises(RuntimeError):
        with utils.atomic_path(target) as tmp:
            tmp.write_text("half-written")
            raise RuntimeError("boom")
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]  # no leftover temp file


def test_write_summary_one_file_per_task(tmp_path):
    cfg = {"paths": {"summaries_dir": str(tmp_path)}}
    utils.write_summary("task_a", {"rows_in": 10}, cfg)
    utils.write_summary("task_b", {"rows_in": 20}, cfg)
    assert json.loads((tmp_path / "task_a.json").read_text()) == {"rows_in": 10}
    assert json.loads((tmp_path / "task_b.json").read_text()) == {"rows_in": 20}


def test_timed_records_duration():
    summary = {}
    with utils.timed(summary):
        pass
    assert summary["duration_s"] >= 0
