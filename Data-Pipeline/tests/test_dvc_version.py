"""The DVC task: tracks data, pushes only when asked, never blocks a fresh clone."""

import subprocess
from pathlib import Path

import pytest

from scripts import dvc_version


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_version, "_repo_root", lambda: tmp_path)
    for name in ("raw", "processed"):
        (tmp_path / "data" / name).mkdir(parents=True)
    (tmp_path / "reports" / "schema" / "stats").mkdir(parents=True)
    return {
        "paths": {"raw_dir": str(tmp_path / "data" / "raw"),
                  "processed_dir": str(tmp_path / "data" / "processed"),
                  "reports_dir": str(tmp_path / "reports"),
                  "summaries_dir": str(tmp_path / "summaries")},
        "dvc": {"push": False},
    }


def _fake_dvc(calls, returncode=0, stderr=""):
    def run(args, cwd):
        calls.append(args)
        return subprocess.CompletedProcess(args, returncode, stdout="", stderr=stderr)
    return run


def test_uninitialised_repo_skips_instead_of_failing(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(dvc_version, "_dvc", _fake_dvc(calls))
    summary = dvc_version.run(cfg)
    assert "skipped" in summary and calls == []


def test_tracks_the_data_and_the_tfdv_statistics(cfg, monkeypatch, tmp_path):
    (tmp_path / ".dvc").mkdir()
    calls = []
    monkeypatch.setattr(dvc_version, "_dvc", _fake_dvc(calls))
    summary = dvc_version.run(cfg)
    assert calls == [["add", "data/raw"], ["add", "data/processed"],
                     ["add", "reports/schema/stats"]]
    assert summary["tracked"] == ["data/raw", "data/processed", "reports/schema/stats"]
    assert summary["commit_reminder"] == ["data/raw.dvc", "data/processed.dvc",
                                         "reports/schema/stats.dvc"]


def test_push_is_skipped_by_default(cfg, monkeypatch, tmp_path):
    (tmp_path / ".dvc").mkdir()
    calls = []
    monkeypatch.setattr(dvc_version, "_dvc", _fake_dvc(calls))
    summary = dvc_version.run(cfg)
    assert summary["pushed"] is False
    assert ["push"] not in calls


def test_push_runs_when_enabled(cfg, monkeypatch, tmp_path):
    (tmp_path / ".dvc").mkdir()
    cfg["dvc"]["push"] = True
    calls = []
    monkeypatch.setattr(dvc_version, "_dvc", _fake_dvc(calls))
    assert dvc_version.run(cfg)["pushed"] is True
    assert ["push"] in calls


def test_a_failed_push_warns_but_keeps_the_run_green(cfg, monkeypatch, tmp_path):
    (tmp_path / ".dvc").mkdir()
    cfg["dvc"]["push"] = True
    alerts = []
    monkeypatch.setattr(dvc_version, "send_alert",
                        lambda level, title, details: alerts.append(level))

    def add_ok_push_fails(args, cwd):
        code = 1 if args[0] == "push" else 0
        return subprocess.CompletedProcess(args, code, stdout="", stderr="no credentials")

    monkeypatch.setattr(dvc_version, "_dvc", add_ok_push_fails)
    summary = dvc_version.run(cfg)
    assert summary["pushed"] is False
    # tracking still happened
    assert summary["tracked"] == ["data/raw", "data/processed", "reports/schema/stats"]
    assert alerts == ["warn"]


def test_a_failed_add_fails_the_task(cfg, monkeypatch, tmp_path):
    (tmp_path / ".dvc").mkdir()
    monkeypatch.setattr(dvc_version, "send_alert", lambda *a, **k: None)

    def failing_add(args, cwd):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="dvc add blew up")

    monkeypatch.setattr(dvc_version, "_dvc", failing_add)
    with pytest.raises(RuntimeError, match="dvc add"):
        dvc_version.run(cfg)


def test_missing_data_directory_is_skipped_quietly(cfg, monkeypatch, tmp_path):
    (tmp_path / ".dvc").mkdir()
    Path(cfg["paths"]["processed_dir"]).rmdir()
    calls = []
    monkeypatch.setattr(dvc_version, "_dvc", _fake_dvc(calls))
    assert dvc_version.run(cfg)["tracked"] == ["data/raw", "reports/schema/stats"]


def test_missing_statistics_are_skipped_quietly(cfg, monkeypatch, tmp_path):
    """While schema_stats is a stub there are no statistics; data is still versioned."""
    (tmp_path / ".dvc").mkdir()
    (tmp_path / "reports" / "schema" / "stats").rmdir()
    calls = []
    monkeypatch.setattr(dvc_version, "_dvc", _fake_dvc(calls))
    assert dvc_version.run(cfg)["tracked"] == ["data/raw", "data/processed"]
