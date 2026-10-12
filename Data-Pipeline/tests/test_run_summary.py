"""The final task: merge per-task summaries and post one Slack message."""

import json
from pathlib import Path

import pytest

from scripts import run_summary


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"summaries_dir": str(tmp_path / "summaries"),
                      "reports_dir": str(tmp_path / "reports")}}


def _summary(cfg, task, payload):
    d = Path(cfg["paths"]["summaries_dir"])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task}.json").write_text(json.dumps(payload))


def test_empty_run_still_produces_a_report(cfg, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    Path(cfg["paths"]["summaries_dir"]).mkdir(parents=True)
    report = run_summary.run(cfg)
    assert report["tasks_reported"] == 0
    assert (Path(cfg["paths"]["reports_dir"]) / "run_summary.json").exists()


def test_totals_are_summed_across_tasks(cfg, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    _summary(cfg, "dedupe", {"mode": "real", "rows_out": 100, "duration_s": 1.5})
    _summary(cfg, "split", {"mode": "stub", "rows_out": 50, "duration_s": 0.5})
    report = run_summary.run(cfg)
    assert report["rows_out"] == 150
    assert report["duration_s"] == 2.0
    assert report["stages_real"] == ["dedupe"]
    assert report["stages_stubbed"] == ["split"]


def test_a_corrupt_summary_does_not_lose_the_rest(cfg, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    _summary(cfg, "good", {"mode": "real", "rows_out": 7})
    Path(cfg["paths"]["summaries_dir"], "bad.json").write_text("{not json")
    report = run_summary.run(cfg)
    assert report["rows_out"] == 7
    assert report["unreadable_summaries"] == ["bad.json"]


def test_missing_duration_or_rows_is_tolerated(cfg, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    _summary(cfg, "sparse", {"mode": "real"})
    _summary(cfg, "nulls", {"mode": "real", "rows_out": None, "duration_s": None})
    assert run_summary.run(cfg)["rows_out"] == 0


def test_a_stubbed_run_is_announced_as_a_warning(cfg, monkeypatch):
    sent = {}
    monkeypatch.setattr(run_summary, "send_alert",
                        lambda level, title, details: sent.update(level=level, title=title))
    _summary(cfg, "split", {"mode": "stub", "rows_out": 1})
    run_summary.run(cfg)
    assert sent["level"] == "warn"
    assert "stubbed" in sent["title"]


def test_a_fully_real_run_is_announced_as_success(cfg, monkeypatch):
    sent = {}
    monkeypatch.setattr(run_summary, "send_alert",
                        lambda level, title, details: sent.update(level=level, title=title))
    _summary(cfg, "split", {"mode": "real", "rows_out": 1})
    run_summary.run(cfg)
    assert sent["level"] == "ok"
    assert "stubbed" not in sent["title"]
