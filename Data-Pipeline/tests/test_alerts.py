from types import SimpleNamespace

import pytest

from scripts import alerts


def test_no_webhook_only_logs(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert alerts.send_alert("warn", "test", {"k": "v"}) is False


def test_invalid_level_rejected():
    with pytest.raises(ValueError):
        alerts.send_alert("info", "bad level")


def test_webhook_posts_payload(monkeypatch):
    sent = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        sent["body"] = request.data.decode()
        return FakeResponse()

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    assert alerts.send_alert("fail", "Task failed", {"task": "dedupe"}) is True
    assert "Task failed" in sent["body"] and "dedupe" in sent["body"]


def test_broken_webhook_never_raises(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    monkeypatch.setattr(alerts.urllib.request, "urlopen", broken)
    assert alerts.send_alert("fail", "x") is False


def test_failure_callback_reads_task_instance(monkeypatch):
    captured = {}
    monkeypatch.setattr(alerts, "send_alert",
                        lambda level, title, details: captured.update(details))
    ti = SimpleNamespace(dag_id="data_prep_dag", task_id="dedupe_wikipedia", try_number=2, log_url="http://log")
    alerts.task_failure_callback({"task_instance": ti, "exception": ValueError("bad rows")})
    assert captured["dag"] == "data_prep_dag"
    assert captured["task"] == "dedupe_wikipedia"
    assert "bad rows" in captured["error"]
