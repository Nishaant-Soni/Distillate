"""The §2.4 API: response shape and decisions, input validation, rate limiting, metrics, and
no comment text in logs."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app import RateLimiter, create_app
from scorer import LABELS, Scorer

SECRET = "you are an idiot please see the talk page"


def logs(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def metric(client, line_start: str) -> float:
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start + " "):
            return float(line.split()[-1])
    raise AssertionError(f"{line_start} not in /metrics")


def test_predict_follows_the_contract(client, package):
    r = client.post("/predict", json={"text": SECRET})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"scores", "flagged", "flag", "low_confidence", "model_version",
                         "latency_ms"}
    assert list(body["scores"]) == list(LABELS) and list(body["flagged"]) == list(LABELS)
    assert all(0.0 <= v <= 1.0 for v in body["scores"].values())
    assert body["model_version"] == "student-test-v1"
    assert isinstance(body["latency_ms"], float) and body["latency_ms"] >= 0

    scorer = Scorer(package)
    expected = scorer.scores(SECRET)
    for label in LABELS:
        assert body["scores"][label] == pytest.approx(expected[label], abs=1e-6)
        assert body["flagged"][label] == (expected[label] >= scorer.thresholds[label])
    assert body["flag"] == any(body["flagged"].values())
    assert body["low_confidence"] == any(
        abs(expected[label] - scorer.thresholds[label]) < 0.1 for label in LABELS)


def test_scores_differ_by_text_and_long_text_is_truncated(client):
    a = client.post("/predict", json={"text": "thanks for the edit"}).json()["scores"]
    b = client.post("/predict", json={"text": "you are an idiot"}).json()["scores"]
    assert a != b
    long = client.post("/predict", json={"text": "fact check " * 500})
    assert long.status_code == 200


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "student-test-v1"}


@pytest.mark.parametrize("kwargs, status", [
    ({"content": b'{"text": "hello"}', "headers": {"content-type": "text/plain"}}, 415),
    ({"content": b"not json", "headers": {"content-type": "application/json"}}, 422),
    ({"json": {"txt": "hello"}}, 422),
    ({"json": {"text": 42}}, 422),
    ({"json": ["hello"]}, 422),
    ({"json": {"text": "   "}}, 422),
    ({"content": b'{"text": "\\ud800"}', "headers": {"content-type": "application/json"}}, 422),
    ({"json": {"text": "a" * 10_001}}, 413),
    ({"json": {"text": "a", "pad": "x" * 70_000}}, 413),
])
def test_bad_input_is_rejected_without_counting_as_an_error(client, kwargs, status):
    r = client.post("/predict", **kwargs)
    assert r.status_code == status
    assert "hello" not in r.text and "aaaa" not in r.text      # never echoes the input
    assert metric(client, "distillate_requests_total") == 1
    assert metric(client, "distillate_errors_total") == 0


def test_json_with_charset_is_accepted(client):
    r = client.post("/predict", content=json.dumps({"text": "hello"}).encode(),
                    headers={"content-type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def test_metrics_count_requests_flags_and_scores(client):
    for text in ("thanks for the edit", "you are an idiot", "hello world"):
        client.post("/predict", json={"text": text})
    text = client.get("/metrics").text
    assert metric(client, "distillate_requests_total") == 3
    assert metric(client, "distillate_errors_total") == 0
    assert metric(client, "distillate_request_latency_seconds_count") == 3
    for label in LABELS:
        assert f'distillate_flagged_total{{label="{label}"}}' in text
        assert metric(client, f'distillate_score_count{{label="{label}"}}') == 3
    assert "distillate_flag_total" in text and "distillate_low_confidence_total" in text
    assert 'distillate_request_latency_seconds_bucket{le="0.15"}' in text
    assert "_created" not in text


def test_one_log_line_per_request_and_no_text(config, capsys):
    with TestClient(create_app(config)) as c:
        capsys.readouterr()                                  # drop the startup line
        body = c.post("/predict", json={"text": SECRET}).json()
        c.post("/predict", json={"text": ""})               # rejected: no line
    lines = logs(capsys)
    assert len(lines) == 1
    line = lines[0]
    assert line["text_sha256"] == hashlib.sha256(SECRET.encode()).hexdigest()
    assert line["char_len"] == len(SECRET)
    for key in ("scores", "flagged", "flag", "low_confidence", "model_version", "latency_ms"):
        assert line[key] == body[key]
    assert "idiot" not in json.dumps(lines)


class Broken:
    model_version = "broken"
    max_length = 16
    thresholds = {label: 0.5 for label in LABELS}

    def __init__(self):
        self.calls = 0

    def scores(self, text):
        self.calls += 1
        if self.calls > 1:                                  # the warm-up call succeeds
            raise RuntimeError(f"could not score {text!r}")
        return {label: 0.1 for label in LABELS}


def test_a_failure_is_a_500_and_an_error_without_text_in_the_log(config, capsys):
    with TestClient(create_app(config, scorer=Broken())) as c:
        capsys.readouterr()
        r = c.post("/predict", json={"text": SECRET})
        assert r.status_code == 500 and "idiot" not in r.text
        assert metric(c, "distillate_errors_total") == 1
        assert metric(c, "distillate_requests_total") == 1
    out = capsys.readouterr().out
    assert "idiot" not in out
    assert json.loads(out.strip())["error"] == "RuntimeError"


def test_rate_limit_answers_429_per_client(config):
    config["rate_limit"] = {"requests_per_minute": 60, "burst": 2}
    with TestClient(create_app(config)) as c:
        a = {"x-forwarded-for": "spoofed, 10.0.0.1"}
        codes = [c.post("/predict", json={"text": "hello"}, headers=a).status_code
                 for _ in range(3)]
        assert codes == [200, 200, 429]
        other = c.post("/predict", json={"text": "hello"}, headers={"x-forwarded-for": "10.0.0.2"})
        assert other.status_code == 200
        assert metric(c, "distillate_errors_total") == 0


def test_rate_limiter_refills_and_can_be_off():
    now = [0.0]
    limiter = RateLimiter(per_minute=60, burst=1, clock=lambda: now[0])
    assert limiter.allow("a") and not limiter.allow("a")
    now[0] += 1.0                                            # one token a second
    assert limiter.allow("a")
    off = RateLimiter(per_minute=0, burst=1)
    assert all(off.allow("a") for _ in range(100))


def test_environment_overrides(monkeypatch, package):
    from app import load_config

    monkeypatch.setenv("DISTILLATE_PACKAGE_DIR", str(package))
    monkeypatch.setenv("DISTILLATE_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("DISTILLATE_ORT_THREADS", "4")
    cfg = load_config()
    assert cfg["package_dir"] == str(package)
    assert cfg["rate_limit"]["requests_per_minute"] == 0
    assert cfg["onnx"]["intra_op_threads"] == 4
