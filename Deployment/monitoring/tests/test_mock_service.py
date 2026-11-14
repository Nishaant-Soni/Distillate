"""Tests for mock_service.py: it must follow the serving API (work plan v4, section 2.4)."""

import json
import urllib.error
import urllib.request

import pytest

import mock_service as mock

SECTION_2_4_METRICS = {
    "distillate_request_latency_seconds", "distillate_requests_total",
    "distillate_errors_total", "distillate_flagged_total", "distillate_flag_total",
    "distillate_score", "distillate_low_confidence_total",
}


@pytest.fixture
def server(tmp_path):
    (tmp_path / "canary.prom").write_text("distillate_canary_mean_auc{set=\"civil\"} 0.83\n")
    httpd = mock.serve(0, textfile_dir=tmp_path)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body):
    request = urllib.request.Request(url + "/predict", data=json.dumps(body).encode(),
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as response:
        return response.read().decode()


def _sample(metrics_text, name):
    """The value of an unlabelled sample line."""
    for line in metrics_text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise AssertionError(f"{name} not in /metrics")


def test_response_follows_section_2_4():
    response = mock.score("some comment")
    assert set(response) == {"scores", "flagged", "flag", "low_confidence", "model_version"}
    assert set(response["scores"]) == set(mock.LABELS) == set(response["flagged"])
    assert all(0 <= s <= 1 for s in response["scores"].values())
    assert response["flag"] == any(response["flagged"].values())
    for label, score in response["scores"].items():
        assert response["flagged"][label] == (score >= mock.THRESHOLD)


def test_scores_are_deterministic_and_toxicity_leads():
    assert mock.score("same text") == mock.score("same text")
    response = mock.score("another comment")
    assert response["scores"]["toxicity"] == max(response["scores"].values())


def test_about_a_tenth_of_comments_are_flagged():
    flags = [mock.score(f"comment number {i}")["flag"] for i in range(5000)]
    assert 0.07 < sum(flags) / len(flags) < 0.15


def test_predict_and_health_over_http(server):
    response = _post(server, {"text": "hello there"})
    assert response["model_version"] == mock.MODEL_VERSION
    assert isinstance(response["latency_ms"], float)
    assert json.loads(_get(server, "/health")) == {"status": "ok",
                                                    "model_version": mock.MODEL_VERSION}


def test_metrics_have_every_section_2_4_name(server):
    text = _get(server, "/metrics")
    families = {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")}
    assert SECTION_2_4_METRICS <= families


def test_counters_follow_the_traffic(server):
    texts = [f"comment {i}" for i in range(50)]
    flagged = sum(mock.score(t)["flag"] for t in texts)
    for t in texts:
        _post(server, {"text": t})
    metrics = _get(server, "/metrics")
    assert _sample(metrics, "distillate_requests_total") == 50
    assert _sample(metrics, "distillate_flag_total") == flagged
    assert _sample(metrics, "distillate_request_latency_seconds_count") == 50


def test_histogram_buckets_are_cumulative(server):
    for i in range(20):
        _post(server, {"text": f"x{i}"})
    lines = [line for line in _get(server, "/metrics").splitlines()
             if line.startswith('distillate_score_bucket{label="toxicity"')]
    counts = [float(line.split()[1]) for line in lines]
    assert counts == sorted(counts) and counts[-1] == 20  # le="+Inf" holds every observation


def test_a_bad_body_is_rejected_but_not_a_service_error(server):
    """Bad client input is a 422: counted as a request, not in errors_total, so malformed
    requests can't trip the error-rate alerts."""
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(server, {"not_text": 1})
    assert err.value.code == 422
    metrics = _get(server, "/metrics")
    assert _sample(metrics, "distillate_errors_total") == 0
    assert _sample(metrics, "distillate_requests_total") == 1


def test_a_lone_surrogate_is_rejected_cleanly(server, capfd):
    """Regression: "\\ud800" passed the input check, then crashed score() outside the error
    handling: a dropped connection, no count, and a traceback on stderr."""
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(server, {"text": "\ud800"})
    assert err.value.code == 422
    assert _sample(_get(server, "/metrics"), "distillate_requests_total") == 1
    assert "Traceback" not in capfd.readouterr().err


def test_a_failure_on_our_side_is_a_counted_500(server, monkeypatch, capfd):
    def broken(text):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(mock, "score", broken)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(server, {"text": "hello"})
    assert err.value.code == 500
    assert _sample(_get(server, "/metrics"), "distillate_errors_total") == 1
    assert "Traceback" not in capfd.readouterr().err


def _quantile(q, buckets, observations):
    """What Prometheus's histogram_quantile returns: linear interpolation inside the bucket."""
    cumulative = [sum(v <= upper for v in observations) for upper in buckets]
    rank = q * len(observations)
    for i, count in enumerate(cumulative):
        if count >= rank:
            lower, below = (buckets[i - 1], cumulative[i - 1]) if i else (0.0, 0)
            return lower + (buckets[i] - lower) * (rank - below) / (count - below)
    raise AssertionError("rank beyond the last bucket")


def test_latency_buckets_resolve_a_p95_near_the_fp32_measurement():
    """Regression: with buckets jumping from 0.1 to 0.25 s, a true p95 of 144 ms read as
    about 242 ms. With 0.15 and 0.2 it lands in the right bucket."""
    observations = [0.144] * 100
    assert 0.1 < _quantile(0.95, mock.LATENCY_BUCKETS, observations) <= 0.15
    old = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
    assert _quantile(0.95, old, observations) > 0.24


def test_textfile_metrics_are_served_too(server):
    assert 'distillate_canary_mean_auc{set="civil"} 0.83' in _get(server, "/metrics")


def test_no_text_in_metrics(server):
    _post(server, {"text": "a very distinctive comment"})
    assert "distinctive" not in _get(server, "/metrics")
