"""Tests for scripts/canary.py, against a real local HTTP server that follows the serving API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from scripts import canary
from scripts.stubs import LABELS

TOXIC = "you are a bad person"
CLEAN = "thanks for the helpful edit"


def _set(n_toxic, n_clean, severe=True):
    """A labeled canary set: toxic rows are positive on every label (severe optional)."""
    rows = []
    for i in range(n_toxic):
        rows.append({"id": f"t{i}", "text": f"{TOXIC} {i}",
                     **{lbl: 1 for lbl in LABELS}, "severe_toxicity": int(severe)})
    for i in range(n_clean):
        rows.append({"id": f"c{i}", "text": f"{CLEAN} {i}", **{lbl: 0 for lbl in LABELS}})
    return pd.DataFrame(rows).astype({lbl: "int8" for lbl in LABELS})


def response_for(text, threshold=0.5):
    """A perfect §2.4-shaped answer: high scores for toxic text, low for clean."""
    score = 0.9 if TOXIC in text else 0.1
    scores = {lbl: score for lbl in LABELS}
    flagged = {lbl: s >= threshold for lbl, s in scores.items()}
    return {"scores": scores, "flagged": flagged, "flag": any(flagged.values()),
            "low_confidence": False, "model_version": "mock-1", "latency_ms": 1.0}


class _Handler(BaseHTTPRequestHandler):
    mode = "perfect"   # set per test through the server fixture
    calls = []

    def do_POST(self):  # noqa: N802 - http.server's naming
        text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
        type(self).calls.append(text)
        if self.mode == "fail_some" and text.endswith(("0", "5")):
            self.send_response(500)
            self.end_headers()
            return
        body = response_for(text)
        if self.mode == "bad_shape":
            body = {"scores": body["scores"]}  # no flagged, no flag
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/health" and self.mode != "down":
            payload = json.dumps({"status": "ok", "model_version": "mock-1"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(503)
            self.end_headers()


@pytest.fixture
def server():
    _Handler.mode, _Handler.calls = "perfect", []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def cfg(tmp_path):
    wiki, civil = tmp_path / "canary_wikipedia.parquet", tmp_path / "canary_civil.parquet"
    _set(20, 80).to_parquet(wiki, index=False)
    _set(30, 70, severe=False).to_parquet(civil, index=False)  # Civil: severe never positive
    return {
        "labels": {"names": LABELS},
        "paths": {"summaries_dir": str(tmp_path / "summaries")},
        "canary": {
            "service_url": None, "timeout_s": 5, "workers": 4, "max_rows": None,
            "max_error_fraction": 0.01, "output_dir": str(tmp_path / "canary"),
            "sets": {
                "wikipedia": {"path": str(wiki)},
                "civil": {"path": str(civil), "exclude_labels": ["severe_toxicity"]},
            },
        },
    }


def _report(cfg):
    return json.loads((canary._resolve(cfg["canary"]["output_dir"]) / "canary_report.json")
                      .read_text(encoding="utf-8"))


def test_a_perfect_service_scores_perfectly(cfg, server):
    _, url = server
    canary.run(cfg, url=url)
    for name in ("wikipedia", "civil"):
        result = _report(cfg)["sets"][name]
        assert result["mean_auc"] == 1.0 and result["errors"] == 0
        assert result["flag"]["precision"] == 1.0 and result["flag"]["recall"] == 1.0
        assert result["flag"]["false_positive_rate"] == 0.0


def test_every_canary_comment_goes_through_the_service(cfg, server):
    _, url = server
    canary.run(cfg, url=url)
    assert len(_Handler.calls) == 200  # 100 per set


def test_civil_metrics_cover_five_labels(cfg, server):
    _, url = server
    canary.run(cfg, url=url)
    sets = _report(cfg)["sets"]
    assert "severe_toxicity" not in sets["civil"]["labels"]
    assert len(sets["civil"]["labels"]) == 5 and len(sets["wikipedia"]["labels"]) == 6


def test_prometheus_file_has_the_metrics(cfg, server):
    _, url = server
    canary.run(cfg, url=url)
    prom = (canary._resolve(cfg["canary"]["output_dir"]) / "canary.prom").read_text()
    assert 'distillate_canary_auc{set="civil",label="toxicity"} 1.0' in prom
    assert 'distillate_canary_flag_recall{set="wikipedia"} 1.0' in prom
    assert "# TYPE distillate_canary_mean_auc gauge" in prom
    assert "distillate_canary_last_run_timestamp_seconds" in prom
    for line in prom.splitlines():  # every sample line is "name{...} number"
        if line and not line.startswith("#"):
            float(line.rsplit(" ", 1)[1])


def test_no_comment_text_in_the_outputs(cfg, server):
    _, url = server
    canary.run(cfg, url=url)
    out = canary._resolve(cfg["canary"]["output_dir"])
    for path in out.iterdir():
        content = path.read_text(encoding="utf-8")
        assert TOXIC not in content and CLEAN not in content, path.name


def test_a_few_failed_calls_are_left_out_of_the_metrics(cfg, server):
    _, url = server
    _Handler.mode = "fail_some"
    cfg["canary"]["max_error_fraction"] = 0.5
    canary.run(cfg, url=url)
    result = _report(cfg)["sets"]["wikipedia"]
    assert result["errors"] == 20 and result["rows_scored"] == 80  # ids ending 0 or 5 fail
    assert result["mean_auc"] == 1.0


def test_too_many_failed_calls_fail_the_run(cfg, server):
    _, url = server
    _Handler.mode = "fail_some"
    with pytest.raises(canary.CanaryError, match="calls failed"):
        canary.run(cfg, url=url)


def test_a_response_that_breaks_the_api_counts_as_an_error(cfg, server):
    _, url = server
    _Handler.mode = "bad_shape"
    with pytest.raises(canary.CanaryError, match="bad_response"):
        canary.run(cfg, url=url)


def test_no_url_gives_a_clear_error(cfg, monkeypatch):
    monkeypatch.delenv("CANARY_URL", raising=False)
    with pytest.raises(canary.CanaryError, match="No service URL"):
        canary.run(cfg)


def test_a_label_with_one_class_gets_no_auc(cfg):
    canary.run(cfg, predict=response_for)  # predict injected: no HTTP needed
    civil = _report(cfg)["sets"]["civil"]
    assert civil["labels"]["toxicity"]["auc"] == 1.0
    wiki = _report(cfg)["sets"]["wikipedia"]
    assert all(m["auc"] is not None for m in wiki["labels"].values())
    one_class = canary.metrics(_set(0, 10).drop(columns=["id", "text"]),
                               pd.DataFrame([{"ok": True, "flag": False,
                                              **{f"p_{lbl}": 0.1 for lbl in LABELS},
                                              **{f"flagged_{lbl}": False for lbl in LABELS}}
                                             ] * 10), LABELS)
    assert one_class["labels"]["threat"]["auc"] is None and one_class["mean_auc"] is None


def test_max_rows_limits_the_replay(cfg):
    cfg["canary"]["max_rows"] = 10
    summary = canary.run(cfg, predict=response_for)
    assert summary["rows_in"] == 20


def test_summary_records_counts(cfg):
    summary = canary.run(cfg, predict=response_for)
    assert summary["rows_in"] == 200 and summary["rows_out"] == 200
    assert summary["dropped"] == {"failed_calls": 0}
    assert summary["mean_auc"] == {"wikipedia": 1.0, "civil": 1.0}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.1, "high", None, True])
def test_a_score_that_isnt_a_finite_number_in_0_1_is_a_bad_response(cfg, bad):
    """Regression: NaN crashed roc_auc_score and a string crashed float(), outside the
    per-call error handling, so the whole run failed with an unclear error."""
    def predict(text):
        response = response_for(text)
        if text.endswith("7"):  # 10 of each 100
            response["scores"]["insult"] = bad
        return response

    cfg["canary"]["max_error_fraction"] = 0.2
    canary.run(cfg, predict=predict)
    result = _report(cfg)["sets"]["wikipedia"]
    assert result["errors"] == 10 and result["rows_scored"] == 90
    assert result["mean_auc"] == 1.0


def test_a_flagged_value_that_isnt_true_or_false_is_a_bad_response(cfg):
    def predict(text):
        response = response_for(text)
        response["flagged"]["toxicity"] = "yes"
        return response

    with pytest.raises(canary.CanaryError, match="flagged not true/false"):
        canary.run(cfg, predict=predict)


def test_a_down_service_fails_before_any_replay(cfg, server):
    _, url = server
    _Handler.mode = "down"
    with pytest.raises(canary.CanaryError, match="Health check failed"):
        canary.run(cfg, url=url)
    assert _Handler.calls == []  # nothing was replayed


def test_the_cli_alerts_when_the_run_fails(monkeypatch):
    sent = []
    monkeypatch.setattr(canary, "send_alert", lambda *args: sent.append(args))
    monkeypatch.setattr(canary, "load_config", lambda path: {})

    def failing_run(cfg, url=None):
        raise canary.CanaryError("civil: 50 of 100 calls failed")

    monkeypatch.setattr(canary, "run", failing_run)
    with pytest.raises(canary.CanaryError):
        canary.main(["--url", "http://service"])
    [(level, title, details)] = sent
    assert level == "fail" and title == "Canary run failed"
    assert details == {"error": "civil: 50 of 100 calls failed", "url": "http://service"}


def test_two_model_versions_in_one_run_are_warned_about(cfg, monkeypatch):
    warnings = []
    monkeypatch.setattr(canary.logger, "warning", lambda *args: warnings.append(args))

    def predict(text):
        response = response_for(text)
        response["model_version"] = "v2" if text.endswith(("1", "3")) else "v1"
        return response

    canary.run(cfg, predict=predict)
    assert _report(cfg)["sets"]["wikipedia"]["model_versions"] == ["v1", "v2"]
    assert any("model versions" in args[0] for args in warnings)
