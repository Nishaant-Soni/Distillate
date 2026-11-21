"""Tests for smoke_test.py against a local fake of the serving API (section 2.4)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import smoke_test as st

THRESHOLDS = {lab: 0.5 for lab in st.LABELS}
MODEL_VERSION = "student-test-1"


def expected_scores(text: str) -> dict[str, float]:
    base = (len(text) % 7) / 10
    return {lab: round(base + 0.01 * i, 6) for i, lab in enumerate(st.LABELS)}


REFERENCE = {"texts": ["revert", "thanks for the edit", "please see the talk page"],
             "scores": [expected_scores(t) for t in
                        ["revert", "thanks for the edit", "please see the talk page"]]}


def make_server(score_shift=0.0, model_version=MODEL_VERSION, bad_flag=False,
                health_status="ok"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": health_status, "model_version": model_version})
            else:
                self._send({"detail": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            text = json.loads(self.rfile.read(length))["text"]
            scores = {lab: s + score_shift for lab, s in expected_scores(text).items()}
            flagged = {lab: scores[lab] >= THRESHOLDS[lab] for lab in st.LABELS}
            if bad_flag:
                flagged["threat"] = not flagged["threat"]
            self._send({"scores": scores, "flagged": flagged, "flag": any(flagged.values()),
                        "low_confidence": False, "model_version": model_version,
                        "latency_ms": 1.0})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def serve():
    servers = []

    def start(**kwargs):
        server, url = make_server(**kwargs)
        servers.append(server)
        return url

    yield start
    for server in servers:
        server.shutdown()


def run(url, **kwargs):
    return st.run(url, REFERENCE, THRESHOLDS, MODEL_VERSION, 1e-3, timeout=5,
                  health_attempts=kwargs.pop("health_attempts", 2), health_delay=0.01, **kwargs)


def test_matching_service_passes(serve):
    result = run(serve())
    assert result["passed"], result["problems"]
    assert result["n_checked"] == 3 and result["max_abs_diff"] == pytest.approx(0.0)


def test_scores_that_drift_fail(serve):
    result = run(serve(score_shift=0.01))
    assert not result["passed"]
    assert result["n_problems"] == 3 * len(st.LABELS)
    assert "diff 1.00e-02 > 1e-03" in result["problems"][0]


def test_tiny_differences_within_tolerance_pass(serve):
    assert run(serve(score_shift=5e-4))["passed"]


def test_wrong_model_version_fails(serve):
    result = run(serve(model_version="student-other"))
    assert not result["passed"]
    assert "/health model_version 'student-other'" in result["problems"][0]


def test_flags_must_agree_with_scores_and_thresholds(serve):
    result = run(serve(bad_flag=True))
    assert not result["passed"]
    assert any("threat flagged=" in p for p in result["problems"])


def test_unhealthy_service_fails(serve):
    result = run(serve(health_status="starting"))
    assert not result["passed"] and result["n_checked"] == 0


def test_unreachable_service_fails_after_retries():
    result = st.run("http://127.0.0.1:9", REFERENCE, THRESHOLDS, MODEL_VERSION, 1e-3,
                    timeout=1, health_attempts=2, health_delay=0.01)
    assert not result["passed"]
    assert "unreachable after 2 tries" in result["problems"][0]


def test_cli_writes_summary_and_exit_code(serve, tmp_path):
    url = serve()
    reference = tmp_path / "reference.json"
    thresholds = tmp_path / "thresholds.json"
    reference.write_text(json.dumps(REFERENCE))
    thresholds.write_text(json.dumps(THRESHOLDS))
    out = tmp_path / "smoke.json"
    code = st.main(["--url", url, "--reference", str(reference), "--thresholds",
                    str(thresholds), "--model-version", MODEL_VERSION, "--out", str(out)])
    assert code == 0
    assert json.loads(out.read_text())["passed"] is True


def test_flags_within_tolerance_of_a_threshold_are_not_judged():
    expected = {lab: 0.2 for lab in st.LABELS}
    response = {"scores": {**expected, "threat": 0.5003}, "model_version": MODEL_VERSION,
                "flagged": {lab: False for lab in st.LABELS}, "flag": False}
    expected["threat"] = 0.5003
    problems, _ = st.check_prediction(response, expected, THRESHOLDS, MODEL_VERSION, 1e-3)
    assert problems == []                     # 0.5003 rounds near 0.5: flag not judged
    response["scores"]["threat"] = expected["threat"] = 0.6
    problems, _ = st.check_prediction(response, expected, THRESHOLDS, MODEL_VERSION, 1e-3)
    assert any("threat flagged=" in p for p in problems)
