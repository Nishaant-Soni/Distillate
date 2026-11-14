"""Mock of the serving API (work plan v4, section 2.4), for building monitoring before Phase 3.

    POST /predict   {"text": "..."} -> scores, flagged, flag, low_confidence, model_version,
                    latency_ms, exactly as section 2.4 specifies
    GET  /health    {"status": "ok", "model_version": "..."}
    GET  /metrics   Prometheus text with the section 2.4 metric names, counted from the traffic
                    this mock actually receives, plus any *.prom files in --textfile-dir
                    (for example the canary's reports/canary/canary.prom)

Scores are made up but deterministic: each comment's text hash gives a base score, skewed
so that roughly 10% of comments are flagged, like real traffic, instead of the ~98% that
six independent random scores would give. They carry no signal, so canary AUCs against
this mock sit near 0.5. Text is never logged or stored (section 2.4); only its hash is used.

Standard library only, so it runs anywhere Python 3.10+ does:

    python mock_service.py --port 8000 --textfile-dir ../../Data-Pipeline/reports/canary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
# Each label's score is the base score times its weight, so the labels move together the way
# real toxicity subtypes do, and toxicity is the most often flagged.
LABEL_WEIGHTS = {"toxicity": 1.0, "insult": 0.9, "obscene": 0.85, "identity_attack": 0.7,
                 "threat": 0.6, "severe_toxicity": 0.5}
THRESHOLD = 0.5
LOW_CONFIDENCE_MARGIN = 0.1
MODEL_VERSION = "mock-0"
# Seconds. Fine buckets around the 50 ms latency target (v4 section 6).
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def score(text: str) -> dict:
    """A section 2.4 response for ``text``. Deterministic: the same text, the same answer."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    base = (int.from_bytes(digest[:8], "big") / 2**64) ** 6  # P(base >= 0.5) is about 0.11
    scores = {label: round(base * LABEL_WEIGHTS[label], 4) for label in LABELS}
    flagged = {label: value >= THRESHOLD for label, value in scores.items()}
    return {
        "scores": scores,
        "flagged": flagged,
        "flag": any(flagged.values()),
        "low_confidence": any(abs(v - THRESHOLD) < LOW_CONFIDENCE_MARGIN
                              for v in scores.values()),
        "model_version": MODEL_VERSION,
    }


class Histogram:
    """A Prometheus histogram: cumulative bucket counts, a sum and a count."""

    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1
        self.total += value
        self.n += 1

    def lines(self, name: str, labels: str = "") -> list[str]:
        sep = "," if labels else ""
        out = [f'{name}_bucket{{{labels}{sep}le="{upper}"}} {count}'
               for upper, count in zip(self.buckets, self.counts, strict=True)]
        out.append(f'{name}_bucket{{{labels}{sep}le="+Inf"}} {self.n}')
        suffix = f"{{{labels}}}" if labels else ""
        out.append(f"{name}_sum{suffix} {round(self.total, 6)}")
        out.append(f"{name}_count{suffix} {self.n}")
        return out


class Metrics:
    """The section 2.4 metrics, updated as requests arrive. Thread-safe."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.flag = 0
        self.low_confidence = 0
        self.flagged = {label: 0 for label in LABELS}
        self.latency = Histogram(LATENCY_BUCKETS)
        self.scores = {label: Histogram(SCORE_BUCKETS) for label in LABELS}

    def record(self, response: dict | None, seconds: float) -> None:
        with self.lock:
            self.requests += 1
            self.latency.observe(seconds)
            if response is None:
                self.errors += 1
                return
            self.flag += response["flag"]
            self.low_confidence += response["low_confidence"]
            for label in LABELS:
                self.flagged[label] += response["flagged"][label]
                self.scores[label].observe(response["scores"][label])

    def text(self) -> str:
        with self.lock:
            lines = [
                "# HELP distillate_request_latency_seconds Time to answer /predict.",
                "# TYPE distillate_request_latency_seconds histogram",
                *self.latency.lines("distillate_request_latency_seconds"),
                "# HELP distillate_requests_total /predict requests received.",
                "# TYPE distillate_requests_total counter",
                f"distillate_requests_total {self.requests}",
                "# HELP distillate_errors_total /predict requests that failed.",
                "# TYPE distillate_errors_total counter",
                f"distillate_errors_total {self.errors}",
                "# HELP distillate_flagged_total Comments flagged, per label.",
                "# TYPE distillate_flagged_total counter",
                *[f'distillate_flagged_total{{label="{label}"}} {count}'
                  for label, count in self.flagged.items()],
                "# HELP distillate_flag_total Comments flagged on any label (the decision).",
                "# TYPE distillate_flag_total counter",
                f"distillate_flag_total {self.flag}",
                "# HELP distillate_score Model scores, per label.",
                "# TYPE distillate_score histogram",
            ]
            for label in LABELS:
                lines += self.scores[label].lines("distillate_score", f'label="{label}"')
            lines += [
                "# HELP distillate_low_confidence_total Answers within the margin of a threshold.",
                "# TYPE distillate_low_confidence_total counter",
                f"distillate_low_confidence_total {self.low_confidence}",
            ]
        return "\n".join(lines) + "\n"


def make_handler(metrics: Metrics, textfile_dir: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: dict) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

        def do_POST(self):  # noqa: N802 - http.server's naming
            if self.path != "/predict":
                self._json(404, {"detail": "not found"})
                return
            start = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", 0))
                text = json.loads(self.rfile.read(length))["text"]
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
            except (ValueError, KeyError, TypeError):
                metrics.record(None, time.perf_counter() - start)
                self._json(422, {"detail": 'body must be {"text": "..."}'})
                return
            response = score(text)
            seconds = time.perf_counter() - start
            response["latency_ms"] = round(seconds * 1000, 3)
            metrics.record(response, seconds)
            self._json(200, response)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._json(200, {"status": "ok", "model_version": MODEL_VERSION})
            elif self.path == "/metrics":
                body = metrics.text()
                if textfile_dir is not None and textfile_dir.is_dir():
                    for prom in sorted(textfile_dir.glob("*.prom")):
                        body += prom.read_text(encoding="utf-8")
                self._send(200, body.encode("utf-8"), "text/plain; version=0.0.4")
            else:
                self._json(404, {"detail": "not found"})

        def log_message(self, fmt, *args):
            # Silent: nothing about requests is logged, so no text can end up in logs.
            pass

    return Handler


def serve(port: int, textfile_dir: Path | None = None,
          host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start the mock in a background thread and return the server (port 0 picks a free one)."""
    server = ThreadingHTTPServer((host, port), make_handler(Metrics(), textfile_dir))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to let a Prometheus container scrape it")
    parser.add_argument("--textfile-dir", type=Path, default=None,
                        help="also serve every *.prom file in this folder, e.g. the canary's")
    args = parser.parse_args()
    server = serve(args.port, args.textfile_dir, args.host)
    print(f"mock serving API on http://{args.host}:{server.server_address[1]} (Ctrl+C to stop)")
    try:
        # Short sleeps rather than one long Event().wait(): on Windows a blocking wait never
        # returns to Python, so Ctrl+C wouldn't stop the server.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
