"""Distillate serving API (work plan v5 §2.4): FastAPI over ONNX Runtime, one model package.

    POST /predict   {"text": "..."} -> {"scores": {label: float}, "flagged": {label: bool},
                    "flag": bool, "low_confidence": bool, "model_version": str,
                    "latency_ms": float}
    GET  /health    {"status": "ok", "model_version": str}
    GET  /metrics   Prometheus text: distillate_request_latency_seconds,
                    distillate_requests_total, distillate_errors_total,
                    distillate_flagged_total{label}, distillate_flag_total,
                    distillate_score{label}, distillate_low_confidence_total
                    (for local docker-compose; off on Cloud Run unless expose_metrics is true)

Comment text is never logged. Each /predict writes one JSON line to stdout (Cloud Run sends it
to Cloud Logging) with text_sha256, char_len, scores, flagged, flag, low_confidence,
model_version and latency_ms. Rejected input (413, 415, 422, 429) counts as a request but not
as an error, so bad clients don't trip the error-rate alerts; only failures on our side (500)
count as errors.

Run locally against a package folder:

    DISTILLATE_PACKAGE_DIR=<package> uvicorn app:create_app --factory --port 8080
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from scorer import LABELS, Scorer, decide

HERE = Path(__file__).resolve().parent
# Seconds; the same buckets as Deployment/monitoring/mock_service.py, so the dashboards'
# histogram_quantile resolves both the 50 ms target and the ~144 ms fp32 p95.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Only the counter itself, no *_created series, matching the mock's exposition.
disable_created_metrics()

_stdout_lock = threading.Lock()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """config.yaml (or ``path``) with the environment overrides listed in it."""
    path = Path(path or os.environ.get("DISTILLATE_CONFIG") or HERE / "config.yaml")
    cfg = yaml.safe_load(path.read_text())
    if os.environ.get("DISTILLATE_PACKAGE_DIR"):
        cfg["package_dir"] = os.environ["DISTILLATE_PACKAGE_DIR"]
    if os.environ.get("DISTILLATE_RATE_LIMIT_PER_MINUTE"):
        cfg["rate_limit"]["requests_per_minute"] = float(
            os.environ["DISTILLATE_RATE_LIMIT_PER_MINUTE"])
    if os.environ.get("DISTILLATE_ORT_THREADS"):
        cfg["onnx"]["intra_op_threads"] = int(os.environ["DISTILLATE_ORT_THREADS"])
    if os.environ.get("DISTILLATE_EXPOSE_METRICS"):
        cfg["expose_metrics"] = os.environ["DISTILLATE_EXPOSE_METRICS"].strip().lower()
    return cfg


def metrics_exposed(setting: Any) -> bool:
    """``expose_metrics``: true/false, or "auto" (everywhere but Cloud Run, which sets
    K_SERVICE)."""
    value = str(setting).strip().lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    if value == "auto":
        return not os.environ.get("K_SERVICE")
    raise ValueError(f"expose_metrics must be true, false or auto, not {setting!r}")


def emit(record: dict[str, Any]) -> None:
    """One JSON line to stdout. Callers never put comment text in ``record``."""
    line = json.dumps(record, separators=(",", ":"))
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class RateLimiter:
    """Token bucket per client key: ``per_minute`` sustained, ``burst`` at once. Thread-safe.
    ``per_minute <= 0`` turns it off."""

    MAX_CLIENTS = 10_000

    def __init__(self, per_minute: float, burst: float, clock=time.monotonic):
        self.rate = float(per_minute) / 60.0
        self.burst = max(float(burst), 1.0)
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.rate <= 0:
            return True
        now = self.clock()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            self.buckets[key] = (tokens - 1.0 if allowed else tokens, now)
            if len(self.buckets) > self.MAX_CLIENTS:     # drop clients whose bucket is full
                full = self.burst / self.rate
                self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < full}
            return allowed


class Metrics:
    """The §2.4 Prometheus metrics, in their own registry."""

    def __init__(self):
        self.registry = CollectorRegistry()
        r = self.registry
        self.latency = Histogram("distillate_request_latency_seconds", "Time to answer /predict.",
                                 buckets=LATENCY_BUCKETS, registry=r)
        self.requests = Counter("distillate_requests", "/predict requests received.",
                                registry=r)
        self.errors = Counter("distillate_errors", "/predict requests that failed on our side.",
                              registry=r)
        self.flagged = Counter("distillate_flagged", "Comments flagged, per label.", ["label"],
                               registry=r)
        self.flag = Counter("distillate_flag", "Comments flagged on any label (the decision).",
                            registry=r)
        self.score = Histogram("distillate_score", "Model scores, per label.", ["label"],
                               buckets=SCORE_BUCKETS, registry=r)
        self.low_confidence = Counter("distillate_low_confidence",
                                      "Answers within the margin of a threshold.", registry=r)
        for label in LABELS:          # every label's series exists from the start, at 0
            self.flagged.labels(label=label)
            self.score.labels(label=label)

    def record(self, response: dict[str, Any], seconds: float) -> None:
        self.requests.inc()
        self.latency.observe(seconds)
        self.flag.inc(int(response["flag"]))
        self.low_confidence.inc(int(response["low_confidence"]))
        for label in LABELS:
            self.flagged.labels(label=label).inc(int(response["flagged"][label]))
            self.score.labels(label=label).observe(response["scores"][label])


def client_key(request: Request, trusted_proxy_hops: int = 0) -> str:
    """The caller's address for the rate limit: the X-Forwarded-For entry
    ``trusted_proxy_hops`` places before the last (entries further left can be forged)."""
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",")
                 if part.strip()]
    if forwarded:
        return forwarded[max(0, len(forwarded) - 1 - trusted_proxy_hops)]
    return request.client.host if request.client else "unknown"


class Rejected(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail


async def read_body(request: Request, limit: int) -> bytes:
    """The request body, refusing (413) more than ``limit`` bytes without reading it all."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise Rejected(413, f"body larger than {limit} bytes")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise Rejected(413, f"body larger than {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_text(body: bytes, max_chars: int) -> str:
    """The ``text`` field of a ``{"text": "..."}`` body, or Rejected. Error details never
    echo the input."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise Rejected(422, 'body must be JSON: {"text": "..."}') from None
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise Rejected(422, 'body must be {"text": "..."} with a string text')
    text = payload["text"]
    try:
        text.encode("utf-8")        # a lone surrogate such as "\ud800" can't be hashed or scored
    except UnicodeEncodeError:
        raise Rejected(422, "text is not valid Unicode") from None
    if not text.strip():
        raise Rejected(422, "text is empty")
    if len(text) > max_chars:
        raise Rejected(413, f"text longer than {max_chars} characters")
    return text


def create_app(config: dict[str, Any] | None = None, scorer: Scorer | None = None) -> FastAPI:
    """The app. Without ``scorer``, the package in ``config["package_dir"]`` loads at startup
    (before the port opens, so Cloud Run sends no traffic to an instance still loading)."""
    cfg = config or load_config()
    state: dict[str, Any] = {"scorer": scorer}
    metrics = Metrics()
    limiter = RateLimiter(cfg["rate_limit"]["requests_per_minute"], cfg["rate_limit"]["burst"])
    margin = float(cfg["low_confidence_margin"])
    hops = int(cfg["rate_limit"].get("trusted_proxy_hops", 0))
    expose_metrics = metrics_exposed(cfg.get("expose_metrics", "auto"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["scorer"] is None:
            onnx_cfg = cfg.get("onnx", {})
            state["scorer"] = Scorer(cfg["package_dir"], onnx_cfg.get("intra_op_threads", 0),
                                     onnx_cfg.get("inter_op_threads", 0))
        state["scorer"].scores("warm up")      # the first real request isn't the slow one
        emit({"severity": "INFO", "event": "startup",
              "model_version": state["scorer"].model_version,
              "max_length": state["scorer"].max_length})
        yield

    app = FastAPI(title="Distillate", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)

    def reject(status: int, detail: str) -> JSONResponse:
        metrics.requests.inc()
        headers = {"Retry-After": "1"} if status == 429 else None
        return JSONResponse({"detail": detail}, status_code=status, headers=headers)

    def score(text: str) -> dict[str, Any]:
        s = state["scorer"]
        scores = s.scores(text)
        return {"scores": scores, **decide(scores, s.thresholds, margin),
                "model_version": s.model_version}

    @app.post("/predict")
    async def predict(request: Request):
        start = time.perf_counter()
        if not limiter.allow(client_key(request, hops)):
            return reject(429, "rate limit exceeded")
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            return reject(415, "Content-Type must be application/json")
        try:
            text = parse_text(await read_body(request, int(cfg["max_body_bytes"])),
                              int(cfg["max_text_chars"]))
        except Rejected as e:
            return reject(e.status, e.detail)

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            response = await run_in_threadpool(score, text)
        except Exception as e:  # noqa: BLE001 - a failure on our side: counted and answered
            # 500; the log names the exception type only (a message could quote the input).
            seconds = time.perf_counter() - start
            metrics.requests.inc()
            metrics.errors.inc()
            metrics.latency.observe(seconds)
            emit({"severity": "ERROR", "event": "predict_error", "error": type(e).__name__,
                  "text_sha256": digest, "char_len": len(text),
                  "latency_ms": round(seconds * 1000, 3)})
            return JSONResponse({"detail": "internal error"}, status_code=500)
        seconds = time.perf_counter() - start
        response["latency_ms"] = round(seconds * 1000, 3)
        metrics.record(response, seconds)
        emit({"severity": "INFO", "text_sha256": digest, "char_len": len(text),
              "scores": response["scores"], "flagged": response["flagged"],
              "flag": response["flag"], "low_confidence": response["low_confidence"],
              "model_version": response["model_version"],
              "latency_ms": response["latency_ms"]})
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": state["scorer"].model_version}

    if expose_metrics:
        @app.get("/metrics")
        def prometheus():
            return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # FastAPI's default 422 and error bodies can echo the request; these don't.
    @app.exception_handler(RequestValidationError)
    async def invalid(_request: Request, _exc: RequestValidationError):
        return JSONResponse({"detail": "invalid request"}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    app.state.metrics = metrics
    app.state.limiter = limiter
    return app
