"""The serving API client (contract section 4), with a live and a mock backend.

    POST /predict  {"text": "..."} -> {"scores": {label: float}, "flagged": {label: bool},
                                       "flag": bool, "low_confidence": bool,
                                       "model_version": "...", "latency_ms": float}
    GET  /health   -> {"status": "ok", "model_version": "..."}

The pages only call ``predict`` and ``health``, so they don't know which backend they use. The
mock exists for tests and for running the app before the live endpoint does; its scores are
made up (deterministic, from the text's hash) and the app labels them as mock. For a local
HTTP mock of the same API, use ``Deployment/monitoring/mock_service.py`` and point
``DISTILLATE_API_URL`` at it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import requests

from .config import Settings

LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")
MOCK_VERSION = "mock"


class APIError(Exception):
    """The service couldn't be reached, refused the request, or answered off contract."""


def validate(body: Any) -> dict[str, Any]:
    """``body`` if it is a /predict response in the contract's shape; else APIError."""
    if not isinstance(body, dict):
        raise APIError(f"expected a JSON object, got {type(body).__name__}")
    problems = []
    for key, kind in (("scores", dict), ("flagged", dict), ("flag", bool),
                      ("low_confidence", bool), ("model_version", str)):
        if not isinstance(body.get(key), kind):
            problems.append(f"{key} missing or not {kind.__name__}")
    latency = body.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, int | float):
        problems.append("latency_ms missing or not a number")
    for label in LABELS:
        score = (body.get("scores") or {}).get(label) if isinstance(body.get("scores"), dict) \
            else None
        if isinstance(score, bool) or not isinstance(score, int | float) \
                or not math.isfinite(score) or not 0 <= score <= 1:
            problems.append(f"scores.{label} missing or not in [0, 1]")
        flagged = body.get("flagged")
        if isinstance(flagged, dict) and not isinstance(flagged.get(label), bool):
            problems.append(f"flagged.{label} missing or not bool")
    if problems:
        raise APIError("response off contract: " + "; ".join(problems))
    return body


class LiveAPI:
    """The serving API at ``url``."""

    mock = False

    def __init__(self, url: str, timeout_s: float = 10, session: requests.Session | None = None):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def _call(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self.session.request(method, self.url + path, timeout=self.timeout_s,
                                            **kwargs)
        except requests.RequestException as err:
            raise APIError(f"the service is unreachable ({type(err).__name__})") from err
        if response.status_code == 429:
            raise APIError("the service is rate limiting requests; try again shortly")
        if response.status_code != 200:
            raise APIError(f"the service answered HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as err:
            raise APIError("the service answered with something that isn't JSON") from err

    def predict(self, text: str) -> dict[str, Any]:
        return validate(self._call("POST", "/predict", json={"text": text}))

    def health(self) -> dict[str, Any]:
        body = self._call("GET", "/health")
        if not isinstance(body, dict) or "status" not in body:
            raise APIError("health answer off contract")
        return body


class MockAPI:
    """Made-up, deterministic answers in the contract's shape, for tests and the pre-live app."""

    mock = True
    thresholds = dict.fromkeys(LABELS, 0.5)
    margin = 0.1

    def predict(self, text: str) -> dict[str, Any]:
        scores = {}
        for label in LABELS:
            digest = hashlib.sha256(f"{label}\0{text}".encode()).digest()
            # Squared, so most scores are low and only a few labels cross 0.5, like real traffic.
            scores[label] = round((int.from_bytes(digest[:8], "big") / 2**64) ** 2, 4)
        flagged = {label: scores[label] >= self.thresholds[label] for label in LABELS}
        low = any(abs(scores[label] - self.thresholds[label]) < self.margin for label in LABELS)
        return {"scores": scores, "flagged": flagged, "flag": any(flagged.values()),
                "low_confidence": low, "model_version": MOCK_VERSION, "latency_ms": 0.0}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_version": MOCK_VERSION}


def make_api(settings: Settings) -> LiveAPI | MockAPI:
    """The live API when ``DISTILLATE_API_URL`` is set, else the mock."""
    if settings.api_url:
        return LiveAPI(settings.api_url, settings.api_timeout_s)
    return MockAPI()
