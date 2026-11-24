"""Read-only access to the model registry through MLflow's REST API.

The Status page needs only a few reads, so this uses ``requests`` rather than the ``mlflow``
package (hundreds of MB in the image, and a slower cold start on Cloud Run):

    GET /api/2.0/mlflow/registered-models/alias?name=&alias=   a version and its tags
    GET /api/2.0/mlflow/registered-models/get?name=            the model's aliases and tags
    GET /get-artifact?path=package/thresholds.json&run_uuid=    a version's thresholds

The tags are the ones ``Model-Pipeline/src/pipeline/register_model.py`` and
``rollback_model.py`` write (``model_version``, ``gate_passed``, ``gate_mean_auc``,
``ar_upload``, ``code_commit``, ``last_alias_move``; ``<alias>_previous`` on the model).
Every failure becomes a RegistryError with a short reason, never a crash of the page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import requests

from .api import LABELS


class RegistryError(Exception):
    """The registry couldn't be read (down, login refused, or an unexpected answer)."""


@dataclass(frozen=True)
class VersionInfo:
    alias: str
    version: str
    run_id: str | None
    model_version: str | None
    gate_passed: bool | None
    gate_mean_auc: float | None
    ar_upload: str | None
    code_commit: str | None
    last_alias_move: dict[str, Any] | None = None
    tags: dict[str, str] = field(default_factory=dict)
    created_ms: int | None = None      # when the version was registered (creation_timestamp)


def _millis(value: Any) -> int | None:
    """MLflow's int64 timestamps arrive as numbers or, from some servers, as strings."""
    try:
        return int(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _tags(items: list[dict[str, str]] | None) -> dict[str, str]:
    return {t["key"]: t.get("value", "") for t in items or [] if "key" in t}


def version_info(alias: str, body: dict[str, Any]) -> VersionInfo:
    """A VersionInfo from an MLflow ``model_version`` object."""
    tags = _tags(body.get("tags"))
    try:
        auc = float(tags["gate_mean_auc"]) if tags.get("gate_mean_auc") else None
    except ValueError:
        auc = None
    try:
        move = json.loads(tags["last_alias_move"]) if tags.get("last_alias_move") else None
    except ValueError:
        move = None
    gate = tags.get("gate_passed")
    return VersionInfo(alias=alias, version=str(body["version"]), run_id=body.get("run_id"),
                       model_version=tags.get("model_version"),
                       gate_passed=None if gate is None else gate == "true",
                       gate_mean_auc=auc, ar_upload=tags.get("ar_upload"),
                       code_commit=tags.get("code_commit"), last_alias_move=move, tags=tags,
                       created_ms=_millis(body.get("creation_timestamp")))


class Registry:
    """The registered model ``name`` on the MLflow server at ``uri``."""

    def __init__(self, uri: str, name: str, username: str | None = None,
                 password: str | None = None, timeout_s: float = 10,
                 session: requests.Session | None = None):
        self.uri = uri.rstrip("/")
        self.name = name
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        if username and password:
            self.session.auth = (username, password)

    def _get(self, path: str, params: dict[str, str], missing_ok: bool = False) -> Any:
        try:
            response = self.session.get(self.uri + path, params=params, timeout=self.timeout_s)
        except requests.RequestException as err:
            raise RegistryError(f"registry unreachable ({type(err).__name__})") from err
        if response.status_code == 404 and missing_ok:
            return None
        if response.status_code in (401, 403):
            raise RegistryError("registry login refused (check the MLflow username and password)")
        if response.status_code != 200:
            raise RegistryError(f"registry answered HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as err:
            raise RegistryError("registry answered with something that isn't JSON") from err

    def alias(self, alias: str) -> VersionInfo | None:
        """The version ``alias`` points at, or None if the alias isn't set."""
        body = self._get("/api/2.0/mlflow/registered-models/alias",
                         {"name": self.name, "alias": alias}, missing_ok=True)
        if body is None:
            return None
        if not isinstance(body, dict) or "model_version" not in body:
            raise RegistryError("unexpected answer for an alias")
        return version_info(alias, body["model_version"])

    def model_tags(self) -> dict[str, str]:
        body = self._get("/api/2.0/mlflow/registered-models/get", {"name": self.name})
        if not isinstance(body, dict) or "registered_model" not in body:
            raise RegistryError("unexpected answer for the registered model")
        return _tags(body["registered_model"].get("tags"))

    def thresholds(self, run_id: str) -> dict[str, float] | None:
        """The package's per-label thresholds for the run ``run_id``, or None if unavailable."""
        try:
            body = self._get("/get-artifact", {"path": "package/thresholds.json",
                                               "run_uuid": run_id}, missing_ok=True)
        except RegistryError:
            return None
        if not isinstance(body, dict) or sorted(body) != sorted(LABELS):
            return None
        try:
            return {label: float(body[label]) for label in LABELS}
        except (TypeError, ValueError):
            return None
