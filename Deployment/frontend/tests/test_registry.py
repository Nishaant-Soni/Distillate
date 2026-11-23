"""The registry client, against answers recorded in MLflow's REST format."""

import json

import pytest
import requests

from demo_app.api import LABELS
from demo_app.registry import Registry, RegistryError, version_info

V1 = {"name": "distillate-student", "version": "1", "run_id": "ab34619a", "aliases": ["production"],
      "tags": [{"key": "model_version", "value": "student-distilbert-mixed-5ep-20261112T210931Z"},
               {"key": "gate_passed", "value": "true"},
               {"key": "gate_mean_auc", "value": "0.984500"},
               {"key": "ar_upload", "value": "ok"},
               {"key": "code_commit", "value": "80a4502"},
               {"key": "last_alias_move",
                "value": json.dumps({"alias": "production", "from": "2", "to": "1",
                                     "reason": "rollback demo"})}]}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self.body = status, body

    def json(self):
        if self.body is None:
            raise ValueError("not json")
        return self.body


class FakeSession:
    """Answers each path from ``routes``: a FakeResponse, or an exception to raise."""

    def __init__(self, routes):
        self.routes, self.auth, self.calls = routes, None, []

    def get(self, url, params=None, timeout=None):
        path = url.split("://", 1)[1].split("/", 1)[1]
        self.calls.append(("/" + path, params))
        answer = self.routes["/" + path]
        if isinstance(answer, Exception):
            raise answer
        return answer(params) if callable(answer) else answer


def registry(routes):
    return Registry("http://mlflow:5000/", "distillate-student", "team", "secret",
                    session=FakeSession(routes))


def test_an_alias_and_its_tags():
    reg = registry({"/api/2.0/mlflow/registered-models/alias":
                    FakeResponse(body={"model_version": V1})})
    info = reg.alias("production")
    assert (info.version, info.run_id, info.gate_passed, info.ar_upload) == \
        ("1", "ab34619a", True, "ok")
    assert info.gate_mean_auc == pytest.approx(0.9845)
    assert info.last_alias_move["reason"] == "rollback demo"
    assert reg.session.auth == ("team", "secret")
    assert reg.session.calls == [("/api/2.0/mlflow/registered-models/alias",
                                  {"name": "distillate-student", "alias": "production"})]


def test_a_missing_alias_is_none():
    reg = registry({"/api/2.0/mlflow/registered-models/alias": FakeResponse(status=404, body={})})
    assert reg.alias("staging") is None


@pytest.mark.parametrize("answer, message", [
    (requests.ConnectionError(), "unreachable"),
    (FakeResponse(status=401, body={}), "login refused"),
    (FakeResponse(status=500, body={}), "HTTP 500"),
    (FakeResponse(body=None), "isn't JSON"),
    (FakeResponse(body={"something": "else"}), "unexpected"),
])
def test_failures_become_registry_errors(answer, message):
    reg = registry({"/api/2.0/mlflow/registered-models/alias": answer})
    with pytest.raises(RegistryError, match=message):
        reg.alias("production")


def test_model_tags():
    body = {"registered_model": {"name": "distillate-student",
                                 "tags": [{"key": "production_previous", "value": "2"}]}}
    reg = registry({"/api/2.0/mlflow/registered-models/get": FakeResponse(body=body)})
    assert reg.model_tags() == {"production_previous": "2"}


def test_thresholds_from_the_package():
    good = dict.fromkeys(LABELS, 0.3)
    reg = registry({"/get-artifact": FakeResponse(body=good)})
    assert reg.thresholds("ab34619a") == good
    assert reg.session.calls == [("/get-artifact", {"path": "package/thresholds.json",
                                                    "run_uuid": "ab34619a"})]


@pytest.mark.parametrize("answer", [
    FakeResponse(status=404, body={}), FakeResponse(body={"toxicity": 0.3}),
    FakeResponse(body={**dict.fromkeys(LABELS, 0.3), "toxicity": "high"}),
    requests.Timeout(),
])
def test_unusable_thresholds_are_none(answer):
    assert registry({"/get-artifact": answer}).thresholds("ab34619a") is None


def test_version_info_without_tags():
    info = version_info("staging", {"version": 2, "tags": [{"key": "gate_mean_auc",
                                                            "value": "n/a"}]})
    assert (info.version, info.gate_passed, info.gate_mean_auc, info.last_alias_move) == \
        ("2", None, None, None)
