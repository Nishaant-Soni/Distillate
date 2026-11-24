"""The serving API client: the contract check, the live backend's errors, and the mock."""

import pytest
import requests

from demo_app.api import LABELS, APIError, LiveAPI, MockAPI, make_api, validate
from demo_app.config import load


def good_response():
    return {"scores": dict.fromkeys(LABELS, 0.2), "flagged": dict.fromkeys(LABELS, False),
            "flag": False, "low_confidence": False, "model_version": "v1", "latency_ms": 12.5}


class FakeResponse:
    def __init__(self, status=200, body=None, json_error=False):
        self.status_code, self.body, self.json_error = status, body, json_error

    def json(self):
        if self.json_error:
            raise ValueError("not json")
        return self.body


class FakeSession:
    """Records the requests; answers with ``response``, or raises ``error``."""

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_a_contract_response_passes():
    body = good_response()
    assert validate(body) is body


@pytest.mark.parametrize("change, problem", [
    (lambda b: b.pop("flag"), "flag"),
    (lambda b: b["scores"].pop("threat"), "scores.threat"),
    (lambda b: b["scores"].update(insult=1.5), "scores.insult"),
    (lambda b: b["scores"].update(insult=True), "scores.insult"),
    (lambda b: b["scores"].update(insult=float("nan")), "scores.insult"),
    (lambda b: b["flagged"].update(obscene="yes"), "flagged.obscene"),
    (lambda b: b.update(latency_ms="fast"), "latency_ms"),
    (lambda b: b.update(model_version=3), "model_version"),
])
def test_an_off_contract_response_is_refused(change, problem):
    body = good_response()
    change(body)
    with pytest.raises(APIError, match=problem):
        validate(body)


def test_live_predict_posts_the_text_and_validates():
    session = FakeSession(FakeResponse(body=good_response()))
    api = LiveAPI("http://svc/", session=session)
    assert api.predict("hello")["model_version"] == "v1"
    assert session.calls == [("POST", "http://svc/predict", {"json": {"text": "hello"}})]


@pytest.mark.parametrize("session, message", [
    (FakeSession(error=requests.ConnectionError()), "unreachable"),
    (FakeSession(FakeResponse(status=429)), "rate limiting"),
    (FakeSession(FakeResponse(status=503)), "HTTP 503"),
    (FakeSession(FakeResponse(json_error=True)), "isn't JSON"),
    (FakeSession(FakeResponse(body={"scores": {}})), "off contract"),
])
def test_live_errors_become_api_errors(session, message):
    with pytest.raises(APIError, match=message):
        LiveAPI("http://svc", session=session).predict("hello")


def test_live_health():
    session = FakeSession(FakeResponse(body={"status": "ok", "model_version": "v1"}))
    assert LiveAPI("http://svc", session=session).health()["model_version"] == "v1"
    with pytest.raises(APIError, match="health"):
        LiveAPI("http://svc", session=FakeSession(FakeResponse(body=[]))).health()


def test_the_mock_follows_the_contract_and_is_deterministic():
    mock = MockAPI()
    first = mock.predict("some comment")
    assert validate(first) is first
    assert first == mock.predict("some comment")
    assert first["model_version"] == "mock"
    assert first["flag"] == any(first["flagged"].values())
    assert first["flagged"] == {k: first["scores"][k] >= 0.5 for k in LABELS}
    assert mock.predict("another comment") != first


def test_make_api_picks_the_mock_unless_a_url_is_set():
    assert isinstance(make_api(load({})), MockAPI)
    assert isinstance(make_api(load({"DISTILLATE_API_URL": "http://svc"})), LiveAPI)
