"""Tests for scripts/validate_teacher_shards.py.

Uses the teacher lane's synthetic fixtures (Model-Pipeline/tests/fixtures/teacher/): 25 made-up
ids in shards of 10. A correct validator finds no errors in valid/ and exactly one in broken/
(shard_0001's manifest says row_count 11 for a 10-row Parquet). No network is used.
"""

import json

import pytest

from scripts import validate_teacher_shards as vts
from scripts.errors import TeacherShardError
from scripts.utils import PIPELINE_ROOT

MODEL_PIPELINE = PIPELINE_ROOT.parent / "Model-Pipeline"
FIXTURES = MODEL_PIPELINE / "tests" / "fixtures" / "teacher"

pytestmark = pytest.mark.skipif(
    not FIXTURES.exists(), reason="Model-Pipeline teacher fixtures not available"
)


def teacher_cfg(tmp_path, prefix, kinds=None, required=False):
    """A teacher config section that reads the fixtures from disk."""
    return {
        "teacher": {
            "model_pipeline_dir": str(MODEL_PIPELINE),
            "backend": None,
            "local_root": str(FIXTURES),
            "required": required,
            "report": str(tmp_path / "validation.json"),
            "plan": None,
            "shard_size": 10,
            "kinds": kinds or {"teacher": {"model_version": None}},
            "sources": {"fixture": str(FIXTURES / "source_ids.csv")},
            "runs": [{"prefix": prefix, "source": "fixture", "kind": "teacher"}],
        }
    }


def report(tmp_path):
    return json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))


# ---- the fixtures


def test_valid_fixtures_pass(tmp_path):
    summary = vts.run(teacher_cfg(tmp_path, "valid/"))
    result = summary["runs"]["valid/"]
    assert summary["skipped"] is False
    assert (result["shards"], result["rows"], result["errors"]) == (3, 25, 0)
    assert report(tmp_path)["runs"]["valid/"]["errors"] == 0


def test_broken_fixtures_fail_with_exactly_one_error(tmp_path):
    with pytest.raises(TeacherShardError):
        vts.run(teacher_cfg(tmp_path, "broken/"))
    result = report(tmp_path)["runs"]["broken/"]
    assert result["errors"] == 1
    assert "shard_0001" in result["first_errors"][0]


def test_wrong_fingerprint_fails(tmp_path):
    artifact = tmp_path / "stack.json"
    artifact.write_text('{"weights": [1, 2, 3]}\n', encoding="utf-8")
    kinds = {
        "teacher": {
            "model_version": None,
            "hash_field": "stack_sha256",
            "hash_file": str(artifact),
        }
    }
    with pytest.raises(TeacherShardError):
        vts.run(teacher_cfg(tmp_path, "valid/", kinds=kinds))
    result = report(tmp_path)["runs"]["valid/"]
    assert result["errors"] == 3  # the fixtures' three manifests carry no stack_sha256
    assert all("stack_sha256" in error for error in result["first_errors"])


# ---- the fingerprint check


def test_check_manifest_hash():
    manifests = {"shard_0000": {"stack_sha256": "abc"}, "shard_0001": {"stack_sha256": "xyz"}}
    errors = vts.check_manifest_hash(manifests, "stack_sha256", "abc")
    assert len(errors) == 1 and errors[0].startswith("shard_0001")


def test_file_sha256_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert vts.file_sha256(lf) == vts.file_sha256(crlf)


# ---- an unreachable store


def _unreachable(*args, **kwargs):
    raise OSError("no credentials")


def test_unreachable_store_skips(tmp_path, monkeypatch):
    storage = vts.teacher_modules(MODEL_PIPELINE)["storage"]
    monkeypatch.setattr(storage, "get_storage", _unreachable)
    summary = vts.run(teacher_cfg(tmp_path, "valid/"))
    assert summary["skipped"] is True
    assert "no credentials" in summary["skip_reason"]


def test_unreachable_store_fails_when_required(tmp_path, monkeypatch):
    storage = vts.teacher_modules(MODEL_PIPELINE)["storage"]
    monkeypatch.setattr(storage, "get_storage", _unreachable)
    with pytest.raises(TeacherShardError):
        vts.run(teacher_cfg(tmp_path, "valid/", required=True))
