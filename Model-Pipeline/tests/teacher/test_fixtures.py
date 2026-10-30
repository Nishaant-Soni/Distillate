import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.teacher.config import PIPELINE_ROOT
from src.teacher.shards import read_ids
from src.teacher.storage import LocalStorage
from src.teacher.validate_local import check_run, load_run

FIXTURES = PIPELINE_ROOT / "tests" / "fixtures" / "teacher"
_spec = importlib.util.spec_from_file_location("make_fixtures", FIXTURES / "make_fixtures.py")
assert _spec is not None and _spec.loader is not None
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def validate(prefix, tmp_path):
    frames, manifests, schemas, errors = load_run(LocalStorage(FIXTURES), prefix, tmp_path)
    return errors + check_run(frames, manifests, schemas, read_ids(FIXTURES / "source_ids.csv"),
                              fx.SHARD_SIZE, fx.PROMPT_VERSION, fx.MODEL_VERSION)


def test_valid_run_passes(tmp_path):
    assert validate("valid/", tmp_path) == []


def test_broken_run_fails_only_on_the_row_count(tmp_path):
    assert validate("broken/", tmp_path) == [
        f"{fx.BROKEN_SHARD}: 10 rows, manifest says {fx.BROKEN_ROW_COUNT}"]


@pytest.mark.parametrize("run", ["valid", "broken"])
def test_fixtures_match_the_generator(tmp_path, run):
    fx.write(tmp_path)
    committed, fresh = sorted((FIXTURES / run).iterdir()), sorted((tmp_path / run).iterdir())
    assert [p.name for p in committed] == [p.name for p in fresh]
    for a, b in zip(committed, fresh, strict=True):
        if a.suffix == ".parquet":
            pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
        else:
            assert json.loads(a.read_text()) == json.loads(b.read_text()), a.name
    assert (FIXTURES / "source_ids.csv").read_text() == (tmp_path / "source_ids.csv").read_text()


def test_fixtures_hold_no_real_ids():
    real = PIPELINE_ROOT / "artifacts" / "teacher_shards.json"
    plan = json.loads(Path(real).read_text())
    ids = read_ids(FIXTURES / "source_ids.csv")
    assert all(i.startswith("f1x") for i in ids)
    assert not {s["first_id"] for s in plan["shards"]} & set(ids)
