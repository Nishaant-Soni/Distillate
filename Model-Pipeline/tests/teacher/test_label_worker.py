import json

import pandas as pd
import pytest

from src.teacher.label_worker import run_worker, select_rows
from src.teacher.prompts import PromptBuilder
from src.teacher.scoring import Score
from src.teacher.shards import plan_shards
from src.teacher.storage import LocalStorage, shard_paths
from src.teacher.validate_local import check_run, load_run

PREFIX, MV = "teacher/p2-zs-v2b-dryrun/", "Qwen/Qwen3-4B-Instruct-2507@abc"
IDS = [f"{i:04x}" for i in range(25)]


@pytest.fixture
def rows():
    return pd.DataFrame({"id": IDS, "text": [f"comment number {i}" for i in range(25)]})


@pytest.fixture
def builder(tokenizer):
    return PromptBuilder(tokenizer, "p2", "zs", text_version="v2b")


class FakeScorer:
    """Counts prompts; every answer is a confident, finite "no"."""

    def __init__(self):
        self.prompts = 0

    def __call__(self, prompts):
        self.prompts += len(prompts)
        return [Score(0.01, 0.99, -4.6) for _ in prompts]


class Clock:
    def __init__(self, step=10.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def run(rows, builder, storage, worker, num_workers=2, scorer=None, deadline=1e9, clock=None):
    return run_worker(rows, plan_shards(rows["id"].tolist(), 10), worker, num_workers, storage,
                      PREFIX, builder, scorer or FakeScorer(), MV, deadline=deadline,
                      clock=clock or Clock(), extra_manifest={"code_commit": "abc"},
                      log=lambda line: None)


def validate(tmp_path, storage, **kwargs):
    frames, manifests, schemas, errors = load_run(storage, PREFIX, tmp_path / "cache")
    return errors + check_run(frames, manifests, schemas, IDS, 10, "p2-zs-v2b", MV, **kwargs)


def test_two_workers_cover_the_run(tmp_path, rows, builder):
    storage = LocalStorage(tmp_path / "store")
    w0 = run(rows, builder, storage, 0)
    assert w0["written"] == ["shard_0000", "shard_0002"]
    assert validate(tmp_path, storage, partial=True) == []
    w1 = run(rows, builder, storage, 1)
    assert w1["written"] == ["shard_0001"] and w1["rows"] == 10
    assert validate(tmp_path, storage) == []
    manifest = json.loads((tmp_path / "store" / shard_paths(PREFIX, "shard_0002")[1])
                          .read_text())
    assert manifest["row_count"] == 5 and manifest["code_commit"] == "abc"
    assert manifest["worker"] == 0 and manifest["failed_rows"] == 0


def test_rerun_skips_finished_shards(tmp_path, rows, builder):
    storage = LocalStorage(tmp_path / "store")
    run(rows, builder, storage, 0, num_workers=1)
    scorer = FakeScorer()
    again = run(rows, builder, storage, 0, num_workers=1, scorer=scorer)
    assert again["written"] == [] and len(again["skipped"]) == 3 and scorer.prompts == 0


def test_interrupted_shard_is_redone(tmp_path, rows, builder):
    storage = LocalStorage(tmp_path / "store")
    run(rows, builder, storage, 0, num_workers=1)
    # A crash between the two uploads: Parquet present, manifest missing.
    (tmp_path / "store" / shard_paths(PREFIX, "shard_0001")[1]).unlink()
    scorer = FakeScorer()
    again = run(rows, builder, storage, 0, num_workers=1, scorer=scorer)
    assert again["written"] == ["shard_0001"] and scorer.prompts == 10 * 6
    assert validate(tmp_path, storage) == []


def test_stops_before_the_deadline(tmp_path, rows, builder):
    storage = LocalStorage(tmp_path / "store")
    # Each clock read advances 50 s. First check at t=50: 50 + 10/4 + 300 = 352.5 < 400, so
    # shard 0 runs and measures 50 s for 10 rows (0.2 rows/s). Next check at t=200:
    # 200 + 50 + 300 = 550 > 400, so it stops before shard 1.
    summary = run(rows, builder, storage, 0, num_workers=1, deadline=400.0,
                  clock=Clock(step=50.0))
    assert summary["written"] == ["shard_0000"] and summary["stopped_before"] == "shard_0001"
    assert validate(tmp_path, storage, partial=True) == []


def test_select_rows(tmp_path):
    data = pd.DataFrame({"id": ["c", "a", "b"], "text": ["x", "y", "z"], "toxicity": [0, 1, 0]})
    data.to_parquet(tmp_path / "data.parquet")
    assert select_rows(tmp_path / "data.parquet")["id"].tolist() == ["a", "b", "c"]
    assert select_rows(tmp_path / "data.parquet", limit=2)["id"].tolist() == ["a", "b"]
    pd.DataFrame({"id": ["c", "a"]}).to_csv(tmp_path / "ids.csv", index=False)
    kept = select_rows(tmp_path / "data.parquet", tmp_path / "ids.csv")
    assert kept["id"].tolist() == ["a", "c"] and list(kept.columns) == ["id", "text"]
    pd.DataFrame({"id": ["q"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="not in"):
        select_rows(tmp_path / "data.parquet", tmp_path / "bad.csv")
