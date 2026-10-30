import json

import pandas as pd
import pytest

from src.teacher.config import load_config
from src.teacher.shards import build_manifest, shard_frame
from src.teacher.storage import GCSStorage, get_storage, shard_done, shard_paths, with_retries
from src.teacher.validate_local import check_run, load_run
from tests.teacher.test_shards import scored


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def exists(self):
        return self.name in self.bucket.objects

    def upload_from_filename(self, filename):
        self.bucket.fail_next("upload")
        with open(filename, "rb") as f:
            self.bucket.objects[self.name] = f.read()

    def download_to_filename(self, filename):
        with open(filename, "wb") as f:
            f.write(self.bucket.objects[self.name])

    def delete(self):
        del self.bucket.objects[self.name]


class FakeBucket:
    def __init__(self, name):
        self.name, self.objects, self.failures = name, {}, {}

    def blob(self, name):
        return FakeBlob(self, name)

    def fail_next(self, op):
        if self.failures.get(op, 0):
            self.failures[op] -= 1
            raise ConnectionError(f"transient {op} failure")


class FakeClient:
    """The few ``google.cloud.storage.Client`` calls GCSStorage makes."""

    def __init__(self):
        self.buckets = {}

    def bucket(self, name):
        return self.buckets.setdefault(name, FakeBucket(name))

    def list_blobs(self, bucket, prefix=""):
        return [FakeBlob(self.buckets[bucket], n) for n in sorted(self.buckets[bucket].objects)
                if n.startswith(prefix)]


@pytest.fixture
def store():
    return GCSStorage("distillate-mlops-data", client=FakeClient(), backoff_s=0.0)


def test_round_trip(store, tmp_path):
    local = tmp_path / "x.parquet"
    local.write_bytes(b"data")
    store.upload(local, "teacher/v/shard_0000.parquet")
    assert store.exists("teacher/v/shard_0000.parquet")
    assert not store.exists("teacher/v/shard_0001.parquet")
    got = store.download("teacher/v/shard_0000.parquet", tmp_path / "cache")
    assert got == tmp_path / "cache" / "teacher/v/shard_0000.parquet"
    assert got.read_bytes() == b"data"
    store.delete("teacher/v/shard_0000.parquet")
    assert not store.exists("teacher/v/shard_0000.parquet")


def test_list_skips_folder_markers_and_other_prefixes(store, tmp_path):
    local = tmp_path / "x"
    local.write_bytes(b"")
    for name in ("teacher/.keep", "teacher/v/", "teacher/v/shard_0000.parquet",
                 "soft_labels/v/shard_0000.parquet"):
        store.upload(local, name)
    assert store.list("teacher/v/") == ["teacher/v/shard_0000.parquet"]
    assert store.list("teacher/") == ["teacher/.keep", "teacher/v/shard_0000.parquet"]


def test_upload_retries_transient_failures(store, tmp_path):
    local = tmp_path / "x"
    local.write_bytes(b"ok")
    store.bucket.failures["upload"] = 2
    store.upload(local, "teacher/v/a")
    assert store.bucket.objects["teacher/v/a"] == b"ok"


def test_with_retries_gives_up_after_the_last_attempt():
    calls, waits = [], []

    def always_fails():
        calls.append(1)
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_retries("op", always_fails, retries=3, backoff_s=1.0, sleep=waits.append)
    assert len(calls) == 3 and waits == [1.0, 2.0]          # exponential backoff


def test_shards_written_to_gcs_validate(store, tmp_path):
    from src.teacher.storage import write_shard

    prefix, ids = "teacher/p2-zs-v2b/", ["a", "b", "c"]
    frame = shard_frame(scored(ids), "p2-zs-v2b", "m@r")
    write_shard(store, prefix, "shard_0000", frame, build_manifest(frame, "p2-zs-v2b", "m@r"))
    assert shard_done(store, prefix, "shard_0000")
    manifest = json.loads(store.bucket.objects[shard_paths(prefix, "shard_0000")[1]])
    assert manifest["row_count"] == 3
    frames, manifests, schemas, errors = load_run(store, prefix, tmp_path / "cache")
    assert errors == [] and check_run(frames, manifests, schemas, ids, 10, "p2-zs-v2b",
                                      "m@r") == []
    assert pd.read_parquet(tmp_path / "cache" / shard_paths(prefix, "shard_0000")[0]).equals(frame)


def test_get_storage_backends(tmp_path, monkeypatch):
    cfg = load_config()
    assert cfg["storage"]["backend"] == "gcs"
    assert cfg["storage"]["gcs_bucket"] == "distillate-mlops-data"
    monkeypatch.setattr("src.teacher.storage.gcs_client", FakeClient)
    assert isinstance(get_storage(cfg), GCSStorage)
    assert type(get_storage(cfg, tmp_path)).__name__ == "LocalStorage"
    with pytest.raises(ValueError, match="gcs_bucket"):
        get_storage({"storage": {"backend": "gcs", "gcs_bucket": None}})
    with pytest.raises(ValueError, match="unknown"):
        get_storage({"storage": {"backend": "s3"}})


def test_gcs_client_prefers_the_key_from_the_environment(monkeypatch):
    from google.cloud import storage as gcs

    from src.teacher.storage import GCS_KEY_ENV, gcs_client

    seen = {}
    monkeypatch.setattr(gcs.Client, "from_service_account_info",
                        classmethod(lambda cls, info: seen.setdefault("info", info)))
    monkeypatch.setenv(GCS_KEY_ENV, json.dumps({"type": "service_account", "x": 1}))
    assert gcs_client() == {"type": "service_account", "x": 1}
