"""Dispatch behaviour: real module when merged, contract-shaped stub when not."""

import json
import sys
import types

import pytest

from scripts import stage


@pytest.fixture
def cfg(tmp_path):
    """A config pointing every path at a temp dir, shaped like pipeline.yaml."""
    from scripts.utils import load_config

    cfg = load_config()
    for key in ("raw_dir", "processed_dir", "splits_dir", "schema_dir",
                "reports_dir", "summaries_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["stubs"] = {"rows": 40}
    return cfg


def _install_fake(monkeypatch, name, run):
    module = types.ModuleType(name)
    if run is not None:
        module.run = run
    monkeypatch.setitem(sys.modules, name, module)


def test_unknown_stage_is_rejected(cfg):
    with pytest.raises(KeyError):
        stage.dispatch("not_a_stage", cfg)


def test_stage_ids_match_the_cli_module_names():
    for name, module in stage.STAGE_MODULES.items():
        assert module == f"scripts.{name}", "task id must map to python -m scripts.<name>"


def test_real_module_is_used_once_merged(cfg, monkeypatch):
    _install_fake(monkeypatch, "scripts.dedupe",
                  lambda c: {"rows_in": 10, "rows_out": 9, "dropped": {"exact": 1}})
    summary = stage.dispatch("dedupe", cfg)
    assert summary["mode"] == "real"
    assert summary["rows_out"] == 9
    assert summary["task"] == "dedupe"


def test_real_module_must_return_a_dict(cfg, monkeypatch):
    _install_fake(monkeypatch, "scripts.bias", lambda c: "done")
    with pytest.raises(TypeError, match="must return a dict"):
        stage.dispatch("bias", cfg)


def test_module_without_run_falls_back_to_the_stub(cfg, monkeypatch):
    pytest.importorskip("pandas")
    _install_fake(monkeypatch, "scripts.split", None)
    assert stage.dispatch("split", cfg)["mode"] == "stub"


def test_a_broken_import_inside_a_real_module_is_not_hidden(cfg, monkeypatch):
    """A missing third-party package must fail loudly, not look like 'not merged'."""
    def explode(name):
        raise ModuleNotFoundError("No module named 'datasketch'", name="datasketch")

    monkeypatch.setattr(stage.importlib, "import_module", explode)
    with pytest.raises(ModuleNotFoundError):
        stage.dispatch("dedupe", cfg)


def test_every_stage_has_a_working_stub(cfg):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    for name in stage.STAGE_MODULES:
        summary = stage.dispatch(name, cfg)
        assert summary["mode"] == "stub", name
        assert "duration_s" in summary, name


def test_stub_writes_one_summary_per_task(cfg):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    stage.dispatch("preprocess", cfg)
    written = json.loads((__import__("pathlib").Path(cfg["paths"]["summaries_dir"])
                          / "preprocess.json").read_text())
    assert written["mode"] == "stub"
    assert set(written["parts"]) == {"preprocess_wikipedia", "preprocess_civil"}


def test_stub_output_matches_the_contract_columns(cfg):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from pathlib import Path

    stage.dispatch("split", cfg)
    train = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "wikipedia" / "train.parquet")
    labels = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
    assert list(train.columns) == ["id", "text", *labels, "char_len",
                                   "has_identity_term", "has_obfuscation", "sample_weight"]
    assert all(str(train[c].dtype) == "int8" for c in labels)
    assert str(train["char_len"].dtype) == "int32"
    assert str(train["sample_weight"].dtype) == "float32"
    assert train["id"].is_unique


def test_civil_stub_is_sorted_by_created_date(cfg):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from pathlib import Path

    stage.dispatch("subsets", cfg)
    drift = Path(cfg["paths"]["processed_dir"]) / "civil" / "drift"
    batch = pd.read_parquet(drift / "batch_00.parquet")
    assert batch["created_date"].is_monotonic_increasing
    # CONTRACTS.md fixes the timezone, not the resolution: pandas 3.x in the main
    # environment writes microseconds and pandas 1.5.3 in /opt/tfenv reads
    # nanoseconds, so only tz-awareness is portable across the two.
    assert batch["created_date"].dt.tz is not None
    assert str(batch["created_date"].dt.tz) == "UTC"


def test_stub_never_overwrites_real_output(cfg):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from pathlib import Path

    target = Path(cfg["paths"]["processed_dir"]) / "obfuscation.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["real_1"], "text": ["produced by the real stage"]}).to_parquet(target)
    stage.dispatch("obfuscate", cfg)
    assert pd.read_parquet(target)["id"].tolist() == ["real_1"]


def test_stubs_are_deterministic(cfg, tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from pathlib import Path

    from scripts import stubs

    first = stubs.wikipedia_frame(20, cfg["seed"], "wiki_train")
    second = stubs.wikipedia_frame(20, cfg["seed"], "wiki_train")
    pd.testing.assert_frame_equal(first, second)
    assert Path  # keeps the import used if the assertions above change
