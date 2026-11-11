import pytest

from src.student.config import (
    DEFAULT_CONFIG,
    PIPELINE_ROOT,
    apply_overrides,
    load_config,
    uses_soft_labels,
)


def test_default_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg["model"]["name"] == "distilbert"
    assert cfg["paths"]["stack_file"].startswith(str(PIPELINE_ROOT))
    assert not uses_soft_labels(cfg)   # the Phase 1 baseline is gold-only


def test_overrides_keep_yaml_types():
    cfg = apply_overrides({"loss": {"alpha": 1.0, "temperature": 1.0}, "train": {"max_steps": 5}},
                          ["loss.alpha=0.25", "train.max_steps=null"])
    assert cfg["loss"]["alpha"] == 0.25 and cfg["train"]["max_steps"] is None


def test_unknown_override_key_fails():
    with pytest.raises(KeyError, match="alhpa"):
        load_config(DEFAULT_CONFIG, ["loss.alhpa=0.5"])


def test_override_without_equals_fails():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({"a": 1}, ["a"])


@pytest.mark.parametrize("override, message", [
    ("loss.alpha=1.5", "alpha"),
    ("loss.temperature=0", "temperature"),
    ("model.name=gpt", "model.name"),
    ("data.source=tfds", "data.source"),
])
def test_bad_values_fail(override, message):
    with pytest.raises(ValueError, match=message):
        load_config(DEFAULT_CONFIG, [override])


def test_overridden_relative_path_resolves_against_model_pipeline():
    cfg = load_config(DEFAULT_CONFIG, ["paths.raw_train=data/other.parquet"])
    assert cfg["paths"]["raw_train"] == str(PIPELINE_ROOT / "data" / "other.parquet")


def test_mixed_run_uses_soft_labels():
    assert uses_soft_labels(load_config(DEFAULT_CONFIG, ["loss.alpha=0.5"]))
