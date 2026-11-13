"""Student config: ``configs/student.yaml`` plus ``--set key=value`` overrides.

Shares ``LABELS`` and the logger with the teacher lane, so both use one label order and one
log format. Relative paths resolve against ``Model-Pipeline/`` after overrides are applied,
so ``--set paths.raw_train=...`` behaves like the file.

``soft_labels.arm`` picks the teacher: after the overrides, the chosen arm's values in
``soft_labels.arms`` replace the stack's, so the rest of the code reads one flat
``soft_labels`` section whatever the arm (``--set soft_labels.arm=calqwen``).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.teacher.config import LABELS, PIPELINE_ROOT

DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "student.yaml"
SOURCES = ("temp", "ids", "official")
ARMS = ("stack", "calqwen", "tfidf")
FORMATS = ("shards", "oof")


def apply_overrides(cfg: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """``["loss.alpha=0.5", "train.epochs=1"]`` -> a copy of ``cfg`` with those values set.

    Values are parsed as YAML, so ``1``, ``1e-5``, ``true``, ``null``, and ``[1, 2]`` keep
    their types. Only existing keys can be set, so a typo fails instead of being ignored.
    """
    out = copy.deepcopy(dict(cfg))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        *parents, leaf = key.split(".")
        node = out
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"override {key!r}: {part!r} is not a section of the config")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"override {key!r}: unknown key {leaf!r}")
        node[leaf] = yaml.safe_load(raw)
    return out


def _resolve(value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    return str(p if p.is_absolute() else PIPELINE_ROOT / p)


def load_config(path: str | Path | None = None,
                overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load the student config, apply overrides, check labels, and resolve paths."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        cfg = apply_overrides(yaml.safe_load(f) or {}, overrides)
    if tuple(cfg.get("labels", ())) != LABELS:
        raise ValueError(f"config labels {cfg.get('labels')} must equal {list(LABELS)}")
    if cfg["model"]["name"] not in cfg["models"]:
        raise ValueError(f"model.name {cfg['model']['name']!r} is not one of "
                         f"{sorted(cfg['models'])}")
    if not 0.0 <= float(cfg["loss"]["alpha"]) <= 1.0:
        raise ValueError(f"loss.alpha must be in [0, 1], got {cfg['loss']['alpha']}")
    if float(cfg["loss"]["temperature"]) <= 0:
        raise ValueError(f"loss.temperature must be positive, got {cfg['loss']['temperature']}")
    if cfg["data"]["source"] not in SOURCES:
        raise ValueError(f"data.source must be one of {list(SOURCES)}, got "
                         f"{cfg['data']['source']!r}")
    cfg["soft_labels"] = resolve_arm(cfg["soft_labels"])
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = _resolve(value)
    for key in ("local_root", "check_file"):
        cfg["soft_labels"][key] = _resolve(cfg["soft_labels"].get(key))
    return cfg


def resolve_arm(soft_labels: Mapping[str, Any]) -> dict[str, Any]:
    """The flat ``soft_labels`` section for ``soft_labels.arm`` (``arms`` is kept as is)."""
    sl = copy.deepcopy(dict(soft_labels))
    arm = sl.get("arm", "stack")
    if arm not in ARMS:
        raise ValueError(f"soft_labels.arm must be one of {list(ARMS)}, got {arm!r}")
    arms = sl.get("arms") or {}
    for name, values in arms.items():   # every arm, so a typo fails before that arm is run
        if name not in ARMS or name == "stack":
            raise ValueError(f"soft_labels.arms.{name}: not an attribution arm "
                             f"({[a for a in ARMS if a != 'stack']})")
        unknown = sorted(set(values or {}) - set(sl) - {"arms"})
        if unknown:
            raise KeyError(f"soft_labels.arms.{name}: unknown keys {unknown}")
    if arm != "stack":
        if arm not in arms:
            raise ValueError(f"soft_labels.arms has no {arm!r} entry")
        sl.update(copy.deepcopy(arms[arm]))
    if sl.get("format", "shards") not in FORMATS:
        raise ValueError(f"soft_labels.format must be one of {list(FORMATS)}")
    return sl


def uses_soft_labels(cfg: Mapping[str, Any]) -> bool:
    """Gold-only runs (alpha = 1) never read the soft labels, so they need no HF token."""
    return float(cfg["loss"]["alpha"]) < 1.0
