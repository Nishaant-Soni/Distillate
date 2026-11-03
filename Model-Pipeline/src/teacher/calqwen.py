"""Calibrated-Qwen soft labels: the ablation's second attribution arm.

The student's main targets are the stack's (``soft_labels.py``), which lean on TF-IDF. This
arm isolates the teacher: Qwen's own ``logit_<label>`` through the pilot's frozen Platt
calibration (``calibrate.py``, ``pilot_dir/calibration_<prompt_version>.json``), with no
TF-IDF input. Platt scaling is monotone per label, so it keeps raw Qwen's ranking and only
fixes its calibration.

No row is labeled by a calibration that saw its gold label: the 2,000 pilot rows, which the
calibration was fitted on, get five-fold cross-fitted values (as their stack labels do);
every other training row and the gate subset get the frozen calibration.

Shards have the same plan, columns, and layout as the stack's soft labels, under
``soft_labels/<calqwen.version>/<prompt_version>/`` (plus ``gate/``); ``logit_<label>`` is the
calibrated log-odds. Manifests record the calibration file and its SHA-256.

    python -m src.teacher.calqwen            # write artifacts/calqwen/calqwen_check.json
    python -m src.teacher.calqwen --upload   # also write the shards

``--upload`` requires the code, ``artifacts/calqwen/`` and ``artifacts/pilot/`` to be
committed, and every manifest records the commit.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger, load_config

logger = get_logger(__name__)


def calqwen_model_version(version: str, calibration: Mapping[str, Any]) -> str:
    """``model_version`` of calibrated-Qwen shards: this arm's version and the teacher's."""
    return f"{version}+{calibration['model_version']}"


def calibrated_frame(teacher, scores: Mapping[str, Any]):
    """``id``, ``p_<label>``, ``logit_<label>``, ``status`` from calibrated log-odds."""
    import pandas as pd
    from scipy.special import expit

    out = pd.DataFrame({"id": teacher["id"].to_numpy()})
    for label in LABELS:
        out[f"p_{label}"] = expit(scores[label]).astype("float32")
    for label in LABELS:
        out[f"logit_{label}"] = scores[label].astype("float32")
    out["status"] = teacher["status"].to_numpy()
    return out


def run(cfg: Mapping[str, Any], teacher_storage, out_storage=None,
        code_commit: str | None = None,
        before_upload: Callable[[], None] | None = None) -> dict[str, Any]:
    """Write ``calqwen_check.json``, and the shards if ``out_storage`` is given.

    ``before_upload`` runs after the report is written and before any shard is, e.g. to check
    that the files just written match the committed ones.
    """
    import pandas as pd

    from src.teacher.calibrate import cross_fit_log_odds, load_calibration, log_odds
    from src.teacher.metrics import evaluate
    from src.teacher.shards import check_plan, plan_shards
    from src.teacher.soft_labels import (
        load_teacher,
        sha256_file,
        shard_frames,
        soft_prefix,
        table,
        upload_shards,
    )

    pv, bins, paths = cfg["shift"]["prompt_version"], cfg["metrics"]["ece_bins"], cfg["paths"]
    version, data_dir = cfg["calqwen"]["version"], Path(paths["data_dir"])
    teacher_prefix = f"teacher/{pv}/"

    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    test = pd.read_parquet(data_dir / cfg["data"]["files"]["test"])
    for frame in (train, test):
        frame["id"] = frame["id"].astype(str)
    gate_ids = pd.read_parquet(paths["gate_ids"])
    gate_ids["id"] = gate_ids["id"].astype(str)
    pilot_ids = pd.read_parquet(Path(paths["pilot_dir"]) / "pilot_ids.parquet")
    pilot_ids["id"] = pilot_ids["id"].astype(str)
    train_truth = train[["id", *LABELS]]
    gate_truth = test.loc[test["id"].isin(set(gate_ids["id"])), ["id", *LABELS]]

    teacher = load_teacher(teacher_storage, teacher_prefix)
    gate_teacher = load_teacher(teacher_storage, f"{teacher_prefix}gate/")
    if set(teacher["id"]) != set(train["id"]) or set(gate_teacher["id"]) != set(gate_ids["id"]):
        raise ValueError("teacher shards do not cover the training set and gate subset")
    if not set(pilot_ids["id"]) <= set(train["id"]):
        raise ValueError("some pilot ids are not training rows")

    cal_path = Path(paths["pilot_dir"]) / f"calibration_{pv}.json"
    calibration, cal_sha = load_calibration(cal_path), sha256_file(cal_path)

    pilot = teacher["id"].isin(set(pilot_ids["id"]))
    pilot_truth = train_truth[train_truth["id"].isin(set(pilot_ids["id"]))]
    rest = calibrated_frame(teacher[~pilot], log_odds(teacher[~pilot], calibration))
    pilot_part = calibrated_frame(teacher[pilot], cross_fit_log_odds(
        teacher[pilot], pilot_truth, pilot_ids, cfg["calibration"]["folds"], cfg["seed"]))
    train_cal = pd.concat([rest, pilot_part], ignore_index=True)
    gate_cal = calibrated_frame(gate_teacher, log_odds(gate_teacher, calibration))
    cross_fitted = (f"pilot rows only (the calibration was fitted on them): "
                    f"{cfg['calibration']['folds']} folds (seed {cfg['seed']})")

    model_version = calqwen_model_version(version, calibration)
    plan = plan_shards(train["id"].tolist(), cfg["labeling"]["shard_size"])
    check_plan(plan, json.loads(Path(paths["shard_plan"]).read_text()))
    train_frames = shard_frames(train_cal, plan, pv, model_version)
    gate_frames = shard_frames(gate_cal, plan_shards(gate_ids["id"].tolist(),
                                                     cfg["labeling"]["shard_size"]),
                               pv, model_version)

    # Metrics from the exact frames that get uploaded (float32), to compare after upload.
    train_out = pd.concat(train_frames)
    non_pilot = ~train_truth["id"].isin(set(pilot_ids["id"]))
    report: dict[str, Any] = {
        "calqwen_version": version, "model_version": model_version, "prompt_version": pv,
        "calibration_file": cal_path.resolve().relative_to(PIPELINE_ROOT).as_posix(),
        "calibration_sha256": cal_sha, "calibration": calibration,
        "train_cross_fitted": cross_fitted,
        "train": table(evaluate(train_out, train_truth, None, bins)),
        "train_non_pilot": table(evaluate(train_out, train_truth[non_pilot], None, bins)),
        "train_teacher_raw": table(evaluate(teacher, train_truth, None, bins)),
        "gate_subset": table(evaluate(pd.concat(gate_frames), gate_truth, gate_ids, bins)),
        "gate_subset_teacher_raw": table(evaluate(gate_teacher, gate_truth, gate_ids, bins)),
        "note": "gate subset weighted by its inclusion weights; training rows unweighted",
    }
    out_dir = Path(paths["calqwen_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calqwen_check.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("calibrated Qwen: train %s (raw %s), gate %s (raw %s) mean AUC; ECE train %s, "
                "gate %s", report["train"]["mean_auc"], report["train_teacher_raw"]["mean_auc"],
                report["gate_subset"]["mean_auc"],
                report["gate_subset_teacher_raw"]["mean_auc"], report["train"]["ece"],
                report["gate_subset"]["ece"])

    if out_storage is not None:
        if before_upload is not None:
            before_upload()
        prefix = soft_prefix(version, pv)
        extra = {"calqwen_version": version,
                 "calibration_version": calibration["calibration_version"],
                 "calibration_file": report["calibration_file"],
                 "teacher_prefix": teacher_prefix, "code_commit": code_commit}
        report["upload"] = {
            "prefix": prefix,
            "train": upload_shards(out_storage, prefix, train_frames, pv, model_version,
                                   cal_sha, {**extra, "cross_fitted": cross_fitted},
                                   sha_field="calibration_sha256"),
            "gate": upload_shards(out_storage, f"{prefix}gate/", gate_frames, pv,
                                  model_version, cal_sha,
                                  {**extra, "cross_fitted": "no (held-out rows)",
                                   "teacher_prefix": f"{teacher_prefix}gate/"},
                                  sha_field="calibration_sha256"),
        }
    return report


def main() -> None:
    from src.teacher.soft_labels import git_state
    from src.teacher.storage import get_storage

    parser = argparse.ArgumentParser(description="Calibrated-Qwen soft-label shards.")
    parser.add_argument("--upload", action="store_true", help="write the shards")
    parser.add_argument("--teacher-root", default=None,
                        help="local copy of the teacher shards (default: the configured store)")
    parser.add_argument("--local-root", default=None, help="write shards here, not the store")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tracked = ["src", "configs", cfg["paths"]["calqwen_dir"], cfg["paths"]["pilot_dir"]]
    commit, dirty = git_state(tracked)
    teacher_storage = get_storage(cfg, args.teacher_root)

    def still_committed() -> None:
        if git_state(tracked)[1] and args.local_root is None:
            raise SystemExit("artifacts/calqwen/ as rewritten by this run differs from the "
                             "committed files; nothing was uploaded")

    out_storage = None
    if args.upload:
        if dirty and args.local_root is None:
            raise SystemExit("commit src/, configs/, artifacts/calqwen/ and artifacts/pilot/ "
                             "before --upload")
        out_storage = get_storage(cfg, args.local_root)
    report = run(cfg, teacher_storage, out_storage, commit,
                 still_committed if args.upload else None)
    print(json.dumps({k: report[k]["mean_auc"] for k in
                      ("train", "train_non_pilot", "train_teacher_raw", "gate_subset",
                       "gate_subset_teacher_raw")} | {"upload": report.get("upload")},
                     indent=2))


if __name__ == "__main__":
    main()
