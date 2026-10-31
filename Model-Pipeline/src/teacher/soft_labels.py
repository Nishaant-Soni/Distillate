"""The final stack and the soft-label shards the student trains on.

The raw teacher shards rank well but are poorly calibrated, so the student never trains on
their ``p_*``. It trains on the stack's output, written here as shards with the same plan,
layout, and columns as the teacher shards, under ``soft_labels/<stack_version>/<prompt>/``.

Two candidate stacks, same model (``stack.py``):
* A, ``stack.version``: fitted on the 2,000 pilot rows with pilot weights (``tfidf_oof.py``);
* B, ``stack.full_version``: fitted on all training rows, unweighted (they are the population).

Both are compared on rows neither was fitted on: the gate subset (weighted by its inclusion
weights) and the Civil Comments shift sample. B is adopted unless its mean AUC is
significantly lower than A's on either (95% bootstrap interval entirely below 0), or its
gate-subset ECE is worse by more than ``stack.max_ece_increase``.

Training rows get cross-fitted soft labels, so no row's label comes from a stack that saw its
gold label: with B, each ``tfidf_oof`` fold is scored by a stack fitted on the other folds;
with A, non-pilot rows get A's output and pilot rows the cross-fitted pilot outputs. Gate
rows get the adopted stack as frozen.

The drift stream (``--drift``) gets the adopted stack as frozen, like the gate subset, with
one shard per delivered batch under ``.../drift/``; nothing is refitted or re-chosen.

    python -m src.teacher.soft_labels            # choose the stack, write artifacts/stack/
    python -m src.teacher.soft_labels --upload   # also write the soft-label shards
    python -m src.teacher.soft_labels --drift [--upload]

``--upload`` requires the code, ``artifacts/stack/`` and ``artifacts/oof/`` to be committed,
and every manifest records the commit and the stack file's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger, load_config

logger = get_logger(__name__)


def soft_model_version(stack: Mapping[str, Any]) -> str:
    """``model_version`` of soft-label shards: the stack and both of its inputs."""
    return f"{stack['stack_version']}+{stack['model_version']}+{stack['baseline_version']}"


def soft_prefix(stack_version: str, prompt_version: str) -> str:
    return f"soft_labels/{stack_version}/{prompt_version}/"


def unit_weights(ids: Sequence[str]):
    import pandas as pd

    return pd.DataFrame({"id": list(ids), "weight": 1.0})


def cross_fit_by_folds(teacher, baseline, truth, version: str, C: float):
    """Stacked outputs for every row, each fold scored by a stack fitted on the other folds.

    ``baseline`` carries the ``fold`` column (``tfidf_oof.parquet``), so the stack's folds are
    the folds TF-IDF's scores were made with.
    """
    import pandas as pd

    from src.teacher.stack import apply_stack, fit_stack

    meta = teacher[["prompt_version", "model_version"]].iloc[0]
    baseline_version = str(baseline["model_version"].iloc[0])
    parts = []
    for k in sorted(baseline["fold"].unique()):
        held = set(baseline.loc[baseline["fold"] == k, "id"])
        fit_rows = teacher[~teacher["id"].isin(held)]
        stack = fit_stack(fit_rows, baseline, truth, unit_weights(fit_rows["id"]), version,
                          meta["prompt_version"], meta["model_version"], baseline_version, C)
        parts.append(apply_stack(teacher[teacher["id"].isin(held)], baseline, stack))
    out = pd.concat(parts, ignore_index=True)
    if len(out) != len(teacher) or not out["id"].is_unique:
        raise ValueError("every teacher row must be in exactly one fold")
    return out.sort_values("id", kind="stable").reset_index(drop=True)


def choose(bootstrap: Mapping[str, Mapping[str, Any]], gate_ece: Mapping[str, float],
           max_ece_increase: float) -> dict[str, Any]:
    """``{"adopt": "B" or "A", "reasons": [...]}`` from the B - A comparisons."""
    reasons = []
    for domain, result in bootstrap.items():
        if result["mean_auc_ci95"][1] < 0:
            reasons.append(f"{domain}: B's mean AUC is significantly lower than A's")
    if gate_ece["B"] - gate_ece["A"] > max_ece_increase:
        reasons.append(f"gate subset: B's ECE is worse by {gate_ece['B'] - gate_ece['A']:.4f}")
    return {"adopt": "A" if reasons else "B", "reasons": reasons}


def shard_frames(pred, plan: Sequence[Sequence[str]], prompt_version: str, model_version: str):
    """``pred`` cut into the shard plan, as spec shard frames (``shard_frame`` types)."""
    from src.teacher.shards import shard_frame

    by_id = pred.set_index("id")
    frames = []
    for ids in plan:
        part = by_id.loc[list(ids)].reset_index()
        frames.append(shard_frame(part, prompt_version, model_version))
    if sum(len(f) for f in frames) != len(pred):
        raise ValueError("the plan does not cover every prediction")
    return frames


def upload_shards(storage, prefix: str, frames: Sequence[Any], prompt_version: str,
                  model_version: str, stack_sha256: str, extra: Mapping[str, Any],
                  log: Callable[[str], None] = logger.info) -> dict[str, list[str]]:
    """Write each shard (Parquet, then manifest), skipping shards this stack already wrote.

    Raises if a finished shard under ``prefix`` came from a different stack file.
    """
    from src.teacher.shards import build_manifest, shard_name
    from src.teacher.storage import shard_done, shard_paths, write_shard

    summary: dict[str, list[str]] = {"written": [], "skipped": []}
    for k, frame in enumerate(frames):
        name = shard_name(k)
        if shard_done(storage, prefix, name):
            with tempfile.TemporaryDirectory() as tmp:
                local = storage.download(shard_paths(prefix, name)[1], Path(tmp))
                previous = json.loads(Path(local).read_text()).get("stack_sha256")
            if previous != stack_sha256:
                raise ValueError(f"{prefix}{name} was written by another stack "
                                 f"({previous}); remove that prefix first")
            summary["skipped"].append(name)
            continue
        manifest = build_manifest(frame, prompt_version, model_version,
                                  stack_sha256=stack_sha256, **extra)
        write_shard(storage, prefix, name, frame, manifest)
        summary["written"].append(name)
        log(f"{prefix}{name}: {len(frame)} rows")
    return summary


def git_state(paths: Sequence[str]) -> tuple[str, bool]:
    """``(HEAD commit, whether any of paths has uncommitted changes)``."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT,
                                   text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *paths],
                                     cwd=PIPELINE_ROOT, text=True).strip()
    return head, bool(status)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_teacher(storage, prefix: str):
    """All shards directly under ``prefix`` as one frame, sorted by id."""
    import pandas as pd

    from src.teacher.validate_local import load_run

    with tempfile.TemporaryDirectory() as tmp:
        frames, manifests, _, errors = load_run(storage, prefix, Path(tmp))
    if errors or set(frames) != set(manifests) or not frames:
        raise ValueError(f"{prefix}: incomplete or unexpected files {errors}")
    out = pd.concat([frames[name] for name in sorted(frames)], ignore_index=True)
    out["id"] = out["id"].astype(str)
    return out.sort_values("id", kind="stable").reset_index(drop=True)


def tfidf_oof(cfg: Mapping[str, Any], storage):
    """``tfidf_oof.parquet``, downloaded if absent, checked against its manifest."""
    import pandas as pd

    oof_dir = Path(cfg["paths"]["oof_dir"])
    manifest = json.loads((oof_dir / "tfidf_oof.manifest.json").read_text())
    local = oof_dir / manifest["file"]
    if not local.exists():
        with tempfile.TemporaryDirectory() as tmp:
            local.write_bytes(Path(storage.download(manifest["hf_path"], Path(tmp))).read_bytes())
    if sha256_file(local) != manifest["sha256"]:
        raise ValueError(f"{local} does not match its manifest's SHA-256")
    frame = pd.read_parquet(local)
    frame["id"] = frame["id"].astype(str)
    return frame


def full_tfidf(cfg: Mapping[str, Any], train, rows, name: str):
    """Full-train TF-IDF scores for held-out ``rows``, computed once (a few minutes) and
    cached as ``<oof_dir>/<name>_tfidf_full.parquet`` (ids and scores only)."""
    import pandas as pd

    from src.teacher.baseline_tfidf import train_predict

    path = Path(cfg["paths"]["oof_dir"]) / f"{name}_tfidf_full.parquet"
    if not path.exists():
        train_predict(train, rows[["id", "text"]], cfg["baseline"],
                      cfg["seed"]).to_parquet(path, index=False)
    frame = pd.read_parquet(path)
    frame["id"] = frame["id"].astype(str)
    if set(frame["id"]) != set(rows["id"]):
        raise ValueError(f"{path} does not cover the {name} rows; delete it to recompute")
    return frame


def table(t) -> dict[str, Any]:
    """The mean row and per-label AUC of a metrics table, rounded, for the JSON report."""
    import pandas as pd

    return {"mean_auc": round(float(t.loc["mean", "auc"]), 5),
            "mean_pr_auc": round(float(t.loc["mean", "pr_auc"]), 5),
            "ece": round(float(t.loc["mean", "ece"]), 5),
            "brier": round(float(t.loc["mean", "brier"]), 5),
            "auc": {k: None if pd.isna(t.loc[k, "auc"]) else round(float(t.loc[k, "auc"]), 5)
                    for k in LABELS}}


def run(cfg: Mapping[str, Any], teacher_storage, out_storage=None, n_boot: int = 1000,
        code_commit: str | None = None,
        before_upload: Callable[[], None] | None = None) -> dict[str, Any]:
    """Choose the stack, write ``artifacts/stack/``, and upload if ``out_storage`` is given.

    ``before_upload`` runs after ``artifacts/stack/`` is written and before any shard is,
    e.g. to check that the files just written match the committed ones.
    """
    import pandas as pd

    from src.teacher.metrics import evaluate, paired_bootstrap
    from src.teacher.shards import check_plan, plan_shards
    from src.teacher.shift import download_civil, load_civil, shift_rows
    from src.teacher.stack import apply_stack, cross_fit_stack, fit_stack, load_stack, save_stack

    pv, bins, st = cfg["shift"]["prompt_version"], cfg["metrics"]["ece_bins"], cfg["stack"]
    paths, data_dir = cfg["paths"], Path(cfg["paths"]["data_dir"])
    teacher_prefix = f"teacher/{pv}/"

    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    test = pd.read_parquet(data_dir / cfg["data"]["files"]["test"])
    for frame in (train, test):
        frame["id"] = frame["id"].astype(str)
    gate_ids = pd.read_parquet(paths["gate_ids"])
    gate_ids["id"] = gate_ids["id"].astype(str)
    gate_rows = test[test["id"].isin(set(gate_ids["id"]))]
    train_truth, gate_truth = train[["id", *LABELS]], gate_rows[["id", *LABELS]]

    teacher = load_teacher(teacher_storage, teacher_prefix)
    gate_teacher = load_teacher(teacher_storage, f"{teacher_prefix}gate/")
    if set(teacher["id"]) != set(train["id"]) or set(gate_teacher["id"]) != set(gate_ids["id"]):
        raise ValueError("teacher shards do not cover the training set and gate subset")
    oof = tfidf_oof(cfg, teacher_storage)
    gate_base = full_tfidf(cfg, train, gate_rows, "gate")

    shift_dir, oof_dir = Path(paths["shift_dir"]), Path(paths["oof_dir"])
    civil_ids = pd.read_parquet(shift_dir / "civil_shift_ids.parquet")
    civil_truth = shift_rows(load_civil(download_civil(cfg, data_dir / "civil"),
                                        cfg["shift"]["split"], cfg["shift"]["threshold"]),
                             civil_ids)
    civil_teacher = pd.read_parquet(shift_dir / f"civil_{pv}.parquet")
    civil_base = pd.read_parquet(oof_dir / "civil_tfidf_full.parquet")

    # The two candidates.
    stack_a = load_stack(oof_dir / f"stack_{pv}.json")
    stack_b = fit_stack(teacher, oof, train_truth, unit_weights(teacher["id"]),
                        st["full_version"], pv, stack_a["model_version"],
                        stack_a["baseline_version"], st["C"])
    stack_b["method"] = "logistic regression"
    stack_b["fit_on"] = f"all {len(teacher)} training rows, unweighted"
    stack_b["baseline_scores"] = stack_a.get("baseline_scores")
    stack_a.setdefault("fit_on", "the 2,000 pilot rows, pilot inclusion weights")
    stacks = {"A": stack_a, "B": stack_b}

    # Held-out comparison.
    domains = {"gate_subset": (gate_truth, gate_ids, gate_teacher, gate_base),
               "civil_shift": (civil_truth, civil_ids, civil_teacher, civil_base)}
    report: dict[str, Any] = {"prompt_version": pv, "stacks": stacks, "domains": {}}
    b_minus_a, gate_ece = {}, {}
    for domain, (truth, weights, t_pred, b_pred) in domains.items():
        preds = {"A": apply_stack(t_pred, b_pred, stack_a),
                 "B": apply_stack(t_pred, b_pred, stack_b),
                 "tfidf": b_pred.assign(status="ok"), "teacher": t_pred}
        tables = {name: evaluate(p, truth, weights, bins) for name, p in preds.items()}
        boot = paired_bootstrap({k: preds[k] for k in ("A", "B", "tfidf")}, truth, weights,
                                [("B", "A"), ("B", "tfidf"), ("A", "tfidf")], strata=weights,
                                n_boot=n_boot, seed=cfg["seed"])
        report["domains"][domain] = {"table": {k: table(t) for k, t in tables.items()},
                                     "bootstrap": boot}
        b_minus_a[domain] = boot["B - A"]
        if domain == "gate_subset":
            gate_ece = {k: float(tables[k].loc["mean", "ece"]) for k in ("A", "B")}
        logger.info("%s: mean AUC A %.4f, B %.4f, TF-IDF %.4f; B - A %s", domain,
                    tables["A"].loc["mean", "auc"], tables["B"].loc["mean", "auc"],
                    tables["tfidf"].loc["mean", "auc"], boot["B - A"]["mean_auc_ci95"])
    decision = choose(b_minus_a, gate_ece, st["max_ece_increase"])
    report["decision"] = decision
    adopted = stacks[decision["adopt"]]

    # Soft labels: cross-fitted for training rows, the frozen stack for the gate subset.
    if decision["adopt"] == "B":
        train_soft = cross_fit_by_folds(teacher, oof, train_truth, adopted["stack_version"],
                                        st["C"])
        cross_fitted = f"yes: {cfg['oof']['folds']} folds of tfidf_oof (seed {cfg['seed']})"
    else:
        pilot_ids = pd.read_parquet(Path(paths["pilot_dir"]) / "pilot_ids.parquet")
        pilot_ids["id"] = pilot_ids["id"].astype(str)
        pilot = teacher["id"].isin(set(pilot_ids["id"]))
        rest = apply_stack(teacher[~pilot], oof, adopted)
        pilot_soft = cross_fit_stack(teacher[pilot], oof, train_truth, pilot_ids,
                                     cfg["calibration"]["folds"], cfg["seed"], st["C"])
        train_soft = pd.concat([rest, pilot_soft], ignore_index=True)
        cross_fitted = "pilot rows only (the stack was fitted on them)"
    gate_soft = apply_stack(gate_teacher, gate_base, adopted)

    model_version = soft_model_version(adopted)
    plan = plan_shards(train["id"].tolist(), cfg["labeling"]["shard_size"])
    check_plan(plan, json.loads(Path(paths["shard_plan"]).read_text()))
    train_frames = shard_frames(train_soft, plan, pv, model_version)
    gate_frames = shard_frames(gate_soft, plan_shards(gate_ids["id"].tolist(),
                                                      cfg["labeling"]["shard_size"]),
                               pv, model_version)
    # Metrics from the exact frames that get uploaded (float32), to compare after upload.
    report["soft_labels"] = {
        "model_version": model_version,
        "train_cross_fitted": cross_fitted,
        "train": table(evaluate(pd.concat(train_frames), train_truth, None, bins)),
        "train_teacher_raw": table(evaluate(teacher, train_truth, None, bins)),
        "train_tfidf_oof": table(evaluate(oof.assign(status="ok"), train_truth, None, bins)),
        "gate_subset": table(evaluate(pd.concat(gate_frames), gate_truth, gate_ids, bins)),
    }

    stack_dir = Path(paths["stack_dir"])
    stack_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stack_dir / f"stack_{pv}.json"
    save_stack(adopted, stack_path)
    (stack_dir / "stack_check.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("adopted %s (%s); soft labels: train %s, gate %s", decision["adopt"],
                adopted["stack_version"], report["soft_labels"]["train"]["mean_auc"],
                report["soft_labels"]["gate_subset"]["mean_auc"])

    if out_storage is not None:
        if before_upload is not None:
            before_upload()
        prefix = soft_prefix(adopted["stack_version"], pv)
        stack_file = stack_path.resolve().relative_to(PIPELINE_ROOT).as_posix()
        extra = {"stack_version": adopted["stack_version"], "stack_file": stack_file,
                 "teacher_prefix": teacher_prefix,
                 "baseline_version": adopted["baseline_version"], "code_commit": code_commit}
        report["upload"] = {
            "prefix": prefix,
            "train": upload_shards(out_storage, prefix, train_frames, pv, model_version,
                                   sha256_file(stack_path),
                                   {**extra, "cross_fitted": cross_fitted,
                                    "baseline_scores": "tfidf_oof (out-of-fold)"}),
            "gate": upload_shards(out_storage, f"{prefix}gate/", gate_frames, pv,
                                  model_version, sha256_file(stack_path),
                                  {**extra, "cross_fitted": "no (held-out rows)",
                                   "teacher_prefix": f"{teacher_prefix}gate/",
                                   "baseline_scores": "TF-IDF trained on all of train"}),
        }
    return report


def run_drift(cfg: Mapping[str, Any], teacher_storage, out_storage=None,
              code_commit: str | None = None,
              before_upload: Callable[[], None] | None = None) -> dict[str, Any]:
    """Soft labels for the drift stream from the adopted stack, frozen; nothing is refitted.

    Reads the data lane's batches from ``paths.drift_dir`` and the teacher's drift shards,
    checks that ``artifacts/stack/`` holds the stack that wrote the training soft labels,
    writes ``drift_check.json`` (teacher, TF-IDF and stack against the batches' Civil gold
    labels, overall and per batch), and uploads one shard per batch if ``out_storage`` is
    given.
    """
    import pandas as pd

    from src.teacher.label_worker import select_drift_rows
    from src.teacher.metrics import evaluate
    from src.teacher.shards import plan_batches
    from src.teacher.stack import apply_stack, load_stack
    from src.teacher.storage import shard_paths

    pv, bins, paths = cfg["shift"]["prompt_version"], cfg["metrics"]["ece_bins"], cfg["paths"]
    stack_path = Path(paths["stack_dir"]) / f"stack_{pv}.json"
    stack, stack_sha = load_stack(stack_path), sha256_file(stack_path)
    prefix = soft_prefix(stack["stack_version"], pv)
    with tempfile.TemporaryDirectory() as tmp:
        local = teacher_storage.download(shard_paths(prefix, "shard_0000")[1], Path(tmp))
        trained_sha = json.loads(Path(local).read_text()).get("stack_sha256")
    if trained_sha != stack_sha:
        raise ValueError(f"{stack_path} is not the stack that wrote {prefix} "
                         f"({trained_sha}); the drift labels must come from the same stack")

    drift_dir = Path(paths["drift_dir"])
    rows = select_drift_rows(drift_dir, drift_dir / cfg["drift"]["ids_file"],
                             ["id", "text", *LABELS])
    teacher_prefix = f"teacher/{pv}/drift/"
    teacher = load_teacher(teacher_storage, teacher_prefix)
    if set(teacher["id"]) != set(rows["id"]):
        raise ValueError(f"{teacher_prefix} does not cover the drift rows")
    train = pd.read_parquet(Path(paths["data_dir"]) / cfg["data"]["files"]["train"])
    train["id"] = train["id"].astype(str)
    base = full_tfidf(cfg, train, rows, "drift")

    soft = apply_stack(teacher, base, stack)
    model_version = soft_model_version(stack)
    frames = shard_frames(soft, plan_batches(dict(zip(rows["id"], rows["batch"], strict=True))),
                          pv, model_version)
    preds = {"stack": pd.concat(frames), "teacher_raw": teacher,
             "tfidf": base.assign(status="ok")}
    truth = rows[["id", *LABELS]]

    def mean_auc(pred, part) -> float | None:
        value = evaluate(pred, part, None, bins).loc["mean", "auc"]
        return None if pd.isna(value) else round(float(value), 5)

    per_batch = []
    for b, part in rows.groupby("batch", sort=True):
        per_batch.append({"batch": int(b), "rows": len(part),
                          "positives": {k: int(part[k].sum()) for k in LABELS},
                          **{f"{name}_mean_auc": mean_auc(p, part[["id", *LABELS]])
                             for name, p in preds.items()}})
    report = {
        "stack_version": stack["stack_version"], "stack_sha256": stack_sha,
        "model_version": model_version, "rows": len(rows), "batches": len(frames),
        "note": "Civil Comments gold labels (positive at half the raters or more); a batch's "
                "mean AUC skips labels with no positive in that batch",
        "overall": {name: table(evaluate(p, truth, None, bins)) for name, p in preds.items()},
        "per_batch": per_batch,
    }
    report_path = Path(paths["stack_dir"]) / "drift_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("drift: stack %s, teacher %s, TF-IDF %s (mean AUC)",
                report["overall"]["stack"]["mean_auc"],
                report["overall"]["teacher_raw"]["mean_auc"],
                report["overall"]["tfidf"]["mean_auc"])

    if out_storage is not None:
        if before_upload is not None:
            before_upload()
        extra = {"stack_version": stack["stack_version"],
                 "stack_file": stack_path.resolve().relative_to(PIPELINE_ROOT).as_posix(),
                 "teacher_prefix": teacher_prefix,
                 "baseline_version": stack["baseline_version"], "code_commit": code_commit,
                 "cross_fitted": "no (held-out rows)",
                 "baseline_scores": "TF-IDF trained on all of train",
                 "layout": "one shard per batch: shard_<k> is batch k"}
        report["upload"] = {"prefix": f"{prefix}drift/",
                            "drift": upload_shards(out_storage, f"{prefix}drift/", frames, pv,
                                                   model_version, stack_sha, extra)}
    return report


def main() -> None:
    from src.teacher.storage import get_storage

    parser = argparse.ArgumentParser(description="Final stack and soft-label shards.")
    parser.add_argument("--upload", action="store_true", help="write the soft-label shards")
    parser.add_argument("--teacher-root", default=None,
                        help="local copy of the teacher shards (default: the configured store)")
    parser.add_argument("--local-root", default=None, help="write shards here, not the store")
    parser.add_argument("--drift", action="store_true",
                        help="the drift stream, from the frozen stack (nothing is refitted)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tracked = ["src", "configs", cfg["paths"]["stack_dir"], cfg["paths"]["oof_dir"]]
    commit, dirty = git_state(tracked)
    teacher_storage = get_storage(cfg, args.teacher_root)
    def still_committed() -> None:
        if git_state(tracked)[1] and args.local_root is None:
            raise SystemExit("artifacts/stack/ or artifacts/oof/ as rewritten by this run "
                             "differs from the committed files; nothing was uploaded")

    out_storage = None
    if args.upload:
        if dirty and args.local_root is None:
            raise SystemExit("commit src/, configs/, artifacts/stack/ and artifacts/oof/ "
                             "before --upload")
        out_storage = get_storage(cfg, args.local_root)
    before_upload = still_committed if args.upload else None
    if args.drift:
        report = run_drift(cfg, teacher_storage, out_storage, commit, before_upload)
        print(json.dumps({"drift": {k: v["mean_auc"] for k, v in report["overall"].items()},
                          "upload": report.get("upload")}, indent=2))
        return
    report = run(cfg, teacher_storage, out_storage, code_commit=commit,
                 before_upload=before_upload)
    print(json.dumps({"decision": report["decision"],
                      "soft_labels": {k: v["mean_auc"] if isinstance(v, dict) else v
                                      for k, v in report["soft_labels"].items()},
                      "upload": report.get("upload")}, indent=2))


if __name__ == "__main__":
    main()
