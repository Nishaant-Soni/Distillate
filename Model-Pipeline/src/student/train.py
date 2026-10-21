"""Student training: gold-only, soft-only, or mixed distillation, set by ``loss.alpha``.

One run: load the split (``data.py``), join the soft labels by id when ``alpha < 1``, train
with the distillation loss (``loss.py``), keep the epoch with the best validation mean AUC,
and write to ``paths.out_dir/<run>/``:

* ``model/``: weights and tokenizer (``save_pretrained``) plus ``meta.json`` (lineage);
* ``predictions/val.parquet``: the validation prediction file (CONTRACTS.md §2.2);
* ``splits/{train,val}_ids.txt``, ``metrics.json``, ``summary.json``.

Params, per-epoch metrics, and those files (except the weights) go to MLflow: the local
SQLite store in ``paths.mlflow_db``, or ``MLFLOW_TRACKING_URI`` when set (the VM later).

    python -m src.student.train                                    # configs/student.yaml
    python -m src.student.train --set loss.alpha=0.5 loss.temperature=2
    python -m src.student.train --config configs/smoke.yaml        # offline, seconds
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.student.config import load_config, uses_soft_labels
from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger

logger = get_logger(__name__)
TASK_ID = "train"


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def code_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(PIPELINE_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"loss.alpha": 1.0, ...}`` for MLflow params; paths and model ids are left out."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in ("paths", "models", "labels", "storage"):
            continue
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


class Rows:
    """Tokenized rows for a DataLoader; padding happens per batch in ``collate``."""

    def __init__(self, frame, tokenizer, max_length: int, soft: bool):
        import numpy as np

        enc = tokenizer(frame["text"].tolist(), truncation=True, max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.labels = frame[list(LABELS)].to_numpy(dtype="float32")
        self.weights = frame["sample_weight"].to_numpy(dtype="float32")
        self.soft = soft
        if soft:
            self.teacher = frame[[f"logit_{k}" for k in LABELS]].to_numpy(dtype="float32")
            self.soft_ok = frame["soft_ok"].to_numpy(dtype=bool)
        self.truncated_share = float(np.mean([len(x) >= max_length for x in self.input_ids]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {"input_ids": self.input_ids[i], "labels": self.labels[i],
                "sample_weight": self.weights[i]}
        if self.soft:
            item["teacher_logits"] = self.teacher[i]
            item["soft_ok"] = self.soft_ok[i]
        return item


def make_collate(tokenizer):
    import numpy as np
    import torch

    def collate(items):
        batch = tokenizer.pad({"input_ids": [x["input_ids"] for x in items]},
                              return_tensors="pt")
        out = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
               "labels": torch.from_numpy(np.stack([x["labels"] for x in items])),
               "sample_weight": torch.tensor([x["sample_weight"] for x in items])}
        if "teacher_logits" in items[0]:
            out["teacher_logits"] = torch.from_numpy(np.stack([x["teacher_logits"]
                                                               for x in items]))
            out["soft_ok"] = torch.tensor([bool(x["soft_ok"]) for x in items])
        return out

    return collate


def evaluate_split(model, tokenizer, frame, cfg: Mapping[str, Any], device,
                   model_version: str):
    """``(prediction_frame, metrics_table)`` on gold labels (unweighted)."""
    from src.student.predict import predict_logits, prediction_frame
    from src.teacher.metrics import evaluate

    t = cfg["train"]
    logits = predict_logits(model, tokenizer, frame["text"].tolist(),
                            int(cfg["model"]["max_length"]), int(t["eval_batch_size"]), device,
                            bool(t["fp16"]))
    pred = prediction_frame(frame["id"], logits, model_version)
    return pred, evaluate(pred, frame)


def _mlflow_start(cfg: Mapping[str, Any], run_name: str):
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        db = Path(cfg["paths"]["mlflow_db"])
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = None if os.environ.get("MLFLOW_TRACKING_URI") else \
            Path(cfg["paths"]["mlflow_artifacts"]).resolve().as_uri()
        mlflow.create_experiment(name, artifact_location=artifacts)
    mlflow.set_experiment(name)
    return mlflow.start_run(run_name=run_name)


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Train one student; return the JSON summary (also written to reports/summaries/)."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from src.student import data as data_mod
    from src.student.loss import distillation_loss
    from src.student.model import build
    from src.student.predict import check_prediction_frame
    from src.teacher.shards import read_ids

    start_time = time.perf_counter()
    seed, t, alpha = int(cfg["seed"]), cfg["train"], float(cfg["loss"]["alpha"])
    temperature = float(cfg["loss"]["temperature"])
    set_seed(seed)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{cfg['run_name']}-{stamp}"
    model_version = f"student-{run_name}"
    run_dir = Path(cfg["paths"]["out_dir"]) / run_name
    for sub in ("model", "predictions", "splits"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    train_df, val_df, data_info = data_mod.load_splits(cfg)
    soft = uses_soft_labels(cfg)
    soft_info = None
    if soft:
        # temp has every raw train id, so the teacher validator's full checks can run.
        source_ids = read_ids(cfg["paths"]["raw_train"]) if cfg["data"]["source"] == "temp" \
            else None
        labels, soft_info = data_mod.load_soft_labels(cfg, source_ids)
        train_df = data_mod.attach_soft_labels(train_df, labels)
    for name, frame in (("train", train_df), ("val", val_df)):
        (run_dir / "splits" / f"{name}_ids.txt").write_text("\n".join(frame["id"]) + "\n")
    logger.info("data: %s", json.dumps(data_info))

    device = pick_device(t["device"])
    model, tokenizer, model_id = build(cfg)
    model.to(device)
    max_length = int(cfg["model"]["max_length"])
    train_rows = Rows(train_df, tokenizer, max_length, soft)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_rows, batch_size=int(t["batch_size"]), shuffle=True,
                        generator=generator, collate_fn=make_collate(tokenizer),
                        num_workers=int(t["num_workers"]))

    epochs, max_steps = int(t["epochs"]), t.get("max_steps")
    total_steps = min(epochs * len(loader), int(max_steps)) if max_steps else epochs * len(loader)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": float(t["weight_decay"])},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=float(t["lr"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(float(t["warmup_ratio"]) * total_steps), total_steps)
    use_amp = bool(t["fp16"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mlflow_on = bool(cfg["mlflow"]["enabled"])
    mlflow_run = None
    if mlflow_on:
        import mlflow

        mlflow_run = _mlflow_start(cfg, run_name)
        mlflow.log_params({**flatten(cfg), "model_id": model_id,
                           "train_rows": len(train_df), "val_rows": len(val_df)})
        mlflow.set_tags({"model_version": model_version, "code_commit": code_commit() or "",
                         "data_source": data_info["source"], "lane": "student"})

    history: list[dict[str, Any]] = []
    best_auc, best_epoch, bad_epochs, step = -1.0, -1, 0, 0
    best_table = None
    try:
        for epoch in range(epochs):
            model.train()
            running = {"loss": 0.0, "gold": 0.0, "soft": 0.0, "n": 0}
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"]).logits
                loss, parts = distillation_loss(
                    logits, batch["labels"], batch.get("teacher_logits"), alpha, temperature,
                    batch["sample_weight"], batch.get("soft_ok"))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                running["loss"] += float(loss.detach())
                running["gold"] += parts["gold"]
                running["soft"] += parts["soft"] if soft else 0.0
                running["n"] += 1
                if step % int(t["log_every"]) == 0:
                    logger.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps,
                                running["loss"] / running["n"])
                    if mlflow_on:
                        mlflow.log_metric("train_loss", float(loss.detach()), step=step)
                if max_steps and step >= int(max_steps):
                    break

            pred, table = evaluate_split(model, tokenizer, val_df, cfg, device, model_version)
            n = max(running["n"], 1)
            record = {"epoch": epoch, "step": step, "train_loss": running["loss"] / n,
                      "train_gold": running["gold"] / n,
                      "train_soft": running["soft"] / n if soft else None,
                      "val_mean_auc": float(table.loc["mean", "auc"]),
                      "val_mean_pr_auc": float(table.loc["mean", "pr_auc"]),
                      **{f"val_auc_{k}": float(table.loc[k, "auc"]) for k in LABELS}}
            history.append(record)
            logger.info("epoch %d: val mean AUC %.4f, PR-AUC %.4f", epoch,
                        record["val_mean_auc"], record["val_mean_pr_auc"])
            if mlflow_on:
                mlflow.log_metrics({k: v for k, v in record.items()
                                    if isinstance(v, float)}, step=epoch)
            if record["val_mean_auc"] > best_auc:
                best_auc, best_epoch, bad_epochs = record["val_mean_auc"], epoch, 0
                model.save_pretrained(run_dir / "model")
                tokenizer.save_pretrained(run_dir / "model")
                pred.to_parquet(run_dir / "predictions" / "val.parquet", index=False)
                best_table = table
            else:
                bad_epochs += 1
                if bad_epochs > int(t["patience"]):
                    logger.info("early stop after epoch %d", epoch)
                    break
            if max_steps and step >= int(max_steps):
                break

        best_pred = pd.read_parquet(run_dir / "predictions" / "val.parquet")
        check_prediction_frame(best_pred)
        meta = {
            "model_version": model_version, "model_id": model_id, "max_length": max_length,
            "labels": list(LABELS), "alpha": alpha, "temperature": temperature,
            "teacher_version": cfg["soft_labels"]["teacher_version"] if soft else None,
            "teacher_model_version": cfg["soft_labels"]["teacher_model_version"] if soft
            else None,
            "prompt_version": cfg["soft_labels"]["prompt_version"] if soft else None,
            "stack_sha256": soft_info["stack_sha256"] if soft else None,
            "soft_label_model_version": cfg["soft_labels"]["model_version"] if soft else None,
            "data_source": data_info["source"], "training_data_sha256":
            data_info["source_sha256"], "training_data_dvc_hash": None,   # set in Phase 2
            "seed": seed, "best_epoch": best_epoch, "code_commit": code_commit(),
            "mlflow_run_id": mlflow_run.info.run_id if mlflow_run else None,
        }
        (run_dir / "model" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        metrics = json.loads(best_table.round(6).to_json(orient="index"))
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        summary = {
            "task": TASK_ID, "run_name": run_name, "model_version": model_version,
            "model_id": model_id, "device": device.type, "alpha": alpha,
            "temperature": temperature, "rows_in": data_info["train_rows"] + data_info[
                "val_rows"], "data": data_info, "soft_labels": soft_info,
            "truncated_share_train": round(train_rows.truncated_share, 5),
            "steps": step, "best_epoch": best_epoch, "best_val_mean_auc": best_auc,
            "best_val_mean_pr_auc": float(best_table.loc["mean", "pr_auc"]),
            "history": history, "run_dir": str(run_dir),
            "mlflow_run_id": meta["mlflow_run_id"],
            "duration_s": round(time.perf_counter() - start_time, 1),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        reports = Path(cfg["paths"]["reports_dir"])
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"{TASK_ID}.json").write_text(json.dumps(summary, indent=2) + "\n")
        if mlflow_on:
            mlflow.log_metrics({"best_val_mean_auc": best_auc,
                                "best_val_mean_pr_auc": summary["best_val_mean_pr_auc"],
                                "truncated_share_train": train_rows.truncated_share})
            for name in ("summary.json", "metrics.json"):
                mlflow.log_artifact(str(run_dir / name))
            mlflow.log_artifact(str(run_dir / "model" / "meta.json"), "model")
            mlflow.log_artifacts(str(run_dir / "predictions"), "predictions")
            mlflow.log_artifacts(str(run_dir / "splits"), "splits")
            if cfg["mlflow"]["log_model"]:
                mlflow.log_artifacts(str(run_dir / "model"), "model")
    finally:
        if mlflow_on:
            mlflow.end_run()
    logger.info("done: best epoch %d, val mean AUC %.4f, %s", best_epoch, best_auc, run_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Distillate student.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. loss.alpha=0.5")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
