"""Civil Comments shift check: do TF-IDF, the teacher, and the stack hold up off Wikipedia?

Data: ``google/civil_comments`` on Hugging Face at a pinned revision (TFDS's download now
returns 403). It has no id column, so a row's id is ``civil-<split>-<row number>`` in that
file at that revision. Labels are rater fractions; a label is positive when the fraction is
at least ``shift.threshold``. Civil's ``severe_toxicity`` almost never reaches 0.5, so that
label usually has no positives and drops out of the AUC means.

The sample is enriched like the pilot (``sampling.enriched_sample``), with inclusion weights,
so the metrics estimate the whole split. Only ids, weights, and scores are committed.

Everything is scored by models fitted on Wikipedia only: TF-IDF trained on the same rows as
for the pilot, the teacher's pilot calibration, and the stack fitted on the pilot. Nothing
is refitted on Civil.

    python -m src.teacher.shift sample     # writes artifacts/shift/civil_shift_ids.parquet
    python -m src.teacher.shift evaluate   # needs artifacts/shift/civil_<prompt_version>.parquet
    python -m src.teacher.shift evaluate --prompt-version p2-zs-v2a --pilot-dir D --out-dir D
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger, load_config

logger = get_logger(__name__)


def download_civil(cfg: Mapping[str, Any], local_dir: str | Path) -> Path:
    """The pinned Civil Comments split file, downloaded once to ``local_dir``."""
    from huggingface_hub import hf_hub_download

    s = cfg["shift"]
    return Path(hf_hub_download(s["dataset"], s["file"], repo_type="dataset",
                                revision=s["revision"], local_dir=str(local_dir)))


def load_civil(path: str | Path, split: str, threshold: float):
    """``id``, cleaned ``text``, and the six labels (0/1 at ``threshold``), in file order."""
    import pandas as pd

    from src.teacher.data import clean_text

    raw = pd.read_parquet(path)
    missing = [c for c in ("text", *LABELS) if c not in raw]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    width = len(str(len(raw) - 1))
    out = pd.DataFrame({
        "id": pd.Series([f"civil-{split}-{i:0{width}d}" for i in range(len(raw))],
                        dtype="string"),
        "text": raw["text"].fillna("").map(clean_text),
    })
    for label in LABELS:
        out[label] = (raw[label].to_numpy() >= threshold).astype("int8")
    return out


def shift_rows(civil, ids):
    """The rows of ``civil`` whose ids are in ``ids`` (a frame with ``id``), in ``ids`` order."""
    rows = ids[["id"]].merge(civil, on="id", how="left", validate="one_to_one", indicator=True)
    if (rows.pop("_merge") != "both").any():
        raise ValueError("some shift ids are not in the Civil Comments file")
    return rows


def summary(civil, sample) -> dict[str, Any]:
    """Positives per label in the sample, and weighted vs. true prevalence."""
    import numpy as np

    rows = shift_rows(civil, sample)
    w = sample["weight"].to_numpy()
    out: dict[str, Any] = {"rows": len(sample), "population": len(civil),
                           "weight_sum": round(float(w.sum()), 1), "labels": {}}
    for label in LABELS:
        y = rows[label].to_numpy()
        out["labels"][label] = {
            "positives": int(y.sum()),
            "weighted_prevalence": round(float(np.average(y, weights=w)), 5),
            "true_prevalence": round(float(civil[label].mean()), 5),
        }
    out["strata"] = {str(k): int(v) for k, v in sample["stratum"].value_counts().items()}
    return out


def run_sample(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Draw and save the enriched shift sample."""
    from src.teacher.sampling import enriched_sample

    s = cfg["shift"]
    civil = load_civil(download_civil(cfg, Path(cfg["paths"]["data_dir"]) / "civil"),
                       s["split"], s["threshold"])
    sample = enriched_sample(civil, s["size"], s["enrich"], cfg["seed"])
    info = {"dataset": s["dataset"], "revision": s["revision"], "file": s["file"],
            "threshold": s["threshold"], "seed": cfg["seed"], **summary(civil, sample)}
    out_dir = Path(cfg["paths"]["shift_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_dir / "civil_shift_ids.parquet", index=False)
    (out_dir / "civil_shift_summary.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _table(name: str, metrics) -> dict[str, Any]:
    """One results-table row from an ``evaluate`` table."""
    import math

    row: dict[str, Any] = {"model": name}
    for label in LABELS:
        row[f"auc_{label}"] = metrics.loc[label, "auc"]
        row[f"pr_auc_{label}"] = metrics.loc[label, "pr_auc"]
    for key, column in (("mean_auc", "auc"), ("mean_pr_auc", "pr_auc"), ("ece", "ece"),
                        ("brier", "brier"), ("failure_rate", "failure_rate")):
        row[key] = metrics.loc["mean", column]
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def run_evaluate(cfg: Mapping[str, Any], teacher_path: str | Path | None = None,
                 n_boot: int = 1000, prompt_version: str | None = None,
                 pilot_pred_dir: str | Path | None = None,
                 out_dir: str | Path | None = None) -> dict[str, Any]:
    """Score the shift sample with TF-IDF and the stack, evaluate all models on both domains.

    ``prompt_version`` defaults to ``shift.prompt_version``. The teacher's pilot predictions
    and calibration (``pilot_<pv>.parquet``, ``calibration_<pv>.json``) are read from
    ``pilot_pred_dir`` (default ``paths.pilot_dir``), its Civil predictions from
    ``teacher_path`` (default ``<out_dir>/civil_<pv>.parquet``). Writes to ``out_dir``
    (default ``paths.shift_dir``): ``civil_tfidf.parquet``, ``civil_stack_<pv>.parquet``,
    ``stack_<pv>.json``, and ``shift_results_<pv>.json``.
    """
    import pandas as pd

    from src.teacher.baseline_tfidf import train_predict
    from src.teacher.calibrate import apply_calibration, cross_fit, load_calibration
    from src.teacher.metrics import evaluate, paired_bootstrap
    from src.teacher.pilot import pilot_truth
    from src.teacher.stack import apply_stack, cross_fit_stack, fit_stack, save_stack

    s, bins = cfg["shift"], cfg["metrics"]["ece_bins"]
    pv, tfidf_name = prompt_version or s["prompt_version"], cfg["baseline"]["model_version"]
    data_dir, pilot_dir = Path(cfg["paths"]["data_dir"]), Path(cfg["paths"]["pilot_dir"])
    shift_dir = Path(cfg["paths"]["shift_dir"])
    pred_dir = Path(pilot_pred_dir) if pilot_pred_dir else pilot_dir
    out_dir = Path(out_dir) if out_dir else shift_dir

    # Wikipedia side: the pilot, with TF-IDF trained on train minus pilot and few-shot rows.
    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    pilot_ids = pd.read_parquet(pilot_dir / "pilot_ids.parquet")
    pilot = pilot_truth(train, pilot_ids)
    fewshot = json.loads((pilot_dir / "fewshot_ids.json").read_text())["ids"]
    held_out = set(pilot_ids["id"]) | {i for ids in fewshot.values() for i in ids}
    wiki_teacher = pd.read_parquet(pred_dir / f"pilot_{pv}.parquet")
    wiki_tfidf = pd.read_parquet(pilot_dir / "pilot_tfidf.parquet")
    calibration = load_calibration(pred_dir / f"calibration_{pv}.json")

    # Civil side.
    sample = pd.read_parquet(shift_dir / "civil_shift_ids.parquet")
    civil = load_civil(download_civil(cfg, data_dir / "civil"), s["split"], s["threshold"])
    civil_rows = shift_rows(civil, sample)
    civil_teacher = pd.read_parquet(teacher_path or out_dir / f"civil_{pv}.parquet")
    if set(civil_teacher["id"]) != set(sample["id"]):
        raise ValueError("the teacher's Civil predictions do not cover the shift sample")

    # One TF-IDF fit predicts both the pilot (a check against the committed predictions)
    # and Civil.
    fit_rows = train[~train["id"].isin(held_out)]
    both = pd.concat([train.loc[train["id"].isin(set(pilot_ids["id"])), ["id", "text"]],
                      civil_rows[["id", "text"]]], ignore_index=True)
    tfidf = train_predict(fit_rows, both, {**cfg["baseline"]}, cfg["seed"])
    civil_tfidf = tfidf[tfidf["id"].isin(set(sample["id"]))].reset_index(drop=True)
    refit = wiki_tfidf[["id"]].merge(tfidf, on="id")
    tfidf_max_diff = max(float((refit[f"p_{label}"] - wiki_tfidf[f"p_{label}"]).abs().max())
                         for label in LABELS)
    if tfidf_max_diff > 1e-4:
        raise ValueError(f"TF-IDF refit differs from pilot_tfidf.parquet by {tfidf_max_diff}")

    stack = fit_stack(wiki_teacher, wiki_tfidf, pilot, pilot_ids, cfg["stack"]["version"], pv,
                      calibration["model_version"], tfidf_name, cfg["stack"]["C"])
    wiki_stack = cross_fit_stack(wiki_teacher, wiki_tfidf, pilot, pilot_ids,
                                 cfg["calibration"]["folds"], cfg["seed"], cfg["stack"]["C"])
    civil_stack = apply_stack(civil_teacher, civil_tfidf, stack)

    stack_name, cal_name = cfg["stack"]["version"], f"{pv}-calibrated"
    domains = {
        "wikipedia_pilot": (pilot, pilot_ids, {
            tfidf_name: wiki_tfidf, pv: wiki_teacher,
            cal_name: cross_fit(wiki_teacher, pilot, pilot_ids,       # both cross-fitted,
                                cfg["calibration"]["folds"], cfg["seed"]),
            stack_name: wiki_stack}),                                   # as in the pilot table
        "civil_shift": (civil_rows, sample, {
            tfidf_name: civil_tfidf, pv: civil_teacher,
            cal_name: apply_calibration(civil_teacher, calibration),
            stack_name: civil_stack}),
    }
    pairs = [(stack_name, tfidf_name), (pv, tfidf_name), (stack_name, pv)]
    results: dict[str, Any] = {
        "prompt_version": pv, "model_version": calibration["model_version"],
        "baseline_version": tfidf_name, "stack_version": stack_name,
        "civil": {k: s[k] for k in ("dataset", "revision", "split", "threshold")},
        "tfidf_refit_max_abs_diff": tfidf_max_diff, "stack": stack, "domains": {},
    }
    for domain, (truth, weights, preds) in domains.items():
        tables = {name: evaluate(pred, truth, weights, bins) for name, pred in preds.items()}
        ranking = {k: v for k, v in preds.items() if k != cal_name}   # same ranking as pv
        results["domains"][domain] = {
            "rows": len(truth),
            "positives": {label: int(truth[label].sum()) for label in LABELS},
            "table": [_table(name, t) for name, t in tables.items()],
            "bootstrap": paired_bootstrap(ranking, truth, weights, pairs, strata=weights,
                                          n_boot=n_boot, seed=cfg["seed"]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    civil_tfidf.to_parquet(out_dir / "civil_tfidf.parquet", index=False)
    civil_stack.to_parquet(out_dir / f"civil_stack_{pv}.parquet", index=False)
    save_stack(stack, out_dir / f"stack_{pv}.json")
    (out_dir / f"shift_results_{pv}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Civil Comments shift check.")
    parser.add_argument("what", choices=["sample", "evaluate"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--teacher", default=None, help="the teacher's Civil predictions")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--pilot-dir", default=None, help="teacher pilot predictions")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.what == "sample":
        print(json.dumps(run_sample(cfg), indent=2))
    else:
        results = run_evaluate(cfg, args.teacher, prompt_version=args.prompt_version,
                               pilot_pred_dir=args.pilot_dir, out_dir=args.out_dir)
        for domain, r in results["domains"].items():
            print(domain)
            print(json.dumps({t["model"]: (t["mean_auc"], t["mean_pr_auc"], t["ece"])
                              for t in r["table"]}, indent=2))
            print(json.dumps(r["bootstrap"], indent=2))


if __name__ == "__main__":
    main()
