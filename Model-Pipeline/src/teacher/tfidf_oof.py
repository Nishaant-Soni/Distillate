"""Out-of-fold TF-IDF scores for the training set, and the stack check that depends on them
(implementation plan, Step 6b).

The stack combines the teacher's logit with TF-IDF's score. A TF-IDF model scoring its own
training rows nearly reproduces their gold labels, so every training row's TF-IDF score must
come from a model that never saw it: the 159,571 ids are split into ``oof.folds`` folds
(seed 42), and each fold is scored by a model trained on the others.

The check refits the stack on the pilot with the pilot rows' out-of-fold scores (the scores
the stack will actually receive on the training shards) and accepts if it still beats
TF-IDF, with the 95% bootstrap interval above 0, on the pilot and on Civil Comments. Civil,
like the gate subset and drift rows, is scored by a TF-IDF model trained on all of train.

    python -m src.teacher.tfidf_oof   # ~20 min CPU; writes artifacts/oof/
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger, load_config

logger = get_logger(__name__)


def assign_folds(ids: Sequence[str], folds: int, seed: int):
    """``id`` and ``fold`` (0 to ``folds - 1``), a seeded shuffle of the sorted ids."""
    import numpy as np
    import pandas as pd

    ordered = np.array(sorted(str(i) for i in ids))
    if len(set(ordered)) != len(ordered):
        raise ValueError("ids contain duplicates")
    fold = np.empty(len(ordered), dtype="int8")
    fold[np.random.default_rng(seed).permutation(len(ordered))] = np.arange(len(ordered)) % folds
    return pd.DataFrame({"id": pd.Series(ordered, dtype="string"), "fold": fold})


def oof_predict(train, folds_frame, params: Mapping[str, Any], seed: int,
                train_predict=None, log=logger.info):
    """Every row of ``train`` scored by a TF-IDF model trained on the other folds.

    Returns ``id``, ``fold``, ``p_<label>`` (float32), ``model_version``, sorted by id.
    """
    import pandas as pd

    if train_predict is None:
        from src.teacher.baseline_tfidf import train_predict
    frame = train.merge(folds_frame, on="id", how="inner", validate="one_to_one")
    if len(frame) != len(train):
        raise ValueError("every training row needs a fold")
    parts = []
    for k in sorted(frame["fold"].unique()):
        start = time.perf_counter()
        held = frame["fold"] == k
        pred = train_predict(frame[~held], frame[held], params, seed)
        pred["fold"] = pd.Series([int(k)] * len(pred), dtype="int8").to_numpy()
        parts.append(pred)
        log(f"fold {k}: trained on {int((~held).sum())}, scored {int(held.sum())} "
            f"in {time.perf_counter() - start:.0f} s")
    out = pd.concat(parts, ignore_index=True)
    out["id"] = out["id"].astype("string")
    columns = ["id", "fold", *[f"p_{label}" for label in LABELS], "model_version"]
    return out[columns].sort_values("id", kind="stable").reset_index(drop=True)


def run(cfg: Mapping[str, Any], n_boot: int = 1000) -> dict[str, Any]:
    """Write ``tfidf_oof.parquet``, refit and evaluate the stack, write ``oof_check.json``."""
    import pandas as pd

    from src.teacher.baseline_tfidf import train_predict
    from src.teacher.metrics import evaluate, paired_bootstrap
    from src.teacher.pilot import pilot_truth
    from src.teacher.shift import download_civil, load_civil, shift_rows
    from src.teacher.stack import apply_stack, cross_fit_stack, fit_stack, save_stack

    s, bins = cfg["shift"], cfg["metrics"]["ece_bins"]
    pv, tfidf_name, stack_name = s["prompt_version"], cfg["baseline"]["model_version"], \
        cfg["stack"]["version"]
    data_dir, pilot_dir = Path(cfg["paths"]["data_dir"]), Path(cfg["paths"]["pilot_dir"])
    shift_dir, out_dir = Path(cfg["paths"]["shift_dir"]), Path(cfg["paths"]["oof_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    train["id"] = train["id"].astype(str)
    folds = assign_folds(train["id"].tolist(), cfg["oof"]["folds"], cfg["seed"])
    oof = oof_predict(train[["id", "text", *LABELS]], folds, cfg["baseline"], cfg["seed"],
                      train_predict)
    oof.to_parquet(out_dir / "tfidf_oof.parquet", index=False)

    # Pilot: the stack refitted on out-of-fold TF-IDF scores.
    pilot_ids = pd.read_parquet(pilot_dir / "pilot_ids.parquet")
    pilot = pilot_truth(train, pilot_ids)
    teacher = pd.read_parquet(pilot_dir / f"pilot_{pv}.parquet")
    calibration = json.loads((pilot_dir / f"calibration_{pv}.json").read_text())
    pilot_oof = pilot[["id"]].merge(oof, on="id", validate="one_to_one")
    pilot_tfidf = pd.read_parquet(pilot_dir / "pilot_tfidf.parquet")    # held-out model
    stack = fit_stack(teacher, pilot_oof, pilot, pilot_ids, stack_name, pv,
                      calibration["model_version"], tfidf_name, cfg["stack"]["C"])
    stack["baseline_scores"] = (f"{tfidf_name}, out-of-fold on train ({cfg['oof']['folds']} "
                                "folds); full-train model for the gate subset and drift rows")
    wiki_stack = cross_fit_stack(teacher, pilot_oof, pilot, pilot_ids,
                                 cfg["calibration"]["folds"], cfg["seed"], cfg["stack"]["C"])

    # Civil: TF-IDF trained on all of train, the frozen stack applied.
    sample = pd.read_parquet(shift_dir / "civil_shift_ids.parquet")
    civil = shift_rows(load_civil(download_civil(cfg, data_dir / "civil"), s["split"],
                                  s["threshold"]), sample)
    start = time.perf_counter()
    civil_tfidf = train_predict(train, civil[["id", "text"]], cfg["baseline"], cfg["seed"])
    logger.info(f"full-train TF-IDF for Civil in {time.perf_counter() - start:.0f} s")
    civil_teacher = pd.read_parquet(shift_dir / f"civil_{pv}.parquet")
    civil_stack = apply_stack(civil_teacher, civil_tfidf, stack)
    civil_stack_v1 = pd.read_parquet(shift_dir / f"civil_stack_{pv}.parquet")   # earlier stack

    domains = {
        "wikipedia_pilot": (pilot, pilot_ids, {
            "stack_oof": wiki_stack, "tfidf_oof": pilot_oof, "tfidf_heldout": pilot_tfidf,
            pv: teacher}),
        "civil_shift": (civil, sample, {
            "stack_oof": civil_stack, "tfidf_full": civil_tfidf,
            "stack_heldout": civil_stack_v1, pv: civil_teacher}),
    }
    pairs = {"wikipedia_pilot": [("stack_oof", "tfidf_oof"), ("stack_oof", "tfidf_heldout")],
             "civil_shift": [("stack_oof", "tfidf_full"), ("stack_oof", "stack_heldout")]}
    results: dict[str, Any] = {"prompt_version": pv, "folds": cfg["oof"]["folds"],
                               "rows": len(oof), "stack": stack, "domains": {}}
    for domain, (truth, weights, preds) in domains.items():
        tables = {name: evaluate(pred, truth, weights, bins) for name, pred in preds.items()}
        results["domains"][domain] = {
            "table": {name: {"mean_auc": round(float(t.loc["mean", "auc"]), 5),
                             "mean_pr_auc": round(float(t.loc["mean", "pr_auc"]), 5),
                             "ece": round(float(t.loc["mean", "ece"]), 5),
                             "auc": {k: (None if pd.isna(t.loc[k, "auc"])
                                         else round(float(t.loc[k, "auc"]), 5))
                                     for k in LABELS}}
                      for name, t in tables.items()},
            "bootstrap": paired_bootstrap(preds, truth, weights, pairs[domain], strata=weights,
                                          n_boot=n_boot, seed=cfg["seed"]),
        }
    pilot_gap = results["domains"]["wikipedia_pilot"]["bootstrap"]["stack_oof - tfidf_oof"]
    civil_gap = results["domains"]["civil_shift"]["bootstrap"]["stack_oof - tfidf_full"]
    results["accept"] = {"pilot_beats_tfidf": pilot_gap["mean_auc_ci95"][0] > 0,
                         "civil_beats_tfidf": civil_gap["mean_auc_ci95"][0] > 0}
    results["accepted"] = all(results["accept"].values())

    save_stack(stack, out_dir / f"stack_{pv}.json")
    civil_tfidf.to_parquet(out_dir / "civil_tfidf_full.parquet", index=False)
    civil_stack.to_parquet(out_dir / f"civil_stack_{pv}.parquet", index=False)
    (out_dir / "oof_check.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-fold TF-IDF and the stack check.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    results = run(load_config(args.config))
    print(json.dumps({"accepted": results["accepted"], **results["accept"]}, indent=2))
    for domain, r in results["domains"].items():
        print(domain, json.dumps({k: v["mean_auc"] for k, v in r["table"].items()}))
        for pair, b in r["bootstrap"].items():
            print("  ", pair, b["mean_auc_diff"], b["mean_auc_ci95"])


if __name__ == "__main__":
    main()
