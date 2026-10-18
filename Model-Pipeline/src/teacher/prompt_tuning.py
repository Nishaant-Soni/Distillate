"""Obscene prompt follow-up (``p2-zs-v2*``): choose a rewording on half the pilot, test it
on the other half and on Civil Comments.

The rules are fixed before any v2 scoring (``prompt_tuning`` in teacher.yaml):

1. **Halves.** The pilot is split in two, stratified by pilot stratum and by the target
   label, and the split is committed (``halves.parquet``) before scoring.
2. **Selection, on half A only.** The candidate with the highest weighted AUC for the target
   label. No candidate is chosen if none beats v1 on half A.
3. **Acceptance, on half B and Civil (never used for selection).** All must hold:
   a. the target label's AUC improves on half B, with the 95% bootstrap interval above 0;
   b. the target label's AUC improves on Civil, with the interval above 0;
   c. no other label's AUC on half B drops by more than ``max_other_drop``;
   d. the stack's mean AUC on Civil does not drop (point estimate).

The candidates change only the target label's question, so the other labels' prompts are
byte-identical to v1; check (c) and ``unchanged_labels_max_logit_diff`` confirm it.

    python -m src.teacher.prompt_tuning halves     # writes artifacts/prompt_v2/halves.parquet
    python -m src.teacher.prompt_tuning evaluate   # after the Colab run, from the same folder
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, load_config

HALVES = ("A", "B")


def pilot_halves(truth, pilot_ids, label: str, seed: int):
    """``id`` and ``half`` (A or B): each (stratum, label value) group split evenly."""
    import numpy as np
    import pandas as pd

    frame = pilot_ids[["id", "stratum"]].merge(truth[["id", label]], on="id",
                                               validate="one_to_one")
    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    half = np.empty(len(frame), dtype=object)
    for _, group in frame.groupby(["stratum", label], sort=True):
        order = rng.permutation(group.index.to_numpy())
        half[order[0::2]] = "A"
        half[order[1::2]] = "B"
    return pd.DataFrame({"id": frame["id"].astype("string"), "half": pd.Series(half,
                                                                               dtype="string")})


def _subset(frame, ids):
    return frame[frame["id"].isin(set(ids))].reset_index(drop=True)


def label_auc(pred, truth, weights, label: str) -> float:
    """Weighted AUC for one label, ranking on ``logit_<label>`` (``ok`` rows only)."""
    from sklearn.metrics import roc_auc_score

    frame = truth[["id", label]].merge(pred[["id", f"logit_{label}", "status"]], on="id",
                                       validate="one_to_one")
    frame = frame.merge(weights[["id", "weight"]], on="id", validate="one_to_one")
    frame = frame[frame["status"] == "ok"]
    return float(roc_auc_score(frame[label], frame[f"logit_{label}"],
                               sample_weight=frame["weight"]))


def select(runs: Mapping[str, Any], v1, truth, weights, halves, label: str) -> dict[str, Any]:
    """Rule 2: the candidate with the best half-A AUC for ``label``, if it beats v1 there."""
    ids_a = halves.loc[halves["half"] == "A", "id"]
    truth_a = _subset(truth, ids_a)
    aucs = {pv: round(label_auc(pred, truth_a, weights, label), 5) for pv, pred in runs.items()}
    v1_auc = round(label_auc(v1, truth_a, weights, label), 5)
    best = max(aucs, key=lambda pv: aucs[pv])   # first in config order on a tie
    return {"label": label, "half": "A", "rows": len(truth_a),
            "positives": int(truth_a[label].sum()), "v1_auc": v1_auc, "candidate_auc": aucs,
            "chosen": best if aucs[best] > v1_auc else None}


def acceptance(checks: Mapping[str, Any], max_other_drop: float) -> dict[str, bool]:
    """Rule 3 from the computed checks."""
    target = checks["label"]
    others = {k: v for k, v in checks["half_b_auc_diff"].items() if k != target}
    return {
        "a_half_b_target_improves": checks["half_b_target"]["mean_auc_ci95"][0] > 0,
        "b_civil_target_improves": checks["civil_target"]["mean_auc_ci95"][0] > 0,
        "c_no_other_label_drops": all(d >= -max_other_drop for d in others.values()),
        "d_civil_stack_not_worse": checks["civil_stack"]["mean_auc_diff"] >= 0,
    }


def run_halves(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Write the pre-registered pilot halves."""
    import pandas as pd

    from src.teacher.pilot import pilot_truth

    t = cfg["prompt_tuning"]
    train = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / cfg["data"]["files"]["train"])
    pilot_ids = pd.read_parquet(Path(cfg["paths"]["pilot_dir"]) / "pilot_ids.parquet")
    truth = pilot_truth(train, pilot_ids)
    halves = pilot_halves(truth, pilot_ids, t["label"], cfg["seed"])
    out_dir = Path(cfg["paths"]["prompt_v2_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    halves.to_parquet(out_dir / "halves.parquet", index=False)
    joined = halves.merge(truth, on="id")
    return {h: {"rows": int((joined["half"] == h).sum()),
                "positives": {label: int(joined.loc[joined["half"] == h, label].sum())
                              for label in LABELS}} for h in HALVES}


def run_evaluate(cfg: Mapping[str, Any], n_boot: int = 1000) -> dict[str, Any]:
    """Recheck the selection, evaluate the chosen candidate, and apply rule 3.

    Reads the Colab output from ``paths.prompt_v2_dir``; writes ``v2_results.json`` there,
    next to the chosen candidate's shift-check outputs.
    """
    import pandas as pd

    from src.teacher.injection import TARGET
    from src.teacher.metrics import evaluate, paired_bootstrap
    from src.teacher.pilot import pilot_truth
    from src.teacher.shift import download_civil, load_civil, shift_rows
    from src.teacher.shift import run_evaluate as shift_evaluate

    t, s = cfg["prompt_tuning"], cfg["shift"]
    label, v1_pv = t["label"], t["baseline"]
    v2_dir = Path(cfg["paths"]["prompt_v2_dir"])
    pilot_dir, shift_dir = Path(cfg["paths"]["pilot_dir"]), Path(cfg["paths"]["shift_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])

    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    pilot_ids = pd.read_parquet(pilot_dir / "pilot_ids.parquet")
    truth = pilot_truth(train, pilot_ids)
    halves = pd.read_parquet(v2_dir / "halves.parquet")
    selection = json.loads((v2_dir / "selection.json").read_text())
    v1_pilot = pd.read_parquet(pilot_dir / f"pilot_{v1_pv}.parquet")
    runs = {f"p2-zs-{tv}": pd.read_parquet(v2_dir / f"pilot_p2-zs-{tv}.parquet")
            for tv in t["candidates"]}

    recheck = select(runs, v1_pilot, truth, pilot_ids, halves, label)
    if recheck["chosen"] != selection["chosen"]:
        raise ValueError(f"selection differs: Colab {selection['chosen']}, "
                         f"recomputed {recheck['chosen']}")
    results: dict[str, Any] = {"label": label, "v1": v1_pv, "selection": recheck,
                               "code_commit": selection.get("code_commit")}
    if recheck["chosen"] is None:
        results["accepted"] = False
        results["reason"] = "no candidate beats v1 on half A"
        (v2_dir / "v2_results.json").write_text(json.dumps(results, indent=2) + "\n")
        return results

    v2_pv = recheck["chosen"]
    v2_pilot = runs[v2_pv]
    shift = shift_evaluate(cfg, n_boot=n_boot, prompt_version=v2_pv, pilot_pred_dir=v2_dir,
                           out_dir=v2_dir)

    # Half B (rule 3a, 3c).
    ids_b = halves.loc[halves["half"] == "B", "id"]
    truth_b, ids_b_frame = _subset(truth, ids_b), _subset(pilot_ids, ids_b)
    table_b = {pv: evaluate(pred, truth_b, ids_b_frame, cfg["metrics"]["ece_bins"])
               for pv, pred in ((v1_pv, v1_pilot), (v2_pv, v2_pilot))}
    half_b_auc = {pv: {k: round(float(tab.loc[k, "auc"]), 5) for k in LABELS}
                  for pv, tab in table_b.items()}
    half_b_diff = {k: round(half_b_auc[v2_pv][k] - half_b_auc[v1_pv][k], 5) for k in LABELS}
    half_b_target = paired_bootstrap({v2_pv: v2_pilot, v1_pv: v1_pilot}, truth_b, ids_b_frame,
                                     [(v2_pv, v1_pv)], strata=ids_b_frame, n_boot=n_boot,
                                     seed=cfg["seed"], labels=[label])[f"{v2_pv} - {v1_pv}"]

    # Civil (rule 3b, 3d).
    sample = pd.read_parquet(shift_dir / "civil_shift_ids.parquet")
    civil = shift_rows(load_civil(download_civil(cfg, data_dir / "civil"), s["split"],
                                  s["threshold"]), sample)
    v1_civil = pd.read_parquet(shift_dir / f"civil_{v1_pv}.parquet")
    v2_civil = pd.read_parquet(v2_dir / f"civil_{v2_pv}.parquet")
    civil_target = paired_bootstrap({v2_pv: v2_civil, v1_pv: v1_civil}, civil, sample,
                                    [(v2_pv, v1_pv)], strata=sample, n_boot=n_boot,
                                    seed=cfg["seed"], labels=[label])[f"{v2_pv} - {v1_pv}"]
    stack_v1 = pd.read_parquet(shift_dir / f"civil_stack_{v1_pv}.parquet")
    stack_v2 = pd.read_parquet(v2_dir / f"civil_stack_{v2_pv}.parquet")
    civil_stack = paired_bootstrap({"stack_v2": stack_v2, "stack_v1": stack_v1}, civil, sample,
                                   [("stack_v2", "stack_v1")], strata=sample, n_boot=n_boot,
                                   seed=cfg["seed"])["stack_v2 - stack_v1"]

    # Sanity: the unchanged labels' prompts are identical, so their scores should match v1.
    both = v1_pilot.merge(v2_pilot, on="id", suffixes=("_v1", "_v2"))
    unchanged = {k: round(float((both[f"logit_{k}_v1"] - both[f"logit_{k}_v2"]).abs().max()), 5)
                 for k in LABELS if k != label}

    injection = {}
    for pv, path in ((v1_pv, pilot_dir / f"injection_{v1_pv}.csv"),
                     (v2_pv, v2_dir / f"injection_{v2_pv}.csv")):
        flips = pd.read_csv(path)
        flips = flips[flips["label"] == "all"].set_index("direction")
        injection[pv] = {d: (None if pd.isna(flips.loc[d, "flip_rate"])   # nothing eligible
                             else round(float(flips.loc[d, "flip_rate"]), 4)) for d in TARGET}

    checks = {"label": label, "half_b_auc": half_b_auc, "half_b_auc_diff": half_b_diff,
              "half_b_target": half_b_target, "civil_target": civil_target,
              "civil_stack": civil_stack, "unchanged_labels_max_logit_diff": unchanged,
              "injection_flip_rate": injection}
    verdict = acceptance(checks, t["max_other_drop"])
    civil_tables = {r["model"]: r for r in shift["domains"]["civil_shift"]["table"]}
    results.update({
        "chosen": v2_pv, "checks": checks, "rules": verdict, "accepted": all(verdict.values()),
        "civil_auc": {v1_pv: _civil_auc(shift_dir, v1_pv, label),
                      v2_pv: civil_tables[f"{v2_pv}"][f"auc_{label}"]},
        "civil_stack_mean_auc": {
            "stack_v1": _civil_stack_mean(shift_dir, v1_pv, cfg["stack"]["version"]),
            "stack_v2": civil_tables[cfg["stack"]["version"]]["mean_auc"]},
    })
    (v2_dir / "v2_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def _shift_table(shift_dir: Path, pv: str) -> dict[str, Any]:
    results = json.loads((shift_dir / f"shift_results_{pv}.json").read_text())
    return {r["model"]: r for r in results["domains"]["civil_shift"]["table"]}


def _civil_auc(shift_dir: Path, pv: str, label: str) -> float:
    return _shift_table(shift_dir, pv)[pv][f"auc_{label}"]


def _civil_stack_mean(shift_dir: Path, pv: str, stack_name: str) -> float:
    return _shift_table(shift_dir, pv)[stack_name]["mean_auc"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Obscene prompt follow-up (p2-zs-v2).")
    parser.add_argument("what", choices=["halves", "evaluate"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_halves(cfg) if args.what == "halves" else run_evaluate(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
