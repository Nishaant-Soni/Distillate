"""Bias detection and mitigation for the toxicity baseline (Ruchita).

Inputs (CONTRACTS.md section 2.2):
    data/processed/wikipedia/train.parquet
    data/processed/wikipedia/val.parquet
    data/processed/civil/bias_eval.parquet      (identity-annotated Civil test rows)

Outputs:
    sample_weight written into data/processed/wikipedia/train.parquet
    reports/bias/bias_report.md
    reports/bias/bias_metrics.json

Steps:
    1. Train a TF-IDF + logistic regression toxicity baseline on Wikipedia train.
    2. Slice Wikipedia validation (identity mention, length band, obfuscation) and the
       Civil identity rows (each identity attribute >= 0.5, plus one slice per group).
    3. Data level: toxicity rate per slice against the overall rate.
       Model level: Fairlearn MetricFrame with FPR, FNR, AUC and size. Only slices with
       at least ``min_slice_size`` rows are judged; FPR >= overall + margin is flagged.
    4. Mitigation: reweight non-toxic identity-mention training rows by each factor and
       keep the smallest that brings the validation identity slice under the margin with
       an AUC drop of at most ``max_auc_drop``. The decision threshold is chosen on
       validation for every factor.
    5. Report before and after on the Civil identity rows only.

The slice functions take a prediction file (id, p_<label>, model_version) plus gold
labels, so Phase 2 reuses them on student predictions with no changes.

    python -m scripts.bias --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from scripts.alerts import send_alert
from scripts.utils import atomic_path, atomic_write, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "bias"
MODEL_VERSION = "tfidf-lr-bias-v1"

# Civil Comments identity columns, grouped as the plan asks (Section 4, Ruchita).
IDENTITY_GROUPS: dict[str, list[str]] = {
    "gender": ["female", "male", "transgender", "other_gender"],
    "sexual_orientation": [
        "homosexual_gay_or_lesbian", "bisexual", "heterosexual", "other_sexual_orientation",
    ],
    "religion": [
        "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    ],
    "race_or_ethnicity": ["black", "white", "asian", "latino", "other_race_or_ethnicity"],
    "disability": [
        "physical_disability", "intellectual_or_learning_disability",
        "psychiatric_or_mental_illness", "other_disability",
    ],
}
IDENTITY_SLICE = "identity_mention"


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def load_lexicon(path: str | Path) -> list[str]:
    """Read identity terms: one per line, lowercase; blank lines and # comments skipped."""
    terms = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        term = line.strip().lower()
        if term and not term.startswith("#"):
            terms.append(term)
    if not terms:
        raise ValueError(f"No identity terms found in {path}")
    # Longest first, so "mentally ill" is tried before any shorter overlapping term.
    return sorted(set(terms), key=lambda t: (-len(t), t))


def compile_lexicon(terms: list[str]) -> re.Pattern[str]:
    """One case-insensitive word-boundary regex for all terms ("trans" never hits "transport")."""
    alternation = "|".join(r"\s+".join(map(re.escape, t.split())) for t in terms)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def mentions_identity(texts: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    """True where a text mentions any identity term."""
    return texts.fillna("").map(lambda t: pattern.search(t) is not None).astype(bool)


# ---------------------------------------------------------------------------
# Baseline model and prediction files
# ---------------------------------------------------------------------------

def train_baseline(
    texts: pd.Series,
    y: pd.Series,
    bias_cfg: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit TF-IDF + logistic regression for one label."""
    tfidf = bias_cfg["tfidf"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf["max_features"],
        ngram_range=(1, tfidf["ngram_max"]),
        min_df=tfidf["min_df"],
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(texts.fillna(""))
    logreg = bias_cfg["logreg"]
    model = LogisticRegression(C=logreg["C"], max_iter=logreg["max_iter"], random_state=seed)
    model.fit(features, np.asarray(y), sample_weight=sample_weight)
    return vectorizer, model


def score(model: tuple[TfidfVectorizer, LogisticRegression], texts: pd.Series) -> np.ndarray:
    """Probability of the positive class for each text."""
    vectorizer, clf = model
    return clf.predict_proba(vectorizer.transform(texts.fillna("")))[:, 1]


def predictions_frame(
    ids: pd.Series, scores: np.ndarray, label: str, labels: list[str]
) -> pd.DataFrame:
    """A prediction file in the contract format. Labels the baseline doesn't score are NaN."""
    frame = pd.DataFrame({"id": ids.astype(str).to_numpy()})
    for name in labels:
        values = scores if name == label else np.full(len(frame), np.nan)
        frame[f"p_{name}"] = np.asarray(values, dtype="float32")
    frame["model_version"] = MODEL_VERSION
    return frame


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The threshold on a 0.05..0.95 grid that maximizes F1 (ties go to the lowest)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y_true, scores >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------

def wikipedia_slices(df: pd.DataFrame, length_bands: list[int]) -> dict[str, pd.Series]:
    """Boolean masks (indexed by id) for identity mention, length band and obfuscation."""
    ids = df["id"].astype(str).to_numpy()

    def mask(values: Any) -> pd.Series:
        return pd.Series(np.asarray(values, dtype=bool), index=ids)

    identity = df["has_identity_term"].astype(bool).to_numpy()
    obfuscated = df["has_obfuscation"].astype(bool).to_numpy()
    slices = {
        IDENTITY_SLICE: mask(identity),
        "no_identity_mention": mask(~identity),
        "obfuscation": mask(obfuscated),
        "no_obfuscation": mask(~obfuscated),
    }
    edges = [0, *sorted(length_bands)]
    char_len = df["char_len"].to_numpy()
    for low, high in zip(edges, [*edges[1:], None], strict=True):
        if high is None:
            slices[f"length_{low}+"] = mask(char_len >= low)
        else:
            slices[f"length_{low}_{high - 1}"] = mask((char_len >= low) & (char_len < high))
    return slices


def civil_slices(df: pd.DataFrame, threshold: float) -> dict[str, pd.Series]:
    """One mask per identity attribute (>= threshold) and one per group (any attribute)."""
    ids = df["id"].astype(str).to_numpy()
    slices: dict[str, pd.Series] = {}
    for group, columns in IDENTITY_GROUPS.items():
        present = [c for c in columns if c in df.columns]
        if not present:
            continue
        group_mask = np.zeros(len(df), dtype=bool)
        for column in present:
            hit = (pd.to_numeric(df[column], errors="coerce").fillna(0.0) >= threshold).to_numpy()
            slices[f"{group}:{column}"] = pd.Series(hit, index=ids)
            group_mask |= hit
        slices[f"{group}:any"] = pd.Series(group_mask, index=ids)
    return slices


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negatives = np.asarray(y_true) == 0
    return float(np.asarray(y_pred)[negatives].mean()) if negatives.any() else float("nan")


def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = np.asarray(y_true) == 1
    return float(1 - np.asarray(y_pred)[positives].mean()) if positives.any() else float("nan")


def _size(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int(len(y_true))


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def slice_report(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    slices: dict[str, pd.Series],
    threshold: float,
    label: str,
    min_size: int,
    margin_pts: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-slice data-level and model-level metrics for one prediction file.

    Returns (table, overall). Each table row has the slice size, its toxicity rate and
    gap to overall (data level), FPR, FNR and AUC from a Fairlearn MetricFrame, the FPR
    gap to overall in points, and whether the slice was judged and flagged.
    """
    merged = gold[["id", label]].assign(id=lambda d: d["id"].astype(str)).merge(
        predictions[["id", f"p_{label}"]].assign(id=lambda d: d["id"].astype(str)),
        on="id", how="inner", validate="one_to_one",
    )
    if len(merged) != len(gold):
        raise ValueError(
            f"Prediction file covers {len(merged)} of {len(gold)} gold ids for {label}"
        )
    y_true = merged[label].to_numpy().astype(int)
    y_score = merged[f"p_{label}"].to_numpy().astype(float)
    y_pred = (y_score >= threshold).astype(int)

    overall = {
        "n": int(len(merged)),
        "toxicity_rate": float(y_true.mean()),
        "fpr": _fpr(y_true, y_pred),
        "fnr": _fnr(y_true, y_pred),
        "auc": _auc(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    rows = []
    for name, slice_mask in slices.items():
        in_slice = slice_mask.reindex(merged["id"], fill_value=False).to_numpy(dtype=bool)
        row: dict[str, Any] = {"slice": name, "n": int(in_slice.sum())}
        if row["n"] == 0:
            rows.append({**row, "toxicity_rate": np.nan, "fpr": np.nan, "fnr": np.nan,
                         "auc": np.nan, "judged": False, "flagged": False})
            continue
        groups = pd.Series(np.where(in_slice, "in", "out"), name="slice")
        rates = MetricFrame(
            metrics={"fpr": _fpr, "fnr": _fnr, "size": _size},
            y_true=y_true, y_pred=y_pred, sensitive_features=groups,
        ).by_group.loc["in"]
        auc = MetricFrame(
            metrics={"auc": _auc}, y_true=y_true, y_pred=y_score, sensitive_features=groups,
        ).by_group.loc["in", "auc"]
        judged = row["n"] >= min_size
        fpr = float(rates["fpr"])
        flagged = bool(judged and not np.isnan(fpr)
                       and (fpr - overall["fpr"]) * 100 >= margin_pts)
        rows.append({**row, "toxicity_rate": float(y_true[in_slice].mean()), "fpr": fpr,
                     "fnr": float(rates["fnr"]), "auc": float(auc),
                     "judged": bool(judged), "flagged": flagged})

    table = pd.DataFrame(rows)
    table["toxicity_rate_gap_pts"] = (table["toxicity_rate"] - overall["toxicity_rate"]) * 100
    table["fpr_gap_pts"] = (table["fpr"] - overall["fpr"]) * 100
    return table, overall


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------

def reweight(train: pd.DataFrame, factor: float, label: str) -> pd.Series:
    """sample_weight: ``factor`` on non-toxic identity-mention rows, 1.0 everywhere else."""
    target = train["has_identity_term"].astype(bool) & (train[label] == 0)
    weights = np.where(target.to_numpy(), float(factor), 1.0).astype("float32")
    return pd.Series(weights, index=train.index, name="sample_weight")


def select_factor(
    trials: list[dict[str, Any]], margin_pts: float, max_auc_drop: float
) -> tuple[float, bool]:
    """Smallest factor whose identity FPR gap is under the margin and AUC drop is allowed.

    ``trials`` must include factor 1 (the unweighted baseline). Marks each trial with
    ``passes``. Returns (factor, True), or (1, False) when no factor passes.
    """
    ordered = sorted(trials, key=lambda t: t["factor"])
    base = next(t for t in ordered if t["factor"] == 1)
    chosen = None
    for trial in ordered:
        gap = trial["identity_fpr_gap_pts"]
        under_margin = gap is not None and not np.isnan(gap) and gap < margin_pts
        auc_ok = base["val_auc"] - trial["val_auc"] <= max_auc_drop + 1e-12
        trial["auc_drop"] = base["val_auc"] - trial["val_auc"]
        trial["passes"] = bool(under_margin and auc_ok)
        if trial["passes"] and chosen is None:
            chosen = trial["factor"]
    return (chosen, True) if chosen is not None else (1, False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x: Any) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _num(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def _pts(x: Any) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.1f}"


def _status(row: pd.Series) -> str:
    if not row["judged"]:
        return "too small"
    return "**flagged**" if row["flagged"] else "ok"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _slice_table(table: pd.DataFrame) -> str:
    rows = [[r["slice"], str(r["n"]), _pct(r["toxicity_rate"]), _pts(r["toxicity_rate_gap_pts"]),
             _pct(r["fpr"]), _pts(r["fpr_gap_pts"]), _pct(r["fnr"]), _num(r["auc"]), _status(r)]
            for _, r in table.iterrows()]
    return _md_table(["Slice", "n", "Toxic rate", "Rate gap (pts)", "FPR", "FPR gap (pts)",
                      "FNR", "AUC", "Status"], rows)


def _overall_line(overall: dict[str, float]) -> str:
    return (f"n = {overall['n']}, toxic rate {_pct(overall['toxicity_rate'])}, "
            f"threshold {overall['threshold']:.2f}, AUC {_num(overall['auc'])}, "
            f"F1 {_num(overall['f1'])}, FPR {_pct(overall['fpr'])}, FNR {_pct(overall['fnr'])}")


def build_report(result: dict[str, Any], bias_cfg: dict[str, Any]) -> str:
    """Markdown report. Interpretation goes in by hand after the first real run."""
    before, after = result["civil_before"], result["civil_after"]
    flagged_before = before["table"].query("flagged")["slice"].tolist()
    flagged_after = after["table"].query("flagged")["slice"].tolist()

    trial_rows = [[str(t["factor"]), _num(t["val_auc"], 4), _num(t["auc_drop"], 4),
                   _pts(t["identity_fpr_gap_pts"]), f"{t['threshold']:.2f}",
                   "yes" if t["passes"] else "no"] for t in result["trials"]]

    compare = before["table"][["slice", "n", "fpr", "fpr_gap_pts", "judged", "flagged"]].merge(
        after["table"][["slice", "fpr", "fpr_gap_pts", "flagged"]],
        on="slice", suffixes=("_before", "_after"),
    )
    compare_rows = [[r["slice"], str(r["n"]), _pct(r["fpr_before"]), _pct(r["fpr_after"]),
                     _pts(r["fpr_gap_pts_before"]), _pts(r["fpr_gap_pts_after"]),
                     ("too small" if not r["judged"] else
                      ("**flagged**" if r["flagged_before"] else "ok") + " → "
                      + ("**flagged**" if r["flagged_after"] else "ok"))]
                    for _, r in compare.iterrows()]

    chosen = result["chosen_factor"]
    mitigation = (f"Chosen factor: **{chosen}**." if result["mitigation_passed"] else
                  "No factor met both conditions, so no reweighting was applied (factor 1).")

    return "\n\n".join([
        "# Bias report: TF-IDF + logistic regression toxicity baseline",
        f"Model `{MODEL_VERSION}`, label `{result['label']}`. Slices with at least "
        f"{bias_cfg['min_slice_size']} rows are judged; a slice is flagged when its FPR is "
        f"{bias_cfg['fpr_flag_margin_pts']:.0f} or more points above the overall FPR. The "
        "decision threshold maximizes F1 on Wikipedia validation.",
        "## 1. Data-level and model-level slices before mitigation",
        "### Civil identity rows (bias set)",
        _overall_line(before["overall"]),
        _slice_table(before["table"]),
        "### Wikipedia validation",
        _overall_line(result["wiki_before"]["overall"]),
        _slice_table(result["wiki_before"]["table"]),
        "## 2. Biases found",
        ("Flagged on the Civil identity rows before mitigation: "
         + (", ".join(f"`{s}`" for s in flagged_before) if flagged_before else "none") + "."),
        "## 3. Mitigation",
        f"Non-toxic Wikipedia training rows that mention an identity term were reweighted by "
        f"each factor in {bias_cfg['reweight_factors']}. A factor passes when the validation "
        f"`{IDENTITY_SLICE}` slice's FPR gap is under {bias_cfg['fpr_flag_margin_pts']:.0f} "
        f"points and validation AUC drops by at most {bias_cfg['max_auc_drop']}. The smallest "
        f"passing factor is kept. {mitigation}",
        _md_table(["Factor", "Val AUC", "AUC drop", "Identity FPR gap (pts)", "Threshold",
                   "Passes"], trial_rows),
        "## 4. Before and after on the Civil identity rows",
        f"Before: {_overall_line(before['overall'])}",
        f"After: {_overall_line(after['overall'])}",
        _md_table(["Slice", "n", "FPR before", "FPR after", "Gap before (pts)",
                   "Gap after (pts)", "Status"], compare_rows),
        ("Still flagged after mitigation: "
         + (", ".join(f"`{s}`" for s in flagged_after) if flagged_after else "none") + "."),
        "## 5. Trade-off",
        f"AUC {_num(before['overall']['auc'])} → {_num(after['overall']['auc'])}, "
        f"F1 {_num(before['overall']['f1'])} → {_num(after['overall']['f1'])} "
        "on the Civil identity rows.",
        "## 6. Limits",
        "- **Approximate lexicon.** Wikipedia slices rely on word matching against "
        "`config/identity_terms.txt`, which misses identity references it doesn't list and "
        "matches innocent uses (\"black car\").\n"
        "- **Rater bias.** Gold labels come from crowd raters, whose own biases are in the "
        "labels the model is measured against.\n"
        "- **No age or location fields.** Neither dataset records them, so those groups "
        "can't be checked.\n"
        "- **No per-group thresholds.** They would need the commenter's identity at "
        "prediction time, which the service doesn't have and shouldn't infer, and they "
        "treat groups differently by design. One threshold plus reweighting is used instead.",
    ]) + "\n"


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(table.to_json(orient="records"))


def _clean(value: Any) -> Any:
    """NaN -> None and numpy scalars -> Python, so the JSON is valid."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def _read(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in {path.parent.as_posix()}; run the upstream stage first"
        )
    df = pd.read_parquet(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df.reset_index(drop=True)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Detect slice bias, pick a reweighting factor, write sample_weight and the report."""
    bias_cfg = cfg["bias"]
    label = bias_cfg["label"]
    labels = cfg["labels"]["names"]
    seed = cfg["seed"]
    margin, min_size = bias_cfg["fpr_flag_margin_pts"], bias_cfg["min_slice_size"]
    processed = Path(cfg["paths"]["processed_dir"])
    train_path = processed / "wikipedia" / "train.parquet"
    val_path = processed / "wikipedia" / "val.parquet"
    civil_path = processed / "civil" / "bias_eval.parquet"
    out_dir = Path(cfg["paths"]["reports_dir"]) / "bias"

    summary: dict[str, Any] = {"task": TASK_ID}
    with timed(summary):
        train = _read(train_path, ["id", "text", label])
        val = _read(val_path, ["id", "text", label, "char_len", "has_obfuscation"])
        civil = _read(civil_path, ["id", "text", label])

        pattern = compile_lexicon(load_lexicon(cfg["paths"]["identity_terms"]))
        for name, df in (("train", train), ("val", val)):
            if "has_identity_term" not in df.columns:
                logger.warning("%s has no has_identity_term column; computing it from the "
                               "lexicon", name)
                df["has_identity_term"] = mentions_identity(df["text"], pattern)

        wiki_masks = wikipedia_slices(val, bias_cfg["length_bands"])
        civil_masks = civil_slices(civil, bias_cfg["identity_threshold"])
        factors = sorted({1, *bias_cfg["reweight_factors"]})

        trials, fitted = [], {}
        for factor in factors:
            weights = reweight(train, factor, label).to_numpy()
            model = train_baseline(train["text"], train[label], bias_cfg, seed, weights)
            val_scores = score(model, val["text"])
            threshold = choose_threshold(val[label].to_numpy().astype(int), val_scores)
            preds = predictions_frame(val["id"], val_scores, label, labels)
            table, overall = slice_report(preds, val, wiki_masks, threshold, label,
                                          min_size, margin)
            gap = table.loc[table["slice"] == IDENTITY_SLICE, "fpr_gap_pts"].iloc[0]
            trials.append({"factor": factor, "val_auc": overall["auc"],
                           "identity_fpr_gap_pts": float(gap), "threshold": threshold})
            fitted[factor] = {"model": model, "threshold": threshold,
                              "wiki": {"table": table, "overall": overall}}
            logger.info("factor=%s val_auc=%.4f identity_fpr_gap=%+.1f pts threshold=%.2f",
                        factor, overall["auc"], gap, threshold)

        chosen, passed = select_factor(trials, margin, bias_cfg["max_auc_drop"])

        civil_results = {}
        for tag, factor in (("civil_before", 1), ("civil_after", chosen)):
            entry = fitted[factor]
            preds = predictions_frame(civil["id"], score(entry["model"], civil["text"]),
                                      label, labels)
            table, overall = slice_report(preds, civil, civil_masks, entry["threshold"],
                                          label, min_size, margin)
            civil_results[tag] = {"table": table, "overall": overall}

        result = {"label": label, "trials": trials, "chosen_factor": chosen,
                  "mitigation_passed": passed, "wiki_before": fitted[1]["wiki"],
                  "wiki_after": fitted[chosen]["wiki"], **civil_results}

        # Write sample_weight into train.parquet, re-read so only that column changes.
        train_file = pd.read_parquet(train_path)
        train_file["sample_weight"] = reweight(train, chosen, label).to_numpy()
        with atomic_path(train_path) as tmp:
            train_file.to_parquet(tmp, index=False)

        atomic_write(out_dir / "bias_report.md", build_report(result, bias_cfg))
        metrics = {
            "model_version": MODEL_VERSION, "label": label, "chosen_factor": chosen,
            "mitigation_passed": passed, "trials": trials,
            **{key: {"overall": result[key]["overall"], "slices": _records(result[key]["table"])}
               for key in ("wiki_before", "wiki_after", "civil_before", "civil_after")},
        }
        atomic_write(out_dir / "bias_metrics.json",
                     json.dumps(_clean(metrics), indent=2, sort_keys=True))

        flagged_before = civil_results["civil_before"]["table"].query("flagged")["slice"].tolist()
        flagged_after = civil_results["civil_after"]["table"].query("flagged")["slice"].tolist()
        if not passed:
            send_alert("warn", "Bias mitigation: no reweighting factor passed",
                       {"factors": factors, "margin_pts": margin})
        if flagged_after:
            send_alert("warn", "Bias slices still flagged on the Civil identity rows",
                       {"slices": ", ".join(flagged_after)})

        summary.update({
            "rows_in": {"train": len(train), "val": len(val), "bias_eval": len(civil)},
            "rows_out": len(train_file),
            "chosen_factor": chosen,
            "mitigation_passed": passed,
            "reweighted_rows": int((train_file["sample_weight"] != 1.0).sum()),
            "threshold": fitted[chosen]["threshold"],
            "val_auc": {"before": trials[0]["val_auc"],
                        "after": fitted[chosen]["wiki"]["overall"]["auc"]},
            "civil_auc": {"before": civil_results["civil_before"]["overall"]["auc"],
                          "after": civil_results["civil_after"]["overall"]["auc"]},
            "slices_judged": int(civil_results["civil_before"]["table"]["judged"].sum()),
            "flagged_before": flagged_before,
            "flagged_after": flagged_after,
        })
    logger.info("bias: chosen factor %s, flagged before %s, after %s",
                chosen, flagged_before, flagged_after)
    return _clean(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = run(cfg)
    write_summary(TASK_ID, summary, cfg)
    print(json.dumps(summary, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
