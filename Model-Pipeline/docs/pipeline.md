# The model pipeline: train, validate, select, report

Work plan v4 §5.1. One entry point, one image, the same steps everywhere.

## `python -m src.pipeline.run`

For the student in the config (`configs/student.yaml` plus `--set` overrides):

| Step | What it does | Output in the run folder |
|---|---|---|
| 1. Data | The evaluation tables through `src.data.load_data`: files on disk, a DVC revision (`pipeline.data_rev`), or a folder in the same layout (`pipeline.data_dir`). Read **before** training, so a missing table fails in seconds | `pipeline.json` → `data` (hashes, rows) |
| 2. Train | `src.student.train`: model, thresholds, validation predictions, the §2.5 package | `model/`, `package/`, `summary.json` |
| 3. Validate | Prediction files for gate, H2, bias and obfuscation; gold files (ids and labels, no text); every metric (`src.eval.metrics`) at the tuned thresholds | `predictions/`, `gold/`, `eval/metrics.json` |
| 4. Bias check and gate | `src.eval.gate` with `configs/gate.yaml`, no production model | `eval/gate_report.{json,md}` |
| 5. Latency | The package's batch-1 CPU latency, tokenization included, on real comments (`pipeline.latency_threads`, default 2) | `latency.json` |
| 6. Plots | Per-label confusion matrices at the tuned thresholds; per-label AUC and PR-AUC by split | `plots/` |
| 7. MLflow | All of the above into the training run | |

A step that fails after training is recorded in `pipeline.json` (`errors`); the steps before
it are kept, and the exit code is 1.

## Selection (`python -m src.student.select_best`)

Among the runs that pass the **bias check** (the gate's slice-FPR limits on the Civil bias set)
and the **CPU latency budget** (`select.latency_stat` p95 at most `select.latency_budget_ms`,
50 ms, batch size 1), the highest **validation** mean AUC. Runs within `select.auc_tie`
(0.0002; one pair of runs at one seed differed by 0.00015, so this is a lower bound on
run-to-run noise, not its size, `docs/tuning.md`) count as tied, and the
higher validation mean PR-AUC wins. The gate set and H2 are never used to choose.
`select.candidates` limits who competes (the other runs stay in the table as controls), and
`select.latency_enforced: false` reports the budget without excluding anyone; `select.note`
records why. Phase 2 uses both (decided after the run, `reports/student/phase2/README.md`).
Writes `selection.json` and `selection.md` (every candidate, eligible or not, with the reason).

## Results across arms (`python -m src.eval.results`)

The model card's H1 (gate) and H2 (Civil, five labels) tables with 1,000-resample bootstrap
intervals; paired bootstrap differences for mixed vs gold, stack vs TF-IDF (the part creditable
to the LLM), calibrated Qwen vs TF-IDF, stack vs calibrated Qwen, and mixed vs stack; the
validation-to-gate drop of mixed vs gold (the soft-label leak check); obfuscation recall against
the same comments' originals; latency; the truncation sweep. Plots: per-label AUC (dots) and
PR-AUC (bars) across arms per split, the H1 / H2 figure, the selected arm's confusion matrices.

## Where it runs

* **Kaggle** (the reported run): `kaggle/13_student_ablation_all.ipynb`. It builds the evaluation
  tables from the two competitions by the id lists (`src/student/eval_sets.py`, the data
  pipeline's own obfuscation script), runs every arm through `src.pipeline.run` (two at a time
  on T4 x2), then latency, selection, truncation and results, and zips everything small into
  `phase2_results.zip` (no comment text, no weights).
* **Docker** (CI, Vertex): `docker build -f Model-Pipeline/Dockerfile -t distillate-train .` from
  the repository root; exact versions in `requirements-pipeline.txt`. Smoke test:
  `docker run --rm --entrypoint python distillate-train -m pytest tests/pipeline -q`.
* **Locally**, offline: `pytest tests/pipeline` runs every step on the tiny model.
