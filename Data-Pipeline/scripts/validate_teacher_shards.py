"""Validate the teacher's label shards before anything reads them (Phase 2).

Runs the checks in the teacher shard spec (Model-Pipeline/docs/teacher_prompts_and_shard_spec.md,
section 6) on every run listed in config teacher.runs, by calling the teacher lane's own validator
(src.teacher.validate_local), so the checks have one implementation. Adds one check of its own:
for kinds that name a frozen artifact, every manifest's fingerprint field must equal that file's
SHA-256 (stack_sha256 for the stack's soft labels, calibration_sha256 for calibrated Qwen).

Training and gate runs are checked against their id lists (training also against the committed
shard plan); drift runs against civil_drift_ids.csv, one shard per batch.

If the shard store can't be reached (no credentials, as on a fresh clone or in CI), the stage warns
and skips instead of failing the DAG, unless teacher.required is true. If the store is reachable
and any check fails, the stage writes its report and fails.

Run on its own: python -m scripts.validate_teacher_shards
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts.alerts import send_alert
from scripts.errors import TeacherShardError
from scripts.utils import PIPELINE_ROOT, atomic_write, get_logger, load_config, timed

logger = get_logger(__name__)

MAX_ERRORS_SHOWN = 10


def resolve(path: str | Path | None) -> Path | None:
    """Paths in the teacher section are relative to Data-Pipeline/."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


def teacher_modules(model_pipeline_dir: Path) -> dict[str, ModuleType]:
    """Import the teacher lane's validator from Model-Pipeline/.

    The folder is appended (not prepended) to sys.path, so it can never shadow this pipeline's
    own packages.
    """
    root = str(model_pipeline_dir)
    if root not in sys.path:
        sys.path.append(root)
    from src.teacher import config, prompts, shards, storage, validate_local

    return {
        "config": config,
        "prompts": prompts,
        "shards": shards,
        "storage": storage,
        "validate": validate_local,
    }


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file, with line endings normalized to LF.

    A Windows checkout can turn LF into CRLF, which would change the hash of an unchanged file.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_manifest_hash(
    manifests: dict[str, dict[str, Any]], field: str, expected: str
) -> list[str]:
    """Every manifest's ``field`` must name the committed artifact's SHA-256."""
    errors = []
    for name in sorted(manifests):
        found = manifests[name].get(field)
        if found != expected:
            errors.append(f"{name}: {field} {found!r}, expected {expected}")
    return errors


def source_ids(source: str, sources: dict[str, Path], shards: ModuleType):
    """The sorted ids a run must cover, and the id -> batch map for drift runs."""
    if source == "drift":
        batches = shards.read_batches(sources["drift"])
        return sorted(batches), batches
    return sorted(shards.read_ids(sources[source])), None


def kind_expectations(
    kinds: dict[str, dict[str, Any]], teacher_model_version: str
) -> dict[str, dict[str, Any]]:
    """For each kind: the expected model_version, and the fingerprint to check (if any)."""
    expected = {}
    for kind, spec in kinds.items():
        model = spec.get("model_version") or teacher_model_version
        entry: dict[str, Any] = {"model_version": model}
        if spec.get("hash_field"):
            entry["hash_field"] = spec["hash_field"]
            entry["hash"] = file_sha256(resolve(spec["hash_file"]))
        expected[kind] = entry
    return expected


def skip(
    summary: dict[str, Any], reason: str, required: bool, report: Path
) -> dict[str, Any]:
    """The store can't be reached: fail if required, otherwise warn, record the skip, and skip."""
    if required:
        raise TeacherShardError(f"teacher shard store unavailable: {reason}")
    logger.warning("Skipping teacher shard validation: %s", reason)
    send_alert("warn", "validate_teacher_shards skipped: store unavailable", {"reason": reason})
    summary["skipped"] = True
    summary["skip_reason"] = reason
    atomic_write(report, json.dumps(summary, indent=2, default=str))
    return summary


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate every configured run. Returns the stage summary."""
    summary: dict[str, Any] = {"task": "validate_teacher_shards", "runs": {}, "dropped": {}}
    with timed(summary):
        tcfg = cfg["teacher"]
        required = bool(tcfg.get("required", False))
        mods = teacher_modules(resolve(tcfg["model_pipeline_dir"]))

        mcfg = mods["config"].load_config(None)
        prompt = mcfg["prompt"]
        expected_prompt = mods["prompts"].prompt_version(
            prompt["protocol"], prompt["shots"], prompt.get("text_version", "v1")
        )
        expected = kind_expectations(tcfg["kinds"], mods["config"].model_version(mcfg))
        shard_size = tcfg.get("shard_size") or mcfg["labeling"]["shard_size"]
        plan_path = resolve(tcfg.get("plan"))
        plan_record = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path else None
        sources = {name: resolve(path) for name, path in tcfg["sources"].items()}

        try:
            storage = mods["storage"].get_storage(
                mcfg, resolve(tcfg.get("local_root")), tcfg.get("backend")
            )
        except ImportError:  # a missing package is a setup error, not an unreachable store
            raise
        except Exception as err:  # no credentials or project: the store is unreachable
            return skip(summary, f"{type(err).__name__}: {err}", required, resolve(tcfg["report"]))

        failed_runs = []
        for index, run_cfg in enumerate(tcfg["runs"]):
            prefix, kind = run_cfg["prefix"], run_cfg["kind"]
            want = expected[kind]
            source, batches = source_ids(run_cfg["source"], sources, mods["shards"])
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    frames, manifests, schemas, errors = mods["validate"].load_run(
                        storage, prefix, Path(tmp)
                    )
                except ImportError:
                    raise
                except Exception as err:
                    if index == 0:  # nothing read yet: treat as an unreachable store
                        return skip(
                            summary, f"{type(err).__name__}: {err}", required,
                            resolve(tcfg["report"]),
                        )
                    raise
                errors += mods["validate"].check_run(
                    frames,
                    manifests,
                    schemas,
                    source,
                    shard_size,
                    expected_prompt,
                    want["model_version"],
                    partial=False,
                    plan_record=plan_record if run_cfg["source"] == "train" else None,
                    batches=batches,
                )
            if "hash_field" in want:
                errors += check_manifest_hash(manifests, want["hash_field"], want["hash"])

            summary["runs"][prefix] = {
                "kind": kind,
                "source": run_cfg["source"],
                "shards": len(set(frames) & set(manifests)),
                "rows": sum(len(frame) for frame in frames.values()),
                "errors": len(errors),
                "first_errors": errors[:MAX_ERRORS_SHOWN],
            }
            logger.info(
                "%s: %d shards, %d rows, %d errors",
                prefix,
                summary["runs"][prefix]["shards"],
                summary["runs"][prefix]["rows"],
                len(errors),
            )
            if errors:
                failed_runs.append(prefix)

        summary["skipped"] = False
        summary["artifact_hashes"] = {
            kind: want["hash"] for kind, want in expected.items() if "hash" in want
        }
        atomic_write(resolve(tcfg["report"]), json.dumps(summary, indent=2, default=str))
        if failed_runs:
            raise TeacherShardError(
                f"{len(failed_runs)} teacher shard run(s) failed validation: {failed_runs}. "
                f"Details in {tcfg['report']}"
            )
    return summary


def main() -> None:
    print(json.dumps(run(load_config()), indent=2, default=str))


if __name__ == "__main__":
    main()
