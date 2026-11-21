"""Helpers for deploy.sh and rollback.sh (work plan v5, section 6.1). Owner: Ruchita.

The shell scripts run gcloud; everything that needs parsing, hashing, the MLflow registry or a
record goes through this module, so it can be unit-tested without GCP:

    resolve    --spec <n|alias>       the model version to deploy (JSON on stdout)
    previous                           the version rollback.sh --previous goes back to
    verify     --package DIR           every file against its SHA-256 in meta.json
    reference  --package DIR           make sure the package has reference_outputs.json
    tag-url    --service-json F --tag T       the URL of a tagged Cloud Run revision
    find-tag   --service-json F --version N   the newest revision tag for model version N
    record     --log PATH ...           append to deployments.jsonl, the job summary and Slack

Reference outputs: the smoke test compares the live service with the package's own scores on
neutral inputs (no comment text, since packages are published). If the package already holds
``reference_outputs.json`` it is used as is; otherwise ``reference`` computes it from the
package's ``model.onnx`` and tokenizer and writes it next to them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")
REFERENCE_FILE = "reference_outputs.json"
REQUIRED = ("model.onnx", "tokenizer", "thresholds.json", "meta.json")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Neutral filler, the same sentences as Model-Pipeline/src/student/package.py PARITY_TEXTS:
# no comment text, so the reference inputs can live in a published package.
FILLER = ("thanks for the edit to the article", "please see the talk page section",
          "the source link in the image note was changed back", "revert",
          "user page fact check")
# Repeats per round: short inputs through ones longer than the 256-token limit.
REPEATS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 80)


class DeployError(RuntimeError):
    """A step can't continue; the message says why."""


# ---------------------------------------------------------------------------
# Reference inputs and outputs
# ---------------------------------------------------------------------------

def reference_texts() -> list[str]:
    """50 neutral inputs of increasing length (5 filler sentences x 10 lengths)."""
    return [" ".join([sentence] * repeat) for repeat in REPEATS for sentence in FILLER]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class OnnxScorer:
    """Scores texts with the package's model.onnx and tokenizer, as serving does."""

    def __init__(self, package: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta = json.loads((package / "meta.json").read_text())
        self._np = np
        self._tokenizer = Tokenizer.from_file(str(package / "tokenizer" / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=int(meta["max_length"]))
        self._tokenizer.no_padding()
        self._session = ort.InferenceSession(str(package / "model.onnx"),
                                             providers=["CPUExecutionProvider"])
        self._labels = tuple(meta.get("labels", LABELS))

    def __call__(self, texts: Sequence[str]) -> list[dict[str, float]]:
        np = self._np
        out = []
        for text in texts:
            enc = self._tokenizer.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            logits = self._session.run(["logits"], {"input_ids": ids,
                                                    "attention_mask": mask})[0][0]
            out.append({label: sigmoid(float(z)) for label, z in zip(self._labels, logits,
                                                                    strict=True)})
        return out


def ensure_reference(package: Path,
                     scorer: Callable[[Sequence[str]], list[dict[str, float]]] | None = None,
                     ) -> tuple[dict[str, Any], bool]:
    """The package's reference outputs, computing and writing them if absent.

    Returns (reference, created).
    """
    path = package / REFERENCE_FILE
    if path.exists():
        reference = json.loads(path.read_text())
        check_reference(reference)
        return reference, False
    meta = json.loads((package / "meta.json").read_text())
    texts = reference_texts()
    scores = (scorer or OnnxScorer(package))(texts)
    reference = {"model_version": meta.get("model_version"), "texts": texts, "scores": scores,
                 "source": "computed by deploy.sh from model.onnx"}
    check_reference(reference)
    path.write_text(json.dumps(reference, indent=2) + "\n")
    return reference, True


def check_reference(reference: dict[str, Any]) -> None:
    texts, scores = reference.get("texts"), reference.get("scores")
    if not texts or not scores or len(texts) != len(scores):
        raise DeployError(f"{REFERENCE_FILE} needs matching, non-empty 'texts' and 'scores'")
    for i, row in enumerate(scores):
        missing = [label for label in LABELS if label not in row]
        if missing:
            raise DeployError(f"{REFERENCE_FILE}: input {i} has no score for {missing}")


# ---------------------------------------------------------------------------
# Package check
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package(folder: Path, expect_model_version: str | None = None) -> list[str]:
    """Problems with a package folder; empty means valid (same rule as package.py)."""
    problems = [f"missing {name}" for name in REQUIRED if not (folder / name).exists()]
    if problems:
        return problems
    meta = json.loads((folder / "meta.json").read_text())
    files = meta.get("files")
    if not files:
        return ["meta.json has no 'files' (SHA-256 per file)"]
    for name, digest in files.items():
        path = folder / name
        if not path.exists():
            problems.append(f"{name} listed in meta.json but missing")
        elif sha256_file(path) != digest:
            problems.append(f"{name}: SHA-256 differs from meta.json")
    if expect_model_version and meta.get("model_version") != expect_model_version:
        problems.append(f"meta.json model_version {meta.get('model_version')!r}, registry says "
                        f"{expect_model_version!r}")
    return problems


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def mlflow_client():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise DeployError("registry unreachable: MLFLOW_TRACKING_URI is not set")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def _alias(client, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception as exc:  # MlflowException when nothing holds the alias
        if "not found" in str(exc).lower() or "RESOURCE_DOES_NOT_EXIST" in str(exc):
            return None
        raise


def resolve(client, name: str, spec: str, allow_unusable: bool = False) -> dict[str, Any]:
    """The version for ``spec`` (a number or an alias) and the current production version."""
    try:
        version = (client.get_model_version(name, spec) if spec.isdigit()
                   else _alias(client, name, spec))
    except Exception as exc:
        raise DeployError(f"registry unreachable or {name} v{spec} not found: {exc}") from exc
    if version is None:
        raise DeployError(f"no version of {name} holds @{spec}")
    tags = version.tags or {}
    problems = []
    if getattr(version, "status", "READY") != "READY":
        problems.append(f"status {version.status}")
    if tags.get("gate_passed") != "true":
        problems.append("the gate didn't pass")
    if tags.get("ar_upload") != "ok":
        problems.append(f"package not published (ar_upload={tags.get('ar_upload')})")
    if problems and not allow_unusable:
        raise DeployError(f"{name} v{version.version} can't be deployed: {'; '.join(problems)}")
    production = _alias(client, name, "production")
    return {"version": str(version.version), "model_version": tags.get("model_version", ""),
            "production_version": str(production.version) if production else "",
            "spec": spec}


def previous(client, name: str) -> dict[str, Any]:
    """The version ``rollback.sh --previous`` returns to (rollback_model's own rule)."""
    sys.path.insert(0, str(REPO_ROOT / "Model-Pipeline"))
    from src.pipeline.rollback_model import previous_version

    current = _alias(client, name, "production")
    if current is None:
        raise DeployError("no version holds @production, so there is no previous one")
    target = previous_version(client, name, str(current.version))
    return {"version": str(target.version), "current": str(current.version),
            "model_version": (target.tags or {}).get("model_version", "")}


# ---------------------------------------------------------------------------
# Cloud Run
# ---------------------------------------------------------------------------

def tagged_routes(service: dict[str, Any]) -> list[dict[str, Any]]:
    """Traffic entries that carry a tag, from ``gcloud run services describe --format=json``."""
    return [t for t in (service.get("status", {}).get("traffic") or []) if t.get("tag")]


def tag_url(service: dict[str, Any], tag: str) -> str:
    for route in tagged_routes(service):
        if route["tag"] == tag and route.get("url"):
            return route["url"]
    raise DeployError(f"no URL for tag {tag!r} on the service")


def _revision_number(name: str) -> int:
    match = re.search(r"-(\d+)-[a-z0-9]+$", name or "")
    return int(match.group(1)) if match else -1


def find_tag(service: dict[str, Any], version: str) -> str:
    """The newest revision tag for model version ``version`` (tags look like v3-abc1234)."""
    prefix = f"v{version}-"
    routes = [r for r in tagged_routes(service) if r["tag"].startswith(prefix)]
    if not routes:
        raise DeployError(f"no Cloud Run revision is tagged for v{version}; deploy it first "
                          f"with deploy.sh --version {version}")
    return max(routes, key=lambda r: _revision_number(r.get("revisionName", "")))["tag"]


# ---------------------------------------------------------------------------
# Record: deployments.jsonl, job summary, Slack
# ---------------------------------------------------------------------------

OK_STATUSES = {"deployed", "rolled-back", "smoke-passed-not-promoted"}


def build_entry(args: argparse.Namespace) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "action": args.action, "status": args.status, "step": args.step,
        "service": args.service, "version": args.version or None,
        "model_version": args.model_version or None, "tag": args.tag or None,
        "image": args.image or None, "from_version": args.from_version or None,
        "detail": args.detail or None,
        "git_commit": os.environ.get("GITHUB_SHA") or _git_commit(),
        "actor": os.environ.get("GITHUB_ACTOR") or os.environ.get("USER"),
    }
    if os.environ.get("GITHUB_RUN_ID"):
        entry["run_url"] = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
                            f"{os.environ['GITHUB_RUN_ID']}")
    if args.smoke and Path(args.smoke).exists():
        smoke = json.loads(Path(args.smoke).read_text())
        entry["smoke"] = {k: smoke.get(k) for k in ("passed", "n_inputs", "max_abs_diff")}
    return {k: v for k, v in entry.items() if v is not None}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


NOT_FOUND = ("not found", "notfound", "404", "no urls matched", "does not exist")
PRECONDITION = ("precondition", "412")


def _gcs_read(log: str, runner) -> tuple[str, int]:
    """(content, generation) of a gs:// object; ("", 0) only when it doesn't exist yet.

    Any other failure (network, permissions) raises, so a transient error can never be taken
    for an empty log and overwrite the history.
    """
    described = runner(["gcloud", "storage", "objects", "describe", log,
                        "--format=value(generation)"], capture_output=True, text=True)
    if described.returncode != 0:
        if any(marker in described.stderr.lower() for marker in NOT_FOUND):
            return "", 0
        raise DeployError(f"could not read {log}: {described.stderr.strip()[-300:]}")
    generation = int(described.stdout.strip())
    content = runner(["gcloud", "storage", "cat", f"{log}#{generation}"], capture_output=True,
                     text=True)
    if content.returncode != 0:
        raise DeployError(f"could not read {log}: {content.stderr.strip()[-300:]}")
    return content.stdout, generation


def append_log(log: str, entry: dict[str, Any], runner=subprocess.run, attempts: int = 3) -> None:
    """Append one JSON line to a local file or a gs:// object.

    For gs://, the write only succeeds if the object is still the generation that was read
    (``--if-generation-match``; 0 means "doesn't exist yet"), so two writers at the same time
    can't drop each other's line: the loser re-reads and tries again.
    """
    line = json.dumps(entry, sort_keys=True) + "\n"
    if not log.startswith("gs://"):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(line)
        return
    for _ in range(attempts):
        body, generation = _gcs_read(log, runner)
        if body and not body.endswith("\n"):
            body += "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            tmp.write(body + line)
        try:
            result = runner(["gcloud", "storage", "cp", tmp.name, log,
                             f"--if-generation-match={generation}"], capture_output=True,
                            text=True)
        finally:
            os.unlink(tmp.name)
        if result.returncode == 0:
            return
        if not any(marker in result.stderr.lower() for marker in PRECONDITION):
            raise DeployError(f"could not write {log}: {result.stderr.strip()[-300:]}")
    raise DeployError(f"{log} kept changing under another writer ({attempts} tries)")


def summary_markdown(entry: dict[str, Any]) -> str:
    icon = "✅" if entry["status"] in OK_STATUSES else "❌"
    rows = [f"| {k} | {entry[k]} |" for k in ("status", "step", "version", "model_version",
                                             "tag", "from_version", "image", "detail")
            if entry.get(k)]
    if "smoke" in entry:
        rows.append(f"| smoke test | {json.dumps(entry['smoke'])} |")
    return "\n".join([f"## {icon} {entry['action']}: {entry['status']}", "",
                      "| | |", "|---|---|", *rows, ""]) + "\n"


def post_slack(entry: dict[str, Any]) -> bool:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    icon = ":white_check_mark:" if entry["status"] in OK_STATUSES else ":x:"
    lines = [f"{icon} *{entry['action']} {entry['status']}* ({entry.get('service', '')})"]
    lines += [f"• {k}: {entry[k]}" for k in ("version", "model_version", "tag", "from_version",
                                              "detail", "run_url") if entry.get(k)]
    request = urllib.request.Request(url, data=json.dumps({"text": "\n".join(lines)}).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # a broken webhook must never fail a deploy
        print(f"warning: Slack post failed: {exc}", file=sys.stderr)
        return False


def record(args: argparse.Namespace) -> dict[str, Any]:
    entry = build_entry(args)
    print(json.dumps(entry, sort_keys=True))
    try:
        append_log(args.log, entry)
    except Exception as exc:
        print(f"warning: deployments log not written: {exc}", file=sys.stderr)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary_markdown(entry))
    post_slack(entry)
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("resolve")
    p.add_argument("--model", default="distillate-student")
    p.add_argument("--spec", required=True, help="a version number, or an alias")
    p.add_argument("--allow-unusable", action="store_true")

    p = sub.add_parser("previous")
    p.add_argument("--model", default="distillate-student")

    p = sub.add_parser("verify")
    p.add_argument("--package", required=True)
    p.add_argument("--expect-model-version", default=None)

    p = sub.add_parser("reference")
    p.add_argument("--package", required=True)

    for name in ("tag-url", "find-tag"):
        p = sub.add_parser(name)
        p.add_argument("--service-json", required=True)
        p.add_argument("--tag" if name == "tag-url" else "--version", required=True)

    p = sub.add_parser("record")
    p.add_argument("--log", required=True, help="a local path or gs:// URI")
    p.add_argument("--action", default="deploy", choices=["deploy", "rollback"])
    p.add_argument("--status", required=True)
    for flag in ("--step", "--service", "--version", "--model-version", "--tag", "--image",
                 "--from-version", "--detail", "--smoke"):
        p.add_argument(flag, default="")

    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            print(json.dumps(resolve(mlflow_client(), args.model, args.spec,
                                     args.allow_unusable)))
        elif args.command == "previous":
            print(json.dumps(previous(mlflow_client(), args.model)))
        elif args.command == "verify":
            problems = verify_package(Path(args.package), args.expect_model_version)
            for problem in problems:
                print(f"package problem: {problem}", file=sys.stderr)
            return 1 if problems else 0
        elif args.command == "reference":
            reference, created = ensure_reference(Path(args.package))
            print(f"{REFERENCE_FILE}: {len(reference['texts'])} inputs "
                  f"({'computed from model.onnx' if created else 'from the package'})")
        elif args.command in ("tag-url", "find-tag"):
            service = json.loads(Path(args.service_json).read_text())
            print(tag_url(service, args.tag) if args.command == "tag-url"
                  else find_tag(service, args.version))
        elif args.command == "record":
            record(args)
    except DeployError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
