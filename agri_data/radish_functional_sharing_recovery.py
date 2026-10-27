"""Score-only recovery for an immutable FHS answer-free V build.

This adapter permits one reviewed scorer-source transition. It never fits or
rewrites the original phase artifacts; the recovered report is stored in a
separate directory bound to the original run and to this adapter's source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import statsmodels

from . import radish_functional_sharing as fhs

ORIGINAL_PROTOCOL_SHA256 = "3aa5b1a6511c9414b534a743d07496670c0926af9458dd0d0f4447d9f61f3959"
APPROVED_SCORER_SHA256 = "0a0f87157b1797451a1853ed5a97e702c702ef109c87e182f045b94d9933ea3d"
RECOVERY_SCHEMA = 1
RECOVERY_DIRNAME = "score-recovery-v1"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_or_identical(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing recovery artifact differs: {path}")
        return
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikitLearn": sklearn.__version__, "statsmodels": statsmodels.__version__}


def _original_protocol(original_dir: Path) -> dict[str, Any]:
    path = original_dir / "protocol.json"
    doc = _json(path)
    if doc.get("protocolSha256") != ORIGINAL_PROTOCOL_SHA256:
        raise ValueError("original FHS protocol does not match the authorized V run")
    if doc.get("protocolSha256") != sha256_bytes(canonical(doc.get("protocol"))):
        raise ValueError("original FHS protocol digest mismatch")
    return doc


def _v_artifact_hashes(original_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    names = ["protocol.json", "validation-claim.json", "validation-terminal.json", "validation-manifest.json",
             "validation-predictions.csv", "fit-diagnostics.json"]
    if (original_dir / "validation-score-failure.json").is_file():
        names.append("validation-score-failure.json")
    fit_ids = [str(item["fitId"]) for item in manifest.get("jointFitDiagnostics", [])]
    if len(fit_ids) != 3 or len(set(fit_ids)) != 3:
        raise ValueError("V answer-free manifest does not contain the exact three joint fits")
    for fit_id in fit_ids:
        prefix = f"v-{fit_id}"
        names.extend((f"fit-checkpoints/{prefix}.claim.json", f"fit-checkpoints/{prefix}.json",
                      f"fit-checkpoints/{prefix}.terminal.json"))
    result = {}
    for name in names:
        path = original_dir / name
        if not path.is_file():
            raise ValueError(f"required original V artifact is missing: {name}")
        result[name] = file_sha(path)
    return result


def _current_body() -> dict[str, Any]:
    return json.loads(canonical(fhs._protocol_body()))


def _source_hashes() -> dict[str, str]:
    return {name: file_sha(path) for name, path in fhs._source_paths().items()}


def _input_hashes() -> dict[str, str]:
    return {name: file_sha(path) for name, path in fhs._input_paths().items()}


def _recovery_body(original_dir: Path, recovery_dir: Path, *, approved_scorer_sha256: str) -> dict[str, Any]:
    original = _original_protocol(original_dir)
    pinned = original["protocol"]
    current = _current_body()
    current_sources = dict(current.get("sourcesSha256", {}))
    old_sources = dict(pinned.get("sourcesSha256", {}))
    if set(current_sources) != set(old_sources) or "scorer" not in old_sources:
        raise ValueError("FHS source inventory differs from original protocol")
    if approved_scorer_sha256 != APPROVED_SCORER_SHA256:
        raise ValueError("scorer SHA is not the exact Astra-reviewed recovery source")
    if old_sources["scorer"] == approved_scorer_sha256:
        raise ValueError("recovery requires an actual approved scorer-source transition")
    current_scorer = _source_hashes().get("scorer")
    if approved_scorer_sha256 != current_scorer:
        raise ValueError("current scorer source does not match the explicitly approved SHA-256")
    expected = json.loads(canonical(pinned))
    expected["sourcesSha256"]["scorer"] = approved_scorer_sha256
    if current != expected:
        raise ValueError("runtime, inputs, model sources, or frozen protocol behavior drifted beyond the approved scorer change")
    if _runtime() != pinned.get("runtime"):
        raise ValueError("runtime differs from the original FHS protocol")

    manifest_path = original_dir / "validation-manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("phase") != "V" or manifest.get("status") != "answer_free_locked":
        raise ValueError("original V manifest is absent or not answer-free locked")
    if manifest.get("protocolSha256") != original["protocolSha256"]:
        raise ValueError("original V manifest is not bound to the original protocol")
    terminal = _json(original_dir / "validation-terminal.json")
    claim = _json(original_dir / "validation-claim.json")
    if (terminal.get("status") != "complete_answer_free" or terminal.get("attemptId") != claim.get("attemptId") or
            terminal.get("protocolSha256") != original["protocolSha256"] or
            terminal.get("manifestSha256") != file_sha(manifest_path) or claim.get("protocolSha256") != original["protocolSha256"]):
        raise ValueError("original V claim/terminal does not bind the immutable manifest")

    sources = _source_hashes()
    if sources != {**old_sources, "scorer": approved_scorer_sha256}:
        raise ValueError("current source hashes differ from the single approved scorer transition")
    inputs = _input_hashes()
    if inputs != pinned.get("inputsSha256"):
        raise ValueError("current input hashes differ from the original FHS protocol")
    artifacts = _v_artifact_hashes(original_dir, manifest)
    recovery_code = Path(__file__).resolve()
    if recovery_dir.resolve() == original_dir.resolve():
        raise ValueError("recovery artifacts must be stored outside the original run directory")
    return {"schemaVersion": RECOVERY_SCHEMA, "kind": "score_only_recovery", "phase": "V",
            "recoveryId": str(uuid.uuid4()), "originalOutputDir": str(original_dir.resolve()),
            "recoveryOutputDir": str(recovery_dir.resolve()), "originalProtocolSha256": original["protocolSha256"],
            "originalMainSourceSha256": old_sources["functionalSharing"], "originalSourceSha256": old_sources,
            "currentSourceSha256": sources, "approvedSourceTransition": {"module": "scorer",
                "fromSha256": old_sources["scorer"], "toSha256": approved_scorer_sha256},
            "inputSha256": inputs, "runtime": _runtime(), "originalVArtifactsSha256": artifacts,
            "attemptId": claim["attemptId"], "fitIds": fit_ids_from_manifest(manifest),
            "recoveryImplementationSha256": file_sha(recovery_code),
            "verificationAdapter": "temporarily supply the exact frozen protocol body to the unchanged FHS verify_phase; restore in finally",
            "fitCount": 0}


def fit_ids_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    return [str(item["fitId"]) for item in manifest.get("jointFitDiagnostics", [])]


def freeze_recovery(original_dir: Path = fhs.OUT, recovery_dir: Path | None = None, *,
                    approved_scorer_sha256: str = APPROVED_SCORER_SHA256) -> dict[str, Any]:
    original_dir = Path(original_dir).resolve()
    recovery_dir = (Path(recovery_dir) if recovery_dir is not None else original_dir / RECOVERY_DIRNAME).resolve()
    body = _recovery_body(original_dir, recovery_dir, approved_scorer_sha256=approved_scorer_sha256)
    doc = {"recovery": body, "recoveryProtocolSha256": sha256_bytes(canonical(body))}
    if recovery_dir.exists() and any(recovery_dir.iterdir()) and not (recovery_dir / "recovery-protocol.json").exists():
        raise ValueError("score-recovery directory already contains unbound artifacts")
    _write_once_or_identical(recovery_dir / "recovery-protocol.json", canonical(doc))
    return doc


@contextmanager
def _frozen_protocol_adapter(protocol_body: Mapping[str, Any]) -> Iterator[None]:
    """Narrow compatibility bridge for the unchanged FHS verifier only."""
    original = fhs._protocol_body
    frozen_body = json.loads(canonical(protocol_body))
    fhs._protocol_body = lambda: frozen_body
    try:
        yield
    finally:
        fhs._protocol_body = original


def _load_recovery(recovery_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = recovery_dir / "recovery-protocol.json"
    doc = _json(path)
    body = doc.get("recovery")
    if doc.get("recoveryProtocolSha256") != sha256_bytes(canonical(body)):
        raise ValueError("score-recovery protocol digest mismatch")
    if body.get("schemaVersion") != RECOVERY_SCHEMA or body.get("kind") != "score_only_recovery" or body.get("phase") != "V":
        raise ValueError("unsupported score-recovery protocol")
    if body.get("recoveryImplementationSha256") != file_sha(Path(__file__).resolve()):
        raise ValueError("score-recovery implementation differs from its frozen protocol")
    return doc, body


def _verify_recovery_lineage(original_dir: Path, recovery_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recovery_doc, body = _load_recovery(recovery_dir)
    if str(original_dir.resolve()) != body.get("originalOutputDir") or str(recovery_dir.resolve()) != body.get("recoveryOutputDir"):
        raise ValueError("score-recovery protocol directory binding mismatch")
    original = _original_protocol(original_dir)
    pinned = original["protocol"]
    if original["protocolSha256"] != body.get("originalProtocolSha256"):
        raise ValueError("original protocol changed after recovery freeze")
    if body.get("originalMainSourceSha256") != pinned.get("sourcesSha256", {}).get("functionalSharing"):
        raise ValueError("original FHS implementation lineage mismatch")
    current = _current_body()
    expected = json.loads(canonical(pinned))
    transition = body.get("approvedSourceTransition", {})
    if (transition.get("module") != "scorer" or transition.get("fromSha256") != pinned.get("sourcesSha256", {}).get("scorer") or
            transition.get("toSha256") != body.get("currentSourceSha256", {}).get("scorer")):
        raise ValueError("score-recovery scorer transition is invalid")
    if transition.get("toSha256") != APPROVED_SCORER_SHA256 or transition.get("fromSha256") == APPROVED_SCORER_SHA256:
        raise ValueError("score-recovery transition is not the exact approved scorer allowlist entry")
    expected["sourcesSha256"]["scorer"] = transition["toSha256"]
    if current != expected or _runtime() != pinned.get("runtime"):
        raise ValueError("current protocol body/runtime differs beyond the approved scorer transition")
    sources = _source_hashes()
    if sources != body.get("currentSourceSha256") or sources.get("functionalSharing") != body.get("originalMainSourceSha256"):
        raise ValueError("current source hashes differ from recovery lineage")
    inputs = _input_hashes()
    if inputs != body.get("inputSha256") or inputs != pinned.get("inputsSha256"):
        raise ValueError("current input hashes differ from recovery lineage")
    manifest = _json(original_dir / "validation-manifest.json")
    artifacts = _v_artifact_hashes(original_dir, manifest)
    if artifacts != body.get("originalVArtifactsSha256"):
        raise ValueError("an original V artifact changed after score-recovery freeze")
    if fit_ids_from_manifest(manifest) != body.get("fitIds"):
        raise ValueError("V fit IDs differ from score-recovery protocol")
    if manifest.get("protocolSha256") != original["protocolSha256"] or manifest.get("status") != "answer_free_locked":
        raise ValueError("original V manifest is not eligible for score-only recovery")
    claim = _json(original_dir / "validation-claim.json")
    if claim.get("attemptId") != body.get("attemptId"):
        raise ValueError("original V attempt differs from recovery protocol")
    return recovery_doc, body, manifest


def _score_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    from . import equal_info_online_forecasts as base
    from . import radish_functional_sharing_score as scorer
    actuals = fhs.rhs._actuals_by_product_target(_json(base.PRICES_PATH))
    return scorer.score_manifest(manifest, baseline_rows=fhs._baseline_rows_for_scorer("V"),
                                 actuals=actuals, fit_diagnostics=manifest["fitDiagnostics"], phase="V")


def recover_score(original_dir: Path = fhs.OUT, recovery_dir: Path | None = None) -> dict[str, Any]:
    original_dir = Path(original_dir).resolve()
    recovery_dir = (Path(recovery_dir) if recovery_dir is not None else original_dir / RECOVERY_DIRNAME).resolve()
    recovery_doc, body, _ = _verify_recovery_lineage(original_dir, recovery_dir)
    original_protocol = _original_protocol(original_dir)
    # The adapter is process-local, narrow, restored even on failure, and allows
    # only the already-validated original protocol body while the frozen verifier runs.
    with _frozen_protocol_adapter(original_protocol["protocol"]):
        verified_manifest = fhs.verify_phase("V", original_dir)
    if verified_manifest.get("status") != "answer_free_locked":
        raise ValueError("verified original V manifest is not answer-free locked")
    _verify_recovery_lineage(original_dir, recovery_dir)
    report = _score_report(verified_manifest)
    # Recheck all pins immediately before committing the recovered score.
    _verify_recovery_lineage(original_dir, recovery_dir)
    report_path = recovery_dir / "recovered-validation-report.json"
    terminal_path = recovery_dir / "recovery-terminal.json"
    report_bytes = canonical(report)
    _write_once_or_identical(report_path, report_bytes)
    terminal = {"kind": "score_only_recovery", "phase": "V", "status": "scored",
                "recoveryProtocolSha256": recovery_doc["recoveryProtocolSha256"],
                "originalProtocolSha256": body["originalProtocolSha256"],
                "originalManifestSha256": body["originalVArtifactsSha256"]["validation-manifest.json"],
                "attemptId": body["attemptId"], "fitCount": 0,
                "reportSha256": sha256_bytes(report_bytes),
                "vGatePassed": bool(report.get("vGateMetrics", {}).get("passed"))}
    _write_once_or_identical(terminal_path, canonical(terminal))
    return {"recovery": recovery_doc, "report": report, "terminal": terminal}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "score"))
    parser.add_argument("--original-dir", type=Path, default=fhs.OUT)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--approved-scorer-sha256")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_recovery(args.original_dir, args.recovery_dir,
                                 approved_scorer_sha256=args.approved_scorer_sha256 or APPROVED_SCORER_SHA256)
    else:
        result = recover_score(args.original_dir, args.recovery_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
