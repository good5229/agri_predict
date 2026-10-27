import json

import pytest

from agri_data import radish_functional_sharing as fhs
from agri_data import radish_functional_sharing_recovery as recovery


def _fixture_protocol(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    recovery_dir = tmp_path / "recovery"
    original_dir.mkdir()
    old_sources = {"functionalSharing": "main-old", "scorer": "scorer-old", "rhs": "rhs-pin"}
    new_sources = {**old_sources, "scorer": "scorer-reviewed"}
    inputs = {"prices": "prices-pin"}
    runtime = {"python": "3.11.4", "numpy": "1.26.4", "scipy": "1.10.1",
               "scikitLearn": "1.3.0", "statsmodels": "0.14.0"}
    body = {"sourcesSha256": old_sources, "inputsSha256": inputs, "runtime": runtime,
            "scoreProtocol": {"familySize": 60}}
    protocol_sha = recovery.sha256_bytes(recovery.canonical(body))
    monkeypatch.setattr(recovery, "ORIGINAL_PROTOCOL_SHA256", protocol_sha)
    monkeypatch.setattr(recovery, "APPROVED_SCORER_SHA256", "scorer-reviewed")
    protocol = {"protocol": body, "protocolSha256": protocol_sha}
    (original_dir / "protocol.json").write_bytes(recovery.canonical(protocol))
    manifest = {"phase": "V", "status": "answer_free_locked", "protocolSha256": protocol_sha,
                "jointFitDiagnostics": [{"fitId": f"radish-2026-0{month}-01"} for month in (4, 5, 6)]}
    manifest_path = original_dir / "validation-manifest.json"
    manifest_path.write_bytes(recovery.canonical(manifest))
    attempt_id = "attempt-v"
    claim = {"protocolSha256": protocol_sha, "attemptId": attempt_id}
    (original_dir / "validation-claim.json").write_bytes(recovery.canonical(claim))
    terminal = {"status": "complete_answer_free", "protocolSha256": protocol_sha,
                "attemptId": attempt_id, "manifestSha256": recovery.file_sha(manifest_path)}
    (original_dir / "validation-terminal.json").write_bytes(recovery.canonical(terminal))
    for name in ("validation-predictions.csv", "fit-diagnostics.json"):
        (original_dir / name).write_bytes(name.encode())
    fit_ids = recovery.fit_ids_from_manifest(manifest)
    for fit_id in fit_ids:
        prefix = f"v-{fit_id}"
        path = original_dir / "fit-checkpoints" / f"{prefix}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(fit_id.encode())
        (path.parent / f"{prefix}.claim.json").write_bytes(b"claim")
        (path.parent / f"{prefix}.terminal.json").write_bytes(b"terminal")

    current_body = {**body, "sourcesSha256": new_sources}
    monkeypatch.setattr(recovery, "_current_body", lambda: json.loads(recovery.canonical(current_body)))
    monkeypatch.setattr(recovery, "_source_hashes", lambda: new_sources.copy())
    monkeypatch.setattr(recovery, "_input_hashes", lambda: inputs.copy())
    monkeypatch.setattr(recovery, "_runtime", lambda: runtime.copy())
    monkeypatch.setattr(recovery, "_v_artifact_hashes", lambda d, m: {"manifest": recovery.file_sha(manifest_path)})
    return original_dir, recovery_dir, body, manifest, old_sources, new_sources, inputs, runtime


def test_recovery_freeze_pins_one_explicit_scorer_transition_and_original_artifacts(tmp_path, monkeypatch):
    original_dir, recovery_dir, body, manifest, old_sources, new_sources, inputs, runtime = _fixture_protocol(tmp_path, monkeypatch)
    doc = recovery.freeze_recovery(original_dir, recovery_dir, approved_scorer_sha256="scorer-reviewed")
    frozen = doc["recovery"]
    assert frozen["originalProtocolSha256"] == recovery.sha256_bytes(recovery.canonical(body))
    assert frozen["originalSourceSha256"] == old_sources
    assert frozen["currentSourceSha256"] == new_sources
    assert frozen["approvedSourceTransition"] == {"module": "scorer", "fromSha256": "scorer-old", "toSha256": "scorer-reviewed"}
    assert frozen["inputSha256"] == inputs and frozen["runtime"] == runtime
    assert frozen["fitIds"] == recovery.fit_ids_from_manifest(manifest)
    assert frozen["fitCount"] == 0
    assert (recovery_dir / "recovery-protocol.json").exists()
    assert not (original_dir / "validation-report.json").exists()


def test_recovery_freeze_rejects_unapproved_scorer_or_other_source_drift(tmp_path, monkeypatch):
    original_dir, recovery_dir, body, *_ = _fixture_protocol(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exact Astra-reviewed recovery source"):
        recovery.freeze_recovery(original_dir, recovery_dir, approved_scorer_sha256="different")
    changed = {**body, "sourcesSha256": {**body["sourcesSha256"], "rhs": "changed"}}
    monkeypatch.setattr(recovery, "_current_body", lambda: changed)
    with pytest.raises(ValueError, match="beyond the approved scorer change"):
        recovery.freeze_recovery(original_dir, recovery_dir, approved_scorer_sha256="scorer-reviewed")


def test_score_time_rejects_self_rehashed_unapproved_scorer_transition(tmp_path, monkeypatch):
    original_dir, recovery_dir, *_ = _fixture_protocol(tmp_path, monkeypatch)
    recovery.freeze_recovery(original_dir, recovery_dir, approved_scorer_sha256="scorer-reviewed")
    recovery_path = recovery_dir / "recovery-protocol.json"
    altered = json.loads(recovery_path.read_text())
    altered_body = altered["recovery"]
    altered_body["approvedSourceTransition"]["toSha256"] = "scorer-evil"
    altered_body["currentSourceSha256"]["scorer"] = "scorer-evil"
    altered["recoveryProtocolSha256"] = recovery.sha256_bytes(recovery.canonical(altered_body))
    recovery_path.write_bytes(recovery.canonical(altered))
    with pytest.raises(ValueError, match="not the exact approved scorer allowlist"):
        recovery._verify_recovery_lineage(original_dir, recovery_dir)


def test_frozen_protocol_adapter_restores_even_when_verifier_raises(monkeypatch):
    original = fhs._protocol_body
    body = {"source": "frozen"}
    with pytest.raises(RuntimeError, match="synthetic verifier failure"):
        with recovery._frozen_protocol_adapter(body):
            assert fhs._protocol_body() == body
            raise RuntimeError("synthetic verifier failure")
    assert fhs._protocol_body is original


def test_recovery_scores_only_to_separate_artifacts_and_is_idempotent(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    recovery_dir = tmp_path / "recovery"
    original_dir.mkdir()
    recovery_dir.mkdir()
    body = {"originalProtocolSha256": "protocol", "originalVArtifactsSha256": {"validation-manifest.json": "manifest"},
            "attemptId": "attempt", "fitIds": ["fit-1"], "fitCount": 0}
    recovery_doc = {"recovery": body, "recoveryProtocolSha256": "recovery-protocol"}
    monkeypatch.setattr(recovery, "_verify_recovery_lineage",
                        lambda _original, _recovery: (recovery_doc, body, {"status": "answer_free_locked"}))
    original_protocol = {"protocol": {"frozen": True}, "protocolSha256": "protocol"}
    monkeypatch.setattr(recovery, "_original_protocol", lambda _path: original_protocol)
    verify_fn = fhs.verify_phase
    monkeypatch.setattr(fhs, "verify_phase", lambda phase, path: {"phase": phase, "status": "answer_free_locked"})
    report = {"vGateMetrics": {"passed": False}, "synthetic": "report"}
    monkeypatch.setattr(recovery, "_score_report", lambda _manifest: report)
    try:
        first = recovery.recover_score(original_dir, recovery_dir)
        second = recovery.recover_score(original_dir, recovery_dir)
    finally:
        monkeypatch.setattr(fhs, "verify_phase", verify_fn)
    assert first["report"] == second["report"] == report
    assert first["terminal"]["fitCount"] == 0 and first["terminal"]["vGatePassed"] is False
    assert (recovery_dir / "recovered-validation-report.json").exists()
    assert (recovery_dir / "recovery-terminal.json").exists()
    assert not (original_dir / "validation-report.json").exists()
