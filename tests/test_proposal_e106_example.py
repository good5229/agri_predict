import ast
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "proposal/build_idea_proposal.py"


def _loader_namespace(tmp_path):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_verified_radish_example")
    namespace = {"hashlib": hashlib, "json": json, "datetime": datetime, "timedelta": timedelta,
                 "PROJECT": tmp_path, "SystemExit": SystemExit}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _artifact(tmp_path):
    run_id = "20261010T120014-01ac7655"
    run_dir = tmp_path / "outputs/daily/2026-10-10" / run_id
    run_dir.mkdir(parents=True)
    cutoff = "2026-10-09"
    prices = {"schemaVersion": 1, "meta": {"evaluationEnd": cutoff}, "products": []}
    for product in ("cabbage", "radish", "onion", "garlic"):
        prices["products"].append({"id": product, "observations": [
            {"date": cutoff, "actual": 150.0, "volumeKg": 20.0, "marketCodes": ["110001", "110008"]}]})
    prices_raw = json.dumps(prices).encode()
    (run_dir / "prices.json").write_bytes(prices_raw)
    provenance = [{"date": cutoff, "market": market, "product": product,
                   "jobId": 1, "epochId": f"{market}-{product}", "responseAt": "2026-10-10T00:00:00+00:00",
                   "rowCount": 2, "selectedPrevious": False}
                  for product in ("cabbage", "radish", "onion", "garlic")
                  for market in ("110001", "110008")]
    provenance_raw = (json.dumps(provenance) + "\n").encode()
    (run_dir / "selected-provenance.json").write_bytes(provenance_raw)
    selection_raw = b'{"selectionSha256":"synthetic"}\n'
    selection_path = tmp_path / "outputs/experiments/trend-response-correction-20261010/selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_bytes(selection_raw)
    source = {"pricesPath": str((run_dir / "prices.json").relative_to(tmp_path)),
              "provenancePath": str((run_dir / "selected-provenance.json").relative_to(tmp_path)),
              "pricesFileSha256": hashlib.sha256(prices_raw).hexdigest(),
              "provenanceFileSha256": hashlib.sha256(provenance_raw).hexdigest(),
              "provenanceCanonicalSha256": hashlib.sha256(
                  (json.dumps(provenance, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest(),
              "selectionFileSha256": hashlib.sha256(selection_raw).hexdigest(),
              "dailyTrainingRunId": run_id, "dailyTrainingSnapshotRecordSha256": "snapshot-sha",
              "dailyTrainingFinishedAtKst": "2026-10-10T12:00:00+09:00",
              "dailyTrainingCutoff": cutoff, "sourceCutoff": cutoff, "noTargetOutcomesRead": True}
    forecasts = [{"product": product, "horizon": h, "target": (datetime.fromisoformat("2026-10-11").date() +
                   timedelta(days=h)).isoformat(), "p0": 150.0, "p0Date": cutoff,
                   "E106": {"status": "issued", "prediction": 150 + h}}
                 for product in ("cabbage", "radish", "onion", "garlic") for h in range(1, 11)]
    issue = {"schemaVersion": 1, "phase": "prospective_forecast_only", "targetValuesAttached": False,
             "issueDateKst": "2026-10-11", "issueAtKst": "2026-10-11T12:00:00+09:00",
             "methods": ["causal50", "E", "E106"],
             "products": ["cabbage", "radish", "onion", "garlic"], "sources": source,
             "sourceAvailability": {"radish": {"lastObservedPriceDate": cutoff,
                 "sourceStates90d": [{"date": cutoff, "state": "complete_rows",
                     "selectedSources": {m: {"rowCount": 2, "jobId": 1,
                        "epochId": f"{m}-radish", "responseAt": "2026-10-10T00:00:00+00:00",
                        "selectedPrevious": False} for m in ("110001", "110008")}}]}},
             "forecasts": forecasts}
    canonical = (json.dumps(issue, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False) + "\n").encode()
    issue["forecastSha256"] = hashlib.sha256(canonical).hexdigest()
    artifact_raw = (json.dumps(issue, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    artifact_path = tmp_path / "outputs/prospective-e106/2026-10-11/forecast.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(artifact_raw)
    latest = {"runId": run_id, "snapshotSha256": "snapshot-sha", "cutoff": cutoff,
              "runDirectory": str(run_dir.relative_to(tmp_path)),
              "selectedProvenance": str((run_dir / "selected-provenance.json").relative_to(tmp_path)),
              "finishedAtKst": "2026-10-10T12:00:00+09:00"}
    status_raw = (json.dumps({"latestSuccess": latest}) + "\n").encode()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/daily-training-status.json").write_bytes(status_raw)
    run_status = {**latest, "status": "complete"}
    run_status_raw = json.dumps(run_status).encode()
    (run_dir / "run-status.json").write_bytes(run_status_raw)
    index = {"artifact": "outputs/prospective-e106/2026-10-11/forecast.json",
             "artifactSha256": hashlib.sha256(artifact_raw).hexdigest(),
             "canonicalSha256": issue["forecastSha256"], "issueDateKst": issue["issueDateKst"],
             "issueAtKst": issue["issueAtKst"], "dailyTrainingRunId": run_id,
             "dailyTrainingSnapshotRecordSha256": "snapshot-sha",
             "runStatusFileSha256": hashlib.sha256(run_status_raw).hexdigest(),
             "sourceCutoffDate": cutoff, "pricesFileSha256": source["pricesFileSha256"],
             "provenanceFileSha256": source["provenanceFileSha256"],
             "provenanceCanonicalSha256": source["provenanceCanonicalSha256"],
             "selectionFileSha256": source["selectionFileSha256"]}
    return index, artifact_path, run_dir


def test_loader_checks_pinned_prospective_artifact_and_source_lineage(tmp_path):
    namespace = _loader_namespace(tmp_path)
    index, _, _ = _artifact(tmp_path)
    namespace["EXAMPLE_FORECAST"] = index
    returned_index, issue, product, predictions, flagged = namespace["load_verified_radish_example"]()
    assert returned_index == index
    assert issue["sourceCutoffDate"] == "2026-10-09"
    assert product["lastObservedPrice"] == 150.0
    assert product["method"] == "E106"
    assert len(predictions) == 10 and predictions[0]["target"] == "2026-10-12"
    assert predictions[-1]["predictedPrice"] == 160
    assert flagged is None


def test_loader_rejects_changed_price_bytes(tmp_path):
    namespace = _loader_namespace(tmp_path)
    index, _, run_dir = _artifact(tmp_path)
    namespace["EXAMPLE_FORECAST"] = index
    (run_dir / "prices.json").write_text("{}")
    with pytest.raises(SystemExit, match="price snapshot file hash"):
        namespace["load_verified_radish_example"]()


def test_loader_rejects_duplicate_forecast_record_even_when_grid_is_complete(tmp_path):
    namespace = _loader_namespace(tmp_path)
    index, artifact_path, _ = _artifact(tmp_path)
    issue = json.loads(artifact_path.read_bytes())
    issue["forecasts"].append(dict(issue["forecasts"][0]))
    unsigned = dict(issue)
    unsigned.pop("forecastSha256")
    canonical = (json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False) + "\n").encode()
    issue["forecastSha256"] = hashlib.sha256(canonical).hexdigest()
    payload = (json.dumps(issue, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    artifact_path.write_bytes(payload)
    index["artifactSha256"] = hashlib.sha256(payload).hexdigest()
    index["canonicalSha256"] = issue["forecastSha256"]
    namespace["EXAMPLE_FORECAST"] = index
    with pytest.raises(SystemExit, match="exactly 40 forecast records"):
        namespace["load_verified_radish_example"]()


def test_loader_is_independent_of_mutable_latest_success_status(tmp_path):
    namespace = _loader_namespace(tmp_path)
    index, _, _ = _artifact(tmp_path)
    namespace["EXAMPLE_FORECAST"] = index
    (tmp_path / "runs/daily-training-status.json").write_text('{"latestSuccess":{"runId":"later-run"}}')
    returned_index, _, product, predictions, flagged = namespace["load_verified_radish_example"]()
    assert returned_index == index
    assert product["productId"] == "radish"
    assert len(predictions) == 10 and flagged is None
