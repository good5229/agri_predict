import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agri_data import e106_prospective_eval as ev

KST = ZoneInfo("Asia/Seoul")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ev.pretty(value))
    return path


def _forecast(issue_date=dt.date(2026, 10, 11), issue_at="2026-10-11T15:19:46+09:00"):
    rows = []
    for product in ev.PRODUCTS:
        for horizon in ev.HORIZONS:
            rows.append({"product": product, "origin": issue_date.isoformat(),
                "target": (issue_date + dt.timedelta(days=horizon)).isoformat(), "horizon": horizon,
                "sourceCutoff": (issue_date - dt.timedelta(days=1)).isoformat(),
                **{m: {"status": "issued", "prediction": 100.0 + horizon,
                       "fallback": False, "fallbackReason": None, "reason": None} for m in ev.METHODS}})
    doc = {"schemaVersion": 1, "phase": "prospective_forecast_only", "issueDateKst": issue_date.isoformat(),
        "issueAtKst": issue_at, "targetValuesAttached": False, "methods": list(ev.METHODS),
        "products": list(ev.PRODUCTS), "horizons": list(ev.HORIZONS), "plannedCount": 40,
        "sources": {"sourceCutoff": (issue_date-dt.timedelta(days=1)).isoformat(),
                    "noTargetOutcomesRead": True, "dailyTrainingRunId": "run-1",
                    "dailyTrainingSnapshotRecordSha256": "snap", "pricesFileSha256": "prices",
                    "provenanceFileSha256": "provenance"}, "forecasts": rows}
    doc["forecastSha256"] = ev.sha(ev.canonical(doc))
    return doc


def _registry(registry, project, *, issue_date=dt.date(2026, 10, 11), monkeypatch=None):
    doc = _forecast(issue_date)
    artifact_rel = "outputs/prospective-e106/2026-10-11/validated-runtime/forecast.json"
    artifact = _write_json(project / artifact_rel, doc)
    raw_sha = ev.sha(artifact.read_bytes())
    unsigned = dict(doc); embedded = unsigned.pop("forecastSha256")
    canonical_sha = ev.sha(ev.canonical(unsigned))
    if monkeypatch:
        monkeypatch.setattr(ev, "INITIAL_RAW_SHA256", raw_sha)
        monkeypatch.setattr(ev, "INITIAL_CANONICAL_SHA256", canonical_sha)
        monkeypatch.setattr(ev, "INITIAL_RUN_ID", "run-1")
        monkeypatch.setattr(ev, "INITIAL_RUN_STATUS_SHA256", "status")
    initial = {"issueDateKst": issue_date.isoformat(), "attemptId": "initial-e106-success",
        "startedAtKst": doc["issueAtKst"], "artifactTarget": artifact_rel,
        "artifactRawSha256": raw_sha, "artifactCanonicalSha256": canonical_sha,
        "source": {"runId": "run-1", "snapshotSha256": "snap", "cutoff": "2026-10-10",
                   "pricesFileSha256": "prices", "provenanceFileSha256": "provenance",
                   "runStatusFileSha256": ev.INITIAL_RUN_STATUS_SHA256}}
    protocol = ev._protocol_payload({**initial, "artifactRawSha256": ev.INITIAL_RAW_SHA256,
        "artifactCanonicalSha256": ev.INITIAL_CANONICAL_SHA256})
    protocol["protocolSha256"] = ev.sha(ev.canonical(protocol))
    _write_json(registry / "protocol.json", protocol)
    day = registry / "days" / issue_date.isoformat()
    _write_json(day / "claim.json", {"schemaVersion": 1, "issueDateKst": issue_date.isoformat(),
        "attemptId": initial["attemptId"], "startedAtKst": initial["startedAtKst"],
        "attemptOrdinal": 1, "plannedForecastKeyCount": 40,
        "plannedForecastGridSha256": ev.grid_sha256(issue_date), "source": initial["source"],
        "artifactTarget": artifact_rel, "bootstrapPin": True})
    _write_json(day / "terminal.json", {"schemaVersion": 1, "attemptId": initial["attemptId"],
        "status": "complete", "artifactPath": artifact_rel, "artifactRawSha256": raw_sha,
        "artifactCanonicalSha256": canonical_sha, "bootstrapPin": True})
    return doc


def _truth(end=dt.date(2026, 10, 21), missing_scope=None):
    actuals, states = {p: {} for p in ev.PRODUCTS}, {}
    for p in ev.PRODUCTS:
        for d in (dt.date(2026, 10, 12) + dt.timedelta(days=i) for i in range(10)):
            actuals[p][d] = 100.0
            for market in ("110001", "110008"):
                if missing_scope == (p, d, market):
                    continue
                states[(p, d, market)] = {"rowCount": 1, "state": "complete", "epochId": "e",
                                          "responseAt": "2026-10-22T00:00:00+09:00", "jobId": 1,
                                          "selectedPrevious": False}
    return {"runId": "truth", "runStatusSha256": "status", "snapshotSha256": "snap",
        "pricesSha256": "price", "provenanceSha256": "prov", "provenanceCanonicalSha256": "prov-c",
        "dataEnd": end.isoformat(), "evaluationEnd": end.isoformat(),
        "finishedAtKst": "2026-10-22T00:00:00+09:00", "actuals": actuals, "scopeStates": states,
        "prices": {"products": []}}


def _pair(product, horizon, target, issue="2026-10-11", mature=True):
    methods = {m: {"status": "issued", "prediction": 101.0, "fallback": False,
                   "fallbackReason": None, "reason": None} for m in ev.METHODS}
    return {"issueDateKst": issue, "origin": issue, "product": product, "horizon": horizon,
        "target": target.isoformat(), "cohortMature10d": mature, "truthState": "positive_actual_observed",
        "actual": 100.0, "sourceScopes": {}, "methods": methods,
        "pairs": {name: {"status": "paired_positive_actual", "aPrediction": 101.0, "bPrediction": 100.0}
                  for name in ev.COMPARISONS}}


def test_grid_and_calendar_scheduling_are_frozen():
    issue = dt.date(2026, 10, 11)
    assert len(ev._grid(issue)) == 40
    assert ev.grid_sha256(issue) == ev.sha(ev.canonical(ev._grid(issue)))
    assert ev._scheduled_dates(dt.datetime(2026, 10, 12, 17, 59, tzinfo=KST)) == [issue]
    assert ev._scheduled_dates(dt.datetime(2026, 10, 12, 18, 0, tzinfo=KST)) == [issue, dt.date(2026, 10, 12)]
    assert ev._scheduled_dates(dt.datetime(2026, 10, 11, 15, 18, tzinfo=KST)) == []
    assert ev._scheduled_dates(dt.datetime(2026, 10, 11, 15, 20, tzinfo=KST)) == [issue]


def test_forecast_reader_rejects_bool_horizon_and_modified_digest(tmp_path):
    doc = _forecast()
    doc["forecasts"][0]["horizon"] = True
    unsigned = dict(doc); unsigned.pop("forecastSha256")
    doc["forecastSha256"] = ev.sha(ev.canonical(unsigned))
    path = _write_json(tmp_path / "forecast.json", doc)
    with pytest.raises(ValueError, match="40-key grid"):
        ev._forecast_doc(path)


def test_target_state_preserves_truth_and_source_states():
    truth = _truth()
    now = dt.datetime(2026, 10, 25, 12, tzinfo=KST)
    day = dt.date(2026, 10, 12)
    assert ev._target_state("garlic", day, truth, now)[0] == "positive_actual_observed"
    assert ev._target_state("garlic", now.date() + dt.timedelta(days=1), truth, now)[0] == "target_not_yet_arrived"
    short = _truth(end=dt.date(2026, 10, 20))
    assert ev._target_state("garlic", dt.date(2026, 10, 21), short, now)[0] == "truth_source_extent_not_reached"
    incomplete = _truth(missing_scope=("garlic", day, "110008"))
    assert ev._target_state("garlic", day, incomplete, now)[0] == "source_error_or_incomplete"
    empty = _truth(); empty["actuals"]["garlic"].pop(day)
    empty["scopeStates"][("garlic", day, "110001")]["rowCount"] = 0
    empty["scopeStates"][("garlic", day, "110008")]["rowCount"] = 0
    assert ev._target_state("garlic", day, empty, now)[0] == "normal_complete_no_eligible_price"
    unclassified = _truth(); unclassified["actuals"]["garlic"].pop(day)
    assert ev._target_state("garlic", day, unclassified, now)[0] == "complete_rows_no_eligible_price_unclassified"
    contradictory = _truth(); contradictory["scopeStates"][("garlic", day, "110001")]["rowCount"] = 0
    contradictory["scopeStates"][("garlic", day, "110008")]["rowCount"] = 0
    assert ev._target_state("garlic", day, contradictory, now)[0] == "source_error_or_incomplete"


def test_partial_pairs_are_retained_and_daily_never_runs_inference(tmp_path, monkeypatch):
    registry, project, output = tmp_path / "registry", tmp_path / "project", tmp_path / "output"
    _registry(registry, project, monkeypatch=monkeypatch)
    truth = _truth()
    monkeypatch.setattr(ev, "_review_inference", lambda *a: pytest.fail("daily mode must not infer"))
    as_of = dt.datetime(2026, 10, 25, 12, tzinfo=KST)
    report = ev.build_report(registry, truth, as_of, output_root=output, project_root=project, mode="daily")
    assert report["status"] == "not_ready"
    assert report["inference"]["status"] == "not_run_before_explicit_review"
    assert len(report["outcomes"]) == 40
    assert all(row["cohortMature10d"] for row in report["outcomes"])
    assert report["completeCase"]["positiveAllTenTargetCaseRows"] == 40
    assert report["comparisonSummaries"]["E106_minus_E"]["fixedMacro"]["status"] == "complete_fixed_cell_macro"
    partial_truth = _truth(missing_scope=("garlic", dt.date(2026, 10, 12), "110008"))
    partial = ev.build_report(registry, partial_truth, as_of, output_root=output, project_root=project)
    assert len(partial["outcomes"]) == 40
    assert sum(row["truthState"] == "source_error_or_incomplete" for row in partial["outcomes"]) == 1
    assert partial["comparisonSummaries"]["E106_minus_E"]["fixedMacro"]["status"] == "incomplete_cell_support"
    assert partial["comparisonSummaries"]["E106_minus_E"]["fixedMacro"]["mapeDifference"] is None


def test_threshold_counts_unique_target_days_not_rows():
    base = dt.date(2026, 10, 20)
    rows = []
    for h in ev.HORIZONS:
        rows += [_pair("garlic", h, base + dt.timedelta(days=i)) for i in range(30)]
        rows.append(_pair("garlic", h, base, issue="2026-11-01"))
    result = ev._thresholds(rows)
    assert result["allHorizonsReady"] is True
    assert all(v["uniquePositivePairedTargetDates"] == 30 for v in result["byGarlicHorizon"].values())


def test_first_ready_month_end_is_pinned_from_observation_date():
    assert ev._eligible_review_date(dt.date(2026, 11, 20)) == dt.date(2026, 11, 30)
    assert ev._eligible_review_date(dt.date(2026, 11, 30)) == dt.date(2026, 11, 30)
    assert ev._eligible_review_date(dt.date(2026, 12, 2)) == dt.date(2026, 12, 31)
    assert ev._eligible_review_date(dt.date(2026, 10, 20)) == dt.date(2026, 11, 30)


def test_first_ready_marker_is_content_and_protocol_bound(tmp_path, monkeypatch):
    registry, project = tmp_path / "registry", tmp_path / "project"
    _registry(registry, project, monkeypatch=monkeypatch)
    marker = {"schemaVersion": 1, "protocolId": ev.PROTOCOL_ID,
        "protocolSha256": ev._load_protocol(registry)["protocolSha256"],
        "firstReadyDetectedAtKst": "2026-11-20T18:00:00+09:00", "truthSource": {"runId": "x"},
        "thresholds": {"allHorizonsReady": True}}
    marker["readinessSha256"] = ev.sha(ev.canonical(marker))
    path = _write_json(registry / "readiness" / "first-ready.json", marker)
    loaded, raw_sha = ev._load_first_ready(registry)
    assert loaded == marker and raw_sha == ev.sha(path.read_bytes())
    assert ev._first_review_window(registry, dt.datetime(2026, 11, 30, 18, tzinfo=KST), True)["eligible"]
    lost = ev._first_review_window(registry, dt.datetime(2026, 11, 30, 18, tzinfo=KST), False)
    assert lost["reason"] == "support_threshold_lost_on_first_eligible_month_end"
    missed = ev._first_review_window(registry, dt.datetime(2026, 12, 31, 18, tzinfo=KST), True)
    assert missed["reason"] == "first_eligible_month_end_was_missed"
    marker["thresholds"]["allHorizonsReady"] = False
    _write_json(path, marker)
    with pytest.raises(ValueError, match="marker is invalid"):
        ev._load_first_ready(registry)


def test_claims_and_terminals_require_schema_first_attempt_and_complete_hashes(tmp_path, monkeypatch):
    registry, project = tmp_path / "registry", tmp_path / "project"
    _registry(registry, project, monkeypatch=monkeypatch)
    issue = dt.date(2026, 10, 11)
    claim_path = registry / "days" / issue.isoformat() / "claim.json"
    claim = json.loads(claim_path.read_bytes())
    claim["plannedForecastGridSha256"] = "wrong"
    _write_json(claim_path, claim)
    with pytest.raises(ValueError, match="immutable first 40-key"):
        protocol = ev._load_protocol(registry)
        ev._load_claim(registry, issue, protocol["initialSelectedIssue"], protocol["protocolSha256"])
    day = registry / "days" / issue.isoformat()
    terminal = json.loads((day / "terminal.json").read_bytes())
    terminal["artifactCanonicalSha256"] = None
    _write_json(day / "terminal.json", terminal)
    valid_claim = {"attemptId": "initial-e106-success"}
    with pytest.raises(ValueError, match="artifactCanonicalSha256"):
        ev._terminal(registry, issue, valid_claim)


def test_future_claim_and_terminal_self_hashes_and_protocol_link(tmp_path):
    registry = tmp_path / "registry"
    issue = dt.date(2026, 10, 12)
    source = {"runId": "run", "snapshotSha256": "snap", "cutoff": "2026-10-11",
              "pricesFileSha256": "prices", "provenanceFileSha256": "prov"}
    protocol_sha = "protocol-sha"
    claim = {"schemaVersion": 1, "protocolSha256": protocol_sha, "issueDateKst": issue.isoformat(),
        "attemptId": "attempt-2", "startedAtKst": "2026-10-12T18:00:00+09:00", "attemptOrdinal": 1,
        "plannedForecastKeyCount": 40, "plannedForecastGridSha256": ev.grid_sha256(issue),
        "source": source, "artifactTarget": "outputs/prospective-e106/2026-10-12/forecast.json"}
    claim["claimSha256"] = ev.sha(ev.canonical(claim))
    day = registry / "days" / issue.isoformat()
    _write_json(day / "claim.json", claim)
    loaded = ev._load_claim(registry, issue, {"issueDateKst": "2026-10-11"}, protocol_sha)
    terminal = {"schemaVersion": 1, "protocolSha256": protocol_sha, "claimSha256": claim["claimSha256"],
        "attemptId": claim["attemptId"], "status": "complete", "finishedAtKst": "2026-10-12T18:01:00+09:00",
        "artifactPath": claim["artifactTarget"], "artifactRawSha256": "raw", "artifactCanonicalSha256": "canonical",
        "source": source}
    terminal["terminalSha256"] = ev.sha(ev.canonical(terminal))
    _write_json(day / "terminal.json", terminal)
    assert ev._terminal(registry, issue, loaded) == terminal
    reserved_path, reserved, created = ev._reserve_review(registry, protocol_sha, "ready-sha",
        {"runId": "truth-a"}, dt.datetime(2026, 11, 30, 18, tzinfo=KST))
    assert created is True and reserved_path.exists()
    same_path, same, same_created = ev._reserve_review(registry, protocol_sha, "ready-sha",
        {"runId": "truth-a"}, dt.datetime(2026, 11, 30, 18, tzinfo=KST))
    assert same_path == reserved_path and same == reserved and same_created is False
    _, different, matches = ev._reserve_review(registry, protocol_sha, "ready-sha",
        {"runId": "truth-b"}, dt.datetime(2026, 11, 30, 18, tzinfo=KST))
    assert matches is False and different == reserved


def test_successful_forecast_requires_claim_source_even_when_terminal_has_source():
    doc = {"sources": {"dailyTrainingRunId": "run", "dailyTrainingSnapshotRecordSha256": "snap",
        "sourceCutoff": "2026-10-11", "pricesFileSha256": "prices", "provenanceFileSha256": "prov"}}
    terminal_source = {"runId": "run", "snapshotSha256": "snap", "cutoff": "2026-10-11",
        "pricesFileSha256": "prices", "provenanceFileSha256": "prov"}
    with pytest.raises(ValueError, match="successful claim source binding"):
        ev._validate_forecast_sources(doc, {"source": None}, terminal_source)


def test_run_evaluation_persists_first_ready_marker_and_versioned_reports(tmp_path, monkeypatch):
    registry, project, output = tmp_path / "registry", tmp_path / "project", tmp_path / "output"
    _registry(registry, project, monkeypatch=monkeypatch)
    now = dt.datetime.now(KST).replace(microsecond=100000)
    truth = _truth()
    monkeypatch.setattr(ev, "_validate_truth_run", lambda *a: truth)
    def ready_report(_registry, truth_doc, as_of, **kwargs):
        return {"schemaVersion": 1, "protocolId": ev.PROTOCOL_ID, "status": "not_ready",
            "mode": kwargs["mode"], "evaluatedAsOfKst": as_of.isoformat(),
            "truthSource": {"runId": truth_doc["runId"]},
            "thresholds": {"allHorizonsReady": True}}
    monkeypatch.setattr(ev, "build_report", ready_report)
    first_path, _ = ev.run_evaluation(registry, tmp_path / "truth-run", output,
        as_of=now, mode="daily")
    marker_path = registry / "readiness" / "first-ready.json"
    marker = json.loads(marker_path.read_bytes())
    first_digest = marker["readinessSha256"]
    second_path, _ = ev.run_evaluation(registry, tmp_path / "truth-run", output,
        as_of=now + dt.timedelta(microseconds=1), mode="daily")
    assert first_path != second_path
    assert json.loads(marker_path.read_bytes())["readinessSha256"] == first_digest
    assert first_path.exists() and second_path.exists()


def test_fixed_primary_family_and_both_auxiliary_losses_use_shared_blocks(monkeypatch):
    rows = [_pair(product, h, dt.date(2026, 10, 12) + dt.timedelta(days=h-1))
            for product in ev.PRODUCTS for h in ev.HORIZONS]
    dm_calls, holm_calls, bootstrap_calls = [], [], []
    original_dm, original_holm = ev.score._dm, ev.score._holm
    monkeypatch.setattr(ev.score, "_dm", lambda diff, bw: (dm_calls.append((len(diff), bw)) or original_dm(diff, bw)))
    monkeypatch.setattr(ev.score, "_holm", lambda family: (holm_calls.append(len(family)) or original_holm(family)))
    def bootstrap(sample, **kwargs):
        assert {(r["product"], r["horizon"]) for r in sample} == set(kwargs["cells"])
        assert all(r["target"] != "2026-11-20" for r in sample)
        bootstrap_calls.append(kwargs)
        return {"status": "ok", "interval95": [-1.0, 1.0], "validReplicates": 2000}
    monkeypatch.setattr(ev.intervals, "bootstrap_estimate", bootstrap)
    truth = _truth()
    immature = _pair("garlic", 1, dt.date(2026, 11, 20), mature=False)
    rows.append(immature)
    result = ev._review_inference(rows, truth)
    assert result["primary"]["familySizePerBandwidth"] == 20
    assert len(result["primary"]["tests"]) == 40
    assert len(dm_calls) == 40
    assert dm_calls.count((1, 9)) == 20 and dm_calls.count((1, 28)) == 20
    assert holm_calls == [20, 20]
    assert len(bootstrap_calls) == 12
    assert {x["metric"] for x in bootstrap_calls} == {"ae", "ape"}
    assert {x["block"] for x in bootstrap_calls} == {14, 28}
    assert {x["cells"] for x in bootstrap_calls} == {
        frozenset({("garlic", h) for h in ev.HORIZONS}),
        frozenset({("cabbage", h) for h in ev.HORIZONS}),
        frozenset({(p, h) for p in ev.PRODUCTS for h in ev.HORIZONS})}
    assert all(x["replicates"] == 2000 and x["seed"] == 20261011 for x in bootstrap_calls)
