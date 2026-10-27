import datetime as dt
import json

import pytest

from agri_data import dashboard
from agri_data import e106_prospective as issuer


def _sources(cutoff=dt.date(2026, 10, 9), row_count=0):
    products = []
    for product in dashboard.PRODUCTS:
        products.append({"id": product, "observations": [{"date": cutoff.isoformat(), "actual": 100.0,
            "volumeKg": 10.0, "rowCount": 2, "marketCodes": ["110001", "110008"]}]})
    prices = {"schemaVersion": 1, "generatedAt": "synthetic", "meta": {
        "evaluationEnd": cutoff.isoformat(), "dataEnd": cutoff.isoformat(),
        "aggregation": "signed total amount per kg", "marketNames": list(dashboard.MARKETS.values())},
        "products": products}
    provenance = [{"date": cutoff.isoformat(), "market": market, "product": product,
                   "jobId": 1, "epochId": f"{market}-{product}", "responseAt": "2026-10-10T00:00:00+00:00",
                   "rowCount": row_count, "selectedPrevious": False}
                  for product in dashboard.PRODUCTS for market in issuer.MARKETS]
    return json.dumps(prices).encode(), json.dumps(provenance).encode()


def _fake_base(crop, issue_date, cutoff):
    return [{"product": crop, "target": (issue_date + dt.timedelta(days=h)).isoformat(),
             "origin": issue_date.isoformat(), "horizon": h, "fitOrigin": issue_date.replace(day=1).isoformat(),
             "sourceCutoff": cutoff.isoformat(), "models": {
                 "causal50": {"status": "issued", "prediction": 100.0 + h, "reason": None},
                 "causalDirect": {"status": "issued", "prediction": 99.0 + h, "reason": None},
                 "causalSarima": {"status": "issued", "prediction": 101.0 + h, "reason": None}}}
            for h in issuer.HORIZONS]


def test_trend_scaling_uses_frozen_conflicting_and_neutral_rules():
    selected = {("cabbage", "conflicting"): {"supported": True, "scale": .75, "regime": "conflicting"}}
    result = issuer._trend_prediction("cabbage", 4, 120.0, 100.0,
        {"regime": "conflicting", "momentum7Log": .1, "momentum28Log": -.1}, selected)
    assert result["prediction"] == pytest.approx(100 * 1.2 ** .75)
    assert result["applied"] is True
    neutral = issuer._trend_prediction("cabbage", 4, 120.0, 100.0,
        {"regime": "neutral", "momentum7Log": 0.0, "momentum28Log": .1}, selected)
    assert neutral["prediction"] == 120.0
    assert neutral["applied"] is False


def test_provenance_distinguishes_complete_empty_from_missing_scope():
    _, raw = _sources(row_count=0)
    rows = json.loads(raw)
    scopes = issuer._validate_provenance(rows, dt.date(2026, 10, 9))
    audit = issuer._source_audit(
        {crop: {dt.date(2026, 10, 9): {"price": 100.0}} for crop in issuer.PRODUCTS},
        {crop: {dt.date(2026, 10, 9): (100.0, 10.0)} for crop in issuer.PRODUCTS},
        scopes, dt.date(2026, 10, 9), dt.date(2026, 10, 11))
    states = audit["garlic"]["sourceStates90d"]
    assert states[-2]["state"] == "complete_empty_response"
    assert states[-2]["selectedSources"]["110001"]["epochId"] == "110001-garlic"
    assert states[-1]["date"] == "2026-10-10"
    assert states[-1]["state"] == "incomplete_or_unknown"
    assert audit["garlic"]["positivePriceAndVolumeObservationCount28d"] == 1
    assert audit["garlic"]["sourceWindowEnd"] == "2026-10-10"
    assert audit["garlic"]["actualSnapshotCutoff"] == "2026-10-09"
    assert audit["garlic"]["unknownDaysAfterSnapshotCutoff"] == 1


def test_build_forecast_is_answer_free_same_issue_grid(monkeypatch):
    issue = dt.date(2026, 10, 11)
    prices_raw, provenance_raw = _sources(row_count=1)
    selection_raw = issuer.SELECTION_PATH.read_bytes()
    monkeypatch.setattr(issuer.four, "_forecast_crop", lambda records, series, crop, day, cutoff, progress=None, **kwargs:
                       _fake_base(crop, day, cutoff))
    monkeypatch.setattr(issuer, "_fit_garlic_106", lambda raw, day, rows, progress=None, **kwargs: {
        h: {"direct106": {"status": "issued", "prediction": 120.0 + h, "reason": None},
            "blend106": {"status": "issued", "prediction": 121.0 + h, "reason": None},
            "forecast": {"status": "issued", "prediction": 121.0 + h, "reason": None, "fallbackReason": None}}
        for h in issuer.HORIZONS})
    result = issuer.build_forecast(prices_raw, provenance_raw, selection_raw, issue,
                                   dt.datetime(2026, 10, 11, 12, tzinfo=issuer.KST))
    assert result["targetValuesAttached"] is False
    assert result["issueAtKst"] == "2026-10-11T12:00:00+09:00"
    assert len(result["forecasts"]) == 40
    assert len({(r["product"], r["origin"], r["target"], r["horizon"], r["sourceCutoff"])
                for r in result["forecasts"]}) == 40
    assert result["forecasts"][0]["E106"]["prediction"] == result["forecasts"][0]["E"]["prediction"]
    assert result["forecasts"][0]["causalDirect"]["prediction"] is not None
    assert result["forecasts"][0]["causalSarima"]["prediction"] is not None
    garlic = next(r for r in result["forecasts"] if r["product"] == "garlic")
    assert garlic["E106"]["prediction"] == garlic["E106Blend"]["prediction"]
    assert result["sourceAvailability"]["garlic"]["sourceAgeDaysAtIssue"] == 2
    assert result["sources"]["noTargetOutcomesRead"] is True


def test_future_source_data_is_rejected():
    prices_raw, _ = _sources(cutoff=dt.date(2026, 10, 11))
    prices = json.loads(prices_raw)
    with pytest.raises(ValueError, match="after the issue-date cutoff"):
        issuer._validate_prices(prices, dt.date(2026, 10, 11), prices_raw)


def test_selected_response_after_captured_issue_time_is_rejected():
    _, raw = _sources(row_count=1)
    rows = json.loads(raw)
    rows[0]["responseAt"] = "2026-10-11T04:00:01+00:00"
    with pytest.raises(ValueError, match="after the captured issue timestamp"):
        issuer._validate_provenance(rows, dt.date(2026, 10, 9),
                                    dt.datetime(2026, 10, 11, 13, tzinfo=issuer.KST))


def test_r106_support_mismatch_stops_before_fit(monkeypatch):
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(150)]
    direct = {"garlic": {day: (100.0, 10.0) for day in dates}}
    records = {"garlic": {day: {"price": 100.0, "volume": 10.0} for day in dates}}
    ordered = {"garlic": dates}
    prices = {"garlic": {day: 100.0 for day in dates}}
    monkeypatch.setattr(issuer, "_prepare_preserved_inputs",
                        lambda _raw: (records, direct, ordered, prices, {"garlic": {}}))
    monkeypatch.setattr(issuer.preserved, "_feature_pair",
                        lambda *args, **kwargs: {"preserved106": [0.0] * 106, "p0": 100.0,
                                                  "p0Date": dates[-1], "p0AgeDays": 0})
    monkeypatch.setattr(issuer.online, "_fit_ridge",
                        lambda *args, **kwargs: pytest.fail("fit must not run before support validation"))
    causal_rows = {h: {"models": {"causal50": {"status": "issued", "prediction": 100.0},
                                   "causalSarima": {"status": "issued", "prediction": 100.0},
                                   "causalDirect": {"training": {"count": 1, "firstTarget": "2025-01-01",
                                       "lastTarget": "2025-01-01", "fitOrigin": "2026-10-01"}}}}
                   for h in issuer.HORIZONS}
    with pytest.raises(ValueError, match="training support mismatch"):
        issuer._fit_garlic_106({}, dt.date(2026, 10, 11), causal_rows)


def test_blend_sarima_fallback_is_reported():
    blend = {"status": "issued", "prediction": 123.0, "provenance": {"fallback": True}}
    assert issuer._blend_fallback_reason(blend) == "sarima_unavailable_direct106_only_fallback"


def test_daily_run_completion_must_be_present_complete_and_precede_issue(tmp_path, monkeypatch):
    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    run_dir = tmp_path / "outputs/daily/2026-10-10/run"
    run_dir.mkdir(parents=True)
    prices = run_dir / "prices.json"
    provenance = run_dir / "selected-provenance.json"
    prices.write_text("{}")
    provenance.write_text("[]")
    finished = "2026-10-11T03:00:00+00:00"
    latest = {"runId": "run", "runDirectory": "outputs/daily/2026-10-10/run",
              "selectedProvenance": "outputs/daily/2026-10-10/run/selected-provenance.json",
              "cutoff": "2026-10-09", "snapshotSha256": "abc", "finishedAtKst": finished}
    status_path = tmp_path / "daily-training-status.json"
    status_path.write_text(json.dumps({"latestSuccess": latest}))
    run_status = {"status": "complete", "runId": "run", "cutoff": "2026-10-09",
                  "snapshotSha256": "abc", "finishedAtKst": finished}
    (run_dir / "run-status.json").write_text(json.dumps(run_status))
    issue_at = dt.datetime(2026, 10, 11, 12, tzinfo=issuer.KST)
    assert issuer._verify_latest_success(status_path, prices, provenance,
                                         captured_issue_at=issue_at)[0] == latest
    (run_dir / "run-status.json").unlink()
    with pytest.raises(ValueError, match="record is missing"):
        issuer._verify_latest_success(status_path, prices, provenance, captured_issue_at=issue_at)
    (run_dir / "run-status.json").write_text(json.dumps(run_status))
    run_status["status"] = "failed"
    (run_dir / "run-status.json").write_text(json.dumps(run_status))
    with pytest.raises(ValueError, match="complete immutable"):
        issuer._verify_latest_success(status_path, prices, provenance, captured_issue_at=issue_at)
    run_status["status"] = "complete"
    run_status["finishedAtKst"] = "2026-10-11T05:00:00+00:00"
    latest["finishedAtKst"] = run_status["finishedAtKst"]
    status_path.write_text(json.dumps({"latestSuccess": latest}))
    (run_dir / "run-status.json").write_text(json.dumps(run_status))
    with pytest.raises(ValueError, match="after the captured issue timestamp"):
        issuer._verify_latest_success(status_path, prices, provenance, captured_issue_at=issue_at)


def test_snapshot_content_binding_recomputes_fingerprint_and_keeps_signed_provenance(monkeypatch, tmp_path):
    prices_raw, provenance_raw = _sources(row_count=1)
    prices_path, provenance_path = tmp_path / "prices.json", tmp_path / "provenance.json"
    prices_path.write_bytes(prices_raw)
    provenance_path.write_bytes(provenance_raw)
    interpreter = tmp_path / "python"
    interpreter.write_text("stub")
    interpreter.chmod(0o755)
    binding = {"dailyTrainingFingerprintRecomputed": "frozen", "selectedScopeSha256": "scope-sha",
               "selectedScopeCount": 123, "selectedRequestCount": 1,
               "selectedQueryStates": {"complete": 4}, "selectedRequestHashes": {"10": "response-sha"},
               "reconciliation": {"checkedProductDates": 20, "positiveTargetsReconciled": 10},
               "broadScopesSha256": "broad-sha", "broadScopeCount": 30}
    probe = {"fingerprint": "frozen", "pricesSha256": issuer._sha(prices_raw),
             "provenanceSha256": issuer._sha(provenance_raw),
             "runtime": {"python": "3.12", "executable": "/venv/bin/python",
                         "packages": {"numpy": "2.5"}, "sqliteVersion": "3.53.1"},
             "databaseReconciliation": binding}
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return issuer.subprocess.CompletedProcess(command, 0, stdout=json.dumps(probe), stderr="")
    monkeypatch.setattr(issuer.subprocess, "run", fake_run)
    bound = issuer._verify_snapshot_content(prices_raw, provenance_raw,
        {"snapshotSha256": "frozen"}, tmp_path / "agri.sqlite", prices_path=prices_path,
        provenance_path=provenance_path, fingerprint_python=interpreter)
    assert bound["dailyTrainingFingerprintRecomputed"] == "frozen"
    assert bound["selectedRequestHashes"] == {"10": "response-sha"}
    assert bound["reconciliation"]["positiveTargetsReconciled"] == 10
    assert bound["fingerprintValidationRuntime"]["sqliteVersion"] == "3.53.1"
    assert "--db" in calls[0][0] and calls[0][1]["timeout"] == 900
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        issuer._verify_snapshot_content(prices_raw, provenance_raw,
            {"snapshotSha256": "different"}, tmp_path / "agri.sqlite", prices_path=prices_path,
            provenance_path=provenance_path, fingerprint_python=interpreter)


def test_snapshot_fingerprint_subprocess_failure_stops_before_reconciliation(monkeypatch, tmp_path):
    prices_raw, provenance_raw = _sources(row_count=1)
    prices_path, provenance_path = tmp_path / "prices.json", tmp_path / "provenance.json"
    prices_path.write_bytes(prices_raw)
    provenance_path.write_bytes(provenance_raw)
    interpreter = tmp_path / "python"
    interpreter.write_text("stub")
    interpreter.chmod(0o755)
    monkeypatch.setattr(issuer.subprocess, "run", lambda *a, **kw: issuer.subprocess.CompletedProcess(
        a[0], 17, stdout="", stderr="missing frozen dependency"))
    with pytest.raises(ValueError, match="failed before fitting"):
        issuer._verify_snapshot_content(prices_raw, provenance_raw, {"snapshotSha256": "x"},
            tmp_path / "agri.sqlite", prices_path=prices_path, provenance_path=provenance_path,
            fingerprint_python=interpreter)


def test_source_validation_timeout_stops_issue_without_fallback(monkeypatch, tmp_path):
    prices_raw, provenance_raw = _sources(row_count=1)
    prices_path, provenance_path = tmp_path / "prices.json", tmp_path / "provenance.json"
    prices_path.write_bytes(prices_raw)
    provenance_path.write_bytes(provenance_raw)
    interpreter = tmp_path / "python"
    interpreter.write_text("stub")
    interpreter.chmod(0o755)
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 900
        raise issuer.subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(issuer.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="exceeded the 900-second limit"):
        issuer._verify_snapshot_content(prices_raw, provenance_raw, {"snapshotSha256": "x"},
            tmp_path / "agri.sqlite", prices_path=prices_path, provenance_path=provenance_path,
            fingerprint_python=interpreter)


def test_pinned_helper_reconciles_signed_source_and_records_sqlite(monkeypatch, tmp_path):
    prices_raw, provenance_raw = _sources(row_count=1)
    prices_path, provenance_path = tmp_path / "prices.json", tmp_path / "provenance.json"
    prices_path.write_bytes(prices_raw)
    provenance_path.write_bytes(provenance_raw)
    monkeypatch.setattr(issuer.daily_training, "_pipeline_fingerprint", lambda prices, rows: "frozen")
    monkeypatch.setattr(issuer.online.direct_horizon, "_observations", lambda prices: {"core": True})
    expected = {"selected": {"selectedRequestHashes": {"10": "response-sha"},
                              "selectedScopesSha256": "scope-sha", "selectedScopeCount": 123,
                              "selectedQueryStates": {"complete": 4}, "selectedRequestCount": 1},
                "reconciliation": {"checkedProductDates": 20, "positiveTargetsReconciled": 10},
                "broad": {"scopesSha256": "broad-sha", "scopeCount": 30}}
    calls = []
    monkeypatch.setattr(issuer.market_mix, "extract_market_data", lambda db, raw, core: (
        calls.append((db, raw, core)) or ({}, expected)))
    result = issuer._fingerprint_probe(prices_path, provenance_path, issuer._sha(prices_raw),
                                       issuer._sha(provenance_raw), tmp_path / "agri.sqlite")
    assert len(calls) == 1 and calls[0][0] == tmp_path / "agri.sqlite"
    assert result["databaseReconciliation"]["selectedRequestHashes"] == {"10": "response-sha"}
    assert result["runtime"]["sqliteVersion"] == issuer.sqlite3.sqlite_version


def test_model_runtime_requires_original_statsmodels_before_fitting(monkeypatch):
    expected = dict(issuer.ISSUER_RUNTIME)
    monkeypatch.setattr(issuer, "_package_version", lambda name: "0.14.6" if name == "statsmodels"
                        else expected.get(name))
    with pytest.raises(RuntimeError, match=r"statsmodels=0.14.6 \(expected 0.14.0\)"):
        issuer._validate_model_runtime()


def test_idempotent_write_never_overwrites_different_bytes(tmp_path):
    first = {"issueDateKst": "2026-10-11", "value": 1}
    target, created = issuer._write_idempotent(tmp_path, first)
    assert created is True and json.loads(target.read_text()) == first
    assert issuer._write_idempotent(tmp_path, first) == (target, False)
    with pytest.raises(FileExistsError):
        issuer._write_idempotent(tmp_path, {"issueDateKst": "2026-10-11", "value": 2})
