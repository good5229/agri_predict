import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from agri_data import cabbage_national_daily as daily
from agri_data import cabbage_national_eval as evaluate
from agri_data import cabbage_national_prospective as prospective

KST = prospective.KST


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _html(day, value="1000"):
    compact = day.replace("-", "")
    label = f"{day[:4]}년 {day[5:7]}월 {day[8:]}일"
    return (f'<form name="actForm"><input name="searchSymbol1" value="100000">'
            f'<input name="searchSymbol2" value="1001">'
            f'<input name="searchName1" value="*전국도매시장">'
            f'<input name="searchName2" value="배추">'
            f'<input name="searchDate" value="{label}">'
            f'<input name="searchName3" value=""></form>'
            '<p class="grade">고</p><p class="grade">중</p><p class="grade">저</p>'
            '<p class="grade">평균</p><strong>1000</strong><span>1kg</span>'
            f'function makeGradeChart(gradeNm,index,avgPrice) {{ else {{ tmpObj["date"]="{compact}"; '
            f'tmpObj["value"]="{value}"; }} }}').encode()


def _baseline(day, values):
    return {"schemaVersion": 1, "issueDateKst": day.isoformat(),
            "issueAtKst": dt.datetime.combine(day, dt.time(10, 1), KST).isoformat(),
            "sourceCutoffDate": (day - dt.timedelta(days=1)).isoformat(),
            "products": [{"productId": "cabbage", "status": "available", "method": "ridge",
                          "predictions": [{"horizonDays": h,
                                           "target": (day + dt.timedelta(days=h)).isoformat(),
                                           "predictedPrice": values[h - 1], "sourceDate": (day - dt.timedelta(days=1)).isoformat()}
                                          for h in range(1, 11)]}]}


def _put(root, rel, payload):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_daily_artifact(root, issue, baseline_values=None, candidate_values=None, *, corrupt_baseline=False):
    baseline_values = baseline_values or [1000 + h for h in range(10)]
    candidate_values = candidate_values or list(baseline_values)
    base_rel = Path("outputs/live-forecasts") / issue.isoformat() / "ridge.json"
    baseline_doc = _baseline(issue, baseline_values)
    baseline_bytes = json.dumps(baseline_doc, sort_keys=True).encode()
    _put(root, base_rel, baseline_bytes)
    local_rel = Path("outputs/daily") / issue.isoformat() / "local-prices.json"
    local_doc = {"schemaVersion": 1, "meta": {"contextEnd": (issue-dt.timedelta(days=2)).isoformat()},
                 "products": [{"id": "cabbage", "observations": [{"date": (issue-dt.timedelta(days=3)).isoformat(),
                    "actual": 900, "marketCodes": ["110001", "110008"]}]}]}
    local_bytes = json.dumps(local_doc, sort_keys=True).encode()
    _put(root, local_rel, local_bytes)
    selected = issue - dt.timedelta(days=1)
    capture_at = dt.datetime.combine(issue, dt.time(10, 2), KST)
    html = _html(selected.isoformat())
    raw_rel = Path("outputs/prospective-cabbage/daily") / issue.isoformat() / "captures/page.html"
    _put(root, raw_rel, html)
    manifest = {"schemaVersion": 1, "captureState": "validated_price_observations", "issueDateKst": issue.isoformat(),
                "selectedDate": selected.isoformat(), "httpStatus": 200, "bodyComplete": True,
                "requestUrl": daily.build_page_url(selected), "capturedAtKst": capture_at.isoformat(),
                "rawResponse": {"path": str(raw_rel), "sha256": _sha(html)},
                "meanPricesWonPerKg": {selected.isoformat(): 1000.0},
                "selection": daily._response_selection(html.decode(), selected),
                "validatedSelection": daily._response_selection(html.decode(), selected)}
    manifest_rel = raw_rel.with_suffix(".json")
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    _put(root, manifest_rel, manifest_bytes)
    artifact = {"schemaVersion": 1, "artifactType": "cabbage_national_level_daily_challenger",
                "issueDateKst": issue.isoformat(), "originDate": issue.isoformat(),
                "issueAtKst": dt.datetime.combine(issue, dt.time(10, 3), KST).isoformat(),
                "inputs": {"nationalHtmlPath": str(raw_rel), "nationalHtmlSha256": _sha(html),
                           "nationalManifestPath": str(manifest_rel), "nationalManifestSha256": _sha(manifest_bytes),
                           "captureManifestPath": str(manifest_rel), "captureManifestSha256": _sha(manifest_bytes),
                           "requestUrl": daily.build_page_url(selected), "httpStatus": 200,
                           "nationalSelection": manifest["validatedSelection"],
                           "baselineArtifactPath": str(base_rel), "baselineArtifactSha256": _sha(baseline_bytes),
                           "baselineIssueAtKst": baseline_doc["issueAtKst"], "localPricesPath": str(local_rel),
                           "localPricesSha256": _sha(local_bytes),
                           "localObservationDates": [(issue-dt.timedelta(days=3)).isoformat()],
                           "capturedAtKst": capture_at.isoformat()},
                "levelRatio": {"windowStart": (issue-dt.timedelta(days=15)).isoformat(),
                               "windowEnd": (issue-dt.timedelta(days=2)).isoformat(),
                               "eligiblePairCount": 0, "eligiblePairs": [], "ratio": 1.0,
                               "fallbackReason": "fewer than 3 eligible same-day positive national/local pairs (0)",
                               "sourcePublicationTimeKnown": False},
                "forecasts": [{"horizonDays": h, "target": (issue+dt.timedelta(days=h)).isoformat(),
                               "baselineRidgePrediction": baseline_values[h-1],
                               "challengerPrediction": candidate_values[h-1], "nationalLevelRatio": 1.0}
                              for h in range(1, 11)]}
    artifact_rel = evaluate.DAILY_ARTIFACT_ROOT / issue.isoformat() / "challenger.json"
    artifact_bytes = json.dumps(artifact, sort_keys=True).encode()
    _put(root, artifact_rel, artifact_bytes)
    if corrupt_baseline:
        _put(root, base_rel, b"tampered")
    return artifact_rel, artifact_bytes


def _write_truth_capture(root, target, value="1000", *, issue_time="10:05:00", state="validated_price_observations"):
    issue = target + dt.timedelta(days=1)
    selected = target
    html = _html(target.isoformat(), value)
    raw_rel = Path("outputs/prospective-cabbage/daily") / issue.isoformat() / "captures" / f"national-{issue.isoformat()}T{issue_time.replace(':', '')}.html"
    _put(root, raw_rel, html)
    at = dt.datetime.combine(issue, dt.time.fromisoformat(issue_time), KST)
    manifest = {"schemaVersion": 1, "captureState": state, "issueDateKst": issue.isoformat(),
                "selectedDate": selected.isoformat(), "httpStatus": 200, "bodyComplete": True,
                "requestUrl": daily.build_page_url(selected), "capturedAtKst": at.isoformat(),
                "rawResponse": {"path": str(raw_rel), "sha256": _sha(html)},
                "meanPricesWonPerKg": {selected.isoformat(): float(value)},
                "selection": daily._response_selection(html.decode(), selected),
                "validatedSelection": daily._response_selection(html.decode(), selected)}
    manifest_rel = raw_rel.with_suffix(".json")
    _put(root, manifest_rel, json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode())
    return manifest_rel


def test_build_assessment_pairs_only_first_next_day_capture_and_preserves_exclusions(tmp_path):
    issue = dt.date(2026, 10, 8)
    artifact_rel, artifact_bytes = _write_daily_artifact(tmp_path, issue)
    for h in range(1, 11):
        _write_truth_capture(tmp_path, issue + dt.timedelta(days=h), str(1000+h))
    # The target 1 truth is valid, but a later revised capture must not replace it.
    _write_truth_capture(tmp_path, issue + dt.timedelta(days=1), "2000", issue_time="10:30:00")
    result = evaluate.build_assessment(tmp_path, [(artifact_rel, artifact_bytes)],
        dt.datetime(2026, 10, 25, 13, 0, tzinfo=KST))
    assert len(result["coverage"]) == 10
    row = next(row for row in result["coverage"] if row["horizon"] == 1)
    assert row["truthStatus"] == "paired" and row["actual"] == 1001
    assert row["candidate"] == row["baseline"]
    assert len(result["comparisons"]) == 20
    assert all(row["dm"]["9"]["pValue"] == 1 for row in result["comparisons"])
    assert all("holmPValue" in row["dm"]["9"] and "holmPValue" in row["dm"]["28"] for row in result["comparisons"])
    assert result["protocol"]["familySizePerBandwidth"] == 20
    assert result["macroMetrics"]["candidate"]["equalHorizonMape"] is not None


def test_truth_states_separate_immature_missed_invalid_empty_and_nonpositive(tmp_path):
    target = dt.date(2026, 10, 9)
    before_noon = dt.datetime(2026, 10, 10, 11, 59, tzinfo=KST)
    assert evaluate._truth_for_target(tmp_path, target, before_noon)["status"] == "immature"
    after_noon = dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST)
    assert evaluate._truth_for_target(tmp_path, target, after_noon)["status"] == "missed_next_day_capture"
    _write_truth_capture(tmp_path, target, "1000", state="http_success_empty_chart")
    assert evaluate._truth_for_target(tmp_path, target, after_noon)["status"] == "empty_capture"

    target2 = dt.date(2026, 10, 11)
    _write_truth_capture(tmp_path, target2, "0")
    truth = evaluate._truth_for_target(tmp_path, target2, dt.datetime(2026, 10, 12, 12, 1, tzinfo=KST))
    assert truth["status"] == "nonpositive_truth"


def test_earliest_successfully_validated_capture_follows_and_audits_prior_failure(tmp_path):
    target = dt.date(2026, 10, 9)
    _write_truth_capture(tmp_path, target, "1000", issue_time="10:01:00", state="selection_mismatch")
    _write_truth_capture(tmp_path, target, "1100", issue_time="10:20:00")
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST))
    assert result["status"] == "paired"
    assert result["actual"] == 1100
    assert len(result["priorCaptureAttempts"]) == 1


def test_tampered_baseline_source_is_rejected(tmp_path):
    issue = dt.date(2026, 10, 8)
    rel, payload = _write_daily_artifact(tmp_path, issue, corrupt_baseline=True)
    with pytest.raises(evaluate.CabbageEvalError, match="baseline artifact SHA-256 mismatch"):
        evaluate._validate_artifact(tmp_path, tmp_path / rel, payload)


def test_thirty_pair_hac_family_and_holm_are_complete(tmp_path):
    artifacts = []
    first = dt.date(2026, 10, 8)
    for offset in range(30):
        issue = first + dt.timedelta(days=offset)
        artifacts.append(_write_daily_artifact(tmp_path, issue))
    targets = {issue + dt.timedelta(days=h) for issue in [first + dt.timedelta(days=i) for i in range(30)] for h in evaluate.HORIZONS}
    for target in targets:
        _write_truth_capture(tmp_path, target, str(1200 + target.toordinal() % 17))
    as_of_day = max(targets) + dt.timedelta(days=1)
    result = evaluate.build_assessment(tmp_path, artifacts,
        dt.datetime.combine(as_of_day, dt.time(12, 1), KST))
    h1 = [row for row in result["comparisons"] if row["horizon"] == 1]
    assert all(row["n"] == 30 for row in h1)
    assert all(row["dm"]["9"]["status"] in ("tested", "zero_or_invalid_hac_variance") for row in h1)
    assert len(result["comparisons"]) == 20
    assert all(row["dm"]["9"]["pValue"] == 1.0 or row["dm"]["9"]["pValue"] < 1.0 for row in result["comparisons"])
    assert all("holmSignificant" in row["dm"]["9"] for row in result["comparisons"])
    json.dumps(result, allow_nan=False)
    assert all(isinstance(row["issuedAtKst"], str) for row in result["baselineIssueDays"])


def test_wrong_capture_day_and_path_tampering_do_not_pair(tmp_path):
    target = dt.date(2026, 10, 9)
    rel = _write_truth_capture(tmp_path, target)
    manifest = json.loads((tmp_path / rel).read_text())
    manifest["rawResponse"]["path"] = "outputs/prospective-cabbage/daily/2026-10-10/captures/../../elsewhere.html"
    (tmp_path / rel).write_text(json.dumps(manifest))
    truth = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST))
    assert truth["status"] == "invalid_capture"


def test_first_static_issue_path_and_source_metadata_are_validated(tmp_path, monkeypatch):
    issue = evaluate.FIRST_ISSUE
    raw_rel = Path("data/raw/frozen.html")
    manifest_rel = Path("data/raw/frozen.json")
    raw = b"synthetic immutable initial page"
    manifest = b"synthetic immutable transcription"
    _put(tmp_path, raw_rel, raw)
    _put(tmp_path, manifest_rel, manifest)
    monkeypatch.setattr(prospective, "RAW_HTML_PATH", raw_rel)
    monkeypatch.setattr(prospective, "MANIFEST_PATH", manifest_rel)
    monkeypatch.setattr(prospective, "RAW_HTML_SHA256", _sha(raw))
    monkeypatch.setattr(prospective, "MANIFEST_SHA256", _sha(manifest))
    monkeypatch.setattr(prospective, "validate_national_inputs", lambda *_args, **_kwargs: {"nationalPrices": {"2026-10-01": 1000.0}})

    base_rel = Path("outputs/live-forecasts") / issue.isoformat() / "ridge.json"
    baseline = _baseline(issue, [1100 + h for h in range(10)])
    baseline_bytes = json.dumps(baseline, sort_keys=True).encode()
    _put(tmp_path, base_rel, baseline_bytes)
    local_rel = Path("outputs/daily/initial-prices.json")
    local_doc = {"schemaVersion": 1, "products": [{"id": "cabbage", "observations": [
        {"date": "2026-10-01", "actual": 900, "marketCodes": ["110001", "110008"]}]}]}
    local_bytes = json.dumps(local_doc, sort_keys=True).encode()
    _put(tmp_path, local_rel, local_bytes)
    monkeypatch.setattr(prospective, "LOCAL_PRICES_PATH", local_rel)
    monkeypatch.setattr(prospective, "LOCAL_PRICES_SHA256", _sha(local_bytes))
    artifact = {"schemaVersion": 1, "artifactType": "cabbage_national_level_prospective_challenger",
                "issueDateKst": issue.isoformat(), "originDate": issue.isoformat(),
                "issueAtKst": "2026-10-07T10:03:59+09:00",
                "inputs": {"nationalHtmlPath": str(raw_rel), "nationalHtmlSha256": _sha(raw),
                           "nationalManifestPath": str(manifest_rel), "nationalManifestSha256": _sha(manifest),
                           "nationalPageUrl": prospective.NATIONAL_PAGE_URL,
                           "nationalSelection": prospective.EXPECTED_SELECTION,
                           "localPricesPath": str(local_rel), "localPricesSha256": _sha(local_bytes),
                           "localObservationDates": ["2026-10-01"],
                           "baselineArtifactPath": str(base_rel), "baselineArtifactSha256": _sha(baseline_bytes),
                           "baselineIssueAtKst": baseline["issueAtKst"]},
                "levelRatio": {"windowStart": "2026-09-22", "windowEnd": "2026-10-05",
                               "eligiblePairCount": 1,
                               "eligiblePairs": [{"date": "2026-10-01", "nationalPrice": 1000.0,
                                                  "localPrice": 900.0, "ratio": 1000.0/900.0}],
                               "ratio": 1.0,
                               "fallbackReason": "fewer than 3 eligible same-day positive national/local pairs (1)",
                               "sourcePublicationTimeKnown": False},
                "forecasts": [{"horizonDays": h, "target": (issue+dt.timedelta(days=h)).isoformat(),
                               "baselineRidgePrediction": baseline["products"][0]["predictions"][h-1]["predictedPrice"],
                               "challengerPrediction": baseline["products"][0]["predictions"][h-1]["predictedPrice"]}
                              for h in range(1, 11)]}
    artifact_rel = evaluate.STATIC_ARTIFACT
    artifact_bytes = json.dumps(artifact, sort_keys=True).encode()
    _put(tmp_path, artifact_rel, artifact_bytes)
    parsed = evaluate._validate_artifact(tmp_path, tmp_path / artifact_rel, artifact_bytes)
    assert parsed["issueDate"] == issue
    assert parsed["rows"][10]["target"] == dt.date(2026, 10, 17)

    artifact["forecasts"][0]["target"] = "2026-10-17"
    tampered = json.dumps(artifact).encode()
    _put(tmp_path, artifact_rel, tampered)
    with pytest.raises(evaluate.CabbageEvalError, match="target does not equal"):
        evaluate._validate_artifact(tmp_path, tmp_path / artifact_rel, tampered)


def test_modified_candidate_ratio_is_rejected(tmp_path):
    issue = dt.date(2026, 10, 8)
    rel, payload = _write_daily_artifact(tmp_path, issue)
    artifact = json.loads(payload)
    artifact["forecasts"][0]["challengerPrediction"] += 1
    changed = json.dumps(artifact).encode()
    _put(tmp_path, rel, changed)
    with pytest.raises(evaluate.CabbageEvalError, match="baseline × level ratio"):
        evaluate._validate_artifact(tmp_path, tmp_path / rel, changed)


def test_later_valid_baseline_is_rejected_when_earlier_valid_ridge_exists(tmp_path):
    issue = dt.date(2026, 10, 8)
    rel, payload = _write_daily_artifact(tmp_path, issue)
    earlier = _baseline(issue, [999 + h for h in range(10)])
    earlier["issueAtKst"] = "2026-10-08T10:00:01+09:00"
    earlier_path = tmp_path / "outputs/live-forecasts/2026-10-08/earlier.json"
    earlier_path.write_text(json.dumps(earlier))
    with pytest.raises(evaluate.CabbageEvalError, match="not the earliest valid"):
        evaluate._validate_artifact(tmp_path, tmp_path / rel, payload)


def test_baseline_only_issue_day_reduces_candidate_coverage(tmp_path):
    candidate_day = dt.date(2026, 10, 8)
    artifact = _write_daily_artifact(tmp_path, candidate_day)
    baseline_only_day = candidate_day + dt.timedelta(days=1)
    baseline = _baseline(baseline_only_day, [1020 + h for h in range(10)])
    baseline_path = tmp_path / "outputs/live-forecasts" / baseline_only_day.isoformat() / "ridge.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps(baseline))
    targets = {candidate_day + dt.timedelta(days=h) for h in evaluate.HORIZONS}
    targets |= {baseline_only_day + dt.timedelta(days=h) for h in evaluate.HORIZONS}
    for target in targets:
        _write_truth_capture(tmp_path, target, str(1000 + target.toordinal() % 13))
    as_of_day = max(targets) + dt.timedelta(days=1)
    report = evaluate.build_assessment(tmp_path, [artifact], dt.datetime.combine(as_of_day, dt.time(12, 1), KST))
    h1 = next(item for item in report["conclusions"] if item["horizon"] == 1)
    assert h1["candidatePairedCoverage"] == 1
    assert h1["baselineEligibleCoverage"] == 2
    assert h1["coverageNondeclining"] is False
    missing = [row for row in report["coverage"] if row["origin"] == baseline_only_day.isoformat()]
    assert len(missing) == 10 and all(row["candidateStatus"] == "not_issued" for row in missing)
    assert all(row["unpairedReason"] == "challenger_not_issued" for row in missing)


def test_null_target_value_is_not_imputed_when_no_later_valid_capture_exists(tmp_path):
    target = dt.date(2026, 10, 9)
    rel = _write_truth_capture(tmp_path, target, "1000")
    manifest_path = tmp_path / rel
    manifest = json.loads(manifest_path.read_text())
    raw_path = tmp_path / manifest["rawResponse"]["path"]
    html = raw_path.read_bytes().decode().replace('tmpObj["value"]="1000";', '')
    raw_path.write_bytes(html.encode())
    manifest["rawResponse"]["sha256"] = _sha(html.encode())
    manifest["captureState"] = "chart_validation_failed"
    manifest["meanPricesWonPerKg"] = None
    manifest_path.write_text(json.dumps(manifest))
    # A null target remains an explicit validation failure, not an imputed value.
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST))
    assert result["status"] == "invalid_capture"
    assert result["reason"] == "target_value_null_or_invalid_in_first_selected_chart"


def test_first_http_failure_is_audited_before_first_validated_truth(tmp_path):
    target = dt.date(2026, 10, 9)
    failed_rel = _write_truth_capture(tmp_path, target, "1000", issue_time="10:01:00", state="http_failure")
    failed_path = tmp_path / failed_rel
    failed = json.loads(failed_path.read_text())
    failed["httpStatus"] = 503
    failed["bodyComplete"] = True
    failed_path.write_text(json.dumps(failed))
    _write_truth_capture(tmp_path, target, "1100", issue_time="10:20:00")
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST))
    assert result["status"] == "paired" and result["actual"] == 1100
    assert result["priorCaptureAttempts"][0]["httpStatus"] == 503
    assert result["priorCaptureAttempts"][0]["rawHtmlSha256"] is not None


def test_early_success_is_mature_before_next_day_window_closes(tmp_path):
    target = dt.date(2026, 10, 9)
    _write_truth_capture(tmp_path, target, "1000", issue_time="10:05:00")
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 10, 10, tzinfo=KST))
    assert result["status"] == "paired"
    assert result["actual"] == 1000


def test_capture_after_historical_as_of_is_ignored_until_window_closes(tmp_path):
    target = dt.date(2026, 10, 9)
    _write_truth_capture(tmp_path, target, "1000", issue_time="10:20:00")
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 10, 10, tzinfo=KST))
    assert result["status"] == "immature"
    assert result["priorCaptureAttempts"] == []


def test_run_rejects_future_as_of_with_injectable_clock(tmp_path):
    now = dt.datetime(2026, 10, 10, 10, 0, tzinfo=KST)
    with pytest.raises(evaluate.CabbageEvalError, match="later than the current KST time"):
        evaluate.run(tmp_path, "report.json", now + dt.timedelta(microseconds=1), clock=lambda: now)
    assert evaluate._check_as_of(now - dt.timedelta(days=1), now) == now - dt.timedelta(days=1)


def test_null_or_nonpositive_first_chart_is_audited_before_later_positive_truth(tmp_path):
    target = dt.date(2026, 10, 9)
    first_rel = _write_truth_capture(tmp_path, target, "1000", issue_time="10:01:00")
    first_manifest_path = tmp_path / first_rel
    first_manifest = json.loads(first_manifest_path.read_text())
    raw_path = tmp_path / first_manifest["rawResponse"]["path"]
    html = raw_path.read_bytes().decode().replace('tmpObj["value"]="1000";', '')
    raw_path.write_bytes(html.encode())
    first_manifest["rawResponse"]["sha256"] = _sha(html.encode())
    first_manifest["captureState"] = "chart_validation_failed"
    first_manifest["meanPricesWonPerKg"] = None
    first_manifest_path.write_text(json.dumps(first_manifest))
    _write_truth_capture(tmp_path, target, "1200", issue_time="10:20:00")
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST))
    assert result["status"] == "paired" and result["actual"] == 1200
    assert result["priorCaptureAttempts"][0]["reason"] == "target_value_null_or_invalid_in_first_selected_chart"


def test_nonpositive_first_chart_is_audited_before_later_positive_truth(tmp_path):
    target = dt.date(2026, 10, 9)
    first_rel = _write_truth_capture(tmp_path, target, "0", issue_time="10:01:00", state="chart_validation_failed")
    _write_truth_capture(tmp_path, target, "1200", issue_time="10:20:00")
    result = evaluate._truth_for_target(tmp_path, target, dt.datetime(2026, 10, 10, 12, 1, tzinfo=KST))
    assert result["status"] == "paired" and result["actual"] == 1200
    assert result["priorCaptureAttempts"][0]["reason"] == "first_selected_capture_target_price_not_positive"
