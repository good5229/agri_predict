import datetime as dt
import json
import urllib.error
from pathlib import Path

import pytest

from agri_data import cabbage_national_daily as daily
from agri_data import cabbage_national_prospective as base


def _html(day="20261006", value="934", *, market="100000", product="1001",
          market_name="*전국도매시장", product_name="배추", variety="",
          grade=None, date_label="2026년 10월 06일", chart=True, unit=True):
    rows = (f'function makeGradeChart(gradeNm,index,avgPrice) {{ else {{ '
            f'tmpObj["date"]="{day}"; tmpObj["value"]="{value}"; }} }}') if chart else ""
    grades = grade if grade is not None else "<p class=\"grade\">고</p><p class=\"grade\">중</p><p class=\"grade\">저</p><p class=\"grade\">평균</p>"
    return (f'<form name="actForm"><input name="searchSymbol1" value="{market}">'
            f'<input name="searchSymbol2" value="{product}">'
            f'<input name="searchName1" value="{market_name}">'
            f'<input name="searchName2" value="{product_name}">'
            f'<input id="pickerDate" name="searchDate" value="{date_label}">'
            f'<input name="searchName3" value="{variety}"></form>'
            f'{grades}<strong>934</strong>{"<span>1kg</span>" if unit else ""}{rows}').encode()


def _baseline(day=dt.date(2026, 10, 8), issue_time="2026-10-08T10:15:00+09:00"):
    return {"schemaVersion": 1, "issueDateKst": day.isoformat(), "issueAtKst": issue_time,
            "sourceCutoffDate": (day - dt.timedelta(days=1)).isoformat(),
            "products": [{"productId": "cabbage", "status": "available", "method": "ridge",
                          "predictions": [{"horizonDays": h,
                                           "target": (day + dt.timedelta(days=h)).isoformat(),
                                           "predictedPrice": 1000 + h, "sourceDate": (day-dt.timedelta(days=1)).isoformat()}
                                          for h in range(1, 11)]}]}


def _prices(day=dt.date(2026, 10, 6)):
    rows = []
    for offset, price in ((4, 900), (3, 950), (2, 1000)):
        observed = day - dt.timedelta(days=offset)
        rows.append({"date": observed.isoformat(), "actual": price,
                     "marketCodes": ["110001", "110008"]})
    return {"schemaVersion": 1, "generatedAt": "2026-10-06T23:00:00+00:00",
            "meta": {"contextEnd": day.isoformat()},
            "products": [{"id": "cabbage", "observations": rows}]}


def _clock(*stamps):
    values = iter(stamps)
    return lambda: next(values)


def _transport(status=200, body=None):
    payload = body if body is not None else _html()
    return lambda url, timeout, limit: (status, payload, {
        "Content-Type": "text/html; charset=utf-8", "Set-Cookie": "session=synthetic-secret"})


def test_build_page_url_encodes_prior_day_and_frozen_selection():
    url = daily.build_page_url(dt.date(2026, 10, 6))
    parsed = daily.urllib.parse.urlparse(url)
    query = daily.urllib.parse.parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert query["searchDate"] == ["2026년 10월 06일"]
    assert query["searchName1"] == ["*전국도매시장"]
    assert query["searchName2"] == ["배추"]
    assert query["searchSymbol1"] == ["100000"]
    assert query["searchSymbol2"] == ["1001"]


def test_success_capture_preserves_exact_html_manifest_and_validated_selection(tmp_path):
    issue = dt.date(2026, 10, 7)
    payload = _html()
    times = _clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                   dt.datetime(2026, 10, 7, 10, 30, 1, tzinfo=base.KST))
    capture = daily.capture_national_page(root=tmp_path, issue_date=issue,
                                          transport=_transport(body=payload), clock=times,
                                          sleeper=lambda _: None)
    assert capture["state"] == "validated_price_observations"
    raw_path = tmp_path / capture["rawPath"]
    manifest_path = tmp_path / capture["manifestPath"]
    assert raw_path.read_bytes() == payload
    assert base._sha256(payload) == capture["rawSha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["httpStatus"] == 200
    assert manifest["requestUrl"] == daily.build_page_url(issue - dt.timedelta(days=1))
    assert manifest["selection"]["selectedDate"] == "2026-10-06"
    assert manifest["selection"]["varietyEvidence"].startswith("searchName3 is empty")
    assert "Set-Cookie" not in manifest["responseHeaders"]


def test_selection_is_scoped_to_actform_and_does_not_accept_decoy_controls():
    selected = dt.date(2026, 10, 6)
    html = (_html(market="999999").decode().replace('<form name="actForm">',
            '<form name="decoy"><input name="searchSymbol1" value="100000">'
            '<input name="searchSymbol2" value="1001"><input name="searchName1" value="*전국도매시장">'
            '<input name="searchName2" value="배추"></form><form name="actForm">'))
    with pytest.raises(daily.DailyCabbageError, match="marketCode"):
        daily._response_selection(html, selected)


@pytest.mark.parametrize("payload,expected_state", [
    (_html(chart=False), "http_success_empty_chart"),
    (_html(market="000000"), "selection_mismatch"),
    (_html(product="9999"), "selection_mismatch"),
    (_html(market_name="서울도매시장"), "selection_mismatch"),
    (_html(product_name="무"), "selection_mismatch"),
    (_html(variety="특품"), "selection_mismatch"),
    (_html(grade="특품"), "selection_mismatch"),
    (_html(unit=False), "selection_mismatch"),
    (_html(date_label="2026년 10월 05일"), "selection_mismatch"),
    (_html(value="-15"), "chart_validation_failed"),
    (_html(day="20261005"), "chart_validation_failed"),
])
def test_nonmatching_or_empty_successful_page_is_captured_but_not_validated(tmp_path, payload, expected_state):
    capture = daily.capture_national_page(
        root=tmp_path, issue_date=dt.date(2026, 10, 7), transport=_transport(body=payload),
        clock=_clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                     dt.datetime(2026, 10, 7, 10, 30, 1, tzinfo=base.KST)), sleeper=lambda _: None)
    assert capture["state"] == expected_state
    assert (tmp_path / capture["rawPath"]).read_bytes() == payload
    manifest = json.loads((tmp_path / capture["manifestPath"]).read_text(encoding="utf-8"))
    if expected_state == "http_success_empty_chart":
        assert "not evidence of closure or no-trade" in manifest["reason"]
    if expected_state == "selection_mismatch":
        assert manifest["selection"] is None
        assert manifest["validatedSelection"] is None
        assert manifest["requestedSelection"]["selectedDate"] == "2026-10-06"


def test_chart_with_future_dates_is_captured_but_rejected(tmp_path):
    html = _html().decode().replace(
        'tmpObj["date"]="20261006"; tmpObj["value"]="934"; } }',
        'tmpObj["date"]="20261006"; tmpObj["value"]="934"; } else { '
        'tmpObj["date"]="20261008"; tmpObj["value"]="1200"; } }').encode()
    capture = daily.capture_national_page(
        root=tmp_path, issue_date=dt.date(2026, 10, 7),
        transport=_transport(body=html),
        clock=_clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                     dt.datetime(2026, 10, 7, 10, 30, 1, tzinfo=base.KST)), sleeper=lambda _: None)
    assert capture["state"] == "chart_validation_failed"
    assert "after selected prior day" in capture["reason"]


def test_http_failure_retries_are_bounded_and_response_is_saved(tmp_path):
    calls, sleeps = [], []
    def transport(url, timeout, limit):
        calls.append((url, timeout, limit))
        return 503, b"service unavailable", {"Retry-After": "1"}
    capture = daily.capture_national_page(root=tmp_path, issue_date=dt.date(2026, 10, 7),
                                          transport=transport,
                                          clock=_clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                                                       dt.datetime(2026, 10, 7, 10, 30, 2, tzinfo=base.KST)),
                                          sleeper=sleeps.append)
    assert capture["state"] == "http_failure"
    assert len(calls) == daily.MAX_ATTEMPTS == 3
    assert len(sleeps) == 2
    assert (tmp_path / capture["rawPath"]).read_bytes() == b"service unavailable"
    manifest = json.loads((tmp_path / capture["manifestPath"]).read_text())
    attempts = manifest["httpAttempts"]
    assert [item["status"] for item in attempts] == [503, 503, 503]
    assert all(item["attemptedAtKst"] and item["bodyComplete"] for item in attempts)
    assert all((tmp_path / item["bodyPath"]).read_bytes() == b"service unavailable" for item in attempts)
    assert all(item["bodySha256"] == base._sha256(b"service unavailable") for item in attempts)


def test_transport_failure_is_distinct_from_http_failure_and_empty_chart(tmp_path):
    calls = []
    def transport(*_args):
        calls.append(1)
        raise urllib.error.URLError("synthetic connection failure")
    capture = daily.capture_national_page(root=tmp_path, issue_date=dt.date(2026, 10, 7),
                                          transport=transport,
                                          clock=_clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                                                       dt.datetime(2026, 10, 7, 10, 30, 1, tzinfo=base.KST)),
                                          sleeper=lambda _: None)
    assert capture["state"] == "transport_failure"
    assert capture["httpStatus"] is None and capture["rawPath"] is None
    assert len(calls) == daily.MAX_ATTEMPTS
    manifest = json.loads((tmp_path / capture["manifestPath"]).read_text(encoding="utf-8"))
    assert manifest["reason"].startswith("transport_failed_after_3_attempts")


def _install_daily_inputs(root: Path, *, baseline=True):
    issue = dt.date(2026, 10, 8)
    local_dir = root / "outputs" / "daily" / "2026-10-07" / "run-a"
    local_dir.mkdir(parents=True)
    snapshot = _prices()
    local_bytes = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()
    (local_dir / "prices.json").write_bytes(local_bytes)
    (local_dir / "run-status.json").write_text(json.dumps({
        "status": "complete", "cutoff": "2026-10-06",
        "finishedAtKst": "2026-10-07T23:00:00+09:00",
        "snapshotSha256": "a" * 64,
    }), encoding="utf-8")
    if baseline:
        base_dir = root / "outputs" / "live-forecasts" / issue.isoformat()
        base_dir.mkdir(parents=True)
        (base_dir / "baseline.json").write_text(json.dumps(_baseline()), encoding="utf-8")
    return local_bytes


def test_newest_completed_snapshot_and_baseline_produce_immutable_daily_challenger(tmp_path):
    _install_daily_inputs(tmp_path)
    issue = dt.date(2026, 10, 8)
    clock = _clock(*[dt.datetime(2026, 10, 8, 10, 30, s, tzinfo=base.KST) for s in range(5)])
    output = tmp_path / "outputs/prospective-cabbage/daily/2026-10-08/challenger.json"
    result = daily.issue_daily(root=tmp_path, issue_date=issue, transport=_transport(body=_html(day="20261007", date_label="2026년 10월 07일")),
                               clock=clock, sleeper=lambda _: None)
    assert result["status"] == "complete"
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["issueAtKst"] == "2026-10-08T10:30:04.000000+09:00"
    assert artifact["artifactType"] == "cabbage_national_level_daily_challenger"
    assert artifact["inputs"]["localPricesSha256"] == base._sha256(
        (tmp_path / "outputs/daily/2026-10-07/run-a/prices.json").read_bytes())
    assert len(artifact["forecasts"]) == 10
    assert artifact["levelRatio"]["fallbackReason"]
    assert result["capture"]["state"] == "validated_price_observations"


def test_newest_completed_snapshot_is_selected_and_stale_context_is_not_substituted(tmp_path):
    _install_daily_inputs(tmp_path)
    issue = dt.date(2026, 10, 8)
    second = tmp_path / "outputs" / "daily" / "2026-10-07" / "run-b"
    second.mkdir()
    newer = _prices()
    newer["products"][0]["observations"][0]["actual"] = 910
    (second / "prices.json").write_text(json.dumps(newer), encoding="utf-8")
    (second / "run-status.json").write_text(json.dumps({
        "status": "complete", "cutoff": "2026-10-06", "finishedAtKst": "2026-10-07T23:30:00+09:00"}),
        encoding="utf-8")
    as_of = dt.datetime(2026, 10, 8, 10, 0, tzinfo=base.KST)
    selected, payload, snapshot, _ = daily.select_newest_local_snapshot(tmp_path, issue, not_after=as_of)
    assert selected.parent.name == "run-b"
    assert snapshot["products"][0]["observations"][0]["actual"] == 910
    assert json.loads(payload)["meta"]["contextEnd"] == "2026-10-06"
    snapshot["meta"]["contextEnd"] = "2026-10-05"
    (second / "prices.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (second / "run-status.json").write_text(json.dumps({
        "status": "complete", "cutoff": "2026-10-06", "finishedAtKst": "2026-10-07T23:45:00+09:00"}),
        encoding="utf-8")
    selected, _, snapshot, _ = daily.select_newest_local_snapshot(tmp_path, issue, not_after=as_of)
    assert selected.parent.name == "run-a"
    assert snapshot["meta"]["contextEnd"] == "2026-10-06"


def test_snapshot_completed_or_generated_after_issue_is_not_selected(tmp_path):
    _install_daily_inputs(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "outputs/daily/2026-10-07/run-a")
    issue = dt.date(2026, 10, 8)
    late = tmp_path / "outputs" / "daily" / "2026-10-07" / "run-late"
    late.mkdir()
    snapshot = _prices()
    snapshot["generatedAt"] = "2026-10-08T01:00:00+00:00"
    (late / "prices.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (late / "run-status.json").write_text(json.dumps({
        "status": "complete", "cutoff": "2026-10-06",
        "finishedAtKst": "2026-10-08T10:20:00+09:00"}), encoding="utf-8")
    with pytest.raises(daily.DailyCabbageError, match="no completed prior-day snapshot"):
        daily.select_newest_local_snapshot(
            tmp_path, issue, not_after=dt.datetime(2026, 10, 8, 10, 0, tzinfo=base.KST))


def test_capture_is_kept_when_baseline_is_missing(tmp_path):
    _install_daily_inputs(tmp_path, baseline=False)
    issue = dt.date(2026, 10, 8)
    clock = _clock(*[dt.datetime(2026, 10, 8, 10, 30, s, tzinfo=base.KST) for s in range(4)])
    result = daily.issue_daily(root=tmp_path, issue_date=issue, transport=_transport(body=_html(day="20261007", date_label="2026년 10월 07일")),
                               clock=clock, sleeper=lambda _: None)
    assert result["status"] == "not_issued"
    assert result["reason"].startswith("baseline_unavailable:")
    assert (tmp_path / result["capture"]["rawPath"]).is_file()
    assert (tmp_path / result["capture"]["manifestPath"]).is_file()
    assert (tmp_path / result["skipRecordPath"]).is_file()

    # A later retry reuses the earliest validated capture rather than fetching
    # a revised page, then issues once a valid baseline appears.
    base_dir = tmp_path / "outputs" / "live-forecasts" / issue.isoformat()
    base_dir.mkdir(parents=True)
    (base_dir / "baseline.json").write_text(json.dumps(_baseline()), encoding="utf-8")
    calls = []
    def must_not_fetch(*_args):
        calls.append(True)
        raise AssertionError("validated same-day capture should be reused")
    retry_clock = _clock(*[dt.datetime(2026, 10, 8, 10, 31, s, tzinfo=base.KST) for s in range(3)])
    second = daily.issue_daily(root=tmp_path, issue_date=issue, transport=must_not_fetch,
                               clock=retry_clock, sleeper=lambda _: None)
    assert second["status"] == "complete"
    assert second["capture"]["reused"] is True
    assert calls == []
    third = daily.issue_daily(root=tmp_path, issue_date=issue, transport=must_not_fetch,
                              clock=lambda: dt.datetime(2026, 10, 8, 10, 40, tzinfo=base.KST))
    assert third["status"] == "already_issued"
    assert calls == []


def test_noon_crossing_after_build_skips_issue_and_keeps_capture(tmp_path):
    _install_daily_inputs(tmp_path)
    issue = dt.date(2026, 10, 8)
    stamps = [dt.datetime(2026, 10, 8, 11, 59, 55, tzinfo=base.KST),
              dt.datetime(2026, 10, 8, 11, 59, 56, tzinfo=base.KST),
              dt.datetime(2026, 10, 8, 11, 59, 57, tzinfo=base.KST),
              dt.datetime(2026, 10, 8, 11, 59, 58, tzinfo=base.KST),
              dt.datetime(2026, 10, 8, 12, 0, 1, tzinfo=base.KST)]
    output = tmp_path / "outputs/prospective-cabbage/daily/2026-10-08/challenger.json"
    result = daily.issue_daily(root=tmp_path, issue_date=issue, transport=_transport(body=_html(day="20261007", date_label="2026년 10월 07일")),
                               clock=_clock(*stamps), sleeper=lambda _: None)
    assert result["status"] == "not_issued"
    assert result["reason"].startswith("issue_window_closed_before_publication:")
    assert not output.exists()
    assert (tmp_path / result["capture"]["rawPath"]).exists()


def test_daily_issuer_rejects_first_issue_date_and_older_dates_without_output_override(tmp_path):
    for issue in (dt.date(2026, 10, 7), dt.date(2026, 10, 6)):
        result = daily.issue_daily(root=tmp_path, issue_date=issue,
                                   clock=lambda: dt.datetime(2026, 10, 8, 10, 30, tzinfo=base.KST))
        assert result["status"] == "not_issued"
        assert result["reason"].startswith("daily_issuance_available_only_after_first_issue_date")
    assert "output_path" not in __import__("inspect").signature(daily.issue_daily).parameters


def test_retry_after_bounded_wait_is_honored_and_long_wait_defers_capture(tmp_path):
    sleeps, calls = [], []
    def retry_transport(url, timeout, limit):
        calls.append(1)
        return (503, b"busy", {"Retry-After": "2"}) if len(calls) == 1 else (200, _html(), {})
    status, body, headers, attempts, error, attempt_records = daily._request_page(
        "https://example.invalid", transport=retry_transport, sleeper=sleeps.append)
    assert (status, body, attempts, error) == (200, _html(), 2, None)
    assert sleeps == [2.0]
    assert [item["status"] for item in attempt_records] == [503, 200]
    assert [item["body"] for item in attempt_records] == [b"busy", _html()]

    calls.clear()
    capture = daily.capture_national_page(
        root=tmp_path, issue_date=dt.date(2026, 10, 7),
        transport=lambda *_: (503, b"busy", {"Retry-After": str(daily.MAX_RETRY_AFTER_SECONDS + 10)}),
        clock=_clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                     dt.datetime(2026, 10, 7, 10, 30, 1, tzinfo=base.KST)), sleeper=sleeps.append)
    assert capture["state"] == "retry_deferred"
    manifest = json.loads((tmp_path / capture["manifestPath"]).read_text())
    assert manifest["responseHeaders"]["Retry-After"] == str(daily.MAX_RETRY_AFTER_SECONDS + 10)
    assert manifest["bodyComplete"] is True
    assert manifest["httpAttempts"][0]["bodyComplete"] is True
    assert (tmp_path / manifest["httpAttempts"][0]["bodyPath"]).read_bytes() == b"busy"
    assert (tmp_path / capture["rawPath"]).read_bytes() == b"busy"


def test_prior_day_snapshot_finished_after_midnight_is_rejected(tmp_path):
    _install_daily_inputs(tmp_path)
    run_status = tmp_path / "outputs/daily/2026-10-07/run-a/run-status.json"
    status = json.loads(run_status.read_text())
    status["finishedAtKst"] = "2026-10-08T00:01:00+09:00"
    run_status.write_text(json.dumps(status))
    with pytest.raises(daily.DailyCabbageError, match="no completed prior-day snapshot"):
        daily.select_newest_local_snapshot(
            tmp_path, dt.date(2026, 10, 8),
            not_after=dt.datetime(2026, 10, 8, 10, 30, tzinfo=base.KST))


def test_late_page_capture_is_preserved_but_not_reused_for_daily_issue(tmp_path):
    issue = dt.date(2026, 10, 8)
    capture = daily.capture_national_page(
        root=tmp_path, issue_date=issue, transport=_transport(body=_html(day="20261007", date_label="2026년 10월 07일")),
        clock=_clock(dt.datetime(2026, 10, 8, 12, 1, tzinfo=base.KST),
                     dt.datetime(2026, 10, 8, 12, 1, 1, tzinfo=base.KST)), sleeper=lambda _: None)
    assert capture["state"] == "validated_price_observations"
    assert (tmp_path / capture["rawPath"]).is_file()
    manifest = json.loads((tmp_path / capture["manifestPath"]).read_text())
    assert manifest["sameDayIssueWindowEligibleAtCapture"] is False
    assert daily.select_earliest_valid_capture(tmp_path, issue) is None


def test_retry_after_applies_to_rate_limit_responses(tmp_path):
    calls, sleeps = [], []
    def transport(*_args):
        calls.append(True)
        return 429, b"rate limited", {"Retry-After": "1"}
    status, body, _, attempts, error, attempt_records = daily._request_page(
        "https://example.invalid", transport=transport, sleeper=sleeps.append)
    assert (status, body, attempts, error) == (429, b"rate limited", 3, None)
    assert len(calls) == 3 and sleeps == [1.0, 1.0]


def test_capture_reuse_rejects_raw_paths_outside_capture_directory(tmp_path):
    issue = dt.date(2026, 10, 7)
    capture = daily.capture_national_page(
        root=tmp_path, issue_date=issue, transport=_transport(body=_html()),
        clock=_clock(dt.datetime(2026, 10, 7, 10, 30, tzinfo=base.KST),
                     dt.datetime(2026, 10, 7, 10, 30, 1, tzinfo=base.KST)), sleeper=lambda _: None)
    manifest_path = tmp_path / capture["manifestPath"]
    manifest = json.loads(manifest_path.read_text())
    manifest["rawResponse"]["path"] = "outputs/prospective-cabbage/daily/2026-10-07/outside.html"
    manifest_path.write_text(json.dumps(manifest))
    assert daily.select_earliest_valid_capture(tmp_path, issue) is None


def test_same_second_capture_precedes_persisted_issue_time_at_microsecond_precision(tmp_path):
    _install_daily_inputs(tmp_path)
    issue = dt.date(2026, 10, 8)
    clock = _clock(
        dt.datetime(2026, 10, 8, 10, 30, 0, 100000, tzinfo=base.KST),
        dt.datetime(2026, 10, 8, 10, 30, 0, 200000, tzinfo=base.KST),
        dt.datetime(2026, 10, 8, 10, 30, 0, 900000, tzinfo=base.KST),
        dt.datetime(2026, 10, 8, 10, 30, 0, 950000, tzinfo=base.KST),
        dt.datetime(2026, 10, 8, 10, 30, 0, 999000, tzinfo=base.KST),
    )
    result = daily.issue_daily(root=tmp_path, issue_date=issue,
                               transport=_transport(body=_html(day="20261007", date_label="2026년 10월 07일")),
                               clock=clock, sleeper=lambda _: None)
    assert result["status"] == "complete"
    artifact_path = tmp_path / result["artifact"]
    artifact = json.loads(artifact_path.read_text())
    captured = dt.datetime.fromisoformat(artifact["inputs"]["capturedAtKst"])
    issued = dt.datetime.fromisoformat(artifact["issueAtKst"])
    assert captured < issued
    assert issued.microsecond == 999000
