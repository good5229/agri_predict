import datetime as dt
import hashlib
import json

import pytest

from agri_data import cabbage_national_prospective as mod


def _html(values=None):
    values = values or [("20261001", 928), ("20261002", 1007), ("20261003", 1008),
                        ("20261005", 1026), ("20261006", 934)]
    branches = "\n".join(
        f'if (gradeNm == "고") {{ tmpObj["date"]="{day}"; tmpObj["value"]="999"; }} else {{ '
        f'tmpObj["date"]="{day}"; tmpObj["value"]="{value}"; }}'
        for day, value in values
    )
    return f"function makeGradeChart(gradeNm,index,avgPrice) {{ {branches} }}".encode()


def _manifest(values=None, html_sha256=None):
    values = values or {"2026-10-01": 928, "2026-10-02": 1007, "2026-10-03": 1008,
                        "2026-10-05": 1026, "2026-10-06": 934}
    return json.dumps({
        "selection": {"market": "*전국도매시장", "product": "배추", "productCode": "1001",
                      "variety": "전체", "priceType": "평균", "unit": "원/kg", "selectedDate": "2026-10-06"},
        "visibleMeanPricesWonPerKg": values,
        "rawResponse": {"path": str(mod.RAW_HTML_PATH),
                        "sha256": html_sha256 or hashlib.sha256(_html()).hexdigest()},
        "capturedAtUtcMinute": "2026-10-07T00:26:00+00:00", "sourceUrl": mod.NATIONAL_PAGE_URL,
    }, ensure_ascii=False, sort_keys=True).encode()


def _baseline(issue_date=dt.date(2026, 10, 7), issue_time="2026-10-07T10:15:00+09:00"):
    return {
        "schemaVersion": 1, "issueDateKst": issue_date.isoformat(), "issueAtKst": issue_time,
        "sourceCutoffDate": (issue_date - dt.timedelta(days=1)).isoformat(),
        "products": [{"productId": "cabbage", "status": "available", "method": "ridge",
                      "predictions": [{"horizonDays": h,
                                       "target": (issue_date + dt.timedelta(days=h)).isoformat(),
                                       "predictedPrice": 1000 + h, "sourceDate": "2026-10-06"}
                                      for h in range(1, 11)]}],
    }


def _national_inputs():
    return {"nationalPrices": {"2026-10-01": 928, "2026-10-02": 1007,
                               "2026-10-03": 1008, "2026-10-05": 1026,
                               "2026-10-06": 934},
            "selection": mod.EXPECTED_SELECTION, "htmlSha256": "h", "manifestSha256": "m",
            "pageUrl": mod.NATIONAL_PAGE_URL, "pageMetadata": {"capturedAtUtcMinute": "2026-10-07T00:26:00+00:00"}}


def _local_snapshot():
    return {"products": [{"id": "cabbage", "observations": [
        {"date": "2026-10-01", "actual": 900, "marketCodes": ["110001", "110008"]},
        {"date": "2026-10-02", "actual": 950, "marketCodes": ["110001", "110008"]},
        {"date": "2026-10-03", "actual": 1000, "marketCodes": ["110001", "110008"]},
        {"date": "2026-10-05", "actual": 940, "marketCodes": ["110001"]},
        {"date": "2026-10-06", "actual": 930, "marketCodes": ["110001", "110008"]},
    ]}]}


def test_html_manifest_selection_hash_and_values_are_jointly_validated():
    html = _html([("20260928", 840), ("20260929", 850), ("20260930", 870),
                  ("20261001", 928), ("20261002", 1007), ("20261003", 1008),
                  ("20261005", 1026), ("20261006", 934)])
    manifest = _manifest(html_sha256=hashlib.sha256(html).hexdigest())
    checked = mod.validate_national_inputs(
        html, manifest, expected_html_sha256=hashlib.sha256(html).hexdigest(),
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest())
    assert checked["nationalPrices"] == {
        "2026-09-28": 840.0, "2026-09-29": 850.0, "2026-09-30": 870.0,
        **{k: float(v) for k, v in mod.EXPECTED_NATIONAL_VALUES.items()},
    }
    bad_manifest = _manifest({"2026-10-01": 928, "2026-10-02": 1007,
                              "2026-10-03": 1008, "2026-10-05": 1025,
                              "2026-10-06": 934})
    with pytest.raises(mod.ProspectiveCabbageError, match="frozen transcribed"):
        mod.validate_national_inputs(_html(), bad_manifest,
                                     expected_html_sha256=hashlib.sha256(_html()).hexdigest(),
                                     expected_manifest_sha256=hashlib.sha256(bad_manifest).hexdigest())


def test_national_page_hash_is_pinned():
    html = _html()
    with pytest.raises(mod.ProspectiveCabbageError, match="HTML SHA-256"):
        mod.validate_national_inputs(html, _manifest())


def test_manifest_source_url_requires_the_exact_selected_query():
    html, manifest = _html(), json.loads(_manifest())
    manifest["sourceUrl"] = "https://www.nongnet.or.kr/front/M000000259/marketInfo/nationWide.do"
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    with pytest.raises(mod.ProspectiveCabbageError, match="sourceUrl"):
        mod.validate_national_inputs(html, payload,
                                     expected_html_sha256=hashlib.sha256(html).hexdigest(),
                                     expected_manifest_sha256=hashlib.sha256(payload).hexdigest())


def test_baseline_requires_exact_ridge_horizons_and_cutoff():
    validated = mod.validate_baseline_artifact(_baseline(), issue_date=dt.date(2026, 10, 7))
    assert set(validated["predictions"]) == set(range(1, 11))
    broken = _baseline()
    broken["products"][0]["predictions"].pop()
    with pytest.raises(mod.ProspectiveCabbageError, match="exactly 10"):
        mod.validate_baseline_artifact(broken, issue_date=dt.date(2026, 10, 7))
    broken = _baseline()
    broken["sourceCutoffDate"] = "2026-10-05"
    with pytest.raises(mod.ProspectiveCabbageError, match="source cutoff"):
        mod.validate_baseline_artifact(broken, issue_date=dt.date(2026, 10, 7))


def test_baseline_selection_uses_earliest_valid_same_day_artifact(tmp_path):
    directory = tmp_path / "outputs" / "live-forecasts" / "2026-10-07"
    directory.mkdir(parents=True)
    for name, clock in (("later.json", "10:40:00"), ("earlier.json", "10:05:00")):
        artifact = _baseline(issue_time=f"2026-10-07T{clock}+09:00")
        (directory / name).write_text(json.dumps(artifact), encoding="utf-8")
    (directory / "invalid.json").write_text("{}", encoding="utf-8")
    path, _, validated = mod.select_earliest_baseline(tmp_path, dt.date(2026, 10, 7))
    assert path.name == "earlier.json"
    assert validated["issueAtKst"] == "2026-10-07T10:05:00+09:00"


def test_ratio_uses_o_minus_15_through_o_minus_2_and_excludes_partial_market_days():
    issue_date = dt.date(2026, 10, 7)
    prices = _local_snapshot()
    prices_bytes = json.dumps(prices, sort_keys=True).encode()
    baseline = mod.validate_baseline_artifact(_baseline(), issue_date=issue_date)
    result = mod.build_challenger(issue_date=issue_date,
                                  issued_at=dt.datetime(2026, 10, 7, 10, 30, tzinfo=mod.KST),
                                  national_inputs=_national_inputs(), local_snapshot_bytes=prices_bytes,
                                  local_snapshot=prices, baseline=baseline,
                                  baseline_path="baseline.json", baseline_sha256="b" * 64,
                                  local_expected_sha256=hashlib.sha256(prices_bytes).hexdigest())
    pairs = result["levelRatio"]["eligiblePairs"]
    assert [row["date"] for row in pairs] == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert result["levelRatio"]["windowStart"] == "2026-09-22"
    assert result["levelRatio"]["windowEnd"] == "2026-10-05"
    assert result["levelRatio"]["ratio"] == pytest.approx(1.031111111111111)
    assert result["forecasts"][0]["challengerPrediction"] == pytest.approx(
        result["forecasts"][0]["baselineRidgePrediction"] * result["levelRatio"]["ratio"])


def test_ratio_falls_back_to_one_with_a_reason_under_three_pairs():
    issue_date = dt.date(2026, 10, 7)
    local = {"products": [{"id": "cabbage", "observations": [
        {"date": "2026-10-01", "actual": 900, "marketCodes": ["110001", "110008"]},
        {"date": "2026-10-02", "actual": 0, "marketCodes": ["110001", "110008"]},
    ]}]}
    local_bytes = json.dumps(local, sort_keys=True).encode()
    result = mod.build_challenger(issue_date=issue_date,
                                  issued_at=dt.datetime(2026, 10, 7, 10, 30, tzinfo=mod.KST),
                                  national_inputs=_national_inputs(), local_snapshot_bytes=local_bytes,
                                  local_snapshot=local,
                                  baseline=mod.validate_baseline_artifact(_baseline(), issue_date=issue_date),
                                  baseline_path="baseline.json", baseline_sha256="b" * 64,
                                  local_expected_sha256=hashlib.sha256(local_bytes).hexdigest())
    assert result["levelRatio"]["ratio"] == 1
    assert result["levelRatio"]["fallbackReason"].startswith("fewer than 3")


def test_future_national_dates_and_invalid_timestamp_are_rejected_or_unused():
    issue_date = dt.date(2026, 10, 7)
    national = _national_inputs()
    national["nationalPrices"]["2026-10-08"] = 5000
    local = _local_snapshot()
    local_bytes = json.dumps(local, sort_keys=True).encode()
    result = mod.build_challenger(issue_date=issue_date,
                                  issued_at=dt.datetime(2026, 10, 7, 10, 30, tzinfo=mod.KST),
                                  national_inputs=national, local_snapshot_bytes=local_bytes,
                                  local_snapshot=local,
                                  baseline=mod.validate_baseline_artifact(_baseline(), issue_date=issue_date),
                                  baseline_path="baseline.json", baseline_sha256="b" * 64,
                                  local_expected_sha256=hashlib.sha256(local_bytes).hexdigest())
    assert all(row["date"] <= "2026-10-05" for row in result["levelRatio"]["eligiblePairs"])
    with pytest.raises(mod.ProspectiveCabbageError, match="issue timestamp"):
        mod.build_challenger(issue_date=issue_date,
                             issued_at=dt.datetime(2026, 10, 7, 9, 59, tzinfo=mod.KST),
                             national_inputs=_national_inputs(), local_snapshot_bytes=local_bytes,
                             local_snapshot=local,
                             baseline=mod.validate_baseline_artifact(_baseline(), issue_date=issue_date),
                             baseline_path="baseline.json", baseline_sha256="b" * 64,
                             local_expected_sha256=hashlib.sha256(local_bytes).hexdigest())


def test_live_issue_timestamp_must_match_current_clock_and_window():
    issue_date = dt.date(2026, 10, 7)
    now = dt.datetime(2026, 10, 7, 10, 30, tzinfo=mod.KST)
    assert mod.validate_current_issue_time(issue_date, now, now) == now
    with pytest.raises(mod.ProspectiveCabbageError, match="inconsistent with current"):
        mod.validate_current_issue_time(issue_date,
                                       dt.datetime(2026, 10, 7, 10, 0, tzinfo=mod.KST), now)
    with pytest.raises(mod.ProspectiveCabbageError, match="issue time"):
        mod.validate_current_issue_time(issue_date, now,
                                       dt.datetime(2026, 10, 7, 9, 59, tzinfo=mod.KST))
    with pytest.raises(mod.ProspectiveCabbageError, match="inconsistent with current"):
        mod.issue_from_files(root="/missing", baseline_path=None, issue_date=issue_date,
                             issued_at=dt.datetime(2026, 10, 7, 10, 0, tzinfo=mod.KST),
                             clock=lambda: now)


def test_issue_from_files_rechecks_clock_before_publication_after_noon_crossing(tmp_path, monkeypatch):
    issue_date = dt.date(2026, 10, 7)
    stamps = iter((dt.datetime(2026, 10, 7, 11, 59, 59, 500000, tzinfo=mod.KST),
                   dt.datetime(2026, 10, 7, 11, 59, 59, 800000, tzinfo=mod.KST),
                   dt.datetime(2026, 10, 7, 12, 0, 1, tzinfo=mod.KST)))
    baseline_path = tmp_path / "outputs" / "live-forecasts" / issue_date.isoformat() / "base.json"
    baseline_path.parent.mkdir(parents=True)
    baseline = mod.validate_baseline_artifact(_baseline(), issue_date=issue_date)
    national = _national_inputs()
    monkeypatch.setattr(mod, "_read_verified", lambda *args: b"synthetic")
    monkeypatch.setattr(mod, "_json_bytes", lambda _data, label: _local_snapshot())
    monkeypatch.setattr(mod, "validate_national_inputs", lambda *_args: national)
    monkeypatch.setattr(mod, "select_earliest_baseline",
                        lambda *_args: (baseline_path, b"baseline", baseline))
    monkeypatch.setattr(mod, "build_challenger", lambda **_kwargs: {"artifactType": "synthetic"})
    output = tmp_path / "challenger.json"
    with pytest.raises(mod.ProspectiveCabbageError, match="issue time"):
        mod.issue_from_files(root=tmp_path, baseline_path=None, issue_date=issue_date,
                             issued_at=dt.datetime(2026, 10, 7, 11, 59, 59, 500000, tzinfo=mod.KST),
                             output_path=output, clock=lambda: next(stamps))
    assert not output.exists()


def test_baseline_cannot_be_issued_after_challenger():
    issue_date = dt.date(2026, 10, 7)
    local = _local_snapshot()
    local_bytes = json.dumps(local, sort_keys=True).encode()
    later = mod.validate_baseline_artifact(
        _baseline(issue_time="2026-10-07T10:31:00+09:00"), issue_date=issue_date)
    with pytest.raises(mod.ProspectiveCabbageError, match="must not follow"):
        mod.build_challenger(issue_date=issue_date,
                             issued_at=dt.datetime(2026, 10, 7, 10, 30, tzinfo=mod.KST),
                             national_inputs=_national_inputs(), local_snapshot_bytes=local_bytes,
                             local_snapshot=local, baseline=later,
                             baseline_path="baseline.json", baseline_sha256="b" * 64,
                             local_expected_sha256=hashlib.sha256(local_bytes).hexdigest())


def test_immutable_writer_refuses_to_replace_existing_artifact(tmp_path):
    path = tmp_path / "issued.json"
    mod._write_immutable(path, b"first")
    assert path.read_bytes() == b"first"
    with pytest.raises(mod.ProspectiveCabbageError, match="refusing to overwrite"):
        mod._write_immutable(path, b"second")
    assert path.read_bytes() == b"first"
