import datetime as dt
import hashlib
import json
import math

import pytest

from agri_data import four_crop_forecast as forecast


def test_snapshot_rejects_price_rows_after_declared_cutoff(tmp_path):
    raw = {"schemaVersion": 1, "meta": {"evaluationEnd": "2026-10-08"},
           "products": [{"id": crop, "observations": [
               {"date": "2026-10-09", "actual": 100, "volumeKg": 10, "rowCount": 1}
           ]} for crop in (*forecast.PRODUCTS, "pepper")]}
    data = json.dumps(raw).encode()
    path = tmp_path / "prices.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="after declared source cutoff"):
        forecast._validate_snapshot(raw, data, path)


def test_snapshot_rejects_changed_bytes_at_alternate_path(tmp_path):
    raw = forecast._read_json(forecast.LIVE_PRICES)
    original = forecast.LIVE_PRICES.read_bytes()
    copy = tmp_path / "same-name-prices.json"
    copy.write_bytes(original + b" ")
    changed = json.loads(copy.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="file hash changed"):
        forecast._validate_snapshot(changed, copy.read_bytes(), copy)


def test_snapshot_allows_byte_identical_copy_and_records_external_path(tmp_path):
    copy = tmp_path / "prices.json"
    copy.write_bytes(forecast.LIVE_PRICES.read_bytes())
    raw = forecast._read_json(copy)
    cutoff, sha, _, _ = forecast._validate_snapshot(raw, copy.read_bytes(), copy)
    assert cutoff == forecast.LIVE_CUTOFF
    assert sha == forecast.LIVE_PRICES_SHA256
    assert forecast._display_source_path(copy) == str(copy.resolve())


def test_future_month_fit_issue_origin_and_daily_nan_update_horizon_steps(monkeypatch):
    issue = dt.date(2026, 10, 10)
    dates = [dt.date(2026, 10, day) for day in range(1, 9)]
    rec = {day: {"price": 100.0, "volume": 1000.0} for day in dates}
    series = {day: (100.0, 1000.0) for day in dates}
    seen_fits, seen_appends, seen_steps = [], [], []

    class Model:
        def predict(self, _rows):
            return [math.log(1.1)]

    class Forecast:
        def __init__(self, n):
            self.predicted_mean = [math.log(120.0)] * n

    class Sarima:
        def append(self, values, refit=False):
            assert refit is False
            seen_appends.append(values)
            return self

        def get_forecast(self, steps):
            seen_steps.append(steps)
            return Forecast(steps)

    def period_fit(_records, _series, crop, h, origin, alpha, kind):
        seen_fits.append((crop, h, origin, alpha, kind))
        return Model(), {"count": 200, "firstTarget": "2020-01-01", "lastTarget": "2026-09-30"}, None

    monkeypatch.setattr(forecast.online, "_period_fit", period_fit)
    monkeypatch.setattr(forecast.online, "_fit_sarima", lambda _s, origin: (Sarima(), {"trainEnd": "2026-09-30"}))
    monkeypatch.setattr(forecast.online, "_direct_feature", lambda _s, target, h, _c:
                        ([0.0] * 40, 100.0, {"p0Date": "2026-10-08", "p0AgeDays": 2,
                                             "cutoff": "2026-10-09"}))
    monkeypatch.setattr(forecast.online.ten, "_calendar_values", lambda _s, start, end:
                        {"date": start, "value": float("nan") if start == dt.date(2026, 10, 9) else 100.0})
    rows = forecast._forecast_crop({"cabbage": rec}, {"cabbage": series}, "cabbage", issue,
                                   dt.date(2026, 10, 8))
    assert len(rows) == 10
    assert len(seen_fits) == 10 and all(item[2:] == (dt.date(2026, 10, 1), 10.0, "direct") for item in seen_fits)
    assert rows[0]["target"] == "2026-10-11" and rows[-1]["target"] == "2026-10-20"
    assert all(row["origin"] == "2026-10-10" for row in rows)
    assert len(seen_appends) == 9 and seen_appends[-1]["date"] == dt.date(2026, 10, 9)
    assert math.isnan(seen_appends[-1]["value"])
    assert seen_steps == list(range(2, 12))
    assert rows[0]["models"]["causal50"]["prediction"] == pytest.approx(115.0)


def test_historical_score_uses_exact_four_method_pairs_and_fixed_dm_families(tmp_path, monkeypatch):
    day = dt.date(2026, 1, 11)
    end = dt.date(2026, 1, 21)
    monkeypatch.setattr(forecast, "EVAL_START", day)
    monkeypatch.setattr(forecast, "EVAL_END", end)
    price_doc = {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [{"date": (day+dt.timedelta(days=j)).isoformat(), "actual": 100.0 + i + j,
                                          "volumeKg": 100.0, "rowCount": 1} for j in range(11)]}
        for i, crop in enumerate((*forecast.PRODUCTS, "pepper"))]}
    price_path = tmp_path / "history.json"
    price_bytes = json.dumps(price_doc).encode()
    price_path.write_bytes(price_bytes)
    monkeypatch.setattr(forecast, "HISTORICAL_PRICES_SHA256", hashlib.sha256(price_bytes).hexdigest())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    synthetic_rows = []
    for i, crop in enumerate(forecast.PRODUCTS):
        for j in range(11):
            target = day + dt.timedelta(days=j)
            for h in forecast.HORIZONS:
                origin = target - dt.timedelta(days=h)
                if origin < day:
                    continue
                values = {"causal50": 101.0 + i + j, "causalDirect": 102.0 + i + j,
                          "causalSarima": 103.0 + i + j, "persistence": 104.0 + i + j}
                synthetic_rows.append({"product": crop, "target": target.isoformat(), "horizon": h,
                    "origin": origin.isoformat(), "eligible": True,
                    "models": {method: {"status": "issued", "prediction": value} for method, value in values.items()}})
    missing = next(r for r in synthetic_rows if r["product"] == "cabbage" and r["horizon"] == 1 and r["target"] == "2026-01-12")
    missing["models"]["persistence"] = {"status": "not_issued", "prediction": None, "reason": "synthetic"}
    monkeypatch.setattr(forecast, "_historical_rows", lambda *_args: synthetic_rows)
    result = forecast._score_historical(manifest_path, price_path)
    assert len(result["metricsByProductHorizon"]) == 40
    assert result["metricsByProductHorizon"][0]["allFourWayCommonPairs"] == 9
    assert all(row["allFourWayCommonPairs"] == 11 - row["horizon"]
               for row in result["metricsByProductHorizon"][1:])
    assert len(result["dm"]) == 480
    for bandwidth in forecast.BANDWIDTHS:
        family = [row for row in result["dm"] if row["bandwidth"] == bandwidth]
        assert len(family) == 240
        assert all(row["status"] == "insufficient_pairs" and row["pValue"] == 1.0 for row in family)


def test_historical_manifest_full_file_hash_and_source_binding_are_checked(tmp_path):
    raw = forecast.HISTORICAL_MANIFEST.read_bytes()
    manifest = json.loads(raw)
    assert forecast._historical_rows(manifest, {}, raw)
    with pytest.raises(ValueError, match="file hash mismatch"):
        forecast._historical_rows(manifest, {}, raw + b" ")


def test_html_is_self_contained_and_includes_proposal_and_prediction_grid():
    issue = dt.date(2026, 10, 10)
    rows = []
    for crop in forecast.PRODUCTS:
        for h in forecast.HORIZONS:
            rows.append({"product": crop, "target": (issue+dt.timedelta(days=h)).isoformat(),
                "horizon": h, "models": {m: {"status": "issued", "prediction": 1000.0} for m in forecast.METHODS}})
    future = {"issueDate": issue.isoformat(), "issuedAtKst": "2026-10-10T14:00:00+09:00",
        "targetDateRange": ["2026-10-11", "2026-10-20"], "source": {"sourceCutoff": "2026-10-08", "pricesFileSha256": "abc", "dailyTrainingStatusSnapshotRecordSha256": "ghi"},
        "inputSha256": {"prices": "abc", "module": "def"},
        "sourceAvailability": {crop: {"lastObservedPriceDate": "2026-10-08", "sourceAgeDaysAtIssue": 2, "status": "source_available"} for crop in forecast.PRODUCTS},
        "forecasts": rows}
    metrics = [{"product": crop, "method": method, "horizonsIncluded": 10,
                "meanHorizonMae": 12.0, "meanHorizonMape": 1.2} for crop in forecast.PRODUCTS for method in forecast.METHODS]
    evaluation = {"equalHorizonMetrics": metrics, "dmSummaryByBandwidth": [
        {"bandwidth": 9, "tested": 100, "significantImprovement": 1, "significantWorsening": 2, "inferenceUnavailable": 140},
        {"bandwidth": 28, "tested": 100, "significantImprovement": 0, "significantWorsening": 1, "inferenceUnavailable": 140}]}
    prices = {"schemaVersion": 1, "products": [{"id": crop, "observations": [
        {"date": "2026-10-08", "actual": 980.0, "volumeKg": 100.0, "rowCount": 1}]} for crop in (*forecast.PRODUCTS, "pepper")]}
    page = forecast.render_html(future, evaluation, prices)
    assert "<!doctype html>" in page and "예측아이디어" in page and "활용안" in page
    assert page.count("<td>2026-10-") == 40
    assert "https://" not in page and "2026-10-08" in page and "미평가 상태" in page


def test_output_bundle_is_create_only(tmp_path):
    out = tmp_path / "bundle"
    forecast._write_bundle(out, {"forecasts": []}, {"dm": []}, "<html></html>")
    assert {p.name for p in out.iterdir()} == {"forecast.json", "evaluation.json", "forecast.csv", "index.html"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        forecast._write_bundle(out, {"forecasts": []}, {"dm": []}, "replacement")
    assert (out / "index.html").read_text() == "<html></html>"
