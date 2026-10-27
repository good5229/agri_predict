import datetime as dt
import math

import pytest

from agri_data import national_gap_residual as gap


def _row(day, *, target=None, horizon=1, ridge=100.0):
    origin = dt.date.fromisoformat(day)
    target_day = target or (origin + dt.timedelta(days=horizon)).isoformat()
    return {"product": "cabbage", "origin": origin.isoformat(), "target": target_day,
            "horizon": horizon, "cutoff": (origin - dt.timedelta(days=2)).isoformat(),
            "predictions": {"ridge_alpha_100": ridge, "calendar_sarima": ridge * 1.01}}


def test_month_fit_excludes_labels_after_asof_cutoff():
    rows = []
    national = {}
    market = {}
    start = dt.date(2025, 10, 1)
    for i in range(100):
        origin = start + dt.timedelta(days=i)
        row = _row(origin.isoformat())
        rows.append(row)
        target = dt.date.fromisoformat(row["target"])
        national[target] = 110.0
        market[origin] = 120.0
    fit = gap._fit_month(rows, "cabbage", dt.date(2026, 1, 1), national, market)
    assert fit["fitCutoff"] == "2025-12-30"
    assert fit["maxLabelDate"] <= fit["fitCutoff"]
    mature_with_features = {r["target"] for r in rows if r["target"] <= fit["fitCutoff"]
                            and gap._features("cabbage", r, national, market)[2]["commonFeaturesAvailable"]}
    assert fit["uniqueTargetDays"] == len(mature_with_features)


def test_candidate_manifest_hash_and_answer_tampering_are_rejected():
    doc = {"protocolId": gap.PROTOCOL_ID, "phase": "forecast_only", "targetValuesAttached": False,
           "codeSha256": gap._sha(__import__("pathlib").Path(gap.__file__).read_bytes()),
           "plannedForecastCount": 1,
           "forecasts": [{"product": "cabbage", "origin": "2026-01-01", "target": "2026-01-02", "horizon": 1}]}
    doc["manifestSha256"] = gap._hash(doc)
    gap.verify_manifest(doc)
    changed = dict(doc)
    changed["forecasts"] = [dict(doc["forecasts"][0], actual=3.0)]
    with pytest.raises(ValueError, match="manifest hash mismatch|Answer|answer"):
        gap.verify_manifest(changed)
    changed = dict(doc)
    changed["forecasts"] = [dict(doc["forecasts"][0], actual=3.0)]
    changed.pop("manifestSha256")
    changed["manifestSha256"] = gap._hash(changed)
    with pytest.raises(ValueError, match="answer field"):
        gap.verify_manifest(changed)


def test_manifest_replay_rejects_rebuilt_forecast_change(monkeypatch):
    forecast = {"product": "cabbage", "origin": "2026-01-01", "target": "2026-01-02", "horizon": 1}
    doc = {"protocolId": gap.PROTOCOL_ID, "phase": "forecast_only", "targetValuesAttached": False,
           "codeSha256": gap._sha(__import__("pathlib").Path(gap.__file__).read_bytes()),
           "plannedForecastCount": 1, "forecasts": [forecast]}
    doc["manifestSha256"] = gap._hash(doc)
    rebuilt = dict(doc)
    rebuilt["forecasts"] = [dict(forecast, predictions={"gap_residual_full": 123.0})]
    rebuilt["manifestSha256"] = gap._hash(rebuilt)
    monkeypatch.setattr(gap, "build_manifest", lambda *args: rebuilt)
    with pytest.raises(ValueError, match="replay mismatch"):
        gap._verify_replay(doc, {}, {}, {}, "unused")


def test_input_manifest_and_snapshot_hashes_fail_closed(tmp_path, monkeypatch):
    doc = {"window": ["2025-10-01", "2026-06-30"], "forecasts": [],
           "sourceDigests": {"archiveSha256": "archive"}, "manifestSha256": ""}
    doc["manifestSha256"] = gap._hash({k: v for k, v in doc.items() if k != "manifestSha256"})
    frozen = tmp_path / "frozen.json"
    frozen.write_text(__import__("json").dumps(doc), encoding="utf-8")
    monkeypatch.setitem(gap.MANIFEST_PATHS, "development", str(frozen))
    monkeypatch.setitem(gap.MANIFEST_SHA256, "development", gap._sha(frozen.read_bytes()))
    monkeypatch.setitem(gap.MANIFEST_CONTENT_SHA256, "development", doc["manifestSha256"])
    source = {"sourceDigests": {"archiveSha256": "archive"}}
    monkeypatch.setattr(gap.base, "_verify_manifest", lambda *_: None)
    monkeypatch.setattr(gap.base, "_load_source", lambda *_: source)
    monkeypatch.setattr(gap.base, "_build_manifest", lambda *_args, **_kwargs: {"forecasts": []})
    assert gap._verify_input_manifest(doc, "development", {"archivePath": "synthetic.zip"}) is source
    tampered = {**doc, "window": ["2025-10-02", "2026-06-30"]}
    with pytest.raises(ValueError, match="differs from the frozen input"):
        gap._verify_input_manifest(tampered, "development", {"archivePath": "synthetic.zip"})
    snapshot = tmp_path / "prices.json"
    snapshot.write_text('{"products": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        gap._market_series(snapshot)


def test_development_and_fresh_manifest_use_distinct_source_bundles(monkeypatch):
    development_bundle = {"archivePath": "national.zip"}
    fresh_bundle = {"archivePath": "national.zip", "coverageReportPath": "coverage.json"}
    development_source = {"sourceDigests": {"archiveSha256": "archive", "combinedObservationsSha256": "archive-observations"}}
    fresh_source = {"sourceDigests": {"archiveSha256": "archive", "coverageReportSha256": "coverage",
                                       "combinedObservationsSha256": "fresh-observations"}}
    calls = []

    def verify(doc, kind, bundle):
        calls.append((kind, bundle))
        return development_source if kind == "development" else fresh_source

    monkeypatch.setattr(gap, "_verify_input_manifest", verify)
    dev, fresh = gap._verified_sources({}, {}, {"development": development_bundle, "fresh": fresh_bundle})
    assert dev is development_source and fresh is fresh_source
    assert calls == [("development", development_bundle), ("fresh", fresh_bundle)]
    with pytest.raises(ValueError, match="archive-only"):
        gap._verified_sources({}, {}, {"development": fresh_bundle, "fresh": fresh_bundle})


def test_market_features_require_asof_overlap_and_preserve_no_market_fallback():
    national = {dt.date(2026, 1, 1): 100.0, dt.date(2026, 1, 8): 110.0}
    market = {dt.date(2026, 1, 1): 120.0, dt.date(2026, 1, 8): 132.0,
              dt.date(2026, 1, 12): 9999.0}
    row = _row("2026-01-14")
    full, common, audit = gap._features("cabbage", row, national, market)
    assert len(full) == 5 and len(common) == 3
    assert audit["gapDate"] == "2026-01-08"  # Jan 12 has no corresponding national observation.
    assert audit["nationalChangeReferenceDate"] == "2026-01-01"
    assert audit["nationalChangeRecentDate"] == "2026-01-08"
    assert math.isfinite(full[0])
    no_market = {}
    _, _, missing = gap._features("cabbage", row, national, no_market)
    assert missing["marketFeaturesAvailable"] is False
    sparse_national = {dt.date(2026, 1, 8): 110.0}
    _, common_missing, audit_missing = gap._features("cabbage", row, sparse_national, market)
    assert common_missing[0] is None
    assert audit_missing["nationalChangeAvailable"] is False
    assert audit_missing["nationalChangeReferenceDate"] is None


def test_missing_sarima_ridge_ratio_is_explicit_and_falls_back_for_both_candidates():
    national = {dt.date(2026, 1, 1) + dt.timedelta(days=i): 100.0 + i for i in range(25)}
    market = {d: v * 1.1 for d, v in national.items()}
    row = _row("2026-01-24")
    row["predictions"]["calendar_sarima"] = None
    full, common, audit = gap._features("cabbage", row, national, market)
    assert common[1] is None and full[3] is None
    assert audit["componentRatioAvailable"] is False
    assert audit["commonFeaturesAvailable"] is False
    assert audit["commonFeaturesReason"] == "ridge_or_sarima_not_positive_finite"
    for features in (full, common):
        value, lineage = gap._correct(object(), features, 100.0, audit["commonFeaturesReason"])
        assert value == 100.0
        assert lineage["fallback"] is True
        assert lineage["fallbackReason"] == "ridge_or_sarima_not_positive_finite"


def test_correction_is_finite_capped_and_falls_back_with_reason():
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, _rows):
            return [self.value]

    value, audit = gap._correct(Constant(3.0), [1.0], 100.0)
    assert value == pytest.approx(100.0 * math.exp(gap.CLIP))
    assert audit["appliedLogCorrection"] == gap.CLIP
    fallback, audit = gap._correct(Constant(float("nan")), [1.0], 100.0)
    assert fallback == 100.0
    assert audit["fallback"] is True
    assert audit["fallbackReason"] == "nonfinite_log_correction"
    fallback, audit = gap._correct(Constant(1.0), [1.0], 100.0, "market_features_unavailable")
    assert fallback == 100.0 and audit["fallbackReason"] == "market_features_unavailable"
    value, audit = gap._correct(Constant(1.0), [1.0], float("inf"))
    assert value is None and audit["fallbackReason"] == "ridge100_not_positive_finite"


def test_two_market_snapshot_allows_unmapped_garlic_but_rejects_duplicate_ids(tmp_path, monkeypatch):
    import json

    snapshot = tmp_path / "prices.json"
    rows = [{"id": p, "observations": []} for p in (*gap.PRODUCTS, "garlic")]
    snapshot.write_text(json.dumps({"products": rows}), encoding="utf-8")
    monkeypatch.setattr(gap, "_sha", lambda _raw: gap.MARKET_SHA256)
    series, _ = gap._market_series(snapshot)
    assert set(series) == set(gap.PRODUCTS)
    rows.append({"id": "cabbage", "observations": []})
    snapshot.write_text(json.dumps({"products": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="product keys mismatch"):
        gap._market_series(snapshot)


def test_calendar_dm_holm_fixed_family_and_unavailable_members():
    diffs = {dt.date(2026, 1, 1) + dt.timedelta(days=i): (1.0 if i % 2 else -1.0) for i in range(40)}
    available = gap._dm(diffs, 9)
    assert available["status"] == "available"
    short = gap._dm(dict(list(diffs.items())[:10]), 9)
    assert short["status"] == "inference_unavailable_insufficient_pairs"
    assert short["pValue"] == 1.0
    family = [{"dm": {"9": gap._dm(diffs if i == 0 else dict(list(diffs.items())[:10]), 9)}} for i in range(320)]
    gap._apply_family_holm(family, 9)
    assert len(family) == 320
    assert all("holmAdjustedPValue" in r["dm"]["9"] for r in family)
    assert family[-1]["dm"]["9"]["holmAdjustedPValue"] == 1.0
    assert family[-1]["dm"]["9"]["conclusion"] == "inference_unavailable"


def test_predeclared_gate_reports_failed_crop_threshold():
    macro = {m: {"mae": 100.0, "mape": 10.0} for m in gap.METHODS}
    per_product = {p: {m: {"equalHorizonMape": 10.0, "equalHorizonMae": 100.0} for m in gap.METHODS}
                   for p in gap.PRODUCTS}
    per_product["pepper"]["gap_residual_full"]["equalHorizonMae"] = 106.0
    coverage = {m: {"issued": 4000} for m in gap.METHODS}
    joint = [{"product": p, "horizon": h, "comparator": "ridge_alpha_100",
              "verdictByBandwidth": {"9": {"joint": "improvement"}, "28": {"joint": "improvement"}}}
             for p in gap.PRODUCTS for h in gap.HORIZONS]
    result = gap._period_gate(True, macro, per_product, coverage, joint)
    assert result["status"] == "fail"
    assert "eachProductMaeWorseningAtMost5pct" in result["failureReasons"]


def test_json_output_is_create_only(tmp_path):
    output = tmp_path / "manifest.json"
    gap._write_json_create_only(output, {"value": 1})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        gap._write_json_create_only(output, {"value": 2})
    assert output.read_bytes() == before
