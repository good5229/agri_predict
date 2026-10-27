from __future__ import annotations

import datetime as dt
import json
import math

import pytest

from agri_data import external_reference_v1 as exp


def _query(day: dt.date, market: str, *, state_scope: str = ""):
    return {"cond[trd_clcln_ymd::EQ]": day.isoformat(),
            "cond[whsl_mrkt_cd::EQ]": market,
            "cond[gds_lclsf_cd::EQ]": "10",
            "cond[gds_mclsf_cd::EQ]": "01",
            "cond[gds_sclsf_cd::EQ]": state_scope}


def _row(day, amount, volume, *, grade="A", subclass="07", market="110001"):
    return {"gds_lclsf_cd": "10", "gds_mclsf_cd": "01",
            "gds_sclsf_cd": subclass, "gds_sclsf_nm": "배추",
            "grd_cd": grade, "grd_nm": "상",
            "whsl_mrkt_cd": market, "trd_clcln_ymd": day.isoformat(),
            "trd_se": "경매", "unit_nm": "kg",
            "totprc": amount, "unit_tot_qty": volume}


def test_signed_transaction_aggregation_and_target_replay():
    day = dt.date(2025, 2, 3)
    rows = [
        {"query": _query(day, "110001"), "payload": None, "selectedState": "complete"},
        {"query": _query(day, "110008"), "payload": None, "selectedState": "empty"},
        {"query": _query(day, "110001"), "payload": _row(day, "100", "1"), "selectedState": "complete"},
        {"query": _query(day, "110001"), "payload": _row(day, "-10", "-0.1"), "selectedState": "complete"},
    ]
    channels, audit = exp._transactions_to_channels(rows)
    market = channels["cabbage"][("110001",)][day]
    assert market["amountSigned"] == "90"
    assert market["volumeKgSigned"] == "0.9"
    assert market["price"] == pytest.approx(100)
    prices = {"products": [{"id": "cabbage", "observations": [
        {"date": day.isoformat(), "actual": 100, "volumeKg": 0.9}]}]}
    checked = exp.verify_target_reconciliation(audit, prices, day, day)
    assert checked["status"] == "verified"
    prices["products"][0]["observations"][0]["actual"] = 101
    with pytest.raises(ValueError, match="raw target price mismatch"):
        exp.verify_target_reconciliation(audit, prices, day, day)


def test_channel_selection_uses_only_pre_v_support_and_falls_back_coarsely():
    cutoff = exp.CHANNEL_SUPPORT_END
    days = [cutoff - dt.timedelta(days=i) for i in range(120)]
    channels = {p: {} for p in exp.PRODUCTS}
    channels["cabbage"][("110001", "07", "A")] = {d: {"price": 100} for d in days}
    channels["cabbage"][("110008", "08", "B")] = {d: {"price": 120} for d in days[:119]}
    channels["cabbage"][("110008", "08", "B")][cutoff + dt.timedelta(days=10)] = {"price": 130}
    channels["garlic"][("110001",)] = {d: {"price": 500} for d in days}
    selected = exp.select_channels(channels)
    assert selected["cabbage"]["selected"][0]["channelId"] == ["110001", "07", "A"]
    assert selected["cabbage"]["selected"][0]["supportDays"] == 120
    assert selected["cabbage"]["selected"][0]["supportEnd"] == cutoff.isoformat()
    assert selected["garlic"]["selected"][0]["grain"] == "market"
    assert "overlaps prior market-context" in selected["garlic"]["selected"][0]["noveltyNote"]
    assert selected["onion"]["status"] == "no_supported_channel"


def test_channel_paths_keep_cutoff_asof_age_exact_mask_and_source_date():
    cutoff = dt.date(2025, 3, 10)
    series = {cutoff - dt.timedelta(days=1): 110.0,
              cutoff - dt.timedelta(days=9): 90.0,
              cutoff + dt.timedelta(days=1): 9000.0}
    values, audit = exp.channel_path_features([(("110001", "07", "A"), series)], 100.0, cutoff)
    # Five values per lag: as-of relative price, age, availability, exact price, exact mask.
    assert len(values) == 5 * len(exp.CHANNEL_LAGS)
    assert values[:5] == pytest.approx([math.log(1.1), 1.0, 1.0, 0.0, 0.0])
    assert audit[0]["asOfSourceDate"] == (cutoff - dt.timedelta(days=1)).isoformat()
    assert audit[0]["exactDateObserved"] is False
    lag1_offset = 5 * exp.CHANNEL_LAGS.index(1)
    assert values[lag1_offset + 4] == 1.0
    assert audit[0]["asOfSourceDate"] <= cutoff.isoformat()


def test_basic40_and_candidate_features_use_origin_minus_one_cutoff():
    origin = dt.date(2025, 4, 10)
    target = origin + dt.timedelta(days=3)
    cutoff = origin - dt.timedelta(days=1)
    series = {cutoff - dt.timedelta(days=1): (10.0, 10.0),
              cutoff: (11.0, 10.0), origin: (1000.0, 10.0)}
    channels = [(("110001",), {cutoff: 12.0, origin: 10000.0}, tuple(sorted((cutoff, origin))))]
    row = exp._feature_row(series, channels, origin, target, 3, {})
    assert row is not None
    assert row["cutoff"] == cutoff.isoformat()
    assert row["p0Date"] == cutoff.isoformat()
    assert row["p0"] == 11.0
    assert row["basic40"][0] == pytest.approx(math.log(11.0))


def _forecast_manifest(monkeypatch):
    start, end, origin_floor = exp.PHASES["V"]
    target = start + dt.timedelta(days=20)
    keys = set()
    rows = []
    actuals = {}
    for product in exp.PRODUCTS:
        actuals[(product, target)] = 100.0
        for h in exp.HORIZONS:
            origin = target - dt.timedelta(days=h)
            keys.add((product, target, h))
            rows.append({"product": product, "origin": origin.isoformat(),
                         "cutoff": (origin-dt.timedelta(days=1)).isoformat(),
                         "target": target.isoformat(), "horizon": h,
                         "baseline": {"E106_RHS": 120.0, "E106": 125.0,
                                      "rhsStatus": "issued", "e106Status": "issued"},
                         "CTRL_LGB": {"status": "issued", "prediction": 105.0},
                         "CAND_A": {"status": "issued", "prediction": 100.0}})
    monkeypatch.setattr(exp, "_baseline_index_rows_for_phase", lambda phase: keys)
    baseline_index = {(p, target, h): {"product": p, "origin": target-dt.timedelta(days=h),
                                       "target": target, "horizon": h,
                                       "E106_RHS": 120.0, "E106": 125.0,
                                       "rhsStatus": "issued", "e106Status": "issued"}
                      for p in exp.PRODUCTS for h in exp.HORIZONS}
    monkeypatch.setattr(exp, "_baseline_data_for_phase", lambda phase: baseline_index)
    doc = {"schemaVersion": 1, "phase": "V", "targetValuesAttached": False,
           "forecasts": rows, "coverage": {"portfolio": {"rhsEligibleKeys": 40,
                                                            "rhsIssued": 40, "candidateIssued": 40}}}
    doc["manifestSha256"] = exp._sha(exp._canonical(doc))
    return doc, actuals


def test_fixed_240_dm_families_gate_and_joint_bootstrap(monkeypatch):
    doc, actuals = _forecast_manifest(monkeypatch)
    calls = []
    def fake_bootstrap(rows, **kwargs):
        calls.append(kwargs["block"])
        return {"blockDays": kwargs["block"], "calendarDays": 81,
                "requested": kwargs["replicates"], "valid": kwargs["replicates"],
                "invalid": 0, "interval95": [-1.0, 1.0]}
    monkeypatch.setattr(exp.bootstrap, "circular_block_bootstrap", fake_bootstrap)
    report = exp.score_manifest(doc, actuals, phase="V")
    assert len(report["dm"]) == 480
    assert all(x["familySize"] == 240 for x in report["dmSummaryByBandwidth"])
    assert report["validationGate"]["status"] == "pass"
    assert len(calls) == 12
    assert set(calls) == {14, 28}
    assert all(r["status"] == "insufficient_pairs" for r in report["dm"])
    assert report["storedRHSvsE106Reference"]["cellsPresent"] == 40


def test_fixed_holm_retains_unavailable_tests_as_one():
    records = [{"status": "tested", "pValue": 0.001},
               {"status": "insufficient_pairs", "pValue": 1.0}]
    exp._fixed_holm(records)
    assert records[0]["holmAdjustedPValue"] == pytest.approx(0.002)
    assert records[1]["holmAdjustedPValue"] == 1.0
    assert records[0]["holmSignificant"]
    assert not records[1]["holmSignificant"]


def test_frozen_runtime_matches_reviewed_lightgbm_environment():
    versions = exp._check_runtime()
    assert versions["lightgbm"] == "4.6.0"
    assert versions["python"] == "3.11.4"


def test_source_end_is_bound_to_frozen_snapshot_context_not_generated_date():
    prices = {"meta": {"contextStart": "2014-01-01", "contextEnd": "2026-09-27",
                        "evaluationEnd": "2026-09-27"}}
    assert exp.SOURCE_END == dt.date(2026, 9, 27)
    assert exp.validate_snapshot_range(prices) == {"contextStart": "2014-01-01", "contextEnd": "2026-09-27"}
    prices["meta"]["contextEnd"] = "2026-10-05"
    with pytest.raises(ValueError, match="differs from frozen snapshot"):
        exp.validate_snapshot_range(prices)


def test_execution_identity_freezes_code_dependencies_runtime_and_model_config():
    identity = exp._execution_identity()
    assert len(identity["sourceCodeSha256"]) == 64
    assert len(identity["protocolSha256"]) == 64
    assert len(identity["modelConfigSha256"]) == 64
    assert {"dashboard", "modeling", "equal_info_online_forecasts"} <= set(identity["dependencyInitFileSha256"])
    assert identity["runtime"]["lightgbm"] == "4.6.0"


def test_fixed_month_fit_origin_uses_july_first_even_for_h1_july10():
    key = ("cabbage", dt.date(2026, 7, 11), 1)
    baseline = {key: {"origin": dt.date(2026, 7, 10)}}
    assert exp._fixed_month_fit_origin(dt.date(2026, 7, 10), [key], baseline) == dt.date(2026, 7, 1)
    with pytest.raises(ValueError, match="precedes fixed monthly fit boundary"):
        exp._fixed_month_fit_origin(dt.date(2026, 7, 1), [key],
                                    {key: {"origin": dt.date(2026, 6, 30)}})


def test_price_conversion_rejects_nonfinite_and_overflow_without_clipping():
    assert exp._price_from_log_ratio(100, 0) == pytest.approx(100)
    with pytest.raises(ValueError, match="finite"):
        exp._price_from_log_ratio(100, float("nan"))
    with pytest.raises(ValueError, match="positive"):
        exp._price_from_log_ratio(0, 0)
    with pytest.raises(ValueError, match="overflows"):
        exp._price_from_log_ratio(100, 1000)


def test_p0_staleness_is_measured_from_origin_and_query_controls_classification():
    origin = dt.date(2025, 5, 20)
    target = origin + dt.timedelta(days=1)
    day14, day15 = origin-dt.timedelta(days=14), origin-dt.timedelta(days=15)
    assert exp._feature_row({day14: (100.0, 1.0)}, [], origin, target, 1, {}) is not None
    assert exp._feature_row({day15: (100.0, 1.0)}, [], origin, target, 1, {}) is None
    day = dt.date(2025, 2, 3)
    mismatch = _row(day, "100", "1", subclass="08")
    mismatch["gds_lclsf_cd"] = "11"
    records = [{"query": _query(day, "110001"), "payload": mismatch, "selectedState": "complete",
                "sourceUrl": "source", "requestId": 7}]
    channels, audit = exp._transactions_to_channels(records)
    assert not channels["cabbage"]
    assert audit["excludedRows"]["category_mismatch"] == 1
    assert audit["selectedQueryStates"] == {"complete": 1}
    assert audit["selectedRequestIds"] == ["7"]


def test_signed_cell_issues_are_audited_without_dropping_negative_contributions():
    day = dt.date(2025, 2, 3)
    records = [
        {"query": _query(day, "110001"), "payload": None, "selectedState": "complete"},
        {"query": _query(day, "110008"), "payload": None, "selectedState": "empty"},
        {"query": _query(day, "110001"), "payload": _row(day, "10", "1"), "selectedState": "complete",
         "requestId": 10, "sourceUrl": "url"},
        {"query": _query(day, "110001"), "payload": _row(day, "-20", "-2"), "selectedState": "complete",
         "requestId": 11, "sourceUrl": "url"},
    ]
    channels, audit = exp._transactions_to_channels(records)
    assert not channels["cabbage"]
    assert audit["negativeAdjustmentRows"] == 1
    assert audit["cellIssueCounts"]["nonpositive_net_volume"] >= 1
    assert audit["cellIssues"]
    assert audit["selectedRequestIds"] == ["10", "11"]


def test_gate_detects_swapped_missing_rhs_issued_key_not_just_counts():
    targets = (dt.date(2026, 4, 20), dt.date(2026, 4, 21), dt.date(2026, 4, 22))
    baseline, forecasts = {}, []
    actuals = {(product, day): 100.0 for product in exp.PRODUCTS for day in targets}
    for product in exp.PRODUCTS:
        for horizon in exp.HORIZONS:
            for target in targets:
                key = product, target, horizon
                rhs_issued = target != targets[2]
                baseline[key] = {"E106_RHS": 120.0 if rhs_issued else None, "E106": 125.0,
                                 "rhsStatus": "issued" if rhs_issued else "not_issued", "e106Status": "issued"}
                status = "not_issued" if product == "cabbage" and horizon == 1 and target == targets[0] else "issued"
                forecasts.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                                  "CAND_A": {"status": status, "prediction": 100.0 if status == "issued" else None},
                                  "CTRL_LGB": {"status": "issued", "prediction": 105.0}})
    coverage = exp._coverage(forecasts, baseline, actuals)
    cell = coverage["cabbage/1"]
    assert cell["candidateIssued"] == cell["rhsIssued"]
    assert cell["candidateMissingRhsIssuedKeys"] == [targets[0].isoformat()]
    assert exp._validation_gate([], coverage, "V")["status"] == "fail"


class _ZeroPredictor:
    def predict(self, rows):
        return [0.0 for _ in rows]


class _FirstPathPredictor:
    def predict(self, rows):
        return [float(row[40]) for row in rows]


def test_saved_model_replay_is_persisted_and_tampered_prediction_blocks(tmp_path, monkeypatch):
    import joblib

    phase = "V"
    product = exp.PRODUCTS[0]
    target = dt.date(2026, 4, 25)
    horizon = 1
    origin = target-dt.timedelta(days=horizon)
    key = (product, target, horizon)
    baseline = {key: {"product": product, "origin": origin, "target": target,
                      "horizon": horizon, "E106_RHS": 120.0, "E106": 125.0,
                      "rhsStatus": "issued", "e106Status": "issued"}}
    monkeypatch.setattr(exp, "_baseline_data_for_phase", lambda phase: baseline)
    identity = {"sourceCodeSha256": "code", "runtime": {"python": "synthetic"}}
    monkeypatch.setattr(exp, "_execution_identity", lambda: identity)
    prices_path = tmp_path / "prices.json"
    observations = []
    for offset in range(-60, 1):
        day = origin + dt.timedelta(days=offset)
        observations.append({"date": day.isoformat(), "actual": 100 + offset, "volumeKg": 1})
    prices = {"schemaVersion": 1, "products": [{"id": product, "observations": observations}]}
    prices_path.write_text(json.dumps(prices))
    monkeypatch.setattr(exp, "PRICES_PATH", prices_path)
    series = exp.modeling._observed_series(prices["products"][0])
    channel_key = ("110001", "07", "A")
    selection = {p: {"selected": []} for p in exp.PRODUCTS}
    selection[product] = {"selected": [{"channelId": list(channel_key), "grain": "market_subclass_grade"}]}
    channels = {p: {} for p in exp.PRODUCTS}
    channels[product][channel_key] = {origin-dt.timedelta(days=1): {"price": 200.0}}
    channel_series = exp._channel_series(channels, selection, product)
    feature = exp._feature_row(series, channel_series, origin, target, horizon, {})
    model_path = tmp_path / phase / "models" / "fake.joblib"
    model_path.parent.mkdir(parents=True)
    checkpoint_id = {"executionIdentity": identity, "trainingRows": 100}
    joblib.dump({"checkpointIdentity": checkpoint_id, "CTRL_LGB": _ZeroPredictor(),
                 "CAND_A": _FirstPathPredictor()}, model_path)
    model_hash = exp._file_sha(model_path)
    fit_record = {"product": product, "horizon": horizon, "fitMonth": origin.replace(day=1).isoformat(),
                  "fitOrigin": origin.replace(day=1).isoformat(), "trainingTargetsStrictlyBeforeFitOrigin": True,
                  "modelStatus": "fit", "modelFile": str(model_path.relative_to(tmp_path)),
                  "modelFileSha256": model_hash,
                  "checkpointIdentitySha256": exp._sha(exp._canonical(checkpoint_id))}
    predictions = exp._price_from_log_ratio(feature["p0"], 0.0)
    candidate_prediction = exp._price_from_log_ratio(feature["p0"], feature["candidate"][40])
    row = {"product": product, "origin": origin.isoformat(),
           "cutoff": (origin-dt.timedelta(days=1)).isoformat(), "target": target.isoformat(),
           "horizon": horizon,
           "baseline": {"E106_RHS": 120.0, "E106": 125.0,
                        "rhsStatus": "issued", "e106Status": "issued"},
           "CTRL_LGB": {"status": "issued", "prediction": predictions},
           "CAND_A": {"status": "issued", "prediction": candidate_prediction,
                      "p0": feature["p0"], "p0Date": feature["p0Date"]}}
    manifest = {"schemaVersion": 1, "phase": phase, "targetValuesAttached": False,
                "priceSnapshotSha256": exp._file_sha(prices_path), "executionIdentity": identity,
                "fitRecords": [fit_record], "forecasts": [row]}
    manifest["manifestSha256"] = exp._sha(exp._canonical(manifest))
    result = exp.verify_saved_model_replay(phase, manifest, prices, channels, selection, out_dir=tmp_path)
    assert result["status"] == "pass"
    assert result["pairedForecastsReplayed"] == 1
    manifest["forecasts"][0]["CAND_A"]["prediction"] += 1
    unsigned = dict(manifest)
    unsigned.pop("manifestSha256")
    manifest["manifestSha256"] = exp._sha(exp._canonical(unsigned))
    other_dir = tmp_path / "tampered"
    # Use an otherwise identical checkpoint under the second output root.
    other_path = other_dir / fit_record["modelFile"]
    other_path.parent.mkdir(parents=True)
    joblib.dump({"checkpointIdentity": checkpoint_id, "CTRL_LGB": _ZeroPredictor(),
                 "CAND_A": _FirstPathPredictor()}, other_path)
    manifest["fitRecords"][0]["modelFileSha256"] = exp._file_sha(other_path)
    unsigned = dict(manifest)
    unsigned.pop("manifestSha256")
    manifest["manifestSha256"] = exp._sha(exp._canonical(unsigned))
    with pytest.raises(ValueError, match="replay verification failed"):
        exp.verify_saved_model_replay(phase, manifest, prices, channels, selection, out_dir=other_dir)
