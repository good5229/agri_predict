import datetime as dt

from agri_data import failure_mechanism_diagnostic as diag


def _cell_row(actual, base, candidate, multiplier=1.0, factor=1.0, status="corrected"):
    return {
        "product": "cabbage", "target": dt.date(2025, 3, 1), "horizon": 1,
        "actual": actual, "fixed50": base, "candidate": candidate,
        "correction": {"status": status, "multiplier": multiplier,
                       "appliedFactor": factor, "n": 30},
    }


def test_deltas_count_helpful_harmful_unchanged_and_hindsight_is_constrained_optimum():
    rows = [
        _cell_row(110.0, 100.0, 105.0, multiplier=1.2, factor=1.05),
        _cell_row(90.0, 100.0, 105.0, multiplier=1.2, factor=1.05),
        _cell_row(100.0, 100.0, 100.0, multiplier=1.0, factor=1.0),
    ]
    delta = diag._error_deltas(rows, "candidate")
    assert (delta["helpful"], delta["harmful"], delta["unchanged"]) == (1, 1, 1)
    oracle = diag._hindsight_oracle(rows)
    assert oracle["appliedFactor"] == 1.0
    assert oracle["label"].startswith("HINDSIGHT_ORACLE_NOT_DEPLOYABLE")
    assert oracle["mape"] <= oracle["fixed50Mape"]


def test_group_rates_distinguish_fallback_clipping_and_near_zero_residual():
    rows = [
        _cell_row(110.0, 100.0, 105.0, multiplier=1.4, factor=1.05),
        _cell_row(100.5, 100.0, 100.0, multiplier=1.0, factor=1.0, status="fallback"),
    ]
    summary = diag._group_stats(rows)
    assert summary["clipped"] == 1
    assert summary["appliedAtCap"] == 1
    assert summary["fallback"] == 1
    assert summary["nearZeroRealizedLogResidual"] == 1
    assert summary["nearZeroAppliedCorrection"] == 1


def test_direction_signs_compare_training_multiplier_to_realized_log_residual():
    rows = [
        _cell_row(110.0, 100.0, 105.0, multiplier=1.2, factor=1.05),
        _cell_row(90.0, 100.0, 95.0, multiplier=1.2, factor=1.05),
    ]
    out = diag._signed_direction(rows)
    assert out["sameNonzeroDirection"] == 1
    assert out["oppositeNonzeroDirection"] == 1
    assert out["sameDirectionRateAmongBothNonzero"] == 0.5


def test_2026_common_key_metrics_are_paired_and_stratified():
    target = dt.date(2026, 1, 20)
    origin = target - dt.timedelta(days=2)
    methods = {m: {"status": "issued", "prediction": 100.0 + i,
                   "provenance": {"sourceAgeDays": 2}} for i, m in enumerate(diag.ONLINE_METHODS)}
    raw = {"phase": "forecast_only", "targetValuesAttached": False,
           "evaluationForecasts": [{"product": "cabbage", "target": target.isoformat(),
             "origin": origin.isoformat(), "horizon": 2, "eligible": True, "models": methods}]}
    prices = {p: {} for p in diag.PRODUCTS}
    prices["cabbage"] = {origin - dt.timedelta(days=3): 90.0, origin - dt.timedelta(days=1): 100.0, target: 105.0}
    result = diag._diagnose_2026(raw, prices)
    assert result["exactCommonKeysWithPositiveActualAndAllFiveIssued"] == 1
    assert result["uniqueTargetDates"] == 1
    metrics = result["metrics"]
    assert metrics["nPairedKeys"] == 1
    assert metrics["pairedDeltaVsFixed50"]["causal50"]["maeChange"] == -1.0
    assert result["strata"]["horizonBand"]["1-3"]["nPairedKeys"] == 1
    assert result["strata"]["p0AgeBand"]["0-3d"]["nPairedKeys"] == 1
    assert result["features"]["volatilityUsesOnlyObservationsThroughOriginMinus1"] is True


def test_later_price_answers_are_not_parsed_by_2026_cutoff_parser():
    raw = {"schemaVersion": 1, "products": [
        {"id": p, "observations": [{"date": "2026-09-27", "actual": "10"},
                                     {"date": "2026-09-28", "actual": "not-an-answer"}]}
        for p in diag.PRODUCTS
    ]}
    prices = diag._all_price_maps(raw, diag.ONLINE_END)
    assert all(list(v) == [diag.ONLINE_END] for v in prices.values())


def test_feature_inventory_confirms_dropped_original_schema_and_scale_invariance():
    report = diag._feature_inventory()
    assert report["originalDirectFeatureCount"] == 40
    assert report["fullHistoryFeatureCount"] == 71
    assert len(report["originalFeatureNamesNotReused"]) == 40
    audit = report["syntheticScaleInvarianceAudit"]
    assert audit["fullHistory71InvariantToJointScale"] is True
    assert audit["originalDirect40ChangesUnderScale"] is True


def test_weighted_median_oracle_hits_direct_factor_constraint():
    # MAPE-optimal unconstrained multiplier is 2.0; direct constrained optimum is 1.05.
    rows = [_cell_row(200.0, 100.0, 100.0, multiplier=1.0, factor=1.0)]
    oracle = diag._hindsight_oracle(rows)
    assert oracle["multiplier"] == 2.0
    assert oracle["appliedFactor"] == 1.05
