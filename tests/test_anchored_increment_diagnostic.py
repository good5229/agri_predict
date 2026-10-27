from __future__ import annotations

import json
import math
import datetime as dt

import pytest

from agri_data import anchored_increment_diagnostic as aid


def test_fixed_increment_and_explicit_numerical_fallbacks():
    assert aid._compose(100.0, 7.0) == (107.0, 107.0, None)
    assert aid._compose(100.0, -100.0) == (100.0, 0.0, "raw_nonpositive")
    pred, raw, reason = aid._compose(1.7e308, 1.7e308)
    assert pred == 1.7e308 and raw == "nonfinite" and reason == "raw_overflow"
    assert json.dumps({"raw": raw}, allow_nan=False) == '{"raw": "nonfinite"}'


def test_full_40_cell_weights_contributions_reconcile_and_keep_boundary_loss():
    pairs = []
    for product in aid.PRODUCTS:
        for horizon in aid.HORIZONS:
            # All cells have the same n. Include a tiny tolerance-classified
            # unchanged delta whose true loss difference remains nonzero.
            for offset, shift in enumerate((1e-14, 20.0)):
                baseline = 1.0
                actual = 100.0
                prediction = baseline + shift
                pairs.append({"product": product, "horizon": horizon,
                              "target": f"2026-07-{11+offset:02d}", "origin": "2026-07-01",
                              "actual": actual, "B": baseline, "prediction": prediction})
    report = aid._contribution(pairs)
    assert report["reconciled"]
    assert report["residualTypes"]["cabbage:unchanged"]["n"] == 10
    assert report["residualTypes"]["cabbage:unchanged"]["maeGain"] > 0.0
    # The tolerance is a diagnostic classification only; it never overwrites
    # the actual dAE/dAPE arithmetic in the contribution table.
    assert report["netSums"]["mae"] < 0


def test_equal_absolute_error_boundary_is_separate_from_exact_loss():
    pairs = []
    for product in aid.PRODUCTS:
        for horizon in aid.HORIZONS:
            pairs.append({"product": product, "horizon": horizon,
                          "target": "2026-07-11", "actual": 100.0,
                          "B": 90.0, "prediction": 110.0})
    report = aid._contribution(pairs)
    assert report["residualTypes"]["cabbage:equal_absolute_error_boundary"]["n"] == 10
    assert report["netSums"]["mae"] == 0.0
    assert report["netSums"]["mape"] == 0.0


def test_80_test_family_keeps_unchanged_cells_as_p_one():
    pairs = []
    for product in aid.PRODUCTS:
        for horizon in aid.HORIZONS:
            for day in range(40):
                pairs.append({"product": product, "horizon": horizon,
                              "target": (dt.date(2026, 7, 11) + dt.timedelta(days=day % 28)).isoformat(),
                              "actual": 100.0, "B": 90.0,
                              "prediction": 90.0 if product in ("cabbage", "garlic") else 91.0})
    result = aid._inference(pairs)
    assert set(result["mainFamilyByBandwidth"]) == {"9", "28"}
    for bandwidth_rows in result["mainFamilyByBandwidth"].values():
        assert len(bandwidth_rows) == 80
        frozen = [row for row in bandwidth_rows if row["product"] in ("cabbage", "garlic")]
        assert len(frozen) == 40
        assert all(row["status"] == "identical_prediction" and row["pValue"] == 1.0 for row in frozen)
        assert all(row["holmPValue"] == 1.0 for row in frozen)
    assert result["protocol"]["HolmFamilySizePerBandwidth"] == 80


def test_score_rejects_unpinned_forecast_before_opening_prices(tmp_path):
    forecast = tmp_path / "forecasts.json"
    forecast.write_text("{}", encoding="utf-8")
    absent_prices = tmp_path / "no-snapshot.json"
    with pytest.raises(ValueError, match="caller-pinned forecast file hash mismatch"):
        aid.score("V", forecast, "0" * 64, absent_prices)


def test_rows_reject_duplicate_or_shifted_horizon_keys():
    row = {"product": "radish", "origin": "2026-07-01", "target": "2026-07-11", "horizon": 10}
    assert aid._key(row) == ("radish", "2026-07-01", "2026-07-11", 10)
    with pytest.raises(ValueError, match="origin/target/horizon"):
        aid._key({**row, "horizon": 9})
    with pytest.raises(ValueError, match="duplicate"):
        aid._rows({"forecasts": [row, row]})


def test_direct_and_baseline_residual_gains_are_reported_separately():
    pairs=[]
    for p in aid.PRODUCTS:
        for h in aid.HORIZONS:
            pairs.append({"product":p,"horizon":h,"target":"2026-07-11","actual":100.0,
                          "B":90.0,"prediction":91.0,"G0":80.0,"G":81.0,
                          "D":90.0 if p=="radish" else None,
                          "E":91.0 if p=="radish" else None,"delta":1.0 if p=="radish" else 0.0,
                          "fallbackReason":None})
    report=aid._contribution(pairs)
    direct=report["directEminusDProductHorizonMonth"]
    assert len(direct)==10
    assert all(r["maeNet"] < 0 for r in direct)


def test_intervention_summary_separates_full_and_paired_denominators_and_reasons():
    rows=[]
    categories=[("cabbage",0.0,1.0,None),
                ("radish",0.0,1.0,"raw_nonpositive"),
                ("onion",0.0,1.0,None),
                ("radish",1e-14,1.0,None)]
    for i,(product,delta,base,fallback) in enumerate(categories):
        origin=dt.date(2026,7,1)+dt.timedelta(days=i)
        horizon=1; target=origin+dt.timedelta(days=horizon)
        applied=base if fallback else base+delta
        rows.append({"product":product,"origin":origin.isoformat(),"target":target.isoformat(),
                     "horizon":horizon,"baselineB":base,"prediction":applied,
                     "delta":delta,"fallbackReason":fallback})
    summary=aid._intervention_summary(rows,{aid._key(rows[0]),aid._key(rows[3])})
    assert summary["fullForecastGrid"]["denominator"]==4
    assert summary["positiveActualPairedGrid"]["denominator"]==2
    counts=summary["fullForecastGrid"]["counts"]
    assert counts["policy_unchanged"]==1
    assert counts["fallback:raw_nonpositive"]==1
    assert counts["exact_zero_delta"]==1
    assert counts["tolerance_unchanged_nonzero_delta"]==1
    assert summary["fullForecastGrid"]["requestedNonzeroDeltaRows"]==1
    assert summary["fullForecastGrid"]["realizedNonzeroChangeRows"]==1
    assert summary["fullForecastGrid"]["aboveToleranceChangeRows"]==0
