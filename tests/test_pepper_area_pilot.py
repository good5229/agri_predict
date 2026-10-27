import datetime as dt
import json

import pytest

from agri_data import pepper_area_pilot as pilot


def test_area_feature_uses_official_preliminary_values_only_after_both_postings():
    feature, provenance, reason = pilot.area_feature(2023, dt.date(2023, 8, 29))
    assert feature is None and reason == "area_not_yet_published"
    value, provenance, reason = pilot.area_feature(2023, dt.date(2023, 8, 31))
    assert reason is None
    assert value == pytest.approx(__import__("math").log(27129 / 29770))
    assert provenance["currentHa"] == 27129
    assert provenance["availableFrom"] == "2023-08-30"
    # The historical feature references the contemporaneous preliminary value,
    # not a later finalized acreage revision.
    assert pilot.AREA[2023][0] != 27132


def test_training_samples_are_strictly_before_each_origin(monkeypatch):
    days = {dt.date(2016, 1, 1) + dt.timedelta(days=i): (100.0 + i, 10.0) for i in range(1400)}
    monkeypatch.setattr(pilot, "_feature", lambda s, o, t, h: ((1.0,) * 15, o, .01, {}) if o.year in pilot.TRAIN_YEARS and o.month in (9, 10) else None)
    monkeypatch.setattr(pilot.live_forecast, "_strict_feature_row", lambda *args: (dt.date(2016,1,1), (1.0,)*13))
    rows, direct = pilot._train_rows(days, dt.date(2019, 12, 31))
    assert rows
    assert all(row[0] < dt.date(2019, 12, 31) for row in rows)
    assert all((row[0] - dt.timedelta(days=1)).year in pilot.TRAIN_YEARS for row in rows)
    assert any(row[0].month == 11 for row in rows)
    assert all(row[0] >= pilot.DIRECT_TRAIN_START for horizon_rows in direct.values() for row in horizon_rows)


def test_coverage_counts_all_planned_origins_and_missing_targets():
    rows = [{"horizon":1,"targetStatus":"present" if i < 50 else "missing_observation",
             "pairedEvaluable":False,"statuses":{model:"unavailable" for model in pilot.MODELS}}
            for i in range(61)]
    result = pilot._coverage_summary(rows, 1)
    assert result["planned"] == 61
    assert result["targetObserved"] == 50
    assert result["targetMissing"] == 11


def test_pooled_models_have_16_and_15_columns_and_direct_keeps_full_vector():
    rows = [(dt.date(2018, 9, 1) + dt.timedelta(days=i), (float(i),) * 15, .02, float(i), {}) for i in range(100)]
    candidate = pilot._fit(rows, 16)
    ablation = pilot._fit(rows, 15)
    direct_rows = [(r[0], (float(i),) * 13, float(i)) for i, r in enumerate(rows)]
    direct = pilot._fit(direct_rows, None)
    assert candidate[-1].n_features_in_ == 16
    assert ablation[-1].n_features_in_ == 15
    assert direct[-1].n_features_in_ == 13


def test_growth_gate_applies_pair_coverage_and_metric_thresholds():
    metrics = {name: {"macroMape": mape, "macroMae": mae} for name, mape, mae in (
        ("candidatePrice", 8.0, 10.0), ("ablationPrice", 9.0, 10.2),
        ("directPrice", 9.5, 10.1), ("persistencePrice", 12.0, 11.0))}
    coverage = [{"allComparatorsAtLeast30": True, "candidateCoverageNoDecline": True} for _ in pilot.HORIZONS]
    assert pilot._growth_gate(metrics, coverage)["passed"]
    coverage[1]["allComparatorsAtLeast30"] = False
    assert not pilot._growth_gate(metrics, coverage)["allHorizonPairsAtLeast30"]
    coverage[1]["allComparatorsAtLeast30"] = True
    coverage[1]["candidateCoverageNoDecline"] = False
    assert not pilot._growth_gate(metrics, coverage)["candidateCoverageNoDecline"]


def test_dm_pair_minimum_and_invalid_variance_are_explicit():
    short = pilot._dm({dt.date(2020, 1, 1) + dt.timedelta(days=i): 1.0 for i in range(29)}, 9)
    assert short["status"] == "insufficient_pairs"
    constant = pilot._dm({dt.date(2020, 1, 1) + dt.timedelta(days=i): 0.0 for i in range(40)}, 9)
    assert constant["status"] == "invalid_hac_variance"
    rows = [{"target": (dt.date(2020, 1, 1) + dt.timedelta(days=i)).isoformat(),
             "horizon": h, "targetStatus": "present", "actual": 10. + i * .1, "candidatePrice": 9., "ablationPrice": 11.,
             "directPrice": 11., "persistencePrice": 11., "statuses": {k:"available" for k in pilot.MODELS},
             "pairedEvaluable": True} for h in pilot.HORIZONS for i in range(40)]
    tests = pilot._dm_family(rows, 9)
    assert len(tests) == 18
    assert all(t["status"] == "tested" for t in tests)
    assert all("holmPValue" in t for t in tests)


def test_immutable_writer_preserves_first_result(tmp_path):
    output = tmp_path / "result.json"
    pilot._write_immutable(output, {"run": 1})
    with pytest.raises(FileExistsError):
        pilot._write_immutable(output, {"run": 2})
    assert json.loads(output.read_text()) == {"run": 1}
