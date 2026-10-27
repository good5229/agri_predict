import datetime as dt
import json
import math

import pytest

from agri_data import onion_garlic_7pct as study


def _snapshot():
    products = []
    start, end = dt.date(2025, 4, 1), study.TEST_END
    for product in study.PRODUCTS:
        obs, persistence, ridge, hist = [], [], [], []
        day = start
        while day <= end:
            obs.append({"date": day.isoformat(), "actual": 100})
            for h in study.HORIZONS:
                target, origin = day, day-dt.timedelta(days=h)
                persistence.append({"target": target.isoformat(), "origin": origin.isoformat(), "horizon": h, "predicted": 112})
                ridge.append({"target": target.isoformat(), "origin": origin.isoformat(), "horizon": h, "predicted": 110})
                hist.append({"target": target.isoformat(), "origin": origin.isoformat(), "horizon": h, "predicted": 105})
            day += dt.timedelta(days=1)
        products.append({"id": product, "observations": obs, "predictions": persistence,
                         "runs": [{"modelId": "ridge", "cadence": "daily", "predictions": ridge},
                                  {"modelId": "hist_gb", "cadence": "daily", "predictions": hist}]})
    return {"schemaVersion": 2, "meta": {"contextEnd": study.TEST_END.isoformat()}, "products": products}


def test_selection_locks_one_rule_per_crop_and_uses_frozen_criteria():
    data = study._parse_snapshot(_snapshot())
    selection_rows, _, _ = study._build_rows(data, study.SELECTION_START, study.SELECTION_END)
    selected = study._select(selection_rows)
    assert set(selected) == set(study.PRODUCTS)
    assert all(v["selected"] == "D" for v in selected.values())
    assert all(v["metrics"]["candidateD"]["equal10HorizonMae"] <=
               v["metrics"]["ridge"]["equal10HorizonMae"] * 1.01 for v in selected.values())
    assert all(v["metrics"]["candidateD"]["equal10HorizonMape"] <=
               v["metrics"]["ridge"]["equal10HorizonMape"] - .1 for v in selected.values())


def test_temporal_residual_cutoff_is_strictly_causal_and_90_calendar_days():
    raw = _snapshot()
    data = study._parse_snapshot(raw)
    target = dt.date(2025, 10, 6)
    row = next(r for r in study._build_rows(data, target, target)[0] if r["product"] == "onion" and r["horizon"] == 1)
    correction = row["correction"]["C"]
    assert correction["n"] == 90
    assert correction["residualWindowStart"] == "2025-06-24"
    assert correction["residualCutoff"] == "2025-09-21"

    # Change an actual one day after the allowed cutoff; the forecast correction must not move.
    for obs in raw["products"][0]["observations"]:
        if obs["date"] == "2025-09-22":
            obs["actual"] = 10000
    changed = study._parse_snapshot(raw)
    after = next(r for r in study._build_rows(changed, target, target)[0] if r["product"] == "onion" and r["horizon"] == 1)
    assert after["correction"]["C"]["scaledLogCorrection"] == correction["scaledLogCorrection"]


def test_insufficient_residual_history_falls_back_and_cap_is_applied():
    values = study._candidate_values(100, 90, [math.log(2)] * 14, 0)
    assert values["C"][0] == 100
    assert values["C"][1]["reason"] == "insufficient_residual_pairs"
    assert values["D"][0] == 95
    capped = study._candidate_values(100, 100, [1.0] * 15, 0)
    assert capped["C"][1]["medianLogResidual"] == 1.0
    assert capped["C"][1]["scaledLogCorrection"] == .04
    capped_negative = study._candidate_values(100, 100, [-1.0] * 15, 0)
    assert capped_negative["C"][1]["scaledLogCorrection"] == -.04
    invalid = study._candidate_values(100, 90, [0.1] * 20, 1)
    assert invalid["C"][0] == 100 and invalid["C"][1]["reason"] == "invalid_residual_input"


def test_test_window_has_frozen_80_test_family_and_holm():
    assert study.SELECTION_END <= study.TEST_START-dt.timedelta(days=10+14)
    assert study.FIRST_TEST_ORIGIN == study.TEST_START-dt.timedelta(days=10)
    raw = _snapshot()
    result = study.build(raw)
    assert result["selection"]["onion"]["selected"] == "D"
    assert result["selection"]["garlic"]["selected"] == "D"
    assert len(result["test"]["comparisons"]) == 80
    assert all("holmPValue" in row["dm"]["9"] and "holmPValue" in row["dm"]["28"]
               for row in result["test"]["comparisons"])
    assert all(row["n"] == 365 for row in result["test"]["comparisons"])
    seasons = {row["value"] for row in result["test"]["yearSeasonMetrics"] if row["stratum"] == "season"}
    assert seasons == {"DJF", "MAM", "JJA", "SON"}
    assert result["protocol"]["earliestTestOrigin"] == "2025-09-26"
    json.dumps(result, allow_nan=False)


def test_selection_falls_back_to_ridge_when_no_candidate_clears_gate():
    rows = []
    for p in study.PRODUCTS:
        for h in study.HORIZONS:
            rows.append({"product": p, "horizon": h, "actual": 100, "ridge": 100,
                         "candidateA": 100, "candidateB": 100, "candidateC": 100, "candidateD": 100})
    selected = study._select(rows)
    assert all(v["selected"] == "ridge" for v in selected.values())


def test_rejects_origin_mismatch_and_duplicate_forecast():
    raw = _snapshot()
    raw["products"][0]["runs"][0]["predictions"][0]["origin"] = "2025-01-01"
    with pytest.raises(ValueError, match="origin/horizon mismatch"):
        study._parse_snapshot(raw)
    raw = _snapshot()
    raw["products"][0]["predictions"].append(dict(raw["products"][0]["predictions"][0]))
    with pytest.raises(ValueError, match="duplicate persistence forecast"):
        study._parse_snapshot(raw)


def test_forecast_coverage_gap_is_preserved_and_blocks_nondeclining_gate():
    raw = _snapshot()
    hist_rows = raw["products"][0]["runs"][1]["predictions"]
    hist_rows[:] = [r for r in hist_rows if not (r["target"] == "2025-10-06" and r["horizon"] == 1)]
    data = study._parse_snapshot(raw)
    rows, exclusions, _ = study._build_rows(data, study.TEST_START, study.TEST_START)
    coverage = study._coverage_audit(data, rows, study.TEST_START, study.TEST_START)
    row = coverage["onion/1"]
    assert row["eligiblePositiveActualCount"] == 1
    assert row["candidateCoverage"] == 0
    assert row["ridgeCoverage"] == row["persistenceCoverage"] == 1
    assert row["commonPairedCount"] == 0
    assert row["eligibleMinusPaired"] == 1
    assert not row["coverageNondeclining"]
    assert exclusions["onion/1"]["histGb_missing"] == 1


def test_immutable_writer_refuses_overwrite_and_cleans_temp_after_write_failure(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_bytes(b"frozen")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        study._write_immutable(output, b"replacement")
    assert output.read_bytes() == b"frozen"
    assert list(tmp_path.iterdir()) == [output]

    output.unlink()
    real_fdopen = study.os.fdopen

    class FailingStream:
        def __init__(self, fd):
            self.stream = real_fdopen(fd, "wb")
        def __enter__(self): return self
        def write(self, _payload): raise OSError("simulated write failure")
        def __exit__(self, *exc): self.stream.close()

    monkeypatch.setattr(study.os, "fdopen", lambda fd, _mode: FailingStream(fd))
    with pytest.raises(OSError, match="simulated write failure"):
        study._write_immutable(output, b"partial")
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
