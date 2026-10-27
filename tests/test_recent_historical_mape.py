import datetime as dt

import pytest

from agri_data import recent_historical_mape as report


def _two_market(days=35):
    products = []
    start = dt.date(2025, 10, 6)
    for pid in report.PRODUCTS_2M:
        observations = []
        persistence = []
        ridge = []
        for i in range(days):
            target = start + dt.timedelta(days=i)
            observations.append({"date": target.isoformat(), "actual": 100 + i})
            for h in report.HORIZONS:
                origin = target - dt.timedelta(days=h)
                ridge.append({"target": target.isoformat(), "origin": origin.isoformat(), "horizon": h, "predicted": 101 + i})
                persistence.append({"target": target.isoformat(), "origin": origin.isoformat(), "horizon": h, "predicted": 102 + i})
        products.append({"id": pid, "observations": observations, "predictions": persistence,
                         "runs": [{"modelId": "ridge", "cadence": "daily", "predictions": ridge}]})
    return {"schemaVersion": 2, "meta": {"contextEnd": "2026-10-05"}, "products": products}


def _national(days=35):
    products = []
    start = dt.date(2025, 7, 1)
    for pid in report.PRODUCTS_NATIONAL:
        for i in range(days):
            target = start + dt.timedelta(days=i)
            for h in report.HORIZONS:
                origin = target - dt.timedelta(days=h)
                products.append({"product": pid, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                                 "nationalActual": 100 + i, "candidatePrediction": 99 + i,
                                 "directRidge": 102 + i, "persistence": 103 + i})
    coverage = {f"{pid}/{h}": {"candidateCoverageLossKeys": {"directRidge": [], "persistence": []}}
                for pid in report.PRODUCTS_NATIONAL for h in report.HORIZONS}
    return {"schemaVersion": 1, "status": "complete",
            "protocol": {"id": "national-wholesale-target-transfer-2026-10-07"},
            "hashes": {"archiveSha256": "a" * 64}, "coverage": coverage, "pairedRows": products}


def test_frozen_families_windows_holm_and_macro_are_constructed():
    result = report.build(_two_market(), _national())
    two = result["twoMarket"]
    nat = result["national"]
    assert two["familySizePerBandwidth"] == 100
    assert [w["window"] for w in two["windows"]] == ["1y", "2y", "3y"]
    one = two["windows"][0]
    assert len(one["comparisons"]) == 100
    assert len(one["pairedRows"]) == 5 * 10 * 35
    assert all("holmPValue" in c["dm"]["9"] for c in one["comparisons"])
    assert one["macroMetrics"]["ridge"]["byProduct"]["cabbage"]["equalHorizonMape"] is not None
    assert nat["familySizePerBandwidth"] == 160
    assert [w["window"] for w in nat["windows"]] == ["1y", "2y_available"]
    assert len(nat["windows"][0]["comparisons"]) == 160
    assert all(c["dm"]["9"]["n"] == 35 for c in nat["windows"][0]["comparisons"])
    assert all("coverageSource" in c for c in one["coverage"].values())


def test_no_mature_pairs_keeps_p_one_tests_and_macro_null():
    result = report.build(_two_market(days=0), _national(days=0))
    two = result["twoMarket"]["windows"][0]
    assert all(c["dm"]["9"]["pValue"] == 1.0 for c in two["comparisons"])
    assert all(c["dm"]["9"]["status"] == "insufficient_pairs" for c in two["comparisons"])
    assert two["macroMetrics"]["ridge"]["byProduct"]["cabbage"]["equalHorizonMape"] is None


def test_rejects_duplicate_or_misaligned_forecast_rows():
    raw = _two_market(days=1)
    row = raw["products"][0]["runs"][0]["predictions"][0]
    row["origin"] = "2025-01-01"
    with pytest.raises(ValueError, match="origin/horizon mismatch"):
        report._load_two_market(raw)


def test_rejects_duplicate_national_identity():
    raw = _national(days=1)
    raw["pairedRows"].append(dict(raw["pairedRows"][0]))
    with pytest.raises(ValueError, match="duplicate national paired row"):
        report._load_national(raw)


def test_rejects_nonpositive_truth():
    raw = _two_market(days=1)
    raw["products"][0]["observations"][0]["actual"] = 0
    rows, _, quality = report._load_two_market(raw)
    assert not any(r["product"] == raw["products"][0]["id"] and r["target"] == dt.date(2025, 10, 6) for r in rows)
    assert quality[raw["products"][0]["id"]]["observationExcluded"] == {"nonpositive": 1}


def test_national_period_reports_source_coverage_losses_and_nonpaired_denominator_limit():
    nat = _national(days=2)
    coverage = nat["coverage"]["cabbage/1"]["candidateCoverageLossKeys"]
    coverage["directRidge"] = [["cabbage", "2025-06-30", "2025-07-01", 1]]
    result = report.build(_two_market(days=1), nat)
    window = result["national"]["windows"][0]
    row = window["coverage"]["cabbage/1"]
    assert row["candidateCoverageLossDates"]["directRidge"] == ["2025-07-01"]
    assert row["coverageNondeclining"] is False
    assert row["candidateCoverageLossCounts"]["directRidge"] == 1
    assert row["commonPairedCount"] == 2
    assert row["candidateCoverage"] is None
    assert row["comparatorCoverage"] == {"directRidge": None, "persistence": None}
    assert row["eligibleCount"] is None and row["eligibleMinusPaired"] is None
    assert row["denominatorScope"].startswith("conditional on source pairedRows")


def test_immutable_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "report.json"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report._write_immutable(output, b"replacement")
    assert output.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [output]


def test_immutable_writer_leaves_no_partial_final_or_temp_after_write_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    real_fdopen = report.os.fdopen

    class FailingStream:
        def __init__(self, fd):
            self.stream = real_fdopen(fd, "wb")

        def __enter__(self):
            return self

        def write(self, _payload):
            raise OSError("simulated disk write failure")

        def __exit__(self, *exc):
            self.stream.close()

    monkeypatch.setattr(report.os, "fdopen", lambda fd, _mode: FailingStream(fd))
    with pytest.raises(OSError, match="simulated disk write failure"):
        report._write_immutable(output, b"partial")
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
