import datetime as dt
import json

import pytest

from agri_data import third_crop_audit as audit


def _reports(*, daily_missing=None, ridge_shift=False, omit_market=None):
    direct_rows, recent_rows, market_rows = [], [], []
    spans = [(dt.date(2024, 9, 25), 30), (dt.date(2025, 9, 25), 35)]
    for product in audit.PRODUCTS:
        for start, count in spans:
            for i in range(count):
                target = start+dt.timedelta(days=i)
                for horizon in audit.HORIZONS:
                    origin = target-dt.timedelta(days=horizon)
                    key=(product,target.isoformat(),horizon)
                    direct = {"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":horizon,
                              "actual":100,"direct":110 if not ridge_shift else 111,"persistence":120,
                              "monthlyPooledRidge":100,"dailyPooledRidge":None if key==daily_missing else 101}
                    recent = {"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":horizon,
                              "actual":100,"directRidge":110 if not ridge_shift else 111,"persistence":120,
                              "recentWindowHistGb":102,"fullHistoryHistGb":103,"monthlyPooledHistGb":104}
                    market = {"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":horizon,
                              "actual":100,"directRidge":110 if not ridge_shift else 111,"persistence":120,"candidatePrediction":105}
                    direct_rows.append(direct); recent_rows.append(recent)
                    if key != omit_market: market_rows.append(market)
    return ({"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":direct_rows},
            {"schemaVersion":1,"protocol":{"id":"recent-730-day-direct-hist-gb-2026-10-06"},"pairedRows":recent_rows},
            {"schemaVersion":1,"status":"complete","protocol":{"id":"market-wise-direct-ridge-exante-share-2026-10-07"},"pairedRows":market_rows})


def _build(reports):
    return audit.build(*reports, expected_paired_rows=len(reports[0]["pairedRows"]))


def test_frozen_windows_selection_test_family_and_json_serialization():
    reports=_reports()
    result=_build(reports)
    assert result["selection"]["onion"]["selected"]=="monthlyPooledRidge"
    assert all(value["selected"]=="monthlyPooledRidge" for value in result["selection"].values())
    assert len(result["test"]["comparisons"])==200
    assert all("holmPValue" in row["dm"]["9"] and "holmPValue" in row["dm"]["28"] for row in result["test"]["comparisons"])
    assert all(row["n"]==35 for row in result["test"]["comparisons"])
    assert result["test"]["goal"]["cropsBelow10Percent"]==5
    assert all(row["selectedCoverageNondeclining"] for row in result["test"]["coverage"].values())
    assert {audit._season(dt.date(2025,m,15)) for m in (1,4,7,10)}=={"DJF","MAM","JJA","SON"}
    json.dumps(result,allow_nan=False)


def test_selection_embargo_is_before_first_test_origin():
    assert audit.FIRST_TEST_ORIGIN==dt.date(2025,9,15)
    assert audit.SELECTION_END==dt.date(2025,9,1)
    assert audit.SELECTION_END<=audit.FIRST_TEST_ORIGIN-dt.timedelta(days=14)


def test_missing_method_is_audited_and_excluded_without_false_coverage():
    key=("onion","2025-09-25",1)
    result=_build(_reports(daily_missing=key))
    assert result["alignment"]["alignedRows"]==len(result["alignment"]["invalidRows"])+len(result["selectionAudit"]["pairedRows"])+len(result["test"]["pairedRows"])
    cov=result["test"]["coverage"]["onion/1"]
    assert cov["methodCoverage"]["dailyPooledRidge"]==34
    assert cov["coverageNondeclining"]["dailyPooledRidge"] is False
    assert result["alignment"]["invalidRows"]


def test_rejects_row_identity_and_baseline_mismatches():
    reports=list(_reports())
    reports[2]["pairedRows"].pop()
    with pytest.raises(ValueError,match="pairedRows count"):
        _build(tuple(reports))
    reports=list(_reports())
    reports[1]["pairedRows"][0]["directRidge"] += 1e-12
    with pytest.raises(ValueError,match="directRidge mismatch"):
        _build(tuple(reports))


def test_rejects_origin_mismatch_duplicate_and_invalid_baseline():
    reports=list(_reports())
    reports[0]["pairedRows"][0]["origin"]="2020-01-01"
    with pytest.raises(ValueError,match="origin/horizon mismatch"):
        _build(tuple(reports))
    reports=list(_reports())
    reports[1]["pairedRows"].append(dict(reports[1]["pairedRows"][0]))
    with pytest.raises(ValueError,match="duplicate recent identity"):
        _build(tuple(reports))
    reports=list(_reports())
    reports[2]["pairedRows"][0]["persistence"]=0
    with pytest.raises(ValueError,match="positive and finite"):
        _build(tuple(reports))


def test_immutable_writer_refuses_overwrite_and_cleans_up_failed_write(tmp_path,monkeypatch):
    path=tmp_path/"report.json"; path.write_bytes(b"old")
    with pytest.raises(FileExistsError): audit._write_immutable(path,b"new")
    assert path.read_bytes()==b"old" and list(tmp_path.iterdir())==[path]
    path.unlink(); real_fdopen=audit.os.fdopen
    class Failure:
        def __init__(self,fd): self.stream=real_fdopen(fd,"wb")
        def __enter__(self): return self
        def write(self,_): raise OSError("simulated")
        def __exit__(self,*exc): self.stream.close()
    monkeypatch.setattr(audit.os,"fdopen",lambda fd,_mode:Failure(fd))
    with pytest.raises(OSError,match="simulated"): audit._write_immutable(path,b"partial")
    assert not path.exists() and list(tmp_path.iterdir())==[]
