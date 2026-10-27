import datetime as dt
import json

import pytest

from agri_data import national_nonlinear_direct as nd


def _series(end=dt.date(2026, 6, 30)):
    start = end - dt.timedelta(days=300)
    return {start + dt.timedelta(days=i): 100.0 + i * 0.1 for i in range((end-start).days + 1)}


def test_july_first_horizon_ten_uses_june_nineteen_cutoff():
    target = dt.date(2026, 7, 1)
    series = _series(dt.date(2026, 6, 19))
    series[dt.date(2026, 6, 20)] = 999999.0
    feat = nd._feature(series, target, 10)
    assert feat is not None
    assert feat[0] == pytest.approx(__import__("math").log(series[dt.date(2026, 6, 19)]))


def test_monthly_fit_cutoff_is_earliest_eligible_origin_minus_two():
    # Jan 1 / h10 has Dec 22 origin, earlier than Jan 1 / h1.
    origin = dt.date(2025, 12, 22)
    fit_cutoff = origin - dt.timedelta(days=2)
    assert fit_cutoff == dt.date(2025, 12, 20)
    # Forecast targets after the monthly cutoff must not become fit labels.
    series = _series(dt.date(2026, 1, 15))
    _, provenance = nd._fit_pooled(series, fit_cutoff)
    assert provenance["maxTrainingTargetDate"] <= "2025-12-20"
    assert set(provenance["horizonTrainingRows"]) == {str(h) for h in nd.HORIZONS}
    assert all(value >= nd.MIN_TRAINING_ROWS for value in provenance["horizonTrainingRows"].values())


def test_one_pooled_fit_is_reused_across_all_horizons_in_origin_month(monkeypatch):
    series = _series(dt.date(2026, 1, 15))
    calls=[]
    original=nd._fit_pooled
    def counted(s, cutoff):
        calls.append(cutoff)
        return original(s, cutoff)
    monkeypatch.setattr(nd,"_fit_pooled",counted)
    state={}
    a=nd._forecast(series,dt.date(2026,1,1),10,dt.date(2025,12,20),state)
    b=nd._forecast(series,dt.date(2026,1,2),9,dt.date(2025,12,20),state)
    assert len(calls)==1
    assert calls==[dt.date(2025,12,20)]
    assert a[1]["horizonTrainingRows"]==b[1]["horizonTrainingRows"]


def test_manifest_rejects_actuals_and_prediction_tampering():
    rows=[]
    for i in range(273):
        target=dt.date(2026,1,1)+dt.timedelta(days=i)
        for product in nd.PRODUCTS:
            for h in nd.HORIZONS:
                origin=target-dt.timedelta(days=h); cutoff=origin-dt.timedelta(days=2)
                rows.append({"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"cutoff":cutoff.isoformat(),
                    "predictions":{nd.METHOD:1},"lineage":{"candidate":{"featureCutoff":cutoff.isoformat()}}})
    doc = {"protocolId": nd.PROTOCOL_ID, "phase": "forecast_only", "targetValuesAttached": False,
           "codeSha256": nd._sha(__import__("pathlib").Path(nd.__file__).read_bytes()),
           "plannedForecastCount": 10920, "forecasts": rows}
    doc["manifestSha256"] = nd._hash({k:v for k,v in doc.items() if k != "manifestSha256"})
    doc["forecasts"][0]["actual"] = 12
    doc["manifestSha256"] = nd._hash({k:v for k,v in doc.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="target leaked"):
        nd.verify_manifest(doc)
    doc["forecasts"][0].pop("actual",None)
    doc["forecasts"][0]["predictions"][nd.METHOD]=2
    with pytest.raises(ValueError, match="manifest content hash"):
        nd.verify_manifest(doc)


def test_hac_constant_variance_is_unavailable():
    diffs = {dt.date(2026, 1, 1) + dt.timedelta(days=i): 2.0 for i in range(40)}
    result = nd._dm(diffs, 9)
    assert result["status"] == "inference_unavailable_nonpositive_variance"
    assert result["pValue"] == 1.0


def test_hac_repeated_decimal_difference_has_zero_variance():
    diffs={dt.date(2026,1,1)+dt.timedelta(days=i):0.1 for i in range(40)}
    result=nd._dm(diffs,28)
    assert result["status"]=="inference_unavailable_nonpositive_variance"
    assert result["pValue"]==1.0


def test_build_cli_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.inputs,"_load_frozen_manifest",lambda kind,path: {"kind":kind})
    monkeypatch.setattr(nd,"build_manifest",lambda development,fresh,bundles:{"result":"new"})
    output=tmp_path/"manifest.json"
    output.write_text('{"result":"old"}\n',encoding="utf-8")
    argv=["build","--archive",str(tmp_path/"archive.zip"),"--coverage-report",str(tmp_path/"coverage.json"),
          "--development-manifest",str(tmp_path/"dev.json"),"--fresh-manifest",str(tmp_path/"fresh.json"),"--output",str(output)]
    with pytest.raises(FileExistsError): nd.main(argv)
    assert json.loads(output.read_text(encoding="utf-8"))=={"result":"old"}
