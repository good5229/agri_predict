import datetime as dt
import json
from decimal import Decimal
import hashlib
import io
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import pytest

from agri_data import seoan_pepper_blend_eval as blend


def test_latest_source_requires_prior_day_and_allows_three_day_age():
    origin=dt.date(2025,1,10)
    medians={"2025-01-06":Decimal("1"),"2025-01-07":Decimal("2"),"2025-01-10":Decimal("999")}
    assert blend._latest_source(medians,origin)==(dt.date(2025,1,7),Decimal("2"))


def test_daily_audit_retains_null_and_negative_raw_values():
    medians,audit=blend._daily_source({"2025-01-01":[
        {"date":"2025-01-01","sourceRow":2,"priceKgRaw":"10","issues":[]},
        {"date":"2025-01-01","sourceRow":3,"priceKgRaw":"-2","issues":["negative"]},
        {"date":"2025-01-01","sourceRow":4,"priceKgRaw":"NULL","issues":["invalid"]}]})
    assert medians["2025-01-01"]==Decimal("10")
    assert audit["2025-01-01"]["excludedCount"]==2
    assert [x["priceKgRaw"] for x in audit["2025-01-01"]["sourceRows"]]==["10","-2","NULL"]


def test_scale_fit_uses_unique_a_h1_target_and_source_age_rule():
    direct={"pairedRows":[
        {"product":"pepper","horizon":1,"target":"2024-09-25","actual":"20"},
        {"product":"pepper","horizon":2,"target":"2024-09-25","actual":"999"},
        {"product":"pepper","horizon":1,"target":"2025-09-25","actual":"777"}]}
    audit=blend._fit_scale(direct,{"2024-09-22":Decimal("10")})
    assert audit["scale"]=="2"
    assert audit["includedCount"]==1
    assert audit["includedRows"][0]["sourceDate"]=="2024-09-22"


def _xlsx(rows):
    ns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root=ET.Element(f"{{{ns}}}worksheet");sheet=ET.SubElement(root,f"{{{ns}}}sheetData")
    for rno,row in enumerate(rows,1):
        r=ET.SubElement(sheet,f"{{{ns}}}row",{"r":str(rno)})
        for cno,value in enumerate(row):
            letters=chr(65+cno);cell=ET.SubElement(r,f"{{{ns}}}c",{"r":f"{letters}{rno}","t":"inlineStr"})
            inn=ET.SubElement(cell,f"{{{ns}}}is");ET.SubElement(inn,f"{{{ns}}}t").text=value
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,"w") as z:z.writestr("xl/worksheets/sheet1.xml",ET.tostring(root))
    return payload.getvalue()


def test_history_loader_checks_all_month_files_and_preserves_bad_raw(monkeypatch,tmp_path):
    source_dir=Path("source");out=tmp_path/source_dir;out.mkdir()
    months=[]
    for month in blend._month_plan():
        rid=month["requestId"];source_rows=[["일자","품목","단가(kg)"]]
        if rid=="2024-07":source_rows.append(["2024-07-15",blend.FROZEN_NAME,"-4"])
        raw= _xlsx(source_rows);raw_name=f"{rid}.raw";manifest_name=f"{rid}.manifest.json"
        (out/raw_name).write_bytes(raw)
        meta={"kind":"seoan_history_export","http_status":200,"truncated":False,"transport_error":None,
              "request":{"method":"POST","url":blend.seoan.POST_URL,"params":month["params"]},
              "raw_file":raw_name,"response_bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
        meta_bytes=json.dumps(meta).encode();(out/manifest_name).write_bytes(meta_bytes)
        columns=blend.seoan._columns(source_rows[0])
        count=1 if rid=="2024-07" else 0
        months.append({"requestId":rid,"state":"parsed" if count else "normal_empty","usable":True,
                       "rawFile":raw_name,"manifestFile":manifest_name,"sha256":hashlib.sha256(raw).hexdigest(),
                       "responseBytes":len(raw),"header":source_rows[0],"schema":sorted(columns),"rowCount":count,
                       "frozenNameCounts":{blend.FROZEN_NAME:{"rowCount":count}},
                       "attemptOutcomes":[{"manifestFile":manifest_name,"sha256":hashlib.sha256(raw).hexdigest()}]})
    summary={"complete":False,"allMonthsUsable":True,"allMonthsStructurallyReadable":True,"months":months}
    summary_bytes=json.dumps(summary).encode();(out/"summary.json").write_bytes(summary_bytes)
    monkeypatch.setattr(blend,"HISTORY_DIR",source_dir)
    monkeypatch.setattr(blend,"HISTORY_SUMMARY",source_dir/"summary.json")
    monkeypatch.setattr(blend,"HISTORY_SUMMARY_SHA256",hashlib.sha256(summary_bytes).hexdigest())
    _,daily,hashes=blend._load_history(tmp_path)
    assert len(hashes["months"])==24
    assert daily["2024-07-15"][0]["priceKgRaw"]=="-4"
    assert daily["2024-07-15"][0]["issues"]==["nonpositive_or_nonfinite_price_kg"]


def _write_forecast_manifest(tmp_path,phase,monkeypatch):
    doc={"phase":phase,"targetValuesAttached":False,"forecasts":[
        {"product":"pepper","horizon":1,"origin":"2025-09-24","target":"2025-09-25",
         "predictions":{"fixed50":50.}},
        {"product":"cabbage","horizon":1,"origin":"2025-09-24","target":"2025-09-25",
         "predictions":{"fixed50":70.}}]}
    doc["manifestSha256"]=blend._canonical_hash(doc)
    raw=json.dumps(doc,separators=(",",":"),sort_keys=True).encode()
    (tmp_path/"forecast.json").write_bytes(raw)
    monkeypatch.setattr(blend,"FORECAST_MANIFEST",Path("forecast.json"))
    monkeypatch.setattr(blend,"FORECAST_MANIFEST_SHA256",hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(blend,"FORECAST_CONTENT_SHA256",doc["manifestSha256"])


def test_load_forecasts_accepts_pinned_forecast_only_phase(tmp_path,monkeypatch):
    _write_forecast_manifest(tmp_path,"forecast_only",monkeypatch)
    rows,_=blend._load_forecasts(tmp_path)
    assert len(rows)==1
    assert rows[0]["fixed50"]==50.


def test_load_forecasts_rejects_altered_phase(tmp_path,monkeypatch):
    _write_forecast_manifest(tmp_path,"selected_test_forecast_only",monkeypatch)
    with pytest.raises(blend.BlendError,match="not answer-free"):
        blend._load_forecasts(tmp_path)


def test_answer_free_build_uses_fixed50_fallback_without_b_actual(monkeypatch):
    daily={"2024-09-22":[{"date":"2024-09-22","sourceRow":2,"product":blend.FROZEN_NAME,
                            "priceKgRaw":"10","issues":[]}],
           "2025-09-22":[{"date":"2025-09-22","sourceRow":3,"product":blend.FROZEN_NAME,
                            "priceKgRaw":"20","issues":[]}]}
    direct={"pairedRows":[{"product":"pepper","horizon":1,"target":"2024-09-25","actual":"20"},
                           {"product":"pepper","horizon":1,"target":"2025-09-25","actual":"DO NOT USE B ACTUAL"}]}
    forecasts=[
        {"product":"pepper","horizon":1,"origin":"2025-09-24","target":"2025-09-25","fixed50":50,
         "key":["pepper",1,"2025-09-24","2025-09-25"]},
        {"product":"pepper","horizon":1,"origin":"2026-02-28","target":"2026-03-01","fixed50":50,
         "key":["pepper",1,"2026-02-28","2026-03-01"]},
    ]
    monkeypatch.setattr(blend,"_load_history",lambda root:({"complete":False},daily,{"synthetic":True}))
    monkeypatch.setattr(blend,"_read_direct",lambda root:(direct,b"synthetic direct"))
    monkeypatch.setattr(blend,"_load_forecasts",lambda root:(forecasts,b'{"manifestSha256":"synthetic"}'))
    manifest=blend.build_manifest()
    assert blend.build_manifest()["manifestSha256"]==manifest["manifestSha256"]
    rows=manifest["testB"]["forecasts"]
    assert manifest["testB"]["targetValuesAttached"] is False
    assert manifest["trainingA"]["scaleFit"]["scale"]=="2"
    assert rows[0]["fallback"] is False and rows[0]["sourceDate"]=="2025-09-22"
    assert rows[1]["fallback"] is True and rows[1]["prediction"]==50
    assert all("actual" not in row for row in rows)
    assert "DO NOT USE B ACTUAL" not in json.dumps(manifest,ensure_ascii=False)


def test_prediction_manifest_is_immutable(tmp_path,monkeypatch):
    doc={"manifestSha256":"synthetic","testB":{"targetValuesAttached":False,"forecasts":[]}}
    monkeypatch.setattr(blend,"build_manifest",lambda root=blend.ROOT:doc)
    path=tmp_path/"manifest.json"
    assert blend.write_manifest(path)==doc
    with pytest.raises(FileExistsError):blend.write_manifest(path)


def test_score_reports_coverage_and_uninferable_small_sample(tmp_path,monkeypatch):
    target="2025-09-25"
    rows=[];actual_rows=[]
    for h in range(1,11):
        origin=(dt.date.fromisoformat(target)-dt.timedelta(days=h)).isoformat()
        key=["pepper",h,origin,target]
        rows.append({"key":key,"origin":origin,"target":target,"horizon":h,"fixed50":50.,
                     "prediction":50.,"fallback":True,"sourceDate":None,"sourcePriceKg":None})
        actual_rows.append({"product":"pepper","target":target,"horizon":h,"actual":60.})
    direct_bytes=json.dumps({"pairedRows":actual_rows},separators=(",",":"),sort_keys=True).encode()
    (tmp_path/"direct.json").write_bytes(direct_bytes)
    monkeypatch.setattr(blend,"DIRECT_REPORT",blend.Path("direct.json"))
    doc={"manifestSha256":"","trainingA":{"scaleFit":{"scale":"2"}},
         "sourceHashes":{"directReport":{"sha256":hashlib.sha256(direct_bytes).hexdigest()}},
         "testB":{"targetValuesAttached":False,"forecasts":rows}}
    doc["manifestSha256"]=blend._canonical_hash({k:v for k,v in doc.items() if k!="manifestSha256"})
    manifest_path=tmp_path/"answer-free.json";manifest_path.write_text(json.dumps(doc),encoding="utf-8")
    result=blend.score_manifest(manifest_path,tmp_path/"score.json",tmp_path)
    assert result["plannedPairs"]==10 and result["positiveActualCommonPairs"]==10
    assert result["coverageByHorizon"]["1"]["planned"]==1
    assert result["coverageByHorizon"]["1"]["positiveActual"]==1
    assert len(result["dmComparisons"])==20
    assert result["holmFamilySizePerBandwidth"]==20
    assert all(row["dm"]["9"]["status"]=="inference_unavailable_insufficient_pairs" for row in result["dmComparisons"])
    assert result["complete"] is False


def test_score_stops_if_actual_values_for_one_target_disagree(tmp_path,monkeypatch):
    target="2025-09-25";actual_rows=[];forecasts=[]
    for h in range(1,11):
        origin=(dt.date.fromisoformat(target)-dt.timedelta(days=h)).isoformat()
        forecasts.append({"key":["pepper",h,origin,target],"origin":origin,"target":target,"horizon":h,
                          "fixed50":50.,"prediction":50.,"fallback":True,"sourceDate":None,"sourcePriceKg":None})
        actual_rows.append({"product":"pepper","target":target,"horizon":h,"actual":61. if h==2 else 60.})
    direct_bytes=json.dumps({"pairedRows":actual_rows},separators=(",",":"),sort_keys=True).encode()
    (tmp_path/"direct.json").write_bytes(direct_bytes);monkeypatch.setattr(blend,"DIRECT_REPORT",Path("direct.json"))
    doc={"trainingA":{"scaleFit":{"scale":"2"}},"sourceHashes":{"directReport":{"sha256":hashlib.sha256(direct_bytes).hexdigest()}},
         "testB":{"targetValuesAttached":False,"forecasts":forecasts}}
    doc["manifestSha256"]=blend._canonical_hash(doc)
    path=tmp_path/"manifest.json";path.write_text(json.dumps(doc),encoding="utf-8")
    with pytest.raises(blend.BlendError,match="target-day actual prices disagree"):
        blend.score_manifest(path,tmp_path/"score.json",tmp_path)


@pytest.mark.parametrize("constant",[0.1,-0.1])
def test_dm_constant_float_differences_are_zero_spread(constant):
    start=dt.date(2025,1,1)
    diffs={start+dt.timedelta(days=i):constant for i in range(40)}
    for bandwidth in blend.BANDWIDTHS:
        result=blend._dm(diffs,bandwidth)
        assert result["status"]=="inference_unavailable_zero_spread"
        assert result["pValue"]==1.0
        assert result["statistic"] is None


def test_dm_near_constant_but_varying_differences_remain_estimable():
    start=dt.date(2025,1,1)
    diffs={start+dt.timedelta(days=i):0.1+((i%4)-1.5)*1e-12 for i in range(40)}
    for bandwidth in blend.BANDWIDTHS:
        result=blend._dm(diffs,bandwidth)
        assert result["status"]=="available"
        assert result["statistic"] is not None


def test_constant_apparent_improvement_cannot_pass_gate():
    actual={};forecasts=[];start=dt.date(2025,10,1)
    for i in range(40):
        target=(start+dt.timedelta(days=i)).isoformat()
        for h in range(1,11):
            origin=(dt.date.fromisoformat(target)-dt.timedelta(days=h)).isoformat()
            key=("pepper",h,origin,target)
            actual[key]=100.
            forecasts.append({"key":list(key),"origin":origin,"target":target,"horizon":h,
                              "fixed50":101.1,"prediction":100.,"fallback":False})
    result=blend._scores(actual,{"testB":{"forecasts":forecasts}})
    assert result["macroMapeImprovementPoints"]>0.5
    assert result["gatePassed"] is False
    assert result["bilateralImprovementHorizons"]==[]
    assert all(t["dm"]["9"]["status"]=="inference_unavailable_zero_spread" for t in result["dmComparisons"])
