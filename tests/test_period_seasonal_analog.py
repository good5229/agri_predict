import datetime as dt
import json
import math
import sys

import numpy as np
import pytest

from agri_data import period_proxy_experiment as proxy
from agri_data import period_seasonal_analog as m
from agri_data import period_seasonal_analog_score as score


def _target(start):
    start=dt.date.fromisoformat(start) if isinstance(start,str) else start
    return proxy.dekad_bounds(start)

def _candidate(target,origin,ratio,features=None,crop="cabbage",horizon=2):
    lo,hi=target
    return {"crop":crop,"horizon":horizon,"targetStart":lo.isoformat(),"targetEnd":hi.isoformat(),
        "targetKey":m.period_key(target),"origin":origin.isoformat(),"ratio":float(ratio),"referencePrice":100.,
        "features":features or {f:0. for f in m.FEATURES}}

def _raw_series(start,end):
    days=proxy.dekads(start,end);series={};obs=[]
    for i,(lo,hi) in enumerate(days):
        base=100.+i*.07+8*math.sin(i/9)
        for offset in (1,4,8):
            day=min(lo+dt.timedelta(days=offset),hi)
            price=base*(1+.01*math.sin(offset+i));volume=18.+i%5-offset
            series[day]=(price,volume)
            obs.append({"date":day.isoformat(),"actual":str(price),"volumeKg":str(volume)})
    return series,{"products":[{"id":c,"observations":obs} for c in proxy.CROPS]}


def test_calendar_season_wrap_and_inverse_ratio_median_ties():
    assert m.cyclic_season_distance((dt.date(2024,12,21),dt.date(2024,12,31)),
                                    (dt.date(2025,1,1),dt.date(2025,1,10)))==1
    assert m.weighted_ratio_median([1.,2.,4.])==1.
    # Ratios tie only by value; ascending selection is deterministic.
    assert m.weighted_ratio_median([2.,1.,2.,1.])==1.

def test_feature_windows_signed_volumes_and_no_skipping_invalid_dekad():
    dates={}
    # Latest completed period 2020-12-21..31 is valid; predecessor has only two days.
    for day in (21,24,28,31):dates[dt.date(2020,12,day)]=(100.+day,12.)
    for day in (11,18):dates[dt.date(2020,12,day)]=(90.+day,10.)
    for day in (1,4,8,10):dates[dt.date(2020,12,day)]=(80.+day,8.)
    # Inclusive 30-day price support with exactly nine observations; signed quantity retained.
    for day in range(2,11):dates[dt.date(2020,11,day)]=(75.+day,20.)
    targets=proxy._target_periods(dates)
    f=m.origin_features(dates,targets,dt.date(2021,1,2))
    assert f["inputEnd"]=="2020-12-31"
    assert f["recentCompletedPeriod"]=="2020-12-21|2020-12-31"
    assert f["previousCompletedPeriod"]=="2020-12-11|2020-12-20"
    assert f["previousPeriodPrice"] is None
    assert f["features"]["trend_1"] is not None
    assert f["features"]["trend_2"] is None
    assert f["vol30ObservedDays"]>=9 and f["features"]["vol_30"]==pytest.approx(
        np.std([math.log(p) for d,(p,_) in dates.items() if dt.date(2020,12,2)<=d<=dt.date(2020,12,31)],ddof=0))

def test_volume_ratio_signed_values_and_nonpositive_means_become_missing():
    dates={dt.date(2020,1,1)+dt.timedelta(days=i):(100.+i,(-2. if i<10 else 10.)) for i in range(30)}
    targets=proxy._target_periods(dates)
    f=m.origin_features(dates,targets,dt.date(2020,2,1))
    # input_end=2020-01-30; use inclusive recent 10 and preceding 20 windows.
    recent=[v for d,(_,v) in dates.items() if dt.date(2020,1,21)<=d<=dt.date(2020,1,30)]
    prior=[v for d,(_,v) in dates.items() if dt.date(2020,1,1)<=d<=dt.date(2020,1,20)]
    assert f["features"]["volume_ratio"]==pytest.approx(math.log(np.mean(recent)/np.mean(prior)))
    for d in list(dates):
        if d>=dt.date(2020,1,21):dates[d]=(dates[d][0],-1.)
    targets=proxy._target_periods(dates)
    assert m.origin_features(dates,targets,dt.date(2020,2,1))["features"]["volume_ratio"] is None

def test_q1_uses_pool_only_iqr_and_zero_iqr_coordinates_have_zero_weight():
    target=_target("2025-01-01");origin=dt.date(2024,12,12)
    candidates=[]
    for i in range(8):
        f={"trend_1":float(i),"trend_2":2.,"vol_30":3.,"volume_ratio":4.}
        candidates.append(_candidate(_target(dt.date(2020+i,1,1)),dt.date(2020+i,1,2),1.+i/100,f))
    # Force circular season-compatible fixtures independent of each year date.
    for c in candidates:c["targetStart"]="2024-01-01";c["targetEnd"]="2024-01-10";c["targetKey"]=c["targetStart"]+"|"+c["targetEnd"]
    candidates=[{**c,"targetStart":f"{2014+i}-01-01","targetEnd":f"{2014+i}-01-10",
       "targetKey":f"{2014+i}-01-01|{2014+i}-01-10"} for i,c in enumerate(candidates)]
    current={"features":{"trend_1":0.,"trend_2":99.,"vol_30":99.,"volume_ratio":99.},"referencePrice":100.}
    selected=m.select_analogs(current,candidates,"cabbage",2,target)
    assert selected["q1ValidCandidates"]==8 and len(selected["q1Selected"])==8
    assert selected["zeroIQRMask"]=={"trend_1":False,"trend_2":True,"vol_30":True,"volume_ratio":True}
    iqr=np.quantile(np.arange(8),.75,method="linear")-np.quantile(np.arange(8),.25,method="linear")
    assert all(x["_distance"]==pytest.approx(abs(x["features"]["trend_1"])/iqr)
               for x in selected["q1Selected"])

def test_fallbacks_are_support_only_and_preserve_missing_baseline():
    target=_target("2025-01-01");origin=dt.date(2024,12,12)
    f={x:0. for x in m.FEATURES};current={"features":f,"referencePrice":100.}
    seven=[_candidate(_target(dt.date(2014+i,1,1)),dt.date(2014+i,1,2),1.1+i*.01) for i in range(7)]
    row,analogs,forecast,sel=m.analog_forecasts(current,seven,"cabbage",2,target,origin,None)
    assert forecast["forecasts"]["Q0"]["predicted"] is None
    assert forecast["forecasts"]["Q1"]["predicted"] is None
    assert sel["q1Effective"]=="B0" and sel["q1FallbackReason"]=="q1_and_q0_insufficient_analogs"
    eight=seven+[_candidate(_target(dt.date(2021,1,1)),dt.date(2021,1,2),1.12,{**f,"trend_2":None})]
    _,_,forecast,sel=m.analog_forecasts(current,eight,"cabbage",2,target,origin,123.)
    assert forecast["forecasts"]["Q0"]["predicted"] is not None
    assert forecast["forecasts"]["Q1"]["effectiveMethod"]=="Q0"
    assert forecast["forecasts"]["Q1"]["predicted"]==forecast["forecasts"]["Q0"]["predicted"]

def test_nonempty_subk_q1_pool_has_oracle_range_and_selection_metadata_is_origin_local():
    target=_target("2025-01-01");origin=dt.date(2024,12,12)
    pool=[_candidate(_target(dt.date(2014+i,1,1)),dt.date(2014+i,1,2),1.05+i*.01) for i in range(8)]
    for c in pool[3:]:c["features"]["trend_1"]=None
    for c in pool:c["_distance"]=999.;c["_seasonDistance"]=999
    current={"features":{"trend_1":0.,"trend_2":0.,"vol_30":0.,"volume_ratio":0.},"referencePrice":100.}
    feature,_,forecast,selected=m.analog_forecasts(current,pool,"cabbage",2,target,origin,100.)
    assert selected["q1ValidCandidates"]==3
    assert feature["oracleEligible"] is True
    assert forecast["forecasts"]["Q1"]["predicted"] is not None
    no_features={"features":{"trend_1":None,"trend_2":0.,"vol_30":0.,"volume_ratio":0.}}
    missing_selection=m.select_analogs(no_features,pool,"cabbage",2,target)
    assert missing_selection["q0Selected"] and all("_distance" not in c and "_seasonDistance" not in c for c in missing_selection["q0Selected"])
    assert all(c["_distance"]==999. for c in pool)  # catalog objects are untouched

def test_candidate_eligibility_maturity_dedup_and_future_mutation_invariance():
    series,raw=_raw_series(dt.date(2014,1,1),dt.date(2025,12,31));targets=proxy._target_periods(series)
    horizon=2;catalog=m._historical_catalog(series,targets,horizon)
    origin=dt.date(2024,1,2);target=_target("2024-01-11")
    current=m.origin_features(series,targets,origin)
    candidates=m.eligible_candidates(catalog,"cabbage",horizon,target,origin)
    before=m.select_analogs(current,[dict(c) for c in candidates],"cabbage",horizon,target)
    # Change all future prices AND quantities after input_end. No current feature,
    # eligible target, per-origin scaling, analog key, or prediction can move.
    changed={d:v for d,v in series.items()}
    for d,(price,volume) in list(changed.items()):
        if d>proxy.input_cutoff(origin):changed[d]=(price*4.,volume+500.)
    targets2=proxy._target_periods(changed);catalog2=m._historical_catalog(changed,targets2,horizon)
    current2=m.origin_features(changed,targets2,origin)
    candidates2=m.eligible_candidates(catalog2,"cabbage",horizon,target,origin)
    after=m.select_analogs(current2,[dict(c) for c in candidates2],"cabbage",horizon,target)
    assert current["features"]==current2["features"]
    assert [(c["targetKey"],c["ratio"]) for c in before["q0Selected"]]==[(c["targetKey"],c["ratio"]) for c in after["q0Selected"]]
    assert [(c["targetKey"],c["ratio"]) for c in before["q1Selected"]]==[(c["targetKey"],c["ratio"]) for c in after["q1Selected"]]
    assert before["scale"]==after["scale"] and before["q1Ratio"]==after["q1Ratio"]
    assert all(proxy.label_matured((dt.date.fromisoformat(c["targetStart"]),dt.date.fromisoformat(c["targetEnd"])),origin)
               and dt.date.fromisoformat(c["origin"])<origin for c in candidates)

def test_forecast_artifact_is_label_free_and_exact_format_path(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,"pyarrow",None)
    rows=[{"targetKey":"a","forecasts":{"Q1":{"predicted":1.}}}]
    meta=m.write_rows(tmp_path/"forecasts_test.parquet",rows)
    assert meta["format"]==m.FORMAT_JSONL
    # A stale file at the conventional Parquet path must never override checkpoint metadata.
    (tmp_path/"forecasts_test.parquet").write_text("corrupt stale parquet")
    assert m.read_artifact(meta)==rows
    with pytest.raises(ValueError,match="realized labels"):
        m.write_rows(tmp_path/"x.parquet",[{"targetObserved":True}])

def test_checkpoint_identity_stale_preflight_refuses_resume(tmp_path,monkeypatch):
    stable={"runId":m.RUN_ID,"synthetic":True}
    monkeypatch.setattr(m,"run_identity",lambda:stable)
    cp=m.save_checkpoint("preflight",{"runId":m.RUN_ID,"stage":"preflight","complete":True},tmp_path)
    assert m.load_checkpoint("preflight",tmp_path)["runIdentity"]==stable
    monkeypatch.setattr(m,"run_identity",lambda:{**stable,"synthetic":False})
    with pytest.raises(ValueError,match="stale code/protocol/source/dependency"):
        m.load_checkpoint("preflight",tmp_path)

def test_scorer_synthetic_full_calendar_and_no_refit(tmp_path,monkeypatch):
    start,end=dt.date(2024,1,1),dt.date(2026,9,30);periods=proxy.dekads(start,end)
    raw={"products":[]};actuals={}
    for ci,crop in enumerate(proxy.CROPS):
        obs=[]
        for i,(lo,hi) in enumerate(periods):
            value=100.+ci*25+(i%11)*1.7;actuals[(crop,m.period_key((lo,hi)))]=value
            for j,day in enumerate((lo,lo+dt.timedelta(days=2),lo+dt.timedelta(days=5))):
                obs.append({"date":day.isoformat(),"actual":str(value+j*.1),"volumeKg":str(15+j)})
        raw["products"].append({"id":crop,"observations":obs})
    monkeypatch.setattr(score.proxy,"validate_inputs",lambda:({},raw))
    stable={"runId":m.RUN_ID,"synthetic":True,"dependencies":{"synthetic":"test"}};monkeypatch.setattr(m,"run_identity",lambda:stable)
    m.save_checkpoint("preflight",{"runId":m.RUN_ID,"stage":"preflight","complete":True},tmp_path)
    m.save_checkpoint("replay",{"runId":m.RUN_ID,"stage":"replay","complete":True},tmp_path)
    m.save_checkpoint("develop",{"runId":m.RUN_ID,"stage":"develop","complete":True},tmp_path)
    features=[];rows=[]
    for ci,crop in enumerate(proxy.CROPS):
      for h in proxy.HORIZONS:
       schedules=proxy.planned_origins(start,end,h)
       for i,s in enumerate(schedules):
        lo,hi=s["target"];key=m.period_key((lo,hi));actual=actuals[(crop,key)]
        unavailable=i<(1 if h==2 else 2)
        ratios={"P":1.08,"B0":1.05,"Q0":1.04,"Q1":1.02}
        fc={k:{"predicted":None if unavailable and k in ("P","B0") else actual*v,
               "effectiveMethod":k,"fallbackReason":None} for k,v in ratios.items()}
        rows.append({"crop":crop,"horizon":h,"targetStart":lo.isoformat(),"targetEnd":hi.isoformat(),
          "targetKey":key,"origin":s["origin"].isoformat(),"inputCutoff":s["inputCutoff"].isoformat(),
          "policyUnavailable":unavailable,"forecasts":fc})
        features.append({"crop":crop,"horizon":h,"targetStart":lo.isoformat(),"targetEnd":hi.isoformat(),
          "referencePrice":actual*.98,"features":{"trend_1":0.,"volume_ratio":0.},"oraclePoolMinRatio":.9,
          "oraclePoolMaxRatio":1.2,"oracleCandidateCount":12,"oracleEligible":True})
    forecast_meta=m.write_rows(tmp_path/"forecasts_test.parquet",rows)
    feature_meta=m.write_rows(tmp_path/"origin_features_test.parquet",features)
    m.save_checkpoint("forecast",{"runId":m.RUN_ID,"stage":"forecast","complete":True,
      "forecastArtifact":forecast_meta,"originFeaturesTest":feature_meta},tmp_path)
    bootstrap_inputs=[];original_bootstrap=score.stats.bootstrap
    def capture_bootstrap(rows,comparison,length,reps=2000):
        bootstrap_inputs.append((len(rows),min(r["targetStart"] for r in rows),length))
        return original_bootstrap(rows,comparison,length,reps)
    monkeypatch.setattr(score.stats,"bootstrap",capture_bootstrap)
    score.score_saved(tmp_path)
    report=json.loads((tmp_path/"score_report.json").read_text())
    assert report["support"]["plannedRows"]==792
    assert report["support"]["policyUnavailableRows"]==12
    assert report["support"]["pairedPAndB0"]==780
    assert report["support"]["commonComparisonRows"]==780
    assert len(bootstrap_inputs)==6 and all(n==792 and start=="2024-01-01" for n,start,_ in bootstrap_inputs)
    assert all(report["metricsByCell"][c][str(h)]["Q1"]["n"]==(98 if h==2 else 97)
               for c in proxy.CROPS for h in proxy.HORIZONS)
    assert report["hypothesisCounts"]=={"mainPerLag":32,"mechanismPerLag":16}
    assert len(report["main32"])==32 and len(report["mechanism16"])==16
    assert set(report["jointTargetDekadBlockBootstrap"]["Q1-P"])=={"6","12"}
    assert (tmp_path/"result.md").is_file()
    assert m.load_checkpoint("score",tmp_path)["complete"]

