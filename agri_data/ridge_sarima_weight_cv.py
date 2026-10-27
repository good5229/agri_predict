"""Three-phase historical SARIMA component forecasts and Ridge weight selection."""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, uuid
from pathlib import Path
from typing import Any
from . import dashboard, ten_strategy_benchmark as ten, ridge_sarima_blend as fixed
from . import multiscale_residual as stats

ROOT=Path(__file__).resolve().parents[1]
PRICES=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
DIRECT=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
TEN_MANIFEST=ROOT/"outputs/experiments/ten-strategy-benchmark-20261008/forecast-manifest.json"
TEN_REPORT=ROOT/"outputs/experiments/ten-strategy-benchmark-20261008/report.json"
FIXED_MANIFEST=ROOT/"outputs/experiments/ridge-sarima-blend-20261008/forecast-manifest.json"
FIXED_REPORT=ROOT/"outputs/experiments/ridge-sarima-blend-20261008/report.json"
PRICES_SHA="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
TEN_MANIFEST_SHA="30cc2c86e23027902667d554f8cddcec630172a1019a47b838b954b528adbeea"
TEN_REPORT_SHA="a63726a72fa9671556926ee87833b416b60f78bcf02247c0b3a813d3b85c2a41"
FIXED_MANIFEST_SHA="e73cd6b3c47b163000d020dbf60373af67d4edb79309af3a91b349167d945d40"
FIXED_REPORT_SHA="423c70ce3c34dc4eaf80c3b3fb3367fa6eedc73745d2098bf66c8f62c1a5a822"
PRODUCTS=tuple(dashboard.PRODUCTS);HORIZONS=tuple(range(1,11));BANDWIDTHS=(9,28)
HIST_START=dt.date(2024,9,25);HIST_END=dt.date(2025,8,31);FIRST_HIST_ORIGIN=dt.date(2024,9,15)
CUTOFF=FIRST_HIST_ORIGIN-dt.timedelta(days=1);TRAIN_DAYS=730;MIN_CELL_PAIRS=30
TEST_START=dt.date(2025,9,25);TEST_END=dt.date(2026,9,24);TEST_COUNT=12_508
WEIGHTS=(0.,.25,.5,.75,1.)
PROTOCOL="ridge-sarima-historical-weight-cv-2026-10-08"
METHODS=("candidate","fixed50","direct","sarima","persistence")

def _sha(b:bytes)->str:return stats.sha(b)
def _canonical(x):return stats.canonical(x)
def _code_sha():return _sha(Path(__file__).read_bytes())
def _positive(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0
def _key(row,*,period=None):
    p=row.get("product");h=row.get("horizon")
    if p not in PRODUCTS or isinstance(h,bool) or not isinstance(h,int) or h not in HORIZONS:raise ValueError("forecast_key_invalid")
    target=dt.date.fromisoformat(row["target"]);origin=dt.date.fromisoformat(row["origin"]) if row.get("origin") else target-dt.timedelta(days=h)
    if target-origin!=dt.timedelta(days=h):raise ValueError("origin_horizon_mismatch")
    if period and not period[0]<=target<=period[1]:raise ValueError("target_outside_frozen_period")
    return p,origin,target,h

def _direct_rows(raw,period):
    doc=json.loads(raw);rows=doc.get("pairedRows")
    if not isinstance(rows,list):raise ValueError("direct_paired_rows_missing")
    result={}
    for row in rows:
        target=dt.date.fromisoformat(row["target"])
        if not period[0]<=target<=period[1]:continue
        k=(row.get("product"),target-dt.timedelta(days=int(row["horizon"])),target,int(row["horizon"]))
        if k in result:raise ValueError("direct_duplicate_key")
        if k[0] not in PRODUCTS or k[3] not in HORIZONS:raise ValueError("direct_key_invalid")
        try:direct=float(row["direct"])
        except (KeyError,TypeError,ValueError):direct=None
        result[k]={"direct":direct,"raw":row}
    return result

def _historical_keys(direct_raw):
    rows=_direct_rows(direct_raw,(HIST_START,HIST_END))
    if not rows:raise ValueError("historical_direct_rows_empty")
    return rows

def _history_rows(prices_raw,direct_raw):
    if _sha(prices_raw)!=PRICES_SHA or _sha(direct_raw)!=DIRECT_SHA:raise ValueError("historical_input_sha_mismatch")
    price_doc=json.loads(prices_raw);series=ten._price_series(price_doc);direct=_historical_keys(direct_raw)
    train_start=CUTOFF-dt.timedelta(days=TRAIN_DAYS-1)
    audits={};fits={};failed={};
    for product in PRODUCTS:
        selection=ten._choose_difference(series[product],"sarima",train_start,CUTOFF)
        d=selection.get("selectedD");fit=None;fit_error=None
        if d is not None:
            try:fit=ten._fit_calendar(series[product],train_start,CUTOFF,"sarima",d)
            except Exception as exc:fit_error=type(exc).__name__+":"+str(exc)
        audits[product]={"selection":selection,"fitStatus":"available" if fit is not None else "unavailable",
            "fitError":fit_error,"selectedD":d,"trainStart":train_start.isoformat(),"trainEnd":CUTOFF.isoformat()}
        fits[product]=fit;failed[product]=None
    states={p:CUTOFF for p in PRODUCTS};out=[]
    for k,source in sorted(direct.items(),key=lambda item:(item[0][0],item[0][1],item[0][3])):
        product,origin,target,h=k;ps=series[product]
        available=[day for day in ps if day<=origin]
        source_day=max(available) if available else None
        age=(origin-source_day).days if source_day else None
        prediction=None;status="unavailable";reason=None
        fit=fits[product]
        if source_day is None:reason="no_source_observation"
        elif age>ten.MAX_SOURCE_AGE:reason="latest_source_older_than_14_days"
        elif fit is None:reason=audits[product]["fitError"] or "sarima_fit_unavailable"
        else:
            while states[product]<origin and failed[product] is None:
                day=states[product]+dt.timedelta(days=1)
                try:fit=fit.append(ten._calendar_values(ps,day,day),refit=False)
                except Exception as exc:failed[product]="state_update_failed:"+type(exc).__name__
                states[product]=day
            fits[product]=fit
            if failed[product]:reason=failed[product]
            else:
                try:
                    means=fit.get_forecast(steps=h).predicted_mean
                    logp=float(means.iloc[-1] if hasattr(means,"iloc") else means[-1]);value=math.exp(logp)
                    if not _positive(value):raise ValueError("nonpositive_or_nonfinite_forecast")
                    prediction=value;status="available"
                except Exception as exc:reason="forecast_failed:"+type(exc).__name__
        direct_value=source["direct"]
        if not _positive(direct_value):direct_value=None
        out.append({"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,
            "direct":direct_value,"sarima":prediction,"status":status,"reason":reason,
            "sourceDate":source_day.isoformat() if source_day else None,"sourceAgeDays":age})
    manifest={"schemaVersion":1,"protocol":PROTOCOL,"phase":"historical_components_only","targetValuesAttached":False,
        "inputSha256":{"prices":PRICES_SHA,"direct":DIRECT_SHA},"codeSha256":_code_sha(),
        "evaluation":{"start":HIST_START.isoformat(),"end":HIST_END.isoformat(),"firstOrigin":FIRST_HIST_ORIGIN.isoformat(),
            "cutoff":CUTOFF.isoformat(),"plannedRows":len(out),"products":list(PRODUCTS),"horizons":list(HORIZONS)},
        "sarimaProtocol":{"trainingCalendarDays":TRAIN_DAYS,"validationCalendarDays":60,"order":"(1,d,1)",
            "seasonalOrder":[1,0,0,7],"maxIter":ten.MAXITER,"statsmodelsVersion":"0.14.0",
            "missingDays":"NaN; append each calendar day; no interpolation","sourceAgeDaysMaximum":ten.MAX_SOURCE_AGE},
        "audits":audits,"coverage":{"planned":len(out),"sarimaAvailable":sum(r["sarima"] is not None for r in out),
            "directAvailable":sum(r["direct"] is not None for r in out)},"forecasts":out}
    manifest["manifestSha256"]=_canonical(manifest);return manifest

def build_historical(prices_raw,direct_raw):return _history_rows(prices_raw,direct_raw)

def _verify_historical(raw,prices,direct):
    m=json.loads(raw)
    if m.get("phase")!="historical_components_only" or m.get("targetValuesAttached") is not False:raise ValueError("historical_manifest_phase_invalid")
    saved=m.get("manifestSha256");clean=dict(m);clean.pop("manifestSha256",None)
    if saved!=_canonical(clean) or _sha(raw)!=_sha((json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()):raise ValueError("historical_manifest_tampered")
    if m!=build_historical(prices,direct):raise ValueError("historical_manifest_reconstruction_mismatch")
    keyed={}
    for row in m["forecasts"]:
        k=_key(row,period=(HIST_START,HIST_END))
        if k in keyed:raise ValueError("historical_duplicate_key")
        if "actual" in row or "targetValue" in row:raise ValueError("historical_answer_leakage")
        keyed[k]=row
    return m,keyed

def _cell_metrics(rows,predfield):
    cells={}
    for product in PRODUCTS:
        for h in HORIZONS:
            group=[r for r in rows if r["product"]==product and r["horizon"]==h]
            n=len(group)
            cells[(product,h)]={"n":n,"mae":sum(abs(r["actual"]-r[predfield]) for r in group)/n if n else None,
                "mape":100*sum(abs(r["actual"]-r[predfield])/r["actual"] for r in group)/n if n else None}
    return cells

def _weight_scores(common):
    all_scores=[]
    for w in WEIGHTS:
        scored=[]
        for row in common:
            item=dict(row);item["candidate"]=w*row["direct"]+(1-w)*row["sarima"];scored.append(item)
        cells=_cell_metrics(scored,"candidate")
        if any(c["n"]<MIN_CELL_PAIRS for c in cells.values()):raise ValueError("selection_cell_below_30_common_pairs")
        all_scores.append({"weight":w,"macro50CellMape":sum(c["mape"] for c in cells.values())/len(cells),
            "macro50CellMae":sum(c["mae"] for c in cells.values())/len(cells),"commonPairs":len(scored),
            "pairsByCell":{f"{p}|{h}":c["n"] for (p,h),c in cells.items()}})
    selected=min(all_scores,key=lambda x:(x["macro50CellMape"],x["macro50CellMae"],abs(x["weight"]-.5),x["weight"]))
    return selected["weight"],all_scores

def _source_manifests(ten_raw,fixed_raw):
    ten_doc=json.loads(ten_raw);fixed_doc=json.loads(fixed_raw)
    ten_rows=fixed._check_source_manifest(ten_doc,ten_raw)
    fixed_rows=fixed._verify_forecast(fixed_doc,fixed_raw)
    if set(ten_rows)!=set(fixed_rows) or len(ten_rows)!=TEST_COUNT:raise ValueError("test_source_keyset_invalid")
    fixed._validate_source_lineage(fixed_rows,ten_rows)
    return ten_doc,ten_rows,fixed_doc,fixed_rows

def build_selection(historical_raw,prices_raw,direct_raw,ten_raw,fixed_raw):
    if _sha(ten_raw)!=TEN_MANIFEST_SHA or _sha(fixed_raw)!=FIXED_MANIFEST_SHA:raise ValueError("test_source_manifest_sha_mismatch")
    hist,hist_rows=_verify_historical(historical_raw,prices_raw,direct_raw)
    prices=ten._price_series(json.loads(prices_raw));direct=_historical_keys(direct_raw)
    common=[];by_cell={}
    for k,row in hist_rows.items():
        if row["status"]!="available" or row["sarima"] is None or row["direct"] is None:continue
        source=direct[k]["raw"]
        try:actual=float(source["actual"])
        except (KeyError,TypeError,ValueError):continue
        actual_price=prices[k[0]].get(k[2])
        if not _positive(actual) or actual_price is None or not math.isclose(actual,actual_price,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("selection_actual_mismatch_against_prices")
        item={"product":k[0],"horizon":k[3],"target":k[2].isoformat(),"actual":actual,
              "direct":row["direct"],"sarima":row["sarima"]};common.append(item)
    cells=_cell_metrics(common,"sarima")
    if any(c["n"]<MIN_CELL_PAIRS for c in cells.values()):raise ValueError("selection_cell_below_30_common_pairs")
    weight,scores=_weight_scores(common)
    _,ten_rows,_,fixed_rows=_source_manifests(ten_raw,fixed_raw)
    test_direct=_direct_rows(direct_raw,(TEST_START,TEST_END))
    if set(test_direct)!=set(ten_rows):raise ValueError("test_direct_keyset_mismatch")
    forecasts=[]
    for k in sorted(ten_rows):
        source=ten_rows[k];fixed_row=fixed_rows[k];sp=source["predictions"];fp=fixed_row["predictions"]
        direct_value=float(sp["direct_ridge"]);persist=float(sp["naive"])
        report_direct=test_direct[k]["direct"]
        if report_direct is None or not math.isclose(direct_value,report_direct,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_direct_report_lineage_mismatch")
        try:report_persistence=float(test_direct[k]["raw"]["persistence"])
        except (KeyError,TypeError,ValueError):raise ValueError("test_persistence_report_invalid")
        if not math.isclose(persist,report_persistence,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_persistence_report_lineage_mismatch")
        status=source.get("methodStatus",{}).get("calendar_sarima_111_100_7",{})
        fallback=bool(status.get("fallback",source.get("fallback",{}).get("calendar_sarima_111_100_7",False)))
        native=source.get("nativePredictions",{}).get("calendar_sarima_111_100_7")
        sarima_value=None if fallback or native is None else float(native)
        if not math.isclose(direct_value,float(fp["direct_ridge"]),rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_direct_lineage_mismatch")
        if not math.isclose(persist,float(fp["persistence"]),rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_persistence_lineage_mismatch")
        if fallback:
            if not math.isclose(float(fp["blend"]),direct_value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("fixed_blend_fallback_mismatch")
        elif sarima_value is None or not math.isclose(float(fp["blend"]),.5*direct_value+.5*sarima_value,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("fixed_blend_component_mismatch")
        candidate=weight*direct_value+(1-weight)*sarima_value if sarima_value is not None else (direct_value if weight==1 else None)
        forecasts.append({"product":k[0],"origin":k[1].isoformat(),"target":k[2].isoformat(),"horizon":k[3],
            "predictions":{"candidate":candidate,"fixed50":float(fp["blend"]),"direct":direct_value,"sarima":sarima_value,"persistence":persist},
            "lineage":{"sarimaFallback":fallback,"sarimaFallbackReason":status.get("fallbackReason"),"weight":weight,
                "candidateStatus":"available" if candidate is not None else "held_sarima_unavailable"}})
    selection={"period":{"start":HIST_START.isoformat(),"end":HIST_END.isoformat()},"commonPairs":len(common),
        "pairsByCell":{f"{p}|{h}":c["n"] for (p,h),c in cells.items()},"weightScores":scores,"selectedWeight":weight}
    result={"schemaVersion":1,"protocol":PROTOCOL,"phase":"test_forecast_only","targetValuesAttached":False,
        "inputSha256":{"prices":PRICES_SHA,"direct":DIRECT_SHA,"tenManifest":TEN_MANIFEST_SHA,"fixed50Manifest":FIXED_MANIFEST_SHA},
        "codeSha256":_code_sha(),"historicalManifestSha256":hist["manifestSha256"],"selection":selection,
        "test":{"start":TEST_START.isoformat(),"end":TEST_END.isoformat(),"plannedPairs":TEST_COUNT},
        "coverage":{"testManifestPairs":len(forecasts),"candidateAvailable":sum(r["predictions"]["candidate"] is not None for r in forecasts),
            "sarimaNativeAvailable":sum(r["predictions"]["sarima"] is not None for r in forecasts)},"forecasts":forecasts}
    result["manifestSha256"]=_canonical(result);return result

def _verify_test(m,raw):
    if m.get("protocol")!=PROTOCOL or m.get("phase")!="test_forecast_only" or m.get("targetValuesAttached") is not False:raise ValueError("test_manifest_phase_invalid")
    if m.get("codeSha256")!=_code_sha():raise ValueError("test_manifest_code_hash_mismatch")
    clean=dict(m);saved=clean.pop("manifestSha256",None)
    if saved!=_canonical(clean) or _sha(raw)!=_sha((json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()):raise ValueError("test_manifest_hash_mismatch")
    rows={}
    for row in m.get("forecasts",[]):
        if "actual" in row or "targetValue" in row:raise ValueError("test_answer_leakage")
        k=_key(row,period=(TEST_START,TEST_END))
        if k in rows:raise ValueError("test_manifest_duplicate_key")
        rows[k]=row
    if len(rows)!=TEST_COUNT:raise ValueError("test_manifest_key_count_mismatch")
    return rows

def _check_report_metrics(report,scored,methods):
    refs={(row.get("product"),row.get("horizon")):row.get("metrics",{}) for row in report.get("cellMetrics",[])}
    if len(refs)!=len(PRODUCTS)*len(HORIZONS):raise ValueError("scoring_report_cells_invalid")
    for product in PRODUCTS:
        for horizon in HORIZONS:
            group=[row for row in scored if row["product"]==product and row["horizon"]==horizon]
            ref=refs.get((product,horizon),{})
            for score_field,report_field in methods:
                n=len(group)
                mae=sum(abs(r["actual"]-r[score_field]) for r in group)/n if n else None
                mape=100*sum(abs(r["actual"]-r[score_field])/r["actual"] for r in group)/n if n else None
                expected=ref.get(report_field,{})
                if expected.get("n")!=n:raise ValueError(f"scoring_report_pair_count_mismatch:{score_field}:{report_field}")
                if mae is None or not math.isclose(mae,float(expected["mae"]),rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f"scoring_report_mae_mismatch:{score_field}:{report_field}")
                if not math.isclose(mape,float(expected["mape"]),rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f"scoring_report_mape_mismatch:{score_field}:{report_field}")

def score(test_raw,historical_raw,prices_raw,direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw):
    hashes={"prices":_sha(prices_raw),"direct":_sha(direct_raw),"tenManifest":_sha(ten_raw),"tenReport":_sha(ten_report_raw),
            "fixed50Manifest":_sha(fixed_raw),"fixed50Report":_sha(fixed_report_raw)}
    expected={"prices":PRICES_SHA,"direct":DIRECT_SHA,"tenManifest":TEN_MANIFEST_SHA,"tenReport":TEN_REPORT_SHA,
              "fixed50Manifest":FIXED_MANIFEST_SHA,"fixed50Report":FIXED_REPORT_SHA}
    if hashes!=expected:raise ValueError("pinned_score_input_sha_mismatch")
    ten_report=json.loads(ten_report_raw);fixed_report=json.loads(fixed_report_raw);ten_doc=json.loads(ten_raw);fixed_doc=json.loads(fixed_raw)
    if ten_report.get("protocol")!=ten.PROTOCOL or ten_report.get("manifestSha256")!=ten_doc.get("manifestSha256"):raise ValueError("ten_report_lineage_mismatch")
    if ten_report.get("inputSha256",{}).get("direct")!=DIRECT_SHA or ten_report.get("inputSha256",{}).get("prices")!=PRICES_SHA:raise ValueError("ten_report_input_lineage_mismatch")
    if fixed_report.get("protocol")!=fixed.PROTOCOL or fixed_report.get("manifestSha256")!=fixed_doc.get("manifestSha256"):raise ValueError("fixed_report_lineage_mismatch")
    fixed_inputs=fixed_report.get("inputSha256",{})
    if fixed_inputs.get("directReport")!=DIRECT_SHA or fixed_inputs.get("tenStrategyManifest")!=TEN_MANIFEST_SHA or fixed_inputs.get("tenStrategyReport")!=TEN_REPORT_SHA:
        raise ValueError("fixed_report_input_lineage_mismatch")
    manifest=json.loads(test_raw);rows=_verify_test(manifest,test_raw)
    expected_manifest=build_selection(historical_raw,prices_raw,direct_raw,ten_raw,fixed_raw)
    if manifest!=expected_manifest:raise ValueError("test_manifest_reconstruction_mismatch")
    direct_rows=stats._pairs(json.loads(direct_raw));price_series=ten._price_series(json.loads(prices_raw))
    _,ten_rows,_,fixed_rows=_source_manifests(ten_raw,fixed_raw)
    source_scored=[];common=[]
    for k,source_row in ten_rows.items():
        direct_source=direct_rows.get(k)
        if direct_source is None:raise ValueError("test_direct_actual_key_missing")
        actual=float(direct_source["actual"]);price=price_series[k[0]].get(k[2])
        if not _positive(actual) or price is None or not math.isclose(actual,float(price),rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_actual_mismatch_against_frozen_sources")
        tenpred=source_row["predictions"];fixedpred=fixed_rows[k]["predictions"]
        direct_value=float(direct_source["direct"]);persistence=float(direct_source["persistence"])
        if not math.isclose(float(tenpred["direct_ridge"]),direct_value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_direct_prediction_mismatch")
        if not math.isclose(float(tenpred["naive"]),persistence,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_persistence_prediction_mismatch")
        source_scored.append({"product":k[0],"horizon":k[3],"actual":actual,
            "direct_ridge":direct_value,"calendar_sarima_111_100_7":float(tenpred["calendar_sarima_111_100_7"]),"persistence":persistence,
            "blend":float(fixedpred["blend"])})
        row=rows[k];pred=row["predictions"]
        if all(pred.get(m) is not None for m in METHODS):common.append({"product":k[0],"horizon":k[3],"target":k[2],"actual":actual,
            "candidate":float(pred["candidate"]),"fixed50":float(pred["fixed50"]),"direct":float(pred["direct"]),
            "sarima":float(pred["sarima"]),"persistence":float(pred["persistence"])})
    _check_report_metrics(ten_report,source_scored,(("direct_ridge","direct_ridge"),
        ("calendar_sarima_111_100_7","calendar_sarima_111_100_7"),("persistence","naive")))
    _check_report_metrics(fixed_report,source_scored,(("direct_ridge","direct_ridge"),
        ("calendar_sarima_111_100_7","calendar_sarima_111_100_7"),("persistence","persistence"),("blend","blend")))
    metrics={}
    for method in METHODS:
        cells={}
        for p in PRODUCTS:
            for h in HORIZONS:
                group=[r for r in common if r["product"]==p and r["horizon"]==h];n=len(group)
                cells[f"{p}|{h}"]={"n":n,"mae":sum(abs(r["actual"]-r[method]) for r in group)/n if n else None,
                    "mape":100*sum(abs(r["actual"]-r[method])/r["actual"] for r in group)/n if n else None}
        byp={}
        for p in PRODUCTS:
            group=[r for r in common if r["product"]==p];n=len(group)
            byp[p]={"n":n,"mae":sum(abs(r["actual"]-r[method]) for r in group)/n if n else None,
                "mape":100*sum(abs(r["actual"]-r[method])/r["actual"] for r in group)/n if n else None}
        metrics[method]={"cells":cells,"byProduct":byp,"macro50CellMae":sum(x["mae"] for x in cells.values())/50 if len(cells)==50 and all(x["mae"] is not None for x in cells.values()) else None,
            "macro50CellMape":sum(x["mape"] for x in cells.values())/50 if len(cells)==50 and all(x["mape"] is not None for x in cells.values()) else None,"coverage":len(common)}
    tests=[]
    for comparator in ("fixed50","direct","sarima","persistence"):
        for p in PRODUCTS:
            for h in HORIZONS:
                group={ (r["product"],r["target"],r["horizon"]):{"actual":r["actual"],"predictions":{"candidate":r["candidate"],comparator:r[comparator]}}
                    for r in common if r["product"]==p and r["horizon"]==h}
                for loss in ("mae","mape"):
                    diffs=[(k[1],(abs(r["actual"]-r["predictions"]["candidate"])-abs(r["actual"]-r["predictions"][comparator])) if loss=="mae" else
                        (abs(r["actual"]-r["predictions"]["candidate"])-abs(r["actual"]-r["predictions"][comparator]))/r["actual"]) for k,r in group.items()]
                    for bw in BANDWIDTHS:
                        dm=stats._dm(diffs,bw);tests.append({"comparator":comparator,"product":p,"horizon":h,"loss":loss,"bandwidth":bw,
                            "meanLossDifference":sum(v for _,v in diffs)/len(diffs) if diffs else None,**dm})
    for bw in BANDWIDTHS:stats._holm([t for t in tests if t["bandwidth"]==bw])
    grouped={}
    for item in tests:grouped.setdefault((item["comparator"],item["product"],item["horizon"]),[]).append(item)
    conclusions=[]
    for (comp,p,h),items in grouped.items():
        states=[x["conclusion"] for x in items]
        decision="improvement" if len(states)==4 and all(x=="improvement" for x in states) else "worsening" if len(states)==4 and all(x=="worsening" for x in states) else "inference_unavailable" if "inference_unavailable" in states else "inconclusive"
        conclusions.append({"comparator":comp,"product":p,"horizon":h,"conclusion":decision})
    return {"protocol":PROTOCOL,"manifestSha256":manifest["manifestSha256"],"selectedWeight":manifest["selection"]["selectedWeight"],
        "coverage":{"planned":TEST_COUNT,"manifestPairs":len(rows),"commonScoredPairs":len(common),
            "candidateAvailable":manifest["coverage"]["candidateAvailable"],"sarimaNativeAvailable":manifest["coverage"]["sarimaNativeAvailable"]},
        "metrics":metrics,"dmTests":tests,"cellConclusions":conclusions,"inputSha256":hashes,
        "interpretation":"Retrospective, repeatedly inspected revised-price data; nonsignificance is not equivalence or independent confirmation."}

def _immutable(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x",encoding="utf-8") as stream:stream.write(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        try:temp.unlink()
        except FileNotFoundError:pass

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="phase",required=True)
    a=sub.add_parser("historical");a.add_argument("--manifest",type=Path,required=True);a.add_argument("--prices",type=Path,default=PRICES);a.add_argument("--direct",type=Path,default=DIRECT)
    b=sub.add_parser("select");b.add_argument("--historical",type=Path,required=True);b.add_argument("--manifest",type=Path,required=True);b.add_argument("--prices",type=Path,default=PRICES);b.add_argument("--direct",type=Path,default=DIRECT);b.add_argument("--ten-manifest",type=Path,default=TEN_MANIFEST);b.add_argument("--fixed-manifest",type=Path,default=FIXED_MANIFEST)
    c=sub.add_parser("score");c.add_argument("--historical",type=Path,required=True);c.add_argument("--manifest",type=Path,required=True);c.add_argument("--report",type=Path,required=True);c.add_argument("--prices",type=Path,default=PRICES);c.add_argument("--direct",type=Path,default=DIRECT);c.add_argument("--ten-manifest",type=Path,default=TEN_MANIFEST);c.add_argument("--ten-report",type=Path,default=TEN_REPORT);c.add_argument("--fixed-manifest",type=Path,default=FIXED_MANIFEST);c.add_argument("--fixed-report",type=Path,default=FIXED_REPORT)
    args=parser.parse_args(argv);prices=args.prices.read_bytes();direct=args.direct.read_bytes()
    if args.phase=="historical":_immutable(args.manifest,build_historical(prices,direct))
    elif args.phase=="select":_immutable(args.manifest,build_selection(args.historical.read_bytes(),prices,direct,args.ten_manifest.read_bytes(),args.fixed_manifest.read_bytes()))
    else:_immutable(args.report,score(args.manifest.read_bytes(),args.historical.read_bytes(),prices,direct,args.ten_manifest.read_bytes(),args.ten_report.read_bytes(),args.fixed_manifest.read_bytes(),args.fixed_report.read_bytes()))
    return 0
if __name__=="__main__":raise SystemExit(main())
