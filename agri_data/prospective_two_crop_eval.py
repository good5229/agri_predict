"""Evaluate immutable onion and garlic prospective forecast issues against verified truth."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from . import dashboard, direct_horizon as dh, market_mix, modeling

ROOT=Path(__file__).resolve().parents[1]
KST=dt.timezone(dt.timedelta(hours=9))
ONION_HORIZONS={"onion":(1,2),"onion_extension":tuple(range(3,11))}
HORIZONS={**ONION_HORIZONS,"garlic":tuple(range(1,11))}
COMPARATORS={"onion":("directRidgePrediction","persistencePrediction"),"onion_extension":("directRidgePrediction","persistencePrediction"),"garlic":("ridgePrediction","persistencePrediction")}
FAMILY_SIZE={"onion":8,"onion_extension":32,"garlic":40}
HAC=(9,28); MIN_PAIRS=30; ALPHA=.05


def _sha(data): return hashlib.sha256(data).hexdigest()


def _finite_positive(value,label):
    if isinstance(value,bool): raise ValueError(f"{label} must be positive finite")
    try: x=float(value)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(f"{label} must be positive finite") from exc
    if not math.isfinite(x) or x<=0: raise ValueError(f"{label} must be positive finite")
    return x


def _date(value,label):
    if not isinstance(value,str): raise ValueError(f"{label} must be ISO date")
    try: return dt.date.fromisoformat(value)
    except ValueError as exc: raise ValueError(f"{label} must be ISO date") from exc


def _state_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def _onion_family_from_doc(doc):
    products=doc.get("products")
    if not isinstance(products,list) or len(products)!=1 or not isinstance(products[0],dict) or products[0].get("productId")!="onion":
        raise ValueError("onion artifact product schema is invalid")
    forecasts=products[0].get("forecasts")
    if not isinstance(forecasts,list): raise ValueError("forecast rows are invalid")
    hs=[]
    for row in forecasts:
        if not isinstance(row,dict): raise ValueError("forecast row must be an object")
        h=row.get("horizon")
        if type(h) is not int or h in hs: raise ValueError("invalid or duplicate onion horizon")
        hs.append(h)
    observed=set(hs)
    for family,expected in ONION_HORIZONS.items():
        if observed==set(expected) and len(hs)==len(expected): return family
    raise ValueError("onion forecast must contain exactly horizons 1-2 or exactly horizons 3-10")


def _verified_onion_state(doc,path,issue,issued,artifact_bytes_by_path,expected_horizons):
    family=_onion_family_from_doc(doc)
    if tuple(ONION_HORIZONS[family])!=tuple(expected_horizons):
        raise ValueError("onion model-state horizon family differs from forecast horizons")
    reference=doc.get("modelStateReference")
    model=doc.get("modelState")
    if reference is None:
        if not isinstance(model,dict): raise ValueError("onion model state is missing")
        if _sha(_state_bytes(model))!=doc.get("modelStateSha256"): raise ValueError("onion embedded model-state SHA-256 mismatch")
        fit_origin=_date(model.get("direct",{}).get("fitOrigin"),"onion model fitOrigin")
        first_issue=issue
        first_issued=issued
        if fit_origin!=issue: raise ValueError("onion first model-state fit origin mismatch")
    else:
        if model is not None or not isinstance(reference,dict): raise ValueError("onion model-state reference schema is invalid")
        ref_path=reference.get("path"); ref_sha=reference.get("sha256")
        if not isinstance(ref_path,str) or not ref_path or not isinstance(ref_sha,str) or len(ref_sha)!=64 or any(c not in "0123456789abcdef" for c in ref_sha):
            raise ValueError("onion model-state reference is malformed")
        ref=Path(ref_path)
        candidates=(ref.resolve(),(Path(path).parent/ref).resolve()) if not ref.is_absolute() else (ref.resolve(),)
        resolved=next((candidate for candidate in candidates if candidate in artifact_bytes_by_path),None)
        state_bytes=artifact_bytes_by_path.get(resolved) if resolved is not None else None
        if state_bytes is None: raise ValueError("referenced onion first-issue artifact was not supplied")
        if _sha(state_bytes)!=ref_sha: raise ValueError("onion model-state reference byte SHA-256 mismatch")
        try: first_doc=json.loads(state_bytes)
        except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("referenced onion first-issue artifact is invalid JSON") from exc
        if first_doc.get("schemaVersion")!=1 or not isinstance(first_doc.get("products"),list): raise ValueError("referenced artifact is not an onion forecast")
        first_family=_onion_family_from_doc(first_doc)
        expected_family=next((family for family,hs in ONION_HORIZONS.items() if tuple(hs)==tuple(expected_horizons)),None)
        if first_family!=expected_family: raise ValueError("onion model-state reference uses a different horizon family")
        first_issue=_date(first_doc.get("issueDateKst"),"referenced onion issueDateKst")
        try: first_issued=dt.datetime.fromisoformat(first_doc["issuedAtKst"])
        except (KeyError,TypeError,ValueError) as exc: raise ValueError("referenced onion issue timestamp is invalid") from exc
        if first_issued.tzinfo is None or first_issued.astimezone(KST).date()!=first_issue: raise ValueError("referenced onion issue timestamp is invalid")
        if first_issue>=issue or first_issue.strftime("%Y-%m")!=issue.strftime("%Y-%m"): raise ValueError("referenced onion model-state month mismatch")
        if first_issued.astimezone(KST)>=issued.astimezone(KST): raise ValueError("referenced onion state was not issued before this forecast")
        if first_doc.get("modelStateReference") is not None: raise ValueError("referenced onion artifact is not the first issue for the month")
        model=first_doc.get("modelState")
        if not isinstance(model,dict) or _sha(_state_bytes(model))!=first_doc.get("modelStateSha256"):
            raise ValueError("referenced onion model-state SHA-256 mismatch")
        if doc.get("modelStateSha256")!=first_doc.get("modelStateSha256"):
            raise ValueError("onion follow-on model-state SHA-256 differs from referenced first issue")
        fit_origin=_date(model.get("direct",{}).get("fitOrigin"),"referenced onion fitOrigin")
        if fit_origin!=first_issue: raise ValueError("referenced onion model fit origin mismatch")
    if first_issue.strftime("%Y-%m")!=issue.strftime("%Y-%m"): raise ValueError("onion model state month mismatch")
    if set(model)!={"direct","110001","110008"}: raise ValueError("onion model state is incomplete")
    for state_name,state_model in model.items():
        if not isinstance(state_model,dict) or state_model.get("month")!=issue.strftime("%Y-%m") or state_model.get("fitOrigin")!=first_issue.isoformat():
            raise ValueError(f"onion model-state metadata invalid: {state_name}")
        horizons=state_model.get("byHorizon")
        if not isinstance(horizons,dict) or set(horizons)!={str(h) for h in expected_horizons}: raise ValueError(f"onion model-state horizons invalid: {state_name}")
        for h,state in horizons.items():
            if (not isinstance(state,dict) or state.get("fitOrigin")!=first_issue.isoformat() or
                    any(not isinstance(state.get(key),int) or state[key]<0 for key in ("trainingTargetDays","trainingSampleCount"))):
                raise ValueError(f"onion model-state fit audit invalid: {state_name}/{h}")
            end=state.get("trainingTargetEnd")
            if end is not None and _date(end,"onion trainingTargetEnd")>=fit_origin: raise ValueError("onion training target is not strictly before fitOrigin")
            if state.get("available"):
                if end is None or state["trainingTargetDays"]<90: raise ValueError("available onion model state lacks sufficient training history")
                try:
                    mean,scale,coef=state["mean"],state["scale"],state["coef"]
                    if not (len(mean)==len(scale)==len(coef)==len(modeling.FEATURE_NAMES)): raise ValueError
                    values=[*mean,*scale,*coef,state["intercept"]]
                    if not all(math.isfinite(float(value)) for value in values) or any(float(value)<=0 for value in scale): raise ValueError
                except (KeyError,TypeError,ValueError,OverflowError) as exc:
                    raise ValueError(f"onion model-state parameters invalid: {state_name}/{h}") from exc
            elif not state.get("reason"):
                raise ValueError(f"unavailable onion model state lacks reason: {state_name}/{h}")
    return model


def _load_artifact(path, data, artifact_bytes_by_path=None):
    digest=_sha(data)
    try: doc=json.loads(data)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"invalid forecast JSON: {path}") from exc
    if not isinstance(doc,dict) or doc.get("schemaVersion")!=1: raise ValueError(f"unsupported forecast artifact: {path}")
    issue=_date(doc.get("issueDateKst"),"issueDateKst")
    try: issued=dt.datetime.fromisoformat(doc["issuedAtKst"])
    except (KeyError,TypeError,ValueError) as exc: raise ValueError("issuedAtKst must be an aware ISO timestamp") from exc
    if issued.tzinfo is None or issued.astimezone(KST).date()!=issue: raise ValueError("issuedAtKst date/timezone does not match issueDateKst")
    if doc.get("products") is not None:
        crop=_onion_family_from_doc(doc); forecasts=doc["products"][0].get("forecasts")
        price_hash=doc.get("pricesSha256")
        snapshot_cutoff=None
    else:
        crop="garlic"; forecasts=doc.get("forecasts")
        input_meta=doc.get("input",{}); price_hash=input_meta.get("pricesSha256")
        snapshot_cutoff=_date(input_meta.get("snapshotCutoff"),"snapshotCutoff")
        if snapshot_cutoff>=issue: raise ValueError("garlic snapshot cutoff must precede issue date")
        fit=doc.get("fit",{})
        if fit.get("issueCutoffExclusive") not in (None,issue.isoformat()): raise ValueError("garlic fit cutoff does not match issue date")
        fit_end=fit.get("trainingTargetEnd")
        if fit_end is not None and _date(fit_end,"garlic trainingTargetEnd")>=issue: raise ValueError("garlic training target is not strictly before issue date")
    if crop not in HORIZONS or not isinstance(forecasts,list): raise ValueError("forecast rows are invalid")
    if not isinstance(price_hash,str) or len(price_hash)!=64 or any(c not in "0123456789abcdef" for c in price_hash): raise ValueError("forecast source prices SHA-256 is missing or malformed")
    expected=set(HORIZONS[crop]); rows={}
    for row in forecasts:
        if not isinstance(row,dict): raise ValueError("forecast row must be an object")
        h=row.get("horizon")
        if isinstance(h,bool) or h not in expected or h in rows: raise ValueError(f"invalid or duplicate {crop} horizon")
        target=_date(row.get("targetDate"),"targetDate")
        if target!=issue+dt.timedelta(days=h): raise ValueError("forecast targetDate does not equal issueDate + horizon")
        source=_date(row.get("sourceDate"),"sourceDate") if row.get("sourceDate") is not None else None
        if source is not None and source>=issue: raise ValueError("forecast sourceDate must precede issue date")
        if source is not None and row.get("sourceAgeDays") is not None:
            age=row["sourceAgeDays"]
            if isinstance(age,bool) or not isinstance(age,int) or age!=(issue-source).days or age<0: raise ValueError("forecast source age is inconsistent")
        status=row.get("status")
        if status not in ("available","unavailable","held"): raise ValueError("forecast status is invalid")
        candidate_key="prediction" if crop in ONION_HORIZONS else "blendPrediction"
        candidate=_finite_positive(row.get(candidate_key),candidate_key) if status=="available" else None
        comps={}
        for field in COMPARATORS[crop]:
            value=row.get(field)
            comps[field]=_finite_positive(value,field) if value is not None else None
        if status=="available":
            if source is None: raise ValueError("available forecast requires sourceDate")
            if row.get("sourceAgeDays") is None: raise ValueError("available forecast requires sourceAgeDays")
            if row["sourceAgeDays"]>14: raise ValueError("available forecast source is stale")
            if any(v is None for v in comps.values()): raise ValueError("available forecast requires both positive finite comparator forecasts")
            if crop=="garlic":
                expected_weight=1.0 if h in (3,4) else .75
                if row.get("weight")!=expected_weight: raise ValueError("garlic forecast weight differs from frozen horizon weight")
        reason=row.get("holdReason") if crop=="garlic" else row.get("fallbackReason")
        if status!="available" and not reason: reason="forecast_not_issued"
        rows[h]={"horizon":h,"issueDate":issue,"target":target,"targetDate":target,"sourceDate":source,"status":status,"candidate":candidate,
                 "comparators":comps,"reason":reason,"raw":row}
    if set(rows)!=expected: raise ValueError(f"{crop} artifact must include every expected horizon")
    if crop in ONION_HORIZONS:
        rule=doc.get("featureRule","")
        if "issueDateKst-1" not in rule: raise ValueError("onion feature rule does not attest a pre-issue cutoff")
        _verified_onion_state(doc,path,issue,issued,artifact_bytes_by_path or {},HORIZONS[crop])
    return {"crop":crop,"issueDate":issue,"issuedAt":issued,"path":str(path),"sha256":digest,
            "sourcePricesSha256":price_hash,"snapshotCutoff":snapshot_cutoff,"rows":rows,"raw":doc}


def _season(day):
    if day.month in (12,1,2): return "winter"
    if day.month in (3,4,5): return "spring"
    if day.month in (6,7,8): return "summer"
    return "autumn"


def _truth_series(actuals,crop):
    return actuals.get("onion",{}) if crop in ONION_HORIZONS else actuals.get(crop,{})


def _comparator_eligible(forecast_row, comparator, cutoff, actual_series):
    """Whether an issued comparator has a mature positive truth pair."""
    source=forecast_row["sourceDate"]
    raw=forecast_row["raw"]
    age=raw.get("sourceAgeDays")
    target=forecast_row["target"]
    return (source is not None and isinstance(age,int) and not isinstance(age,bool) and
            age==(forecast_row["issueDate"]-source).days and 0<=age<=14 and
            forecast_row["comparators"].get(comparator) is not None and target<=cutoff and
            actual_series.get(target) is not None and actual_series[target][0]>0)


def _test_record(crop,h,comp,loss,rows):
    diff={}
    for row in rows:
        actual=row["actual"]; cand=row["candidate"]; base=row["comparators"][comp]
        target=_date(row["target"],"paired target date")
        diff[target]=(abs(actual-cand)-abs(actual-base)) if loss=="ae" else 100*(abs(actual-cand)-abs(actual-base))/actual
    actuals=[r["actual"] for r in rows]
    candidate=[r["candidate"] for r in rows]; comparator=[r["comparators"][comp] for r in rows]
    return {"product":crop,"horizon":h,"comparator":comp,"loss":loss,"n":len(rows),
            "candidateMetric":dh._metrics(actuals,candidate)["mae" if loss=="ae" else "mape"],
            "comparatorMetric":dh._metrics(actuals,comparator)["mae" if loss=="ae" else "mape"],
            "dm":{"9":dh._dm(diff,9),"28":dh._dm(diff,28)}}


def _conclusion(tests, coverage_nondeclining):
    enough=all(x["n"]>=MIN_PAIRS and x["dm"]["9"].get("status")=="tested" for x in tests)
    metrics_better=all(x["candidateMetric"] is not None and x["comparatorMetric"] is not None and x["candidateMetric"]<x["comparatorMetric"] for x in tests)
    improves=all(x["dm"]["9"].get("holmSignificant",False) and x["dm"]["9"].get("meanLossDifference",0)<0 for x in tests)
    worsens=any(x["dm"]["9"].get("holmSignificant",False) and x["dm"]["9"].get("meanLossDifference",0)>0 for x in tests)
    status="inference_unavailable" if not enough else "improvement" if metrics_better and improves and coverage_nondeclining else "worsening" if worsens else "inconclusive"
    return {"conclusion":status,"inferenceAvailable":enough,"hac28SensitivityAvailable":all(x["dm"]["28"].get("status")=="tested" for x in tests),"pairedMetricsImprove":metrics_better,
            "allFourHac9HolmTestsImprove":improves,"coverageNondeclining":coverage_nondeclining,
            "holdReason":"fewer_than_30_pairs_or_invalid_hac_variance" if not enough else None}


def build_assessment(forecast_files, raw_truth, truth_sha256, provenance):
    """Build paired assessment from (path, bytes) forecast inputs and reconciled truth snapshot."""
    if not isinstance(truth_sha256,str) or len(truth_sha256)!=64: raise ValueError("truth snapshot SHA-256 is invalid")
    cutoff=_date(raw_truth.get("meta",{}).get("contextEnd"),"truth contextEnd")
    actuals=dh._observations(raw_truth)
    artifacts=[]; seen=set()
    artifact_bytes_by_path={Path(path).resolve():data for path,data in forecast_files}
    for path,data in forecast_files:
        a=_load_artifact(path,data,artifact_bytes_by_path); key=(a["crop"],a["issueDate"])
        if key in seen: raise ValueError(f"duplicate product/issue date: {key}")
        seen.add(key); artifacts.append(a)
    allrows=[]; coverage={}; reason_counts={}; conclusions=[]; strata=[]
    for a in artifacts:
        crop=a["crop"]; actual_series=_truth_series(actuals,crop)
        issue_key=f"{crop}/{a['issueDate'].isoformat()}"; coverage[issue_key]={}
        for h in HORIZONS[crop]:
            artifact_row=a["rows"][h]; target=artifact_row["target"]
            state={"product":crop,"horizon":h,"issueDate":a["issueDate"].isoformat(),"target":target.isoformat(),"targetDate":target.isoformat(),"forecastStatus":artifact_row["status"],"reason":None}
            if artifact_row["status"]!="available": state["reason"]="held_forecast:"+str(artifact_row["reason"])
            elif target>cutoff: state["reason"]="immature_target_after_truth_contextEnd"
            elif actual_series.get(target) is None: state["reason"]="completed_scope_without_positive_target_unevaluable"
            else:
                actual=float(actual_series[target][0])
                if not math.isfinite(actual) or actual<=0: raise ValueError("truth actual must be positive and finite")
                state.update(actual=actual,candidate=artifact_row["candidate"],comparators=artifact_row["comparators"],reason="paired_positive_target")
                allrows.append(state)
            reason_counts[state["reason"]]=reason_counts.get(state["reason"],0)+1
            coverage[issue_key][str(h)]={"forecastCount":1,"issuedCount":int(artifact_row["status"]=="available"),"heldCount":int(artifact_row["status"]!="available"),
                "mature":target<=cutoff,"pairedPositiveTargetCount":int("actual" in state),"unpairedReason":None if "actual" in state else state["reason"]}
    comparisons=[]
    for crop,hs in HORIZONS.items():
        for h in hs:
            paired=[r for r in allrows if r["product"]==crop and r["horizon"]==h]
            for comp in COMPARATORS[crop]:
                for loss in ("ae","ape"): comparisons.append(_test_record(crop,h,comp,loss,paired))
            groups={}
            for row in paired:
                target=_date(row["targetDate"],"paired target date")
                for kind,label in (("year",str(target.year)),("season",_season(target))): groups.setdefault((kind,label),[]).append(row)
            for (kind,label),group in sorted(groups.items()):
                for comp in COMPARATORS[crop]:
                    strata.append({"product":crop,"horizon":h,"stratum":kind,"value":label,"comparator":comp,"n":len(group),
                        "candidate":dh._metrics([r["actual"] for r in group],[r["candidate"] for r in group]),
                        "comparatorMetrics":dh._metrics([r["actual"] for r in group],[r["comparators"][comp] for r in group])})
        family=[x for x in comparisons if x["product"]==crop]
        if len(family)!=FAMILY_SIZE[crop]: raise AssertionError(f"{crop} frozen DM family must contain {FAMILY_SIZE[crop]} tests")
        for bw in HAC: dh._holm([r["dm"][str(bw)] for r in family])
        for h in hs:
            tests=[r for r in family if r["horizon"]==h]
            candidate_count=sum(r["product"]==crop and r["horizon"]==h for r in allrows)
            mature_comparators={comp:0 for comp in COMPARATORS[crop]}
            actual_series=_truth_series(actuals,crop)
            for a in artifacts:
                if a["crop"]!=crop: continue
                fr=a["rows"][h]; target=fr["target"]
                for comp in mature_comparators:
                    mature_comparators[comp]+=int(_comparator_eligible(fr,comp,cutoff,actual_series))
            nondeclining=all(candidate_count>=n for n in mature_comparators.values())
            conclusions.append({"product":crop,"horizon":h,**_conclusion(tests,nondeclining)})
        aggregate_coverage={}
        for h in hs:
            forecasts=[a["rows"][h] for a in artifacts if a["crop"]==crop]
            paired=[r for r in allrows if r["product"]==crop and r["horizon"]==h]
            aggregate_coverage[str(h)]={"forecastCount":len(forecasts),"issuedCount":sum(r["status"]=="available" for r in forecasts),
                "heldCount":sum(r["status"]!="available" for r in forecasts),"pairedPositiveTargetCount":len(paired),
                "coverageRate":len(paired)/len(forecasts) if forecasts else None,
                "matureComparatorIssueCounts":{comp:sum(_comparator_eligible(a["rows"][h],comp,cutoff,_truth_series(actuals,crop)) for a in artifacts if a["crop"]==crop) for comp in COMPARATORS[crop]},
                "unpairedReasons":{reason:sum(v==reason for v in [coverage[f"{crop}/{a['issueDate'].isoformat()}"][str(h)]["unpairedReason"] for a in artifacts if a["crop"]==crop]) for reason in sorted({coverage[f"{crop}/{a['issueDate'].isoformat()}"][str(h)]["unpairedReason"] for a in artifacts if a["crop"]==crop} - {None})}}
        coverage[crop]=aggregate_coverage
    macro={}
    for crop,hs in HORIZONS.items():
        metric_by_h=[]; enough=True
        for h in hs:
            rows=[r for r in allrows if r["product"]==crop and r["horizon"]==h]
            enough &= len(rows)>=MIN_PAIRS
            metric_by_h.append(dh._metrics([r["actual"] for r in rows],[r["candidate"] for r in rows])["mape"] if rows else None)
        value=sum(metric_by_h)/len(hs) if all(x is not None for x in metric_by_h) else None
        macro[crop]={"equalHorizonMape":value,"pairedCountsByHorizon":{str(h):sum(r["product"]==crop and r["horizon"]==h for r in allrows) for h in hs},
                      "under10ClaimEligible":bool(enough),"under10Claim":(value<10 if enough and value is not None else None),
                      "reason":None if enough else "requires_at_least_30_positive_target_pairs_at_every_horizon"}
        if crop=="onion_extension":
            macro[crop].update(under10ClaimEligible=False,under10Claim=None,
                under10PointEstimate=(value<10 if enough and value is not None else None),
                interpretation="extension_horizons_3_to_10_only; not a full-onion success claim")
    onion_issue_dates={a["issueDate"] for a in artifacts if a["crop"]=="onion"}
    extension_issue_dates={a["issueDate"] for a in artifacts if a["crop"]=="onion_extension"}
    same_day_issues=onion_issue_dates & extension_issue_dates
    # Admit an issue date to the full-horizon macro only when all ten target
    # dates are mature and all ten candidate/positive-truth pairs exist. The
    # ordinary per-horizon coverage above retains held, immature, and
    # unevaluable rows independently of this complete-case macro cohort.
    paired_horizons_by_issue={}
    for row in allrows:
        if row["product"] in ONION_HORIZONS:
            paired_horizons_by_issue.setdefault(row["issueDate"],set()).add(row["horizon"])
    full_issue_dates=[]
    full_issue_exclusions={}
    full_issue_exclusion_counts={}
    for issue in sorted(same_day_issues):
        issue_text=issue.isoformat()
        missing_reasons=[]
        if issue+dt.timedelta(days=10)>cutoff:
            missing_reasons.append("incomplete_full_horizon_maturity")
        if paired_horizons_by_issue.get(issue_text,set())!=set(range(1,11)):
            for family in ONION_HORIZONS:
                issue_coverage=coverage.get(f"{family}/{issue_text}",{})
                missing_reasons.extend(value["unpairedReason"] for value in issue_coverage.values()
                                       if value["unpairedReason"] is not None)
            if not missing_reasons:
                missing_reasons.append("incomplete_issued_positive_truth_pairs")
        if missing_reasons:
            reasons=sorted(set(missing_reasons))
            full_issue_exclusions[issue_text]=reasons
            for reason in reasons:
                full_issue_exclusion_counts[reason]=full_issue_exclusion_counts.get(reason,0)+1
        else:
            full_issue_dates.append(issue_text)
    full_issue_date_set=set(full_issue_dates)
    full_onion_rows=[r for r in allrows if r["issueDate"] in full_issue_date_set and r["product"] in ONION_HORIZONS]
    full_counts={str(h):sum(r["horizon"]==h for r in full_onion_rows) for h in range(1,11)}
    full_metrics={}
    for h in range(1,11):
        rows=[r for r in full_onion_rows if r["horizon"]==h]
        full_metrics[str(h)]=dh._metrics([r["actual"] for r in rows],[r["candidate"] for r in rows])["mape"] if rows else None
    full_eligible=bool(same_day_issues) and all(full_counts[str(h)]>=MIN_PAIRS for h in range(1,11))
    full_macro=sum(full_metrics.values())/10 if full_eligible and all(v is not None for v in full_metrics.values()) else None
    macro["onionFullHorizonSameIssueDay"]={"equalHorizonMape":full_macro,"pairedCountsByHorizon":full_counts,
        "sameIssueDayCount":len(same_day_issues),"cohortIssueDateCount":len(full_issue_dates),"sameIssueDates":full_issue_dates,
        "excludedIssueDates":full_issue_exclusions,"excludedIssueDateReasonCounts":full_issue_exclusion_counts,
        "eligible":full_eligible,"under10PointEstimate":(full_macro<10 if full_eligible and full_macro is not None else None),
        "interpretation":"descriptive_point_estimate_not_a_success_claim",
        "reason":None if full_eligible else "requires_same_issue_day_artifacts_and_at_least_30_mature_positive_truth_pairs_at_each_of_10_horizons"}
    # Supplementary calendar-maturity cohort: selecting issue dates depends only
    # on required artifacts and whether the tenth target is inside the truth
    # snapshot. Missing positive truth at an individual horizon never removes
    # an otherwise mature issue from this cohort.
    mature_summaries={}
    for product,required_families,hs in (
            ("onion",ONION_HORIZONS,(1,2,3,4,5,6,7,8,9,10)),
            ("garlic",{"garlic":HORIZONS["garlic"]},HORIZONS["garlic"])):
        by_family={family:{a["issueDate"]:a for a in artifacts if a["crop"]==family} for family in required_families}
        available_dates=set.union(*(set(items) for items in by_family.values())) if by_family else set()
        issue_dates=set.intersection(*(set(items) for items in by_family.values())) if by_family else set()
        missing_artifact_dates=sorted(available_dates-issue_dates)
        mature_dates=sorted(day for day in issue_dates if day+dt.timedelta(days=10)<=cutoff)
        immature_dates=sorted(day for day in issue_dates if day+dt.timedelta(days=10)>cutoff)
        actual_series=_truth_series(actuals,"onion" if product=="onion" else "garlic")
        per_horizon={}
        for h in hs:
            pair_rows=[]; held_count=0; absent_positive_count=0
            for issue in mature_dates:
                family="onion" if product=="onion" and h<=2 else "onion_extension" if product=="onion" else "garlic"
                row=by_family[family][issue]["rows"][h]
                if row["status"]!="available":
                    held_count+=1
                    continue
                actual_item=actual_series.get(row["target"])
                if actual_item is None:
                    absent_positive_count+=1
                    continue
                actual=float(actual_item[0])
                if not math.isfinite(actual) or actual<=0: raise ValueError("truth actual must be positive and finite")
                # All methods use the exact same issue/horizon/target pairs.
                pair_rows.append({"actual":actual,"candidate":row["candidate"],"comparators":row["comparators"]})
            metrics={}
            for method in ("candidate",*COMPARATORS["onion" if product=="onion" else "garlic"]):
                preds=([r["candidate"] for r in pair_rows] if method=="candidate" else [r["comparators"][method] for r in pair_rows])
                base=dh._metrics([r["actual"] for r in pair_rows],preds)
                metrics[method]={"n":len(pair_rows),"mae":base["mae"],"mape":base["mape"]}
            deltas={comp:{metric:(metrics["candidate"][metric]-metrics[comp][metric]
                                  if metrics["candidate"][metric] is not None and metrics[comp][metric] is not None else None)
                          for metric in ("mae","mape")}
                    for comp in COMPARATORS["onion" if product=="onion" else "garlic"]}
            n=len(pair_rows)
            per_horizon[str(h)]={"matureIssueCount":len(mature_dates),"heldForecastCount":held_count,
                "absentPositiveTargetCount":absent_positive_count,"pairedCount":n,
                "coverageRate":n/len(mature_dates) if mature_dates else None,
                "methods":metrics,"candidateMinusComparator":deltas}
        sufficient=all(per_horizon[str(h)]["pairedCount"]>=MIN_PAIRS for h in hs)
        aggregate={}
        for method in ("candidate",*COMPARATORS["onion" if product=="onion" else "garlic"]):
            aggregate[method]={metric:(sum(per_horizon[str(h)]["methods"][method][metric] for h in hs)/len(hs)
                                       if sufficient else None) for metric in ("mae","mape")}
        aggregate_deltas={comp:{metric:(aggregate["candidate"][metric]-aggregate[comp][metric]
                                         if sufficient else None)
                                for metric in ("mae","mape")}
                          for comp in COMPARATORS["onion" if product=="onion" else "garlic"]}
        mature_summaries[product]={"cohort":"descriptive_calendar_maturity_observed_target",
            "requiredArtifactFamilies":list(required_families),"availableArtifactIssueDateCount":len(available_dates),
            "sameDayArtifactIssueCount":len(issue_dates),"missingRequiredArtifactIssueDateCount":len(missing_artifact_dates),
            "missingRequiredArtifactIssueDates":[day.isoformat() for day in missing_artifact_dates],
            "matureIssueCount":len(mature_dates),"immatureIssueCount":len(immature_dates),
            "matureIssueDates":[day.isoformat() for day in mature_dates],
            "perHorizon":per_horizon,
            "equalHorizon10":{"supported":sufficient,"methods":aggregate,"candidateMinusComparator":aggregate_deltas,
                "reason":None if sufficient else "requires_at_least_30_same_pair_positive_targets_at_each_horizon"},
            "interpretation":"descriptive_only; existing_complete_case_macro_and_DM_inference_are_unchanged"}
    return {"schemaVersion":1,"status":"complete","protocol":{"minimumPairs":MIN_PAIRS,"hacCalendarDays":list(HAC),"alpha":ALPHA,"familySizes":FAMILY_SIZE,
        "noImputation":True,"maturity":"targetDate <= truth contextEnd; DB broad scopes reconciled complete/empty"},
        "truth":{"sha256":truth_sha256,"contextStart":raw_truth["meta"].get("contextStart"),"contextEnd":cutoff.isoformat(),
                 "selectedRequestHashes":provenance.get("selectedRequestHashes"),"selectedQueryStates":provenance.get("selectedQueryStates"),
                 "selectedScopesSha256":provenance.get("selectedScopesSha256"),"broadScopesSha256":provenance.get("broadScopesSha256"),"reconciliation":provenance.get("reconciliation")},
        "forecastArtifacts":[{"product":"onion" if a["crop"] in ONION_HORIZONS else a["crop"],"horizonFamily":"1-2" if a["crop"]=="onion" else "3-10" if a["crop"]=="onion_extension" else "1-10","issueDate":a["issueDate"].isoformat(),"issuedAtKst":a["issuedAt"].isoformat(),"path":a["path"],"sha256":a["sha256"],"sourcePricesSha256":a["sourcePricesSha256"],"snapshotCutoff":a["snapshotCutoff"].isoformat() if a["snapshotCutoff"] else None} for a in artifacts],
        "coverage":coverage,"reasonCounts":reason_counts,"pairedRows":allrows,"comparisons":comparisons,"conclusions":conclusions,"strata":strata,"macroMetrics":macro,
        "descriptiveMaturityCohort":mature_summaries,
        "limitations":"retrospective truth snapshot revision is used; no publication-time or market-closure inference; held and no-positive targets are not imputed"}


def _atomic_new(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(f".{path.name}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp")
    try:
        with temp.open("xb") as f: f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(temp,path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass


def _forecast_paths(values):
    paths=[]
    for raw in values:
        p=Path(raw)
        if p.is_dir(): paths.extend(sorted(p.glob("*.json")))
        elif p.is_file(): paths.append(p)
        else: raise FileNotFoundError(p)
    if not paths: raise ValueError("no forecast JSON files supplied")
    return paths


def run(forecast_inputs,truth_path,db_path,output_path):
    for path in (truth_path,db_path,*_forecast_paths(forecast_inputs)):
        if Path(output_path).resolve()==Path(path).resolve(): raise ValueError("assessment output must not replace an input or database")
    forecast_files=[(str(path),path.read_bytes()) for path in _forecast_paths(forecast_inputs)]
    truth_bytes=Path(truth_path).read_bytes(); raw=json.loads(truth_bytes); core=dh._observations(raw)
    _,prov=market_mix.extract_market_data(Path(db_path),raw,core)
    selected=prov["selected"]
    provenance={**selected,"broadScopesSha256":prov["broad"].get("scopesSha256"),"reconciliation":prov["reconciliation"]}
    result=build_assessment(forecast_files,raw,_sha(truth_bytes),provenance)
    _atomic_new(Path(output_path),(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
    return {"output":str(output_path),"sha256":_sha((json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()),"pairedRows":len(result["pairedRows"])}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--forecast",action="append",required=True,help="immutable forecast JSON file or narrow directory; repeatable")
    p.add_argument("--truth",type=Path,required=True); p.add_argument("--db",type=Path,default=ROOT/"data/agri.sqlite"); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv); print(json.dumps(run(a.forecast,a.truth,a.db,a.output),ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
