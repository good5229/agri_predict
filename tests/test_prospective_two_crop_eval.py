import datetime as dt
import json

import pytest

from agri_data.prospective_two_crop_eval import build_assessment


def _actual(day):
    return 1000.0 + (day.toordinal() % 100)


def _model_state(fit_origin,horizons=(1,2)):
    return {name:{"month":fit_origin[:7],"fitOrigin":fit_origin,"byHorizon":{
        str(h):{"available":False,"fitOrigin":fit_origin,"trainingTargetEnd":None,"trainingTargetDays":0,
                "trainingSampleCount":0,"reason":"synthetic_unavailable"} for h in horizons}}
        for name in ("direct","110001","110008")}


def _artifact(crop, issue, *, held_horizons=(), stale_horizons=(), onion_reference=None, onion_fit_origin=None):
    is_onion=crop in ("onion","onion_extension")
    hs=(1,2) if crop=="onion" else tuple(range(3,11)) if crop=="onion_extension" else tuple(range(1,11))
    forecasts=[]
    for h in hs:
        target=issue+dt.timedelta(days=h); actual=_actual(target); held=h in held_horizons
        age=20 if h in stale_horizons else 1
        source=issue-dt.timedelta(days=age)
        if is_onion:
            forecasts.append({"horizon":h,"targetDate":target.isoformat(),"sourceDate":source.isoformat(),"sourceAgeDays":age,
                "status":"unavailable" if held else "available","prediction":None if held else actual+(-1 if h%2 else 1),
                "directRidgePrediction":actual+10,"persistencePrediction":actual+20,"fallbackReason":"synthetic_hold" if held else None})
        else:
            forecasts.append({"horizon":h,"targetDate":target.isoformat(),"sourceDate":source.isoformat(),"sourceAgeDays":age,
                "status":"held" if held else "available","blendPrediction":None if held else actual+(-1 if h%2 else 1),
                "ridgePrediction":actual+10,"persistencePrediction":actual+20,"weight":1.0 if h in (3,4) else .75,
                "holdReason":"synthetic_hold" if held else None})
    doc={"schemaVersion":1,"issueDateKst":issue.isoformat(),"issuedAtKst":dt.datetime.combine(issue,dt.time(9),tzinfo=dt.timezone(dt.timedelta(hours=9))).isoformat(),
         "pricesSha256":"1"*64}
    if is_onion:
        origin=(onion_fit_origin or issue).isoformat()
        model=_model_state(origin,hs)
        ref=None
        if onion_reference:
            ref={"path":onion_reference[0],"sha256":onion_reference[1]}
        doc.update(featureRule="all model features use observations on or before issueDateKst-1",modelState=None if ref else model,
                   modelStateSha256=__import__("hashlib").sha256((json.dumps(model,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()).hexdigest(),
                   modelStateReference=ref)
    else: doc.update(input={"pricesSha256":"2"*64,"snapshotCutoff":(issue-dt.timedelta(days=1)).isoformat()},fit={"issueCutoffExclusive":issue.isoformat(),"trainingTargetEnd":(issue-dt.timedelta(days=1)).isoformat()})
    doc["products"]=[{"productId":"onion","forecasts":forecasts}] if is_onion else None
    if crop=="garlic": doc["forecasts"]=forecasts
    if crop=="garlic": doc.pop("products")
    if is_onion: doc["pricesSha256"]="3"*64
    return (f"{crop}-{issue}.json",json.dumps(doc,sort_keys=True).encode())


def _truth(cutoff, omit=()):
    days=set()
    for issue_offset in range(30):
        issue=dt.date(2026,10,7)+dt.timedelta(days=issue_offset)
        days.update(issue+dt.timedelta(days=h) for h in range(1,11))
    return {"schemaVersion":1,"meta":{"contextStart":"2026-10-01","contextEnd":cutoff.isoformat()},"products":[
        {"id":crop,"observations":[{"date":d.isoformat(),"actual":_actual(d),"volumeKg":10.0}
            for d in sorted(days) if d not in omit and d<=cutoff]}
        for crop in ("cabbage","radish","onion","garlic","pepper")
    ]}


def _truth_for_issues(issues, cutoff, omit=()):
    days={issue+dt.timedelta(days=h) for issue in issues for h in range(1,11)}
    return {"schemaVersion":1,"meta":{"contextStart":"2026-10-01","contextEnd":cutoff.isoformat()},"products":[
        {"id":crop,"observations":[{"date":d.isoformat(),"actual":_actual(d),"volumeKg":10.0}
            for d in sorted(days) if d not in omit and d<=cutoff]}
        for crop in ("cabbage","radish","onion","garlic","pepper")
    ]}


def test_pairing_runs_fixed_families_dm_holm_and_macro_gate():
    issues=[dt.date(2026,10,7)+dt.timedelta(days=i) for i in range(30)]
    first_issues={"2026-10":issues[0],"2026-11":dt.date(2026,11,1)}
    first_artifacts={month:_artifact("onion",day) for month,day in first_issues.items()}
    references={month:(path,__import__("hashlib").sha256(data).hexdigest()) for month,(path,data) in first_artifacts.items()}
    files=list(first_artifacts.values())
    files += [_artifact("onion",issue,onion_reference=references[issue.strftime("%Y-%m")],onion_fit_origin=first_issues[issue.strftime("%Y-%m")])
              for issue in issues if issue!=first_issues[issue.strftime("%Y-%m")]]
    files += [_artifact("garlic",issue) for issue in issues]
    truth=_truth(dt.date(2026,12,31))
    result=build_assessment(files,truth,"a"*64,{"selectedRequestHashes":{"1":"abc"},"reconciliation":{"positiveTargetsReconciled":100}})
    assert sum(x["product"]=="onion" for x in result["comparisons"])==8
    assert sum(x["product"]=="garlic" for x in result["comparisons"])==40
    assert all(x["n"]==30 for x in result["comparisons"] if x["product"] in ("onion","garlic"))
    assert all(x["n"]==0 for x in result["comparisons"] if x["product"]=="onion_extension")
    assert all("holmPValue" in x["dm"]["9"] and "holmPValue" in x["dm"]["28"] for x in result["comparisons"])
    assert result["coverage"]["garlic"]["1"]["pairedPositiveTargetCount"]==30
    assert result["macroMetrics"]["garlic"]["under10ClaimEligible"]
    assert result["macroMetrics"]["onionFullHorizonSameIssueDay"]["equalHorizonMape"] is None
    json.dumps(result,allow_nan=False)


def test_maturity_no_positive_and_held_forecasts_remain_distinct():
    issue=dt.date(2026,10,7)
    files=[_artifact("onion",issue,held_horizons=(2,)),_artifact("garlic",issue,held_horizons=(2,))]
    truth=_truth(issue+dt.timedelta(days=1),omit={issue+dt.timedelta(days=1)})
    result=build_assessment(files,truth,"b"*64,{})
    assert result["reasonCounts"]["immature_target_after_truth_contextEnd"]==8
    assert result["reasonCounts"]["completed_scope_without_positive_target_unevaluable"]==2
    assert result["reasonCounts"]["held_forecast:synthetic_hold"]==2
    assert len(result["pairedRows"])==0


def test_malformed_artifact_duplicate_horizon_and_duplicate_issue_rejected():
    issue=dt.date(2026,10,7); item=_artifact("onion",issue)
    doc=json.loads(item[1]); doc["products"][0]["forecasts"][1]["horizon"]=1
    truth=_truth(dt.date(2026,12,31))
    with pytest.raises(ValueError,match="duplicate"):
        build_assessment([(item[0],json.dumps(doc).encode())],truth,"c"*64,{})
    item=_artifact("onion",issue)
    with pytest.raises(ValueError,match="duplicate product/issue"):
        build_assessment([item,item],truth,"c"*64,{})


def test_no_mature_actuals_produce_empty_pairs_and_unavailable_inference():
    issue=dt.date(2026,10,7)
    result=build_assessment([_artifact("onion",issue)],_truth(issue-dt.timedelta(days=1)),"d"*64,{})
    assert result["pairedRows"]==[]
    assert result["reasonCounts"]["immature_target_after_truth_contextEnd"]==2
    assert all(x["dm"]["9"]["pValue"]==1.0 for x in result["comparisons"])
    assert all(x["conclusion"]=="inference_unavailable" for x in result["conclusions"])


def test_stale_held_forecast_is_valid_but_not_counted_as_comparator_coverage():
    issue=dt.date(2026,10,7)
    files=[_artifact("garlic",issue,held_horizons=(2,),stale_horizons=(2,))]
    result=build_assessment(files,_truth(issue+dt.timedelta(days=10)),"e"*64,{})
    assert result["coverage"]["garlic"]["2"]["heldCount"]==1
    assert result["coverage"]["garlic"]["2"]["matureComparatorIssueCounts"]["ridgePrediction"]==0


def test_fresh_comparator_coverage_is_counted_when_candidate_is_held():
    issue=dt.date(2026,10,7)
    files=[_artifact("garlic",issue,held_horizons=(2,))]
    result=build_assessment(files,_truth(issue+dt.timedelta(days=2)),"1"*64,{})
    cov=result["coverage"]["garlic"]["2"]
    assert cov["heldCount"]==1
    assert cov["pairedPositiveTargetCount"]==0
    assert cov["matureComparatorIssueCounts"]["ridgePrediction"]==1


def test_comparator_coverage_is_independent_of_artifact_input_order():
    issues=[dt.date(2026,10,7),dt.date(2026,10,8)]
    files=[_artifact("garlic",issues[0],held_horizons=(10,)),_artifact("garlic",issues[1])]
    truth=_truth(dt.date(2026,10,20))
    forward=build_assessment(files,truth,"2"*64,{})
    reverse=build_assessment(list(reversed(files)),truth,"2"*64,{})
    assert forward["coverage"]["garlic"]["1"]["matureComparatorIssueCounts"]==reverse["coverage"]["garlic"]["1"]["matureComparatorIssueCounts"]
    assert forward["conclusions"]==reverse["conclusions"]


@pytest.mark.parametrize("bad_reference,match",[("wrong_hash","byte SHA-256"),("wrong_model_hash","model-state SHA-256"),
                                                  ("wrong_follow_hash","follow-on model-state SHA-256"),
                                                  ("wrong_month","month mismatch")])
def test_onion_referenced_state_requires_supplied_verified_same_month_first_issue(bad_reference,match):
    first_issue=dt.date(2026,10,7); first_path,first_bytes=_artifact("onion",first_issue)
    path="onion-2026-10-08.json"; issue=first_issue+dt.timedelta(days=1)
    digest=__import__("hashlib").sha256(first_bytes).hexdigest()
    if bad_reference=="wrong_hash": digest="0"*64
    elif bad_reference=="wrong_model_hash":
        changed=json.loads(first_bytes)
        changed["modelStateSha256"]="0"*64
        first_bytes=json.dumps(changed,sort_keys=True).encode()
        digest=__import__("hashlib").sha256(first_bytes).hexdigest()
    second=_artifact("onion",issue,onion_reference=(first_path,digest),onion_fit_origin=first_issue)
    if bad_reference=="wrong_follow_hash":
        second_path,second_bytes=second
        changed=json.loads(second_bytes); changed["modelStateSha256"]="0"*64
        second=(second_path,json.dumps(changed,sort_keys=True).encode())
    if bad_reference=="wrong_month":
        first_issue=dt.date(2026,9,30); first_path,first_bytes=_artifact("onion",first_issue)
        digest=__import__("hashlib").sha256(first_bytes).hexdigest()
        second=_artifact("onion",issue,onion_reference=(first_path,digest),onion_fit_origin=first_issue)
    truth=_truth(dt.date(2026,12,31))
    with pytest.raises(ValueError,match=match):
        build_assessment([(str(first_path),first_bytes),second],truth,"f"*64,{})


def test_hac9_primary_inference_does_not_require_hac28_sensitivity():
    from agri_data.prospective_two_crop_eval import _conclusion
    tests=[{"n":30,"candidateMetric":1.0,"comparatorMetric":2.0,
            "dm":{"9":{"status":"tested","holmSignificant":True,"meanLossDifference":-1.0},
                  "28":{"status":"invalid_variance","holmSignificant":False,"meanLossDifference":-1.0}}}
           for _ in range(4)]
    result=_conclusion(tests,True)
    assert result["inferenceAvailable"] is True
    assert result["hac28SensitivityAvailable"] is False
    assert result["conclusion"]=="improvement"


def test_extension_family_pairs_separately_and_gates_full_onion_macro_by_same_issue_dates():
    from agri_data.prospective_two_crop_eval import build_assessment
    issues=[dt.date(2026,10,7)+dt.timedelta(days=i) for i in range(30)]
    month_starts={"2026-10":issues[0],"2026-11":dt.date(2026,11,1)}
    base_first={m:_artifact("onion",d) for m,d in month_starts.items()}
    ext_first={m:_artifact("onion_extension",d) for m,d in month_starts.items()}
    refs={m:(path,__import__("hashlib").sha256(data).hexdigest()) for m,(path,data) in ext_first.items()}
    base_refs={m:(path,__import__("hashlib").sha256(data).hexdigest()) for m,(path,data) in base_first.items()}
    files=list(base_first.values())+list(ext_first.values())
    for issue in issues:
        month=issue.strftime("%Y-%m")
        if issue!=month_starts[month]:
            files.append(_artifact("onion",issue,onion_reference=base_refs[month],onion_fit_origin=month_starts[month]))
            files.append(_artifact("onion_extension",issue,onion_reference=refs[month],onion_fit_origin=month_starts[month]))
    # This shared issue date has mature short targets but immature long ones;
    # it must not enter any horizon's full-onion macro mean.
    partial_issue=dt.date(2026,11,10)
    files.append(_artifact("onion",partial_issue,onion_reference=base_refs["2026-11"],onion_fit_origin=month_starts["2026-11"]))
    files.append(_artifact("onion_extension",partial_issue,onion_reference=refs["2026-11"],onion_fit_origin=month_starts["2026-11"]))
    truth=_truth(dt.date(2026,11,15))
    result=build_assessment(files,truth,"9"*64,{})
    assert sum(x["product"]=="onion" for x in result["comparisons"])==8
    assert sum(x["product"]=="onion_extension" for x in result["comparisons"])==32
    assert result["protocol"]["familySizes"]=={"onion":8,"onion_extension":32,"garlic":40}
    macro=result["macroMetrics"]["onionFullHorizonSameIssueDay"]
    assert macro["sameIssueDayCount"]==31
    assert macro["cohortIssueDateCount"]==30
    assert "2026-11-10" not in macro["sameIssueDates"]
    assert macro["excludedIssueDates"]["2026-11-10"]==["immature_target_after_truth_contextEnd", "incomplete_full_horizon_maturity"]
    assert macro["eligible"] is True
    assert all(n==30 for n in macro["pairedCountsByHorizon"].values())
    assert macro["under10PointEstimate"] is True
    assert macro["interpretation"]=="descriptive_point_estimate_not_a_success_claim"
    extension_macro=result["macroMetrics"]["onion_extension"]
    assert extension_macro["under10ClaimEligible"] is False
    assert extension_macro["under10Claim"] is None
    assert extension_macro["under10PointEstimate"] is True
    assert extension_macro["interpretation"]=="extension_horizons_3_to_10_only; not a full-onion success claim"
    assert sum(r["product"]=="onion_extension" for r in result["pairedRows"])==8*30+3
    assert len([a for a in result["forecastArtifacts"] if a["horizonFamily"]=="3-10"])==31
    assert result["coverage"]["onion_extension"]["10"]["pairedPositiveTargetCount"]==30
    assert result["reasonCounts"]["immature_target_after_truth_contextEnd"]==5
    json.dumps(result,allow_nan=False)


def test_extension_artifacts_reject_partial_or_mixed_horizon_sets():
    from agri_data.prospective_two_crop_eval import build_assessment
    issue=dt.date(2026,10,7); truth=_truth(issue+dt.timedelta(days=10))
    for kind in ("partial","mixed"):
        path,data=_artifact("onion_extension",issue)
        doc=json.loads(data)
        if kind=="partial": doc["products"][0]["forecasts"].pop()
        else: doc["products"][0]["forecasts"][0]["horizon"]=1
        with pytest.raises(ValueError,match="exactly horizons"):
            build_assessment([(path,json.dumps(doc).encode())],truth,"8"*64,{})


def test_extension_followup_state_must_reference_extension_first_issue():
    from agri_data.prospective_two_crop_eval import build_assessment
    first_issue=dt.date(2026,10,7)
    base_path,base_bytes=_artifact("onion",first_issue)
    ext_path,ext_bytes=_artifact("onion_extension",first_issue)
    ext_hash=__import__("hashlib").sha256(ext_bytes).hexdigest()
    follow=_artifact("onion_extension",first_issue+dt.timedelta(days=1),onion_reference=(ext_path,ext_hash),onion_fit_origin=first_issue)
    result=build_assessment([(base_path,base_bytes),(ext_path,ext_bytes),follow],_truth(dt.date(2026,10,20)),"7"*64,{})
    assert sum(x["product"]=="onion" for x in result["comparisons"])==8
    assert sum(x["product"]=="onion_extension" for x in result["comparisons"])==32
    wrong=_artifact("onion_extension",first_issue+dt.timedelta(days=1),onion_reference=(base_path,__import__("hashlib").sha256(base_bytes).hexdigest()),onion_fit_origin=first_issue)
    with pytest.raises(ValueError,match="different horizon family"):
        build_assessment([(base_path,base_bytes),wrong],_truth(dt.date(2026,10,20)),"6"*64,{})


def test_separate_holm_adjustment_families_do_not_pool_onion_horizons(monkeypatch):
    from agri_data import direct_horizon as dh
    from agri_data.prospective_two_crop_eval import build_assessment
    monkeypatch.setattr(dh,"_dm",lambda diff,bw:{"status":"tested","n":len(diff),"meanLossDifference":0.0,"pValue":0.01})
    issue=dt.date(2026,10,7)
    files=[_artifact("onion",issue),_artifact("onion_extension",issue),_artifact("garlic",issue)]
    result=build_assessment(files,_truth(dt.date(2026,12,31)),"5"*64,{})
    maxima={crop:max(r["dm"]["9"]["holmPValue"] for r in result["comparisons"] if r["product"]==crop) for crop in ("onion","onion_extension","garlic")}
    assert maxima==pytest.approx({"onion":0.08,"onion_extension":0.32,"garlic":0.40})


def test_combined_full_horizon_macro_waits_for_30_mature_pairs_per_distance():
    from agri_data.prospective_two_crop_eval import build_assessment
    issue=dt.date(2026,10,7)
    files=[_artifact("onion",issue),_artifact("onion_extension",issue)]
    result=build_assessment(files,_truth(issue+dt.timedelta(days=5)),"4"*64,{})
    macro=result["macroMetrics"]["onionFullHorizonSameIssueDay"]
    assert macro["sameIssueDayCount"]==1
    assert macro["eligible"] is False
    assert macro["equalHorizonMape"] is None
    assert macro["under10PointEstimate"] is None
    assert "at_least_30" in macro["reason"]


def test_full_horizon_macro_excludes_mature_issue_with_missing_positive_truth():
    issue=dt.date(2026,10,7)
    files=[_artifact("onion",issue),_artifact("onion_extension",issue)]
    omitted=issue+dt.timedelta(days=6)
    result=build_assessment(files,_truth(issue+dt.timedelta(days=10),omit={omitted}),"3"*64,{})
    macro=result["macroMetrics"]["onionFullHorizonSameIssueDay"]
    assert macro["sameIssueDayCount"]==1
    assert macro["cohortIssueDateCount"]==0
    assert macro["sameIssueDates"]==[]
    assert macro["excludedIssueDates"][issue.isoformat()]==["completed_scope_without_positive_target_unevaluable"]
    assert macro["pairedCountsByHorizon"]=={str(h):0 for h in range(1,11)}
    assert result["reasonCounts"]["completed_scope_without_positive_target_unevaluable"]==1


def test_calendar_maturity_cohort_keeps_missing_target_and_excludes_immature_issue():
    mature=dt.date(2026,10,7); immature=dt.date(2026,10,8)
    files=[_artifact("onion",issue) for issue in (mature,immature)]
    files += [_artifact("onion_extension",issue) for issue in (mature,immature)]
    omitted=mature+dt.timedelta(days=5)
    result=build_assessment(files,_truth_for_issues((mature,immature),mature+dt.timedelta(days=10),{omitted}),"a"*64,{})
    old=result["macroMetrics"]["onionFullHorizonSameIssueDay"]
    assert old["cohortIssueDateCount"]==0
    cohort=result["descriptiveMaturityCohort"]["onion"]
    assert cohort["matureIssueDates"]==[mature.isoformat()]
    assert cohort["immatureIssueCount"]==1
    assert cohort["perHorizon"]["5"]["absentPositiveTargetCount"]==1
    assert cohort["perHorizon"]["5"]["pairedCount"]==0
    assert cohort["perHorizon"]["4"]["pairedCount"]==1
    assert cohort["perHorizon"]["4"]["methods"]["candidate"]["n"]==cohort["perHorizon"]["4"]["methods"]["directRidgePrediction"]["n"]==1
    assert cohort["perHorizon"]["4"]["candidateMinusComparator"]["persistencePrediction"]["mae"] is not None
    assert cohort["perHorizon"]["5"]["heldForecastCount"]==0
    assert cohort["equalHorizon10"]["supported"] is False
    assert cohort["equalHorizon10"]["methods"]["candidate"]["mae"] is None


def test_maturity_equal_horizon_aggregate_uses_shared_pairs_and_thirty_pair_gate():
    # Two-day issue spacing makes a missing truth date affect only one parity
    # of horizons, so the mature issue counts are deliberately unequal.
    issues=[dt.date(2026,10,7)+dt.timedelta(days=2*i) for i in range(31)]
    files=[_artifact("garlic",issue) for issue in issues]
    omitted=issues[0]+dt.timedelta(days=1)
    truth=_truth_for_issues(issues,issues[-1]+dt.timedelta(days=10),{omitted})
    result=build_assessment(files,truth,"b"*64,{})
    cohort=result["descriptiveMaturityCohort"]["garlic"]
    counts=[cohort["perHorizon"][str(h)]["pairedCount"] for h in range(1,11)]
    assert len(set(counts))==2 and min(counts)==30 and max(counts)==31
    for h in range(1,11):
        row=cohort["perHorizon"][str(h)]
        assert row["methods"]["candidate"]["n"]==row["methods"]["ridgePrediction"]["n"]==row["methods"]["persistencePrediction"]["n"]==row["pairedCount"]
        assert row["coverageRate"]==pytest.approx(row["pairedCount"]/31)
    eq=cohort["equalHorizon10"]
    assert eq["supported"] is True
    assert eq["methods"]["candidate"]["mae"]==pytest.approx(sum(cohort["perHorizon"][str(h)]["methods"]["candidate"]["mae"] for h in range(1,11))/10)
    assert eq["candidateMinusComparator"]["ridgePrediction"]["mae"]==pytest.approx(eq["methods"]["candidate"]["mae"]-eq["methods"]["ridgePrediction"]["mae"])


def test_maturity_equal_horizon_aggregate_is_null_below_thirty_pairs():
    issue=dt.date(2026,10,7)
    result=build_assessment([_artifact("garlic",issue)],_truth(issue+dt.timedelta(days=10)),"c"*64,{})
    eq=result["descriptiveMaturityCohort"]["garlic"]["equalHorizon10"]
    assert eq["supported"] is False
    assert eq["methods"]["candidate"]["mape"] is None
    assert "30_same_pair" in eq["reason"]
