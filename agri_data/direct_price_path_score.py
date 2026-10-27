"""Integrity-first retrospective scoring for DIRECT_PRICE_PATH_ATTENTION_V1.

The scorer only reads sealed, answer-free forecast artifacts. It never fits a
model or changes the frozen per-crop U/M selection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import chronos2_market_context as design
from . import full_history_return_eval as price_helper
from . import equal_info_online_score as stats
from . import research_progress_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/direct-price-path-attention-v1-20261014"
PHASES = design.PHASES
PRODUCTS = design.PRODUCTS
HORIZONS = design.HORIZONS
BANDWIDTHS = (9, 28)
POLICIES = ("B", "U", "M", "N", "H", "H_U", "H_M")
TRANSITION_PINS = {
    "transitionId": "nested-forecast-manifest-with-hashed-csv-companion-v1",
    "configFileSha256": "236003f2e844e30eaece7a30123e56151c84fcb695e406164d15b222c7d7af93",
    "configInternalSha256": "ed6f49c7ecd0331e1cfef137c9f3c453d1f1d5a2361fd4dfdc43f3cba383aa12",
    "originalMainPath": "agri_data/direct_price_path.py",
    "originalMainSha256": "0a69eea3f55b72574d869e6d17b563b332d18a3cf12d5dd2f4bb41da1a932401",
    "policyFileSha256": "dc42a386878360cde117795393478b0e836b9f538749acafabc08967db1c26f2",
    "policyInternalSha256": "e0cc3d73d2592bf05cce65b4b2e63bf12abb5c182bacbc45435a667d3ef4ca64",
    "developmentFileSha256": "89e9839d11f6e23636e7b92b6d188d8ae8ec166b8ec48e221bf5fd3a378d350b",
    "developmentInternalSha256": "8298477ac10b43097f635acecb3a1e35482fff065ff5fd3caad0b5e95c539d64",
    "trainingLedgerFileSha256": "4b1272de58c3a27af3c019e2f789bcb7c17d9762cfe1af5ca18b365574e01616",
    "completedFitCount": 104,
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    raise ValueError(f"invalid date {value!r}")


def _finite_positive(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _json_value(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _key(row: Mapping[str, Any]) -> tuple[str, dt.date, dt.date, int, str]:
    return (str(row["product"]), _date(row["origin"]), _date(row["target"]),
            int(row["horizon"]), str(row["target_version"]))


def _row_digest(rows: list[dict[str, Any]]) -> str:
    ordered = sorted((_json_value(r) for r in rows),
                     key=lambda r: (r["product"], r["origin"], r["target"], r["horizon"]))
    return sha256(canonical(ordered))


def _key_digest(rows: list[dict[str, Any]]) -> str:
    keys = sorted([[r["product"], _date(r["origin"]).isoformat(),
                    _date(r["target"]).isoformat(), int(r["horizon"]),
                    r["target_version"]] for r in rows])
    return sha256(canonical(keys))


def _output(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = row.get(name)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing output {name}")
    status = result.get("status")
    pred = _finite_positive(result.get("prediction"))
    if status not in ("issued", "fallback", "not_issued"):
        raise ValueError(f"invalid {name} status")
    if (status in ("issued", "fallback")) != (pred is not None):
        raise ValueError(f"{name} status/prediction mismatch")
    return result


def _raw_kind(value: float) -> str:
    if math.isnan(value): return "nan"
    if value == math.inf: return "positive_infinity"
    if value == -math.inf: return "negative_infinity"
    if value > 0: return "issued_positive"
    if value == 0: return "zero"
    return "negative_finite"


def _validate_outputs(row: Mapping[str, Any]) -> None:
    outs = {n: _output(row, n) for n in POLICIES}
    for name in ("U", "M", "N"):
        o = outs[name]
        raw_status = o.get("rawStatus", o.get("status"))
        if raw_status not in ("issued", "not_issued"):
            raise ValueError(f"invalid {name} rawStatus")
        raw_value=o.get("rawPrediction")
        raw = _finite_positive(raw_value)
        raw_kind=o.get("rawNumericKind")
        if raw_kind not in ("issued_positive","zero","negative_finite","positive_infinity","negative_infinity","nan","no_prediction"):
            raise ValueError(f"invalid {name} rawNumericKind")
        if raw_value is not None:
            try: numeric=float(raw_value)
            except (TypeError,ValueError,OverflowError): raise ValueError(f"invalid {name} rawPrediction")
            if not math.isfinite(numeric) or _raw_kind(numeric)!=raw_kind:
                raise ValueError(f"{name} rawNumericKind disagrees with its finite rawPrediction")
        elif raw_kind in ("issued_positive","zero","negative_finite"):
            raise ValueError(f"{name} finite rawNumericKind lacks a rawPrediction")
        # A model may return a non-positive or non-finite raw value. Such a
        # value remains preserved in rawPrediction and is not a valid issue.
        if raw_status == "issued" and raw is None:
            raise ValueError(f"{name} issued without a positive finite raw prediction")
        bpred=_finite_positive(outs["B"].get("prediction"))
        pred=_finite_positive(o.get("prediction"))
        if o.get("status")=="issued":
            if raw_status!="issued" or raw_kind!="issued_positive" or o.get("rawReason") is not None or pred is None or pred!=raw:
                raise ValueError(f"{name} issued prediction does not exactly preserve its raw model output")
        elif o.get("status")=="fallback":
            if raw_status!="not_issued" or bpred is None or pred!=bpred or o.get("fallbackTo")!="B" or not o.get("reason") or o.get("rawReason")!=o.get("reason"):
                raise ValueError(f"{name} fallback must preserve its raw failure and copy B with reason")
        elif o.get("status")=="not_issued":
            if pred is not None or raw_status!="not_issued" or bpred is not None or not o.get("reason") or o.get("rawReason")!=o.get("reason"):
                raise ValueError(f"{name} failed to use available B fallback")
    if row.get("selectedInput") not in ("U", "M"):
        raise ValueError("selectedInput must be U or M")
    chosen=outs[row["selectedInput"]]
    if any(outs["N"].get(k)!=chosen.get(k) for k in ("status","prediction","rawPrediction","rawNumericKind","rawStatus","rawReason","fallbackTo","reason")):
        raise ValueError("N does not mirror the frozen selected input")
    if bool(row.get("neuralUsed")) != (outs["N"].get("rawStatus")=="issued"):
        raise ValueError("neuralUsed disagrees with the selected candidate issue status")
    b = _finite_positive(outs["B"].get("prediction"))
    for hname, nname in (("H", "N"), ("H_U", "U"), ("H_M", "M")):
        hybrid = outs[hname]
        n = _finite_positive(outs[nname].get("prediction"))
        candidate_status=outs[nname].get("status")
        if b is not None and candidate_status=="issued" and n is not None:
            expected = .5 * b + .5 * n
            got = _finite_positive(hybrid.get("prediction"))
            if got is None or not math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{hname} violates the frozen 50/50 blend")
            if hybrid.get("status") != "issued":
                raise ValueError(f"{hname} should be issued when both components issue")
            if (_finite_positive(hybrid.get("rawPrediction"))!=got or hybrid.get("rawStatus")!="issued"
                    or hybrid.get("rawNumericKind")!="issued_positive" or hybrid.get("reason") is not None):
                raise ValueError(f"{hname} raw output/status do not preserve its blended prediction")
        elif b is not None:
            if hybrid.get("status") != "fallback" or not math.isclose(
                    float(hybrid["prediction"]), b, rel_tol=0, abs_tol=0):
                raise ValueError(f"{hname} did not fall back exactly to B")
            if hybrid.get("fallbackTo") != "B" or not hybrid.get("reason"):
                raise ValueError(f"{hname} fallback provenance missing")
            if any(hybrid.get(k)!=outs[nname].get(k) for k in ("rawPrediction","rawNumericKind","rawStatus","reason")):
                raise ValueError(f"{hname} did not preserve underlying neural failure evidence")
            if hybrid.get("reason")!=outs[nname].get("rawReason"):
                raise ValueError(f"{hname} fallback reason differs from raw failure reason")
        elif hybrid.get("status") != "not_issued":
            raise ValueError(f"{hname} issued without baseline B")


def validate_phase_rows(phase: str, rows: list[dict[str, Any]],
                        baseline: Mapping[tuple[str, dt.date, int], Mapping[str, Any]],
                        target_version: str, selected_by_product: Mapping[str,str] | None = None) -> None:
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    start, end, origin_start = PHASES[phase]
    expected = {(p, _date(b["origin"]), _date(b["target"]), int(h))
                for (p, _target, h), b in baseline.items()
                if start <= _date(b["target"]) <= end and _date(b["origin"]) >= origin_start}
    if len(rows) != len(expected):
        raise ValueError(f"{phase} forecast row count does not match exact B issue grid")
    seen = set()
    for r in rows:
        p, origin, target, h, version = _key(r)
        ident = (p, origin, target, h)
        if ident not in expected or ident in seen:
            raise ValueError(f"unexpected or duplicate forecast key {ident}")
        if version != target_version:
            raise ValueError("forecast target_version differs from pinned target snapshot")
        if selected_by_product is not None and r.get("selectedInput") != selected_by_product.get(p):
            raise ValueError("row selectedInput differs from frozen product policy")
        if target < start or target > end or origin < origin_start or origin != target - dt.timedelta(days=h):
            raise ValueError("forecast key outside frozen phase or invalid origin")
        b = baseline.get((p, target, h))
        if b is None or _date(b["origin"]) != origin:
            raise ValueError("forecast rows do not match pinned B key/origin")
        _validate_outputs(r)
        bvalue = _finite_positive(b.get("E106_RHS"))
        out_b = _finite_positive(_output(r, "B").get("prediction"))
        if bvalue != out_b:
            raise ValueError("forecast B differs from saved E106-RHS prediction")
        if _output(r,"B").get("sourceStatus") != b.get("rhsStatus"):
            raise ValueError("forecast B sourceStatus differs from saved E106-RHS status")
        seen.add(ident)
    if seen != expected:
        raise ValueError("forecast rows omit one or more keys from the exact B grid")


def _days(start: dt.date, end: dt.date):
    for i in range((end - start).days + 1):
        yield start + dt.timedelta(days=i)


def _metrics(rows: list[dict[str, Any]], actuals: Mapping[tuple[str, dt.date], float],
             a: str, b: str) -> dict[str, Any]:
    pairs = []
    for r in rows:
        actual = actuals.get((r["product"], _date(r["target"])))
        pa = _finite_positive(_output(r, a).get("prediction"))
        pb = _finite_positive(_output(r, b).get("prediction"))
        if actual is not None and pa is not None and pb is not None:
            pairs.append((r, actual, pa, pb))
    if not pairs:
        return {"n": 0, "maeA": None, "maeB": None, "mapeA": None, "mapeB": None}
    ea = [abs(pa - actual) for _, actual, pa, _ in pairs]
    eb = [abs(pb - actual) for _, actual, _, pb in pairs]
    return {"n": len(pairs), "maeA": math.fsum(ea) / len(ea),
            "maeB": math.fsum(eb) / len(eb),
            "mapeA": 100 * math.fsum(e / pair[1] for e, pair in zip(ea, pairs)) / len(ea),
            "mapeB": 100 * math.fsum(e / pair[1] for e, pair in zip(eb, pairs)) / len(eb)}


def _comparison(rows, actuals, a, b, calendar_start=None, calendar_end=None, inference=True, bootstrap_enabled=True,
                include_pair_rows=True):
    result = {"overall": _metrics(rows, actuals, a, b), "cells": [],
              "perProductHorizonMonth": [], "pairedTargets": [], "dm": [], "decomposition": {}}
    diffs = {"ae": {}, "ape": {}}; pair_records=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            cell = [r for r in rows if r["product"] == p and int(r["horizon"]) == h]
            met = _metrics(cell, actuals, a, b)
            result["cells"].append({"product": p, "horizon": h, **met})
            for r in cell:
                day = _date(r["target"]); actual = actuals.get((p, day))
                pa = _finite_positive(_output(r, a).get("prediction")); pb = _finite_positive(_output(r, b).get("prediction"))
                if actual is None or pa is None or pb is None:
                    continue
                ae = abs(pa-actual)-abs(pb-actual)
                diffs["ae"][(p,h,day)] = ae
                diffs["ape"][(p,h,day)] = ae/actual
                pair_record={"product":p,"horizon":h,"month":day.strftime("%Y-%m"),
                    "target":day.isoformat(),"actual":actual,"aPrediction":pa,"bPrediction":pb,
                    "aAe":abs(pa-actual),"bAe":abs(pb-actual),"aApe":abs(pa-actual)/actual,"bApe":abs(pb-actual)/actual}
                pair_records.append(pair_record)
                if include_pair_rows:
                    result["pairedTargets"].append(pair_record)
    monthly={}
    for r in pair_records:
        key=(r["product"],r["horizon"],r["month"])
        bucket=monthly.setdefault(key,{"n":0,"aAe":0.,"bAe":0.,"aApe":0.,"bApe":0.})
        bucket["n"]+=1
        for k in ("aAe","bAe","aApe","bApe"): bucket[k]+=r[k]
    result["perProductHorizonMonth"]=[{"product":p,"horizon":h,"month":m,"n":v["n"],
        "aMae":v["aAe"]/v["n"],"bMae":v["bAe"]/v["n"],
        "aMapePct":100*v["aApe"]/v["n"],"bMapePct":100*v["bApe"]/v["n"]}
        for (p,h,m),v in sorted(monthly.items())]
    supported=[r for r in result["cells"] if r["n"]]
    result["equalCellMetrics"]={"cellsIncluded":len(supported),"expectedCells":len(PRODUCTS)*len(HORIZONS),
        **{key:(math.fsum(r[key] for r in supported)/len(supported) if supported else None)
           for key in ("maeA","maeB","mapeA","mapeB")}}
    cell_counts={}
    for r in pair_records:
        key=(r["product"],r["horizon"]);cell_counts[key]=cell_counts.get(key,0)+1
    expected_cells={(p,h) for p in PRODUCTS for h in HORIZONS}
    months=sorted({d.strftime("%Y-%m") for d in _days(calendar_start,calendar_end)}) if calendar_start and calendar_end else sorted({r["month"] for r in pair_records})
    by_cell_month={}
    for p in PRODUCTS:
        for h in HORIZONS:
            n=cell_counts.get((p,h),0)
            for month in months:
                by_cell_month[(p,h,month)]={"product":p,"horizon":h,"month":month,"n":0,
                    "grossWorseningMae":0.,"grossImprovementMae":0.,"netMaeContribution":0.,
                    "grossWorseningMapePctPoints":0.,"grossImprovementMapePctPoints":0.,"netMapePctPointContribution":0.}
    if len(expected_cells):
        for r in pair_records:
            cell=(r["product"],r["horizon"]); item=by_cell_month[(cell[0],cell[1],r["month"])]
            item["n"]+=1
            if cell_counts[cell]:
                w=1./(len(expected_cells)*cell_counts[cell])
                d_mae=r["aAe"]-r["bAe"];d_mape=100*(r["aApe"]-r["bApe"])
                item["grossWorseningMae"]+=max(0.,d_mae)*w
                item["grossImprovementMae"]+=max(0.,-d_mae)*w
                item["netMaeContribution"]+=d_mae*w
                item["grossWorseningMapePctPoints"]+=max(0.,d_mape)*w
                item["grossImprovementMapePctPoints"]+=max(0.,-d_mape)*w
                item["netMapePctPointContribution"]+=d_mape*w
    result["cellMonthContributions"]=[by_cell_month[k] for k in sorted(by_cell_month)]
    observed_mae=math.fsum(r["netMaeContribution"] for r in result["cellMonthContributions"])
    observed_mape=math.fsum(r["netMapePctPointContribution"] for r in result["cellMonthContributions"])
    eq=result.get("equalCellMetrics",{})
    supported_cells=eq.get("cellsIncluded")==eq.get("expectedCells")
    accounting=(supported_cells and math.isclose(observed_mae,eq["maeA"]-eq["maeB"],rel_tol=1e-10,abs_tol=1e-10)
                and math.isclose(observed_mape,eq["mapeA"]-eq["mapeB"],rel_tol=1e-10,abs_tol=1e-10))
    result["contributionAccounting"]={"supportedAll40Cells":supported_cells,"passed":accounting,
        "netMaeContributionSum":observed_mae,"equalCellMaeDifference":(eq.get("maeA")-eq.get("maeB")) if supported_cells else None,
        "netMapePctPointContributionSum":observed_mape,
        "equalCellMapePctPointDifference":(eq.get("mapeA")-eq.get("mapeB")) if supported_cells else None}
    if supported_cells and not accounting: raise ArithmeticError("gross gain/loss contribution accounting does not reconcile")
    if inference:
        # Exactly forty product/horizon cells per loss and bandwidth.
        for loss in ("ae", "ape"):
            for p in PRODUCTS:
                for h in HORIZONS:
                    series = {day:value for (pp,hh,day),value in diffs[loss].items() if pp==p and hh==h}
                    for bw in BANDWIDTHS:
                        result["dm"].append({"product":p,"horizon":h,"loss":loss,"bandwidth":bw,**stats._dm(series,bw)})
        for bw in BANDWIDTHS:
            family=[r for r in result["dm"] if r["bandwidth"]==bw]
            if len(family)!=80: raise ValueError("DM family must contain 80 tests")
            stats._holm(family)
        result["dmSummaryByBandwidth"]=[]
        for bw in BANDWIDTHS:
            family=[r for r in result["dm"] if r["bandwidth"]==bw]
            result["dmSummaryByBandwidth"].append({"bandwidth":bw,"familySize":len(family),
                "tested":sum(r["status"]=="tested" for r in family),
                "inferenceUnavailable":sum(r["status"]!="tested" for r in family),
                "holmSignificantImprovement":sum(r.get("holmSignificant") is True and r["meanLossDifference"]<0 for r in family),
                "holmSignificantWorsening":sum(r.get("holmSignificant") is True and r["meanLossDifference"]>0 for r in family),
                "testedInconclusive":sum(r["status"]=="tested" and not r.get("holmSignificant") for r in family)})
    boot=[]
    flat=[]
    for r in rows:
        actual=actuals.get((r["product"],_date(r["target"])))
        pa=_finite_positive(_output(r,a).get("prediction"));pb=_finite_positive(_output(r,b).get("prediction"))
        if actual and pa and pb:
            flat.append({"product":r["product"],"horizon":int(r["horizon"]),
                "target":_date(r["target"]).isoformat(),"actual":actual,"a":pa,"b":pb})
    if inference:
        if bootstrap_enabled:
            for metric in ("ae","ape"):
                for block in (14,28):
                    boot.append(diagnostic.circular_block_bootstrap(flat,a="a",b="b",metric=metric,
                        block=block,replicates=2000,seed=0,calendar_start=calendar_start,calendar_end=calendar_end))
        else:
            result["bootstrapStatus"]="not_requested_for_mechanism_comparison"
        result["bootstrap"] = boot
        for loss in ("ae","ape"):
            cell_coverage={(r["product"],r["horizon"]) for r in flat}
            expected_cells={(p,h) for p in PRODUCTS for h in HORIZONS}
            if cell_coverage != expected_cells:
                dec = {"n":len(flat),"meanDifference":None,"byTarget":[],"concentration":{},"leaveOut":{},
                       "status":"unavailable_incomplete_40_cell_pair_coverage"}
            else:
                dec = diagnostic.decompose_losses(flat,a="a",b="b",metric=loss,expected_cells={(p,h) for p in PRODUCTS for h in HORIZONS})
            result["decomposition"][loss]=dec
    else:
        result["inferenceStatus"]="descriptive_only_no_multiplicity_family"
    return result


def _operational_rows(rows, policy):
    """Apply frozen B fallback to direct U/M output for operational M-U."""
    copied=[]
    for row in rows:
        clone=dict(row); output=dict(row[policy]); pred=_finite_positive(output.get("prediction"))
        if pred is None:
            b=_finite_positive(_output(row,"B").get("prediction"))
            if b is not None:
                output.update(status="fallback",prediction=b,fallbackTo="B")
            else:
                output.update(status="not_issued",prediction=None)
        clone[policy]=output;copied.append(clone)
    return copied


def score_phase(phase: str, rows: list[dict[str, Any]],
                baseline: Mapping[tuple[str, dt.date, int], Mapping[str, Any]],
                prices: Any, target_version: str, selected_by_product: Mapping[str,str] | None = None) -> dict[str, Any]:
    validate_phase_rows(phase, rows, baseline, target_version, selected_by_product)
    start,end,origin_start=PHASES[phase]
    series=price_helper._series(prices)
    actuals={(p,d):v for p,ds in series.items() for d,v in ds.items() if start<=d<=end}
    positive={(p,d) for p,d in actuals}
    issued={name:sum(_finite_positive(_output(r,name).get("prediction")) is not None for r in rows) for name in POLICIES}
    neural={name:sum(_output(r,name).get("rawStatus",_output(r,name).get("status"))=="issued" for r in rows)
            for name in ("U","M")}
    neural["N"]=sum(bool(r.get("neuralUsed")) for r in rows)
    neural_by_product={p:sum(bool(r.get("neuralUsed")) for r in rows if r["product"]==p) for p in PRODUCTS}
    price_changes={name:sum(_finite_positive(_output(r,name).get("prediction")) is not None and _finite_positive(_output(r,name).get("prediction")) != _finite_positive(_output(r,"B").get("prediction")) for r in rows) for name in POLICIES if name!="B"}
    policy_metrics={name:_metrics(rows,actuals,name,"B") for name in POLICIES if name!="B"}
    policy_performance={}
    for name in POLICIES:
        if name=="B": continue
        detail=_comparison(rows,actuals,name,"B",start,end,inference=False,include_pair_rows=False)
        policy_performance[name]={k:detail[k] for k in ("overall","equalCellMetrics","cells","perProductHorizonMonth")}
    comparisons={}
    for a,b,label in (("H","B","H_minus_B"),("M","U","M_minus_U"),("H_U","B","H_U_minus_B"),("H_M","B","H_M_minus_B")):
        pairrows=rows
        if label=="M_minus_U":
            pairrows=_operational_rows(_operational_rows(rows,"M"),"U")
        comparisons[label]=_comparison(pairrows,actuals,a,b,start,end,inference=label in ("H_minus_B","M_minus_U"),bootstrap_enabled=label=="H_minus_B")
    # Common-neural M-U subset isolates model behavior from fallback policy.
    common=[]
    for r in rows:
        if (_output(r,"U").get("status")=="issued" and _output(r,"U").get("rawStatus")=="issued"
                and _output(r,"M").get("status")=="issued" and _output(r,"M").get("rawStatus")=="issued"):
            common.append(r)
    comparisons["M_minus_U_common_neural"]=_comparison(common,actuals,"M","U",start,end,inference=False)
    selected_use=[r for r in rows if bool(r.get("neuralUsed")) and _output(r,"N").get("status")=="issued"]
    selected_subset={}
    for candidate,label in (("N","N_vs_B"),("H","H_vs_B")):
        detail=_comparison(selected_use,actuals,candidate,"B",start,end,inference=False,include_pair_rows=False)
        selected_subset[label]={k:detail[k] for k in ("overall","equalCellMetrics","cells","perProductHorizonMonth")}
    selected_subset["coverage"]={"selectedNeuralUseRows":len(selected_use),
        "sameKeyBIssuedRows":sum(_finite_positive(_output(r,"B").get("prediction")) is not None for r in selected_use),
        "sameKeyPositiveActualPairs":sum((r["product"],_date(r["target"])) in actuals for r in selected_use),
        "NAndHSameKeyIssuedRows":sum(_finite_positive(_output(r,"N").get("prediction")) is not None and
            _finite_positive(_output(r,"H").get("prediction")) is not None and
            _finite_positive(_output(r,"B").get("prediction")) is not None for r in selected_use)}
    expected_targets={(p,d) for p in PRODUCTS for d in _days(start,end)}
    convexity=[]
    for label,hybrid,candidate in (("H_vs_B_N","H","N"),("H_U_vs_B_U","H_U","U"),("H_M_vs_B_M","H_M","M")):
        tested=violations=0; max_excess={"ae":0.,"ape":0.}
        for r in rows:
            b=_finite_positive(_output(r,"B").get("prediction")); n=_finite_positive(_output(r,candidate).get("prediction"))
            hpred=_finite_positive(_output(r,hybrid).get("prediction")); actual=actuals.get((r["product"],_date(r["target"])))
            if b is None or n is None or hpred is None: continue
            tested+=1
            if actual is None: continue
            for loss in ("ae","ape"):
                div=actual if loss=="ape" else 1.
                lhs=abs(hpred-actual)/div
                rhs=.5*(abs(b-actual)+abs(n-actual))/div
                excess=lhs-rhs
                max_excess[loss]=max(max_excess[loss],excess)
                if excess>1e-10*max(1.,rhs): violations+=1
        convexity.append({"comparison":label,"arithmeticIssuedPairs":tested,
            "positiveActualLossChecks":sum(1 for r in rows if actuals.get((r["product"],_date(r["target"]))) is not None and
                _finite_positive(_output(r,"B").get("prediction")) is not None and
                _finite_positive(_output(r,candidate).get("prediction")) is not None),
            "violations":violations,"maxLossExcess":max_excess,"tolerance":1e-10})
    target_coverage=[]
    fit_usage={}
    aux_by_product={p:{"rows":0,"activeRows":0,"activeAuxCountTotal":0,"countByActiveAuxCount":{}} for p in PRODUCTS}
    for r in sorted(rows,key=lambda x:(x["product"],_date(x["target"]),int(x["horizon"]))):
        day=_date(r["target"]); actual=actuals.get((r["product"],day))
        aux=r.get("activeAuxCount")
        if isinstance(aux,int) and aux>=0:
            group=aux_by_product[r["product"]];group["rows"]+=1;group["activeRows"]+=int(aux>0);group["activeAuxCountTotal"]+=aux
            k=str(aux);group["countByActiveAuxCount"][k]=group["countByActiveAuxCount"].get(k,0)+1
        fitkey=(r["product"],str(r.get("fitId")),str(r.get("fitMonth")))
        fg=fit_usage.setdefault(fitkey,{"product":fitkey[0],"fitId":r.get("fitId"),"fitMonth":r.get("fitMonth"),"rows":0,"activeAuxRows":0})
        fg["rows"]+=1;fg["activeAuxRows"]+=int(isinstance(aux,int) and aux>0)
        target_coverage.append({"product":r["product"],"origin":_date(r["origin"]).isoformat(),
            "target":day.isoformat(),"horizon":int(r["horizon"]),"positiveActual":actual is not None,
            "actual":actual,"selectedInput":r.get("selectedInput"),"fitId":r.get("fitId"),
            "fitMonth":r.get("fitMonth"),"activeAuxCount":aux,"neuralUsed":bool(r.get("neuralUsed")),
            "states":{p:{"status":_output(r,p).get("status"),
                "rawStatus":_output(r,p).get("rawStatus"),"issued":_finite_positive(_output(r,p).get("prediction")) is not None}
                for p in POLICIES},
            "outputEvidence":{p:_json_value(dict(_output(r,p))) for p in POLICIES}})
    return {"phase":phase,"targetWindow":{"start":start.isoformat(),"end":end.isoformat(),"originStart":origin_start.isoformat()},
        "grid":{"rows":len(rows),"uniqueKeys":len({_key(r) for r in rows}),"positiveActualProductTargets":len(positive & expected_targets),
                "positiveActualProductTargetHorizonPairs":sum((r["product"],_date(r["target"])) in actuals for r in rows),
                "positiveActualProductTargetsByProduct":{p:sum((p,d) in positive for d in _days(start,end)) for p in PRODUCTS}},
        "issuedRows":issued,"neuralUseRows":neural,"neuralUseRowsByProduct":neural_by_product,
        "neuralUseRates":{k:v/len(rows) if rows else 0. for k,v in neural.items()},
        "priceChangeRowsVsB":price_changes,"priceChangeRatesVsB":{k:v/len(rows) if rows else 0. for k,v in price_changes.items()},
        "policyMetricsVsB":policy_metrics,"policyPerformanceVsB":policy_performance,
        "selectedNeuralUseSubset":selected_subset,
        "activeAuxiliarySupportByProduct":{p:{**v,"activeRate":v["activeRows"]/v["rows"] if v["rows"] else 0.,
            "meanActiveAuxCount":v["activeAuxCountTotal"]/v["rows"] if v["rows"] else None} for p,v in aux_by_product.items()},
        "fitUseByProductFitIdMonth":list(fit_usage.values()),
        "selectedInputByProduct":{p:next(iter({r["selectedInput"] for r in rows if r["product"]==p})) for p in PRODUCTS},
        "comparisons":comparisons,"convexityChecks":convexity,
        "targetCoverageAndStates":target_coverage,"status":"complete"}


def _verify_ref(root: Path, ref: Mapping[str, Any], *, internal_key: str | None = None) -> Path:
    path=Path(ref["path"])
    if not path.is_absolute(): path=root/path
    raw=path.read_bytes()
    if sha256(raw)!=ref.get("sha256"): raise ValueError(f"file hash mismatch: {path}")
    if internal_key:
        doc=json.loads(raw); unsigned=dict(doc); supplied=unsigned.pop(internal_key,None)
        if supplied!=sha256(canonical(unsigned)): raise ValueError(f"internal hash mismatch: {path}")
    return path


def load_rows(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned=dict(manifest); supplied=unsigned.pop("manifestSha256",None)
    if supplied!=sha256(canonical(unsigned)): raise ValueError("forecast manifest self-hash mismatch")
    root=OUT
    art=manifest["forecastArtifact"]
    path=Path(art["path"]); path=path if path.is_absolute() else root/path
    if sha256(path.read_bytes())!=art["sha256"]: raise ValueError("forecast parquet hash mismatch")
    frame=pd.read_parquet(path,engine="pyarrow")
    rows=[_json_value(r) for r in frame.to_dict(orient="records")]
    if len(rows)!=art["rowCount"] or _row_digest(rows)!=art["rowsSha256"] or _key_digest(rows)!=art["keySetSha256"]:
        raise ValueError("forecast parquet content digest mismatch")
    return manifest,rows


def _ref_path(ref: Mapping[str,Any], experiment_root: Path) -> Path:
    path=Path(ref["path"])
    if path.is_absolute(): return path
    base=ref.get("pathBase")
    return (ROOT if base=="repository" else experiment_root)/path


def _checked_ref(ref: Mapping[str,Any], experiment_root: Path, *, file_hash_key: str,
                 internal_hash_key: str | None = None) -> tuple[Path,dict|None]:
    path=_ref_path(ref,experiment_root); raw=path.read_bytes()
    if sha256(raw)!=ref.get(file_hash_key): raise ValueError(f"referenced file hash mismatch: {path}")
    doc=None
    if internal_hash_key:
        doc=json.loads(raw); unsigned=dict(doc); got=unsigned.pop(internal_hash_key,None)
        if got!=sha256(canonical(unsigned)) or got!=ref.get(internal_hash_key):
            raise ValueError(f"referenced internal hash mismatch: {path}")
    return path,doc


def _validate_pinned_target_ref(ref: Mapping[str,Any]) -> None:
    pinned=price_helper.PRICES_SHA256
    if ref.get("sha256")!=pinned or ref.get("targetVersion")!=pinned:
        raise ValueError("target snapshot bytes and targetVersion must both match the pinned answer source")


def _hashed_json(path: Path, expected_file_sha: str, internal_key: str) -> dict:
    raw=path.read_bytes()
    if sha256(raw)!=expected_file_sha: raise ValueError(f"transition-pinned file hash mismatch: {path}")
    doc=json.loads(raw); unsigned=dict(doc); internal=unsigned.pop(internal_key,None)
    if internal!=sha256(canonical(unsigned)): raise ValueError(f"transition-pinned internal hash mismatch: {path}")
    return doc


def _validate_transition_refs(v_ref: Mapping[str,Any], t_ref: Mapping[str,Any],
                              source_code_files: Mapping[str,str], experiment_root: Path,
                              repository_root: Path=ROOT) -> dict:
    if v_ref!=t_ref:
        raise ValueError("V/T implementation transition references differ")
    if (v_ref.get("path")!="implementation-transition.json"
            or v_ref.get("pathBase")!="experimentRoot"):
        raise ValueError("implementation transition path is not the frozen experiment artifact")
    transition_path=experiment_root/"implementation-transition.json"
    raw=transition_path.read_bytes()
    if sha256(raw)!=v_ref.get("fileSha256"):
        raise ValueError("implementation transition file hash mismatch")
    transition=json.loads(raw); unsigned=dict(transition); self_hash=unsigned.pop("transitionSha256",None)
    if (self_hash!=sha256(canonical(unsigned)) or self_hash!=v_ref.get("transitionSha256")):
        raise ValueError("implementation transition self-hash mismatch")
    pins=TRANSITION_PINS
    if (transition.get("schemaVersion")!=1 or transition.get("experiment")!="DIRECT_PRICE_PATH_ATTENTION_V1"
            or transition.get("transitionId")!=pins["transitionId"]
            or transition.get("originalResolvedConfigPath")!="outputs/experiments/direct-price-path-attention-v1-20261014/resolved-config.json"
            or transition.get("originalResolvedConfigFileSha256")!=pins["configFileSha256"]
            or transition.get("originalConfigInternalSha256")!=pins["configInternalSha256"]
            or transition.get("originalMainPath")!=pins["originalMainPath"]
            or transition.get("originalMainSha256")!=pins["originalMainSha256"]
            or transition.get("originalTrainingLedgerFileSha256")!=pins["trainingLedgerFileSha256"]
            or transition.get("revisedMainPath")!="agri_data/direct_price_path_v2.py"
            or transition.get("dataSourcePath")!="agri_data/direct_price_path_data.py"):
        raise ValueError("implementation transition does not match the pinned original-to-V2 handoff")
    original_config_path=repository_root/transition["originalResolvedConfigPath"]
    config=_hashed_json(original_config_path,pins["configFileSha256"],"resolvedConfigSha256")
    if config.get("resolvedConfigSha256")!=pins["configInternalSha256"]:
        raise ValueError("transition original config internal hash is not pinned")
    config_sources=config.get("sourceCodeFiles")
    if not isinstance(config_sources,Mapping) or config_sources.get(pins["originalMainPath"])!=pins["originalMainSha256"]:
        raise ValueError("transition original config does not reference the pinned original producer")
    data_path=transition["dataSourcePath"]
    if transition.get("dataSourceSha256")!=config_sources.get(data_path):
        raise ValueError("transition data-source hash differs from the preserved resolved config")
    for path,expected in config_sources.items():
        if source_code_files.get(path)!=expected:
            raise ValueError(f"V/T manifest source map differs from preserved config for {path}")
    for path,expected in ((pins["originalMainPath"],pins["originalMainSha256"]),
                          (transition["revisedMainPath"],transition.get("revisedMainSha256")),
                          (data_path,transition.get("dataSourceSha256"))):
        if not isinstance(expected,str) or source_code_files.get(path)!=expected:
            raise ValueError(f"V/T manifest does not bind transition source {path}")
        if sha256((repository_root/path).read_bytes())!=expected:
            raise ValueError(f"transition source bytes changed: {path}")
    for phase,expected in design.BASELINE_SHA256.items():
        if transition.get("baselineManifestSha256",{}).get(phase)!=expected:
            raise ValueError(f"transition changed pinned {phase} baseline")
    preserved=transition.get("preservedArtifacts")
    if not isinstance(preserved,Mapping): raise ValueError("transition lacks preserved-artifact inventory")
    policy_path=experiment_root/"model_policy.json"
    development_path=experiment_root/"development-scores.json"
    policy=_hashed_json(policy_path,pins["policyFileSha256"],"modelPolicySha256")
    development=_hashed_json(development_path,pins["developmentFileSha256"],"developmentScoresSha256")
    if (policy.get("modelPolicySha256")!=pins["policyInternalSha256"]
            or development.get("developmentScoresSha256")!=pins["developmentInternalSha256"]
            or transition.get("modelPolicyFileSha256")!=pins["policyFileSha256"]
            or transition.get("developmentScoresFileSha256")!=pins["developmentFileSha256"]
            or preserved.get("resolvedConfigFileSha256")!=pins["configFileSha256"]
            or preserved.get("modelPolicyFileSha256")!=pins["policyFileSha256"]
            or preserved.get("modelPolicyInternalSha256")!=pins["policyInternalSha256"]
            or preserved.get("developmentScoresFileSha256")!=pins["developmentFileSha256"]
            or preserved.get("developmentScoresInternalSha256")!=pins["developmentInternalSha256"]
            or preserved.get("trainingLedgerFileSha256")!=pins["trainingLedgerFileSha256"]):
        raise ValueError("transition policy/development/config provenance differs from the approved frozen artifacts")
    fit_ids=transition.get("completedFitIds")
    inventory=transition.get("completedFitInventory")
    expected_count=pins["completedFitCount"]
    if (not isinstance(fit_ids,list) or fit_ids!=sorted(set(fit_ids)) or len(fit_ids)!=expected_count
            or not isinstance(inventory,list) or len(inventory)!=2*expected_count
            or preserved.get("completedFitCount")!=expected_count
            or preserved.get("completedFitIds")!=fit_ids):
        raise ValueError("transition does not contain all 104 preserved development fits")
    development_fits=development.get("fitArtifacts")
    if not isinstance(development_fits,list) or len(development_fits)!=expected_count:
        raise ValueError("pinned development scores do not contain the 104 preserved fit artifacts")
    development_by_id={}
    for fit in development_fits:
        fit_id=fit.get("fitId")
        if not isinstance(fit_id,str) or fit_id in development_by_id:
            raise ValueError("pinned development fit-artifact list has invalid or duplicate IDs")
        development_by_id[fit_id]=fit
    if sorted(development_by_id)!=fit_ids:
        raise ValueError("transition fit IDs differ from the byte-pinned development result")
    development_inventory=[]
    for fit_id in fit_ids:
        fit=development_by_id[fit_id]
        for artifact,path_key,sha_key in (("weightPath","weightPath","weightSha256"),
                                          ("checkpointPath","checkpointPath","checkpointSha256")):
            development_inventory.append({"fitId":fit_id,"artifact":artifact,
                "path":fit.get(path_key),"sha256":fit.get(sha_key)})
    if inventory!=development_inventory:
        raise ValueError("transition completed-fit inventory differs from pinned development fit artifacts")
    expected_inventory=[]
    seen_artifacts=set()
    for item in inventory:
        if (not isinstance(item,Mapping) or item.get("fitId") not in fit_ids
                or item.get("artifact") not in ("weightPath","checkpointPath")):
            raise ValueError("transition completed-fit inventory row is invalid")
        artifact_key=(item["fitId"],item["artifact"])
        if artifact_key in seen_artifacts: raise ValueError("transition fit inventory contains a duplicate artifact")
        seen_artifacts.add(artifact_key)
        rel=Path(item.get("path",""))
        if rel.is_absolute() or ".." in rel.parts or not isinstance(item.get("sha256"),str):
            raise ValueError("transition fit inventory contains an unsafe path or missing digest")
        file_path=experiment_root/rel
        if not file_path.is_file() or sha256(file_path.read_bytes())!=item["sha256"]:
            raise ValueError(f"preserved development fit artifact changed: {item['fitId']}/{item['artifact']}")
    for fit_id in fit_ids:
        for artifact in ("weightPath","checkpointPath"):
            if (fit_id,artifact) not in seen_artifacts:
                raise ValueError("transition inventory omits a model or checkpoint")
    expected_inventory=sorted(inventory,key=lambda x:(x["fitId"],0 if x["artifact"]=="weightPath" else 1))
    if inventory!=expected_inventory:
        raise ValueError("transition completed-fit inventory order is not canonical")
    inventory_sha=sha256(canonical(inventory))
    if (transition.get("completedFitInventorySha256")!=inventory_sha
            or preserved.get("completedFitInventorySha256")!=inventory_sha):
        raise ValueError("transition completed-fit inventory digest mismatch")
    return transition


def _inference_payload_digest(records: list[Mapping[str,Any]]) -> str:
    payload=[{"product":r["product"],"origin":r["origin"],"window":r["window"],
              "calendar":r["calendar"]} for r in records]
    return sha256(canonical(payload))


def _forecast_fit_rows_digest(rows: list[dict[str,Any]], fit: Mapping[str,Any]) -> str:
    input_type=str(fit["inputType"]); month=str(fit["fitMonth"]); product=str(fit["product"])
    payload=[]
    for row in rows:
        if row["product"]!=product or str(row["origin"])[:7]!=month: continue
        output=row[input_type]
        payload.append({"product":row["product"],"origin":row["origin"],"target":row["target"],
            "horizon":int(row["horizon"]),"inputType":input_type,
            "prediction":output.get("rawPrediction"),"status":output.get("rawStatus"),
            "numericKind":output.get("rawNumericKind"),"rawReason":output.get("rawReason")})
    payload.sort(key=lambda r:(r["product"],r["origin"],r["target"],r["horizon"]))
    return sha256(canonical(_json_value(payload)))


def _validate_replay_grid(phase: str, manifest: Mapping[str,Any], rows: list[dict[str,Any]],
                          fit_by_id: Mapping[str,Mapping[str,Any]], experiment_root: Path) -> None:
    replay=manifest.get("replayVerification")
    if (not isinstance(replay,Mapping) or replay.get("schemaVersion")!=1
            or replay.get("method")!="reload_saved_state_dict_then_infer_before_baseline_row_join"
            or not isinstance(replay.get("byFitId"),Mapping)
            or set(replay["byFitId"])!=set(fit_by_id)):
        raise ValueError(f"{phase} manifest lacks the complete saved-model replay verification grid")
    for fit_id,fit in fit_by_id.items():
        evidence=replay["byFitId"][fit_id]
        if not isinstance(evidence,Mapping) or evidence!=fit.get("replayVerification"):
            raise ValueError(f"{phase} replay record differs from fit-grid copy: {fit_id}")
        unsigned=dict(evidence); replay_sha=unsigned.pop("replaySha256",None)
        if replay_sha!=sha256(canonical(unsigned)):
            raise ValueError(f"{phase} replay self-hash mismatch: {fit_id}")
        if (evidence.get("fitId")!=fit_id or evidence.get("modelPath")!=fit.get("modelPath")
                or evidence.get("modelSha256")!=fit.get("modelSha256")
                or evidence.get("reloadedStateDict") is not True
                or evidence.get("stateAgreement") is not True
                or evidence.get("stateKeyAgreement") is not True
                or evidence.get("allRowsAgree") is not True
                or evidence.get("issueClassAgreement") is not True
                or evidence.get("predictionSource")!="reloaded_saved_state_dict_and_persisted_input_schema"
                or evidence.get("horizonCount")!=len(HORIZONS)
                or evidence.get("inferenceRowCount")!=evidence.get("rowPayloadCount")
                or int(evidence.get("rowPayloadCount",0))<=0
                or evidence.get("absoluteTolerance")!=1e-6
                or evidence.get("relativeTolerance")!=1e-6):
            raise ValueError(f"{phase} replay evidence does not prove reloaded-model agreement: {fit_id}")
        bound=evidence.get("absoluteTolerance")+evidence.get("relativeTolerance")*evidence.get("maxOriginalMagnitude",-1)
        if evidence.get("maxAbsoluteDifference",math.inf)>bound:
            raise ValueError(f"{phase} replay error exceeds its declared tolerance: {fit_id}")
        for key in ("rowPayloadFileSha256","rowPayloadSha256","originalPredictionSha256","predictionSha256",
                    "forecastRowsSha256"):
            value=evidence.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{phase} replay evidence has invalid {key}: {fit_id}")
        rel=f"{phase}/replay-inputs/{fit_id}.json"
        if evidence.get("rowPayloadPath")!=rel or evidence.get("rowPayloadPathBase")!="experimentRoot":
            raise ValueError(f"{phase} replay input path is not the pinned fit payload: {fit_id}")
        path=experiment_root/rel; raw=path.read_bytes()
        if sha256(raw)!=evidence.get("rowPayloadFileSha256"):
            raise ValueError(f"{phase} replay input file hash mismatch: {fit_id}")
        payload=json.loads(raw); records=payload.get("records")
        expected_schema={"contextDays":fit.get("contextDays"),"inputType":fit.get("inputType"),
                         "repair":fit.get("repairSetting"),"selectedInputChannels":fit.get("selectedChannels")}
        if (payload.get("fitId")!=fit_id or payload.get("modelSha256")!=fit.get("modelSha256")
                or payload.get("schema")!=expected_schema
                or payload.get("schemaSha256")!=sha256(canonical(expected_schema))
                or evidence.get("schemaSha256")!=payload.get("schemaSha256")
                or not isinstance(records,list) or len(records)!=evidence.get("rowPayloadCount")
                or payload.get("rowPayloadSha256")!=_inference_payload_digest(records)
                or payload.get("rowPayloadSha256")!=evidence.get("rowPayloadSha256")):
            raise ValueError(f"{phase} replay input payload does not match its hashed schema and records: {fit_id}")
        if evidence.get("forecastRowsSha256")!=_forecast_fit_rows_digest(rows,fit):
            raise ValueError(f"{phase} replay does not bind the emitted forecast rows: {fit_id}")


def _baseline_index(path: Path, phase: str) -> dict:
    from . import external_reference_v1 as ext
    doc=json.loads(path.read_text(encoding="utf-8"))
    return ext._baseline_index(doc,phase)


def _validate_fit_grid(phase: str, manifest: Mapping[str,Any], rows: list[dict[str,Any]], experiment_root: Path) -> None:
    fit_grid=manifest.get("fitGrid")
    if not isinstance(fit_grid,list): raise ValueError(f"{phase} manifest lacks a fit grid")
    months=sorted({_date(r["origin"]).strftime("%Y-%m") for r in rows})
    first_origins={}
    for row in rows:
        key=(row["product"],_date(row["origin"]).strftime("%Y-%m"))
        origin=_date(row["origin"])
        first_origins[key]=min(origin,first_origins.get(key,origin))
    expected={(f"{phase}-{month.replace('-','')}-{p}-{it}",p,it,month)
              for p in PRODUCTS for it in ("U","M") for month in months}
    seen={}
    for fit in fit_grid:
        fit_id=fit.get("fitId"); product=fit.get("product"); input_type=fit.get("inputType")
        month=str(fit.get("fitMonth"))[:7]
        ident=(fit_id,product,input_type,month)
        if ident not in expected or fit_id in seen:
            raise ValueError(f"{phase} fit grid contains an unexpected or duplicate fit: {ident}")
        if _date(fit.get("fitBoundary"))!=first_origins[(product,month)]:
            raise ValueError(f"{phase} fit boundary must equal the first eligible origin for {product}/{month}")
        if type(fit.get("fixedEpoch")) is not int or fit["fixedEpoch"]<1 or type(fit.get("bestEpoch")) is not int or fit["bestEpoch"]<1:
            raise ValueError(f"{phase} fit epoch/replay metadata is missing")
        if fit["fixedEpoch"]!=fit["bestEpoch"]:
            raise ValueError(f"{phase} fixed final epoch differs from model bestEpoch metadata")
        model_path=Path(fit.get("modelPath",""))
        expected_path=Path("models")/f"{fit_id}.pt"
        if model_path.is_absolute() or model_path!=expected_path:
            raise ValueError(f"{phase} model path does not match fit id")
        absolute=experiment_root/model_path
        if not absolute.is_file() or sha256(absolute.read_bytes())!=fit.get("modelSha256"):
            raise ValueError(f"{phase} saved model bytes differ from fit-grid digest: {fit_id}")
        checkpoint_path=Path(fit.get("checkpointPath",""))
        expected_checkpoint=Path("checkpoints")/f"{fit_id}.pt"
        if checkpoint_path.is_absolute() or checkpoint_path!=expected_checkpoint:
            raise ValueError(f"{phase} checkpoint path does not match fit id")
        checkpoint_path=experiment_root/checkpoint_path
        if not checkpoint_path.is_file():
            raise ValueError(f"{phase} checkpoint replay evidence is missing: {fit_id}")
        if sha256(checkpoint_path.read_bytes())!=fit.get("checkpointSha256"):
            raise ValueError(f"{phase} checkpoint bytes differ from fit-grid digest: {fit_id}")
        seen[fit_id]=fit
    if set(seen)!= {item[0] for item in expected}:
        raise ValueError(f"{phase} fit grid does not cover all 24 product/input/month fits")
    model_refs=manifest.get("modelsByFitId")
    if not isinstance(model_refs,Mapping) or set(model_refs)!=set(seen):
        raise ValueError(f"{phase} manifest model replay map does not match the 24-fit grid")
    for fit_id,fit in seen.items():
        ref=model_refs[fit_id]
        if (ref.get("pathBase")!="experimentRoot" or ref.get("path")!=fit["modelPath"]
                or ref.get("sha256")!=fit["modelSha256"]
                or ref.get("checkpointPath")!=fit["checkpointPath"]
                or ref.get("checkpointSha256")!=fit["checkpointSha256"]):
            raise ValueError(f"{phase} model replay reference differs from the fit grid: {fit_id}")
    fit_by_id=seen
    for row in rows:
        product=row["product"]; month=_date(row["origin"]).strftime("%Y-%m")
        if str(row.get("fitMonth"))[:7]!=month:
            raise ValueError("forecast row fitMonth does not match the origin month")
        selected=row.get("selectedInput")
        if row.get("fitId")!=f"{phase}-{month.replace('-','')}-{product}-{selected}":
            raise ValueError("forecast row does not reference its frozen selected monthly fit")
        expected_input_ids={it:f"{phase}-{month.replace('-','')}-{product}-{it}" for it in ("U","M")}
        evidence=row.get("outputEvidence")
        if not isinstance(evidence,Mapping): raise ValueError("forecast row is missing model output evidence")
        for role,it in (("U","U"),("M","M"),("N",selected)):
            fit_id=expected_input_ids[it]; fit=fit_by_id[fit_id]; ev=evidence.get(role)
            if not isinstance(ev,Mapping) or ev.get("fitId")!=fit_id or ev.get("modelSha256")!=fit.get("modelSha256"):
                raise ValueError(f"forecast row {role} model provenance does not match fit grid")
        boundary=_date(fit_by_id[row["fitId"]]["fitBoundary"])
        if boundary>_date(row["origin"]):
            raise ValueError("forecast uses a monthly model fit after its origin")
        aux=row.get("activeAuxCount")
        if type(aux) is not int or aux<0:
            raise ValueError("forecast row has invalid active auxiliary count")
    _validate_replay_grid(phase,manifest,rows,fit_by_id,experiment_root)


def score_saved_run(v_manifest_path: Path, t_manifest_path: Path) -> dict:
    """Verify pinned provenance and score the two sealed phase artifacts."""
    vm,vrows=load_rows(v_manifest_path); tm,trows=load_rows(t_manifest_path)
    for phase,manifest in (("V",vm),("T",tm)):
        if manifest.get("experiment")!="DIRECT_PRICE_PATH_ATTENTION_V1" or manifest.get("stage")!=phase or manifest.get("targetValuesAttached") is not False:
            raise ValueError(f"{phase} manifest is not an answer-free stage artifact")
        if manifest.get("temporalBoundary")!={
                "monthlyFitBoundary":"first_eligible_pinned_B_origin_for_product_and_month",
                "trainingLabels":"representative_targets_strictly_before_each_fitBoundary",
                "inferenceFeatures":"source_values_through_origin_minus_one_calendar_day",
                "targetLabelsAttachedToForecastRows":False}:
            raise ValueError(f"{phase} manifest does not bind the frozen origin-only fit boundary")
        if manifest.get("sourceCodeFiles") is None or manifest.get("executionIdentity") is None:
            raise ValueError("manifest lacks source-code or execution provenance")
        artifact_path=manifest.get("forecastArtifact",{}).get("path")
        if artifact_path!=f"{phase}/forecasts.parquet":
            raise ValueError(f"{phase} forecast artifact path differs from the frozen phase location")
        coverage=manifest.get("coverage",{})
        phase_rows=vrows if phase=="V" else trows
        if (coverage.get("rowCount")!=len(phase_rows)
                or coverage.get("uniqueKeyCount")!=len(phase_rows)
                or coverage.get("expectedKeyCount")!=len(phase_rows)):
            raise ValueError(f"{phase} manifest coverage counts differ from forecast payload")
        identity=manifest["executionIdentity"]
        if manifest.get("executionIdentitySha256")!=sha256(canonical(identity)):
            raise ValueError("execution identity hash mismatch")
        for source_path,expected_sha in manifest["sourceCodeFiles"].items():
            full=ROOT/source_path
            if sha256(full.read_bytes())!=expected_sha:
                raise ValueError(f"source code hash mismatch: {source_path}")
        _validate_fit_grid(phase,manifest,vrows if phase=="V" else trows,OUT)
    root=OUT
    if (vm["targetSnapshot"]!=tm["targetSnapshot"] or vm["modelPolicy"]!=tm["modelPolicy"]
            or vm["sourceCache"]!=tm["sourceCache"] or vm["sourceCodeFiles"]!=tm["sourceCodeFiles"]):
        raise ValueError("V/T target snapshot, frozen policy, or source cache differs")
    if not isinstance(vm.get("implementationTransition"),Mapping):
        raise ValueError("V2 manifests lack the required implementation transition reference")
    _validate_transition_refs(vm["implementationTransition"],tm.get("implementationTransition",{}),
                              vm["sourceCodeFiles"],root,ROOT)
    if vm["targetSnapshot"].get("pathBase")!="repository" or any(
            m.get("baselineSource",{}).get("pathBase")!="repository" for m in (vm,tm)):
        raise ValueError("target and baseline references must be repository-relative")
    if vm["modelPolicy"].get("pathBase")!="experimentRoot" or vm["sourceCache"].get("pathBase")!="experimentRoot":
        raise ValueError("policy and source-cache references must be experiment-root-relative")
    policy_ref=vm["modelPolicy"]
    _,policy_doc=_checked_ref(policy_ref,root,file_hash_key="fileSha256",internal_hash_key="modelPolicySha256")
    if _ref_path(policy_ref,root).resolve()!=(OUT/"model_policy.json").resolve():
        raise ValueError("policy path is not the pinned frozen experiment policy")
    if policy_doc.get("selectedInputByProduct")!=policy_ref.get("selectedInputByProduct"):
        raise ValueError("manifest selected-input map differs from the hashed frozen policy file")
    cache_ref=vm["sourceCache"]
    cache_path,_=_checked_ref(cache_ref,root,file_hash_key="fileSha256",internal_hash_key="sourceCacheSha256")
    from . import direct_price_path_data as source_data
    if cache_path.resolve()!=source_data.SOURCE_CACHE_PATH.resolve():
        raise ValueError("source cache path is not the pinned experiment cache")
    target_ref=vm["targetSnapshot"]
    target_path,_=_checked_ref(target_ref,root,file_hash_key="sha256")
    if target_path.resolve()!=price_helper.PRICES_PATH.resolve():
        raise ValueError("target snapshot path is not the pinned historical answer source")
    _validate_pinned_target_ref(target_ref)
    target_version=target_ref.get("targetVersion")
    prices=json.loads(target_path.read_text(encoding="utf-8"))
    baselines={}
    for phase,manifest,expected_sha in (("V",vm,design.BASELINE_SHA256["V"]),("T",tm,design.BASELINE_SHA256["T"])):
        ref=manifest["baselineSource"]
        if ref.get("sha256")!=expected_sha:
            raise ValueError(f"{phase} baseline hash is not the pinned RHS source")
        path,_=_checked_ref(ref,root,file_hash_key="sha256")
        if path.resolve()!=design.BASELINE_PATHS[phase].resolve():
            raise ValueError(f"{phase} baseline path is not the pinned RHS manifest")
        baseline_doc=json.loads(path.read_text(encoding="utf-8"))
        if baseline_doc.get("protocolSha256")!=ref.get("protocolSha256"):
            raise ValueError("baseline protocol provenance mismatch")
        baselines[phase]=_baseline_index(path,phase)
    report=score_documents(vm,vrows,tm,trows,baselines,prices,target_version)
    for phase,manifest in (("V",vm),("T",tm)):
        if manifest["coverage"]["expectedKeyCount"]!=report[phase]["grid"]["rows"]:
            raise ValueError(f"{phase} exact baseline key count differs from phase coverage claim")
    report["V"]["fitGrid"] = vm.get("fitGrid")
    report["T"]["fitGrid"] = tm.get("fitGrid")
    report["provenance"]={"VManifestSha256":vm["manifestSha256"],"TManifestSha256":tm["manifestSha256"],
        "targetSnapshotSha256":target_ref["sha256"],"modelPolicySha256":policy_ref["modelPolicySha256"],
        "sourceCacheSha256":cache_ref["sourceCacheSha256"],
        "baselineManifestSha256":{"V":vm["baselineSource"]["sha256"],"T":tm["baselineSource"]["sha256"]},
        "sourceCodeFiles":vm["sourceCodeFiles"],
        "executionIdentitySha256":{"V":vm["executionIdentitySha256"],"T":tm["executionIdentitySha256"]},
        "targetVersion":target_version}
    return report


def score_documents(v_manifest, v_rows, t_manifest, t_rows, baselines, prices, target_version):
    """Pure scoring entrypoint for the already verified V/T documents."""
    if v_manifest["modelPolicy"] != t_manifest["modelPolicy"]:
        raise ValueError("V/T model policy differs")
    policy=v_manifest["modelPolicy"]
    if not isinstance(policy,Mapping) or not isinstance(policy.get("selectedInputByProduct"),Mapping):
        raise ValueError("frozen model policy lacks selectedInputByProduct")
    selected=policy["selectedInputByProduct"]
    if set(selected)!=set(PRODUCTS) or any(v not in ("U","M") for v in selected.values()):
        raise ValueError("frozen per-product U/M policy is incomplete or invalid")
    actual_selected={p:next(iter({r["selectedInput"] for r in v_rows if r["product"]==p})) for p in PRODUCTS}
    if selected!=actual_selected:
        raise ValueError("V rows differ from frozen model policy")
    v=score_phase("V",v_rows,baselines["V"],prices,target_version,selected)
    t=score_phase("T",t_rows,baselines["T"],prices,target_version,selected)
    v_h=v["comparisons"]["H_minus_B"]["equalCellMetrics"]
    t_h=t["comparisons"]["H_minus_B"]["equalCellMetrics"]
    v_coverage=(v["issuedRows"]["H"]==v["issuedRows"]["B"])
    t_coverage=(t["issuedRows"]["H"]==t["issuedRows"]["B"])
    v["coveragePreservedForH"] = v_coverage
    t["coveragePreservedForH"] = t_coverage
    v_ok=v_coverage and v_h["cellsIncluded"]==v_h["expectedCells"] and v_h["mapeA"]<v_h["mapeB"] and v_h["maeA"]<=v_h["maeB"]
    t_ok=t_coverage and t_h["cellsIncluded"]==t_h["expectedCells"] and t_h["mapeA"]<t_h["mapeB"] and t_h["maeA"]<=t_h["maeB"]
    incomplete_reasons=[f"{phase}:{p}:no_selected_neural_issues"
        for phase,result in (("V",v),("T",t)) for p in PRODUCTS
        if result["neuralUseRowsByProduct"].get(p,0)==0]
    if incomplete_reasons: status="blocked_integrity"
    elif v_ok and t_ok: status="completed_price_improvement" if t_h["mapeB"]-t_h["mapeA"]>=.5 else "completed_small_improvement"
    else: status="completed_goal_not_met"
    return {"schemaVersion":1,"experiment":"DIRECT_PRICE_PATH_ATTENTION_V1","status":status,
        "targetVersion":target_version,"V":v,"T":t,
        "statusReason":incomplete_reasons if incomplete_reasons else None,
        "interpretation":{"retrospectiveOnly":True,"TScoredRegardlessOfV":True,
            "MminusUOperationalWithFallback":True,"commonNeuralSubsetReported":True,
            "nonsignificantIsNotEquivalence":True}}


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--v-manifest",type=Path,default=OUT/"V/forecast-manifest.json"); ap.add_argument("--t-manifest",type=Path,default=OUT/"T/forecast-manifest.json"); ap.add_argument("--output",type=Path,default=OUT/"score-report.json")
    args=ap.parse_args(argv)
    report=score_saved_run(args.v_manifest,args.t_manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as f:f.write(canonical(report))
    return 0


if __name__=="__main__": main()
