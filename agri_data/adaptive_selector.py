"""Retrospective per-origin selector using only past common forecast errors."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, experiments

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/predictions.json"
INPUT_SHA256 = "c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
MODELS = ("baseline", "ridge", "hist_gb")
MODEL_PRIORITY = {model: i for i, model in enumerate(MODELS)}
EVAL_START, EVAL_END = dt.date(2024, 1, 1), dt.date(2026, 9, 27)
LOOKBACK_DAYS, LEAKAGE_GAP_DAYS, MIN_HISTORY, MIN_TEST_PAIRS = 730, 2, 30, 30
PRIMARY_LAG, SENSITIVITY_LAG, ALPHA = 9, 28, .05
LOSSES = ("absolute_error", "absolute_percentage_error")
PROTOCOL = {
    "id": "adaptive-selector-product-horizon-rolling-history-2026-10-06",
    "inputSha256": INPUT_SHA256,
    "window": "For product x horizon x origin, use previous 730 calendar days ending at origin-2 inclusive; targets must have actual and all three predictions.",
    "historyMinimum": MIN_HISTORY,
    "candidateRule": "select Ridge or HistGradientBoosting only if mean APE is at least 2% relatively below baseline and mean AE is no greater than baseline",
    "ranking": "mean APE, then mean AE, then baseline/ridge/hist_gb deterministic priority",
    "fallback": "baseline for fewer than 30 history dates, no qualifying model, or missing selected current prediction",
    "evaluation": {"targets": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "sample": "same baseline origin-target predictions with an actual target", "pairedLosses": list(LOSSES),
        "DM": "two-sided normal Diebold-Mariano with calendar Bartlett HAC", "primaryLagDays": PRIMARY_LAG,
        "sensitivityLagDays": SENSITIVITY_LAG, "minimumPairs": MIN_TEST_PAIRS,
        "holmFamily": "5 products x 10 horizons x 2 losses (fixed 100) separately by bandwidth", "alpha": ALPHA},
    "limitations": "retrospective selection on current revised actuals; evaluation period was previously inspected; exploratory only; no model promotion",
}


def _date(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return out


def _prediction_map(rows: Any, label: str) -> dict[tuple[dt.date, dt.date, int], float]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} predictions must be an array")
    result = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} prediction {i} must be an object")
        target = _date(row.get("target"), f"{label} target")
        origin = _date(row.get("origin"), f"{label} origin")
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise ValueError(f"{label} horizon must be 1..10")
        if target - origin != dt.timedelta(days=horizon):
            raise ValueError(f"{label} origin/target/horizon are inconsistent")
        pred = _positive(row.get("predicted"), f"{label} predicted")
        key = (origin, target, horizon)
        if key in result:
            raise ValueError(f"duplicate {label} forecast {key}")
        result[key] = pred
    return result


def _validate(source: Any):
    if not isinstance(source, dict) or source.get("schemaVersion") != 2:
        raise ValueError("input must be dashboard schemaVersion 2 prediction export")
    rows = source.get("products")
    if not isinstance(rows, list) or len(rows) != len(PRODUCTS):
        raise ValueError("input must contain exactly five products")
    products = {}
    for product in rows:
        if not isinstance(product, dict) or product.get("id") not in PRODUCTS or product["id"] in products:
            raise ValueError("invalid or duplicate product")
        pid = product["id"]
        actuals = {}
        observations = product.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"{pid} observations must be an array")
        for row in observations:
            day = _date(row.get("date"), f"{pid} actual date")
            value = _positive(row.get("actual"), f"{pid} actual")
            if day in actuals:
                raise ValueError(f"duplicate {pid} actual date {day}")
            actuals[day] = value
        baseline = _prediction_map(product.get("predictions"), f"{pid}/baseline")
        runs = product.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"{pid} runs must be an array")
        model_maps = {"baseline": baseline}
        seen = set()
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"{pid} run must be an object")
            model, cadence = run.get("modelId"), run.get("cadence")
            if cadence != "daily" or model not in ("ridge", "hist_gb"):
                continue
            if model in seen:
                raise ValueError(f"duplicate {pid}/{model}/daily run")
            seen.add(model)
            model_maps[model] = _prediction_map(run.get("predictions"), f"{pid}/{model}/daily")
        if set(model_maps) != set(MODELS):
            raise ValueError(f"{pid} requires baseline, ridge daily, and hist_gb daily predictions")
        products[pid] = {"actuals": actuals, "forecasts": model_maps}
    if set(products) != set(PRODUCTS):
        raise ValueError("input must contain every product")
    return products


def _index_forecasts(forecasts):
    """Build model -> horizon -> target maps once; selectors then do O(1) lookup."""
    indexed={model:defaultdict(dict) for model in MODELS}
    for model, rows in forecasts.items():
        for (origin,target,horizon),predicted in rows.items():
            indexed[model][horizon][target]=predicted
    return {model:{horizon:dict(by_target) for horizon,by_target in horizons.items()}
            for model,horizons in indexed.items()}


def _means(preds, baseline, actuals, targets):
    rows = [(preds[d], baseline[d], actuals[d]) for d in targets]
    n = len(rows)
    if not n:
        return {"n": 0, "mae": None, "mape": None}
    return {"n": n,
        "mae": sum(abs(p-a) for p, _, a in rows)/n,
        "mape": sum(abs(p-a)/a*100 for p, _, a in rows)/n,
        "baselineMae": sum(abs(b-a) for _, b, a in rows)/n,
        "baselineMape": sum(abs(b-a)/a*100 for _, b, a in rows)/n}


def _select_one(pid, horizon, origin, actuals, forecast_index):
    baseline = forecast_index["baseline"].get(horizon,{})
    ridge = forecast_index["ridge"].get(horizon,{})
    hist = forecast_index["hist_gb"].get(horizon,{})
    cutoff = origin - dt.timedelta(days=LEAKAGE_GAP_DAYS)
    window_start = cutoff - dt.timedelta(days=LOOKBACK_DAYS-1)
    expected = {d for d in actuals if window_start <= d <= cutoff}
    common = sorted(d for d in expected if d in baseline and d in ridge and d in hist)
    counts = {"eligibleActualDates": len(expected), "commonThreeModelDates": len(common),
              "missingAnyForecastDates": len(expected)-len(common)}
    base = _means(baseline, baseline, actuals, common)
    candidate_stats = {m: _means(forecast, baseline, actuals, common) for m, forecast in (("ridge", ridge), ("hist_gb", hist))}
    qualified=[]
    if len(common) >= MIN_HISTORY and base["mape"] is not None and base["mape"] > 0:
        threshold=base["mape"]*.98
        for model, stats in candidate_stats.items():
            # Inclusive 2% gate; narrowly absorb binary float error at the boundary.
            at_least_two_percent_below=(stats["mape"] < threshold or
                math.isclose(stats["mape"],threshold,rel_tol=1e-12,abs_tol=1e-12))
            if at_least_two_percent_below and stats["mae"] <= base["mae"]:
                qualified.append((stats["mape"], stats["mae"], MODEL_PRIORITY[model], model))
    selected = min(qualified)[-1] if qualified else "baseline"
    why = ("selected_qualified_candidate" if qualified else "insufficient_common_history" if len(common)<MIN_HISTORY
           else "baseline_ape_zero_no_relative_improvement" if base["mape"] == 0 else "no_candidate_met_gate")
    return {"productId":pid,"horizon":horizon,"origin":origin.isoformat(),"selectedModel":selected,
        "historyNomineeModel":selected,"appliedModel":None,"fallbackApplied":None,"fallbackReason":None,
        "selectionReason":why,"historyWindowStart":window_start.isoformat(),"historyCutoff":cutoff.isoformat(),
        **counts,"historyMeanLosses":{"baseline":base,**candidate_stats}}


def _dm(rows, metric, lag):
    n=len(rows)
    differences=[r["selected"][metric]-r["baseline"][metric] for r in rows]
    difference=sum(differences)/n if n else None
    if n < MIN_TEST_PAIRS:
        return {"n":n,"status":"unavailable_insufficient_pairs","meanLossDifference":difference,
                "pValue":None,"calendarHacMaxLagDays":lag}
    if differences and max(differences)-min(differences) <= 1e-12*max(1.0,max(abs(x) for x in differences)):
        return {"n":n,"status":"unavailable_nonpositive_variance","meanLossDifference":difference,
                "pValue":None,"calendarHacMaxLagDays":lag}
    candidate=[{"target":r["target"],"horizon":r["horizon"],"predicted":r["selectedPred"]} for r in rows]
    baseline=[{"target":r["target"],"horizon":r["horizon"],"predicted":r["baselinePred"]} for r in rows]
    actuals={(r["target"],r["horizon"]):r["actual"] for r in rows}
    loss="absolute_error" if metric=="mae" else "absolute_percentage_error"
    test=experiments._dm(candidate,baseline,actuals,rows[0]["horizon"],loss,max_lag=lag)
    if test.get("status")!="available":
        return {"n":n,"status":test.get("status","unavailable"),"meanLossDifference":difference,
                "pValue":None,"calendarHacMaxLagDays":lag}
    return {"n":n,"status":"available","meanLossDifference":difference,
            "statistic":test["statistic"],"pValue":test["pValue"],"calendarHacMaxLagDays":lag}


def _holm(records, lag_key):
    ordered=sorted(records,key=lambda r:(r[lag_key]["pValue"] if r[lag_key]["pValue"] is not None else 1.0,r["productId"],r["horizon"],r["loss"]))
    running=0.0; m=len(records)
    for i,rec in enumerate(ordered):
        test=rec[lag_key];p=test["pValue"] if test["pValue"] is not None else 1.0
        running=max(running,min(1.0,(m-i)*p))
        test["holmAdjustedPValue"]=running
        test["holmSignificantAt005"]=test["status"]=="available" and running < ALPHA


def _conclusion(test_ae, test_ape):
    tests=(test_ae,test_ape)
    if any(t["status"]!="available" for t in tests):
        return "inference_unavailable"
    improved=[t["meanLossDifference"]<0 and t["holmSignificantAt005"] for t in tests]
    worsened=[t["meanLossDifference"]>0 and t["holmSignificantAt005"] for t in tests]
    if all(improved): return "improved"
    if all(worsened): return "worsened"
    if any(improved) or any(worsened): return "mixed_evidence"
    return "no_evidence"


def analyze(source: dict[str, Any], source_sha256: str | None = None):
    products=_validate(source)
    selections={}
    forecast_indexes={pid:_index_forecasts(p["forecasts"]) for pid,p in products.items()}
    for pid,p in products.items():
        actuals=p["actuals"]; forecasts=p["forecasts"]; forecast_index=forecast_indexes[pid]
        origins={(h,origin) for origin,target,h in forecasts["baseline"] if EVAL_START <= target <= EVAL_END}
        for h,origin in origins:
            selections[(pid,h,origin)]=_select_one(pid,h,origin,actuals,forecast_index)
    by_product_h=defaultdict(list)
    evaluation_coverage={}
    evaluation_metrics={}
    for pid,p in products.items():
        actuals=p["actuals"]; forecasts=p["forecasts"]
        forecast_index=forecast_indexes[pid]
        for h in HORIZONS:
            base_rows=[];selected_missing=0;fallback=Counter();baseline_rows_in_window=0;baseline_without_actual=0
            for target,basepred in forecast_index["baseline"].get(h,{}).items():
                if not (EVAL_START<=target<=EVAL_END):
                    continue
                origin=target-dt.timedelta(days=h)
                baseline_rows_in_window+=1
                selection=selections[(pid,h,origin)]
                nominee=selection["historyNomineeModel"]
                applied=nominee
                selected=forecast_index[applied].get(h,{}).get(target)
                fallback_reason=None
                if selected is None and applied!="baseline":
                    fallback_reason="selected_model_prediction_missing"
                    applied="baseline";selected=basepred
                selection["appliedModel"]=applied
                selection["fallbackApplied"]=fallback_reason is not None
                selection["fallbackReason"]=fallback_reason
                if target not in actuals:
                    baseline_without_actual+=1
                    continue
                if fallback_reason is not None:
                    selected_missing+=1;fallback[nominee]+=1
                base_rows.append({"productId":pid,"horizon":h,"origin":origin.isoformat(),"target":target.isoformat(),
                    "actual":actuals[target],"baselinePred":basepred,"selectedPred":selected,
                    "selectedModel":applied,"selected":selection})
            base_rows.sort(key=lambda r:(r["target"],r["origin"]))
            selected_counts=Counter(r["selectedModel"] for r in base_rows)
            evaluation_coverage[f"{pid}|{h}"]={"baselineRowsInWindow":baseline_rows_in_window,
                "baselineRowsWithActual":len(base_rows),"baselineRowsWithoutActual":baseline_without_actual,"selectedRows":len(base_rows),
                "selectedModelMissingFallbacks":selected_missing,"fallbackByModel":dict(fallback),
                "selectedModelCounts":dict(selected_counts),"evaluationTargetStart":min((r["target"] for r in base_rows),default=None),
                "evaluationTargetEnd":max((r["target"] for r in base_rows),default=None)}
            selected_losses=[];base_losses=[]
            for r in base_rows:
                actual=r["actual"];sp=r["selectedPred"];bp=r["baselinePred"]
                selected_losses.append({"target":r["target"],"horizon":h,"mae":abs(sp-actual),"mape":abs(sp-actual)/actual*100})
                base_losses.append({"target":r["target"],"horizon":h,"mae":abs(bp-actual),"mape":abs(bp-actual)/actual*100})
            means={"selected":{"mae":sum(x["mae"] for x in selected_losses)/len(selected_losses) if selected_losses else None,
                               "mape":sum(x["mape"] for x in selected_losses)/len(selected_losses) if selected_losses else None},
                   "baseline":{"mae":sum(x["mae"] for x in base_losses)/len(base_losses) if base_losses else None,
                               "mape":sum(x["mape"] for x in base_losses)/len(base_losses) if base_losses else None}}
            evaluation_metrics[(pid,h)] = means
            for loss in LOSSES:
                metric="mae" if loss=="absolute_error" else "mape"
                base_rows_metric=[]
                for row in base_rows:
                    selected_error=abs(row["selectedPred"]-row["actual"])
                    baseline_error=abs(row["baselinePred"]-row["actual"])
                    if metric=="mape":
                        selected_error=selected_error/row["actual"]*100
                        baseline_error=baseline_error/row["actual"]*100
                    base_rows_metric.append({**row,"selected":{metric:selected_error},"baseline":{metric:baseline_error}})
                primary=_dm(base_rows_metric,metric,PRIMARY_LAG)
                sensitivity=_dm(base_rows_metric,metric,SENSITIVITY_LAG)
                by_product_h[(pid,h)].append({"productId":pid,"horizon":h,"loss":loss,"primaryHAC9":primary,"sensitivityHAC28":sensitivity})
            by_product_h[(pid,h)].sort(key=lambda x:x["loss"])
    primary_tests=[test for vals in by_product_h.values() for test in vals]
    for lag_key in ("primaryHAC9","sensitivityHAC28"):
        _holm(primary_tests,lag_key)
    report_records=[]
    for (pid,h),tests in by_product_h.items():
        # Metrics can be reconstructed from paired test mean differences and attached coverage below.
        coverage=evaluation_coverage[f"{pid}|{h}"]
        paired={t["loss"]:{"primaryHAC9":t["primaryHAC9"],"sensitivityHAC28":t["sensitivityHAC28"]} for t in tests}
        conclusion=_conclusion(paired["absolute_error"]["primaryHAC9"],paired["absolute_percentage_error"]["primaryHAC9"])
        report_records.append({"productId":pid,"horizon":h,"coverage":coverage,
            "metrics":evaluation_metrics[(pid,h)],"conclusion":conclusion,"pairedTests":paired})
    selection_rows=sorted(selections.values(),key=lambda x:(x["productId"],x["horizon"],x["origin"]))
    nominee_counts=defaultdict(Counter);applied_counts=defaultdict(Counter)
    for row in selection_rows:
        nominee_counts[(row["productId"],row["horizon"])][row["historyNomineeModel"]]+=1
        if row["appliedModel"] is not None:
            applied_counts[(row["productId"],row["horizon"])][row["appliedModel"]]+=1
    selection_summary={f"{p}|{h}":{"historyNominee":dict(nominee_counts[(p,h)]),
        "appliedModel":dict(applied_counts[(p,h)])} for p in PRODUCTS for h in HORIZONS}
    return {"schemaVersion":1,"status":"complete","exploratoryPostHoc":True,
        "protocol":{**PROTOCOL,"sha256":hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()},
        "inputSha256":source_sha256,"selectionSummary":selection_summary,
        "originSelections":selection_rows,"evaluation":report_records,
        "multipleTesting":{"familySize":100,"primaryLagDays":PRIMARY_LAG,"sensitivityLagDays":SENSITIVITY_LAG,
            "tests":[{k:v for k,v in t.items() if k!="selected"} for t in primary_tests]}}


def run(input_path=INPUT_PATH):
    path=Path(input_path).resolve();raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=INPUT_SHA256:raise ValueError(f"input SHA-256 mismatch: expected {INPUT_SHA256}, found {sha}")
    source=json.loads(raw)
    return analyze(source,sha)


def _atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_report(report,json_path,markdown_path):
    _atomic(json_path,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 품목·예측거리별 과거오차 선택 실험","",
        "현재 수정본의 과거 예측과 정답으로 원점별 모델을 선택한 사후 탐색 결과입니다. 이는 실제 최초 공개시점의 재현이나 미확인 미래 성능, 자동 모델 승격을 의미하지 않습니다.","",
        "| 품목 | 거리 | 날짜 수 | 기준 MAE | 선택 MAE | 기준 MAPE% | 선택 MAPE% | 선택기 vs 기준선 |","|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in report["evaluation"]:
        tests=r["pairedTests"]; ae=tests["absolute_error"]["primaryHAC9"]; ape=tests["absolute_percentage_error"]["primaryHAC9"]
        metrics=r["metrics"]
        bmae, smae=metrics["baseline"]["mae"],metrics["selected"]["mae"]
        bmape, smape=metrics["baseline"]["mape"],metrics["selected"]["mape"]
        f=lambda value: f"{value:.3f}" if value is not None else "—"
        ptxt=lambda test: f"{test['holmAdjustedPValue']:.3g}" if test.get("holmAdjustedPValue") is not None else "—"
        lines.append(f"| {r['productId']} | {r['horizon']} | {r['coverage']['baselineRowsWithActual']} | {f(bmae)} | {f(smae)} | {f(bmape)} | {f(smape)} | {r['conclusion']} (p={ptxt(ae)}/{ptxt(ape)}) |")
    lines += ["","선택은 각 원점 기준 이틀 전까지의 직전 730일에서 세 예측이 모두 있는 정답 날짜만 사용하며, 30일 미만이면 기준선입니다. 모델은 평균 APE가 기준선보다 상대 2% 이상 낮고 평균 AE가 기준선 이하일 때만 선택됩니다. 평가 표본은 2024-01-01~2026-09-27 기준선 예측·정답의 동일 원점/목표일입니다.","", "주 분석은 달력간격 Bartlett HAC9, 민감도는 HAC28입니다. 각 대역폭에서 5품목×10거리×2손실의 고정 Holm100을 적용하며, 추론 불가 검정은 p=1로 가족에 남깁니다. 개선은 두 손실 모두 평균차가 음수이고 주 분석 보정 p<0.05일 때만 인정합니다."]
    _atomic(markdown_path,"\n".join(lines)+"\n")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",default=str(INPUT_PATH));ap.add_argument("--output",required=True);ap.add_argument("--report")
    args=ap.parse_args(argv);src=Path(args.input).resolve();out=Path(args.output).resolve();md=Path(args.report).resolve() if args.report else out.with_suffix(".md")
    if out==md or out==src or md==src:raise ValueError("output paths must be distinct and must not overwrite input")
    write_report(run(src),out,md);return 0

if __name__=="__main__":raise SystemExit(main())
