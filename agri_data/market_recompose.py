"""Frozen market-wise Ridge forecast and ex-ante share recomposition experiment."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, market_mix, modeling, recent_window as rw

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_PREDICTIONS = dh.INPUT_PREDICTIONS
INPUT_RIDGE_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
INPUT_MARKETS = tuple(dashboard.MARKETS)
GARAK, GANGSEO = INPUT_MARKETS
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
EXPECTED = {
    "pricesSha256": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
    "predictionsSha256": "c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113",
    "directRidgeReportSha256": "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67",
}
EVAL_START = dt.date(2023, 1, 1)
EVAL_END = dt.date(2026, 9, 27)
MIN_TRAIN_DAYS = 90
MAX_AGE = 14
MIN_SHARE_DAYS = 5
SHARE_WINDOW = 28
PROTOCOL = {
    "id": "market-wise-direct-ridge-exante-share-2026-10-07",
    "fit": "per product, market and horizon; Ridge(alpha=10); first eligible origin per month; targets strictly before fit origin; >=90 distinct target days",
    "marketOriginEligibility": "market has positive price and net weight at last observation <= origin; observation age <=14 calendar days",
    "share": "median Garak net-weight share among positive two-market co-trade dates in origin-inclusive prior 28 calendar days; >=5 dates; newest co-trade <=14 days old",
    "evaluation": "targets 2023-01-01..2026-09-27; origins >=2023-01-01; horizons 1..10 calendar days",
    "targetEligibility": "every frozen target observation with an issued pinned direct Ridge prediction; target-date market activity is diagnostic only",
    "fallback": "either market forecast/share invalid => pinned direct Ridge forecast; garlic and dried red pepper always pinned direct Ridge",
    "comparisonFamily": "5 products x 10 horizons x 2 comparators x 2 losses = 200 per HAC bandwidth; two-sided DM, calendar Bartlett HAC9/HAC28, Holm alpha .05; unavailable inference p=1",
    "limitations": "retrospective revised observations; historical first-publication time unknown",
}


def _series(markets):
    out = {}
    for p in PRODUCTS:
        out[p] = {}
        for m in INPUT_MARKETS:
            out[p][m] = {}
            for day, row in markets.get(p, {}).get(m, {}).items():
                d = day if isinstance(day, dt.date) else dt.date.fromisoformat(day)
                try:
                    price, weight = float(row["price"]), float(row["weight"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(price) and math.isfinite(weight) and price > 0 and weight > 0:
                    out[p][m][d] = (price, weight)
    return out


def _market_forecasts(markets, evaluation_calendar):
    """Fit monthly expanding per-market Ridge models and return origin-keyed forecasts/audits."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    data = _series(markets)
    forecasts, audits = {}, {}
    for p in PRODUCTS[:3]:
        for market in INPUT_MARKETS:
            obs = data[p][market]
            prices = dict(sorted((d, (v[0], v[1])) for d, v in obs.items()))
            cache = {}
            for h in HORIZONS:
                samples = []
                for target, (actual, _) in sorted(prices.items()):
                    origin = target - dt.timedelta(days=h)
                    if origin < dt.date(2014, 1, 1):
                        continue
                    feat = modeling._feature_row(prices, origin, target, h, cache)
                    if feat is None:
                        continue
                    source = feat[0]
                    samples.append((target, origin, feat[1], math.log(actual / prices[source][0])))
                months = {}
                for target in evaluation_calendar.get((p, h), ()):
                    origin = target - dt.timedelta(days=h)
                    if EVAL_START <= target <= EVAL_END and origin >= EVAL_START:
                        feature = modeling._feature_row(prices, origin, target, h, cache)
                        if feature is not None:
                            months.setdefault((origin.year, origin.month), []).append((target, origin, feature[1]))
                for ym, rows in sorted(months.items()):
                    fit_origin = min(origin for _, origin, _ in rows)
                    train = [r for r in samples if r[0] < fit_origin]
                    n = len({r[0] for r in train})
                    key = f"{p}/{market}/{h}/{ym[0]:04d}-{ym[1]:02d}"
                    if n < MIN_TRAIN_DAYS:
                        audits[key] = {"status": "insufficient_training", "fitOrigin": fit_origin.isoformat(), "distinctTargets": n}
                        continue
                    valid = [(target, origin, vector, None) for target, origin, vector in rows]
                    try:
                        with threadpool_limits(limits=1):
                            model = modeling._new_estimator("ridge")
                            model.fit(np.asarray([r[2] for r in train]), np.asarray([r[3] for r in train]))
                            predicted = model.predict(np.asarray([r[2] for r in valid])) if valid else []
                        for r, delta in zip(valid, predicted):
                            src = modeling._feature_row(prices, r[1], r[0], h, cache)[0]
                            value, _ = modeling._inverse_log_ratio(prices[src][0], float(delta))
                            forecasts[(p, market, h, r[1])] = value
                        audits[key] = {"status": "fitted", "fitOrigin": fit_origin.isoformat(), "distinctTargets": n, "issued": len(valid)}
                    except Exception as exc:
                        audits[key] = {"status": "fit_failed", "fitOrigin": fit_origin.isoformat(), "distinctTargets": n, "errorType": type(exc).__name__, "error": str(exc)[:200]}
    return forecasts, audits, data


def _share(data, p, origin):
    start = origin - dt.timedelta(days=SHARE_WINDOW - 1)
    common = sorted(d for d in data[p][INPUT_MARKETS[0]] if start <= d <= origin and d in data[p][INPUT_MARKETS[1]])
    common = [d for d in common if all(data[p][m][d][0] > 0 and data[p][m][d][1] > 0 for m in INPUT_MARKETS)]
    last = common[-1] if common else None
    share = None
    if len(common) >= MIN_SHARE_DAYS and last is not None and (origin - last).days <= MAX_AGE:
        weights = sorted(data[p][INPUT_MARKETS[0]][d][1] / sum(data[p][m][d][1] for m in INPUT_MARKETS) for d in common)
        mid = len(weights) // 2
        share = weights[mid] if len(weights) % 2 else (weights[mid - 1] + weights[mid]) / 2
    return share, common, last


def _decomposition(row):
    """Symmetric audit identity, using actual target values only after issuance."""
    pg, pj = row["actualGarakPrice"], row["actualGangseoPrice"]
    wg, wj = row["actualGarakWeight"], row["actualGangseoWeight"]
    w = wg / (wg + wj)
    wh, pgh, pjh = row["share"], row["predictedGarak"], row["predictedGangseo"]
    price = wh * (pgh - pg) + (1 - wh) * (pjh - pj)
    share = (wh - w) * (pg - pj)
    residual = (wh * pgh + (1 - wh) * pjh) - (w * pg + (1 - w) * pj)
    return {"actualGarakShare": w, "actualComposite": w * pg + (1 - w) * pj,
            "predictedComposite": wh * pgh + (1 - wh) * pjh,
            "priceContribution": price, "shareContribution": share,
            "identityResidual": residual - price - share}


def build(raw, markets, forecasts, ridge_report, *, hashes=None):
    """Build full per-origin evidence and fixed paired statistical comparisons."""
    data = _series(markets)
    actuals = dh._observations(raw)
    ridge, ridge_actual = rw._direct_ridge_reference(ridge_report)
    for (p, h, target), expected_actual in ridge_actual.items():
        if p in actuals and target in actuals[p] and not math.isclose(actuals[p][target][0], expected_actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"pinned Ridge actual differs from frozen target: {p}/{h}/{target}")
    # Frozen persistence arm is carried in predictions.json by run(); tests may inject it.
    persistence = forecasts.get("__persistence__", {}) if isinstance(forecasts, dict) and "__persistence__" in forecasts else {}
    market_preds = forecasts.get("__market__", forecasts) if isinstance(forecasts, dict) else {}
    evidence, candidate = [], {}
    for p in PRODUCTS:
        for h in HORIZONS:
            calendar = sorted(t for pp, hh, t in ridge if pp == p and hh == h and t in actuals[p])
            for target in calendar:
                actual, _ = actuals[p][target]
                origin = target - dt.timedelta(days=h)
                if not (EVAL_START <= target <= EVAL_END and origin >= EVAL_START):
                    continue
                share, common, last_common = _share(data, p, origin)
                row = {"product": p, "horizon": h, "origin": origin.isoformat(), "target": target.isoformat(),
                       "actual": actual, "coTradeDays28": len(common), "lastCoTradeDate": last_common.isoformat() if last_common else None,
                       "share": share}
                reasons = []
                if p not in PRODUCTS[:3]: reasons.append("always_fallback_product")
                for m in INPUT_MARKETS:
                    prev = [d for d in data[p][m] if d <= origin]
                    last = max(prev) if prev else None
                    row[f"last{m.title()}Observation"] = last.isoformat() if last else None
                    row[f"{m}AgeDays"] = (origin-last).days if last else None
                    predicted = market_preds.get((p, m, h, origin))
                    label = "Garak" if m == GARAK else "Gangseo"
                    row[f"predicted{label}"] = predicted
                    if last is None or (origin-last).days > MAX_AGE: reasons.append(f"{label.lower()}_source_missing_or_stale")
                    try:
                        forecast_valid = predicted is not None and math.isfinite(float(predicted)) and float(predicted) > 0
                    except (TypeError, ValueError, OverflowError):
                        forecast_valid = False
                    if not forecast_valid: reasons.append(f"{label.lower()}_forecast_unavailable")
                if share is None: reasons.append("share_insufficient_or_stale")
                base = ridge.get((p, h, target))
                if base is None: reasons.append("baseline_unavailable")
                row["targetCoTrade"] = all(target in data[p][m] for m in INPUT_MARKETS)
                row["fallbackReasons"] = sorted(set(reasons))
                row["fallback"] = bool(reasons)
                row["baselinePrediction"] = base
                row["candidatePrediction"] = base if reasons else share * float(row["predictedGarak"]) + (1-share) * float(row["predictedGangseo"])
                row["persistencePrediction"] = persistence.get((p, h, target))
                if row["persistencePrediction"] is None:
                    row["persistencePrediction"] = persistence.get(p, {}).get((target, h))
                if row["candidatePrediction"] is not None:
                    candidate[(p,h,target)] = row["candidatePrediction"]
                # Decomposition applies only when both actual markets traded and both forecasts/share existed.
                if not reasons and all(target in data[p][m] for m in INPUT_MARKETS):
                    row.update(_decomposition({**row,
                        "actualGarakPrice": data[p][GARAK][target][0], "actualGangseoPrice": data[p][GANGSEO][target][0],
                        "actualGarakWeight": data[p][GARAK][target][1], "actualGangseoWeight": data[p][GANGSEO][target][1]}))
                else:
                    row["decompositionExcludedReason"] = (
                        "target_not_positive_cotrade" if not row["targetCoTrade"] else
                        "candidate_fallback:" + ",".join(row["fallbackReasons"]) if reasons else "decomposition_unavailable")
                evidence.append(row)

    # Pair only dates issued by all three frozen arms; preserve coverage separately.
    paired = []
    for r in evidence:
        p,h,t = r["product"],r["horizon"],dt.date.fromisoformat(r["target"])
        base, pers = r["baselinePrediction"], r["persistencePrediction"]
        if r["candidatePrediction"] is not None and base is not None and pers is not None:
            paired.append({**r, "actual": actuals[p][t][0], "directRidge": base, "persistence": pers})
    comparisons=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            rows=[r for r in paired if r["product"]==p and r["horizon"]==h]
            for comp in ("directRidge","persistence"):
                for loss in ("mae","mape"):
                    diff={dt.date.fromisoformat(r["target"]):(abs(r["actual"]-r["candidatePrediction"])-abs(r["actual"]-r[comp])) if loss=="mae" else 100*(abs(r["actual"]-r["candidatePrediction"])-abs(r["actual"]-r[comp]))/r["actual"] for r in rows}
                    metrics=dh._metrics([r["actual"] for r in rows],[r["candidatePrediction"] for r in rows])
                    baseline_metrics=dh._metrics([r["actual"] for r in rows],[r[comp] for r in rows])
                    comparisons.append({"product":p,"horizon":h,"comparator":comp,"loss":loss,"n":len(rows),
                        "candidateMetric":metrics[loss],"comparatorMetric":baseline_metrics[loss],
                        "dm":{"9":dh._dm(diff,9),"28":dh._dm(diff,28)}})
    for bw in (9,28): dh._holm([r["dm"][str(bw)] for r in comparisons])
    if len(comparisons) != 200:
        raise AssertionError(f"frozen comparison family must contain 200 comparisons, got {len(comparisons)}")
    coverage={}
    for p in PRODUCTS:
        for h in HORIZONS:
            rows=[r for r in evidence if r["product"]==p and r["horizon"]==h]
            issued={name:{r["target"] for r in rows if r[field] is not None} for name,field in (("candidate","candidatePrediction"),("directRidge","baselinePrediction"),("persistence","persistencePrediction"))}
            coverage[f"{p}/{h}"]={"eligible":len(rows),"candidateIssued":len(issued["candidate"]),
                "baselineIssued":sum(r["baselinePrediction"] is not None for r in rows),"persistenceIssued":sum(r["persistencePrediction"] is not None for r in rows),
                "paired":len({(r["product"],r["horizon"],r["target"]) for r in paired if r["product"]==p and r["horizon"]==h}),"fallbackCount":sum(r["fallback"] for r in rows),
                "excludedTargetNotCotrade":sum(not r["targetCoTrade"] for r in rows),
                "candidateCoverageLossDates":{comp:sorted(issued[comp]-issued["candidate"]) for comp in ("directRidge","persistence")},
                "fallbackReasons":{reason:sum(reason in r["fallbackReasons"] for r in rows) for reason in sorted({x for r in rows for x in r["fallbackReasons"]})}}
    strata=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            rows=[r for r in paired if r["product"]==p and r["horizon"]==h]
            for kind, fn in (("year",lambda d:str(d.year)),("season",lambda d:"winter" if d.month in (12,1,2) else "spring" if d.month in (3,4,5) else "summer" if d.month in (6,7,8) else "autumn")):
                groups={}
                all_groups={}
                for r in rows: groups.setdefault(fn(dt.date.fromisoformat(r["target"])),[]).append(r)
                for r in evidence:
                    if r["product"]==p and r["horizon"]==h:
                        all_groups.setdefault(fn(dt.date.fromisoformat(r["target"])),[]).append(r)
                for value, eg in sorted(all_groups.items()):
                    group=groups.get(value,[]); fb=sum(r["fallback"] for r in eg)
                    strata.append({"product":p,"horizon":h,"stratum":kind,"value":value,"n":len(group),"eligibleCount":len(eg),"fallbackCount":fb,"fallbackRate":fb/len(eg) if eg else None,"candidate":dh._metrics([r["actual"] for r in group],[r["candidatePrediction"] for r in group]),"directRidge":dh._metrics([r["actual"] for r in group],[r["directRidge"] for r in group]),"persistence":dh._metrics([r["actual"] for r in group],[r["persistence"] for r in group])})
    joint=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            tests=[r for r in comparisons if r["product"]==p and r["horizon"]==h]
            cov=coverage[f"{p}/{h}"]
            inference=all(r["dm"]["9"]["status"]=="tested" and r["n"]>=30 for r in tests)
            metrics_better=all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"]<r["comparatorMetric"] for r in tests)
            tests_better=all(r["dm"]["9"].get("holmSignificant") and r["dm"]["9"].get("meanLossDifference") is not None and r["dm"]["9"]["meanLossDifference"]<0 for r in tests)
            cov_ok=all(not dates for dates in cov["candidateCoverageLossDates"].values())
            adverse=any(r["dm"]["9"]["status"]=="tested" and r["dm"]["9"].get("holmSignificant") and r["dm"]["9"].get("meanLossDifference",0)>0 for r in tests)
            conclusion="inference_unavailable" if not inference else "improvement" if metrics_better and tests_better and cov_ok else "worsening_or_mixed" if adverse else "inconclusive"
            joint.append({"product":p,"horizon":h,"conclusion":conclusion,"inferenceAvailable":inference,"allFourPairedMetricsImprove":metrics_better,"allFourHac9HolmTestsImprove":tests_better,"coverageNondeclining":cov_ok})
    return {"schemaVersion":1,"status":"complete","protocol":PROTOCOL,"hashes":hashes or {},"pairedRows":paired,"originEvidence":evidence,
            "comparisons":comparisons,"coverage":coverage,"yearSeasonMetrics":strata,
            "jointConclusions":joint,
            "decomposition": {"eligibleCoTradeRows":sum("priceContribution" in r for r in evidence),"excludedRows":sum(r.get("decompositionExcludedReason") is not None for r in evidence),
                "excludedReasons":{reason:sum(r.get("decompositionExcludedReason")==reason for r in evidence) for reason in sorted({r["decompositionExcludedReason"] for r in evidence if r.get("decompositionExcludedReason")})},
                "formula":"P-A = ŵ(p̂g−pg)+(1−ŵ)(p̂j−pj)+(ŵ−w)(pg−pj)"},
            "macroCommonDateMAPE": {name:(sum(v for v in vals)/50 if len(vals)==50 and all(v is not None for v in vals) else None) for name, vals in {
                "candidate":[dh._metrics([r["actual"] for r in paired if r["product"]==p and r["horizon"]==h],[r["candidatePrediction"] for r in paired if r["product"]==p and r["horizon"]==h])["mape"] for p in PRODUCTS for h in HORIZONS],
                "directRidge":[dh._metrics([r["actual"] for r in paired if r["product"]==p and r["horizon"]==h],[r["directRidge"] for r in paired if r["product"]==p and r["horizon"]==h])["mape"] for p in PRODUCTS for h in HORIZONS],
                "persistence":[dh._metrics([r["actual"] for r in paired if r["product"]==p and r["horizon"]==h],[r["persistence"] for r in paired if r["product"]==p and r["horizon"]==h])["mape"] for p in PRODUCTS for h in HORIZONS]}.items()},
            "limitations":PROTOCOL["limitations"]}


def run(prices_path=INPUT_PRICES, predictions_path=INPUT_PREDICTIONS, ridge_path=INPUT_RIDGE_REPORT, db_path=market_mix.DB_PATH, *, output=None, allow_nonfrozen_inputs=False):
    pb,predb,rb=Path(prices_path).read_bytes(),Path(predictions_path).read_bytes(),Path(ridge_path).read_bytes()
    hashes={"pricesSha256":hashlib.sha256(pb).hexdigest(),"predictionsSha256":hashlib.sha256(predb).hexdigest(),"directRidgeReportSha256":hashlib.sha256(rb).hexdigest()}
    if not allow_nonfrozen_inputs and hashes != EXPECTED: raise ValueError("input SHA-256 does not match frozen market-recomposition protocol")
    raw,predictions,ridge=map(json.loads,(pb,predb,rb))
    core=dh._observations(raw)
    markets,provenance=market_mix.extract_market_data(Path(db_path),raw,core)
    core_ridge, _ = rw._direct_ridge_reference(ridge)
    calendar={}
    for p,h,t in core_ridge:
        calendar.setdefault((p,h),set()).add(t)
    pred_map, audits, _ = _market_forecasts(markets,calendar)
    # Existing prediction export supplies the pinned persistence forecast mapping.
    _,_,persist=dh._forecast_maps(predictions)
    result=build(raw,markets,{"__market__":pred_map,"__persistence__":persist},ridge,hashes=hashes)
    result["fitAudits"]=audits
    result["inputs"]={"selectedRequestHashes":provenance["selected"]["selectedRequestHashes"],"selectedScopesSha256":provenance["selected"]["selectedScopesSha256"],"reconciliation":provenance["reconciliation"]}
    if output:
        target=Path(output); target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name(target.name+".tmp")
        tmp.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        tmp.replace(target)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="outputs/experiments/market-recompose-20261007/report.json")
    parser.add_argument("--allow-nonfrozen-inputs",action="store_true")
    args=parser.parse_args(argv)
    run(output=args.output,allow_nonfrozen_inputs=args.allow_nonfrozen_inputs)


if __name__ == "__main__": main()
