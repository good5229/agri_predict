"""Frozen three-phase Ridge/SARIMA component selection and shrinkage study.

No API or database access. Phase 1 writes answer-free component forecasts for
selection and test periods; phase 2 selects settings/weights and freezes test
predictions; phase 3 reconstructs both artifacts before joining test outcomes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

# The Ridge fits below are intentionally single-threaded. Set backend limits
# before importing project modules that may import NumPy/SciPy.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

from . import dashboard, direct_horizon as direct, modeling
from . import ten_strategy_benchmark as ten, ridge_sarima_blend as blend
from . import multiscale_residual as stats

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
DIRECT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
TEN_MANIFEST = ROOT / "outputs/experiments/ten-strategy-benchmark-20261008/forecast-manifest.json"
TEN_REPORT = ROOT / "outputs/experiments/ten-strategy-benchmark-20261008/report.json"
FIXED_MANIFEST = ROOT / "outputs/experiments/ridge-sarima-blend-20261008/forecast-manifest.json"
FIXED_REPORT = ROOT / "outputs/experiments/ridge-sarima-blend-20261008/report.json"
HISTORICAL_BASELINE = ROOT / "outputs/experiments/ridge-sarima-weight-cv-20261008/historical-components.json"
PRICES_SHA = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
TEN_MANIFEST_SHA = "30cc2c86e23027902667d554f8cddcec630172a1019a47b838b954b528adbeea"
TEN_REPORT_SHA = "a63726a72fa9671556926ee87833b416b60f78bcf02247c0b3a813d3b85c2a41"
FIXED_MANIFEST_SHA = "e73cd6b3c47b163000d020dbf60373af67d4edb79309af3a91b349167d945d40"
FIXED_REPORT_SHA = "423c70ce3c34dc4eaf80c3b3fb3367fa6eedc73745d2098bf66c8f62c1a5a822"
HISTORICAL_BASELINE_SHA = "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
HIST_START, HIST_END = dt.date(2024, 9, 25), dt.date(2025, 8, 31)
B1_START, B1_END = HIST_START, dt.date(2025, 2, 28)
B2_START, B2_END = dt.date(2025, 3, 1), HIST_END
TEST_START, TEST_END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
PERIODS = (("selection", HIST_START, HIST_END), ("test", TEST_START, TEST_END))
ALPHAS = (1.0, 10.0, 100.0)
DIFFERENCES = (0, 1)
TRAIN_DAYS = (365, 730)
RIDGE_KEYS = tuple(str(int(a)) for a in ALPHAS)
SARIMA_KEYS = tuple(f"d{d}_w{w}" for d in DIFFERENCES for w in TRAIN_DAYS)
CONFIGS = tuple((a, d, w) for a in ALPHAS for d in DIFFERENCES for w in TRAIN_DAYS)
WEIGHTS = (0.0, .25, .5, .75, 1.0)
SHRINKS = (0.0, .25, .5)
HAC_BANDWIDTHS = (9, 28)
ALPHA = .05
PROTOCOL = "ridge-sarima-components-multidistance-2026-10-08"
PLANNED_HIST = 11_370
PLANNED_TEST = 12_508
DM_FAMILY = 3 * 5 * 10 * 2
TOL = 1e-10


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(obj) -> str:
    return _sha(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _code_sha() -> str:
    return _sha(Path(__file__).read_bytes())


def _pos(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _iso(value):
    return dt.date.fromisoformat(value) if isinstance(value, str) else value


def _key(row):
    product, horizon = row.get("product"), row.get("horizon")
    target = _iso(row.get("target"))
    origin = _iso(row.get("origin")) if row.get("origin") else target - dt.timedelta(days=horizon)
    if product not in PRODUCTS or isinstance(horizon, bool) or horizon not in HORIZONS or target - origin != dt.timedelta(days=horizon):
        raise ValueError("forecast_key_invalid")
    return product, origin, target, horizon


def _immutable(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass


def _source_hashes(prices_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw, historical_baseline_raw):
    got = {"prices": _sha(prices_raw), "direct": _sha(direct_raw), "tenManifest": _sha(ten_raw),
           "tenReport": _sha(ten_report_raw), "fixedManifest": _sha(fixed_raw), "fixedReport": _sha(fixed_report_raw),
           "historicalBaseline": _sha(historical_baseline_raw)}
    expected = {"prices": PRICES_SHA, "direct": DIRECT_SHA, "tenManifest": TEN_MANIFEST_SHA,
                "tenReport": TEN_REPORT_SHA, "fixedManifest": FIXED_MANIFEST_SHA, "fixedReport": FIXED_REPORT_SHA,
                "historicalBaseline": HISTORICAL_BASELINE_SHA}
    if got != expected:
        raise ValueError("pinned_source_sha_mismatch")
    return got


def _maps(direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw):
    direct_doc = json.loads(direct_raw)
    if direct_doc.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("direct_report_protocol_mismatch")
    rows = direct_doc.get("pairedRows")
    if not isinstance(rows, list): raise ValueError("direct_rows_missing")
    direct_rows = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("_summary"): continue
        k = _key(row)
        if k in direct_rows: raise ValueError("direct_duplicate_key")
        direct_rows[k] = row
    ten_doc, fixed_doc = json.loads(ten_raw), json.loads(fixed_raw)
    ten_rows = blend._check_source_manifest(ten_doc, ten_raw)
    fixed_rows = blend._verify_forecast(fixed_doc, fixed_raw)
    if set(ten_rows) != set(fixed_rows): raise ValueError("test_source_keyset_mismatch")
    blend._validate_source_lineage(fixed_rows, ten_rows)
    ten_report, fixed_report = json.loads(ten_report_raw), json.loads(fixed_report_raw)
    if ten_report.get("manifestSha256") != ten_doc.get("manifestSha256") or ten_report.get("protocol") != ten.PROTOCOL:
        raise ValueError("ten_report_lineage_mismatch")
    ten_inputs=ten_report.get("inputSha256",{})
    if ten_inputs.get("direct")!=DIRECT_SHA or ten_inputs.get("prices")!=PRICES_SHA:
        raise ValueError("ten_report_input_lineage_mismatch")
    if fixed_report.get("manifestSha256") != fixed_doc.get("manifestSha256") or fixed_report.get("protocol") != blend.PROTOCOL:
        raise ValueError("fixed_report_lineage_mismatch")
    fixed_inputs=fixed_report.get("inputSha256",{})
    if (fixed_inputs.get("directReport")!=DIRECT_SHA or fixed_inputs.get("tenStrategyManifest")!=TEN_MANIFEST_SHA or
            fixed_inputs.get("tenStrategyReport") != TEN_REPORT_SHA):
        raise ValueError("fixed_report_source_lineage_mismatch")
    return direct_rows, ten_doc, ten_rows, ten_report, fixed_doc, fixed_rows, fixed_report


def _price_series(raw):
    return ten._price_series(json.loads(raw))


@contextmanager
def _single_thread_limit(threadpool_limits):
    """Use threadpoolctl when available; rely on pre-import backend limits on broken builds."""
    try:
        manager = threadpool_limits(limits=1)
        manager.__enter__()
    except Exception as exc:
        # Some older threadpoolctl/OpenBLAS combinations fail while inspecting
        # the linked library. Backend limits were fixed before NumPy import.
        yield f"{type(exc).__name__}:{exc}"
        return
    try:
        yield None
    finally:
        manager.__exit__(*sys.exc_info())


def _direct_fits(series, keys, alpha):
    """Monthly pooled Ridge, exactly matching direct_horizon feature/label rules."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    feature_rows = {}
    for product in PRODUCTS:
        cache = {}
        for target in series[product]:
            for horizon in HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                if origin < dt.date(2014, 1, 1): continue
                feature = modeling._feature_row(series[product], origin, target, horizon, cache)
                if feature is not None:
                    src, vec = feature
                    feature_rows[(product, origin, target, horizon)] = (src, vec)
    by_month = {}
    for k in keys:
        by_month.setdefault((k[0], k[3], k[1].year, k[1].month), []).append(k)
    predictions, audits = {}, {}
    for (product, horizon, year, month), month_keys in sorted(by_month.items()):
        eligible_origins = [target-dt.timedelta(days=horizon) for target in series[product]
            if direct.EVAL_START <= target <= direct.EVAL_END and target-dt.timedelta(days=horizon) >= direct.EVAL_START
            and (target-dt.timedelta(days=horizon)).year==year and (target-dt.timedelta(days=horizon)).month==month]
        if not eligible_origins: raise ValueError("direct_eligible_month_missing")
        fit_origin = min(eligible_origins)
        train = [(k, feature_rows[k]) for k in feature_rows if k[0] == product and k[3] == horizon and k[2] <= fit_origin]
        distinct = len({k[2] for k, _ in train})
        audit = {"fitOrigin": fit_origin.isoformat(), "fitSamples": len(train), "fitDistinctTargets": distinct,
                 "status": "unavailable", "error": None, "threadpoolLimitFallback": None}
        if distinct < direct.MIN_TRAIN_DAYS:
            audit["error"] = "insufficient_training"
            audits[(product, year, month)] = audit
            for k in month_keys: predictions[k] = (None, "insufficient_training")
            continue
        try:
            estimator = modeling._new_estimator("ridge")
            estimator.set_params(ridge__alpha=alpha)
            x = np.asarray([v[1] for _, v in train], dtype=float)
            y = np.asarray([math.log(series[product][k[2]][0])-math.log(series[product][v[0]][0]) for k,v in train], dtype=float)
            with _single_thread_limit(threadpool_limits) as fallback:
                audit["threadpoolLimitFallback"] = fallback
                estimator.fit(x, y)
                for k in month_keys:
                    sample = feature_rows.get(k)
                    if sample is None:
                        predictions[k] = (None, "feature_unavailable_or_source_too_old"); continue
                    src, vector = sample
                    val, _ = modeling._inverse_log_ratio(series[product][src][0], float(estimator.predict(np.asarray([vector], dtype=float))[0]))
                    predictions[k] = (val, None)
            audit["status"] = "fitted"
        except Exception as exc:
            audit["error"] = type(exc).__name__ + ":" + str(exc)
            for k in month_keys: predictions[k] = (None, "ridge_fit_failed")
        audits[(product, horizon, year, month)] = audit
    return predictions, audits


def _sarima_config(series, keys, d, window):
    """Calendar SARIMA fit once per period/config then append actuals to each origin."""
    if not keys: return {}, {}
    # Every product gets the same period-level fit cutoff, even if a product's
    # first eligible forecast key occurs later than another product's.
    first_origin = min(k[1] for k in keys)
    train_end = first_origin - dt.timedelta(days=1)
    grouped = {}
    for k in keys: grouped.setdefault(k[0], []).append(k)
    predictions, audits = {}, {}
    for product, product_keys in grouped.items():
        ps = series[product]
        train_start = train_end - dt.timedelta(days=window - 1)
        try:
            fit = ten._fit_calendar(ps, train_start, train_end, "sarima", d)
            err = None
        except Exception as exc:
            fit = None; err = type(exc).__name__ + ":" + str(exc)
        audits[product] = {"periodFirstOrigin": first_origin.isoformat(), "trainStart": train_start.isoformat(), "trainEnd": train_end.isoformat(),
                           "selectedD": d, "trainingCalendarDays": window,
                           "fitStatus": "available" if fit is not None else "unavailable", "fitError": err}
        state_day = train_end; failed = None
        origins = sorted({k[1] for k in product_keys})
        for origin in origins:
            if fit is not None and failed is None:
                while state_day < origin:
                    day = state_day + dt.timedelta(days=1)
                    try: fit = fit.append(ten._calendar_values(ps, day, day), refit=False)
                    except Exception as exc: failed = "state_update_failed:" + type(exc).__name__; break
                    state_day = day
            for k in (x for x in product_keys if x[1] == origin):
                src = max((x for x in ps if x <= origin), default=None)
                age = (origin-src).days if src else None
                if src is None: predictions[k] = (None, "no_source_observation", None, age)
                elif age > ten.MAX_SOURCE_AGE: predictions[k] = (None, "latest_source_older_than_14_days", src, age)
                elif fit is None: predictions[k] = (None, err or "sarima_fit_unavailable", src, age)
                elif failed: predictions[k] = (None, failed, src, age)
                else:
                    try:
                        means = fit.get_forecast(steps=k[3]).predicted_mean
                        lp = float(means.iloc[-1] if hasattr(means, "iloc") else means[-1])
                        pred = math.exp(lp)
                        if not _pos(pred): raise ValueError("invalid_prediction")
                        predictions[k] = (pred, None, src, age)
                    except Exception as exc:
                        predictions[k] = (None, "forecast_failed:" + type(exc).__name__, src, age)
    return predictions, audits


def _verify_historical_baseline(component_rows, historical_baseline_raw):
    if _sha(historical_baseline_raw) != HISTORICAL_BASELINE_SHA:
        raise ValueError("historical_baseline_sha_mismatch")
    baseline=json.loads(historical_baseline_raw)
    if (baseline.get("protocol")!="ridge-sarima-historical-weight-cv-2026-10-08" or
            baseline.get("phase")!="historical_components_only" or baseline.get("targetValuesAttached") is not False):
        raise ValueError("historical_baseline_protocol_mismatch")
    clean=dict(baseline);saved=clean.pop("manifestSha256",None)
    if saved!=_canonical(clean):raise ValueError("historical_baseline_manifest_hash_mismatch")
    if baseline.get("inputSha256")!={"prices":PRICES_SHA,"direct":DIRECT_SHA}:
        raise ValueError("historical_baseline_input_lineage_mismatch")
    audits=baseline.get("audits",{})
    if set(audits)!=set(PRODUCTS) or any(a.get("selectedD")!=1 for a in audits.values()):
        raise ValueError("historical_baseline_sarima_spec_mismatch")
    prior={}
    for row in baseline.get("forecasts",[]):
        if "actual" in row or "targetValue" in row:raise ValueError("historical_baseline_answer_leakage")
        k=_key(row)
        if not HIST_START<=k[2]<=HIST_END or k in prior:raise ValueError("historical_baseline_key_invalid")
        prior[k]=row
    current={_key(r):r for r in component_rows if HIST_START<=_iso(r["target"])<=HIST_END}
    if set(prior)!=set(current) or len(prior)!=PLANNED_HIST:
        raise ValueError("historical_baseline_keyset_mismatch")
    for k,old in prior.items():
        new=current[k]
        ridge=_finite(new.get("ridge",{}).get("10"));old_ridge=_finite(old.get("direct"))
        if ridge is None:
            status=new.get("ridgeStatus",{}).get("10") or "ridge_fit_unavailable"
            raise ValueError(f"historical_baseline_ridge_fit_unavailable:{status}")
        if old_ridge is None:raise ValueError("historical_baseline_ridge_reference_unavailable")
        if not math.isclose(ridge,old_ridge,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("historical_baseline_ridge_mismatch")
        old_sarima=_finite(old.get("sarima"));new_sarima=_finite(new.get("sarima",{}).get("d1_w730"))
        if (old_sarima is None)!=(new_sarima is None):
            raise ValueError("historical_baseline_sarima_coverage_mismatch")
        if old_sarima is not None and not math.isclose(old_sarima,new_sarima,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("historical_baseline_sarima_mismatch")
        if (old.get("status")=="available") != (new_sarima is not None):
            raise ValueError("historical_baseline_native_status_mismatch")
    return baseline


def build_forecasts(prices_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw, historical_baseline_raw):
    """Phase 1: forecast answer-free component grid for both frozen periods."""
    hashes = _source_hashes(prices_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw, historical_baseline_raw)
    direct_rows, ten_doc, ten_rows, _, fixed_doc, fixed_rows, _ = _maps(direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw)
    series = _price_series(prices_raw)
    direct_series = direct._observations(json.loads(prices_raw))
    keys = sorted((k for k in direct_rows if HIST_START <= k[2] <= HIST_END or TEST_START <= k[2] <= TEST_END),
                  key=lambda x: (x[0], x[1], x[3]))
    if len([k for k in keys if HIST_START <= k[2] <= HIST_END]) != PLANNED_HIST or len([k for k in keys if TEST_START <= k[2] <= TEST_END]) != PLANNED_TEST:
        raise ValueError("frozen_evaluation_grid_count_mismatch")
    direct_results = {str(int(a)): _direct_fits(direct_series, keys, a) for a in ALPHAS}
    sarima_results = {}
    sarima_audits = {}
    # Separate state starts preserve the frozen period's fit origins.
    for period, start, end in PERIODS:
        period_keys = [k for k in keys if start <= k[2] <= end]
        for d in DIFFERENCES:
            for window in TRAIN_DAYS:
                result, audits = _sarima_config(series, period_keys, d, window)
                sarima_results[(period, d, window)] = result
                sarima_audits[f"{period}|d{d}_w{window}"] = audits
    forecasts = []
    for k in keys:
        period = "selection" if HIST_START <= k[2] <= HIST_END else "test"
        direct_values = {};direct_status={}
        for name,result in direct_results.items():
            direct_values[name],direct_status[name]=result[0].get(k,(None,"not_forecasted"))
        sarima_values, statuses = {}, {}
        for d in DIFFERENCES:
            for window in TRAIN_DAYS:
                value, reason, source_date, age = sarima_results[(period, d, window)].get(k, (None, "not_forecasted", None, None))
                name = f"d{d}_w{window}"
                sarima_values[name] = value
                statuses[name] = {"reason": reason, "sourceDate": source_date.isoformat() if source_date else None, "sourceAgeDays": age}
        forecasts.append({"product": k[0], "origin": k[1].isoformat(), "target": k[2].isoformat(), "horizon": k[3],
                          "period": period, "ridge": direct_values, "ridgeStatus": direct_status,
                          "sarima": sarima_values, "sarimaStatus": statuses})
    _verify_historical_baseline(forecasts,historical_baseline_raw)
    doc = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "components_forecast_only", "targetValuesAttached": False,
           "inputSha256": hashes, "codeSha256": _code_sha(),
           "evaluation": {"selectionStart": HIST_START.isoformat(), "selectionEnd": HIST_END.isoformat(),
                          "b1": [B1_START.isoformat(), B1_END.isoformat()], "b2": [B2_START.isoformat(), B2_END.isoformat()],
                          "testStart": TEST_START.isoformat(), "testEnd": TEST_END.isoformat(), "plannedHistorical": PLANNED_HIST, "plannedTest": PLANNED_TEST},
           "candidateGrid": {"ridgeAlpha": list(ALPHAS), "sarimaDifference": list(DIFFERENCES), "trainingCalendarDays": list(TRAIN_DAYS),
                             "sarimaOrder": "(1,d,1)(1,0,0)[7]", "maxIter": ten.MAXITER, "statsmodelsVersion": "0.14.0",
                             "missingDays": "NaN; append calendar days; no interpolation", "sourceAgeMaximumDays": ten.MAX_SOURCE_AGE},
           "ridgeAudits": {a: {f"{p}|{h}|{y:04d}-{m:02d}": v for (p,h,y,m),v in results[1].items()} for a,results in direct_results.items()},
           "sarimaAudits": sarima_audits,
            "coverage": {"selectionPlanned": PLANNED_HIST, "testPlanned": PLANNED_TEST,
                        "ridgeNativeByAlpha": {a: sum(v[0] is not None for v in result[0].values()) for a,result in direct_results.items()},
                        "sarimaNativeByConfig": {f"{p}|d{d}_w{w}": sum(v[0] is not None for v in result.values()) for (p,d,w),result in sarima_results.items()}},
           "forecasts": forecasts}
    doc["manifestSha256"] = _canonical(doc)
    return doc


def _cell(rows, product, horizon, prediction):
    group = [r for r in rows if r["product"] == product and r["horizon"] == horizon and r.get("actual") is not None and r.get(prediction) is not None]
    n = len(group)
    return {"n": n, "mae": sum(abs(r["actual"]-r[prediction]) for r in group)/n if n else None,
            "mape": 100*sum(abs(r["actual"]-r[prediction])/r["actual"] for r in group)/n if n else None}


def _metrics(rows, prediction):
    cells = {(p,h): _cell(rows,p,h,prediction) for p in PRODUCTS for h in HORIZONS}
    return {"cells": {f"{p}|{h}": v for (p,h),v in cells.items()},
            "macro50CellMape": sum(v["mape"] for v in cells.values())/50 if all(v["mape"] is not None for v in cells.values()) else None,
            "macro50CellMae": sum(v["mae"] for v in cells.values())/50 if all(v["mae"] is not None for v in cells.values()) else None,
            "coverage": sum(v["n"] for v in cells.values())}


def _covered_keys(rows, prediction):
    return {(r["product"],r["origin"],r["target"],r["horizon"]) for r in rows
            if r.get("actual") is not None and r.get("predictions", r).get(prediction) is not None}


def _candidate_rows(component_rows, direct_rows, prices, start, end, *, attach_truth):
    out = []
    for row in component_rows:
        target = dt.date.fromisoformat(row["target"])
        if not start <= target <= end: continue
        k = _key(row); source = direct_rows.get(k)
        if source is None: raise ValueError("direct_key_missing")
        actual = None
        if attach_truth:
            try: actual = float(source["actual"])
            except (KeyError,TypeError,ValueError): actual = None
            observed = prices[k[0]].get(k[2])
            if actual is not None and (not _pos(actual) or observed is None or not math.isclose(actual,observed,rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError("actual_price_reconciliation_failed")
        direct_pred = _finite(source.get("direct")); persistence = _finite(source.get("persistence"))
        item = {"product":k[0],"origin":k[1],"target":k[2],"horizon":k[3],"actual":actual,
                "direct":direct_pred,"persistence":persistence}
        for a in RIDGE_KEYS: item[f"ridge{a}"] = row["ridge"].get(a)
        for c in SARIMA_KEYS: item[f"sarima{c}"] = row["sarima"].get(c)
        out.append(item)
    return out


def _blend_component(row, alpha, sarima_key):
    direct_value, sarima_value = row[f"ridge{int(alpha)}"], row[f"sarima{sarima_key}"]
    return .5*direct_value + .5*sarima_value if direct_value is not None and sarima_value is not None else None


def _select_a(history_rows):
    b1 = [r for r in history_rows if B1_START <= r["target"] <= B1_END]
    b2 = [r for r in history_rows if B2_START <= r["target"] <= B2_END]
    results = []
    baseline = [_blend_component(r,10,f"d1_w730") for r in b1]
    baseline_rows = [dict(r, fixed50=p) for r,p in zip(b1,baseline)]
    base_metrics = _metrics(baseline_rows,"fixed50")
    for alpha,d,window in CONFIGS:
        sk=f"d{d}_w{window}"; field=f"candidate"
        candidate = [dict(r,candidate=_blend_component(r,alpha,sk)) for r in b1]
        metrics = _metrics(candidate,field)
        sarima_coverage=sum(r.get(f"sarima{sk}") is not None for r in b1)
        ridge_coverage=sum(r.get(f"ridge{int(alpha)}") is not None for r in b1)
        full_coverage=metrics["coverage"]==len(b1)
        eligible = full_coverage and metrics["macro50CellMape"] is not None and all(v["n"]>=30 for v in metrics["cells"].values())
        results.append({"alpha":alpha,"d":d,"trainingCalendarDays":window,"eligible":eligible,
                        "ineligibleReason":None if eligible else "sarima_fit_or_native_coverage_incomplete" if sarima_coverage<len(b1) else "ridge_fit_or_cell_coverage_incomplete" if ridge_coverage<len(b1) else "cell_below_30",
                        "plannedPairs":len(b1),"ridgeNativeCoverage":ridge_coverage,"sarimaNativeCoverage":sarima_coverage,"metrics":metrics})
    viable=[r for r in results if r["eligible"]]
    selected=min(viable,key=lambda r:(r["metrics"]["macro50CellMape"],r["metrics"]["macro50CellMae"],abs(r["alpha"]-10),abs(r["d"]-1),abs(r["trainingCalendarDays"]-730))) if viable else None
    if selected is None: adopted={"alpha":10.,"d":1,"trainingCalendarDays":730,"source":"current_setting_no_eligible_candidate"}
    else:
        adopted={"alpha":selected["alpha"],"d":selected["d"],"trainingCalendarDays":selected["trainingCalendarDays"],"source":"selected"}
        alpha=adopted["alpha"];sk=f"d{adopted['d']}_w{adopted['trainingCalendarDays']}"
        b2rows=[dict(r,candidate=_blend_component(r,alpha,sk),fixed50=_blend_component(r,10,"d1_w730")) for r in b2]
        cand=_metrics(b2rows,"candidate");base=_metrics(b2rows,"fixed50")
        same_coverage=_covered_keys(b2rows,"candidate")==_covered_keys(b2rows,"fixed50")
        pass_gate=(cand["macro50CellMape"] is not None and base["macro50CellMape"] is not None and
                   cand["macro50CellMape"]<=base["macro50CellMape"]-.10 and
                   cand["macro50CellMae"]<=base["macro50CellMae"]*1.01 and same_coverage)
        adopted.update({"b2GatePassed":pass_gate,"b2Metrics":cand,"currentB2Metrics":base})
        if not pass_gate: adopted={"alpha":10.,"d":1,"trainingCalendarDays":730,"source":"current_setting_b2_gate_failed","b2GatePassed":False,"b2Metrics":cand,"currentB2Metrics":base}
    return {"candidateResults":results,"selectedBeforeB2":selected,"adopted":adopted,"b1CurrentMetrics":base_metrics}


def _weight_fit(rows, groups):
    out={}
    for group_name, predicate in groups:
        planned=[r for r in rows if predicate(r) and r.get("actual") is not None]
        group=[r for r in planned if r.get("direct") is not None and r.get("sarima") is not None]
        scores=[]
        expected_horizons=range(1,4) if group_name.endswith("|1-3") else range(4,11)
        for w in WEIGHTS:
            temp=[dict(r,prediction=w*r["direct"]+(1-w)*r["sarima"]) for r in group]
            by_h=[]
            for h in expected_horizons:
                by_h.append(_cell(temp,planned[0]["product"],h,"prediction")["mape"] if planned else None)
            scores.append({"ridgeWeight":w,"distanceEqualWeightMape":sum(by_h)/len(by_h) if by_h and all(x is not None for x in by_h) else None,"pairs":len(group)})
        valid=[s for s in scores if s["distanceEqualWeightMape"] is not None]
        full_coverage=bool(planned) and len(group)==len(planned)
        chosen=min(valid,key=lambda x:(x["distanceEqualWeightMape"],abs(x["ridgeWeight"]-.5))) if valid and full_coverage else {"ridgeWeight":.5,"distanceEqualWeightMape":None,"pairs":len(group)}
        out[group_name]={"scores":scores,"weight":chosen["ridgeWeight"],"pairs":len(group),"plannedPairs":len(planned),
                         "eligible":full_coverage,"ineligibleReason":None if full_coverage else "component_unavailable"}
    return out


def _select_b(history_rows, a_choice):
    alpha,d,window=a_choice["alpha"],a_choice["d"],a_choice["trainingCalendarDays"]
    sarima_key=f"d{d}_w{window}"
    rows=[]
    for r in history_rows:
        rows.append({**r,"direct":r[f"ridge{int(alpha)}"],"sarima":r[f"sarima{sarima_key}"],"a50":_blend_component(r,alpha,sarima_key),"fixed50":_blend_component(r,10,"d1_w730")})
    groups=[(f"{p}|{g}",lambda r,p=p,g=g:r["product"]==p and ((1<=r["horizon"]<=3) if g=="1-3" else (4<=r["horizon"]<=10))) for p in PRODUCTS for g in ("1-3","4-10")]
    b1=[r for r in rows if B1_START<=r["target"]<=B1_END]
    raw_weights=_weight_fit(b1,groups)
    b2=[r for r in rows if B2_START<=r["target"]<=B2_END]
    shrink_scores=[]
    for shrink in SHRINKS:
        candidate=[]
        for r in b2:
            group=f"{r['product']}|{'1-3' if r['horizon']<=3 else '4-10'}"
            raw=raw_weights[group]["weight"];w=.5+shrink*(raw-.5)
            candidate.append({**r,"candidate":w*r["direct"]+(1-w)*r["sarima"] if r["direct"] is not None and r["sarima"] is not None else None,"shrinkWeight":w})
        m=_metrics(candidate,"candidate")
        shrink_scores.append({"shrink":shrink,"metrics":m})
    valid=[r for r in shrink_scores if r["metrics"]["macro50CellMape"] is not None]
    chosen=min(valid,key=lambda r:(r["metrics"]["macro50CellMape"],r["metrics"]["macro50CellMae"],r["shrink"])) if valid else {"shrink":0.,"metrics":None}
    a_rows=[dict(r,candidate=r["a50"]) for r in b2];am=_metrics(a_rows,"candidate")
    bm=chosen["metrics"]
    chosen_rows=[]
    if bm is not None:
        shrink_value=chosen["shrink"]
        for r in b2:
            group=f"{r['product']}|{'1-3' if r['horizon']<=3 else '4-10'}"
            raw=raw_weights[group]["weight"];w=.5+shrink_value*(raw-.5)
            pred=w*r["direct"]+(1-w)*r["sarima"] if r["direct"] is not None and r["sarima"] is not None else None
            chosen_rows.append({**r,"candidate":pred})
    gate=bool(bm and am["macro50CellMape"] is not None and bm["macro50CellMape"]<=am["macro50CellMape"]-.10 and bm["macro50CellMae"]<=am["macro50CellMae"]*1.01 and _covered_keys(chosen_rows,"candidate")==_covered_keys(a_rows,"candidate"))
    shrink=chosen["shrink"] if gate else 0.
    combined=[r for r in rows if B1_START<=r["target"]<=B2_END]
    refit=_weight_fit(combined,groups)
    return {"rawB1Weights":raw_weights,"b2ShrinkScores":shrink_scores,"selectedB2Shrink":chosen["shrink"],
            "shrinkGatePassed":gate,"adoptedShrink":shrink,"b2MetricsA":am,"b2MetricsB":bm,
            "refitB1B2Weights":refit,"groups":{g[0]:{"rawWeight":refit[g[0]]["weight"],"testWeight":.5+shrink*(refit[g[0]]["weight"]-.5)} for g in groups}}


def build_selection(forecast_raw, prices_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw, historical_baseline_raw):
    """Phase 2: use B1/B2 labels, then freeze test predictions without test truth."""
    hashes=_source_hashes(prices_raw,direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw,historical_baseline_raw)
    forecast=json.loads(forecast_raw)
    if forecast.get("phase")!="components_forecast_only" or forecast.get("targetValuesAttached") is not False: raise ValueError("phase1_manifest_invalid")
    clean=dict(forecast);saved=clean.pop("manifestSha256",None)
    if saved!=_canonical(clean):raise ValueError("phase1_manifest_tampered")
    if forecast.get("inputSha256")!=hashes or forecast.get("codeSha256")!=_code_sha():raise ValueError("phase1_lineage_mismatch")
    direct_map,*_=_maps(direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw)
    prices=_price_series(prices_raw)
    component=[];seen=set()
    for row in forecast.get("forecasts",[]):
        k=_key(row)
        if k in seen:raise ValueError("phase1_duplicate_key")
        seen.add(k)
        if "actual" in row or "targetValue" in row:raise ValueError("phase1_answer_leakage")
        component.append(row)
    _verify_historical_baseline(component,historical_baseline_raw)
    expected_keys={k for k in direct_map if HIST_START<=k[2]<=HIST_END or TEST_START<=k[2]<=TEST_END}
    if seen!=expected_keys:raise ValueError("phase1_component_keyset_mismatch")
    if sum(HIST_START<=k[2]<=HIST_END for k in seen)!=PLANNED_HIST or sum(TEST_START<=k[2]<=TEST_END for k in seen)!=PLANNED_TEST:
        raise ValueError("phase1_component_coverage_mismatch")
    hist=_candidate_rows(component,direct_map,prices,HIST_START,HIST_END,attach_truth=True)
    for row in hist:
        key=(row["product"],row["origin"],row["target"],row["horizon"])
        source_direct=_finite(direct_map[key].get("direct"));recomputed=row["ridge10"]
        if source_direct is None or recomputed is None or not math.isclose(recomputed,source_direct,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("historical_original_ridge_prediction_mismatch")
    a=_select_a(hist)
    b=_select_b(hist,a["adopted"])
    test_rows=_candidate_rows(component,direct_map,prices,TEST_START,TEST_END,attach_truth=False)
    alpha=a["adopted"]["alpha"]; d=a["adopted"]["d"]; window=a["adopted"]["trainingCalendarDays"]
    sk=f"d{d}_w{window}"
    _,ten_doc,ten_rows,_,fixed_doc,fixed_rows,_=_maps(direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw)
    frozen=[]
    for r in test_rows:
        k=(r["product"],r["origin"],r["target"],r["horizon"])
        direct_value=r[f"ridge{int(alpha)}"]; sarima_value=r[f"sarima{sk}"]
        fixed= fixed_rows.get(k); tenrow=ten_rows.get(k)
        if fixed is None or tenrow is None:raise ValueError("test_source_key_missing")
        source_direct=float(tenrow["predictions"]["direct_ridge"]);source_native=tenrow.get("nativePredictions",{}).get("calendar_sarima_111_100_7")
        fixed50=float(fixed["predictions"]["blend"])
        direct_original=float(direct_map[k]["direct"])
        if not math.isclose(source_direct,direct_original,rel_tol=1e-10,abs_tol=1e-8):raise ValueError("ten_direct_report_mismatch")
        source_persistence=_finite(direct_map[k].get("persistence"))
        if source_persistence is None or not math.isclose(source_persistence,float(tenrow["predictions"]["naive"]),rel_tol=1e-10,abs_tol=1e-8) or not math.isclose(source_persistence,float(fixed["predictions"]["persistence"]),rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("persistence_source_mismatch")
        baseline_ridge=r["ridge10"]
        if baseline_ridge is None or not math.isclose(baseline_ridge,direct_original,rel_tol=1e-10,abs_tol=1e-8):raise ValueError("original_ridge_prediction_mismatch")
        baseline_sarima=r["sarimad1_w730"]
        if (baseline_sarima is None) != (source_native is None):
            raise ValueError("original_sarima_coverage_mismatch")
        if source_native is not None and not math.isclose(baseline_sarima,float(source_native),rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("original_sarima_prediction_mismatch")
        fixed_current=(.5*source_direct+.5*float(source_native) if source_native is not None else source_direct)
        if not math.isclose(fixed_current,fixed50,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("original_fixed50_prediction_mismatch")
        candidate_a=.5*direct_value+.5*sarima_value if direct_value is not None and sarima_value is not None else fixed50
        group=f"{r['product']}|{'1-3' if r['horizon']<=3 else '4-10'}";wb=b["groups"][group]["testWeight"]
        candidate_b=wb*direct_value+(1-wb)*sarima_value if direct_value is not None and sarima_value is not None else fixed50
        frozen.append({"product":r["product"],"origin":r["origin"].isoformat(),"target":r["target"].isoformat(),"horizon":r["horizon"],
                       "predictions":{"a":candidate_a,"b":candidate_b,"fixed50":fixed50,"direct":direct_original,"sarima":sarima_value,"persistence":r["persistence"]},
                       "lineage":{"group":group,"ridgeAlpha":alpha,"sarimaDifference":d,"trainingCalendarDays":window,"ridgeWeightB":wb,
                                  "sarimaNative":sarima_value is not None,"componentFallback":direct_value is None or sarima_value is None,
                                  "fallbackReason":"selected_component_unavailable" if direct_value is None or sarima_value is None else None,
                                  "fixed50Reconstructed":fixed_current}})
    result={"schemaVersion":1,"protocol":PROTOCOL,"phase":"selected_test_forecast_only","targetValuesAttached":False,
            "inputSha256":hashes,"codeSha256":_code_sha(),"componentManifestSha256":forecast["manifestSha256"],
            "selection":{"A":a,"B":b},"coverage":{"plannedTest":len(frozen),"aAvailable":sum(x["predictions"]["a"] is not None for x in frozen),
                                                         "bAvailable":sum(x["predictions"]["b"] is not None for x in frozen)},"forecasts":frozen}
    result["manifestSha256"]=_canonical(result)
    return result


def _dm(diffs, bandwidth):
    diffs=[(_iso(day),float(value)) for day,value in diffs]
    n=len(diffs); mean=sum(v for _,v in diffs)/n if n else None
    if n<30:return {"status":"inference_unavailable_insufficient_pairs","n":n,"meanLossDifference":mean,"statistic":None,"pValue":1.}
    by={d:v-mean for d,v in diffs}; variance=sum(x*x for x in by.values())
    for lag in range(1,bandwidth+1):variance+=2*(1-lag/(bandwidth+1))*sum(v*by.get(d-dt.timedelta(days=lag),0.) for d,v in by.items())
    if not math.isfinite(variance) or variance<=0:return {"status":"inference_unavailable_nonpositive_variance","n":n,"meanLossDifference":mean,"statistic":None,"pValue":1.}
    z=mean/(math.sqrt(variance)/n)
    return {"status":"tested","n":n,"meanLossDifference":mean,"statistic":z,"pValue":math.erfc(abs(z)/math.sqrt(2))}


def _holm(rows):
    ordered=sorted(range(len(rows)),key=lambda i:(rows[i]["pValue"],rows[i]["product"],rows[i]["horizon"],rows[i]["comparator"],rows[i]["loss"]))
    running=0.
    for rank,index in enumerate(ordered):
        running=max(running,min(1.,(len(rows)-rank)*rows[index]["pValue"]))
        rows[index]["holmPValue"]=running;rows[index]["holmSignificant"]=rows[index]["status"]=="tested" and running<ALPHA


def _check_source_report(report, rows, mappings):
    refs={(r.get("product"),r.get("horizon")):r.get("metrics",{}) for r in report.get("cellMetrics",[])}
    if len(refs)!=len(PRODUCTS)*len(HORIZONS):raise ValueError("source_report_cells_invalid")
    for product in PRODUCTS:
        for horizon in HORIZONS:
            group=[r for r in rows if r["product"]==product and r["horizon"]==horizon]
            ref=refs.get((product,horizon),{})
            for score_field, report_field in mappings:
                n=len(group)
                mae=sum(abs(r["actual"]-r[score_field]) for r in group)/n if n else None
                mape=100*sum(abs(r["actual"]-r[score_field])/r["actual"] for r in group)/n if n else None
                expected=ref.get(report_field,{})
                if expected.get("n")!=n or mae is None or not math.isclose(mae,float(expected["mae"]),rel_tol=1e-10,abs_tol=1e-8):
                    raise ValueError(f"source_report_mae_mismatch:{score_field}:{report_field}")
                if not math.isclose(mape,float(expected["mape"]),rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError(f"source_report_mape_mismatch:{score_field}:{report_field}")


def score(test_raw, forecast_raw, prices_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw, historical_baseline_raw):
    """Phase 3: rebuild phase 1+2, then join truth and report complete inference."""
    hashes=_source_hashes(prices_raw,direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw,historical_baseline_raw)
    expected_components=build_forecasts(prices_raw,direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw,historical_baseline_raw)
    if json.loads(forecast_raw)!=expected_components:raise ValueError("component_manifest_reconstruction_mismatch")
    expected=build_selection(forecast_raw,prices_raw,direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw,historical_baseline_raw)
    test=json.loads(test_raw)
    if test!=expected:raise ValueError("selected_test_manifest_reconstruction_mismatch")
    clean=dict(test);saved=clean.pop("manifestSha256",None)
    if saved!=_canonical(clean):raise ValueError("test_manifest_tampered")
    direct_rows,ten_doc,ten_rows,ten_report,fixed_doc,fixed_rows,fixed_report=_maps(direct_raw,ten_raw,ten_report_raw,fixed_raw,fixed_report_raw)
    prices=_price_series(prices_raw);scored=[]
    for row in test["forecasts"]:
        k=_key(row);source=direct_rows.get(k)
        if source is None:raise ValueError("test_actual_key_missing")
        actual=float(source["actual"]);observed=prices[k[0]].get(k[2])
        if not _pos(actual) or observed is None or not math.isclose(actual,observed,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("test_actual_reconciliation_failed")
        tenrow=ten_rows.get(k);fixed=fixed_rows.get(k)
        if tenrow is None or fixed is None:raise ValueError("test_frozen_source_missing")
        p=row["predictions"]
        if not math.isclose(p["fixed50"],float(fixed["predictions"]["blend"]),rel_tol=1e-10,abs_tol=1e-8):raise ValueError("fixed50_source_mismatch")
        if not math.isclose(p["direct"],float(tenrow["predictions"]["direct_ridge"]),rel_tol=1e-10,abs_tol=1e-8):raise ValueError("direct_source_mismatch")
        if not math.isclose(p["persistence"],float(fixed["predictions"]["persistence"]),rel_tol=1e-10,abs_tol=1e-8):raise ValueError("persistence_source_mismatch")
        scored.append({**row,"actual":actual})
    source_scored=[]
    for row in scored:
        k=_key(row);tenrow=ten_rows[k];fixed=fixed_rows[k]
        source_scored.append({"product":k[0],"horizon":k[3],"actual":row["actual"],
            "direct_ridge":float(tenrow["predictions"]["direct_ridge"]),
            "calendar_sarima_111_100_7":float(tenrow["predictions"]["calendar_sarima_111_100_7"]),
            "naive":float(tenrow["predictions"]["naive"]),"blend":float(fixed["predictions"]["blend"]),
            "persistence":float(fixed["predictions"]["persistence"])})
    _check_source_report(ten_report,source_scored,( ("direct_ridge","direct_ridge"),
        ("calendar_sarima_111_100_7","calendar_sarima_111_100_7"),("naive","naive")))
    _check_source_report(fixed_report,source_scored,( ("direct_ridge","direct_ridge"),
        ("calendar_sarima_111_100_7","calendar_sarima_111_100_7"),("persistence","persistence"),("blend","blend")))
    methods=("a","b","fixed50","direct","sarima","persistence")
    common_six=[r for r in scored if all(r["predictions"].get(m) is not None for m in methods)]
    adoption_rows=[r for r in scored if all(r["predictions"].get(m) is not None for m in ("a","b","fixed50"))]
    cells=[];tests={bw:[] for bw in HAC_BANDWIDTHS};macro={m:[] for m in ("a","b","fixed50")};six_macro={m:[] for m in methods}
    for product in PRODUCTS:
        for h in HORIZONS:
            group=[r for r in scored if r["product"]==product and r["horizon"]==h]
            adoption_group=[r for r in adoption_rows if r["product"]==product and r["horizon"]==h]
            common_group=[r for r in common_six if r["product"]==product and r["horizon"]==h]
            metrics={};six_metrics={};full_grid={}
            for m in ("a","b","fixed50"):
                n=len(adoption_group)
                mae=sum(abs(r["actual"]-r["predictions"][m]) for r in adoption_group)/n if n else None
                mape=100*sum(abs(r["actual"]-r["predictions"][m])/r["actual"] for r in adoption_group)/n if n else None
                metrics[m]={"n":n,"mae":mae,"mape":mape};macro[m].append(metrics[m])
            for m in methods:
                n=len(common_group)
                mae=sum(abs(r["actual"]-r["predictions"][m]) for r in common_group)/n if n else None
                mape=100*sum(abs(r["actual"]-r["predictions"][m])/r["actual"] for r in common_group)/n if n else None
                six_metrics[m]={"n":n,"mae":mae,"mape":mape};six_macro[m].append(six_metrics[m])
                available=[r for r in group if r["predictions"][m] is not None];full_n=len(available)
                full_grid[m]={"n":full_n,"mae":sum(abs(r["actual"]-r["predictions"][m]) for r in available)/full_n if full_n else None,
                              "mape":100*sum(abs(r["actual"]-r["predictions"][m])/r["actual"] for r in available)/full_n if full_n else None}
            cells.append({"product":product,"horizon":h,"plannedPairs":len(group),"adoptionPairs":len(adoption_group),"metrics":metrics,
                          "sixMethodCommonPairs":len(common_group),"sixMethodCommonMetrics":six_metrics,"fullGridDiagnostics":full_grid,
                          "nativeSarima":sum(bool(r["lineage"]["sarimaNative"]) for r in group),"fallbackSarima":sum(not bool(r["lineage"]["sarimaNative"]) for r in group),
                          "componentFallbackA":sum(bool(r["lineage"]["componentFallback"]) for r in group),
                          "componentFallbackB":sum(bool(r["lineage"]["componentFallback"]) for r in group)})
            for comparator,candidate in (("fixed50","a"),("fixed50","b"),("a","b")):
                for loss in ("AE","APE"):
                    diffs=[]
                    for r in adoption_group:
                        y=r["actual"];x=r["predictions"][candidate];z=r["predictions"][comparator]
                        diffs.append((_iso(r["target"]),abs(y-x)-abs(y-z) if loss=="AE" else 100*(abs(y-x)/y-abs(y-z)/y)))
                    for bw in HAC_BANDWIDTHS:
                        tests[bw].append({"product":product,"horizon":h,"candidate":candidate,"comparator":comparator,"loss":loss,**_dm(diffs,bw)})
    for bw in HAC_BANDWIDTHS:_holm(tests[bw])
    macro_metrics={m:{"meanOf50CellMape":sum(x["mape"] for x in macro[m])/50 if all(x["mape"] is not None for x in macro[m]) else None,
                     "meanOf50CellMae":sum(x["mae"] for x in macro[m])/50 if all(x["mae"] is not None for x in macro[m]) else None} for m in macro}
    six_method_macro={m:{"meanOf50CellMape":sum(x["mape"] for x in six_macro[m])/50 if all(x["mape"] is not None for x in six_macro[m]) else None,
                         "meanOf50CellMae":sum(x["mae"] for x in six_macro[m])/50 if all(x["mae"] is not None for x in six_macro[m]) else None} for m in methods}
    conclusions=[]
    indexed={(bw,t["product"],t["horizon"],t["candidate"],t["comparator"],t["loss"]):t for bw,items in tests.items() for t in items}
    for cell in cells:
        for candidate,comparator in (("a","fixed50"),("b","fixed50"),("b","a")):
            required=[indexed[(bw,cell["product"],cell["horizon"],candidate,comparator,loss)] for bw in HAC_BANDWIDTHS for loss in ("AE","APE")]
            available=all(t["status"]=="tested" and t["n"]>=30 for t in required)
            improve=available and all(t["holmSignificant"] and t["meanLossDifference"]<0 for t in required)
            worsen=available and all(t["holmSignificant"] and t["meanLossDifference"]>0 for t in required)
            conclusions.append({"product":cell["product"],"horizon":cell["horizon"],"candidate":candidate,"comparator":comparator,
                                "conclusion":"improvement" if improve else "worsening" if worsen else "inconclusive" if available else "inference_unavailable",
                                "requiredTests":required})
    product_metrics={}
    for product in PRODUCTS:
        group=[r for r in adoption_rows if r["product"]==product]
        product_metrics[product]={}
        for method in ("a","b","fixed50"):
            n=sum(r["predictions"][method] is not None for r in group)
            usable=[r for r in group if r["predictions"][method] is not None]
            product_metrics[product][method]={"n":n,
                "mape":100*sum(abs(r["actual"]-r["predictions"][method])/r["actual"] for r in usable)/n if n else None,
                "mae":sum(abs(r["actual"]-r["predictions"][method]) for r in usable)/n if n else None}
    expansion={}
    for method in ("a","b"):
        baseline=macro_metrics["fixed50"];candidate=macro_metrics[method]
        product_pass=all(product_metrics[p][method]["mape"] is not None and product_metrics[p]["fixed50"]["mape"] is not None and
                         product_metrics[p][method]["mape"]-product_metrics[p]["fixed50"]["mape"]<=1.0 and
                         product_metrics[p][method]["mae"]-product_metrics[p]["fixed50"]["mae"]<=.05*product_metrics[p]["fixed50"]["mae"] for p in PRODUCTS)
        sig_cell=any(c["candidate"]==method and c["comparator"]=="fixed50" and c["conclusion"]=="improvement" for c in conclusions)
        same_coverage=_covered_keys(scored,method)==_covered_keys(scored,"fixed50")
        pass_gate=(candidate["meanOf50CellMape"] is not None and baseline["meanOf50CellMape"] is not None and
                   candidate["meanOf50CellMape"]<=10 and candidate["meanOf50CellMape"]<=baseline["meanOf50CellMape"]-.5 and
                   same_coverage and product_pass and sig_cell)
        expansion[method]={"passed":pass_gate,"macroMapeAtMost10Percent":candidate["meanOf50CellMape"] is not None and candidate["meanOf50CellMape"]<=10,
                           "improvementOverFixed50AtLeast0_5pp":candidate["meanOf50CellMape"] is not None and baseline["meanOf50CellMape"] is not None and candidate["meanOf50CellMape"]<=baseline["meanOf50CellMape"]-.5,
                           "coverageKeysMatchFixed50":same_coverage,"productLimitsPass":product_pass,"jointSignificantImprovementCell":sig_cell}
    full_grid_summary={}
    for method in methods:
        available=[r for r in scored if r["predictions"][method] is not None];n=len(available)
        full_grid_summary[method]={"availablePairs":n,"plannedPairs":len(scored),
            "mae":sum(abs(r["actual"]-r["predictions"][method]) for r in available)/n if n else None,
            "mape":100*sum(abs(r["actual"]-r["predictions"][method])/r["actual"] for r in available)/n if n else None}
    return {"schemaVersion":1,"protocol":PROTOCOL,"inputSha256":hashes,"componentManifestSha256":json.loads(forecast_raw)["manifestSha256"],
            "manifestSha256":saved,"coverage":{"plannedPairs":PLANNED_TEST,"scoredPairs":len(scored),
                "commonSixMethodPairs":len(common_six),"adoptionPairs":len(adoption_rows),
                "byMethod":{m:{"available":sum(r["predictions"][m] is not None for r in scored),
                               "commonSixMethodPairs":len(common_six),
                               "fallback":sum(bool(r["lineage"].get("componentFallback")) for r in scored) if m in ("a","b") else 0} for m in methods}},
            "cellMetrics":cells,"macroMetrics":macro_metrics,"sixMethodCommonMacroMetrics":six_method_macro,"fullGridDiagnostics":full_grid_summary,
            "dmTests":{"HAC9":tests[9],"HAC28":tests[28],"familySizePerBandwidth":DM_FAMILY,"holmAlpha":ALPHA},
            "productMetrics":product_metrics,"expansionGate":expansion,"conclusions":conclusions,"limitations":["Retrospective revised prices and previously inspected periods are exploratory, not independent confirmation.",
            "Same-day source vintage and first-publication time are unknown.","A/B outcomes do not establish official national-price score or user benefit."]}


def main(argv=None):
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="phase",required=True)
    p1=sub.add_parser("forecast");p1.add_argument("--manifest",type=Path,required=True)
    p2=sub.add_parser("select");p2.add_argument("--components",type=Path,required=True);p2.add_argument("--manifest",type=Path,required=True)
    p3=sub.add_parser("score");p3.add_argument("--components",type=Path,required=True);p3.add_argument("--manifest",type=Path,required=True);p3.add_argument("--report",type=Path,required=True)
    for p in (p1,p2,p3):
        p.add_argument("--prices",type=Path,default=PRICES);p.add_argument("--direct",type=Path,default=DIRECT)
        p.add_argument("--ten-manifest",type=Path,default=TEN_MANIFEST);p.add_argument("--ten-report",type=Path,default=TEN_REPORT)
        p.add_argument("--fixed-manifest",type=Path,default=FIXED_MANIFEST);p.add_argument("--fixed-report",type=Path,default=FIXED_REPORT)
        p.add_argument("--historical-baseline",type=Path,default=HISTORICAL_BASELINE)
    args=parser.parse_args(argv);values=[args.prices.read_bytes(),args.direct.read_bytes(),args.ten_manifest.read_bytes(),args.ten_report.read_bytes(),args.fixed_manifest.read_bytes(),args.fixed_report.read_bytes(),args.historical_baseline.read_bytes()]
    if args.phase=="forecast":result=build_forecasts(*values);_immutable(args.manifest,result)
    elif args.phase=="select":result=build_selection(args.components.read_bytes(),*values);_immutable(args.manifest,result)
    else:result=score(args.manifest.read_bytes(),args.components.read_bytes(),*values);_immutable(args.report,result)
    print(json.dumps({"phase":args.phase,"manifestSha256":result.get("manifestSha256"),"forecastRows":len(result.get("forecasts",[]))}))


if __name__=="__main__":main()
