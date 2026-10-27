"""Pure, fixed-protocol scorer for the retrospective RHS forecasts.

This module does not load prices, perform source verification, fit models, or
write artifacts.  The caller supplies an answer-free forecast manifest plus
already validated baseline forecasts, actuals, and fit diagnostics.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from collections import defaultdict
from typing import Any, Mapping

from . import equal_info_online_score as inference

PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
CONDITIONS = ("B", "D", "C", "P")
COMPARISONS = (("C_vs_B", "C", "B"), ("C_vs_D", "C", "D"), ("C_vs_P", "C", "P"))
TECHNICAL_COMPARISONS = (("D_vs_B", "D", "B"), ("P_vs_B", "P", "B"),
                         ("C_vs_D", "C", "D"), ("C_vs_P", "C", "P"))
LOSSES = ("ae", "ape")
BANDWIDTHS = (9, 28)
TARGET_WINDOWS = {
    "V": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
    "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27)),
}
ORIGIN_CUTOFFS = {"V": dt.date(2026, 4, 1), "T": dt.date(2026, 7, 1)}
MIN_PAIRS = 30
BOOTSTRAP_BLOCKS = (14, 28)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0


def _day(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise ValueError(f"invalid date: {value!r}")


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _replay_close(value: Any, baseline: Any) -> bool:
    value0, base0 = _finite(value), _finite(baseline)
    if value0 is None or base0 is None:
        return value0 is None and base0 is None
    return abs(value0-base0) <= 1e-6 + 1e-8*abs(base0)


def _key(row: Mapping[str, Any]) -> tuple[str, dt.date, dt.date, int]:
    try:
        ident = row.get("key", row)
        product = ident["product"]
        origin = _day(ident.get("originDate", ident.get("origin")))
        target = _day(ident.get("targetDate", ident.get("target")))
        horizon = ident["horizon"]
    except (KeyError, TypeError) as exc:
        raise ValueError("forecast row lacks a complete identity") from exc
    if product not in PRODUCTS or type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("forecast identity is outside the fixed product/horizon domain")
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError("forecast origin/target/horizon mismatch")
    return product, origin, target, horizon


def _loss(actual: float, prediction: float, name: str) -> float:
    error = abs(prediction - actual)
    return error if name == "ae" else error / actual


def _metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "uniqueTargets": 0, "directLogMse": None,
                "uniqueOrigins": 0, "directLogN": 0, "directMae": None,
                "directMape": None, "blendMae": None, "blendMape": None}
    vals = [r.get(condition, r) for r in rows]
    logs = [v["directLogPrediction"] - r["actualLogReturn"] for v, r in zip(vals, rows)
            if r.get("actualLogReturn") is not None]
    direct_ae = [abs(v["directPrediction"] - r["actual"]) for v, r in zip(vals, rows)]
    blend_ae = [abs(v["blendPrediction"] - r["actual"]) for v, r in zip(vals, rows)]
    n = len(rows)
    return {"n": n, "uniqueTargets": len({r["target"] for r in rows}),
            "uniqueOrigins": len({r["origin"] for r in rows}),
            "directLogN": len(logs),
            "directLogMse": math.fsum(x*x for x in logs) / len(logs) if logs else None,
            "directMae": math.fsum(direct_ae) / n,
            "directMape": 100 * math.fsum(e/r["actual"] for e, r in zip(direct_ae, rows)) / n,
            "blendMae": math.fsum(blend_ae) / n,
            "blendMape": 100 * math.fsum(e/r["actual"] for e, r in zip(blend_ae, rows)) / n}


def _validate(doc: Mapping[str, Any], baseline_rows: Mapping[Any, Any], actuals: Mapping[Any, Any], phase: str):
    if phase not in TARGET_WINDOWS:
        raise ValueError("phase must be V or T")
    forecasts = doc.get("forecasts")
    if not isinstance(forecasts, list):
        raise ValueError("manifest forecasts must be a list")
    start, end = TARGET_WINDOWS[phase]
    origin_cutoff = ORIGIN_CUTOFFS[phase]
    by_key: dict[tuple[str, dt.date, dt.date, int], Mapping[str, Any]] = {}
    for row in forecasts:
        key = _key(row)
        if key in by_key:
            raise ValueError("duplicate forecast identity")
        if type(row.get("eligible")) is not bool:
            raise ValueError("eligible must be a boolean")
        if row["eligible"] is not (key[1] >= origin_cutoff):
            raise ValueError("eligibility does not match the frozen origin cutoff")
        if not start <= key[2] <= end:
            raise ValueError("forecast target is outside its fixed phase window")
        conds = row.get("conditions")
        if not isinstance(conds, dict) or set(conds) != set(CONDITIONS):
            raise ValueError("condition set must be exactly B/D/C/P")
        for name, item in conds.items():
            if not isinstance(item, dict) or item.get("status") not in ("issued", "not_issued", "fallback"):
                raise ValueError(f"invalid {name} condition status")
            if item["status"] in ("issued", "fallback"):
                if _finite(item.get("directLogPrediction")) is None or _positive(item.get("directPrediction")) is None or _positive(item.get("blendPrediction")) is None:
                    raise ValueError(f"issued {name} lacks finite log or positive price predictions")
                p0 = _positive(item.get("p0"))
                if p0 is not None:
                    reconstructed = p0 * math.exp(float(item["directLogPrediction"]))
                    direct = float(item["directPrediction"])
                    if abs(reconstructed-direct) > 1e-6 + 1e-8*abs(direct):
                        raise ValueError(f"{name} direct log and price predictions disagree")
        by_key[key] = {**row, "product": key[0], "origin": key[1], "target": key[2], "horizon": key[3]}
    expected = {(product, target-dt.timedelta(days=h), target, h)
                for product in PRODUCTS
                for target in (start+dt.timedelta(days=i) for i in range((end-start).days+1))
                for h in HORIZONS}
    if set(by_key) != expected:
        raise ValueError("manifest does not contain the full frozen product/target/horizon grid")
    # Every B key must retain its original E106 row, independently of D/P availability.
    for key, row in by_key.items():
        if _key(row)[0] == "radish":
            if key not in baseline_rows:
                raise ValueError("radish forecast lacks saved E106 baseline row")
            base = baseline_rows[key]
            if not isinstance(base, dict): raise ValueError("invalid saved baseline row")
            bcond = row["conditions"]["B"]
            if (bcond.get("status") != base.get("status") or
                not _replay_close(bcond.get("blendPrediction"), base.get("prediction")) or
                not _replay_close(bcond.get("directPrediction"), base.get("directPrediction")) or
                _finite(bcond.get("sarimaPrediction")) != _finite(base.get("sarimaPrediction"))):
                raise ValueError("B does not reproduce its saved E106 baseline row")
    normalized_actuals: dict[tuple[str, dt.date], float] = {}
    for (product, target), raw in actuals.items():
        value = _positive(raw)
        if product in PRODUCTS and value is not None:
            normalized_actuals[(product, _day(target))] = value
    rows = []
    for key, row in by_key.items():
        product, origin, target, horizon = key
        if not (start <= target <= end and row["eligible"]):
            continue
        actual = normalized_actuals.get((product, target))
        if actual is None:
            continue
        conds = row["conditions"]
        p0_values = {_positive(conds[c].get("p0")) for c in CONDITIONS}
        p0_values.discard(None)
        if len(p0_values) > 1:
            raise ValueError("conditions do not share the same p0")
        # Scoring rows carry a per-condition value only when that condition issued.
        p0 = _positive(conds["B"].get("p0"))
        out = {"product": product, "origin": origin, "target": target, "horizon": horizon,
               "actual": actual, "actualLogReturn": math.log(actual / p0) if p0 else None}
        for name, item in conds.items():
            if item["status"] in ("issued", "fallback"):
                direct_log = _finite(item.get("directLogPrediction"))
                direct = _positive(item.get("directPrediction"))
                blend = _positive(item.get("blendPrediction"))
                p0 = _positive(item.get("p0")) or p0
                if direct_log is None or direct is None or blend is None or p0 is None:
                    continue
                out[name] = {"directLogPrediction": direct_log, "directPrediction": direct,
                             "blendPrediction": blend}
        rows.append(out)
    return by_key, rows, normalized_actuals


def _pair_rows(rows, left, right, *, require_all=False):
    out = []
    for r in rows:
        if left not in r or right not in r:
            continue
        a, b = r[left], r[right]
        out.append({**{k: r[k] for k in ("product", "origin", "target", "horizon", "actual", "actualLogReturn")},
                    "left": a, "right": b})
    return out


def _statistical_rows(rows: list[dict[str, Any]]):
    by_horizon = {h: [r for r in rows if r["horizon"] == h] for h in HORIZONS}
    dm = []
    for comparison, left, right in COMPARISONS:
        for h in HORIZONS:
            cell = [r for r in by_horizon[h] if left in r and right in r and
                    (comparison == "C_vs_B" or all(c in r for c in CONDITIONS))]
            for loss in LOSSES:
                diffs = {}
                same_prediction = bool(cell)
                for r in cell:
                    a, b = r[left]["blendPrediction"], r[right]["blendPrediction"]
                    same_prediction = same_prediction and a == b
                    diff = abs(a-r["actual"])-abs(b-r["actual"])
                    if loss == "ape": diff = 100*diff / r["actual"]
                    diffs[r["target"]] = diff
                for bandwidth in BANDWIDTHS:
                    result = inference._dm(diffs, bandwidth)
                    if same_prediction and len(diffs) >= MIN_PAIRS:
                        result = {"status": "identical_predictions", "reason": "same_blend_prediction_for_every_common_pair",
                                  "n": len(diffs), "statistic": None,
                                  "meanLossDifference": math.fsum(diffs.values())/len(diffs) if diffs else None,
                                  "pValue": 1.0}
                    dm.append({"comparison": comparison, "horizon": h, "loss": loss,
                               "unit": "price_per_kg" if loss == "ae" else "percentage_points",
                               "samePrediction": same_prediction, "bandwidth": bandwidth, **result})
    for bandwidth in BANDWIDTHS:
        family = [r for r in dm if r["bandwidth"] == bandwidth]
        if len(family) != 60:
            raise ValueError("fixed Holm family must contain 60 tests per bandwidth")
        inference._holm(family)
    summaries = []
    for bandwidth in BANDWIDTHS:
        family = [r for r in dm if r["bandwidth"] == bandwidth]
        summaries.append({"bandwidth": bandwidth, "familySize": len(family),
                          "tested": sum(r["status"] == "tested" for r in family),
                          "unavailable": sum(r["status"] != "tested" for r in family),
                          "identicalPredictionCells": sum(r["status"] == "identical_predictions" for r in family),
                          "holmSignificantImprovement": sum(r["status"] == "tested" and r["holmSignificant"] and r["meanLossDifference"] < 0 for r in family),
                          "holmSignificantWorsening": sum(r["status"] == "tested" and r["holmSignificant"] and r["meanLossDifference"] > 0 for r in family)})
    return dm, summaries


def _bootstrap(rows: list[dict[str, Any]], block: int, window: tuple[dt.date, dt.date]) -> dict[str, Any]:
    """Joint circular calendar-block CI for 10-horizon equal-weight blend loss."""
    first, last = window
    calendar = [first + dt.timedelta(days=i) for i in range((last-first).days + 1)]
    dates = calendar
    horizon_losses: dict[tuple[int, dt.date, str, str], float] = {}
    for r in rows:
        if "C" not in r or "B" not in r: continue
        h, day = r["horizon"], r["target"]
        for comp, left, right in (("C_vs_B", "C", "B"), ("C_vs_D", "C", "D"), ("C_vs_P", "C", "P")):
            if left not in r or right not in r or (comp != "C_vs_B" and any(c not in r for c in CONDITIONS)): continue
            for loss in LOSSES:
                diff = abs(r[left]["blendPrediction"]-r["actual"]) - abs(r[right]["blendPrediction"]-r["actual"])
                if loss == "ape": diff = 100*diff / r["actual"]
                horizon_losses[(h, day, comp, loss)] = diff
    def estimate(sample, comparison, loss):
        vals = []
        for h in HORIZONS:
            cell = [horizon_losses[(h, d, comparison, loss)] for d in sample if (h, d, comparison, loss) in horizon_losses]
            if not cell: return None
            vals.append(math.fsum(cell)/len(cell))
        return math.fsum(vals)/len(vals)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(calendar)
    samples = []
    for _ in range(BOOTSTRAP_REPS):
        sampled_idx = []
        while len(sampled_idx) < n:
            start = rng.randrange(n)
            sampled_idx.extend((start+j) % n for j in range(block))
        sampled = [calendar[i] for i in sampled_idx[:n]]
        samples.append(sampled)
    results = {}
    for comparison, _, _ in COMPARISONS:
        for loss in LOSSES:
            point = estimate(dates, comparison, loss)
            values = [v for sample in samples if (v := estimate(sample, comparison, loss)) is not None and math.isfinite(v)]
            values.sort()
            ci = [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]] if values else None
            results[f"{comparison}/{loss}"] = {"estimate": point, "replicatesRequested": BOOTSTRAP_REPS,
                "replicatesValid": len(values), "ci95": ci}
    import hashlib
    draw_hash = hashlib.sha256(repr([[d.isoformat() for d in sample] for sample in samples]).encode()).hexdigest()
    return {"blockDays": block, "seed": BOOTSTRAP_SEED, "replicatesRequested": BOOTSTRAP_REPS,
            "results": results, "drawTableSha256": draw_hash,
            "units": {"ae": "price_per_kg", "ape": "percentage_points"},
            "scope": "joint calendar target-date draws shared across comparisons, horizons, and missingness masks"}


def _coverage(by_key, actuals, rows, phase):
    start, end = TARGET_WINDOWS[phase]
    planned = [r for r in by_key.values() if start <= _day(r["target"]) <= end and r["eligible"]]
    issued = defaultdict(int); fallback = defaultdict(int); withheld = defaultdict(int)
    common = defaultdict(int)
    missing_candidate = []; new_fallback = []; status_pairs = defaultdict(int)
    for r in planned:
        key = _key(r)
        for name, item in r["conditions"].items():
            if item["status"] == "issued": issued[name] += 1
            elif item["status"] == "fallback": fallback[name] += 1
            else: withheld[name] += 1
        for _, a, b in (COMPARISONS[0],):
            bs, cs = r["conditions"][b]["status"], r["conditions"][a]["status"]
            status_pairs[f"{bs}->{cs}"] += 1
            identity = f"{key[0]}/{key[1].isoformat()}/{key[2].isoformat()}/{key[3]}"
            if bs in ("issued", "fallback") and cs == "not_issued": missing_candidate.append(identity)
            if bs == "issued" and cs == "fallback": new_fallback.append(identity)
            if r["conditions"][a]["status"] in ("issued", "fallback") and r["conditions"][b]["status"] in ("issued", "fallback") and (key[0], key[2]) in actuals:
                common[(key[0], key[3])] += 1
    return {"plannedEligibleKeys": len(planned), "positiveActualProductTargets": sum((p,d) in actuals for p in PRODUCTS for d in { _day(r["target"]) for r in planned if r["product"] == p}),
            "issuedKeys": dict(issued), "fallbackKeys": dict(fallback), "withheldKeys": dict(withheld),
            "BtoCStatusTransitions": dict(status_pairs),
            "candidateMissingWhenBaselineAvailable": missing_candidate,
            "candidateNewFallbackWhenBaselineIssued": new_fallback,
            "C_vs_BCommonPairsByProductHorizon": {f"{p}/{h}": common[(p,h)] for p in PRODUCTS for h in HORIZONS}}


def _training_report(fit_diagnostics: Any) -> dict[str, Any]:
    if not isinstance(fit_diagnostics, list):
        raise ValueError("fit_diagnostics must be a per-fit list")
    expected = {"fitId", "fitOrigin", "horizon", "trainingCount", "trainingRows"}
    seen = set(); seen_origin_h = set(); out = []
    for fit in fit_diagnostics:
        if not isinstance(fit, dict) or not expected <= set(fit):
            raise ValueError("invalid per-fit training diagnostic")
        ident = fit["fitId"]
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("missing/duplicate fitId in training diagnostics")
        seen.add(ident)
        fit_origin = _day(fit["fitOrigin"])
        if (fit_origin, fit["horizon"]) in seen_origin_h:
            raise ValueError("duplicate fitOrigin/horizon training diagnostic")
        seen_origin_h.add((fit_origin, fit["horizon"]))
        train = fit["trainingRows"]
        if not isinstance(train, list) or fit["trainingCount"] != len(train):
            raise ValueError("trainingRows/count mismatch")
        for row in train:
            if not isinstance(row, dict) or not {"targetDate", "actualLogReturn", *CONDITIONS} <= set(row):
                raise ValueError("training row lacks paired log-return values")
            if any(_finite(row[c]) is None for c in (*CONDITIONS, "actualLogReturn")):
                raise ValueError("training row contains nonfinite log return")
        decomp = {}
        for label, left, right in (COMPARISONS[0], *TECHNICAL_COMPARISONS):
            eb = [float(r["actualLogReturn"])-float(r[right]) for r in train]
            delta = [float(r[left])-float(r[right]) for r in train]
            n = len(train)
            mse_delta = (math.fsum((e-d)**2 for e,d in zip(eb,delta))-math.fsum(e*e for e in eb))/n if n else None
            delta_sq = math.fsum(d*d for d in delta)/n if n else None
            align = -2*math.fsum(e*d for e,d in zip(eb,delta))/n if n else None
            decomp[label] = {"n": n, "mseDifference": mse_delta,
                             "meanDeltaSquared": delta_sq,
                             "minus2MeanBaselineResidualTimesDelta": align,
                             "identityResidual": mse_delta-(delta_sq+align) if n else None}
        by_month = defaultdict(list)
        for row in train: by_month[_day(row["targetDate"]).strftime("%Y-%m")].append(row)
        monthly_decomp = []
        for month, month_rows in sorted(by_month.items()):
            for label, left, right in (COMPARISONS[0], *TECHNICAL_COMPARISONS):
                eb = [float(r["actualLogReturn"])-float(r[right]) for r in month_rows]
                delta = [float(r[left])-float(r[right]) for r in month_rows]
                n = len(month_rows)
                mse_delta = (math.fsum((e-d)**2 for e,d in zip(eb,delta))-math.fsum(e*e for e in eb))/n if n else None
                delta_sq = math.fsum(d*d for d in delta)/n if n else None
                align = -2*math.fsum(e*d for e,d in zip(eb,delta))/n if n else None
                monthly_decomp.append({"targetMonth": month, "comparison": label, "n": n,
                                       "uniqueTargetDates": len({_day(r["targetDate"]) for r in month_rows}),
                                       "mseDifference": mse_delta, "meanDeltaSquared": delta_sq,
                                       "minus2MeanBaselineResidualTimesDelta": align,
                                       "identityResidual": mse_delta-(delta_sq+align) if n else None})
        # Preserve each overlapping expanding-fold fit separately.
        out.append({**fit, "logMseDecomposition": decomp,
                    "logMseDecompositionByTargetMonth": monthly_decomp,
                    "conditionDirectLogMse": {c: math.fsum((float(r[c])-float(r["actualLogReturn"]))**2 for r in train)/len(train) if train else None for c in CONDITIONS},
                    "conditionDirectLogMseByTargetMonth": [{"targetMonth": month, "n": len(month_rows),
                        "metrics": {c: math.fsum((float(r[c])-float(r["actualLogReturn"]))**2 for r in month_rows)/len(month_rows) for c in CONDITIONS}}
                        for month, month_rows in sorted(by_month.items())]})
    origin_groups = defaultdict(dict)
    training_months = set()
    for fit in out:
        origin = _day(fit["fitOrigin"])
        origin_groups[origin][fit["horizon"]] = fit["trainingRows"]
        training_months.update(_day(r["targetDate"]).strftime("%Y-%m") for r in fit["trainingRows"])
    origin_equal = []
    for origin, by_h in sorted(origin_groups.items()):
        for month in (*sorted(training_months), "all"):
            for label, left, right in (COMPARISONS[0], *TECHNICAL_COMPARISONS):
                per_h = {}
                for h in HORIZONS:
                    train = [r for r in by_h.get(h, []) if month == "all" or _day(r["targetDate"]).strftime("%Y-%m") == month]
                    if not train:
                        per_h[h] = None
                        continue
                    eb = [float(r["actualLogReturn"])-float(r[right]) for r in train]
                    delta = [float(r[left])-float(r[right]) for r in train]
                    mse_delta = (math.fsum((e-d)**2 for e,d in zip(eb,delta))-math.fsum(e*e for e in eb))/len(train)
                    delta_sq = math.fsum(d*d for d in delta)/len(train)
                    align = -2*math.fsum(e*d for e,d in zip(eb,delta))/len(train)
                    per_h[h] = {"n": len(train), "mseDifference": mse_delta,
                                "meanDeltaSquared": delta_sq,
                                "minus2MeanBaselineResidualTimesDelta": align,
                                "identityResidual": mse_delta-(delta_sq+align)}
                supported = [x for x in per_h.values() if x is not None]
                complete = len(supported) == len(HORIZONS)
                entry = {"fitOrigin": origin.isoformat(), "jointFitId": f"radish-{origin.isoformat()}",
                         "targetMonth": month, "comparison": label,
                         "horizonsIncluded": len(supported), "complete10HorizonSupport": complete,
                         "trainingRowsAcrossHorizons": sum(x["n"] for x in supported)}
                for field in ("mseDifference", "meanDeltaSquared", "minus2MeanBaselineResidualTimesDelta", "identityResidual"):
                    entry[field] = math.fsum(x[field] for x in supported)/len(HORIZONS) if complete else None
                condition_metrics = {}
                for condition in CONDITIONS:
                    vals = []
                    for h in HORIZONS:
                        train = [r for r in by_h.get(h, []) if month == "all" or _day(r["targetDate"]).strftime("%Y-%m") == month]
                        if not train: vals = []; break
                        vals.append(math.fsum((float(r[condition])-float(r["actualLogReturn"]))**2 for r in train)/len(train))
                    condition_metrics[condition] = math.fsum(vals)/len(HORIZONS) if complete and len(vals) == len(HORIZONS) else None
                entry["equalHorizonDirectLogMseByCondition"] = condition_metrics
                origin_equal.append(entry)
    return {"fitCount": len(out), "fitDiagnostics": out,
            "jointFitOriginEqualHorizonDecomposition": origin_equal,
            "scopeNote": "Per-fit expanding-fold aggregates are kept separate; overlapping fits are not pooled as independent observations."}


def _monthly_metrics(rows, phase):
    out = []
    months = _month_labels(*TARGET_WINDOWS[phase])
    for month in months:
        for product in ("radish",):
            for h in HORIZONS:
                cell = [r for r in rows if r["target"].strftime("%Y-%m") == month and r["product"] == product and r["horizon"] == h]
                metrics = {cond: _metrics([r for r in cell if cond in r], cond) for cond in CONDITIONS}
                out.append({"targetMonth": month, "product": product, "horizon": h,
                            "metricsByCondition": metrics, "uniqueTargetDates": len({r["target"] for r in cell}),
                            "uniqueOrigins": len({r["origin"] for r in cell})})
    return out


def _all_four_common_metrics(rows):
    common = [r for r in rows if all(c in r for c in CONDITIONS)]
    return [{"product": product, "horizon": h, "condition": condition,
             **_metrics([r for r in common if r["product"] == product and r["horizon"] == h], condition)}
            for product in PRODUCTS for h in HORIZONS for condition in CONDITIONS]


def _all_four_monthly_metrics(rows, phase):
    common = [r for r in rows if all(c in r for c in CONDITIONS)]
    result = []
    for month in _month_labels(*TARGET_WINDOWS[phase]):
        for product in PRODUCTS:
            for h in HORIZONS:
                cell = [r for r in common if r["product"] == product and r["horizon"] == h and r["target"].strftime("%Y-%m") == month]
                for condition in CONDITIONS:
                    result.append({"product": product, "horizon": h, "targetMonth": month,
                                   "condition": condition, **_metrics(cell, condition)})
    return result


def _month_labels(start, end):
    labels = []
    d = start.replace(day=1)
    while d <= end:
        labels.append(d.strftime("%Y-%m"))
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return labels


def _monthly_coverage(by_key, actuals, phase):
    start, end = TARGET_WINDOWS[phase]
    labels = _month_labels(start, end)
    output = []
    for product in PRODUCTS:
        for h in HORIZONS:
            for month in labels:
                cell = [r for r in by_key.values() if r["product"] == product and r["horizon"] == h
                        and r["eligible"] and start <= r["target"] <= end and r["target"].strftime("%Y-%m") == month]
                targets = {r["target"] for r in cell}
                entry = {"product": product, "horizon": h, "targetMonth": month,
                         "plannedKeys": len(cell), "uniquePlannedTargets": len(targets),
                         "positiveActualTargets": sum((product, day) in actuals for day in targets)}
                for c in CONDITIONS:
                    entry[c+"Issued"] = sum(r["conditions"][c]["status"] == "issued" for r in cell)
                    entry[c+"Fallback"] = sum(r["conditions"][c]["status"] == "fallback" for r in cell)
                    entry[c+"NotIssued"] = sum(r["conditions"][c]["status"] == "not_issued" for r in cell)
                entry["C_vs_BCommonPairs"] = sum((product, r["target"]) in actuals and r["conditions"]["C"]["status"] in ("issued", "fallback") and r["conditions"]["B"]["status"] in ("issued", "fallback") for r in cell)
                entry["allFourCommonPairs"] = sum((product, r["target"]) in actuals and all(r["conditions"][c]["status"] in ("issued", "fallback") for c in CONDITIONS) for r in cell)
                output.append(entry)
    return output


def _decomposition(rows, comparisons, phase, sample_scope):
    results = []
    months = _month_labels(*TARGET_WINDOWS[phase])

    def components(cell):
        if not cell:
            return {"n": 0, "uniqueTargetDates": 0, "uniqueOrigins": 0,
                    "mseDifference": None, "meanDeltaSquared": None,
                    "minus2MeanBaselineResidualTimesDelta": None, "identityResidual": None}
        ebase = [r["actualLogReturn"] - r[right]["directLogPrediction"] for r in cell]
        delta = [r[left]["directLogPrediction"] - r[right]["directLogPrediction"] for r in cell]
        mse_a = math.fsum((e-d)**2 for e,d in zip(ebase,delta))/len(cell)
        mse_b = math.fsum(e*e for e in ebase)/len(cell)
        md2 = math.fsum(d*d for d in delta)/len(cell)
        align = -2*math.fsum(e*d for e,d in zip(ebase,delta))/len(cell)
        return {"n": len(cell), "uniqueTargetDates": len({r["target"] for r in cell}),
                "uniqueOrigins": len({r["origin"] for r in cell}),
                "mseDifference": mse_a-mse_b, "meanDeltaSquared": md2,
                "minus2MeanBaselineResidualTimesDelta": align,
                "identityResidual": (mse_a-mse_b)-(md2+align)}

    for label, left, right in comparisons:
        for product in ("radish",):
            for month in (*months, "all"):
                by_h = {}
                for h in HORIZONS:
                    cell = [r for r in rows if r["product"] == product and r["horizon"] == h
                            and (month == "all" or r["target"].strftime("%Y-%m") == month)
                            and left in r and right in r and r.get("actualLogReturn") is not None]
                    by_h[h] = components(cell)
                    results.append({"comparison": label, "sampleScope": sample_scope,
                                    "aggregation": "horizon", "product": product, "horizon": h,
                                    "targetMonth": month, **by_h[h]})
                supported = [by_h[h] for h in HORIZONS if by_h[h]["n"] > 0]
                complete = len(supported) == len(HORIZONS)
                macro = {"n": sum(x["n"] for x in supported),
                         "horizonsIncluded": len(supported), "complete10HorizonSupport": complete,
                         "uniqueTargetDates": len({r["target"] for r in rows if r["product"] == product and (month == "all" or r["target"].strftime("%Y-%m") == month)}),
                         "uniqueOrigins": len({r["origin"] for r in rows if r["product"] == product and (month == "all" or r["target"].strftime("%Y-%m") == month)})}
                for field in ("mseDifference", "meanDeltaSquared", "minus2MeanBaselineResidualTimesDelta", "identityResidual"):
                    macro[field] = math.fsum(by_h[h][field] for h in HORIZONS)/len(HORIZONS) if complete and all(by_h[h][field] is not None for h in HORIZONS) else None
                results.append({"comparison": label, "sampleScope": sample_scope,
                                "aggregation": "equal_weight_10_horizons", "product": product,
                                "horizon": None, "targetMonth": month, **macro})
    return results


def score_manifest(doc: Mapping[str, Any], *, baseline_rows: Mapping[Any, Any],
                   actuals: Mapping[Any, Any], fit_diagnostics: Any, phase: str) -> dict[str, Any]:
    """Score one answer-free V or T manifest against prevalidated inputs.

    Keys in ``baseline_rows`` are ``(product, origin, target, horizon)``; dates
    may be date or ISO strings. Actual keys are ``(product, target)``.
    """
    normalized_baselines = {}
    for key, value in baseline_rows.items():
        if len(key) != 4: raise ValueError("baseline key must be (product, origin, target, horizon)")
        product, origin, target, h = key
        normalized_baselines[(product, _day(origin), _day(target), h)] = value
    by_key, all_rows, normalized_actuals = _validate(doc, normalized_baselines, actuals, phase)
    # Build scoring payload, preserving all positive actuals while treating unavailable methods as missing.
    scored = []
    scored.extend(all_rows)
    radish = [r for r in scored if r["product"] == "radish"]
    metrics = [{"horizon": h, "condition": cond,
                **_metrics([r for r in radish if r["horizon"] == h and cond in r], cond)}
               for h in HORIZONS for cond in CONDITIONS]
    dm, dm_summaries = _statistical_rows(radish)
    pairs = {label: _pair_rows(radish, a, b) for label, a, b in (*COMPARISONS, *TECHNICAL_COMPARISONS)}
    # Main gate is a strict, all-ten-horizon C−B gain with no issue/coverage loss.
    common_cb = [r for r in radish if "C" in r and "B" in r]
    per_horizon = []
    for h in HORIZONS:
        cell = [r for r in common_cb if r["horizon"] == h]
        cmet, bmet = _metrics(cell, "C"), _metrics(cell, "B")
        cmae, bmae = cmet["blendMae"], bmet["blendMae"]
        cmape, bmape = cmet["blendMape"], bmet["blendMape"]
        per_horizon.append({"horizon": h, "commonPairs": len(cell), "uniqueTargetDates": len({r["target"] for r in cell}),
                            "uniqueOrigins": len({r["origin"] for r in cell}),
                            "candidateMetrics": cmet, "baselineMetrics": bmet,
                            "CBlendMae": cmae, "BBlendMae": bmae, "CBlendMape": cmape, "BBlendMape": bmape})
    supported = [r for r in per_horizon if r["commonPairs"] > 0]
    full_horizon_support = len(supported) == len(HORIZONS)
    c_equal_mape = math.fsum(r["CBlendMape"] for r in supported)/len(HORIZONS) if full_horizon_support else None
    b_equal_mape = math.fsum(r["BBlendMape"] for r in supported)/len(HORIZONS) if full_horizon_support else None
    c_equal_mae = math.fsum(r["CBlendMae"] for r in supported)/len(HORIZONS) if full_horizon_support else None
    b_equal_mae = math.fsum(r["BBlendMae"] for r in supported)/len(HORIZONS) if full_horizon_support else None
    macro_fields = ("directLogMse", "directMae", "directMape", "blendMae", "blendMape")
    equal_horizon_metrics = {"horizonsIncluded": len(supported), "complete": full_horizon_support}
    for field in macro_fields:
        equal_horizon_metrics[field] = {
            "candidate": math.fsum(r["candidateMetrics"][field] for r in per_horizon)/len(HORIZONS)
                if full_horizon_support and all(r["candidateMetrics"][field] is not None for r in per_horizon) else None,
            "baseline": math.fsum(r["baselineMetrics"][field] for r in per_horizon)/len(HORIZONS)
                if full_horizon_support and all(r["baselineMetrics"][field] is not None for r in per_horizon) else None}
    # Issue scope is compared against saved E106 baseline keys, not D/P availability.
    start, end = TARGET_WINDOWS[phase]
    eligible_radish = [r for r in by_key.values() if r["product"] == "radish" and start <= _day(r["target"]) <= end and r["eligible"]]
    changes = _prediction_changes(eligible_radish, phase)
    all_four = [r for r in radish if all(c in r for c in CONDITIONS)]
    return {"schemaVersion": 1, "phase": phase, "targetWindow": [start.isoformat(), end.isoformat()],
            "radishMetricsByHorizonCondition": metrics,
            "allFourCommonMetrics": _all_four_common_metrics(scored),
            "allFourMonthlyMetrics": _all_four_monthly_metrics(scored, phase),
            "radishPrimary": {"comparison": "C_vs_B", "perHorizon": per_horizon,
                "equalHorizonBlendMapeCandidate": c_equal_mape, "equalHorizonBlendMapeBaseline": b_equal_mape,
                "equalHorizonBlendMaeCandidate": c_equal_mae, "equalHorizonBlendMaeBaseline": b_equal_mae,
                "horizonsWithPositiveCommonPairs": len(supported), "equalHorizonMetricsComplete": full_horizon_support,
                "equalHorizonMetrics": equal_horizon_metrics,
                "commonPairScope": "C and B available on the same positive-actual target keys; unaffected by D/P missingness"},
            "technicalPairMetrics": _technical_metrics(radish, all_four, phase),
            "dm": dm, "dmSummaryByBandwidth": dm_summaries,
            "bootstrap": [_bootstrap(radish, block, (start, end)) for block in BOOTSTRAP_BLOCKS],
            "coverage": _coverage(by_key, normalized_actuals, scored, phase),
            "monthlyCoverage": _monthly_coverage(by_key, normalized_actuals, phase),
            "monthlyMetrics": _monthly_metrics(scored, phase),
            "logMseDecomposition": _decomposition(radish, (COMPARISONS[0],), phase, "C_B_common") + _decomposition(all_four, (COMPARISONS[0], *TECHNICAL_COMPARISONS), phase, "all_four_common"),
            "predictionChangesVsB": changes,
            "interpretation": {"retrospectiveOnly": True, "nonsignificantIsNotEquivalence": True,
                               "C_vs_BPrimaryCommonPairsIndependentOfD_P": True,
                               "C_vs_D_and_C_vs_PRequireAllFourCommonPairs": True},
            "fitDiagnostics": _training_report(fit_diagnostics)}


def _technical_metrics(primary_rows, all_four_rows, phase):
    out = []
    specs = [(COMPARISONS[0], primary_rows, "C_B_common"),
             (COMPARISONS[0], all_four_rows, "all_four_common")]
    specs.extend((item, all_four_rows, "all_four_common") for item in TECHNICAL_COMPARISONS)
    for (comparison, left, right), rows, scope in specs:
        for product in ("radish",):
            for h in HORIZONS:
                cell = [r for r in rows if r["product"] == product and r["horizon"] == h and left in r and right in r]
                for target_month in _month_labels(*TARGET_WINDOWS[phase]):
                    part = [r for r in cell if r["target"].strftime("%Y-%m") == target_month]
                    row = {"comparison": comparison, "product": product, "horizon": h,
                           "sampleScope": scope,
                           "targetMonth": target_month, "n": len(part),
                           "uniqueTargets": len({r["target"] for r in part}),
                           "uniqueOrigins": len({r["origin"] for r in part})}
                    for metric, pred in (("directMae", "directPrediction"), ("blendMae", "blendPrediction")):
                        for side, cond in (("left", left), ("right", right)):
                            errs = [abs(r[cond][pred]-r["actual"]) for r in part]
                            row[f"{side}{metric}"] = math.fsum(errs)/len(errs) if errs else None
                    for side, cond in (("left", left), ("right", right)):
                        direct_pct = [100*abs(r[cond]["directPrediction"]-r["actual"])/r["actual"] for r in part]
                        row[f"{side}DirectMape"] = math.fsum(direct_pct)/len(direct_pct) if direct_pct else None
                        errs = [abs(r[cond]["blendPrediction"]-r["actual"])/r["actual"] for r in part]
                        row[f"{side}BlendMape"] = 100*math.fsum(errs)/len(errs) if errs else None
                    out.append(row)
    return out


def _prediction_changes(rows, phase):
    result = []
    for product in ("radish",):
        for h in HORIZONS:
            for month in _month_labels(*TARGET_WINDOWS[phase]):
                cell = [r for r in rows if r["product"] == product and r["horizon"] == h
                        and r["target"].strftime("%Y-%m") == month]
                for condition in ("D", "C", "P"):
                    deltas = []
                    for r in cell:
                        conds = r["conditions"]
                        left, base = conds[condition], conds["B"]
                        if left["status"] not in ("issued", "fallback") or base["status"] not in ("issued", "fallback"):
                            continue
                        x, y = _finite(left.get("directLogPrediction")), _finite(base.get("directLogPrediction"))
                        if x is not None and y is not None: deltas.append(x-y)
                    result.append({"product": product, "horizon": h, "targetMonth": month,
                                   "condition": condition, "nChangedKeys": sum(d != 0 for d in deltas), "nPairedKeys": len(deltas),
                                   "rmsLogPredictionChange": math.sqrt(math.fsum(d*d for d in deltas)/len(deltas)) if deltas else None,
                                   "maxAbsLogPredictionChange": max((abs(d) for d in deltas), default=None)})
    return result
