"""Replay-gated diagnosis and bounded trend-response correction experiment.

This experiment only rescales movement in the already frozen causal50 forecast.
It does not fit a replacement forecaster or add information beyond past prices.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from . import direct_horizon, equal_info_online_forecasts as forecasts
from . import equal_info_online_score as scoring

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
MANIFEST_SHA256 = "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
PRICES = forecasts.PRICES_PATH
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
PRODUCTS = ("cabbage", "radish")
ALL_PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(scoring.HORIZONS)
LONG_HORIZONS = tuple(range(4, 11))
SCALES = (0.75, 1.0, 1.25)
BANDWIDTHS = scoring.BANDWIDTHS
TRAIN_START = dt.date(2025, 10, 1)
TRAIN_END = dt.date(2026, 3, 20)
CONF_START, CONF_END = dt.date(2026, 4, 1), dt.date(2026, 6, 20)
FINAL_START, FINAL_END = dt.date(2026, 7, 11), dt.date(2026, 9, 27)
MIN_GROUP_ORIGINS = 40
MIN_GROUP_TARGETS = 40
MIN_HORIZON_PAIRS = 30
OUT = ROOT / "outputs/experiments/trend-response-correction-20261010"
_DIAG_SELECTION_CACHE = None


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _positive(value):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) and n > 0 else None


def _date(value):
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _input_identity(manifest_raw, prices_raw):
    if _sha(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("frozen equal-info-online manifest SHA-256 mismatch")
    if _sha(prices_raw) != PRICES_SHA256:
        raise ValueError("frozen historical prices SHA-256 mismatch")
    manifest = json.loads(manifest_raw)
    prices = json.loads(prices_raw)
    # The original archive was already replay-verified and frozen. Validate its
    # embedded digest and complete grid without refitting historical SARIMAs.
    forecasts._validate(manifest)
    return manifest, prices


def _implementation_hashes():
    modules = {"trendResponseCorrection": Path(__file__),
               "baselineForecastProtocol": Path(forecasts.__file__),
               "scoreProtocol": Path(scoring.__file__),
               "directHorizonPrices": Path(direct_horizon.__file__),
               "modelingFeatures": Path(forecasts.modeling.__file__),
               "priceSeries": Path(scoring.price_helper.__file__)}
    return {name: _sha(path.read_bytes()) for name, path in modules.items()}


def _actuals(prices, start, end):
    rows = scoring.price_helper._series(prices)
    return {(product, day): value for product, observations in rows.items()
            for day, value in observations.items() if start <= day <= end}


def _price_features(price_series, product, origin):
    cutoff = origin - dt.timedelta(days=1)
    observations = dict(sorted(price_series.get(product, {}).items()))

    def latest(max_day):
        choices = [day for day, pair in observations.items()
                   if day <= max_day and _positive(pair[0]) is not None]
        return max(choices) if choices else None

    # Match the baseline causal50 Ridge feature anchor exactly.
    p0_target = origin + dt.timedelta(days=1)
    _vector, p0, p0_audit = forecasts._direct_feature(observations, p0_target, 1, {})
    if p0 is None:
        return {"reason": p0_audit.get("reason", "p0_missing_or_older_than_14_days"),
                "p0Date": p0_audit.get("p0Date"), "p0AgeDays": p0_audit.get("p0AgeDays"),
                "cutoff": cutoff.isoformat()}
    p0_day = _date(p0_audit.get("p0Date"))
    anchors = {0: (p0_day, _positive(p0))}
    for lag in (7, 28):
        anchor_cutoff = cutoff - dt.timedelta(days=lag)
        day = latest(anchor_cutoff)
        if day is None or (anchor_cutoff - day).days > 14:
            anchors[lag] = (None, None)
        else:
            anchors[lag] = (day, _positive(observations[day][0]))
    values = {f"p{lag}Date": pair[0].isoformat() if pair[0] else None for lag, pair in anchors.items()}
    if any(value is None for _day, value in anchors.values()):
        return {"reason": "momentum_anchor_missing_or_older_than_14_days", **values,
                "p0": p0, "p7": anchors[7][1], "p28": anchors[28][1]}
    m7 = math.log(anchors[0][1] / anchors[7][1])
    m28 = math.log(anchors[0][1] / anchors[28][1])
    if m7 == 0 or m28 == 0:
        regime = "neutral"
    elif (m7 > 0) == (m28 > 0):
        regime = "aligned"
    else:
        regime = "conflicting"
    # Descriptive only; deliberately excluded from regime and correction.
    cutoff_series = sorted((day, _positive(pair[0])) for day, pair in observations.items()
                           if cutoff - dt.timedelta(days=28) <= day <= cutoff and _positive(pair[0]))
    logs = [math.log(value) for _day, value in cutoff_series]
    mean = math.fsum(logs) / len(logs) if logs else None
    vol = math.sqrt(math.fsum((x - mean) ** 2 for x in logs) / (len(logs) - 1)) if len(logs) > 1 else None
    return {"reason": None, "regime": regime, "p0Date": anchors[0][0].isoformat(),
            "p7Date": anchors[7][0].isoformat(), "p28Date": anchors[28][0].isoformat(),
            "cutoff": cutoff.isoformat(),
            "p0": anchors[0][1], "p7": anchors[7][1], "p28": anchors[28][1],
            "momentum7Log": m7, "momentum28Log": m28,
            "volatility28LogStdDescriptiveOnly": vol}


def _baseline_rows(manifest, key):
    return manifest[key]


def _all_rows(manifest):
    return manifest["confirmationForecasts"] + manifest["evaluationForecasts"]


def _make_record(row, price_series, actuals, *, scale=1.0):
    origin, target, h = _date(row["origin"]), _date(row["target"]), row["horizon"]
    features = _price_features(price_series, row["product"], origin)
    baseline = _positive(row["models"]["causal50"].get("prediction"))
    actual = actuals.get((row["product"], target))
    if baseline is None:
        return {"row": row, "features": features, "baseline": None, "candidate": None,
                "actual": actual, "reason": "baseline_causal50_not_issued"}
    if features.get("reason"):
        return {"row": row, "features": features, "baseline": baseline, "candidate": baseline,
                "actual": actual, "reason": features["reason"]}
    p0 = features["p0"]
    movement = math.log(baseline / p0)
    candidate = _scale_movement(baseline, p0, scale)
    if candidate is None or not math.isfinite(candidate) or candidate <= 0:
        return {"row": row, "features": features, "baseline": baseline, "candidate": baseline,
                "actual": actual, "reason": "nonfinite_scaled_forecast_fallback_baseline"}
    return {"row": row, "features": features, "baseline": baseline, "candidate": candidate,
            "actual": actual, "movementLog": movement, "reason": None}


def _scale_movement(baseline, p0, scale):
    """Apply a bounded scalar to the existing baseline's log movement from p0."""
    if scale not in SCALES:
        raise ValueError("scale must be one of the frozen bounded candidates")
    baseline, p0 = _positive(baseline), _positive(p0)
    if baseline is None or p0 is None:
        return None
    if scale == 1.0:
        return baseline
    try:
        value = p0 * math.exp(scale * math.log(baseline / p0))
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _diagnose(manifest, price_series, actuals):
    blocks = (("train_2025q4", TRAIN_START, dt.date(2025, 12, 31)),
              ("train_2026jan_mar20", dt.date(2026, 1, 1), TRAIN_END))
    result = []
    for product in PRODUCTS:
        for block, start, end in blocks:
            for regime in ("aligned", "conflicting"):
                rows = []
                for row in _all_rows(manifest):
                    target, origin = _date(row["target"]), _date(row["origin"])
                    if row["product"] != product or row["horizon"] not in LONG_HORIZONS or not row["eligible"] or not (start <= target <= end):
                        continue
                    rec = _make_record(row, price_series, actuals)
                    if rec["baseline"] is None or rec["actual"] is None or rec["features"].get("reason") or rec["features"].get("regime") != regime:
                        continue
                    m = rec["movementLog"]
                    r = math.log(rec["actual"] / rec["features"]["p0"])
                    if m == 0 or r == 0:
                        label = "exact_or_zero"
                    elif m * r < 0:
                        label = "wrong_direction"
                    elif abs(m) < abs(r):
                        label = "underreaction"
                    elif abs(m) > abs(r):
                        label = "overreaction"
                    else:
                        label = "exact_or_zero"
                    rows.append({"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                                 "horizon": row["horizon"], "actual": rec["actual"], "baseline": rec["baseline"],
                                 "p0": rec["features"]["p0"], "movementLog": m, "realizedLogChange": r,
                                 "residualLog": r - m, "category": label})
                ordered = sorted(rows, key=lambda r: (r["target"], r["horizon"]))
                months = {}
                for row in ordered:
                    months.setdefault(row["target"][:7], []).append(row)
                result.append({"product": product, "trainBlock": block, "regime": regime,
                               "n": len(ordered), "distinctOrigins": len({r["origin"] for r in ordered}),
                               "distinctTargets": len({r["target"] for r in ordered}),
                               **_diagnostic_stats(ordered),
                               "byHorizon": _horizon_metrics(ordered),
                               "byMonth": {month: _diagnostic_stats(month_rows)
                                           for month, month_rows in sorted(months.items())}})
    return result


def _diagnostic_stats(rows):
    categories = {name: sum(row["category"] == name for row in rows)
                  for name in ("wrong_direction", "underreaction", "overreaction", "exact_or_zero")}
    ae = [abs(row["baseline"] - row["actual"]) for row in rows]
    ape = [100 * error / row["actual"] for error, row in zip(ae, rows)]
    return {"n": len(rows), "directionCounts": categories,
            "meanSignedResidualLog": math.fsum(row["residualLog"] for row in rows) / len(rows) if rows else None,
            "meanAbsoluteResidualLog": math.fsum(abs(row["residualLog"]) for row in rows) / len(rows) if rows else None,
            "baselineMae": math.fsum(ae) / len(ae) if ae else None,
            "baselineMape": math.fsum(ape) / len(ape) if ape else None}


def _horizon_metrics(rows):
    out = []
    for h in LONG_HORIZONS:
        subset = [r for r in rows if r["horizon"] == h]
        ae = [abs(r["baseline"] - r["actual"]) for r in subset]
        ape = [100 * e / r["actual"] for e, r in zip(ae, subset)]
        out.append({"horizon": h, "n": len(subset),
                    "directionCounts": {name: sum(r["category"] == name for r in subset)
                                        for name in ("wrong_direction", "underreaction", "overreaction", "exact_or_zero")},
                    "directionCounts": {name: sum(r["category"] == name for r in subset)
                                        for name in ("wrong_direction", "underreaction", "overreaction", "exact_or_zero")},
                    "meanSignedResidualLog": math.fsum(r["residualLog"] for r in subset) / len(subset) if subset else None,
                    "meanAbsoluteResidualLog": math.fsum(abs(r["residualLog"]) for r in subset) / len(subset) if subset else None,
                    "mae": math.fsum(ae) / len(ae) if ae else None,
                    "mape": math.fsum(ape) / len(ape) if ape else None})
    return out


def _training_records(manifest, price_series, actuals, product, regime):
    records = []
    for row in _all_rows(manifest):
        target = _date(row["target"])
        if (row["product"] != product or row["horizon"] not in LONG_HORIZONS or not row["eligible"]
                or not (TRAIN_START <= target <= TRAIN_END)):
            continue
        rec = _make_record(row, price_series, actuals)
        if rec["baseline"] is None or rec["actual"] is None or rec["features"].get("reason") or rec["features"].get("regime") != regime:
            continue
        rec["origin"] = _date(row["origin"])
        rec["target"] = target
        rec["horizon"] = row["horizon"]
        records.append(rec)
    return records


def _losses(records, scale):
    by_h = []
    for h in LONG_HORIZONS:
        rows = [r for r in records if r["horizon"] == h]
        if not rows:
            continue
        mae_values, mape_values = [], []
        for r in rows:
            p0 = r["features"]["p0"]
            pred = _scale_movement(r["baseline"], p0, scale)
            if pred is None:
                pred = r["baseline"]
            mae_values.append(abs(pred - r["actual"]))
            mape_values.append(100 * abs(pred - r["actual"]) / r["actual"])
        by_h.append({"horizon": h, "n": len(rows), "mae": math.fsum(mae_values) / len(rows),
                     "mape": math.fsum(mape_values) / len(rows)})
    return {"horizonsIncluded": len(by_h),
            "equalHorizonMae": math.fsum(r["mae"] for r in by_h) / len(by_h) if by_h else None,
            "equalHorizonMape": math.fsum(r["mape"] for r in by_h) / len(by_h) if by_h else None,
            "byHorizon": by_h}


def _select(manifest, price_series, actuals, diagnostic):
    selected = []
    for product in PRODUCTS:
        for regime in ("aligned", "conflicting"):
            rows = _training_records(manifest, price_series, actuals, product, regime)
            support = {"n": len(rows), "distinctOrigins": len({r["origin"] for r in rows}),
                       "distinctTargets": len({r["target"] for r in rows}),
                       "byHorizon": {str(h): sum(r["horizon"] == h for r in rows) for h in LONG_HORIZONS}}
            supported = (support["distinctOrigins"] >= MIN_GROUP_ORIGINS and support["distinctTargets"] >= MIN_GROUP_TARGETS
                         and all(support["byHorizon"][str(h)] >= MIN_HORIZON_PAIRS for h in LONG_HORIZONS))
            trials = []
            if supported:
                baseline = _losses(rows, 1.0)
                for scale in SCALES:
                    loss = _losses(rows, scale)
                    feasible = loss["equalHorizonMae"] <= baseline["equalHorizonMae"] + 1e-12
                    trials.append({"scale": scale, **loss, "maeConstraintSatisfied": feasible})
                feasible = [r for r in trials if r["maeConstraintSatisfied"]]
                winner = min(feasible, key=lambda r: (r["equalHorizonMape"], abs(r["scale"] - 1.0), r["scale"])) if feasible else next(r for r in trials if r["scale"] == 1.0)
                scale, reason = winner["scale"], "selected_on_train_mape_subject_to_train_mae_nonincrease" if winner["scale"] != 1 else "selected_scale_one"
            else:
                baseline = _losses(rows, 1.0)
                scale, reason = 1.0, "unsupported_group_fallback_scale_one"
            selected.append({"product": product, "regime": regime, "scale": scale, "reason": reason,
                             "supported": supported, "support": support,
                             "baselineTrainLoss": baseline, "trials": trials})
    return {"schemaVersion": 1, "phase": "selection_only", "targetValuesAttached": False,
            "trainingPeriod": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
            "selectionRule": "one scale per crop/regime minimizes pooled equal-horizon MAPE over h4-10 subject to equal-horizon MAE no higher than scale 1; tie closest to 1 then smaller; unsupported scale 1",
            "diagnosticSha256": _sha(_canonical(diagnostic)), "groups": selected}


def diagnose_and_select(manifest_raw: bytes, prices_raw: bytes):
    global _DIAG_SELECTION_CACHE
    cache_key = (_sha(manifest_raw), _sha(prices_raw))
    if _DIAG_SELECTION_CACHE and _DIAG_SELECTION_CACHE[0] == cache_key:
        return copy.deepcopy(_DIAG_SELECTION_CACHE[1]), copy.deepcopy(_DIAG_SELECTION_CACHE[2])
    manifest, prices = _input_identity(manifest_raw, prices_raw)
    price_series = direct_horizon._observations(prices)
    actuals = _actuals(prices, TRAIN_START, TRAIN_END)
    diagnostic = {"schemaVersion": 1, "phase": "diagnostic", "targetValuesAttached": False,
                  "inputSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256},
                  "implementationSha256": _implementation_hashes(),
                  "diagnostic": _diagnose(manifest, price_series, actuals)}
    diagnostic["diagnosticSha256"] = _sha(_canonical(diagnostic))
    selection = _select(manifest, price_series, actuals, diagnostic)
    selection["inputSha256"] = {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256}
    selection["implementationSha256"] = _implementation_hashes()
    selection["selectionSha256"] = _sha(_canonical(selection))
    _DIAG_SELECTION_CACHE = (cache_key, copy.deepcopy(diagnostic), copy.deepcopy(selection))
    return diagnostic, selection


def _selection_map(selection):
    return {(r["product"], r["regime"]): r for r in selection["groups"]}


def _period_rows(manifest, period):
    if period == "confirmation":
        start, end, origin_start = CONF_START, CONF_END, CONF_START
    elif period == "final":
        start, end, origin_start = FINAL_START, FINAL_END, dt.date(2026, 7, 1)
    else:
        raise ValueError("unknown period")
    return [r for r in manifest["evaluationForecasts"]
            if r.get("product") in ALL_PRODUCTS and r.get("eligible") is True
            and start <= _date(r["target"]) <= end and _date(r["origin"]) >= origin_start]


def build_correction_manifest(manifest_raw: bytes, prices_raw: bytes, selection):
    manifest, prices = _input_identity(manifest_raw, prices_raw)
    s = dict(selection); digest = s.pop("selectionSha256", None)
    if digest != _sha(_canonical(s)):
        raise ValueError("selection digest mismatch")
    expected_diagnostic, expected_selection = diagnose_and_select(manifest_raw, prices_raw)
    if _canonical(expected_selection) != _canonical(selection):
        raise ValueError("selection replay mismatch")
    price_series = direct_horizon._observations(prices)
    mapping = _selection_map(selection)
    rows = []
    for baseline in _period_rows(manifest, "confirmation"):
        product, horizon = baseline["product"], baseline["horizon"]
        features = _price_features(price_series, product, _date(baseline["origin"]))
        base_row = baseline["models"]["causal50"]
        base = _positive(base_row.get("prediction"))
        chosen = mapping.get((product, features.get("regime"))) if product in PRODUCTS else None
        reason = None
        value, scale = base, 1.0
        if product not in PRODUCTS:
            reason = "product_not_in_correction_scope"
        elif horizon <= 3:
            reason = "short_horizon_unchanged"
        elif base is None:
            reason = "baseline_causal50_not_issued"
        elif features.get("reason"):
            reason = features["reason"]
        elif features.get("regime") == "neutral":
            reason = "neutral_trend_fallback_baseline"
        elif chosen is None or not chosen["supported"]:
            reason = "unsupported_training_group_fallback_baseline"
        else:
            scale = chosen["scale"]
            value = _scale_movement(base, features["p0"], scale)
            if value is None or not math.isfinite(value) or value <= 0:
                value, scale, reason = base, 1.0, "nonfinite_correction_fallback_baseline"
        model = {"status": "not_issued", "prediction": None, "reason": "baseline_causal50_not_issued"}
        if base is not None:
            model = {"status": "issued", "prediction": value, "reason": None}
        rows.append({"product": product, "target": baseline["target"], "origin": baseline["origin"],
                     "horizon": horizon, "eligible": True, "model": model,
                     "baselinePrediction": base, "features": features, "scale": scale,
                     "correctionApplied": reason is None, "fallbackReason": reason})
    doc = {"schemaVersion": 1, "phase": "correction_forecast_only", "targetValuesAttached": False,
           "baselineManifestSha256": MANIFEST_SHA256, "selectionSha256": selection["selectionSha256"],
           "inputSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256},
           "implementationSha256": _implementation_hashes(),
           "selectionFrozenBeforeConfirmation": True, "confirmationTargetRange": [CONF_START.isoformat(), CONF_END.isoformat()],
           "confirmationForecasts": rows}
    doc["manifestSha256"] = _sha(_canonical(doc))
    return doc


def _validate_correction_manifest(doc):
    unsigned = dict(doc); digest = unsigned.pop("manifestSha256", None)
    if doc.get("phase") != "correction_forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("expected answer-free correction forecast manifest")
    if digest != _sha(_canonical(unsigned)):
        raise ValueError("correction manifest digest mismatch")
    start, end = CONF_START, CONF_END
    rows = doc.get("confirmationForecasts")
    expected = {(p, start + dt.timedelta(days=i), h) for p in ALL_PRODUCTS
                for i in range((end - start).days + 1) for h in HORIZONS
                if start + dt.timedelta(days=i) - dt.timedelta(days=h) >= start}
    seen = set()
    for row in rows or []:
        target, origin, h = _date(row.get("target")), _date(row.get("origin")), row.get("horizon")
        ident = (row.get("product"), target, h)
        if ident not in expected or ident in seen or origin != target - dt.timedelta(days=h) or row.get("eligible") is not True:
            raise ValueError("invalid correction forecast grid")
        model = row.get("model", {})
        pred = _positive(model.get("prediction"))
        if model.get("status") not in ("issued", "not_issued") or (model["status"] == "issued") != (pred is not None):
            raise ValueError("invalid correction prediction")
        if row.get("correctionApplied") and (h <= 3 or row.get("scale") not in SCALES):
            raise ValueError("invalid correction scope or scale")
        seen.add(ident)
    if seen != expected:
        raise ValueError("incomplete correction forecast grid")


def _score_period(rows, actuals, crops_under_test, *, frozen_crops=None):
    frozen_crops = set(frozen_crops if frozen_crops is not None else crops_under_test)
    metrics, dm, coverage, monthly = [], [], {}, {}
    for crop in ALL_PRODUCTS:
        for h in HORIZONS:
            cell = [r for r in rows if r["product"] == crop and r["horizon"] == h]
            pairs = [{"target": _date(r["target"]), "actual": actuals.get((crop, _date(r["target"]))),
                      "candidate": _positive(r["model"].get("prediction")),
                      "baseline": _positive(r.get("baselinePrediction"))} for r in cell]
            pairs = [r for r in pairs if r["actual"] and r["candidate"] and r["baseline"]]
            a = scoring._metrics([{"actual": r["actual"], "a": r["candidate"], "b": r["baseline"]} for r in pairs], "a", "b")
            metrics.append({"product": crop, "horizon": h, **a})
            positive_actual_targets = {_date(r["target"]) for r in cell
                                       if _positive(actuals.get((crop, _date(r["target"])))) is not None}
            coverage[f"{crop}/{h}"] = {"planned": len(cell), "baselineIssued": sum(_positive(r.get("baselinePrediction")) is not None for r in cell),
                                         "candidateIssued": sum(_positive(r["model"].get("prediction")) is not None for r in cell),
                                         "positiveActualTargets": len(positive_actual_targets),
                                         "distinctTargetDates": len({_date(r["target"]) for r in cell}),
                                         "commonPairs": len(pairs), "coveragePreserved": all((r["baselinePrediction"] is None) == (r["model"]["prediction"] is None) for r in cell)}
            for loss in scoring.LOSSES:
                for bw in BANDWIDTHS:
                    if crop not in crops_under_test or crop not in frozen_crops:
                        reason = "crop_not_selected_by_confirmation" if crop in crops_under_test else "crop_not_in_correction_family"
                        test = {"status": "not_tested_by_protocol", "reason": reason, "n": len(pairs), "statistic": None, "meanLossDifference": None, "pValue": 1.0}
                    else:
                        diffs = {}
                        for r in pairs:
                            ea = abs(r["candidate"] - r["actual"]); eb = abs(r["baseline"] - r["actual"])
                            diffs[r["target"]] = ea - eb if loss == "ae" else (ea - eb) / r["actual"]
                        test = scoring._dm(diffs, bw)
                    dm.append({"product": crop, "horizon": h, "loss": loss, "bandwidth": bw, **test})
            target_months = {}
            for r in cell:
                month = _date(r["target"]).strftime("%Y-%m")
                target_months.setdefault(month, []).append(r)
            for month, month_rows in target_months.items():
                month_pairs = [r for r in month_rows if actuals.get((crop, _date(r["target"]))) and _positive(r["model"].get("prediction")) and _positive(r.get("baselinePrediction"))]
                mm = scoring._metrics([{"actual": actuals[(crop, _date(r["target"]))], "a": r["model"]["prediction"], "b": r["baselinePrediction"]} for r in month_pairs], "a", "b")
                month_actuals = {_date(r["target"]) for r in month_rows
                                 if _positive(actuals.get((crop, _date(r["target"])))) is not None}
                monthly[f"{crop}/{h}/{month}"] = {"planned": len(month_rows),
                                                  "positiveActualTargets": len(month_actuals),
                                                  "baselineIssued": sum(_positive(r.get("baselinePrediction")) is not None for r in month_rows),
                                                  "candidateIssued": sum(_positive(r["model"].get("prediction")) is not None for r in month_rows),
                                                  "commonPairs": len(month_pairs),
                                                  "coveragePreserved": all((r["baselinePrediction"] is None) == (r["model"]["prediction"] is None) for r in month_rows),
                                                  **mm}
    for bw in BANDWIDTHS:
        family = [r for r in dm if r["bandwidth"] == bw and r["product"] in crops_under_test]
        _holm(family)
    macro = []
    for crop in ALL_PRODUCTS:
        cells = [r for r in metrics if r["product"] == crop and r["n"]]
        macro.append({"product": crop, "horizonsIncluded": len(cells),
                      "candidateMae": math.fsum(r["aMae"] for r in cells) / len(cells) if cells else None,
                      "baselineMae": math.fsum(r["bMae"] for r in cells) / len(cells) if cells else None,
                      "candidateMape": math.fsum(r["aMape"] for r in cells) / len(cells) if cells else None,
                      "baselineMape": math.fsum(r["bMape"] for r in cells) / len(cells) if cells else None})
    all_cells = [r for r in metrics if r["n"]]
    overall40 = {"cellsIncluded": len(all_cells),
                 "candidateMae": math.fsum(r["aMae"] for r in all_cells) / len(all_cells) if all_cells else None,
                 "baselineMae": math.fsum(r["bMae"] for r in all_cells) / len(all_cells) if all_cells else None,
                 "candidateMape": math.fsum(r["aMape"] for r in all_cells) / len(all_cells) if all_cells else None,
                 "baselineMape": math.fsum(r["bMape"] for r in all_cells) / len(all_cells) if all_cells else None}
    return {"metrics": metrics, "macro": macro, "overall40CellMacro": overall40,
            "dm": dm, "coverage": coverage, "monthly": monthly,
            "dmSummaryByBandwidth": [{"bandwidth": bw, "familySize": sum(r["bandwidth"] == bw and r["product"] in crops_under_test for r in dm),
                                      "tested": sum(r["bandwidth"] == bw and r["product"] in crops_under_test and r["status"] == "tested" for r in dm),
                                      "significantImprovement": sum(r["bandwidth"] == bw and r["product"] in crops_under_test and r.get("holmSignificant") and r.get("meanLossDifference", 0) < 0 for r in dm),
                                      "significantWorsening": sum(r["bandwidth"] == bw and r["product"] in crops_under_test and r.get("holmSignificant") and r.get("meanLossDifference", 0) > 0 for r in dm),
                                      "inferenceUnavailable": sum(r["bandwidth"] == bw and r["product"] in crops_under_test and r["status"] != "tested" for r in dm)} for bw in BANDWIDTHS]}


def _holm(rows):
    running = 0.0
    for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]["pValue"])):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["pValue"]))
        rows[index]["holmPValue"] = running
        rows[index]["holmSignificant"] = running < scoring.ALPHA


def _gate(confirmation):
    by_crop = {r["product"]: r for r in confirmation["macro"]}
    decisions = []
    for crop in PRODUCTS:
        row = by_crop[crop]
        preserve = all(confirmation["coverage"][f"{crop}/{h}"]["coveragePreserved"] for h in HORIZONS)
        full_key_coverage = all(confirmation["coverage"][f"{crop}/{h}"]["planned"] ==
                                (CONF_END - CONF_START).days + 1 - h for h in HORIZONS)
        passed = (row["horizonsIncluded"] == len(HORIZONS) and preserve and full_key_coverage and row["candidateMape"] is not None
                  and row["candidateMae"] is not None and row["candidateMape"] < row["baselineMape"]
                  and row["candidateMae"] < row["baselineMae"])
        decisions.append({"product": crop, "candidateMape": row["candidateMape"], "baselineMape": row["baselineMape"],
                          "candidateMae": row["candidateMae"], "baselineMae": row["baselineMae"],
                          "coveragePreserved": preserve, "allTenHorizonsSupported": row["horizonsIncluded"] == len(HORIZONS),
                          "fullBaselineKeyCoverage": full_key_coverage,
                          "decision": "freeze_for_final_period" if passed else "stop_candidate_work"})
    return decisions


def score_confirmation(manifest_raw: bytes, prices_raw: bytes, selection, correction_manifest):
    # Replay selection and answer-free confirmation forecasts before reading any target outcomes.
    expected = build_correction_manifest(manifest_raw, prices_raw, selection)
    _validate_correction_manifest(correction_manifest)
    if _canonical(expected) != _canonical(correction_manifest):
        raise ValueError("correction forecast replay mismatch")
    _base, prices = _input_identity(manifest_raw, prices_raw)
    actuals = _actuals(prices, CONF_START, CONF_END)
    rows = correction_manifest["confirmationForecasts"]
    scored = _score_period(rows, actuals, PRODUCTS)
    result = {"schemaVersion": 1, "phase": "confirmation_scored", "targetRange": [CONF_START.isoformat(), CONF_END.isoformat()],
              "inputSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256},
              "implementationSha256": _implementation_hashes(), **scored,
              "decisions": _gate(scored), "interpretation": "Retrospective confirmation within an already examined historical archive; no claim of independent prospective validation."}
    result["reportSha256"] = _sha(_canonical(result))
    return result


def build_final_manifest(manifest_raw: bytes, prices_raw: bytes, selection, confirmation_report):
    _expected_diagnostic, expected_selection = diagnose_and_select(manifest_raw, prices_raw)
    if _canonical(expected_selection) != _canonical(selection):
        raise ValueError("selection replay mismatch")
    _base_manifest, _base_prices = _input_identity(manifest_raw, prices_raw)
    replayed_confirmation_forecasts = build_correction_manifest(manifest_raw, prices_raw, selection)
    replayed_confirmation = score_confirmation(manifest_raw, prices_raw, selection, replayed_confirmation_forecasts)
    if _canonical(replayed_confirmation) != _canonical(confirmation_report):
        raise ValueError("confirmation report replay mismatch")
    decisions = {r["product"]: r["decision"] for r in confirmation_report["decisions"]}
    selected = [p for p in PRODUCTS if decisions.get(p) == "freeze_for_final_period"]
    if not selected:
        raise ValueError("confirmation gate passed for no crop; stop before final-period scoring")
    manifest, prices = _input_identity(manifest_raw, prices_raw)
    selection_map = _selection_map(selection)
    series = direct_horizon._observations(prices)
    rows = []
    for baseline in _period_rows(manifest, "final"):
        product, h = baseline["product"], baseline["horizon"]
        features = _price_features(series, product, _date(baseline["origin"]))
        base = _positive(baseline["models"]["causal50"].get("prediction"))
        chosen = selection_map.get((product, features.get("regime"))) if product in selected else None
        reason = None; value, scale = base, 1.0
        if product not in selected: reason = "crop_not_selected_by_confirmation"
        elif h <= 3: reason = "short_horizon_unchanged"
        elif base is None: reason = "baseline_causal50_not_issued"
        elif features.get("reason"): reason = features["reason"]
        elif features.get("regime") == "neutral": reason = "neutral_trend_fallback_baseline"
        elif chosen is None or not chosen["supported"]: reason = "unsupported_training_group_fallback_baseline"
        else:
            scale = chosen["scale"]; value = _scale_movement(base, features["p0"], scale)
            if value is None or not math.isfinite(value) or value <= 0:
                value, scale, reason = base, 1.0, "nonfinite_correction_fallback_baseline"
        rows.append({"product": product, "target": baseline["target"], "origin": baseline["origin"], "horizon": h,
                     "model": {"status": "issued", "prediction": value, "reason": None} if base is not None else {"status": "not_issued", "prediction": None, "reason": "baseline_causal50_not_issued"},
                     "baselinePrediction": base, "features": features, "scale": scale,
                     "correctionApplied": reason is None, "fallbackReason": reason})
    doc = {"schemaVersion": 1, "phase": "final_forecast_only", "targetValuesAttached": False,
           "baselineManifestSha256": MANIFEST_SHA256, "selectionSha256": selection["selectionSha256"],
           "confirmationReportSha256": confirmation_report["reportSha256"], "frozenCrops": selected,
           "inputSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256},
           "implementationSha256": _implementation_hashes(), "finalTargetRange": [FINAL_START.isoformat(), FINAL_END.isoformat()],
           "finalForecasts": rows}
    doc["manifestSha256"] = _sha(_canonical(doc))
    return doc


def score_final(manifest_raw: bytes, prices_raw: bytes, selection, confirmation_report, final_manifest):
    expected = build_final_manifest(manifest_raw, prices_raw, selection, confirmation_report)
    if _canonical(expected) != _canonical(final_manifest):
        raise ValueError("final correction forecast replay mismatch")
    unsigned = dict(final_manifest); digest = unsigned.pop("manifestSha256", None)
    if digest != _sha(_canonical(unsigned)):
        raise ValueError("final correction manifest digest mismatch")
    _base, prices = _input_identity(manifest_raw, prices_raw)
    scored = _score_period(final_manifest["finalForecasts"], _actuals(prices, FINAL_START, FINAL_END), PRODUCTS,
                           frozen_crops=final_manifest["frozenCrops"])
    result = {"schemaVersion": 1, "phase": "final_scored", "targetRange": [FINAL_START.isoformat(), FINAL_END.isoformat()],
              "frozenCrops": final_manifest["frozenCrops"],
              "inputSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256},
              "implementationSha256": _implementation_hashes(), **scored,
              "interpretation": "Retrospective final-period assessment within the frozen archive; candidate selection was frozen before this period."}
    result["reportSha256"] = _sha(_canonical(result))
    return result


def write_exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="produce frozen diagnosis/selection and replay-gated period manifests/reports")
    run.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest_raw, prices_raw = MANIFEST.read_bytes(), PRICES.read_bytes()
        diagnostic, selection = diagnose_and_select(manifest_raw, prices_raw)
        correction_manifest = build_correction_manifest(manifest_raw, prices_raw, selection)
        output = args.output_dir
        write_exclusive(output / "diagnostic.json", diagnostic)
        write_exclusive(output / "selection.json", selection)
        write_exclusive(output / "confirmation-manifest.json", correction_manifest)
        confirmation = score_confirmation(manifest_raw, prices_raw, selection, correction_manifest)
        write_exclusive(output / "confirmation-report.json", confirmation)
        selected = [r["product"] for r in confirmation["decisions"] if r["decision"] == "freeze_for_final_period"]
        if selected:
            final_manifest = build_final_manifest(manifest_raw, prices_raw, selection, confirmation)
            write_exclusive(output / "final-manifest.json", final_manifest)
            final_report = score_final(manifest_raw, prices_raw, selection, confirmation, final_manifest)
            write_exclusive(output / "final-report.json", final_report)
        print(json.dumps({"outputDir": str(output), "selectedCrops": selected}, ensure_ascii=False, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
