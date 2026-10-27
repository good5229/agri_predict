"""Frozen five-crop residual specialists using crop-specific existing signals."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "market": Path("outputs/experiments/market-recompose-20261007/report.json"),
    "radish": Path("outputs/experiments/radish-season-20261007/report.json"),
    "garlic": Path("outputs/experiments/garlic-mix-20261007/report.json"),
    "pepper": Path("outputs/experiments/seasonal-analog-20261007/report.json"),
    "historical": Path("outputs/experiments/ridge-sarima-weight-cv-20261008/historical-components.json"),
    "test_components": Path("outputs/experiments/ridge-sarima-component-cv-20261008/forecast-manifest.json"),
    "direct": Path("outputs/experiments/direct-horizon-20261006/report.json"),
}
INPUT_HASHES = {
    "market": "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581",
    "radish": "994265614c69f17e29a0a7519a2db2300990f58366a2357e83126b7d45450a1e",
    "garlic": "483e5dfe7c8114e4f37c954bf640d03b6d407bc4004d4f3f17890fa76798a359",
    "pepper": "2f754c4ac3cc17b3894441f313e322dc8ca3ca91a9b5274a2d63201e7a896089",
    "historical": "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74",
    "test_components": "a85225b580cc6d55bcd5d49c88ee458520cf6de8ace61250fff36ef1a3d91ebd",
    "direct": "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67",
}
CROPS = ("cabbage", "radish", "onion", "garlic", "pepper")
HORIZONS = tuple(range(1, 11))
A = (dt.date(2024, 9, 25), dt.date(2025, 8, 31))
B = (dt.date(2025, 9, 25), dt.date(2026, 9, 24))
WINDOWS = {"A": A, "B": B}
CANDIDATE = "candidate"
ABLATION = "ablation"
DIRECT = "direct"
SARIMA = "sarima"
FIXED = "fixed50"
EXTRA = "extra_signal"
METHODS = (CANDIDATE, ABLATION, DIRECT, SARIMA, FIXED, EXTRA)
COMPARATORS = (ABLATION, DIRECT, SARIMA, FIXED, EXTRA)
BANDWIDTHS = (9, 28)
MIN_UNIQUE_TARGETS = 60
RESIDUAL_CLIP = 0.12
ALPHA = 0.05
EXPECTED_A = {"cabbage": 2870, "radish": 2870, "onion": 2860, "garlic": 1950, "pepper": 820}
EXPECTED_B = {"cabbage": 3040, "radish": 3050, "onion": 3020, "garlic": 2630, "pepper": 768}
SIGNAL_SOURCE = {"cabbage": ("market", "candidatePrediction"), "radish": ("radish", "candidate"),
                 "onion": ("market", "candidatePrediction"), "garlic": ("garlic", "garlicMix"),
                 "pepper": ("pepper", "mix50")}
PROTOCOL = {
    "id": "five-crop-specialist-residual-2026-10-09",
    "inputSha256": INPUT_HASHES,
    "windows": {"A": [A[0].isoformat(), A[1].isoformat()], "B": [B[0].isoformat(), B[1].isoformat()]},
    "model": "per crop, separate StandardScaler+Ridge(alpha=100) candidate and common-feature ablation; train only A, predict B",
    "target": "log(actual/fixed50); prediction=fixed50*exp(clip(residual,-0.12,+0.12))",
    "baseFeatures": ["log(direct/sarima)", "horizon/10", "sin(2pi*doy/365.25)", "cos(2pi*doy/365.25)"],
    "extraFeatures": ["log(extra/fixed50)", "abs(log(extra/fixed50))", "log(extra/fixed50)*horizon/10"],
    "inference": "5 crops x 10 horizons x 5 comparators x AE/APE=500 two-sided tests per calendar Bartlett HAC9/HAC28 bandwidth; separate Holm .05; n<30 or invalid variance remains p=1",
    "limitations": "retrospective revised material and previously viewed test period; not independent confirmation or official national score",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _positive(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) and n > 0 else None


def _date(value: Any) -> dt.date:
    if not isinstance(value, str):
        raise ValueError("date must be ISO text")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be ISO text") from exc


def _window(target: dt.date) -> str | None:
    return next((name for name, (start, end) in WINDOWS.items() if start <= target <= end), None)


def _key(row: dict) -> tuple[str, int, str, str]:
    crop, horizon = row.get("product"), row.get("horizon")
    if crop not in CROPS or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError("invalid crop/horizon key")
    target, origin = _date(row.get("target")), _date(row.get("origin"))
    if target-origin != dt.timedelta(days=horizon):
        raise ValueError("origin/target/horizon mismatch")
    return crop, horizon, origin.isoformat(), target.isoformat()


def _load_inputs(root: Path = ROOT) -> dict[str, dict]:
    docs = {}
    for name, rel in PATHS.items():
        raw = (root / rel).read_bytes()
        if _sha(raw) != INPUT_HASHES[name]:
            raise ValueError(f"pinned {name} source SHA-256 mismatch")
        docs[name] = json.loads(raw)
    if docs["historical"].get("phase") != "historical_components_only" or docs["historical"].get("targetValuesAttached") is not False:
        raise ValueError("historical component input is not answer-free")
    if docs["test_components"].get("phase") != "selected_test_forecast_only" or docs["test_components"].get("targetValuesAttached") is not False:
        raise ValueError("B component input is not answer-free")
    for name in ("historical", "test_components"):
        _verify_content_hash(docs[name], name)
    return docs


def _verify_content_hash(doc: dict, label: str) -> None:
    body = dict(doc)
    if body.pop("manifestSha256", None) != _hash(body):
        raise ValueError(f"{label} canonical content hash mismatch")


def _signal_rows(docs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """Return A/B source signal maps while touching only feature fields, never B actuals."""
    result = {"A": {}, "B": {}}
    for crop in CROPS:
        source_name, signal_field = SIGNAL_SOURCE[crop]
        source = docs[source_name]
        if crop in ("cabbage", "onion"):
            rows_a = source.get("originEvidence", [])
            rows_b = rows_a
        elif crop == "pepper":
            rows_a = source.get("selectionRows", [])
            rows_b = source.get("pairedRows", [])
        else:
            rows_a = source.get("pairedRows", [])
            rows_b = rows_a
        for window, rows in (("A", rows_a), ("B", rows_b)):
            indexed = {}
            # Several source reports carry the original direct forecast only in
            # pairedRows (market originEvidence intentionally omits it).
            paired_direct = {}
            direct_source_rows = (rows if crop == "pepper" else source.get("pairedRows", []))
            for direct_row in direct_source_rows:
                direct_product = direct_row.get("product", "radish" if crop == "radish" else None)
                if direct_product != crop or _window(_date(direct_row.get("target"))) != window:
                    continue
                direct_key = _key({"product": crop, "horizon": direct_row.get("horizon"),
                                   "origin": direct_row.get("origin"), "target": direct_row.get("target")})
                if direct_key in paired_direct:
                    raise ValueError(f"duplicate {window} source direct key for {crop}")
                paired_direct[direct_key] = _positive(direct_row.get("directRidge"))
            for raw in rows:
                # radish report is a dedicated one-crop report without a product field.
                product = raw.get("product", "radish" if crop == "radish" else None)
                if product != crop:
                    continue
                target = _date(raw.get("target"))
                if _window(target) != window:
                    continue
                row = {"product": crop, "horizon": raw.get("horizon"),
                       "origin": raw.get("origin"), "target": raw.get("target")}
                key = _key(row)
                if key in indexed:
                    raise ValueError(f"duplicate {window} extra-signal key for {crop}")
                signal = _positive(raw.get(signal_field))
                # B actual is deliberately not read into this record.
                fallback_reason = raw.get("fallbackReason", raw.get("fallbackReasons"))
                if "fallback" in raw:
                    fallback = bool(raw["fallback"])
                    fallback_status = "known"
                elif "fallbackReason" in raw:
                    fallback = fallback_reason is not None
                    fallback_status = "known"
                else:
                    fallback = None
                    fallback_status = "unknown"
                indexed[key] = {"key": key, "signal": signal,
                                "actual": _positive(raw.get("actual")) if window == "A" else None,
                                "sourceDirect": paired_direct.get(key),
                                "sourceFallback": fallback,
                                "sourceFallbackStatus": fallback_status,
                                "sourceFallbackReason": fallback_reason}
            if set(paired_direct) != set(indexed):
                raise ValueError(f"{window}/{crop} exact source signal/direct key sets differ: {len(indexed)}/{len(paired_direct)}")
            result[window][crop] = indexed
    return result


def _component_rows(docs: dict[str, dict], window: str) -> dict[str, dict]:
    source = docs["historical"]["forecasts"] if window == "A" else docs["test_components"]["forecasts"]
    result = {}
    for row in source:
        crop = row.get("product")
        if crop not in CROPS or _window(_date(row.get("target"))) != window:
            continue
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate {window} price-component key")
        if window == "A":
            direct, sarima = row.get("direct"), row.get("sarima")
            if row.get("status") != "available":
                raise ValueError(f"historical component unavailable at {key}")
            fixed = (float(direct)+float(sarima))/2 if _positive(direct) and _positive(sarima) else None
        else:
            pred = row.get("predictions", {})
            direct, sarima, fixed = pred.get("direct"), pred.get("sarima"), pred.get("fixed50")
        result[key] = {"key": key, "direct": _positive(direct), "sarima": _positive(sarima), "fixed50": _positive(fixed)}
    return result


def _direct_rows(docs: dict[str, dict], window: str, *, include_b_actual: bool = False) -> dict[str, dict]:
    rows = docs["direct"].get("pairedRows", [])
    result = {}
    for row in rows:
        crop = row.get("product")
        if crop not in CROPS:
            continue
        target = _date(row.get("target"))
        if _window(target) != window:
            continue
        horizon = row.get("horizon")
        origin = target-dt.timedelta(days=horizon)
        normalized = {"product": crop, "horizon": horizon, "origin": origin.isoformat(), "target": target.isoformat()}
        key = _key(normalized)
        if key in result:
            raise ValueError(f"duplicate {window} direct-report key")
        result[key] = {"key": key, "direct": _positive(row.get("direct")),
                       "actual": _positive(row.get("actual")) if window == "A" else None,
                       "actualForScoring": row.get("actual") if window == "B" and include_b_actual else None}
    return result


def _base_features(key: tuple, direct: float, sarima: float) -> list[float]:
    _, h, _, target_text = key
    target = _date(target_text)
    day = target.timetuple().tm_yday
    angle = 2*math.pi*day/365.25
    return [math.log(direct/sarima), h/10., math.sin(angle), math.cos(angle)]


def _features(key: tuple, direct: float, sarima: float, fixed: float, extra: float) -> tuple[list[float], list[float]]:
    base = _base_features(key, direct, sarima)
    x = math.log(extra/fixed)
    return base+[x, abs(x), x*key[1]/10.], base


def _compose(docs: dict[str, dict]) -> tuple[list[dict], dict]:
    signals = _signal_rows(docs)
    components = {w: _component_rows(docs, w) for w in WINDOWS}
    directs = {w: _direct_rows(docs, w) for w in WINDOWS}
    training, test = {}, {}
    for window in WINDOWS:
        expected = EXPECTED_A if window == "A" else EXPECTED_B
        for crop in CROPS:
            comp = {k:v for k,v in components[window].items() if k[0] == crop}
            sig = signals[window][crop]
            direct = {k:v for k,v in directs[window].items() if k[0] == crop}
            if set(comp) != set(sig) or set(comp) != set(direct):
                raise ValueError(f"{window}/{crop} exact component, signal and direct key sets differ: {len(comp)}/{len(sig)}/{len(direct)}")
            if len(comp) != expected[crop]:
                raise ValueError(f"{window}/{crop} planned key count mismatch: expected {expected[crop]}, got {len(comp)}")
            rows = []
            for key in sorted(comp):
                c, s, d = comp[key], sig[key], direct[key]
                if c["direct"] is None or d["direct"] is None or not math.isclose(c["direct"], d["direct"], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError(f"{window}/{crop} direct prediction parity failure at {key}")
                if s["sourceDirect"] is None or not math.isclose(c["direct"], s["sourceDirect"], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError(f"{window}/{crop} source direct prediction parity failure at {key}")
                if c["direct"] is None or c["fixed50"] is None or (window == "A" and c["sarima"] is None):
                    raise ValueError(f"{window}/{crop} nonpositive direct/fixed50 or historical SARIMA component at {key}")
                if window == "A":
                    if s["actual"] is None or d["actual"] is None or not math.isclose(s["actual"], d["actual"], rel_tol=1e-10, abs_tol=1e-8):
                        raise ValueError(f"A/{crop} training actual parity failure at {key}")
                rows.append({"key": key, "direct": c["direct"], "sarima": c["sarima"], "fixed50": c["fixed50"],
                             "extra": s["signal"], "actual": s["actual"] if window == "A" else None,
                             "sourceDirect": s["sourceDirect"], "sourceFallback": s["sourceFallback"],
                             "sourceFallbackStatus": s["sourceFallbackStatus"],
                             "sourceFallbackReason": s["sourceFallbackReason"]})
            (training if window == "A" else test)[crop] = rows
    models, train_audit = {}, {}
    for crop in CROPS:
        rows = training[crop]
        eligible = [r for r in rows if r["extra"] is not None and r["actual"] is not None]
        distinct = len({r["key"][3] for r in eligible})
        audit = {"plannedTrainingRows": len(rows), "eligibleTrainingRows": len(eligible),
                 "uniqueTrainingTargets": distinct, "minimumUniqueTargets": MIN_UNIQUE_TARGETS,
                 "trainingTargetStart": min((r["key"][3] for r in eligible), default=None),
                 "trainingTargetEnd": max((r["key"][3] for r in eligible), default=None),
                 "excludedTrainingRows": len(rows)-len(eligible)}
        signal_source_name, _ = SIGNAL_SOURCE[crop]
        audit["trainingSignalSourcePath"] = str(PATHS[signal_source_name])
        audit["trainingSignalSourceSha256"] = INPUT_HASHES[signal_source_name]
        audit["trainingRowProvenance"] = [{"key": list(r["key"]), "extraSignal": r["extra"],
                                           "baseFeatures": _base_features(r["key"],r["direct"],r["sarima"]),
                                           "candidateFeatures": _features(r["key"],r["direct"],r["sarima"],r["fixed50"],r["extra"])[0],
                                           "targetLogRatio": math.log(r["actual"]/r["fixed50"])} for r in eligible]
        if distinct < MIN_UNIQUE_TARGETS:
            models[crop] = None
            audit.update({"status": "fallback_insufficient_training_targets", "failureReason": "fewer_than_60_unique_training_targets"})
            train_audit[crop] = audit
            continue
        try:
            Xc, Xa, y = [], [], []
            for r in eligible:
                fc, fa = _features(r["key"], r["direct"], r["sarima"], r["fixed50"], r["extra"])
                Xc.append(fc); Xa.append(fa); y.append(math.log(r["actual"]/r["fixed50"]))
            candidate_model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
            ablation_model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
            candidate_model.fit(np.asarray(Xc), np.asarray(y))
            ablation_model.fit(np.asarray(Xa), np.asarray(y))
            models[crop] = (candidate_model, ablation_model)
            audit.update({"status": "fitted", "ridgeAlpha": 100., "candidateFeatureNames": PROTOCOL["baseFeatures"]+PROTOCOL["extraFeatures"],
                          "ablationFeatureNames": PROTOCOL["baseFeatures"],
                          "candidateScalerMean": candidate_model.named_steps["standardscaler"].mean_.tolist(),
                          "candidateScalerScale": candidate_model.named_steps["standardscaler"].scale_.tolist(),
                          "candidateCoefficients": candidate_model.named_steps["ridge"].coef_.tolist(),
                          "candidateIntercept": float(candidate_model.named_steps["ridge"].intercept_),
                          "ablationScalerMean": ablation_model.named_steps["standardscaler"].mean_.tolist(),
                          "ablationScalerScale": ablation_model.named_steps["standardscaler"].scale_.tolist(),
                          "ablationCoefficients": ablation_model.named_steps["ridge"].coef_.tolist(),
                          "ablationIntercept": float(ablation_model.named_steps["ridge"].intercept_)})
        except Exception as exc:
            models[crop] = None
            audit.update({"status": "fallback_fit_failure", "failureReason": f"{type(exc).__name__}:{exc}"})
        train_audit[crop] = audit
    forecasts = []
    fallback_counts = {crop: {"candidate": 0, "ablation": 0, "reasons": {}} for crop in CROPS}
    for crop in CROPS:
        model_pair = models[crop]
        for r in test[crop]:
            key = r["key"]
            reason = None
            if r["extra"] is None:
                reason = "extra_signal_not_positive_finite"
            elif r["sarima"] is None:
                reason = "sarima_component_not_positive_finite"
            elif model_pair is None:
                reason = train_audit[crop].get("failureReason", "training_model_unavailable")
            preds = {}
            feature_candidate = feature_ablation = None
            if reason is None:
                feature_candidate, feature_ablation = _features(key, r["direct"], r["sarima"], r["fixed50"], r["extra"])
                try:
                    candidate_residual = float(model_pair[0].predict(np.asarray([feature_candidate]))[0])
                    ablation_residual = float(model_pair[1].predict(np.asarray([feature_ablation]))[0])
                    if not math.isfinite(candidate_residual) or not math.isfinite(ablation_residual):
                        raise ValueError("nonfinite_model_residual")
                    candidate_residual = max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, candidate_residual))
                    ablation_residual = max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, ablation_residual))
                    preds[CANDIDATE] = r["fixed50"]*math.exp(candidate_residual)
                    preds[ABLATION] = r["fixed50"]*math.exp(ablation_residual)
                    if _positive(preds[CANDIDATE]) is None or _positive(preds[ABLATION]) is None:
                        raise ValueError("prediction_output_not_positive_finite")
                except Exception as exc:
                    reason = f"prediction_failure:{type(exc).__name__}:{exc}"
            if reason is not None:
                preds[CANDIDATE] = preds[ABLATION] = r["fixed50"]
                for m in (CANDIDATE, ABLATION):
                    fallback_counts[crop][m] += 1
                fallback_counts[crop]["reasons"][reason] = fallback_counts[crop]["reasons"].get(reason, 0)+1
            preds.update({DIRECT: r["direct"], SARIMA: r["sarima"], FIXED: r["fixed50"], EXTRA: r["extra"]})
            forecasts.append({"product": crop, "horizon": key[1], "origin": key[2], "target": key[3],
                              "predictions": preds,
                              "features": {"candidate": feature_candidate, "ablation": feature_ablation},
                              "fallback": reason is not None, "fallbackReason": reason,
                              "sourceSignalPath": str(PATHS[SIGNAL_SOURCE[crop][0]]),
                              "sourceSignalSha256": INPUT_HASHES[SIGNAL_SOURCE[crop][0]],
                              "sourceDirect": r["sourceDirect"], "sourceFallback": r["sourceFallback"],
                              "sourceFallbackStatus": r["sourceFallbackStatus"],
                              "sourceFallbackReason": r["sourceFallbackReason"]})
    return forecasts, {"training": train_audit, "testFallbacks": fallback_counts}


def build_manifest(*, root: Path = ROOT) -> dict:
    docs = _load_inputs(root)
    forecasts, audits = _compose(docs)
    result = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
              "inputSha256": dict(INPUT_HASHES), "trainingAudit": audits["training"], "testFallbackAudit": audits["testFallbacks"],
              "plannedForecastCount": len(forecasts), "forecasts": forecasts}
    result["manifestSha256"] = _hash(result)
    return result


def verify_manifest(doc: dict) -> None:
    if doc.get("protocol") != PROTOCOL or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("specialist manifest must be answer-free under the frozen protocol")
    if doc.get("inputSha256") != INPUT_HASHES:
        raise ValueError("specialist manifest input digests differ from frozen values")
    forecasts = doc.get("forecasts")
    if not isinstance(forecasts, list) or len(forecasts) != sum(EXPECTED_B.values()) or doc.get("plannedForecastCount") != len(forecasts):
        raise ValueError("specialist manifest must cover exact 12,508 B keys")
    body = dict(doc)
    if body.pop("manifestSha256", None) != _hash(body):
        raise ValueError("specialist manifest content hash mismatch")
    seen = set()
    for row in forecasts:
        key = _key(row)
        if _window(_date(row["target"])) != "B" or key in seen:
            raise ValueError("specialist manifest contains duplicate or non-B key")
        seen.add(key)
        if "actual" in row or "targetValue" in row:
            raise ValueError("actual leaked into answer-free specialist manifest")
        predictions = row.get("predictions", {})
        if set(predictions) != set(METHODS) or any(_positive(predictions.get(m)) is None for m in (CANDIDATE,ABLATION,DIRECT,FIXED)):
            raise ValueError("specialist prediction set is incomplete or nonpositive")
        if predictions[SARIMA] is not None and _positive(predictions[SARIMA]) is None:
            raise ValueError("SARIMA comparator must be positive or explicitly unavailable")
        if predictions[EXTRA] is None:
            if not row.get("fallback") or row.get("fallbackReason") != "extra_signal_not_positive_finite":
                raise ValueError("missing extra signal lacks the frozen paired fallback reason")
        elif _positive(predictions[EXTRA]) is None:
            raise ValueError("extra signal prediction is nonpositive")
        if predictions[SARIMA] is None:
            expected_reason = "extra_signal_not_positive_finite" if predictions[EXTRA] is None else "sarima_component_not_positive_finite"
            if (not row.get("fallback") or row.get("fallbackReason") != expected_reason
                    or predictions[CANDIDATE] != predictions[FIXED] or predictions[ABLATION] != predictions[FIXED]):
                raise ValueError("unavailable SARIMA must remain missing and trigger paired fixed50 fallback")


def _dm(diffs: dict[dt.date, float], bandwidth: int) -> dict:
    n = len(diffs)
    mean = sum(diffs.values())/n if n else None
    if n < 30:
        return {"status": "inference_unavailable_insufficient_pairs", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    centered = {d: v-mean for d, v in diffs.items()}
    total = sum(v*v for v in centered.values())
    for lag in range(1, bandwidth+1):
        total += 2*(1-lag/(bandwidth+1))*sum(v*centered.get(d+dt.timedelta(days=lag), 0.) for d,v in centered.items())
    variance = total/(n*n)
    if not math.isfinite(variance) or variance <= 0:
        return {"status": "inference_unavailable_nonpositive_variance", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    statistic = mean/math.sqrt(variance)
    return {"status": "available", "n": n, "meanLossDifference": mean, "statistic": statistic, "pValue": math.erfc(abs(statistic)/math.sqrt(2))}


def _holm(tests: list[dict]) -> None:
    for bw in BANDWIDTHS:
        family = [t["dm"][str(bw)] for t in tests]
        if len(family) != 500:
            raise AssertionError("frozen 500-test Holm family mismatch")
        ordered = sorted(family, key=lambda x: x["pValue"])
        running = 0.
        for rank, row in enumerate(ordered):
            running = max(running, min(1., (len(family)-rank)*row["pValue"]))
            row["holmAdjustedPValue"] = running
            row["holmSignificant"] = row["status"] == "available" and running < ALPHA
            row["conclusion"] = ("inference_unavailable" if row["status"] != "available" else
                                 "improvement" if row["holmSignificant"] and row["meanLossDifference"] < 0 else
                                 "worsening" if row["holmSignificant"] and row["meanLossDifference"] > 0 else "inconclusive")


def _joint(labels: list[str]) -> str:
    if "inference_unavailable" in labels: return "inference_unavailable"
    if all(x == "improvement" for x in labels): return "joint_improvement"
    if all(x == "worsening" for x in labels): return "joint_worsening"
    if "improvement" in labels and "worsening" in labels: return "mixed"
    return "inconclusive"


def _score(doc: dict, actual_map: dict[tuple, float], fallback_audit: dict) -> dict:
    scored, exclusions = [], {}
    for row in doc["forecasts"]:
        key = _key(row)
        actual = _positive(actual_map.get(key))
        if actual is None:
            exclusions["missing_or_nonpositive_target"] = exclusions.get("missing_or_nonpositive_target", 0)+1
            continue
        if any(_positive(row["predictions"].get(m)) is None for m in METHODS):
            exclusions["incomplete_common_method_forecast"] = exclusions.get("incomplete_common_method_forecast", 0)+1
            continue
        scored.append({**row, "actual": actual})
    cells, tests = [], []
    metrics_by_cell = {}
    for crop in CROPS:
        for h in HORIZONS:
            cell = [r for r in scored if r["product"] == crop and r["horizon"] == h]
            metric = {m: dh._metrics([r["actual"] for r in cell], [r["predictions"][m] for r in cell]) for m in METHODS}
            metrics_by_cell[(crop,h)] = metric
            cells.append({"product": crop, "horizon": h, "n": len(cell), "metrics": metric})
            for comp in COMPARATORS:
                for loss in ("ae", "ape"):
                    diffs = {}
                    for r in cell:
                        y, candidate, baseline = r["actual"], r["predictions"][CANDIDATE], r["predictions"][comp]
                        diffs[_date(r["target"])] = abs(y-candidate)-abs(y-baseline) if loss == "ae" else 100*(abs(y-candidate)-abs(y-baseline))/y
                    tests.append({"product": crop, "horizon": h, "comparator": comp, "loss": loss, "n": len(diffs),
                                  "candidateMetric": metric[CANDIDATE]["mae" if loss == "ae" else "mape"],
                                  "comparatorMetric": metric[comp]["mae" if loss == "ae" else "mape"],
                                  "dm": {str(bw): _dm(diffs,bw) for bw in BANDWIDTHS}})
    _holm(tests)
    complete_cells = all(next(c for c in cells if c["product"] == crop and c["horizon"] == h)["n"] >= 30 for crop in CROPS for h in HORIZONS)
    macro = {m: {loss: sum(metrics_by_cell[(crop,h)][m][loss] for crop in CROPS for h in HORIZONS)/50 if complete_cells else None
                 for loss in ("mae","mape")} for m in METHODS}
    per_crop = {crop: {m: {loss: sum(metrics_by_cell[(crop,h)][m][loss] for h in HORIZONS)/10
                                   if all(metrics_by_cell[(crop,h)][m][loss] is not None for h in HORIZONS) else None
                           for loss in ("mae","mape")} for m in METHODS} for crop in CROPS}
    joint = []
    idx = {(t["product"],t["horizon"],t["comparator"],t["loss"]):t for t in tests}
    for crop in CROPS:
        for h in HORIZONS:
            for comp in COMPARATORS:
                vals = [idx[(crop,h,comp,loss)]["dm"][str(bw)]["conclusion"] for loss in ("ae","ape") for bw in BANDWIDTHS]
                joint.append({"product":crop,"horizon":h,"comparator":comp,
                              "conclusions":dict(zip(("ae_hac9","ae_hac28","ape_hac9","ape_hac28"),vals)),"jointConclusion":_joint(vals)})
    coverage = {}
    for crop in CROPS:
        for h in HORIZONS:
            plan = [r for r in doc["forecasts"] if r["product"] == crop and r["horizon"] == h]
            valid = [r for r in scored if r["product"] == crop and r["horizon"] == h]
            valid_targets = sum(_positive(actual_map.get(_key(r))) is not None for r in plan)
            coverage[f"{crop}/{h}"] = {"planned":len(plan),"validPositiveTargets":valid_targets,
                "issued":{m:sum(_positive(r["predictions"].get(m)) is not None for r in plan) for m in METHODS},"commonScored":len(valid),
                "candidateFallback":sum(r["fallback"] for r in plan)}
    report = {"planned":len(doc["forecasts"]),"commonScoredPairs":len(scored),"exclusionsByState":exclusions,
              "complete50CellsAt30Pairs":complete_cells,"cells":cells,"macro50CellEqualWeight":macro,
              "perCrop10HorizonEqualWeight":per_crop,"coverageByCell":coverage,
              "dmComparisons":tests,"holmFamilySizePerBandwidth":500,"jointConclusions":joint,
              "fallbackAudit":fallback_audit}
    report["gate"] = _gate(report)
    return report


def _gate(report: dict) -> dict:
    cells, per_crop = report["cells"], report["perCrop10HorizonEqualWeight"]
    by_crop = {}
    all_pass = True
    for crop in CROPS:
        options = [(per_crop[crop][m]["mape"], per_crop[crop][m]["mae"], 0 if m == FIXED else 1, m)
                   for m in (DIRECT,SARIMA,FIXED,EXTRA) if per_crop[crop][m]["mape"] is not None and per_crop[crop][m]["mae"] is not None]
        strongest = min(options)[3] if options else None
        candidate, ablation = per_crop[crop][CANDIDATE], per_crop[crop][ABLATION]
        base = per_crop[crop].get(strongest,{}) if strongest else {}
        gain = base.get("mape")-candidate["mape"] if base.get("mape") is not None and candidate["mape"] is not None else None
        crop_cells = [c for c in cells if c["product"] == crop]
        distance_changes = {}
        for cell in crop_cells:
            h = cell["horizon"]
            comp_metrics = cell["metrics"].get(strongest,{}) if strongest else {}
            cm = cell["metrics"][CANDIDATE]
            distance_changes[str(h)] = {"mapeWorsePp": cm["mape"]-comp_metrics.get("mape") if cm["mape"] is not None and comp_metrics.get("mape") is not None else None,
                                        "maeWorsePct": 100*(cm["mae"]-comp_metrics.get("mae"))/comp_metrics["mae"] if cm["mae"] is not None and comp_metrics.get("mae") not in (None,0) else None}
        joint = any(r["product"]==crop and r["comparator"]==strongest and r["jointConclusion"]=="joint_improvement" for r in report["jointConclusions"]) if strongest else False
        coverage_rows=[report["coverageByCell"].get(f"{crop}/{h}",{}) for h in HORIZONS]
        coverage_ok = strongest is not None and all(x.get("issued",{}).get(CANDIDATE,0)>=x.get("issued",{}).get(strongest,0) for x in coverage_rows)
        passes = (gain is not None and gain>=.25 and candidate["mae"] is not None and base.get("mae") not in (None,0)
                  and 100*(candidate["mae"]-base["mae"])/base["mae"]<=5
                  and all(v["mapeWorsePp"] is not None and v["maeWorsePct"] is not None and v["mapeWorsePp"]<=1 and v["maeWorsePct"]<=5 for v in distance_changes.values())
                  and coverage_ok and joint and ablation["mape"] is not None and candidate["mape"]<ablation["mape"])
        all_pass &= passes
        by_crop[crop]={"strongestBaseline":strongest,"candidateMapeGainPp":gain,"candidateMaeWorseningPct":100*(candidate["mae"]-base["mae"])/base["mae"] if candidate["mae"] is not None and base.get("mae") not in (None,0) else None,
                       "distanceChanges":distance_changes,"coverageNondecreasing":coverage_ok,"jointImprovementEvidence":joint,
                       "candidateBeatsAblationMape":ablation["mape"] is not None and candidate["mape"] is not None and candidate["mape"]<ablation["mape"],"pass":bool(passes)}
    all_metrics = report["macro50CellEqualWeight"]
    overall = {"candidateMapeAtMost10Pct":all_metrics[CANDIDATE]["mape"] is not None and all_metrics[CANDIDATE]["mape"]<=10,
               "candidateMapeGainVsFixed50AtLeast0_5pp":all_metrics[FIXED]["mape"] is not None and all_metrics[CANDIDATE]["mape"] is not None and all_metrics[FIXED]["mape"]-all_metrics[CANDIDATE]["mape"]>=.5}
    return {"perCrop":by_crop,"allFiveCropGatesPass":bool(all_pass),"overallFiveCropMacro":overall,
            "status":"pass" if all_pass else "fail_or_unsupported"}


def score_manifest(manifest: dict, *, root: Path = ROOT) -> dict:
    verify_manifest(manifest)
    docs = _load_inputs(root)
    forecasts, audits = _compose(docs)
    if forecasts != manifest["forecasts"] or audits["training"] != manifest["trainingAudit"] or audits["testFallbacks"] != manifest["testFallbackAudit"]:
        raise ValueError("specialist manifest does not fully regenerate")
    # Only after full answer-free replay access B actuals and verify actual/direct parity.
    direct = _direct_rows(docs,"B",include_b_actual=True)
    signals = _signal_rows(docs)["B"]
    raw_maps = {}
    for crop in CROPS:
        source_name, _ = SIGNAL_SOURCE[crop]
        raw_rows = docs[source_name].get("originEvidence",[]) if crop in ("cabbage","onion") else docs[source_name].get("pairedRows",[])
        raw_maps[crop] = {}
        for raw in raw_rows:
            product = raw.get("product", "radish" if crop=="radish" else None)
            if product != crop or _window(_date(raw.get("target"))) != "B": continue
            raw_maps[crop][_key({**raw,"product":crop})]=raw
    actual_map = {}
    for crop in CROPS:
        for key, row in signals[crop].items():
            direct_row = direct.get(key)
            if direct_row is None:
                raise ValueError(f"B direct target row missing: {key}")
            raw=raw_maps[crop].get(key)
            field = "mix50" if crop=="pepper" else SIGNAL_SOURCE[crop][1]
            actual = _positive(raw.get("actual")) if raw else None
            target_actual = _positive(direct_row.get("actualForScoring"))
            if actual is None or target_actual is None or not math.isclose(actual,target_actual,rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError(f"B target actual parity failure: {key}")
            if _positive(raw.get(field)) != row["signal"]:
                raise ValueError(f"B feature signal replay mismatch: {key}")
            actual_map[key]=target_actual
    output=_score(manifest,actual_map,audits["testFallbacks"])
    output.update({"schemaVersion":1,"protocol":PROTOCOL,"manifestSha256":manifest["manifestSha256"],
                   "resultType":"retrospective_exploratory","independentConfirmation":False})
    output["reportSha256"]=_hash(output)
    return output


def _exclusive_output(path: Path, inputs: tuple[Path,...]) -> None:
    dest=path.resolve()
    if any(dest==x.resolve() for x in inputs): raise ValueError("output must not overwrite an input")
    if dest.exists(): raise FileExistsError(f"output already exists: {dest}")


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build");build.add_argument("output",type=Path)
    score=sub.add_parser("score");score.add_argument("manifest",type=Path);score.add_argument("output",type=Path)
    args=parser.parse_args(argv); inputs=tuple(ROOT/p for p in PATHS.values())
    if args.command=="build":
        _exclusive_output(args.output,inputs); result=build_manifest()
    else:
        _exclusive_output(args.output,(*inputs,args.manifest))
        if not args.manifest.is_file(): raise FileNotFoundError(args.manifest)
        result=score_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream: stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
