"""Frozen Fourier/SARIMAX annual-harmonic price pilot (2025 late summer)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import dashboard, live_forecast
from .kma_warning_pilot import _fit_direct, _positive
from .state_space_pilot import _dm, _holm

PRICE_SHA256 = "eeb014d9c7487aae047eb1a0df153f18dc0d496a457a52c0125430401353926c"
START, END = dt.date(2025, 8, 1), dt.date(2025, 9, 24)
HORIZONS, BANDWIDTHS = (1, 5, 10), (9, 28)
WINDOW_DAYS, MIN_OBSERVATIONS, MAXITER = 730, 90, 50
PLANNED = 55 * len(dashboard.PRODUCTS) * len(HORIZONS)
COMPARATORS, LOSSES = ("direct", "persistence"), ("AE", "APE")
FAMILY = len(dashboard.PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
EPOCH = dt.date(1970, 1, 1)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_statsmodels():
    try:
        import statsmodels
    except ImportError as exc:
        raise RuntimeError("statsmodels_0_14_0_required_for_forecast_phase") from exc
    if statsmodels.__version__ != "0.14.0":
        raise RuntimeError("statsmodels_version_mismatch")


def _immutable(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass


def _dates():
    for i in range((END - START).days + 1):
        yield START + dt.timedelta(days=i)


def _series(product):
    out = {}
    for row in product.get("observations", []):
        day = dt.date.fromisoformat(row["date"])
        price = float(row["actual"])
        if not _positive(price) or day in out:
            raise ValueError("invalid_or_duplicate_price_observation")
        out[day] = (price, float(row.get("volumeKg") or 0))
    days = tuple(sorted(out))
    return out, days, tuple(math.log(out[d][0]) for d in days)


def _fourier_row(day: dt.date):
    elapsed = (day - EPOCH).days
    result = []
    for period, order in ((365.25, 2), (7.0, 1)):
        for k in range(1, order + 1):
            angle = 2 * math.pi * k * elapsed / period
            result.extend((math.sin(angle), math.cos(angle)))
    return result


def _harmonic_forecasts(series, origin: dt.date):
    """Fit once for an origin; return forecasts and a complete fit audit."""
    begin = origin - dt.timedelta(days=WINDOW_DAYS)
    train_dates = [begin + dt.timedelta(days=i) for i in range(WINDOW_DAYS)]
    observed = [(day, series[day][0]) for day in train_dates if day in series]
    audit = {"observations": len(observed), "fitStatus": "unavailable", "reason": None}
    missing = {h: None for h in HORIZONS}
    if len(observed) < MIN_OBSERVATIONS:
        audit["reason"] = "fewer_than_90_observations"
        return missing, audit
    try:
        import numpy as np
        import statsmodels
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.full(WINDOW_DAYS, np.nan, dtype=float)
        for day, price in observed:
            y[(day - begin).days] = math.log(price)
        exog = np.asarray([_fourier_row(day) for day in train_dates], dtype=float)
        model = SARIMAX(y, exog=exog, order=(1, 0, 0), trend="c",
                        enforce_stationarity=True, enforce_invertibility=True)
        fit = model.fit(maxiter=MAXITER, disp=False)
        converged = bool(fit.mle_retvals.get("converged", False))
        audit["fitStatus"] = "converged" if converged else "nonconverged"
        if not converged:
            audit["reason"] = "fit_not_converged"
            return missing, audit
        max_h = max(HORIZONS)
        future_dates = [origin + dt.timedelta(days=i) for i in range(max_h + 1)]
        future_exog = np.asarray([_fourier_row(day) for day in future_dates], dtype=float)
        prediction = np.asarray(fit.get_forecast(steps=max_h + 1, exog=future_exog).predicted_mean, dtype=float)
        out = {}
        for horizon in HORIZONS:
            value = float(prediction[horizon])
            try:
                price = math.exp(value)
            except OverflowError:
                price = math.inf
            out[horizon] = price if _positive(price) else None
        audit.update({"reason": None, "invalidForecastHorizons": [h for h in HORIZONS if out[h] is None],
                      "forecastSteps": max_h + 1, "forecastStepsByHorizon": {str(h): h + 1 for h in HORIZONS},
                      "statsmodelsVersion": statsmodels.__version__})
        # A horizon-specific invalid prediction stays unavailable without discarding valid horizons.
        return out, audit
    except Exception as exc:
        audit["reason"] = "fit_error:" + type(exc).__name__
        return missing, audit


def forecast(prices_path: Path, output_dir: Path) -> dict:
    _check_statsmodels()
    raw = Path(prices_path).read_bytes()
    if _sha(raw) != PRICE_SHA256:
        raise ValueError("prices_snapshot_hash_mismatch")
    doc = json.loads(raw)
    products = {p.get("id"): p for p in doc.get("products", []) if isinstance(p, dict)}
    if doc.get("schemaVersion") != 1 or set(products) != set(dashboard.PRODUCTS):
        raise ValueError("prices_must_contain_dashboard_products")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_directory_must_be_new_or_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in dashboard.PRODUCTS:
        series, days, logs = _series(products[pid])
        for origin in _dates():
            candidates, audit = _harmonic_forecasts(series, origin)
            for h in HORIZONS:
                target = origin + dt.timedelta(days=h)
                direct_model, ntrain = _fit_direct(series, origin, h, days, logs)
                strict = live_forecast._strict_feature_row(series, origin, target, h, days, logs)
                direct = None
                if direct_model is not None and strict:
                    src, x = strict
                    try: direct = series[src][0] * math.exp(float(direct_model.predict([x])[0]))
                    except (ValueError, OverflowError, TypeError): pass
                    if not _positive(direct): direct = None
                srcday = max((d for d in days if d < origin), default=None)
                persistence = series[srcday][0] if srcday and (origin - srcday).days <= dashboard.MAX_SOURCE_AGE_DAYS else None
                candidate = candidates[h]
                rows.append({"productId": pid, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": h, "candidate": candidate, "direct": direct, "persistence": persistence,
                    "fitAudit": audit, "directTrainingTargets": ntrain,
                    "unavailableReasons": {"candidate": (audit["reason"] or "invalid_or_nonpositive_forecast") if candidate is None else None,
                        "direct": None if _positive(direct) else "direct_model_or_features_unavailable",
                        "persistence": None if _positive(persistence) else "no_recent_source_price"}})
    manifest = {"schemaVersion": 1, "protocol": "harmonic-ar-pilot-2026-10-08",
        "phase": "forecast_only", "targetValuesAttached": False, "sourceSha256": _sha(raw),
        "pinnedSourceSha256": PRICE_SHA256, "codeSha256": _sha(Path(__file__).read_bytes()),
        "period": [START.isoformat(), END.isoformat()], "origins": 55,
        "products": list(dashboard.PRODUCTS), "horizons": list(HORIZONS), "plannedRows": PLANNED,
        "candidateProtocol": {"model": "statsmodels 0.14.0 SARIMAX(1,0,0)", "trend": "c",
            "annualFourier": {"periodDays": 365.25, "K": 2}, "weeklyFourier": {"periodDays": 7, "K": 1},
            "epoch": EPOCH.isoformat(), "windowCalendarDays": WINDOW_DAYS,
            "minimumObservations": MIN_OBSERVATIONS, "maxiter": MAXITER,
            "enforceStationarity": True, "enforceInvertibility": True,
            "missingDays": "NaN", "fitPerOrigin": 1, "forecastStepsByHorizon": {str(h): h + 1 for h in HORIZONS}},
        "rows": rows}
    _immutable(output_dir / "forecast-manifest.json", manifest)
    return manifest


def score(prices_path: Path, manifest_path: Path, output_path: Path) -> dict:
    raw, mraw = Path(prices_path).read_bytes(), Path(manifest_path).read_bytes()
    manifest = json.loads(mraw)
    if _sha(raw) != PRICE_SHA256 or manifest.get("sourceSha256") != _sha(raw): raise ValueError("source_hash_mismatch")
    if manifest.get("codeSha256") != _sha(Path(__file__).read_bytes()): raise ValueError("code_hash_mismatch")
    if manifest.get("protocol") != "harmonic-ar-pilot-2026-10-08" or manifest.get("phase") != "forecast_only" or manifest.get("targetValuesAttached") is not False:
        raise ValueError("manifest_not_forecast_only")
    expected_protocol = {"model": "statsmodels 0.14.0 SARIMAX(1,0,0)", "trend": "c",
        "annualFourier": {"periodDays": 365.25, "K": 2}, "weeklyFourier": {"periodDays": 7, "K": 1},
        "epoch": EPOCH.isoformat(), "windowCalendarDays": WINDOW_DAYS, "minimumObservations": MIN_OBSERVATIONS,
        "maxiter": MAXITER, "enforceStationarity": True, "enforceInvertibility": True,
        "missingDays": "NaN", "fitPerOrigin": 1, "forecastStepsByHorizon": {str(h): h + 1 for h in HORIZONS}}
    if (manifest.get("period") != [START.isoformat(), END.isoformat()] or manifest.get("origins") != 55
            or manifest.get("products") != list(dashboard.PRODUCTS) or manifest.get("horizons") != list(HORIZONS)
            or manifest.get("candidateProtocol") != expected_protocol): raise ValueError("frozen_protocol_mismatch")
    if manifest.get("plannedRows") != PLANNED or len(manifest.get("rows", [])) != PLANNED: raise ValueError("forecast_grid_incomplete")
    doc = json.loads(raw); products = {p.get("id"): p for p in doc["products"]}
    actuals = {}
    for pid in dashboard.PRODUCTS:
        for r in products[pid].get("observations", []):
            d, v = dt.date.fromisoformat(r["date"]), float(r["actual"])
            if not _positive(v): raise ValueError("invalid_actual_price")
            actuals[(pid, d)] = v
    expected = {(p, o, h) for p in dashboard.PRODUCTS for o in _dates() for h in HORIZONS}
    seen, scored = set(), []
    for r in manifest["rows"]:
        pid, origin, target, h = r["productId"], dt.date.fromisoformat(r["origin"]), dt.date.fromisoformat(r["target"]), r["horizon"]
        key = (pid, origin, h)
        if key in seen or key not in expected or target != origin + dt.timedelta(days=h): raise ValueError("forecast_grid_or_date_invalid")
        seen.add(key)
        for model in ("candidate", "direct", "persistence"):
            value = r.get(model)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0): raise ValueError("invalid_forecast_value")
        actual = actuals.get((pid, target))
        scored.append({**r, "actual": actual, "paired": _positive(actual) and all(_positive(r.get(m)) for m in ("candidate", "direct", "persistence"))})
    if seen != expected: raise ValueError("forecast_grid_incomplete")
    paired = [r for r in scored if r["paired"]]
    metrics = {}
    for model in ("candidate", "direct", "persistence"):
        metrics[model] = {}
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                g = [r for r in paired if r["productId"] == pid and r["horizon"] == h]
                metrics[model][f"{pid}/h{h}"] = {"n": len(g),
                    "mae": sum(abs(r["actual"] - r[model]) for r in g) / len(g) if g else None,
                    "mape": 100 * sum(abs(r["actual"] - r[model]) / r["actual"] for r in g) / len(g) if g else None}
    macro = {m: (sum(metrics[m][f"{p}/h{h}"]["mape"] for p in dashboard.PRODUCTS for h in HORIZONS) / 15
                 if all(metrics[m][f"{p}/h{h}"]["mape"] is not None for p in dashboard.PRODUCTS for h in HORIZONS) else None) for m in metrics}
    tests = {}
    for bw in BANDWIDTHS:
        family = []
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                group = [r for r in paired if r["productId"] == pid and r["horizon"] == h]
                for comp in COMPARATORS:
                    for loss in LOSSES:
                        diffs = {dt.date.fromisoformat(r["target"]): (abs(r["actual"] - r["candidate"]) - abs(r["actual"] - r[comp]))
                            if loss == "AE" else 100 * (abs(r["actual"] - r["candidate"]) - abs(r["actual"] - r[comp])) / r["actual"] for r in group}
                        x = _dm(diffs, bw); x.update(productId=pid, horizon=h, comparator=comp, loss=loss); family.append(x)
        if len(family) != FAMILY: raise AssertionError("fixed_DM_family_size_mismatch")
        _holm(family)
        for x in family:
            x["conclusion"] = ("inference_unavailable" if x["status"] != "tested" else "inconclusive"
                if not x["holmSignificant"] or x["meanLossDifference"] == 0 else "improvement" if x["meanLossDifference"] < 0 else "worsening")
        tests[str(bw)] = family
    groups = []
    for pid in dashboard.PRODUCTS:
        for h in HORIZONS:
            g = [r for r in scored if r["productId"] == pid and r["horizon"] == h]
            groups.append({"productId": pid, "horizon": h, "planned": len(g), "targetObserved": sum(r["actual"] is not None for r in g),
                "commonPairs": sum(r["paired"] for r in g), "coverageByModel": {m: sum(r["actual"] is not None and _positive(r.get(m)) for r in g) for m in metrics}})
    crop = {}
    for pid in dashboard.PRODUCTS:
        cm = [metrics["candidate"][f"{pid}/h{h}"]["mape"] for h in HORIZONS]
        dm = [metrics["direct"][f"{pid}/h{h}"]["mape"] for h in HORIZONS]
        ca = [metrics["candidate"][f"{pid}/h{h}"]["mae"] for h in HORIZONS]
        da = [metrics["direct"][f"{pid}/h{h}"]["mae"] for h in HORIZONS]
        crop[pid] = {"mapeIncreasePp": sum(a-b for a,b in zip(cm,dm))/3 if all(x is not None for x in cm+dm) else None,
            "maeRelativeIncrease": (sum(ca)/sum(da)-1 if sum(da)>0 else None) if all(x is not None for x in ca+da) else None}
    gain = macro["direct"] - macro["candidate"] if macro["direct"] is not None and macro["candidate"] is not None else None
    gate = {"all15CellsAtLeast30CommonPairs": all(g["commonPairs"] >= 30 for g in groups),
        "candidateCoverageAtLeastDirect": all(g["coverageByModel"]["candidate"] >= g["coverageByModel"]["direct"] for g in groups),
        "candidateMacroMapeAtMost10Percent": macro["candidate"] is not None and macro["candidate"] <= 10,
        "candidateGainAtLeast0_5ppVsDirect": gain is not None and gain >= .5,
        "eachProductMapeWorseningAtMost1pp": all(v["mapeIncreasePp"] is not None and v["mapeIncreasePp"] <= 1 for v in crop.values()),
        "eachProductEqualHorizonMaeWorseningAtMost5Percent": all(v["maeRelativeIncrease"] is not None and v["maeRelativeIncrease"] <= .05 for v in crop.values())}
    gate["passed"] = all(gate.values())
    result = {"schemaVersion": 1, "protocol": manifest["protocol"], "sourceSha256": _sha(raw),
        "forecastManifestSha256": _sha(mraw), "codeSha256": manifest["codeSha256"], "plannedRows": PLANNED,
        "coverage": {"commonPairs": len(paired), "groups": groups,
            "byModel": {m: sum(r["actual"] is not None and _positive(r.get(m)) for r in scored) for m in metrics}},
        "pairedMetrics": metrics, "equalCellMacroMape": macro, "productWorseningVsDirect": crop,
        "frozenExpansionGate": gate, "dmTestsByHacCalendarDays": tests,
        "sampleLimitations": ["Retrospective revised prices; first publication timestamps are unknown.",
            "This 55-origin late-summer slice was previously examined and is not an untouched final holdout."], "rows": scored}
    _immutable(output_path, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("forecast"); f.add_argument("--prices", type=Path, required=True); f.add_argument("--output-dir", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--prices", type=Path, required=True); s.add_argument("--manifest", type=Path, required=True); s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.phase == "forecast": forecast(a.prices, a.output_dir)
    else: score(a.prices, a.manifest, a.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
