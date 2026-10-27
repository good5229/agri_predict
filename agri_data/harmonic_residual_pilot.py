"""Frozen annual-harmonic/calendar-gap residual price pilot."""
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
COMPARATORS, LOSSES = ("ablation", "direct", "persistence"), ("AE", "APE")
WINDOW_DAYS, MIN_OBSERVATIONS, MIN_PAIRS = 730, 90, 30
PHIS, ALPHA, EPOCH = (0.0, 0.5, 0.8, 0.95, 0.99), 30.0, dt.date(1970, 1, 1)
PLANNED = 55 * len(dashboard.PRODUCTS) * len(HORIZONS)
FAMILY = len(dashboard.PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
PROTOCOL = "harmonic-residual-pilot-2026-10-08"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _immutable(path: Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
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
    return (START + dt.timedelta(days=i) for i in range((END - START).days + 1))


def _series(product):
    out = {}
    for row in product.get("observations", []):
        day, price = dt.date.fromisoformat(row["date"]), float(row["actual"])
        if not _positive(price) or day in out: raise ValueError("invalid_or_duplicate_price_observation")
        out[day] = (price, float(row.get("volumeKg") or 0))
    days = tuple(sorted(out))
    return out, days, tuple(math.log(out[d][0]) for d in days)


def _fourier_row(day: dt.date):
    elapsed = (day - EPOCH).days; out = []
    for period, order in ((365.25, 2), (7.0, 1)):
        for k in range(1, order + 1):
            angle = 2 * math.pi * k * elapsed / period
            out.extend((math.sin(angle), math.cos(angle)))
    return out


def _choose_phi(residuals):
    """Input ordered (date, residual) pairs; fit only gap-1..7 residual pairs."""
    pairs = []
    for (d0, e0), (d1, e1) in zip(residuals, residuals[1:]):
        gap = (d1 - d0).days
        if 1 <= gap <= 7 and math.isfinite(e0) and math.isfinite(e1): pairs.append((gap, e0, e1))
    if len(pairs) < MIN_PAIRS: return None, len(pairs), None
    losses = [(sum((e1 - phi ** gap * e0) ** 2 for gap, e0, e1 in pairs) / len(pairs), phi) for phi in PHIS]
    loss, phi = min(losses, key=lambda item: (item[0], item[1]))
    return phi, len(pairs), loss


def _harmonic_forecasts(series, origin):
    """Ridge Fourier fit on observed prices strictly before origin; no target access."""
    begin = origin - dt.timedelta(days=WINDOW_DAYS)
    observed = [(d, series[d][0]) for d in sorted(series) if begin <= d < origin]
    audit = {"observations": len(observed), "residualPairs": 0, "phi": None, "fitStatus": "unavailable", "reason": None,
             "horizonReasons": {str(h): None for h in HORIZONS}}
    missing = {h: None for h in HORIZONS}
    if len(observed) < MIN_OBSERVATIONS:
        audit["reason"] = "fewer_than_90_observations"
        audit["horizonReasons"] = {str(h): audit["reason"] for h in HORIZONS}
        return {"candidate": missing, "ablation": missing.copy()}, audit
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        X = np.asarray([_fourier_row(d) for d, _ in observed], dtype=float)
        y = np.asarray([math.log(p) for _, p in observed], dtype=float)
        model = Ridge(alpha=ALPHA, fit_intercept=True).fit(X, y)
        residuals = [(d, float(actual - predicted)) for (d, actual), predicted in zip(
            [(d, math.log(p)) for d, p in observed], model.predict(X))]
        phi, n_pairs, loss = _choose_phi(residuals)
        audit.update({"fitStatus": "fitted", "residualPairs": n_pairs, "phi": phi, "phiSelectionLoss": loss})
        if phi is None:
            audit["reason"] = "fewer_than_30_residual_pairs"
            audit["horizonReasons"] = {str(h): audit["reason"] for h in HORIZONS}
            return {"candidate": missing, "ablation": missing.copy()}, audit
        last_day, last_price = observed[-1]; last_resid = residuals[-1][1]
        out = {"candidate": {}, "ablation": {}}
        for h in HORIZONS:
            target = origin + dt.timedelta(days=h)
            base = float(model.predict(np.asarray([_fourier_row(target)]))[0])
            gap = (target - last_day).days
            try:
                ablation = math.exp(base)
                candidate = math.exp(base + phi ** gap * last_resid)
            except OverflowError: ablation = candidate = math.inf
            # Candidate and ablation share eligibility; underflow/overflow invalidates both.
            if _positive(candidate) and _positive(ablation):
                out["candidate"][h], out["ablation"][h] = candidate, ablation
            else:
                out["candidate"][h] = out["ablation"][h] = None
                audit["horizonReasons"][str(h)] = "invalid_harmonic_forecast"
        audit["reason"] = None
        return out, audit
    except Exception as exc:
        if audit.get("fitStatus") == "fitted" and audit.get("phi") in PHIS and audit.get("residualPairs", 0) >= MIN_PAIRS:
            # The shared fit and phi remain valid; only this origin's forecast step failed.
            audit["reason"] = None
            audit["horizonReasons"] = {str(h): "invalid_harmonic_forecast" for h in HORIZONS}
        else:
            audit.update(fitStatus="error", reason="fit_error:" + type(exc).__name__)
            audit["horizonReasons"] = {str(h): audit["reason"] for h in HORIZONS}
        return {"candidate": missing, "ablation": missing.copy()}, audit


def forecast(prices_path: Path, output_dir: Path) -> dict:
    raw = Path(prices_path).read_bytes()
    if _sha(raw) != PRICE_SHA256: raise ValueError("prices_snapshot_hash_mismatch")
    doc = json.loads(raw); products = {p.get("id"): p for p in doc.get("products", []) if isinstance(p, dict)}
    if doc.get("schemaVersion") != 1 or set(products) != set(dashboard.PRODUCTS): raise ValueError("prices_must_contain_dashboard_products")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError("output_directory_must_be_new_or_empty")
    output_dir.mkdir(parents=True, exist_ok=True); rows = []
    for pid in dashboard.PRODUCTS:
        series, days, logs = _series(products[pid])
        for origin in _dates():
            harmonic, audit = _harmonic_forecasts(series, origin)
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
                persistence = series[srcday][0] if srcday and (origin-srcday).days <= dashboard.MAX_SOURCE_AGE_DAYS else None
                row = {"productId": pid, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                       "candidate": harmonic["candidate"][h], "ablation": harmonic["ablation"][h],
                       "direct": direct, "persistence": persistence, "harmonicAudit": audit,
                       "directTrainingTargets": ntrain}
                row["unavailableReasons"] = {m: ("harmonic_unavailable" if row[m] is None else None) for m in ("candidate", "ablation", "direct", "persistence")}
                rows.append(row)
    protocol = {"model": "sklearn Ridge(alpha=30, fit_intercept=True)", "annualFourier": {"periodDays": 365.25, "K": 2},
        "weeklyFourier": {"periodDays": 7, "K": 1}, "epoch": EPOCH.isoformat(), "windowCalendarDays": WINDOW_DAYS,
        "minimumObservations": MIN_OBSERVATIONS, "phiGrid": list(PHIS), "phiSelection": "minimum mean squared residual transition error; calendar gaps 1..7; ties smaller phi",
        "minimumResidualPairs": MIN_PAIRS, "residualForecast": "last pre-origin residual times phi to target calendar-day gap",
        "ablation": "same Fourier ridge, no residual adjustment", "fitPerOrigin": 1}
    manifest = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
        "sourceSha256": _sha(raw), "pinnedSourceSha256": PRICE_SHA256, "codeSha256": _sha(Path(__file__).read_bytes()),
        "period": [START.isoformat(), END.isoformat()], "origins": 55, "products": list(dashboard.PRODUCTS),
        "horizons": list(HORIZONS), "plannedRows": PLANNED, "candidateProtocol": protocol, "rows": rows}
    _immutable(output_dir / "forecast-manifest.json", manifest); return manifest


def score(prices_path: Path, manifest_path: Path, output_path: Path) -> dict:
    raw, mraw = Path(prices_path).read_bytes(), Path(manifest_path).read_bytes(); m = json.loads(mraw)
    if _sha(raw) != PRICE_SHA256 or m.get("sourceSha256") != _sha(raw): raise ValueError("source_hash_mismatch")
    if m.get("codeSha256") != _sha(Path(__file__).read_bytes()): raise ValueError("code_hash_mismatch")
    expected_protocol = {"model": "sklearn Ridge(alpha=30, fit_intercept=True)", "annualFourier": {"periodDays": 365.25, "K": 2},
        "weeklyFourier": {"periodDays": 7, "K": 1}, "epoch": EPOCH.isoformat(), "windowCalendarDays": WINDOW_DAYS,
        "minimumObservations": MIN_OBSERVATIONS, "phiGrid": list(PHIS), "phiSelection": "minimum mean squared residual transition error; calendar gaps 1..7; ties smaller phi",
        "minimumResidualPairs": MIN_PAIRS, "residualForecast": "last pre-origin residual times phi to target calendar-day gap",
        "ablation": "same Fourier ridge, no residual adjustment", "fitPerOrigin": 1}
    if (m.get("protocol") != PROTOCOL or m.get("phase") != "forecast_only" or m.get("targetValuesAttached") is not False
        or m.get("period") != [START.isoformat(), END.isoformat()] or m.get("origins") != 55
        or m.get("products") != list(dashboard.PRODUCTS) or m.get("horizons") != list(HORIZONS)
        or m.get("candidateProtocol") != expected_protocol): raise ValueError("frozen_protocol_mismatch")
    if m.get("plannedRows") != PLANNED or len(m.get("rows", [])) != PLANNED: raise ValueError("forecast_grid_incomplete")
    doc = json.loads(raw); products = {p["id"]: p for p in doc["products"]}; actuals = {}
    for pid in dashboard.PRODUCTS:
        for r in products[pid].get("observations", []):
            d, v = dt.date.fromisoformat(r["date"]), float(r["actual"])
            if not _positive(v): raise ValueError("invalid_actual_price")
            if (pid, d) in actuals: raise ValueError("duplicate_actual")
            actuals[(pid, d)] = v
    expected = {(p, o, h) for p in dashboard.PRODUCTS for o in _dates() for h in HORIZONS}; seen = set(); scored = []
    models = ("candidate", "ablation", "direct", "persistence")
    # Validate the complete target-free manifest before looking up any actual values.
    validated = []
    for r in m["rows"]:
        pid, o, t, h = r.get("productId"), dt.date.fromisoformat(r["origin"]), dt.date.fromisoformat(r["target"]), r.get("horizon")
        key = (pid, o, h)
        if key in seen or key not in expected or t != o + dt.timedelta(days=h): raise ValueError("forecast_grid_or_date_invalid")
        seen.add(key)
        if "actual" in r: raise ValueError("target_value_in_forecast_manifest")
        audit = r.get("harmonicAudit")
        if not isinstance(audit, dict): raise ValueError("harmonic_audit_missing")
        nobs, npairs, phi, fit_status = (audit.get("observations"), audit.get("residualPairs"),
                                         audit.get("phi"), audit.get("fitStatus"))
        if (not isinstance(nobs, int) or isinstance(nobs, bool) or not 0 <= nobs <= WINDOW_DAYS
            or not isinstance(npairs, int) or isinstance(npairs, bool) or not 0 <= npairs <= max(0, nobs - 1)):
            raise ValueError("invalid_harmonic_audit_counts")
        horizon_reasons = audit.get("horizonReasons")
        if not isinstance(horizon_reasons, dict) or set(horizon_reasons) != {str(x) for x in HORIZONS}:
            raise ValueError("invalid_harmonic_horizon_reasons")
        if phi is not None and (not isinstance(phi, (int, float)) or isinstance(phi, bool) or phi not in PHIS):
            raise ValueError("invalid_harmonic_phi")
        for model in models:
            v = r.get(model)
            if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0): raise ValueError("invalid_forecast_value")
        available = r["candidate"] is not None
        if (r["ablation"] is not None) != available: raise ValueError("harmonic_pair_availability_mismatch")
        if available:
            if nobs < MIN_OBSERVATIONS or npairs < MIN_PAIRS or phi not in PHIS or fit_status != "fitted" or audit.get("reason") is not None:
                raise ValueError("available_harmonic_forecast_has_invalid_audit")
            horizon_reason = horizon_reasons[str(h)]
            if horizon_reason is not None: raise ValueError("available_harmonic_forecast_has_invalid_horizon_reason")
        elif phi is not None:
            if fit_status != "fitted" or nobs < MIN_OBSERVATIONS or npairs < MIN_PAIRS or audit.get("reason") is not None:
                raise ValueError("unavailable_harmonic_forecast_has_invalid_shared_audit")
            if horizon_reasons[str(h)] != "invalid_harmonic_forecast":
                raise ValueError("unavailable_harmonic_forecast_has_invalid_horizon_reason")
        elif nobs < MIN_OBSERVATIONS:
            if fit_status != "unavailable" or audit.get("reason") != "fewer_than_90_observations" or horizon_reasons[str(h)] != audit.get("reason"): raise ValueError("inconsistent_harmonic_audit")
        elif fit_status == "fitted":
            if npairs >= MIN_PAIRS or audit.get("reason") != "fewer_than_30_residual_pairs" or horizon_reasons[str(h)] != audit.get("reason"): raise ValueError("inconsistent_harmonic_audit")
        elif fit_status == "error":
            if not isinstance(audit.get("reason"), str) or not audit["reason"].startswith("fit_error:") or horizon_reasons[str(h)] != audit.get("reason"): raise ValueError("inconsistent_harmonic_audit")
        else: raise ValueError("inconsistent_harmonic_audit")
        validated.append((r, pid, t))
    if seen != expected: raise ValueError("forecast_grid_incomplete")
    for r, pid, t in validated:
        actual = actuals.get((pid, t)); scored.append({**r, "actual": actual,
            "paired": _positive(actual) and all(_positive(r.get(x)) for x in models)})
    paired = [r for r in scored if r["paired"]]; metrics = {}
    for model in models:
        metrics[model] = {}
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                g = [r for r in paired if r["productId"] == pid and r["horizon"] == h]
                metrics[model][f"{pid}/h{h}"] = {"n": len(g), "mae": sum(abs(r["actual"]-r[model]) for r in g)/len(g) if g else None,
                    "mape": 100*sum(abs(r["actual"]-r[model])/r["actual"] for r in g)/len(g) if g else None}
    macro = {x: (sum(metrics[x][f"{p}/h{h}"]["mape"] for p in dashboard.PRODUCTS for h in HORIZONS)/15
                 if all(metrics[x][f"{p}/h{h}"]["mape"] is not None for p in dashboard.PRODUCTS for h in HORIZONS) else None) for x in models}
    tests = {}
    for bw in BANDWIDTHS:
        family = []
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                g = [r for r in paired if r["productId"] == pid and r["horizon"] == h]
                for comp in COMPARATORS:
                    for loss in LOSSES:
                        diffs = {dt.date.fromisoformat(r["target"]): (abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp])) if loss == "AE" else 100*(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp]))/r["actual"] for r in g}
                        x = _dm(diffs, bw); x.update(productId=pid, horizon=h, comparator=comp, loss=loss); family.append(x)
        if len(family) != FAMILY: raise AssertionError("fixed_DM_family_size_mismatch")
        _holm(family)
        for x in family:
            x["conclusion"] = "inference_unavailable" if x["status"] != "tested" else ("inconclusive" if not x["holmSignificant"] or x["meanLossDifference"] == 0 else "improvement" if x["meanLossDifference"] < 0 else "worsening")
        tests[str(bw)] = family
    groups = [{"productId": p, "horizon": h, "planned": sum(r["productId"] == p and r["horizon"] == h for r in scored),
        "targetObserved": sum(r["productId"] == p and r["horizon"] == h and r["actual"] is not None for r in scored),
        "commonPairs": sum(r["productId"] == p and r["horizon"] == h and r["paired"] for r in scored),
        "coverageByModel": {x: sum(r["productId"] == p and r["horizon"] == h and r["actual"] is not None and _positive(r.get(x)) for r in scored) for x in models}}
        for p in dashboard.PRODUCTS for h in HORIZONS]
    crop = {}
    for p in dashboard.PRODUCTS:
        cm = [metrics["candidate"][f"{p}/h{h}"]["mape"] for h in HORIZONS]; dm = [metrics["direct"][f"{p}/h{h}"]["mape"] for h in HORIZONS]
        ca = [metrics["candidate"][f"{p}/h{h}"]["mae"] for h in HORIZONS]; da = [metrics["direct"][f"{p}/h{h}"]["mae"] for h in HORIZONS]
        crop[p] = {"mapeIncreasePp": sum(a-b for a,b in zip(cm,dm))/3 if all(v is not None for v in cm+dm) else None,
            "maeRelativeIncrease": sum(ca)/sum(da)-1 if all(v is not None for v in ca+da) and sum(da)>0 else None}
    gains = {comp: (macro[comp]-macro["candidate"] if macro[comp] is not None and macro["candidate"] is not None else None) for comp in ("direct", "ablation")}
    gate = {"all15CellsAtLeast30CommonPairs": all(g["commonPairs"] >= 30 for g in groups),
        "candidateCoverageAtLeastDirect": all(g["coverageByModel"]["candidate"] >= g["coverageByModel"]["direct"] for g in groups),
        "candidateMacroMapeAtMost10Percent": macro["candidate"] is not None and macro["candidate"] <= 10,
        "candidateGainAtLeast0_5ppVsDirect": gains["direct"] is not None and gains["direct"] >= .5,
        "candidateGainAtLeast0_5ppVsAblation": gains["ablation"] is not None and gains["ablation"] >= .5,
        "eachProductMapeWorseningAtMost1pp": all(v["mapeIncreasePp"] is not None and v["mapeIncreasePp"] <= 1 for v in crop.values()),
        "eachProductEqualHorizonMaeWorseningAtMost5Percent": all(v["maeRelativeIncrease"] is not None and v["maeRelativeIncrease"] <= .05 for v in crop.values())}
    gate["passed"] = all(gate.values())
    result = {"schemaVersion": 1, "protocol": PROTOCOL, "sourceSha256": _sha(raw), "forecastManifestSha256": _sha(mraw),
        "codeSha256": m["codeSha256"], "plannedRows": PLANNED,
        "coverage": {"commonPairs": len(paired), "groups": groups, "byModel": {x: sum(r["actual"] is not None and _positive(r.get(x)) for r in scored) for x in models}},
        "pairedMetrics": metrics, "equalCellMacroMape": macro, "macroGainVsComparators": gains,
        "productWorseningVsDirect": crop, "frozenExpansionGate": gate, "dmTestsByHacCalendarDays": tests,
        "sampleLimitations": ["Retrospective revised prices; first publication timestamps are unknown.", "This period was previously examined and is not an untouched final holdout."], "rows": scored}
    _immutable(output_path, result); return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("forecast"); f.add_argument("--prices", type=Path, required=True); f.add_argument("--output-dir", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--prices", type=Path, required=True); s.add_argument("--manifest", type=Path, required=True); s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.phase == "forecast": forecast(a.prices, a.output_dir)
    else: score(a.prices, a.manifest, a.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
