"""Frozen two-phase missing-day state-space price pilot (2025 summer)."""
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

PRICE_SHA256 = "eeb014d9c7487aae047eb1a0df153f18dc0d496a457a52c0125430401353926c"
START, END = dt.date(2025, 6, 1), dt.date(2025, 7, 31)
HORIZONS, BANDWIDTHS = (1, 5, 10), (9, 28)
COMPARATORS, LOSSES = ("direct", "persistence"), ("AE", "APE")
PLANNED = 61 * len(dashboard.PRODUCTS) * len(HORIZONS)
FAMILY = len(dashboard.PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)


def _check_statsmodels():
    try:
        import statsmodels
    except ImportError as exc:
        raise RuntimeError("statsmodels_0_14_0_required_for_forecast_phase") from exc
    if statsmodels.__version__ != "0.14.0":
        raise RuntimeError("statsmodels_version_mismatch")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _immutable(path: Path, value) -> None:
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


def _state_space(series, origin, horizon):
    """Return forecast and audit state; target values are never accessed here."""
    begin = origin - dt.timedelta(days=365)
    dates = [begin + dt.timedelta(days=i) for i in range(365)]
    observed = [(d, series[d][0]) for d in dates if d in series]
    audit = {"observations": len(observed), "fitStatus": "unavailable", "reason": None}
    if len(observed) < 90:
        audit["reason"] = "fewer_than_90_observations"
        return None, audit
    try:
        import numpy as np
        import statsmodels
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        y = np.full(365, np.nan, dtype=float)
        for d, price in observed:
            y[(d - begin).days] = math.log(price)
        fit = UnobservedComponents(y, level="local level", seasonal=7,
                                   stochastic_seasonal=False).fit(maxiter=50, disp=False)
        converged = bool(fit.mle_retvals.get("converged", False))
        audit["fitStatus"] = "converged" if converged else "nonconverged"
        if not converged:
            audit["reason"] = "fit_not_converged"
            return None, audit
        value = float(fit.forecast(horizon + 1)[-1])
        price = math.exp(value)
        if not _positive(price):
            audit["reason"] = "invalid_state_space_forecast"
            return None, audit
        audit.update({"reason": None, "forecastSteps": horizon + 1,
                      "statsmodelsVersion": statsmodels.__version__})
        return price, audit
    except Exception as exc:
        audit["reason"] = "fit_error:" + type(exc).__name__
        return None, audit


def forecast(prices_path: Path, output_dir: Path) -> dict:
    _check_statsmodels()
    raw = Path(prices_path).read_bytes()
    if _sha(raw) != PRICE_SHA256:
        raise ValueError("prices_snapshot_hash_mismatch")
    doc = json.loads(raw)
    products = {p.get("id"): p for p in doc.get("products", []) if isinstance(p, dict)}
    if doc.get("schemaVersion") != 1 or set(products) != set(dashboard.PRODUCTS):
        raise ValueError("prices_must_contain_dashboard_products")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_directory_must_be_new_or_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in dashboard.PRODUCTS:
        series, days, logs = _series(products[pid])
        for origin in _dates():
            for h in HORIZONS:
                target = origin + dt.timedelta(days=h)
                candidate, audit = _state_space(series, origin, h)
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
                rows.append({"productId": pid, "origin": origin.isoformat(), "target": target.isoformat(),
                             "horizon": h, "candidate": candidate, "direct": direct,
                             "persistence": persistence, "stateAudit": audit,
                             "directTrainingTargets": ntrain,
                             "unavailableReasons": {"candidate": audit["reason"],
                                "direct": None if _positive(direct) else "direct_model_or_features_unavailable",
                                "persistence": None if _positive(persistence) else "no_recent_source_price"}})
    manifest = {"schemaVersion": 1, "protocol": "state-space-missing-days-pilot-2026-10-08",
                "phase": "forecast_only", "targetValuesAttached": False,
                "sourceSha256": _sha(raw), "pinnedSourceSha256": PRICE_SHA256,
                "codeSha256": _sha(Path(__file__).read_bytes()),
                "period": [START.isoformat(), END.isoformat()], "origins": 61,
                "products": list(dashboard.PRODUCTS), "horizons": list(HORIZONS),
                "plannedRows": PLANNED,
                "candidateProtocol": {"model": "statsmodels 0.14.0 UnobservedComponents",
                    "level": "local level", "seasonal": 7, "stochasticSeasonal": False,
                    "windowCalendarDays": 365, "minimumObservations": 90, "maxiter": 50,
                    "forecastSteps": "horizon + 1 from origin - 1"},
                "rows": rows}
    _immutable(output_dir / "forecast-manifest.json", manifest)
    return manifest


def _dm(diffs, bandwidth):
    n = len(diffs)
    if n < 30:
        return {"status": "insufficient_pairs", "n": n, "meanLossDifference": None, "statistic": None, "pValue": 1.0}
    mean = sum(diffs.values()) / n
    dates = sorted(diffs); centered = {d: v-mean for d, v in diffs.items()}
    s = sum(v*v for v in centered.values())
    for lag in range(1, bandwidth+1):
        s += 2*(1-lag/(bandwidth+1))*sum(centered[d]*centered.get(d-dt.timedelta(days=lag), 0) for d in dates)
    var = s/(n*n)
    if not math.isfinite(var) or var <= 0:
        return {"status": "invalid_hac_variance", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": None}
    z = mean/math.sqrt(var)
    return {"status": "tested", "n": n, "meanLossDifference": mean, "statistic": z, "pValue": math.erfc(abs(z)/math.sqrt(2))}


def _holm(items):
    order = sorted((i for i, x in enumerate(items) if x["status"] == "tested"), key=lambda i: items[i]["pValue"])
    running = 0.0; m = len(items)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m-rank)*items[i]["pValue"]))
        items[i].update(holmAdjustedPValue=running, holmSignificant=running < .05)
    for x in items:
        if x["status"] != "tested": x.update(holmAdjustedPValue=None, holmSignificant=False)


def score(prices_path: Path, manifest_path: Path, output_path: Path) -> dict:
    raw, mraw = Path(prices_path).read_bytes(), Path(manifest_path).read_bytes()
    manifest = json.loads(mraw)
    if _sha(raw) != PRICE_SHA256 or manifest.get("sourceSha256") != _sha(raw): raise ValueError("source_hash_mismatch")
    if manifest.get("codeSha256") != _sha(Path(__file__).read_bytes()): raise ValueError("code_hash_mismatch")
    if manifest.get("targetValuesAttached") is not False or manifest.get("phase") != "forecast_only": raise ValueError("manifest_not_forecast_only")
    if (manifest.get("period") != [START.isoformat(), END.isoformat()]
            or manifest.get("origins") != 61 or manifest.get("products") != list(dashboard.PRODUCTS)
            or manifest.get("horizons") != list(HORIZONS)):
        raise ValueError("frozen_protocol_mismatch")
    if manifest.get("plannedRows") != PLANNED or len(manifest.get("rows", [])) != PLANNED: raise ValueError("forecast_grid_incomplete")
    doc = json.loads(raw); products = {p.get("id"): p for p in doc["products"]}
    actuals = {}
    for pid in dashboard.PRODUCTS:
        for r in products[pid].get("observations", []):
            d = dt.date.fromisoformat(r["date"]); v = float(r["actual"])
            if not _positive(v): raise ValueError("invalid_actual_price")
            actuals[(pid, d)] = v
    expected = {(p, o, h) for p in dashboard.PRODUCTS for o in _dates() for h in HORIZONS}
    seen = set()
    scored = []
    for r in manifest["rows"]:
        pid, origin, target, h = r["productId"], dt.date.fromisoformat(r["origin"]), dt.date.fromisoformat(r["target"]), r["horizon"]
        key = (pid, origin, h)
        if key in seen or key not in expected or target != origin+dt.timedelta(days=h): raise ValueError("forecast_grid_or_date_invalid")
        seen.add(key)
        for model in ("candidate", "direct", "persistence"):
            value = r.get(model)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError("invalid_forecast_value")
        actual = actuals.get((pid, target))
        row = {**r, "actual": actual, "paired": _positive(actual) and all(_positive(r.get(m)) for m in ("candidate", "direct", "persistence"))}
        scored.append(row)
    if seen != expected: raise ValueError("forecast_grid_incomplete")
    paired = [r for r in scored if r["paired"]]
    metrics = {}
    for model in ("candidate", "direct", "persistence"):
        metrics[model] = {}
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                group = [r for r in paired if r["productId"] == pid and r["horizon"] == h]
                metrics[model][f"{pid}/h{h}"] = {"n": len(group),
                    "mae": sum(abs(r["actual"]-r[model]) for r in group)/len(group) if group else None,
                    "mape": 100*sum(abs(r["actual"]-r[model])/r["actual"] for r in group)/len(group) if group else None}
    macro = {m: (sum(metrics[m][f"{p}/h{h}"]["mape"] for p in dashboard.PRODUCTS for h in HORIZONS)/15
                 if all(metrics[m][f"{p}/h{h}"]["mape"] is not None for p in dashboard.PRODUCTS for h in HORIZONS) else None) for m in metrics}
    tests = {}
    for bw in BANDWIDTHS:
        family = []
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                group = [r for r in paired if r["productId"] == pid and r["horizon"] == h]
                for comp in COMPARATORS:
                    for loss in LOSSES:
                        diffs = {dt.date.fromisoformat(r["target"]): (abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp])) if loss == "AE" else 100*(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp]))/r["actual"] for r in group}
                        x = _dm(diffs, bw); x.update(productId=pid, horizon=h, comparator=comp, loss=loss); family.append(x)
        if len(family) != FAMILY: raise AssertionError("fixed_DM_family_size_mismatch")
        _holm(family)
        for x in family:
            x["conclusion"] = ("inference_unavailable" if x["status"] != "tested" else
                "inconclusive" if not x["holmSignificant"] or x["meanLossDifference"] == 0 else
                "improvement" if x["meanLossDifference"] < 0 else "worsening")
        tests[str(bw)] = family
    groups = []
    for pid in dashboard.PRODUCTS:
        for h in HORIZONS:
            g = [r for r in scored if r["productId"] == pid and r["horizon"] == h]
            groups.append({"productId": pid, "horizon": h, "planned": len(g),
                "targetObserved": sum(r["actual"] is not None for r in g), "commonPairs": sum(r["paired"] for r in g),
                "coverageByModel": {m: sum(r["actual"] is not None and _positive(r.get(m)) for r in g) for m in metrics}})
    crop = {}
    for pid in dashboard.PRODUCTS:
        cm = [metrics["candidate"][f"{pid}/h{h}"]["mape"] for h in HORIZONS]
        dm = [metrics["direct"][f"{pid}/h{h}"]["mape"] for h in HORIZONS]
        ca = [metrics["candidate"][f"{pid}/h{h}"]["mae"] for h in HORIZONS]
        da = [metrics["direct"][f"{pid}/h{h}"]["mae"] for h in HORIZONS]
        crop[pid] = {"mapeIncreasePp": sum(a-b for a, b in zip(cm, dm))/3 if all(x is not None for x in cm+dm) else None,
                     "maeRelativeIncrease": (sum(ca)/sum(da)-1 if sum(da) > 0 else None) if all(x is not None for x in ca+da) else None}
    gain = macro["direct"]-macro["candidate"] if macro["direct"] is not None and macro["candidate"] is not None else None
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
            "This 61-origin summer slice is not an untouched final holdout."], "rows": scored}
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
