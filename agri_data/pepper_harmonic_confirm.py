"""Frozen 2024 건고추 confirmation for the reviewed harmonic residual method.

Forecasts are written without target values. Scoring validates the complete
manifest and its hashes before joining any realized prices.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import dashboard, harmonic_residual_pilot as harmonic, live_forecast
from .kma_warning_pilot import _fit_direct, _positive
from .state_space_pilot import _dm, _holm

PRICE_SHA256 = harmonic.PRICE_SHA256
HARMONIC_SHA256 = "90135f0702b2f9498a49965f1d822022dd1debfb2b4261cebfe92cdebd967f4e"
PEPPER = "pepper"
START, END = dt.date(2024, 8, 8), dt.date(2024, 10, 31)
HORIZONS, BANDWIDTHS = (1, 5, 10), (9, 28)
MODELS, COMPARATORS, LOSSES = ("candidate", "ablation", "direct", "persistence"), ("ablation", "direct", "persistence"), ("AE", "APE")
ORIGINS = (END - START).days + 1
PLANNED = ORIGINS * len(HORIZONS)
FAMILY = len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
PROTOCOL = "pepper-harmonic-confirm-2026-10-08"


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
    return (START + dt.timedelta(days=i) for i in range(ORIGINS))


def _protocol():
    return {"model": "sklearn Ridge(alpha=30, fit_intercept=True)",
        "annualFourier": {"periodDays": 365.25, "K": 2}, "weeklyFourier": {"periodDays": 7, "K": 1},
        "epoch": harmonic.EPOCH.isoformat(), "windowCalendarDays": harmonic.WINDOW_DAYS,
        "minimumObservations": harmonic.MIN_OBSERVATIONS, "phiGrid": list(harmonic.PHIS),
        "phiSelection": "minimum mean squared residual transition error; calendar gaps 1..7; ties smaller phi",
        "minimumResidualPairs": harmonic.MIN_PAIRS,
        "residualForecast": "last pre-origin residual times phi to target calendar-day gap",
        "ablation": "same Fourier ridge, no residual adjustment", "fitPerOrigin": 1,
        "direct": "independent horizon-specific Ridge using reviewed _fit_direct and strict pre-origin features",
        "persistence": "latest positive observation strictly before origin if source age <= dashboard limit"}


def forecast(prices_path: Path, output_dir: Path) -> dict:
    raw = Path(prices_path).read_bytes()
    if _sha(raw) != PRICE_SHA256: raise ValueError("prices_snapshot_hash_mismatch")
    if _sha(Path(harmonic.__file__).read_bytes()) != HARMONIC_SHA256: raise ValueError("harmonic_dependency_hash_mismatch")
    doc = json.loads(raw); products = {p.get("id"): p for p in doc.get("products", []) if isinstance(p, dict)}
    if doc.get("schemaVersion") != 1 or PEPPER not in products: raise ValueError("prices_must_contain_pepper")
    outdir = Path(output_dir)
    if outdir.exists() and any(outdir.iterdir()): raise ValueError("output_directory_must_be_new_or_empty")
    outdir.mkdir(parents=True, exist_ok=True)
    series, days, logs = harmonic._series(products[PEPPER]); rows = []
    for origin in _dates():
        harmonic_values, audit = harmonic._harmonic_forecasts(series, origin)
        for h in HORIZONS:
            target = origin + dt.timedelta(days=h)
            model, ntrain = _fit_direct(series, origin, h, days, logs)
            strict = live_forecast._strict_feature_row(series, origin, target, h, days, logs)
            direct = None
            if model is not None and strict:
                src, x = strict
                try: direct = series[src][0] * math.exp(float(model.predict([x])[0]))
                except (ValueError, OverflowError, TypeError): pass
                if not _positive(direct): direct = None
            srcday = max((d for d in days if d < origin), default=None)
            persistence = series[srcday][0] if srcday and (origin-srcday).days <= dashboard.MAX_SOURCE_AGE_DAYS else None
            row = {"productId": PEPPER, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                "candidate": harmonic_values["candidate"][h], "ablation": harmonic_values["ablation"][h],
                "direct": direct, "persistence": persistence, "harmonicAudit": audit,
                "directTrainingTargets": ntrain}
            row["unavailableReasons"] = {m: ((audit["horizonReasons"].get(str(h)) if m in ("candidate", "ablation") else None)
                if row[m] is None else None) or ("forecast_unavailable" if row[m] is None else None) for m in MODELS}
            rows.append(row)
    manifest = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
        "sourceSha256": _sha(raw), "pinnedSourceSha256": PRICE_SHA256,
        "dependencySha256": {"harmonic_residual_pilot.py": _sha(Path(harmonic.__file__).read_bytes())},
        "codeSha256": _sha(Path(__file__).read_bytes()), "period": [START.isoformat(), END.isoformat()],
        "origins": ORIGINS, "products": [PEPPER], "horizons": list(HORIZONS), "plannedRows": PLANNED,
        "candidateProtocol": _protocol(), "missingnessPolicy": "retain every planned origin/horizon; absent target stays null; no forward fill",
        "firstPublicationLimit": "snapshot contains revised observations; first-publication timestamps are unavailable",
        "rows": rows}
    _immutable(outdir / "forecast-manifest.json", manifest)
    return manifest


def _validate_manifest(raw: bytes, m: dict) -> list:
    if _sha(raw) != PRICE_SHA256 or m.get("sourceSha256") != _sha(raw): raise ValueError("source_hash_mismatch")
    if _sha(Path(harmonic.__file__).read_bytes()) != HARMONIC_SHA256: raise ValueError("harmonic_dependency_hash_mismatch")
    if m.get("dependencySha256") != {"harmonic_residual_pilot.py": HARMONIC_SHA256}: raise ValueError("dependency_hash_mismatch")
    if m.get("codeSha256") != _sha(Path(__file__).read_bytes()): raise ValueError("code_hash_mismatch")
    if (m.get("protocol") != PROTOCOL or m.get("phase") != "forecast_only" or m.get("targetValuesAttached") is not False
        or m.get("period") != [START.isoformat(), END.isoformat()] or m.get("origins") != ORIGINS
        or m.get("products") != [PEPPER] or m.get("horizons") != list(HORIZONS)
        or m.get("candidateProtocol") != _protocol()): raise ValueError("frozen_protocol_mismatch")
    rows = m.get("rows")
    if m.get("plannedRows") != PLANNED or not isinstance(rows, list) or len(rows) != PLANNED: raise ValueError("forecast_grid_incomplete")
    expected = {(o, h) for o in _dates() for h in HORIZONS}; seen = set(); checked = []
    for r in rows:
        try: o, t, h = dt.date.fromisoformat(r["origin"]), dt.date.fromisoformat(r["target"]), r["horizon"]
        except (KeyError, TypeError, ValueError): raise ValueError("forecast_grid_or_date_invalid")
        key = (o, h)
        if r.get("productId") != PEPPER or key in seen or key not in expected or t != o + dt.timedelta(days=h): raise ValueError("forecast_grid_or_date_invalid")
        seen.add(key)
        if "actual" in r: raise ValueError("target_value_in_forecast_manifest")
        audit = r.get("harmonicAudit")
        if not isinstance(audit, dict): raise ValueError("harmonic_audit_missing")
        reasons = audit.get("horizonReasons")
        if not isinstance(reasons, dict) or set(reasons) != {str(x) for x in HORIZONS}: raise ValueError("invalid_harmonic_horizon_reasons")
        nobs, npairs = audit.get("observations"), audit.get("residualPairs")
        phi, fit_status, shared_reason = audit.get("phi"), audit.get("fitStatus"), audit.get("reason")
        if (not isinstance(nobs, int) or isinstance(nobs, bool) or not 0 <= nobs <= harmonic.WINDOW_DAYS
            or not isinstance(npairs, int) or isinstance(npairs, bool) or not 0 <= npairs <= max(0, nobs - 1)):
            raise ValueError("invalid_harmonic_audit_counts")
        if phi is not None and (not isinstance(phi, (int, float)) or isinstance(phi, bool) or phi not in harmonic.PHIS):
            raise ValueError("invalid_harmonic_phi")
        for model in MODELS:
            v = r.get(model)
            if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0): raise ValueError("invalid_forecast_value")
        available = r.get("candidate") is not None
        if (r.get("ablation") is not None) != available: raise ValueError("harmonic_pair_availability_mismatch")
        hreason = reasons[str(h)]
        if available:
            if (nobs < harmonic.MIN_OBSERVATIONS or npairs < harmonic.MIN_PAIRS or phi not in harmonic.PHIS
                or fit_status != "fitted" or shared_reason is not None):
                raise ValueError("available_harmonic_forecast_has_invalid_audit")
            if hreason is not None: raise ValueError("available_harmonic_forecast_has_invalid_horizon_reason")
        elif phi is not None:
            if (fit_status != "fitted" or nobs < harmonic.MIN_OBSERVATIONS or npairs < harmonic.MIN_PAIRS
                or shared_reason is not None):
                raise ValueError("unavailable_harmonic_forecast_has_invalid_shared_audit")
            if hreason != "invalid_harmonic_forecast": raise ValueError("unavailable_harmonic_forecast_has_invalid_horizon_reason")
        elif nobs < harmonic.MIN_OBSERVATIONS:
            if (fit_status != "unavailable" or shared_reason != "fewer_than_90_observations"
                or hreason != shared_reason): raise ValueError("inconsistent_harmonic_audit")
        elif fit_status == "fitted":
            if (npairs >= harmonic.MIN_PAIRS or shared_reason != "fewer_than_30_residual_pairs"
                or hreason != shared_reason): raise ValueError("inconsistent_harmonic_audit")
        elif fit_status == "error":
            if (not isinstance(shared_reason, str) or not shared_reason.startswith("fit_error:")
                or hreason != shared_reason): raise ValueError("inconsistent_harmonic_audit")
        else: raise ValueError("inconsistent_harmonic_audit")
        unavailable = r.get("unavailableReasons")
        if not isinstance(unavailable, dict) or set(unavailable) != set(MODELS): raise ValueError("invalid_unavailable_reasons")
        for model in MODELS:
            if r[model] is not None and unavailable[model] is not None: raise ValueError("contradictory_unavailable_reason")
            if r[model] is None and not isinstance(unavailable[model], str): raise ValueError("missing_unavailable_reason")
        for model in ("candidate", "ablation"):
            expected_reason = hreason if r[model] is None else None
            if unavailable[model] != expected_reason: raise ValueError("contradictory_harmonic_unavailable_reason")
        checked.append((r, t))
    if seen != expected: raise ValueError("forecast_grid_incomplete")
    return checked


def score(prices_path: Path, manifest_path: Path, output_path: Path) -> dict:
    raw, mraw = Path(prices_path).read_bytes(), Path(manifest_path).read_bytes(); m = json.loads(mraw)
    validated = _validate_manifest(raw, m)  # no target lookup before complete validation
    doc = json.loads(raw); product = next(p for p in doc["products"] if p.get("id") == PEPPER); actuals = {}
    for r in product.get("observations", []):
        day, value = dt.date.fromisoformat(r["date"]), float(r["actual"])
        if not _positive(value): raise ValueError("invalid_actual_price")
        if day in actuals: raise ValueError("duplicate_actual")
        actuals[day] = value
    scored = []
    for r, target in validated:
        actual = actuals.get(target)
        scored.append({**r, "actual": actual, "paired": _positive(actual) and all(_positive(r.get(x)) for x in MODELS)})
    pairs = [r for r in scored if r["paired"]]
    metrics = {m: {} for m in MODELS}
    for h in HORIZONS:
        g = [r for r in pairs if r["horizon"] == h]
        for model in MODELS:
            metrics[model][f"h{h}"] = {"n": len(g), "mae": sum(abs(r["actual"]-r[model]) for r in g)/len(g) if g else None,
                "mape": 100*sum(abs(r["actual"]-r[model])/r["actual"] for r in g)/len(g) if g else None}
    equal3 = {model: {metric: (sum(metrics[model][f"h{h}"][metric] for h in HORIZONS)/3
        if all(metrics[model][f"h{h}"][metric] is not None for h in HORIZONS) else None) for metric in ("mae", "mape")} for model in MODELS}
    tests = {}
    for bw in BANDWIDTHS:
        family = []
        for h in HORIZONS:
            g = [r for r in pairs if r["horizon"] == h]
            for comp in COMPARATORS:
                for loss in LOSSES:
                    diffs = {dt.date.fromisoformat(r["target"]): (abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp]))
                        if loss == "AE" else 100*(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp]))/r["actual"] for r in g}
                    x = _dm(diffs, bw); x.update(horizon=h, comparator=comp, loss=loss)
                    x["conclusion"] = "inference_unavailable" if x["status"] != "tested" else "inconclusive"
                    family.append(x)
        if len(family) != FAMILY: raise AssertionError("fixed_DM_family_size_mismatch")
        _holm(family)
        for x in family:
            if x["status"] == "tested": x["conclusion"] = ("inconclusive" if not x["holmSignificant"] or x["meanLossDifference"] == 0 else "improvement" if x["meanLossDifference"] < 0 else "worsening")
        tests[str(bw)] = family
    groups = [{"horizon": h, "planned": sum(r["horizon"] == h for r in scored),
        "targetObserved": sum(r["horizon"] == h and r["actual"] is not None for r in scored),
        "commonPairs": sum(r["horizon"] == h and r["paired"] for r in scored),
        "coverageByModel": {x: sum(r["horizon"] == h and r["actual"] is not None and _positive(r.get(x)) for r in scored) for x in MODELS}}
        for h in HORIZONS]
    gate = {"eachHorizonAtLeast30CommonPairs": all(g["commonPairs"] >= 30 for g in groups),
        "candidateCoverageAtLeastDirect": all(g["coverageByModel"]["candidate"] >= g["coverageByModel"]["direct"] for g in groups),
        "candidateEqual3HMapeAtMost10Percent": equal3["candidate"]["mape"] is not None and equal3["candidate"]["mape"] <= 10,
        "candidateImprovesEqual3HMapeVsDirectByAtLeast1pp": equal3["direct"]["mape"] is not None and equal3["candidate"]["mape"] is not None and equal3["direct"]["mape"]-equal3["candidate"]["mape"] >= 1,
        "candidateImprovesEqual3HMapeVsAblationByAtLeast1pp": equal3["ablation"]["mape"] is not None and equal3["candidate"]["mape"] is not None and equal3["ablation"]["mape"]-equal3["candidate"]["mape"] >= 1,
        "candidateEqual3HMaeWorseningVsDirectAtMost5Percent": equal3["direct"]["mae"] is not None and equal3["direct"]["mae"] > 0 and equal3["candidate"]["mae"] is not None and equal3["candidate"]["mae"]/equal3["direct"]["mae"]-1 <= .05}
    gate["passed"] = all(gate.values())
    result = {"schemaVersion": 1, "protocol": PROTOCOL, "sourceSha256": _sha(raw), "forecastManifestSha256": _sha(mraw),
        "dependencySha256": m["dependencySha256"], "codeSha256": m["codeSha256"], "plannedRows": PLANNED,
        "coverage": {"targetObserved": sum(r["actual"] is not None for r in scored), "commonPairs": len(pairs), "groups": groups,
            "byModel": {x: sum(r["actual"] is not None and _positive(r.get(x)) for r in scored) for x in MODELS}},
        "pairedMetrics": metrics, "equal3HorizonMetrics": equal3, "frozenExpansionGate": gate,
        "dmTestsByHacCalendarDays": tests,
        "sampleLimitations": ["Retrospective revised prices; first publication timestamps are unknown.", "2024 period was previously examined; this is not an untouched prospective holdout."],
        "missingnessPolicy": "planned rows retained; absent target values remain null and are not forward-filled",
        "rows": scored}
    _immutable(Path(output_path), result); return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("forecast"); f.add_argument("--prices", type=Path, required=True); f.add_argument("--output-dir", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--prices", type=Path, required=True); s.add_argument("--manifest", type=Path, required=True); s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.phase == "forecast": forecast(a.prices, a.output_dir)
    else: score(a.prices, a.manifest, a.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
