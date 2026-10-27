"""Frozen comparison of two-market onion recomposition with national price baselines."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path("outputs/experiments/market-recompose-20261007/report.json")
HIST_PATH = Path("outputs/experiments/ridge-sarima-weight-cv-20261008/historical-components.json")
B_PATH = Path("outputs/experiments/ridge-sarima-component-cv-20261008/forecast-manifest.json")
INPUT_HASHES = {
    str(REPORT_PATH): "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581",
    str(HIST_PATH): "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74",
    str(B_PATH): "a85225b580cc6d55bcd5d49c88ee458520cf6de8ace61250fff36ef1a3d91ebd",
}
WINDOWS = {
    "A": (dt.date(2024, 9, 25), dt.date(2025, 8, 31)),
    "B": (dt.date(2025, 9, 25), dt.date(2026, 9, 24)),
}
HORIZONS = (1, 2)
PRODUCT = "onion"
CANDIDATE = "market_recompose"
DIRECT = "direct_ridge"
SARIMA = "sarima"
FIXED = "fixed50"
METHODS = (CANDIDATE, DIRECT, SARIMA, FIXED)
COMPARATORS = (DIRECT, SARIMA, FIXED)
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
EXPECTED_COUNTS = {"A": 572, "B": 604}
ALPHA = 0.05
PROTOCOL = {
    "id": "onion-market-vs-price-blend-2026-10-09",
    "inputs": INPUT_HASHES,
    "windows": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in WINDOWS.items()},
    "population": "onion target keys, horizons 1 and 2, with exact origin/target keys shared by frozen market report and period component source",
    "candidate": "market report candidatePrediction for every frozen row, preserving and reporting the original fallback flag and reason; no new fallback or substitution",
    "A_fixed50": "0.5 * historical-components.direct + 0.5 * historical-components.sarima",
    "B_fixed50": "forecast-manifest.predictions.fixed50 exactly",
    "inference": "per window 2 horizons x 3 comparators x AE/APE = 12 two-sided tests; calendar Bartlett HAC9/HAC28; separate Holm .05; fewer than 30 pairs or invalid variance remains p=1",
    "gate": "both windows: candidate 2-distance macro MAPE >=0.25pp below strongest baseline (lowest macro MAPE, ties by MAE then fixed50); each horizon MAPE worsening <=0.25pp and MAE worsening <=5% vs that baseline; issued coverage nondecreasing; at least one of four window x horizon cells joint AE/APE improvement at both Holm HAC9/HAC28",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _iso_day(value: Any) -> dt.date:
    try:
        if not isinstance(value, str):
            raise ValueError
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("forecast dates must be ISO dates") from exc


def _key(row: dict) -> tuple[str, int, str, str]:
    product, horizon = row.get("product"), row.get("horizon")
    if product != PRODUCT or isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("only onion horizons 1 and 2 are in scope")
    origin, target = _iso_day(row.get("origin")), _iso_day(row.get("target"))
    if target <= origin or (target-origin).days != horizon:
        raise ValueError("origin must precede target by the declared horizon")
    return product, horizon, origin.isoformat(), target.isoformat()


def _load_inputs(root: Path = ROOT) -> dict[str, dict]:
    docs = {}
    for rel, expected in INPUT_HASHES.items():
        raw = (root / rel).read_bytes()
        if _sha(raw) != expected:
            raise ValueError(f"pinned input byte SHA-256 mismatch: {rel}")
        doc = json.loads(raw)
        docs[rel] = doc
    report = docs[str(REPORT_PATH)]
    if report.get("schemaVersion") != 1 or report.get("protocol", {}).get("id") != "market-wise-direct-ridge-exante-share-2026-10-07":
        raise ValueError("market report schema/protocol mismatch")
    if report.get("hashes", {}).get("directRidgeReportSha256") != "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67":
        raise ValueError("market report direct-Ridge provenance mismatch")
    hist = docs[str(HIST_PATH)]
    if hist.get("schemaVersion") != 1 or hist.get("phase") != "historical_components_only" or hist.get("targetValuesAttached") is not False:
        raise ValueError("historical component source is not answer-free")
    _verify_content_manifest(hist, "historical component")
    bdoc = docs[str(B_PATH)]
    if bdoc.get("schemaVersion") != 1 or bdoc.get("phase") != "selected_test_forecast_only" or bdoc.get("targetValuesAttached") is not False:
        raise ValueError("B component source is not answer-free")
    _verify_content_manifest(bdoc, "B component")
    return docs


def _verify_content_manifest(doc: dict, label: str) -> None:
    body = dict(doc)
    saved = body.pop("manifestSha256", None)
    if not isinstance(saved, str) or saved != _hash(body):
        raise ValueError(f"{label} canonical content hash mismatch")


def _window_of(row: dict) -> str | None:
    target = _iso_day(row.get("target"))
    for name, (start, end) in WINDOWS.items():
        if start <= target <= end:
            return name
    return None


def _indexed(rows: Any, label: str) -> dict[tuple, dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows are missing")
    result = {}
    for row in rows:
        if row.get("product") != PRODUCT or row.get("horizon") not in HORIZONS or _window_of(row) is None:
            continue
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate {label} forecast key")
        result[key] = row
    return result


def _compose_rows(docs: dict[str, dict]) -> list[dict]:
    report = docs[str(REPORT_PATH)]
    hist = docs[str(HIST_PATH)]
    bdoc = docs[str(B_PATH)]
    markets = _indexed(report.get("originEvidence"), "market report")
    historical = _indexed(hist.get("forecasts"), "historical component")
    brows = _indexed(bdoc.get("forecasts"), "B component")
    output = []
    expected_by_window = {name: set() for name in WINDOWS}
    for key, market in markets.items():
        window = _window_of(market)
        source = historical if window == "A" else brows if window == "B" else None
        if source is None:
            continue
        component = source.get(key)
        if component is None:
            raise ValueError(f"component source is missing exact market key: {key}")
        if not isinstance(market.get("fallback"), bool) or not isinstance(market.get("fallbackReasons"), list):
            raise ValueError(f"market fallback provenance is malformed: {key}")
        direct_market = _positive(market.get("baselinePrediction"))
        if window == "A":
            direct = _positive(component.get("direct"))
            sarima = _positive(component.get("sarima"))
            if component.get("status") != "available" or direct is None or sarima is None:
                raise ValueError(f"A components unavailable/nonpositive: {key}")
            fixed = 0.5*direct + 0.5*sarima
        else:
            preds = component.get("predictions", {})
            direct, sarima, fixed = (_positive(preds.get("direct")), _positive(preds.get("sarima")), _positive(preds.get("fixed50")))
            if direct is None or sarima is None or fixed is None:
                raise ValueError(f"B components unavailable/nonpositive: {key}")
        if direct_market is None or not math.isclose(direct, direct_market, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"direct Ridge equality mismatch: {key}")
        candidate = _positive(market.get("candidatePrediction"))
        if candidate is None:
            raise ValueError(f"market candidate is unavailable/nonpositive: {key}")
        expected_by_window[window].add(key)
        output.append({"product": PRODUCT, "horizon": key[1], "origin": key[2], "target": key[3],
                       "predictions": {CANDIDATE: candidate, DIRECT: direct, SARIMA: sarima, FIXED: fixed},
                       "lineage": {"marketFallback": bool(market.get("fallback")),
                                   "marketFallbackReasons": market.get("fallbackReasons", []),
                                   "periodSource": window, "marketSourceRowKey": list(key),
                                   "fixed50Rule": "0.5*direct+0.5*sarima" if window == "A" else "component predictions.fixed50"}})
    for window, source in (("A", historical), ("B", brows)):
        source_keys = {key for key in source if _window_of({"target": key[3]}) == window}
        if source_keys != expected_by_window[window]:
            raise ValueError(f"{window} market/component exact key-set mismatch: market={len(expected_by_window[window])}, components={len(source_keys)}")
        if len(source_keys) != EXPECTED_COUNTS[window]:
            raise ValueError(f"{window} planned key count mismatch: expected {EXPECTED_COUNTS[window]}, got {len(source_keys)}")
    if not output:
        raise ValueError("no frozen onion 1/2-day comparisons found")
    return sorted(output, key=lambda r: (r["target"], r["horizon"]))


def build_manifest(*, root: Path = ROOT) -> dict:
    """Build a target-free candidate manifest from the three pinned inputs."""
    docs = _load_inputs(root)
    rows = _compose_rows(docs)
    result = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
              "inputSha256": dict(INPUT_HASHES), "windowKeys": {name: sum(_window_of(r) == name for r in rows) for name in WINDOWS},
              "plannedForecastCount": len(rows), "forecasts": rows}
    result["manifestSha256"] = _hash(result)
    return result


def verify_manifest(doc: dict) -> None:
    if doc.get("protocol") != PROTOCOL or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("candidate manifest is not answer-free under frozen protocol")
    if doc.get("inputSha256") != INPUT_HASHES:
        raise ValueError("candidate input hashes differ from frozen values")
    rows = doc.get("forecasts")
    if not isinstance(rows, list) or len(rows) != doc.get("plannedForecastCount"):
        raise ValueError("candidate forecast count mismatch")
    saved = dict(doc).pop("manifestSha256", None)
    body = dict(doc); body.pop("manifestSha256", None)
    if saved != _hash(body):
        raise ValueError("candidate manifest content hash mismatch")
    seen = set()
    for row in rows:
        key = _key(row)
        if key in seen:
            raise ValueError("duplicate candidate manifest key")
        seen.add(key)
        if any(k in row for k in ("actual", "targetValue")):
            raise ValueError("answer-bearing field in candidate manifest")
        if set(row.get("predictions", {})) != set(METHODS):
            raise ValueError("candidate prediction methods differ from frozen set")
        if any(_positive(v) is None for v in row["predictions"].values()):
            raise ValueError("candidate manifest contains nonpositive prediction")


def _dm(diffs: dict[dt.date, float], bandwidth: int) -> dict:
    n = len(diffs)
    mean = sum(diffs.values())/n if n else None
    if n < 30:
        return {"status": "inference_unavailable_insufficient_pairs", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    centered = {day: value-mean for day, value in diffs.items()}
    variance_sum = sum(value*value for value in centered.values())
    for lag in range(1, bandwidth+1):
        cross = sum(value*centered.get(day+dt.timedelta(days=lag), 0.0) for day, value in centered.items())
        variance_sum += 2*(1-lag/(bandwidth+1))*cross
    variance_mean = variance_sum/(n*n)
    if not math.isfinite(variance_mean) or variance_mean <= 0:
        return {"status": "inference_unavailable_nonpositive_variance", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    statistic = mean/math.sqrt(variance_mean)
    return {"status": "available", "n": n, "meanLossDifference": mean, "statistic": statistic,
            "pValue": math.erfc(abs(statistic)/math.sqrt(2))}


def _holm(records: list[dict], alpha: float = ALPHA) -> None:
    ordered = sorted(records, key=lambda r: r["pValue"])
    running = 0.0
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (len(records)-rank)*row["pValue"]))
        row["holmAdjustedPValue"] = running
        row["holmSignificant"] = row["status"] == "available" and running < alpha
        row["conclusion"] = ("inference_unavailable" if row["status"] != "available" else
                             "improvement" if row["holmSignificant"] and row["meanLossDifference"] < 0 else
                             "worsening" if row["holmSignificant"] and row["meanLossDifference"] > 0 else "inconclusive")


def _joint(labels: list[str]) -> str:
    if "inference_unavailable" in labels:
        return "inference_unavailable"
    if all(x == "improvement" for x in labels):
        return "joint_improvement"
    if all(x == "worsening" for x in labels):
        return "joint_worsening"
    if "improvement" in labels and "worsening" in labels:
        return "mixed"
    return "inconclusive"


def _score_window(planned: list[dict], actuals: dict[tuple, Any], name: str) -> dict:
    scored, exclusions = [], {}
    for row in planned:
        key = _key(row)
        actual = _positive(actuals.get(key))
        if actual is None:
            exclusions["report_actual_missing_or_invalid"] = exclusions.get("report_actual_missing_or_invalid", 0)+1
            continue
        if any(_positive(row["predictions"].get(method)) is None for method in METHODS):
            exclusions["incomplete_four_method_forecast"] = exclusions.get("incomplete_four_method_forecast", 0)+1
            continue
        scored.append({**row, "actual": actual})
    pairs = scored
    by_h = {h: [r for r in pairs if r["horizon"] == h] for h in HORIZONS}
    metrics = {h: {m: dh._metrics([r["actual"] for r in by_h[h]], [r["predictions"][m] for r in by_h[h]]) for m in METHODS} for h in HORIZONS}
    complete = all(len(by_h[h]) >= MIN_PAIRS for h in HORIZONS)
    macro = {m: {loss: (sum(metrics[h][m][loss] for h in HORIZONS)/2 if complete else None) for loss in ("mae", "mape")} for m in METHODS}
    tests = []
    for h in HORIZONS:
        cell = by_h[h]
        for comp in COMPARATORS:
            for loss in ("ae", "ape"):
                diffs = {}
                for r in cell:
                    y, a, b = r["actual"], r["predictions"][CANDIDATE], r["predictions"][comp]
                    diffs[_iso_day(r["target"])] = abs(y-a)-abs(y-b) if loss == "ae" else 100*(abs(y-a)-abs(y-b))/y
                test = _dm(diffs, 9)
                tests.append({"horizon": h, "comparator": comp, "loss": loss, "n": len(diffs),
                              "candidateMetric": metrics[h][CANDIDATE]["mae" if loss == "ae" else "mape"],
                              "comparatorMetric": metrics[h][comp]["mae" if loss == "ae" else "mape"],
                              "dm": {"9": test, "28": _dm(diffs, 28)}})
    for bw in BANDWIDTHS:
        family = [t["dm"][str(bw)] for t in tests]
        if len(family) != 12:
            raise AssertionError("frozen 12-test family mismatch")
        _holm(family)
    joint = []
    for h in HORIZONS:
        for comp in COMPARATORS:
            ix = {(t["loss"], t["horizon"], t["comparator"]): t for t in tests}
            vals = [ix[(loss, h, comp)]["dm"][str(bw)]["conclusion"] for loss in ("ae", "ape") for bw in BANDWIDTHS]
            joint.append({"horizon": h, "comparator": comp,
                          "conclusions": dict(zip(("ae_hac9", "ae_hac28", "ape_hac9", "ape_hac28"), vals)),
                          "jointConclusion": _joint(vals)})
    coverage = {m: {"planned": len(planned), "issued": sum(_positive(r["predictions"].get(m)) is not None for r in planned),
                    "commonPositiveTargets": len(pairs)} for m in METHODS}
    fallbacks = [r for r in planned if r.get("lineage", {}).get("marketFallback")]
    fallback_reasons = {}
    for row in fallbacks:
        for reason in row.get("lineage", {}).get("marketFallbackReasons", []):
            fallback_reasons[reason] = fallback_reasons.get(reason, 0)+1
    coverage_by_horizon = {}
    for horizon in HORIZONS:
        horizon_plan = [r for r in planned if r["horizon"] == horizon]
        valid_targets = sum(_positive(actuals.get(_key(row))) is not None for row in horizon_plan)
        coverage_by_horizon[str(horizon)] = {"planned": len(horizon_plan), "validPositiveTargets": valid_targets,
            "issued": {m: sum(_positive(r["predictions"].get(m)) is not None for r in horizon_plan) for m in METHODS},
            "commonPositiveTargets": len(by_h[horizon])}
    return {"window": name, "planned": len(planned), "commonPositiveTargets": len(pairs), "exclusions": exclusions,
            "marketFallbackCount": len(fallbacks), "marketFallbackReasonCounts": fallback_reasons,
            "coverage": coverage, "coverageByHorizon": coverage_by_horizon,
            "completeBothHorizonCellsAt30Pairs": complete,
            "byHorizon": [{"horizon": h, "n": len(by_h[h]), "metrics": metrics[h], "coverage": coverage_by_horizon[str(h)]} for h in HORIZONS],
            "macroTwoHorizonEqualWeight": macro, "dmFamilySizePerBandwidth": 12,
            "dmComparisons": tests, "jointConclusions": joint}


def _gate(windows: dict) -> dict:
    criteria = {}
    for name in WINDOWS:
        report = windows.get(name)
        if not report:
            criteria[name] = {"completeBothHorizonCellsAt30Pairs": False, "strongestBaseline": None}
            continue
        macro = report["macroTwoHorizonEqualWeight"]
        choices = [(macro[m]["mape"], macro[m]["mae"], 0 if m == FIXED else 1, m)
                   for m in COMPARATORS if macro[m]["mape"] is not None and macro[m]["mae"] is not None]
        selected = min(choices)[3] if choices else None
        baseline = macro.get(selected, {}) if selected else {}
        gain = baseline.get("mape")-macro[CANDIDATE]["mape"] if selected and macro[CANDIDATE]["mape"] is not None else None
        per_h = {}
        for row in report["byHorizon"]:
            h, vals = row["horizon"], row["metrics"]
            cb, bb = vals[CANDIDATE], vals.get(selected, {}) if selected else {}
            per_h[str(h)] = {"mapeWorsePp": cb["mape"]-bb.get("mape") if cb["mape"] is not None and bb.get("mape") is not None else None,
                             "maeWorsePct": 100*(cb["mae"]-bb.get("mae"))/bb["mae"] if cb["mae"] is not None and bb.get("mae") not in (None, 0) else None}
        cov = report["coverage"]
        coverage_ok = bool(selected and cov[CANDIDATE]["issued"] >= cov[selected]["issued"])
        joint_ok = any(r["comparator"] == selected and r["jointConclusion"] == "joint_improvement" for r in report["jointConclusions"]) if selected else False
        complete = report["completeBothHorizonCellsAt30Pairs"]
        criteria[name] = {"completeBothHorizonCellsAt30Pairs": complete, "strongestBaseline": selected,
                          "macroMapeGainPp": gain, "macroMapeGainAtLeast0_25pp": gain is not None and gain >= .25,
                          "bothDistanceSafeguardsPass": complete and all(v["mapeWorsePp"] is not None and v["maeWorsePct"] is not None and v["mapeWorsePp"] <= .25 and v["maeWorsePct"] <= 5 for v in per_h.values()),
                          "distanceChanges": per_h, "coverageNondecreasing": coverage_ok,
                          "jointImprovementEvidence": joint_ok}
    complete = all(criteria[n].get("completeBothHorizonCellsAt30Pairs") for n in WINDOWS)
    gate_pass = complete and all(v.get("macroMapeGainAtLeast0_25pp") and v.get("bothDistanceSafeguardsPass") and v.get("coverageNondecreasing") for v in criteria.values()) and any(v.get("jointImprovementEvidence") for v in criteria.values())
    return {"status": "pass" if gate_pass else "unsupported" if not complete else "fail",
            "criteriaByWindow": criteria, "atLeastOneJointImprovement": any(v.get("jointImprovementEvidence") for v in criteria.values())}


def score_manifest(manifest: dict, *, root: Path = ROOT) -> dict:
    """Regenerate the answer-free manifest, then attach report target values."""
    verify_manifest(manifest)
    docs = _load_inputs(root)
    regenerated = _compose_rows(docs)
    if regenerated != manifest["forecasts"]:
        raise ValueError("candidate rows do not regenerate from fixed source inputs")
    if _hash({k: v for k, v in manifest.items() if k != "manifestSha256"}) != manifest["manifestSha256"]:
        raise ValueError("candidate manifest replay hash mismatch")
    report_rows = _indexed(docs[str(REPORT_PATH)].get("originEvidence"), "market target report")
    actuals = {key: row.get("actual") for key, row in report_rows.items()}
    output = {"schemaVersion": 1, "protocol": PROTOCOL, "manifestSha256": manifest["manifestSha256"], "windows": {}}
    for name in WINDOWS:
        planned = [r for r in manifest["forecasts"] if _window_of(r) == name]
        output["windows"][name] = _score_window(planned, actuals, name)
    output["gate"] = _gate(output["windows"])
    output["resultType"] = "post_hoc_retrospective"
    output["independentConfirmation"] = False
    output["interpretation"] = "A pass is limited to this retrospective onion two-market proxy and does not establish prospective 10am benefit or official national performance."
    output["reportSha256"] = _hash(output)
    return output


def _exclusive_output(path: Path, inputs: tuple[Path, ...]) -> None:
    dest = path.resolve()
    if any(dest == source.resolve() for source in inputs):
        raise ValueError("output path must not overwrite an input")
    if dest.exists():
        raise FileExistsError(f"output already exists: {dest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create answer-free frozen comparison manifest")
    build.add_argument("output", type=Path)
    score = sub.add_parser("score", help="replay and score a saved comparison manifest")
    score.add_argument("manifest", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    inputs = tuple(ROOT / path for path in INPUT_HASHES)
    if args.command == "build":
        _exclusive_output(args.output, inputs)
        result = build_manifest()
    else:
        _exclusive_output(args.output, (*inputs, args.manifest))
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        result = score_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
