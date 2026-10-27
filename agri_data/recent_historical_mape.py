"""Frozen retrospective MAPE/DM reaggregation for mature historical windows."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
INPUT_TWO_MARKET = ROOT / "outputs/daily/2026-10-06/20261006T090116-f908e0a4/predictions.json"
INPUT_NATIONAL = ROOT / "outputs/experiments/national-transfer-20261007/report.json"
TWO_MARKET_SHA256 = "5b4d7cbb362a6fa257fb6e6d37b23e5611c3487c8ddc540ecbdbf3f2665c4127"
NATIONAL_SHA256 = "4a22a69e3e39aed56064641a9f5d5eed255057153ee705bc7c40bb22d338817d"
PRODUCTS_2M = tuple(dashboard.PRODUCTS)
PRODUCTS_NATIONAL = ("cabbage", "radish", "onion", "pepper")
HORIZONS = tuple(range(1, 11))
BANDWIDTHS = (9, 28)
LOSSES = ("mae", "mape")
MIN_PAIRS = 30
ALPHA = 0.05
WINDOWS_2M = (("1y", dt.date(2025, 10, 6), dt.date(2026, 10, 5)),
              ("2y", dt.date(2024, 10, 6), dt.date(2026, 10, 5)),
              ("3y", dt.date(2023, 10, 6), dt.date(2026, 10, 5)))
WINDOWS_NATIONAL = (("1y", dt.date(2025, 7, 1), dt.date(2026, 6, 30)),
                    ("2y_available", dt.date(2024, 7, 8), dt.date(2026, 6, 30)))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _positive_or_reason(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, "null"
    if isinstance(value, bool):
        return None, "invalid_type"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_numeric"
    if not math.isfinite(number):
        return None, "nonfinite"
    if number <= 0:
        return None, "nonpositive"
    return number, None


def _day(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def _load_two_market(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if raw.get("schemaVersion") != 2 or not isinstance(raw.get("products"), list):
        raise ValueError("two-market source must be dashboard prediction schemaVersion 2")
    if raw.get("meta", {}).get("contextEnd") != "2026-10-05":
        raise ValueError("two-market source contextEnd differs from frozen cutoff")
    if {p.get("id") for p in raw["products"]} != set(PRODUCTS_2M) or len(raw["products"]) != len(PRODUCTS_2M):
        raise ValueError("two-market source must contain exactly the five dashboard products")
    result = []
    availability, quality = {}, {}
    for product in raw["products"]:
        pid = product["id"]
        observations = {}
        quality[pid] = {"observationExcluded": {}, "ridgeForecastExcluded": {}, "persistenceExcluded": {}}
        if not isinstance(product.get("observations"), list):
            raise ValueError(f"missing observations for {pid}")
        for obs in product["observations"]:
            day = _day(obs.get("date"), f"{pid} observation date")
            if day > dt.date(2026, 10, 5):
                raise ValueError(f"observation exceeds frozen context cutoff: {pid}/{day}")
            if day in observations:
                raise ValueError(f"duplicate observation {pid}/{day}")
            actual, reason = _positive_or_reason(obs.get("actual"))
            if reason is not None:
                quality[pid]["observationExcluded"][reason] = quality[pid]["observationExcluded"].get(reason, 0) + 1
                continue
            observations[day] = actual
        # Precisely one daily Ridge model run; monthly run output is not eligible.
        selected = [r for r in product.get("runs", [])
                    if r.get("modelId") == "ridge" and r.get("cadence") == "daily"]
        if len(selected) != 1 or not isinstance(selected[0].get("predictions"), list):
            raise ValueError(f"expected one daily Ridge run for {pid}")
        ridge = {}
        for row in selected[0]["predictions"]:
            target = _day(row.get("target"), f"{pid} Ridge target")
            origin = _day(row.get("origin"), f"{pid} Ridge origin")
            horizon = row.get("horizon")
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
                raise ValueError(f"invalid Ridge horizon for {pid}/{target}")
            if origin != target - dt.timedelta(days=horizon):
                raise ValueError(f"Ridge origin/horizon mismatch for {pid}/{target}/h{horizon}")
            key = (target, horizon)
            if key in ridge:
                raise ValueError(f"duplicate Ridge forecast {pid}/{target}/h{horizon}")
            prediction, reason = _positive_or_reason(row.get("predicted"))
            if reason is not None:
                quality[pid]["ridgeForecastExcluded"][reason] = quality[pid]["ridgeForecastExcluded"].get(reason, 0) + 1
                continue
            ridge[key] = prediction
        persistence = {}
        if not isinstance(product.get("predictions"), list):
            raise ValueError(f"missing persistence forecasts for {pid}")
        for row in product["predictions"]:
            target = _day(row.get("target"), f"{pid} persistence target")
            origin = _day(row.get("origin"), f"{pid} persistence origin")
            horizon = row.get("horizon")
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
                raise ValueError(f"invalid persistence horizon for {pid}/{target}")
            if origin != target - dt.timedelta(days=horizon):
                raise ValueError(f"persistence origin/horizon mismatch for {pid}/{target}/h{horizon}")
            key = (target, horizon)
            if key in persistence:
                raise ValueError(f"duplicate persistence forecast {pid}/{target}/h{horizon}")
            prediction, reason = _positive_or_reason(row.get("predicted"))
            if reason is not None:
                quality[pid]["persistenceExcluded"][reason] = quality[pid]["persistenceExcluded"].get(reason, 0) + 1
                continue
            persistence[key] = prediction
        availability[pid] = {"actualDates": sorted(observations), "ridgeKeys": sorted(ridge),
                             "persistenceKeys": sorted(persistence)}
        for (target, horizon), prediction in ridge.items():
            actual = observations.get(target)
            baseline = persistence.get((target, horizon))
            if actual is not None and baseline is not None:
                result.append({"product": pid, "target": target, "origin": target-dt.timedelta(days=horizon),
                               "horizon": horizon, "actual": actual, "ridge": prediction, "persistence": baseline})
    return result, availability, quality


def _load_national(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if raw.get("schemaVersion") != 1 or raw.get("status") != "complete":
        raise ValueError("national-transfer source must be complete schemaVersion 1")
    protocol = raw.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("id") != "national-wholesale-target-transfer-2026-10-07":
        raise ValueError("unexpected national-transfer protocol")
    if raw.get("hashes", {}).get("archiveSha256") is None:
        raise ValueError("national-transfer report lacks archive provenance")
    rows = raw.get("pairedRows")
    if not isinstance(rows, list):
        raise ValueError("national-transfer source lacks pairedRows")
    coverage_raw = raw.get("coverage")
    if not isinstance(coverage_raw, dict):
        raise ValueError("national-transfer source lacks coverage audit")
    coverage_losses = {}
    for product in PRODUCTS_NATIONAL:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            item = coverage_raw.get(key)
            if not isinstance(item, dict) or not isinstance(item.get("candidateCoverageLossKeys"), dict):
                raise ValueError(f"national-transfer source lacks coverage loss keys for {key}")
            for comparator in ("directRidge", "persistence"):
                values = item["candidateCoverageLossKeys"].get(comparator)
                if not isinstance(values, list):
                    raise ValueError(f"national-transfer source lacks {comparator} coverage keys for {key}")
                parsed = set()
                for identity in values:
                    if (not isinstance(identity, list) or len(identity) != 4 or identity[0] != product
                            or isinstance(identity[3], bool) or identity[3] != horizon):
                        raise ValueError(f"invalid national coverage identity for {key}/{comparator}")
                    origin, target = _day(identity[1], "coverage origin"), _day(identity[2], "coverage target")
                    if origin != target - dt.timedelta(days=horizon):
                        raise ValueError(f"national coverage origin/horizon mismatch for {key}")
                    parsed.add(target)
                coverage_losses[(product, horizon, comparator)] = parsed
    result, seen = [], set()
    for row in rows:
        product = row.get("product")
        if product not in PRODUCTS_NATIONAL:
            raise ValueError(f"unknown national product {product!r}")
        target, origin = _day(row.get("target"), "national target"), _day(row.get("origin"), "national origin")
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise ValueError("invalid national horizon")
        if origin != target - dt.timedelta(days=horizon):
            raise ValueError(f"national origin/horizon mismatch {product}/{target}/h{horizon}")
        key = (product, origin, target, horizon)
        if key in seen:
            raise ValueError(f"duplicate national paired row {key}")
        seen.add(key)
        result.append({"product": product, "target": target, "origin": origin, "horizon": horizon,
                       "actual": _finite_positive(row.get("nationalActual"), "national actual"),
                       "candidate": _finite_positive(row.get("candidatePrediction"), "candidate"),
                       "directRidge": _finite_positive(row.get("directRidge"), "direct Ridge"),
                       "persistence": _finite_positive(row.get("persistence"), "persistence")})
    return result, coverage_losses


def _metrics(rows: list[dict[str, Any]], candidate: str, comparator: str) -> tuple[dict[str, float | None], dict[str, float | None]]:
    actual = [r["actual"] for r in rows]
    return dh._metrics(actual, [r[candidate] for r in rows]), dh._metrics(actual, [r[comparator] for r in rows])


def _window_analysis(rows: list[dict[str, Any]], products: tuple[str, ...], windows, candidate: str,
                     comparators: tuple[str, ...], family_size: int, *, table: str,
                     availability: dict[str, Any] | None = None,
                     coverage_losses: dict[tuple[str, int, str], set[dt.date]] | None = None) -> dict[str, Any]:
    reports = []
    for window_id, start, end in windows:
        selected = [r for r in rows if start <= r["target"] <= end]
        comparisons, coverage, paired = [], {}, []
        for product in products:
            product_rows = [r for r in selected if r["product"] == product]
            for horizon in HORIZONS:
                group = [r for r in product_rows if r["horizon"] == horizon]
                paired.extend(group)
                eligible_targets = sorted({r["target"] for r in product_rows if r["horizon"] == horizon})
                source = availability.get(product, {}) if availability is not None else {}
                observed_dates = {d for d in source.get("actualDates", []) if start <= d <= end}
                candidate_keys = {key for key in source.get("ridgeKeys", []) if key[1] == horizon and start <= key[0] <= end}
                comparator_keys = {key for key in source.get("persistenceKeys", []) if key[1] == horizon and start <= key[0] <= end}
                losses = {c: sorted(d.isoformat() for d in coverage_losses.get((product,horizon,c), set()) if start <= d <= end)
                          for c in comparators} if coverage_losses is not None else None
                if availability is not None:
                    ridge_dates = {k[0] for k in candidate_keys} & observed_dates
                    persistence_dates = {k[0] for k in comparator_keys} & observed_dates
                    losses = {"persistence": sorted(d.isoformat() for d in persistence_dates-ridge_dates)}
                coverage_row = {"pairedDates": [r["target"].isoformat() for r in group],
                    "firstTarget": min((r["target"] for r in group), default=None).isoformat() if group else None,
                    "lastTarget": max((r["target"] for r in group), default=None).isoformat() if group else None,
                    "candidateCoverageLossDates": losses,
                    "candidateCoverageLossCounts": {c: len(dates) for c, dates in losses.items()} if losses is not None else None,
                    "coverageNondeclining": all(not dates for dates in losses.values()) if losses is not None else None,
                    "coverageSource": "snapshot observations and forecast availability" if availability is not None else "source pairedRows plus source candidateCoverageLossKeys"}
                if availability is not None:
                    paired_count = len(group)
                    coverage_row.update({"pairedCount": paired_count,
                        "candidateCoverage": len({k[0] for k in candidate_keys} & observed_dates),
                        "persistenceCoverage": len({k[0] for k in comparator_keys} & observed_dates),
                        "eligibleCount": len(observed_dates), "eligibleMinusPaired": len(observed_dates)-paired_count,
                        "denominatorScope": "all positive observed dates with target in window"})
                else:
                    # The national report omits date-level eligible origins except for
                    # known comparator-only losses, so its total coverage denominators
                    # cannot be reconstructed without the source archive.
                    paired_count = len(group)
                    coverage_row.update({"commonPairedCount": paired_count,
                        "candidateCoverage": None, "comparatorCoverage": {c: None for c in comparators},
                        "eligibleCount": None, "eligibleMinusPaired": None,
                        "denominatorScope": "conditional on source pairedRows; total eligible and method-specific counts unavailable"})
                coverage[f"{product}/{horizon}"] = coverage_row
                for comparator in comparators:
                    for loss in LOSSES:
                        cand_m, comp_m = _metrics(group, candidate, comparator)
                        diffs = {}
                        for row in group:
                            y, c, b = row["actual"], row[candidate], row[comparator]
                            diffs[row["target"]] = abs(y-c)-abs(y-b) if loss == "mae" else 100.0*(abs(y-c)-abs(y-b))/y
                        comparisons.append({"product": product, "horizon": horizon, "comparator": comparator,
                            "loss": loss, "n": len(group), "candidateMetric": cand_m[loss], "comparatorMetric": comp_m[loss],
                            "dm": {str(bw): dh._dm(diffs, bw) for bw in BANDWIDTHS}})
        if len(comparisons) != family_size:
            raise AssertionError(f"{table}/{window_id}: frozen Holm family size mismatch")
        for bw in BANDWIDTHS:
            dh._holm([row["dm"][str(bw)] for row in comparisons])
        macro = {}
        for method in (candidate, *comparators):
            product_macros = {}
            for product in products:
                horizon_mapes = []
                for horizon in HORIZONS:
                    subset = [r for r in paired if r["product"] == product and r["horizon"] == horizon]
                    horizon_mapes.append(dh._metrics([r["actual"] for r in subset], [r[method] for r in subset])["mape"])
                product_macros[product] = {"equalHorizonMape": sum(horizon_mapes)/10 if all(v is not None for v in horizon_mapes) else None,
                    "mapeByHorizon": {str(h): horizon_mapes[h-1] for h in HORIZONS},
                    "pairedCountsByHorizon": {str(h): (coverage[f"{product}/{h}"].get("pairedCount")
                        if "pairedCount" in coverage[f"{product}/{h}"] else coverage[f"{product}/{h}"]["commonPairedCount"]) for h in HORIZONS},
                    "pointEstimateAvailable": all(v is not None for v in horizon_mapes),
                    "interpretation": "descriptive_point_estimate_not_equivalence_or_success_evidence"}
            vals = [v["equalHorizonMape"] for v in product_macros.values()]
            macro[method] = {"byProduct": product_macros,
                "equalProductMape": sum(vals)/len(vals) if vals and all(v is not None for v in vals) else None,
                "availableProducts": sum(v is not None for v in vals), "requiredProducts": len(products)}
        conclusions = []
        indexed = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in comparisons}
        for product in products:
            for h in HORIZONS:
                tests = [indexed[(product,h,c,l)]["dm"]["9"] for c in comparators for l in LOSSES]
                metrics = [indexed[(product,h,c,l)] for c in comparators for l in LOSSES]
                enough = all(t["status"] == "tested" and t["n"] >= MIN_PAIRS for t in tests)
                lower = all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"] < r["comparatorMetric"] for r in metrics)
                sig = all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"] < 0 for t in tests)
                adverse = any(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"] > 0 for t in tests)
                cov_ok = coverage[f"{product}/{h}"]["coverageNondeclining"]
                conclusions.append({"product": product, "horizon": h,
                    "conclusion": "inference_unavailable" if not enough else "improvement" if lower and sig and cov_ok else "worsening_or_mixed" if adverse else "inconclusive",
                    "inferenceAvailable": enough, "allPairedMetricsLower": lower, "allHac9HolmTestsImprove": sig,
                    "coverageNondeclining": cov_ok, "significantWorseningEvidence": adverse})
        reports.append({"window": window_id, "targetStart": start.isoformat(), "targetEnd": end.isoformat(),
            "rowCount": len(paired), "coverage": coverage, "macroMetrics": macro, "comparisons": comparisons,
            "conclusions": conclusions, "pairedRows": [{**r, "target": r["target"].isoformat(), "origin": r["origin"].isoformat()} for r in paired]})
    return {"familySizePerBandwidth": family_size, "windows": reports}


def build(two_market: dict[str, Any], national: dict[str, Any], *, hashes: dict[str, str] | None = None) -> dict[str, Any]:
    rows2, availability2, quality2 = _load_two_market(two_market)
    rowsn, national_coverage_losses = _load_national(national)
    return {"schemaVersion": 1, "protocol": {"id": "recent-mature-historical-mape-2026-10-07",
        "twoMarketSourceSha256": TWO_MARKET_SHA256, "nationalTransferSourceSha256": NATIONAL_SHA256,
        "horizons": list(HORIZONS), "losses": list(LOSSES), "DM": {"twoSided": True,
        "bartlettCalendarHACDays": list(BANDWIDTHS), "holmAlpha": ALPHA, "minimumPairs": MIN_PAIRS,
        "insufficientOrInvalidPValue": 1.0},
        "twoMarketFamily": "5 products x 10 horizons x 1 comparator x 2 losses = 100 tests/window/bandwidth",
        "nationalFamily": "4 products x 10 horizons x 2 comparators x 2 losses = 160 tests/window/bandwidth",
        "limitations": "Retrospective use of revised prices; first-publication times are unknown. This is not future or official contest-score evidence."},
        "hashes": hashes or {}, "twoMarketInputQuality": quality2,
        "twoMarket": _window_analysis(rows2, PRODUCTS_2M, WINDOWS_2M, "ridge", ("persistence",), 100, table="twoMarket", availability=availability2),
        "national": _window_analysis(rowsn, PRODUCTS_NATIONAL, WINDOWS_NATIONAL, "candidate", ("directRidge", "persistence"), 160, table="national", coverage_losses=national_coverage_losses)}


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"input must be a JSON object: {path}")
    return value, data


def _write_immutable(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite immutable output: {path}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run(two_market_path: Path, national_path: Path, output_path: Path) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_path}")
    two, tb = _read(Path(two_market_path)); nat, nb = _read(Path(national_path))
    hashes = {"twoMarketSha256": _sha(tb), "nationalTransferSha256": _sha(nb)}
    if hashes != {"twoMarketSha256": TWO_MARKET_SHA256, "nationalTransferSha256": NATIONAL_SHA256}:
        raise ValueError("input SHA-256 differs from frozen protocol")
    result = build(two, nat, hashes=hashes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    _write_immutable(output_path, encoded)
    return {"output": str(output_path), "sha256": _sha(encoded), "windows": 5}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-market", type=Path, default=INPUT_TWO_MARKET)
    parser.add_argument("--national", type=Path, default=INPUT_NATIONAL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.two_market, args.national, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
