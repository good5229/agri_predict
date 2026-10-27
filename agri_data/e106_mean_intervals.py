"""Descriptive circular-block intervals for stored E106 forecast aggregates.

This module scores only frozen forecasts from the preserved-improvement
composition; it does not refit models or alter the original DM test family.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import preserved_improvement_composition as protocol
from . import research_progress_diagnostic as bootstrap_protocol

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs/experiments/preserved-improvement-composition-20261011"
OUTPUT_DIR = ROOT / "outputs/experiments/e106-mean-intervals-20261011"
PRICES = protocol.PRICES
PRODUCTS = protocol.PRODUCTS
HORIZONS = protocol.HORIZONS
ALL_CELLS = frozenset((p, h) for p in PRODUCTS for h in HORIZONS)
GARLIC_CELLS = frozenset(("garlic", h) for h in HORIZONS)
PHASES = {
    "confirmation": {
        "manifest": INPUT_DIR / "confirmation-manifest.json",
        "report": INPUT_DIR / "confirmation-report.json",
        "manifestFileSha256": "1c32c11f0db65c1f06724cf8ae4ae02bd3455eedb1bafcdf827c9b0fef7e473b",
        "reportFileSha256": "d2a87314416d4761b480c8a08c37152b28667c2cc5a8b6cb77e25fac0d4b4984",
        "manifestDigest": "43bf757ead50951b93b1a713d40abf5ea21e00afa0ca53a163c9ba9ab9e4004b",
        "reportDigest": "4de80705c52bc764b7cdaef005bed30414171730209e59bcb9b023adc17cfced",
        "phaseName": "confirmation", "start": dt.date(2026, 4, 1), "end": dt.date(2026, 6, 20),
        "originFloor": dt.date(2026, 4, 1),
    },
    "final": {
        "manifest": INPUT_DIR / "final-manifest.json",
        "report": INPUT_DIR / "final-report.json",
        "manifestFileSha256": "c45cc99de60cc9df752194e694c4de8984d424c4edc03de2e283c23ddea9bbe2",
        "reportFileSha256": "7a936f4e50635c8c3a6d59db8afacc405ab55a4d287a2b318ac3eb3232d80074",
        "manifestDigest": "e6e45290486d61c61a8aec5a884db56e1a38570f054bdf6ebcb25b6c13b38f62",
        "reportDigest": "602f68f0522417fc51ccbec8392cd21a0da66e1fcccf5db76126d97c88452369",
        "phaseName": "final", "start": dt.date(2026, 7, 11), "end": dt.date(2026, 9, 27),
        "originFloor": dt.date(2026, 7, 1),
    },
}
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
RESEARCH_BOOTSTRAP_SHA256 = "993a856761324a333ea6520b9ae3f15540b341d8ac9d1f01c26ed4fcb59fd223"
BLOCKS = (14, 28)
REPLICATES = 2000
SEED = 20261011
COMPARISONS = {"E106_minus_E": "E", "E106_minus_causal50": "causal50"}
METRICS = ("ae", "ape")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def positive(value: Any) -> float | None:
    return protocol.positive(value)


def _verify_digest(doc: dict[str, Any], key: str, expected: str, label: str) -> None:
    unsigned = dict(doc)
    digest = unsigned.pop(key, None)
    actual = sha(canonical(unsigned).rstrip(b"\n"))
    if digest != expected or actual != expected:
        raise ValueError(f"{label} internal {key} mismatch")


def _expected_grid(phase: dict[str, Any]) -> set[tuple[str, dt.date, int]]:
    return {
        (product, target, horizon)
        for product in PRODUCTS
        for offset in range((phase["end"] - phase["start"]).days + 1)
        for target in (phase["start"] + dt.timedelta(days=offset),)
        for horizon in HORIZONS
        if target - dt.timedelta(days=horizon) >= phase["originFloor"]
    }


def _index_forecasts(manifest: dict[str, Any], phase_name: str, phase: dict[str, Any]) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    expected_phase = f"preserved_improvement_{phase_name}_forecast_only"
    if manifest.get("phase") != expected_phase or manifest.get("targetValuesAttached") is not False:
        raise ValueError(f"unexpected {phase_name} manifest phase or target attachment")
    rows = manifest.get("forecasts")
    if not isinstance(rows, list):
        raise ValueError("manifest forecasts must be a list")
    expected = _expected_grid(phase)
    indexed = {}
    for row in rows:
        try:
            product = row["product"]
            target = dt.date.fromisoformat(row["target"])
            origin = dt.date.fromisoformat(row["origin"])
            horizon = row["horizon"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed forecast key") from exc
        if product not in PRODUCTS or type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError("invalid forecast key")
        key = (product, target, horizon)
        if key in indexed or key not in expected:
            raise ValueError("duplicate or out-of-grid forecast key")
        if row.get("eligible") is not True or origin != target - dt.timedelta(days=horizon):
            raise ValueError("forecast eligibility/origin mismatch")
        for method in ("candidate", "E", "causal50"):
            model = row.get(method)
            if not isinstance(model, dict) or model.get("status") not in ("issued", "not_issued"):
                raise ValueError(f"missing or invalid {method} status")
            prediction = positive(model.get("prediction"))
            if (model["status"] == "issued") != (prediction is not None):
                raise ValueError(f"{method} status/prediction mismatch")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("forecast manifest does not cover its exact phase grid")
    return indexed


def _close(left: Any, right: Any, *, label: str, rel: float = 1e-9, abs_: float = 1e-8) -> None:
    if left is None or right is None:
        if left is not right:
            raise ValueError(f"saved metric mismatch: {label}")
        return
    if not math.isclose(float(left), float(right), rel_tol=rel, abs_tol=abs_):
        raise ValueError(f"saved metric mismatch: {label}: {left} != {right}")


def _verify_scored_report(manifest: dict[str, Any], report: dict[str, Any], indexed: dict,
                          actuals: dict[tuple[str, dt.date], float], phase_name: str, phase: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The frozen scorer expands this field last, so its persisted value is the
    # plain phase token even though the containing pipeline is named.
    expected_phase = phase_name
    if report.get("phase") != expected_phase or report.get("manifestSha256") != manifest.get("manifestSha256"):
        raise ValueError("scored report phase/manifest binding mismatch")
    if report.get("targetRange") != [phase["start"].isoformat(), phase["end"].isoformat()]:
        raise ValueError("scored report target range mismatch")
    common_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            keys = [(p, day, h) for p, day, h in indexed if p == product and h == horizon]
            cell_rows = [indexed[key] for key in keys]
            common = []
            for row in cell_rows:
                day = dt.date.fromisoformat(row["target"])
                actual = positive(actuals.get((product, day)))
                values = {method: positive(row[method].get("prediction")) for method in ("candidate", "E", "causal50")}
                if actual is not None and all(v is not None for v in values.values()):
                    common.append({"product": product, "target": row["target"], "horizon": horizon,
                                   "actual": actual, **values})
            common_by_cell[(product, horizon)] = common
    saved = {(r.get("product"), r.get("horizon"), r.get("method")): r
             for r in report.get("metricsByProductHorizonMethod", []) if r.get("method")}
    if len(saved) != len(PRODUCTS) * len(HORIZONS) * 3:
        raise ValueError("scored report does not contain exactly 120 method cells")
    cell_metrics = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows = common_by_cell[(product, horizon)]
            for method in ("candidate", "E", "causal50"):
                metric = protocol._metric([r["actual"] for r in rows], [r[method] for r in rows])
                saved_metric = saved.get((product, horizon, method))
                if saved_metric is None or saved_metric.get("n") != metric["n"]:
                    raise ValueError(f"saved cell support mismatch for {product}/{horizon}/{method}")
                _close(metric["mae"], saved_metric.get("mae"), label=f"{product}/{horizon}/{method}/MAE")
                _close(metric["mape"], saved_metric.get("mape"), label=f"{product}/{horizon}/{method}/MAPE")
                cell_metrics[(product, horizon, method)] = metric
            for comparator, base_method in COMPARISONS.items():
                saved_comparison = {"E106_minus_E": "candidateMinusE",
                                    "E106_minus_causal50": "candidateMinusCausal50"}[comparator]
                diff_row = next((r for r in report["metricsByProductHorizonMethod"]
                                 if r.get("product") == product and r.get("horizon") == horizon
                                 and r.get("comparison") == saved_comparison), None)
                if diff_row is None or diff_row.get("n") != len(rows):
                    raise ValueError(f"saved paired cell support missing for {product}/{horizon}/{comparator}")
                _close(cell_metrics[(product, horizon, "candidate")]["mae"] - cell_metrics[(product, horizon, base_method)]["mae"],
                       diff_row.get("maeDelta"), label=f"{product}/{horizon}/{comparator}/MAE delta")
                _close(cell_metrics[(product, horizon, "candidate")]["mape"] - cell_metrics[(product, horizon, base_method)]["mape"],
                       diff_row.get("mapeDelta"), label=f"{product}/{horizon}/{comparator}/MAPE delta")
    # Verify both original crop macros and the 40-cell macro, keeping their
    # equal-cell (not pooled-row) weighting exactly as the saved scorer used.
    saved_macro = {(r.get("product"), r.get("method")): r for r in report.get("productMacros", [])}
    saved_40 = report.get("fourCrop40CellEqualWeightMacro", {})
    for product in PRODUCTS:
        for method in ("candidate", "E", "causal50"):
            cells = [cell_metrics[(product, h, method)] for h in HORIZONS]
            macro = saved_macro.get((product, method))
            if macro is None or macro.get("cellsIncluded") != 10 or macro.get("expectedCells") != 10:
                raise ValueError(f"saved ten-horizon macro missing: {product}/{method}")
            _close(math.fsum(x["mae"] for x in cells) / 10, macro.get("mae"), label=f"{product}/{method} macro MAE")
            _close(math.fsum(x["mape"] for x in cells) / 10, macro.get("mape"), label=f"{product}/{method} macro MAPE")
    for method in ("candidate", "E", "causal50"):
        cells = [cell_metrics[(p, h, method)] for p in PRODUCTS for h in HORIZONS]
        saved_cell = saved_40.get(method)
        if saved_cell is None or saved_cell.get("cellsIncluded") != 40 or saved_cell.get("expectedCells") != 40:
            raise ValueError(f"saved 40-cell macro missing: {method}")
        _close(math.fsum(x["mae"] for x in cells) / 40, saved_cell.get("mae"), label=f"40-cell {method} MAE")
        _close(math.fsum(x["mape"] for x in cells) / 40, saved_cell.get("mape"), label=f"40-cell {method} MAPE")
    # Coverage, unique-date denominators, and exact E/candidate identity outside garlic.
    saved_coverage = report.get("coverage", {})
    common_dates = set()
    eligible_dates = {row["target"] for row in indexed.values()}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [row for (p, _d, h), row in indexed.items() if p == product and h == horizon]
            common = common_by_cell[(product, horizon)]
            common_dates.update(row["target"] for row in common)
            key = f"{product}/{horizon}"
            coverage = saved_coverage.get(key)
            if coverage is None:
                raise ValueError(f"saved coverage missing {key}")
            expected_coverage = {
                "plannedEligible": len(cell),
                "EIssued": sum(positive(r["E"].get("prediction")) is not None for r in cell),
                "candidateIssued": sum(positive(r["candidate"].get("prediction")) is not None for r in cell),
                "causal50Issued": sum(positive(r["causal50"].get("prediction")) is not None for r in cell),
                "candidateMissingEIssuedCount": sum(positive(r["E"].get("prediction")) is not None and positive(r["candidate"].get("prediction")) is None for r in cell),
                "coveragePreserved": all(positive(r["candidate"].get("prediction")) is not None
                                          for r in cell if positive(r["E"].get("prediction")) is not None),
                "positiveActualTargetCount": len({r["target"] for r in cell if positive(actuals.get((product, dt.date.fromisoformat(r["target"])))) is not None}),
                "commonPositivePairs": len(common), "uniquePairTargetDays": len({r["target"] for r in common}),
                "distinctOrigins": len({r["origin"] for r in cell}), "distinctTargets": len({r["target"] for r in cell}),
            }
            for field, value in expected_coverage.items():
                saved_value = coverage.get(field)
                if isinstance(value, bool):
                    if saved_value is not value:
                        raise ValueError(f"saved coverage mismatch {key}/{field}")
                elif saved_value != value:
                    raise ValueError(f"saved coverage mismatch {key}/{field}: {saved_value} != {value}")
            if product != "garlic":
                for r in cell:
                    if (r["candidate"].get("status"), r["candidate"].get("prediction")) != (r["E"].get("status"), r["E"].get("prediction")):
                        raise ValueError(f"candidate differs from E outside garlic at {product}/{r['target']}/{horizon}")
            else:
                if any((r["E"].get("status"), r["E"].get("prediction")) != (r["causal50"].get("status"), r["causal50"].get("prediction")) for r in cell):
                    raise ValueError("garlic E does not exactly equal causal50")
    if report.get("uniqueEligibleTargetDayCount") != len(eligible_dates) or report.get("uniqueCommonPairTargetDayCount") != len(common_dates):
        raise ValueError("saved unique eligible/common target-day count mismatch")
    if not common_by_cell or any(not common_by_cell[cell] for cell in ALL_CELLS):
        raise ValueError("one or more requested aggregate cells have no common pairs")
    return [row for group in common_by_cell.values() for row in group], {"cellMetrics": cell_metrics, "commonByCell": common_by_cell}


def _load_phase(phase_name: str, actuals: dict[tuple[str, dt.date], float]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    phase = PHASES[phase_name]
    manifest_raw = phase["manifest"].read_bytes()
    report_raw = phase["report"].read_bytes()
    if sha(manifest_raw) != phase["manifestFileSha256"]:
        raise ValueError(f"pinned {phase_name} manifest file SHA-256 mismatch")
    if sha(report_raw) != phase["reportFileSha256"]:
        raise ValueError(f"pinned {phase_name} report file SHA-256 mismatch")
    manifest, report = json.loads(manifest_raw), json.loads(report_raw)
    _verify_digest(manifest, "manifestSha256", phase["manifestDigest"], f"{phase_name} manifest")
    _verify_digest(report, "reportSha256", phase["reportDigest"], f"{phase_name} report")
    if manifest.get("sourceSha256", {}).get("prices") != PRICE_SHA256 or report.get("sourceSha256") != manifest.get("sourceSha256"):
        raise ValueError(f"{phase_name} source provenance mismatch")
    for source_key, expected_hash in protocol.EXPECTED.items():
        if manifest.get("sourceSha256", {}).get(source_key) != expected_hash:
            raise ValueError(f"{phase_name} frozen source provenance mismatch: {source_key}")
    if (sha(Path(bootstrap_protocol.__file__).read_bytes()) != RESEARCH_BOOTSTRAP_SHA256):
        raise ValueError("pinned calendar block bootstrap helper SHA-256 mismatch")
    expected_protocol = {"period": [phase["start"].isoformat(), phase["end"].isoformat()],
                         "eligibleOriginAtOrAfter": phase["originFloor"].isoformat(),
                         "noFit": True, "sourceForecastsCopiedWithoutRefit": True,
                         "samePairMethods": ["candidate", "E", "causal50"]}
    if manifest.get("phaseName") != phase_name or any(manifest.get("protocol", {}).get(k) != v for k, v in expected_protocol.items()):
        raise ValueError(f"{phase_name} frozen protocol/provenance mismatch")
    indexed = _index_forecasts(manifest, phase_name, phase)
    common_rows, summary = _verify_scored_report(manifest, report, indexed, actuals, phase_name, phase)
    return phase, manifest, report, common_rows, summary


def _analysis_rows(common_by_cell: dict[tuple[str, int], list[dict[str, Any]]], comparison: str, scope: str) -> list[dict[str, Any]]:
    comparator = COMPARISONS[comparison]
    cells = GARLIC_CELLS if scope == "garlic_h1_10" else ALL_CELLS
    rows = []
    for cell in sorted(cells):
        for row in common_by_cell[cell]:
            rows.append({"product": row["product"], "horizon": row["horizon"], "target": row["target"],
                         "actual": row["actual"], "a": row["candidate"], "b": row[comparator]})
    return rows


def bootstrap_estimate(rows: list[dict[str, Any]], *, metric: str, block: int, cells: frozenset,
                       replicates: int = REPLICATES, seed: int = SEED,
                       calendar_start: dt.date, calendar_end: dt.date) -> dict[str, Any]:
    """Run the frozen joint-calendar circular-block helper on explicit paired rows."""
    if metric not in METRICS:
        raise ValueError("metric must be ae or ape")
    expected_present = {(r["product"], r["horizon"]) for r in rows}
    if expected_present != set(cells):
        raise ValueError("paired rows do not cover the requested frozen cell family")
    result = bootstrap_protocol.circular_block_bootstrap(
        rows, a="a", b="b", metric=metric, block=block, replicates=replicates, seed=seed,
        calendar_start=calendar_start, calendar_end=calendar_end)
    if result["valid"] + result["invalid"] != replicates:
        raise ValueError("bootstrap valid/invalid replicate accounting mismatch")
    return result


def build_report() -> tuple[dict[str, Any], str]:
    prices_raw = PRICES.read_bytes()
    if sha(prices_raw) != PRICE_SHA256:
        raise ValueError("pinned historical prices SHA-256 mismatch")
    prices = json.loads(prices_raw)
    actuals = protocol._actual_series(prices)
    input_records = []
    phase_summaries = {}
    all_rows_for_relationships = {}
    for phase_name, spec in PHASES.items():
        phase, manifest, report, _flat, verification = _load_phase(phase_name, actuals)
        input_records.append({"phase": phase_name, "manifestPath": str(spec["manifest"]),
                              "manifestFileSha256": spec["manifestFileSha256"],
                              "manifestDigest": manifest["manifestSha256"], "reportPath": str(spec["report"]),
                              "reportFileSha256": spec["reportFileSha256"], "reportDigest": report["reportSha256"],
                              "originalDmProtocol": report.get("dmProtocol"),
                              "originalDmConclusions": report.get("dmConclusions")})
        common_by_cell = verification["commonByCell"]
        all_rows_for_relationships[phase_name] = common_by_cell
        spec_summaries = {}
        for comparison in COMPARISONS:
            for scope, cells in (("garlic_h1_10", GARLIC_CELLS), ("four_crop_40_cells", ALL_CELLS)):
                rows = _analysis_rows(common_by_cell, comparison, scope)
                if not rows:
                    raise ValueError(f"no paired rows for {phase_name}/{comparison}/{scope}")
                expected = bootstrap_protocol._cell_means(rows, "a", "b", "ae", expected_cells=cells)
                if expected is None:
                    raise ValueError(f"aggregate cells are incomplete: {phase_name}/{comparison}/{scope}")
                analyses = {}
                for metric in METRICS:
                    point = bootstrap_protocol._cell_means(rows, "a", "b", metric, expected_cells=cells)
                    if point is None:
                        raise ValueError("could not compute point estimate")
                    estimates = {}
                    for block in BLOCKS:
                        estimates[str(block)] = bootstrap_estimate(
                            rows, metric=metric, block=block, cells=cells,
                            calendar_start=spec["start"], calendar_end=spec["end"])
                    analyses[metric] = {"pointDifference": point["difference"],
                                        "differenceDefinition": "E106(candidate) minus comparator; negative favors E106",
                                        "unit": "price units" if metric == "ae" else "percentage points",
                                        "pairedRows": len(rows), "cellsIncluded": len(cells), "cellsExpected": len(cells),
                                        "intervals95": estimates}
                spec_summaries[f"{comparison}/{scope}"] = analyses
        # For E106-E, the all-40-cell mean is exactly 1/4 of garlic's ten-cell
        # equal-weight mean under the same rows and weights. Assert this identity.
        for metric in METRICS:
            garlic = spec_summaries[f"E106_minus_E/garlic_h1_10"][metric]["pointDifference"]
            four_crop = spec_summaries[f"E106_minus_E/four_crop_40_cells"][metric]["pointDifference"]
            if not math.isclose(four_crop, garlic / 4, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"E106-E four-crop {metric} mean is not exactly one quarter of garlic's ten-cell mean")
            full = report["fourCrop40CellEqualWeightMacro"]
            reported_full = full["candidate"]["mae" if metric == "ae" else "mape"] - full["E"]["mae" if metric == "ae" else "mape"]
            _close(four_crop, reported_full, label=f"{phase_name} E106-E 40-cell {metric} report delta")
            reported_garlic = next(x for x in report["productMacros"] if x["product"] == "garlic" and x["method"] == "candidate")["mae" if metric == "ae" else "mape"] - next(x for x in report["productMacros"] if x["product"] == "garlic" and x["method"] == "E")["mae" if metric == "ae" else "mape"]
            _close(garlic, reported_garlic, label=f"{phase_name} E106-E garlic {metric} report delta")
        phase_summaries[phase_name] = {"targetRange": [spec["start"].isoformat(), spec["end"].isoformat()],
                                       "calendarDays": (spec["end"] - spec["start"]).days + 1,
                                       "uniqueEligibleTargetDays": report["uniqueEligibleTargetDayCount"],
                                       "uniqueCommonPairTargetDays": report["uniqueCommonPairTargetDayCount"],
                                       "coverageRecomputedFromStoredPredictions": True,
                                       "storedScoresRecomputedAndVerified": True,
                                       "comparisons": spec_summaries,
                                       "originalDmResults": {"protocol": report.get("dmProtocol"),
                                                              "conclusions": report.get("dmConclusions"),
                                                              "reportSha256": report["reportSha256"],
                                                              "recomputedOrChanged": False}}
    sources = {
        "historicalPricesPath": str(PRICES), "historicalPricesFileSha256": PRICE_SHA256,
        "bootstrapProtocol": "agri_data.research_progress_diagnostic.circular_block_bootstrap",
        "bootstrapProtocolSha256": RESEARCH_BOOTSTRAP_SHA256,
        "compositionScorer": "agri_data.preserved_improvement_composition._score_rows",
        "compositionScorerSha256": sha(Path(protocol.__file__).read_bytes()),
        "intervalModuleSha256": sha(Path(__file__).read_bytes()),
    }
    doc = {"schemaVersion": 1, "phase": "e106_mean_interval_diagnostic",
           "purpose": "descriptive retrospective uncertainty intervals for stored E106 aggregate loss differences; not an adoption test",
           "protocol": {"methods": {"E106": "candidate", "E": "stored E baseline", "causal50": "stored causal50 comparator"},
                        "aggregates": {"garlic_h1_10": "ten garlic horizons, equally weighted after within-cell mean",
                                       "four_crop_40_cells": "four crops × ten horizons, equally weighted after within-cell mean"},
                        "metrics": {"ae": "MAE difference in price units", "ape": "MAPE difference in percentage points"},
                        "blocksCalendarDays": list(BLOCKS), "primaryBlockDays": 14, "sensitivityBlockDays": 28,
                        "replicates": REPLICATES, "seed": SEED,
                        "resampling": "circular calendar blocks jointly across all rows; retain the original per-cell observation masks; recalculate each included cell mean on every draw, then give cells equal weight",
                        "invalidReplicates": "explicitly counted when a draw leaves any expected cell without a sampled common pair; no missing calendar prices are imputed",
                        "direction": "E106 minus comparator; negative favors E106",
                        "garlicRelationship": "On the same common pairs and equal-cell weights, the E106-E 40-cell mean difference equals exactly one quarter of the garlic ten-cell mean difference; these are linked summaries, not independent evidence.",
                        "dmHandling": "Existing DM/HAC/Holm results are copied by reference from frozen reports and are neither recomputed nor extended."},
           "sources": sources, "sourceArtifacts": input_records, "periods": phase_summaries,
           "limitations": ["Retrospective periods have been repeatedly inspected; these bootstrap intervals are descriptive and do not create independent validation.",
                           "Bootstrap draws preserve joint target-day dependence and observed missingness, but do not model future regime change.",
                           "The four-crop E106-E point/interval is linked to the garlic-only replacement and must not be counted as a separate independent improvement." ]}
    md = render_markdown(doc)
    return doc, md


def render_markdown(doc: dict[str, Any]) -> str:
    lines = ["# E106 aggregate mean-difference intervals", "",
             "These are descriptive retrospective intervals from stored forecasts. They do not add an adoption gate or change any original DM/HAC/Holm result.", "",
             "Differences are E106 minus the named comparator: negative values favor E106. MAE is in price units; MAPE is in percentage points.", "",
             "| Period | Comparison | Aggregate | Metric | Point difference | 14-day 95% interval (valid/invalid) | 28-day sensitivity (valid/invalid) |",
             "|---|---|---|---|---:|---|---|"]
    for phase, period in doc["periods"].items():
        for label, analyses in period["comparisons"].items():
            comparison, scope = label.split("/", 1)
            for metric, result in analyses.items():
                vals=[]
                for block in BLOCKS:
                    item=result["intervals95"][str(block)]
                    interval=item["interval95"]
                    interval_text="unavailable" if interval is None else f"[{interval[0]:.5f}, {interval[1]:.5f}]"
                    vals.append(f"{interval_text} ({item['valid']}/{item['invalid']})")
                displayed = {"E106_minus_E": "E106 − E", "E106_minus_causal50": "E106 − causal50"}[comparison]
                lines.append(f"| {phase} | {displayed} | {scope} | {metric.upper()} | {result['pointDifference']:.6f} | {vals[0]} | {vals[1]} |")
    lines.extend(["", "The E106−E 40-cell estimate is exactly one quarter of the garlic ten-horizon estimate because the other three crops are unchanged. The two summaries therefore describe the same replacement effect at different aggregation scales.", "",
                  "The original reports’ DM/HAC/Holm results remain the inferential record. This interval addendum does not recalculate or extend those test families. The periods are previously inspected retrospective data, so the intervals are descriptive rather than prospective confirmation.", "",
                  "Frozen report file SHA-256 values:"])
    for rec in doc["sourceArtifacts"]:
        lines.append(f"- {rec['phase']}: manifest `{rec['manifestFileSha256']}`; scored report `{rec['reportFileSha256']}`.")
    return "\n".join(lines) + "\n"


def write_exclusive(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    report, markdown = build_report()
    output_dir = Path(output_dir)
    write_exclusive(output_dir / "report.json", canonical(report))
    write_exclusive(output_dir / "report.md", markdown.encode())
    return {"outputDir": str(output_dir), "phase": report["phase"],
            "reportSha256": sha(canonical(report)),
            "jsonPath": str(output_dir / "report.json"), "markdownPath": str(output_dir / "report.md")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
