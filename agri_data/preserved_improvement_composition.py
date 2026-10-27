"""Compose the frozen preserved-improvement candidate without fitting models.

The answer-free composition copies the frozen E forecast for cabbage, radish,
and onion and replaces only garlic with its stored preserved106Blend forecast.
Scoring is replay-gated and writes each phase to a new immutable directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import direct_horizon
from . import equal_info_online_forecasts as e_forecasts
from . import equal_info_online_score as e_score
from . import full_history_return_eval as price_helper
from . import preserved_feature_forecasts as preserved
from . import trend_response_correction as trend

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/experiments/preserved-improvement-composition-20261011"
E_MANIFEST = ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
PRESERVED_EVALUATION = ROOT / "outputs/experiments/preserved-feature-20261010/evaluation-manifest.json"
TREND_DIR = ROOT / "outputs/experiments/trend-response-correction-20261010"
TREND_CONFIRMATION = TREND_DIR / "confirmation-manifest.json"
TREND_FINAL = TREND_DIR / "final-manifest.json"
TREND_SELECTION = TREND_DIR / "selection.json"
PRICES = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
EXPECTED = {
    "eManifest": "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df",
    "preservedEvaluation": "35164a11143768f26e7567e80643ec323c7279083ac5e6ef166a7fe8d185aa20",
    "trendConfirmation": "bb929cadbdee4862b71ac93f97a0ecbed9ea7ecb3ff78ac5251e281bdea5d74b",
    "trendFinal": "7ec6453660e020298bf407b829fdda9c8807c739769b5e58abc4d04631662502",
    "trendSelection": "59fa52c3f7c9664d52ad8ce23987eb4d6c70af0fca9907c25596a84eb77ba26b",
    "trendCorrectionBuilder": "f3993916dda53e278f1a2b05bb24089a45831f9f6c88afcc80415d7573861ec4",
    "prices": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
    "eForecastBuilder": "84e68a9a3eb362e59a336440a0bf9254eacb68dfb881e6cf3abdeeba4a215471",
    "preservedForecastBuilder": "f7bee365ce894f7046904a64f575bb64c80099a483d12a2b0016417a8a84c49e",
    "eScorer": "603e49b412bc51d8832807a1ff60aad548a9346e4e3754c59502f119cbae0886",
    "directHorizon": "1cf3f8ce5455d9ce1f8d4eeb6e14f84e465acf4f5e6d512670b102115fb96d40",
    "priceHelper": "276f389e7153e07ef02836183d55e9823a6eb7c9bf24acc882d663946c55c5c5",
}
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
CONFIRMATION = ("confirmation", dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1))
FINAL = ("final", dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1))
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
ALPHA = .05


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def parse_day(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _source_hashes() -> dict[str, str]:
    paths = {
        "eForecastBuilder": Path(e_forecasts.__file__),
        "preservedForecastBuilder": Path(preserved.__file__),
        "eScorer": Path(e_score.__file__),
        "directHorizon": Path(direct_horizon.__file__),
        "priceHelper": Path(price_helper.__file__),
    }
    hashes = {name: file_sha(path) for name, path in paths.items()}
    if hashes != {name: EXPECTED[name] for name in paths}:
        raise ValueError("pinned forecast or scoring helper SHA-256 mismatch")
    return hashes


def _index(rows: Any, label: str, allowed_products=None) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows must be a list")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"malformed {label} row")
        product, day, horizon = row.get("product"), parse_day(row.get("target")), row.get("horizon")
        if product not in (allowed_products or e_forecasts.PRODUCTS) or day is None or type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError(f"invalid {label} key")
        key = (product, day, horizon)
        if key in out:
            raise ValueError(f"duplicate {label} key: {key}")
        out[key] = row
    return out


def _check_source_documents(e_doc: dict[str, Any], p_doc: dict[str, Any],
                            trend_confirmation: dict[str, Any], trend_final: dict[str, Any],
                            selection: dict[str, Any]) -> None:
    e_score._validate_manifest(e_doc)
    preserved._validate_manifest(p_doc, "evaluation")
    if p_doc.get("selectedCrops") is None or "garlic" not in p_doc["selectedCrops"]:
        raise ValueError("preserved-feature evaluation did not select garlic")
    e_hashes = e_doc.get("inputSha256", {})
    p_hashes = p_doc.get("inputSha256", {})
    if e_hashes.get("module") != EXPECTED["eForecastBuilder"]:
        raise ValueError("E manifest builder lineage mismatch")
    if e_hashes.get("prices") != EXPECTED["prices"]:
        raise ValueError("E manifest price lineage mismatch")
    expected_p = {
        "equalInfoBaseline": EXPECTED["eManifest"], "prices": EXPECTED["prices"],
        "equalInfoBuilder": EXPECTED["eForecastBuilder"],
        "preservedFeatureGenerator": EXPECTED["preservedForecastBuilder"],
        "directHorizonHelper": EXPECTED["directHorizon"],
    }
    for key, digest in expected_p.items():
        if p_hashes.get(key) != digest:
            raise ValueError(f"preserved-feature {key} lineage mismatch")
    if p_doc.get("protocol", {}).get("featureSchema", {}).get("totalCount") != 106:
        raise ValueError("preserved-feature candidate is not the frozen 106-feature manifest")
    selection_unsigned = dict(selection)
    selection_digest = selection_unsigned.pop("selectionSha256", None)
    if selection_digest != sha(canonical(selection_unsigned)):
        raise ValueError("trend selection digest mismatch")
    if (selection.get("inputSha256", {}).get("baselineManifest") != EXPECTED["eManifest"]
            or selection.get("inputSha256", {}).get("prices") != EXPECTED["prices"]):
        raise ValueError("trend selection is not bound to the frozen equal-info baseline")
    for doc, phase in ((trend_confirmation, "correction_forecast_only"),
                       (trend_final, "final_forecast_only")):
        _validate_trend_phase_manifest(doc, phase, selection_digest)
    trend._validate_correction_manifest(trend_confirmation)
    for row in trend_final.get("finalForecasts", []):
        if row.get("product") not in PRODUCTS or parse_day(row.get("target")) is None or parse_day(row.get("origin")) is None:
            raise ValueError("invalid trend final source key")
        if row.get("horizon") not in HORIZONS or parse_day(row["target"]) - dt.timedelta(days=row["horizon"]) != parse_day(row["origin"]):
            raise ValueError("invalid trend final source origin")
    if len(_index(trend_final.get("finalForecasts"), "trend-final")) != len(trend_final.get("finalForecasts", [])):
        raise ValueError("duplicate trend final source key")


def _phase_grid(start: dt.date, end: dt.date, eligible_origin: dt.date):
    for product in PRODUCTS:
        for offset in range((end - start).days + 1):
            target = start + dt.timedelta(days=offset)
            for horizon in HORIZONS:
                if target - dt.timedelta(days=horizon) >= eligible_origin:
                    yield (product, target, horizon)


def _same_control(left: Any, right: Any, key: tuple[str, dt.date, int], method: str) -> None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError(f"missing {method} control at {key}")
    if left.get("status") != right.get("status") or left.get("prediction") != right.get("prediction"):
        raise ValueError(f"{method} differs between E and preserved-feature source at {key}")


def _validate_trend_phase_manifest(doc: dict[str, Any], expected_phase: str, selection_digest: str) -> None:
    unsigned = dict(doc)
    digest = unsigned.pop("manifestSha256", None)
    if digest != sha(canonical(unsigned)) or doc.get("targetValuesAttached") is not False:
        raise ValueError("trend E source manifest digest/answer-free check failed")
    if doc.get("phase") != expected_phase or doc.get("baselineManifestSha256") != EXPECTED["eManifest"]:
        raise ValueError("trend E phase or baseline binding mismatch")
    if (doc.get("selectionSha256") != selection_digest
            or doc.get("inputSha256", {}).get("baselineManifest") != EXPECTED["eManifest"]
            or doc.get("inputSha256", {}).get("prices") != EXPECTED["prices"]):
        raise ValueError("trend E manifest is not bound to the frozen selection/baseline")
    if doc.get("implementationSha256", {}).get("trendResponseCorrection") != EXPECTED["trendCorrectionBuilder"]:
        raise ValueError("trend E implementation lineage mismatch")


def _compose_rows(e_rows: dict, p_rows: dict, trend_rows: dict, selection: dict,
                  start: dt.date, end: dt.date, eligible_origin: dt.date):
    selection_map = {(r["product"], r["regime"]): r for r in selection.get("groups", [])}
    expected_keys = set(_phase_grid(start, end, eligible_origin))
    if set(trend_rows) != expected_keys:
        raise ValueError("saved E trend manifest does not exactly cover its frozen eligible phase grid")
    rows = []
    for key in _phase_grid(start, end, eligible_origin):
        product, target, horizon = key
        e_row = e_rows.get(key)
        p_row = p_rows.get(key)
        if e_row is None or p_row is None:
            raise ValueError(f"source forecast key missing at {key}")
        origin = target - dt.timedelta(days=horizon)
        if parse_day(e_row.get("origin")) != origin or parse_day(p_row.get("origin")) != origin:
            raise ValueError(f"source origin mismatch at {key}")
        e_models = e_row.get("models", {})
        p_models = p_row.get("models", {})
        for method in ("causalDirect", "causalSarima", "causal50"):
            _same_control(e_models.get(method), p_models.get(method), key, method)
        correction = trend_rows[key]
        if parse_day(correction.get("origin")) != origin:
            raise ValueError(f"saved E correction origin mismatch at {key}")
        base = e_models["causal50"]
        correction_base = positive(correction.get("baselinePrediction"))
        correction_base_model = {"status": "issued", "prediction": correction_base} if correction_base is not None else {"status": "not_issued", "prediction": None}
        _same_control(correction_base_model, base, key, "causal50 source")
        saved = correction.get("model", {})
        if product == "cabbage":
            features = correction.get("features", {})
            regime = features.get("regime")
            selected = selection_map.get((product, regime))
            if correction.get("correctionApplied"):
                if not selected or not selected.get("supported") or correction.get("scale") != selected.get("scale"):
                    raise ValueError(f"trend correction scale does not replay frozen selection at {key}")
                replay = trend._scale_movement(correction.get("baselinePrediction"), features.get("p0"), correction["scale"])
            else:
                if correction.get("scale") != 1.0:
                    raise ValueError(f"uncorrected E row has non-unit scale at {key}")
                replay = positive(correction.get("baselinePrediction"))
            if saved.get("prediction") != replay or (saved.get("status") == "issued") != (replay is not None):
                raise ValueError(f"saved cabbage E forecast fails pure-transform replay at {key}")
            baseline = dict(saved)
        else:
            _same_control(saved, base, key, "saved E causal50")
            baseline = dict(e_models["causal50"])
        e_model = "trendResponseCorrection" if product == "cabbage" else "causal50"
        candidate = dict(p_models["preserved106Blend"]) if product == "garlic" else dict(baseline)
        causal50 = e_models["causal50"]
        for label, model in (("candidate", candidate), ("E", baseline), ("causal50", causal50)):
            if not isinstance(model, dict) or model.get("status") not in ("issued", "not_issued"):
                raise ValueError(f"invalid {label} forecast at {key}")
            if (model.get("status") == "issued") != (positive(model.get("prediction")) is not None):
                raise ValueError(f"invalid {label} status/prediction at {key}")
        rows.append({"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                     "horizon": horizon, "eligible": origin >= eligible_origin,
                     "candidate": dict(candidate), "E": dict(baseline), "causal50": dict(causal50),
                     "candidateSource": "preserved106Blend" if product == "garlic" else "E",
                     "eSource": e_model})
    return rows


def build_phase_manifest(e_doc: dict[str, Any], p_doc: dict[str, Any], phase: str,
                         trend_confirmation: dict[str, Any], trend_final: dict[str, Any],
                         selection: dict[str, Any], source_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    """Compose a phase from stored manifests only; never refit or replay-fit."""
    _check_source_documents(e_doc, p_doc, trend_confirmation, trend_final, selection)
    spec = CONFIRMATION if phase == "confirmation" else FINAL if phase == "final" else None
    if spec is None:
        raise ValueError("phase must be confirmation or final")
    name, start, end, eligible_origin = spec
    # Both frozen 2026 subperiods come from the answer-free E evaluation grid;
    # its separate confirmation grid is the unrelated 2025Q4 protocol period.
    e_rows = _index(e_doc["evaluationForecasts"], "E")
    p_rows = _index(p_doc["forecasts"], "preserved-feature")
    trend_doc = trend_confirmation if phase == "confirmation" else trend_final
    trend_key = "confirmationForecasts" if phase == "confirmation" else "finalForecasts"
    trend_rows = _index(trend_doc[trend_key], f"trend-{phase}", PRODUCTS)
    rows = _compose_rows(e_rows, p_rows, trend_rows, selection, start, end, eligible_origin)
    doc = {"schemaVersion": 1, "phase": f"preserved_improvement_{name}_forecast_only",
           "phaseName": name, "targetValuesAttached": False,
           "protocol": {"candidate": "saved trend-response E for cabbage; causal50 for radish/onion; stored preserved106Blend for garlic",
                        "primaryComparison": "candidate_vs_E; garlic E is causal50",
                        "reference": "causal50; equal to E for radish/onion/garlic",
                        "period": [start.isoformat(), end.isoformat()],
                        "eligibleOriginAtOrAfter": eligible_origin.isoformat(),
                        "noFit": True, "sourceForecastsCopiedWithoutRefit": True,
                        "samePairMethods": ["candidate", "E", "causal50"]},
           "sourceSha256": source_hashes or {}, "forecasts": rows}
    doc["manifestSha256"] = sha(canonical(doc).rstrip(b"\n"))
    return doc


def verify_phase_manifest(candidate: dict[str, Any], e_doc: dict[str, Any], p_doc: dict[str, Any], phase: str,
                          trend_confirmation: dict[str, Any], trend_final: dict[str, Any],
                          selection: dict[str, Any], source_hashes: dict[str, str] | None = None) -> bool:
    expected = build_phase_manifest(e_doc, p_doc, phase, trend_confirmation, trend_final, selection, source_hashes)
    return canonical(candidate) == canonical(expected)


def _actual_series(prices: dict[str, Any]) -> dict[tuple[str, dt.date], float]:
    series = price_helper._series(prices)
    return {(product, day): value for product, observations in series.items() for day, value in observations.items()}


def _metric(actuals: list[float], predictions: list[float]) -> dict[str, Any]:
    if not actuals:
        return {"n": 0, "mae": None, "mape": None}
    errors = [abs(actual - prediction) for actual, prediction in zip(actuals, predictions)]
    return {"n": len(actuals), "mae": math.fsum(errors) / len(errors),
            "mape": 100 * math.fsum(error / actual for error, actual in zip(errors, actuals)) / len(actuals)}


def _dm_conclusions(dm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the frozen garlic AE/APE family without treating null tests as evidence."""
    by_band = {}
    by_horizon = {}
    for bandwidth in BANDWIDTHS:
        family = [r for r in dm_rows if r.get("product") == "garlic" and r.get("bandwidth") == bandwidth]
        counts = {label: 0 for label in ("improved", "worsened", "inconclusive", "inferenceUnavailable")}
        horizon_results = {}
        for horizon in HORIZONS:
            pair = [r for r in family if r.get("horizon") == horizon]
            if len(pair) != 2:
                status, reason = "inferenceUnavailable", "expected two losses (AE and APE) were not present"
            elif any(r.get("status") != "tested" or r.get("pValue") is None for r in pair):
                status, reason = "inferenceUnavailable", "; ".join(sorted({str(r.get("reason", r.get("status"))) for r in pair}))
            else:
                signs = [(-1 if r["meanLossDifference"] < 0 else 1 if r["meanLossDifference"] > 0 else 0) for r in pair]
                sig = [bool(r.get("holmSignificant")) for r in pair]
                if all(sig) and all(sign < 0 for sign in signs):
                    status, reason = "improved", "both Holm-adjusted AE and APE tests favor candidate"
                elif all(sig) and all(sign > 0 for sign in signs):
                    status, reason = "worsened", "both Holm-adjusted AE and APE tests favor E"
                elif any(sig):
                    status, reason = "inconclusive", "loss directions or significance disagree"
                else:
                    status, reason = "inconclusive", "no joint significant improvement; nonsignificance is not equivalence"
            counts[status] += 1
            horizon_results[str(horizon)] = {"conclusion": status, "reason": reason}
        by_band[str(bandwidth)] = {"familyTests": len(family), "horizons": counts,
                                  "interpretation": "Nonsignificance is not equivalence."}
        by_horizon[str(bandwidth)] = horizon_results
    return {"testedProduct": "garlic", "byBandwidth": by_band, "byHorizon": by_horizon}


def _method_prediction(row: dict[str, Any], method: str) -> float | None:
    return positive(row.get(method, {}).get("prediction"))


def _score_rows(rows: list[dict[str, Any]], actuals: dict[tuple[str, dt.date], float], phase: str) -> dict[str, Any]:
    scoped = [r for r in rows if r.get("eligible") is True]
    cell_rows = []
    coverage = {}
    monthly = []
    common = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in scoped if r["product"] == product and r["horizon"] == horizon]
            issue_count = {method: sum(_method_prediction(r, method) is not None for r in cell)
                           for method in ("candidate", "E", "causal50")}
            e_issued = [r for r in cell if _method_prediction(r, "E") is not None]
            candidate_lost = [r["target"] for r in e_issued if _method_prediction(r, "candidate") is None]
            coverage_preserved = not candidate_lost
            usable = []
            for row in cell:
                day = parse_day(row["target"])
                actual = actuals.get((product, day))
                preds = {method: _method_prediction(row, method) for method in ("candidate", "E", "causal50")}
                if positive(actual) and all(value is not None for value in preds.values()):
                    usable.append({"row": row, "actual": float(actual), **preds})
            common[(product, horizon)] = usable
            for method in ("candidate", "E", "causal50"):
                metric = _metric([x["actual"] for x in usable], [x[method] for x in usable])
                cell_rows.append({"product": product, "horizon": horizon, "method": method, **metric})
            for comparison, baseline in (("candidateMinusE", "E"), ("candidateMinusCausal50", "causal50")):
                cand_metric = _metric([x["actual"] for x in usable], [x["candidate"] for x in usable])
                base_metric = _metric([x["actual"] for x in usable], [x[baseline] for x in usable])
                cell_rows.append({"product": product, "horizon": horizon, "comparison": comparison,
                                  "n": len(usable), "maeDelta": (cand_metric["mae"] - base_metric["mae"] if usable else None),
                                  "mapeDelta": (cand_metric["mape"] - base_metric["mape"] if usable else None)})
            coverage[f"{product}/{horizon}"] = {
                "plannedEligible": len(cell), "EIssued": issue_count["E"], "candidateIssued": issue_count["candidate"],
                "causal50Issued": issue_count["causal50"], "candidateMissingEIssuedCount": len(candidate_lost),
                "candidateMissingEIssuedTargets": candidate_lost, "coveragePreserved": coverage_preserved,
                "positiveActualTargetCount": len({r["target"] for r in cell if positive(actuals.get((product, parse_day(r["target"])))) is not None}),
                "commonPositivePairs": len(usable), "uniquePairTargetDays": len({x["row"]["target"] for x in usable}),
                "distinctOrigins": len({r["origin"] for r in cell}), "distinctTargets": len({r["target"] for r in cell}),
            }
            for month in sorted({r["target"][:7] for r in cell}):
                mrows = [x for x in usable if x["row"]["target"].startswith(month)]
                metrics = {method: _metric([x["actual"] for x in mrows], [x[method] for x in mrows])
                           for method in ("candidate", "E", "causal50")}
                monthly.append({"product": product, "horizon": horizon, "month": month,
                                "commonPairs": len(mrows), "methods": metrics,
                                "candidateMinusE": {k: (metrics["candidate"][k] - metrics["E"][k] if mrows else None) for k in ("mae", "mape")},
                                "candidateMinusCausal50": {k: (metrics["candidate"][k] - metrics["causal50"][k] if mrows else None) for k in ("mae", "mape")}})
    macros = []
    for product in PRODUCTS:
        for method in ("candidate", "E", "causal50"):
            cells = [x for x in cell_rows if x.get("product") == product and x.get("method") == method and x["n"] > 0]
            metrics = {k: (math.fsum(x[k] for x in cells) / len(cells) if len(cells) == len(HORIZONS) else None)
                       for k in ("mae", "mape")}
            macros.append({"product": product, "method": method, "cellsIncluded": len(cells),
                           "expectedCells": len(HORIZONS), **metrics})
    full_macro = {}
    for method in ("candidate", "E", "causal50"):
        cells = [x for x in cell_rows if x.get("method") == method and x["n"] > 0]
        full_macro[method] = {"cellsIncluded": len(cells), "expectedCells": len(PRODUCTS) * len(HORIZONS),
                              "mae": (math.fsum(x["mae"] for x in cells) / len(cells) if len(cells) == len(PRODUCTS) * len(HORIZONS) else None),
                              "mape": (math.fsum(x["mape"] for x in cells) / len(cells) if len(cells) == len(PRODUCTS) * len(HORIZONS) else None)}
    dm = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            pairs = common[(product, horizon)]
            for loss in ("ae", "ape"):
                if product != "garlic":
                    for bandwidth in BANDWIDTHS:
                        dm.append({"product": product, "horizon": horizon, "loss": loss, "bandwidth": bandwidth,
                                   "status": "not_tested_by_protocol", "reason": "non_garlic_candidate_is_identical_to_E",
                                   "n": len(pairs), "statistic": None, "meanLossDifference": None,
                                   "pValue": 1.0, "holmPValue": None, "holmSignificant": False})
                    continue
                diffs = {}
                for item in pairs:
                    actual = item["actual"]
                    diff = abs(item["candidate"] - actual) - abs(item["E"] - actual)
                    diffs[parse_day(item["row"]["target"])] = diff if loss == "ae" else 100 * diff / actual
                for bandwidth in BANDWIDTHS:
                    test = e_score._dm(diffs, bandwidth)
                    dm.append({"product": product, "horizon": horizon, "loss": loss, "bandwidth": bandwidth, **test})
    for bandwidth in BANDWIDTHS:
        family = [x for x in dm if x["product"] == "garlic" and x["bandwidth"] == bandwidth]
        if len(family) != 20:
            raise AssertionError("each phase's garlic DM family must contain exactly 20 tests")
        e_score._holm(family)
    dm_conclusion = _dm_conclusions(dm)
    target_days = sorted({r["target"] for r in scoped})
    unique_days = sorted({item["row"]["target"] for group in common.values() for item in group})
    return {"phase": phase, "metricsByProductHorizonMethod": cell_rows, "productMacros": macros,
            "fourCrop40CellEqualWeightMacro": full_macro, "coverage": coverage,
            "uniqueEligibleTargetDayCount": len(target_days), "uniqueCommonPairTargetDayCount": len(unique_days),
            "monthlyAndDistanceDiagnostics": monthly, "dm": dm, "dmConclusions": dm_conclusion,
            "dmProtocol": {"testedProduct": "garlic", "familySizePerBandwidth": 20,
                           "losses": ["ae", "ape"], "horizons": list(HORIZONS),
                           "calendarHacDays": list(BANDWIDTHS), "holmAlpha": ALPHA,
                           "minimumPairs": MIN_PAIRS, "nonGarlicPValue": 1.0}}


def score_phase(candidate: dict[str, Any], e_doc: dict[str, Any], p_doc: dict[str, Any], prices: dict[str, Any],
                phase: str, trend_confirmation: dict[str, Any] | None = None,
                trend_final: dict[str, Any] | None = None, selection: dict[str, Any] | None = None,
                source_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    if not verify_phase_manifest(candidate, e_doc, p_doc, phase, trend_confirmation, trend_final, selection, source_hashes):
        raise ValueError("answer-free composition manifest rederivation mismatch")
    actuals = _actual_series(prices)
    spec = CONFIRMATION if phase == "confirmation" else FINAL
    scored = _score_rows(candidate["forecasts"], actuals, phase)
    confirmation_gate = None
    if phase == "confirmation":
        by_method = {(x.get("product"), x.get("horizon"), x.get("method")): x
                     for x in scored["metricsByProductHorizonMethod"] if x.get("method")}
        candidate_rows = [by_method[("garlic", h, "candidate")] for h in HORIZONS]
        baseline_rows = [by_method[("garlic", h, "E")] for h in HORIZONS]
        counts = {str(h): by_method[("garlic", h, "candidate")]["n"] for h in HORIZONS}
        candidate_mae = math.fsum(x["mae"] for x in candidate_rows) / 10 if all(x["mae"] is not None for x in candidate_rows) else None
        candidate_mape = math.fsum(x["mape"] for x in candidate_rows) / 10 if all(x["mape"] is not None for x in candidate_rows) else None
        baseline_mae = math.fsum(x["mae"] for x in baseline_rows) / 10 if all(x["mae"] is not None for x in baseline_rows) else None
        baseline_mape = math.fsum(x["mape"] for x in baseline_rows) / 10 if all(x["mape"] is not None for x in baseline_rows) else None
        garlic_coverage = all(scored["coverage"][f"garlic/{h}"]["coveragePreserved"] for h in HORIZONS)
        all_coverage = all(x["coveragePreserved"] for x in scored["coverage"].values())
        pass_gate = (all(n >= MIN_PAIRS for n in counts.values()) and garlic_coverage and all_coverage
                     and candidate_mae is not None and baseline_mae is not None and candidate_mape is not None and baseline_mape is not None
                     and candidate_mae < baseline_mae and candidate_mape < baseline_mape)
        confirmation_gate = {"passed": bool(pass_gate), "allTenHorizonsAtLeast30Pairs": all(n >= MIN_PAIRS for n in counts.values()),
            "pairedCountsByHorizon": counts, "candidateMeanMae": candidate_mae, "eMeanMae": baseline_mae,
            "candidateMeanMape": candidate_mape, "eMeanMape": baseline_mape,
            "allEIssuedKeysCoveragePreserved": all_coverage, "garlicEIssuedKeysCoveragePreserved": garlic_coverage,
            "rule": "strictly lower equal-horizon garlic MAE and MAPE, all horizons >=30 pairs, and all E-issued keys retained"}
    result = {"schemaVersion": 1, "phase": f"preserved_improvement_{phase}_scored",
              "manifestSha256": candidate["manifestSha256"], "sourceSha256": source_hashes or {},
              "targetRange": [spec[1].isoformat(), spec[2].isoformat()], **scored}
    if confirmation_gate is not None:
        result["confirmationGate"] = confirmation_gate
    else:
        horizon_rows = {(x.get("product"), x.get("horizon"), x.get("method")): x
                        for x in scored["metricsByProductHorizonMethod"] if x.get("method")}
        garlic_candidate = [horizon_rows[("garlic", h, "candidate")] for h in HORIZONS]
        garlic_e = [horizon_rows[("garlic", h, "E")] for h in HORIZONS]
        garlic_counts = {str(h): horizon_rows[("garlic", h, "candidate")]["n"] for h in HORIZONS}
        candidate_mae = math.fsum(x["mae"] for x in garlic_candidate) / 10 if all(x["mae"] is not None for x in garlic_candidate) else None
        candidate_mape = math.fsum(x["mape"] for x in garlic_candidate) / 10 if all(x["mape"] is not None for x in garlic_candidate) else None
        e_mae = math.fsum(x["mae"] for x in garlic_e) / 10 if all(x["mae"] is not None for x in garlic_e) else None
        e_mape = math.fsum(x["mape"] for x in garlic_e) / 10 if all(x["mape"] is not None for x in garlic_e) else None
        coverage_preserved = all(x["coveragePreserved"] for x in scored["coverage"].values())
        retained = (all(n >= MIN_PAIRS for n in garlic_counts.values()) and coverage_preserved
                    and candidate_mae is not None and e_mae is not None and candidate_mae < e_mae
                    and candidate_mape is not None and e_mape is not None and candidate_mape < e_mape)
        result["retentionAssessment"] = {
            "retainedForNextResearchCandidate": retained,
            "decision": "retain_composition" if retained else "retain_E_and_preserve_composition_as_failed_candidate",
            "allTenGarlicHorizonsAtLeast30Pairs": all(n >= MIN_PAIRS for n in garlic_counts.values()),
            "garlicPairedCountsByHorizon": garlic_counts,
            "allEIssuedKeysCoveragePreserved": coverage_preserved,
            "garlicCandidateMeanMae": candidate_mae, "garlicEMeanMae": e_mae,
            "garlicCandidateMeanMape": candidate_mape, "garlicEMeanMape": e_mape,
            "candidateVsE": {"meanMaeDelta": scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mae"] - scored["fourCrop40CellEqualWeightMacro"]["E"]["mae"] if scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mae"] is not None and scored["fourCrop40CellEqualWeightMacro"]["E"]["mae"] is not None else None,
                              "meanMapeDelta": scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mape"] - scored["fourCrop40CellEqualWeightMacro"]["E"]["mape"] if scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mape"] is not None and scored["fourCrop40CellEqualWeightMacro"]["E"]["mape"] is not None else None},
            "candidateVsCausal50": {"meanMaeDelta": scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mae"] - scored["fourCrop40CellEqualWeightMacro"]["causal50"]["mae"] if scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mae"] is not None and scored["fourCrop40CellEqualWeightMacro"]["causal50"]["mae"] is not None else None,
                                     "meanMapeDelta": scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mape"] - scored["fourCrop40CellEqualWeightMacro"]["causal50"]["mape"] if scored["fourCrop40CellEqualWeightMacro"]["candidate"]["mape"] is not None and scored["fourCrop40CellEqualWeightMacro"]["causal50"]["mape"] is not None else None},
            "interpretation": "retrospective_candidate_only; no prior strict106 adoption verdict is changed"}
    result["reportSha256"] = sha(canonical(result).rstrip(b"\n"))
    return result


def _read_sources() -> tuple[dict[str, Any], ...]:
    paths = {"eManifest": E_MANIFEST, "preservedEvaluation": PRESERVED_EVALUATION,
             "prices": PRICES, "trendConfirmation": TREND_CONFIRMATION,
             "trendFinal": TREND_FINAL, "trendSelection": TREND_SELECTION}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    for name, expected in (("eManifest", EXPECTED["eManifest"]), ("preservedEvaluation", EXPECTED["preservedEvaluation"]),
                           ("prices", EXPECTED["prices"]), ("trendConfirmation", EXPECTED["trendConfirmation"]),
                           ("trendFinal", EXPECTED["trendFinal"]), ("trendSelection", EXPECTED["trendSelection"])):
        if sha(raw[name]) != expected:
            raise ValueError(f"pinned {name} SHA-256 mismatch")
    hashes = {name: sha(data) for name, data in raw.items()}
    hashes.update(_source_hashes())
    hashes["trendCorrectionBuilder"] = file_sha(Path(trend.__file__))
    if hashes["trendCorrectionBuilder"] != EXPECTED["trendCorrectionBuilder"]:
        raise ValueError("pinned trend-response correction helper SHA-256 mismatch")
    docs = {name: json.loads(data) for name, data in raw.items()}
    _check_source_documents(docs["eManifest"], docs["preservedEvaluation"], docs["trendConfirmation"],
                            docs["trendFinal"], docs["trendSelection"])
    return (docs["eManifest"], docs["preservedEvaluation"], docs["prices"], docs["trendConfirmation"],
            docs["trendFinal"], docs["trendSelection"], hashes)


def write_exclusive(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    write_exclusive(path, (json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def run(output_dir: Path = OUT_DIR) -> dict[str, Any]:
    e_doc, p_doc, prices, trend_confirmation, trend_final, selection, source_hashes = _read_sources()
    output_dir = Path(output_dir)
    confirmation = build_phase_manifest(e_doc, p_doc, "confirmation", trend_confirmation, trend_final, selection, source_hashes)
    _write_json(output_dir / "confirmation-manifest.json", confirmation)
    # Only after the immutable answer-free confirmation manifest exists do we
    # attach target values and decide whether the final phase may be created.
    confirmation_report = score_phase(confirmation, e_doc, p_doc, prices, "confirmation", trend_confirmation,
                                      trend_final, selection, source_hashes)
    _write_json(output_dir / "confirmation-report.json", confirmation_report)
    gate = confirmation_report["confirmationGate"]
    decision = {"schemaVersion": 1, "phase": "preserved_improvement_confirmation_decision",
                "confirmationManifestSha256": confirmation["manifestSha256"],
                "confirmationReportSha256": confirmation_report["reportSha256"],
                "passed": gate["passed"], "gate": gate,
                "decision": "generate_final" if gate["passed"] else "stop_before_final_generation"}
    decision["decisionSha256"] = sha(canonical(decision).rstrip(b"\n"))
    _write_json(output_dir / "confirmation-decision.json", decision)
    result = {"outputDir": str(output_dir), "confirmationGatePassed": gate["passed"], "finalScored": False}
    if not gate["passed"]:
        return result
    final = build_phase_manifest(e_doc, p_doc, "final", trend_confirmation, trend_final, selection, source_hashes)
    _write_json(output_dir / "final-manifest.json", final)
    final_report = score_phase(final, e_doc, p_doc, prices, "final", trend_confirmation,
                               trend_final, selection, source_hashes)
    _write_json(output_dir / "final-report.json", final_report)
    result["finalScored"] = True
    result["finalReportSha256"] = final_report["reportSha256"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
