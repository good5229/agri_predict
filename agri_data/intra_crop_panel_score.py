"""Scoring and fixed diagnostics for the CAND_B intra-crop panel.

Consumes an answer-free forecast manifest before joining pinned target prices.
It does not fit models or access data outside the supplied files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import equal_info_online_score as shared_stats
from . import full_history_return_eval as price_helper
from . import research_progress_diagnostic as shared_diagnostics

PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
BANDWIDTHS = (9, 28)
BLOCKS = (14, 28)
REPLICATES = 2000
SEED = 0
PERIODS = {
    "V": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
    "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27)),
}
ORIGIN_STARTS = {"V": dt.date(2026, 4, 1), "T": dt.date(2026, 7, 1)}
CONTRASTS = {
    "V": (("P_vs_B", "P", "E106_RHS"), ("P_vs_L", "P_raw", "L"), ("P_vs_Q", "P_raw", "Q")),
    "T": (("P_vs_B", "P", "E106_RHS"), ("P_vs_L", "P_raw", "L")),
}
LOSSES = ("ae", "ape")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _validate_prediction(value: Any, name: str) -> None:
    if not isinstance(value, dict) or value.get("status") not in ("issued", "not_issued"):
        raise ValueError(f"{name} prediction has invalid status")
    pred = _positive(value.get("prediction"))
    if (value["status"] == "issued") != (pred is not None):
        raise ValueError(f"{name} status/prediction mismatch")


def _validate_manifest(doc: dict[str, Any]) -> str:
    phase = doc.get("phase")
    if phase not in PERIODS or doc.get("targetValuesAttached") is not False:
        raise ValueError("expected V/T answer-free forecast manifest")
    unsigned = dict(doc)
    digest = unsigned.pop("manifestSha256", None)
    if digest != _sha(_canonical(unsigned)):
        raise ValueError("forecast manifest digest mismatch")
    rows = doc.get("forecasts")
    if not isinstance(rows, list):
        raise ValueError("forecast rows are missing")
    seen = set()
    start, end = PERIODS[phase]
    origin_start = ORIGIN_STARTS[phase]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed forecast row")
        product, origin, target, horizon = (row.get("product"), _date(row.get("origin")),
                                             _date(row.get("target")), row.get("horizon"))
        ident = (product, origin, target, horizon)
        if product not in PRODUCTS or origin is None or target is None or horizon not in HORIZONS:
            raise ValueError("invalid forecast identity")
        if ident in seen or origin != target - dt.timedelta(days=horizon):
            raise ValueError("duplicate forecast or origin/target mismatch")
        if row.get("fitMonth") != origin.strftime("%Y-%m"):
            raise ValueError("forecast fit month differs from its frozen monthly origin")
        if target < start or target > end or origin < origin_start:
            raise ValueError("forecast falls outside the frozen eligible period")
        seen.add(ident)
        baseline = row.get("B")
        if not isinstance(baseline, dict):
            raise ValueError("stored baseline predictions are missing")
        for key in ("E106_RHS", "E106"):
            _validate_prediction(baseline.get(key), f"B/{key}")
        if phase == "V":
            legacy = row.get("A_old")
            if not isinstance(legacy, dict):
                raise ValueError("saved A_old comparison predictions are missing")
            for key in ("CTRL_LGB", "CAND_A"):
                _validate_prediction(legacy.get(key), f"A_old/{key}")
        models = row.get("models")
        if not isinstance(models, dict):
            raise ValueError("candidate prediction map is missing")
        required = ("L", "P") if phase == "T" else ("L", "P", "Q")
        if any(name not in models for name in required):
            raise ValueError("candidate prediction grid is incomplete")
        for name in required:
            _validate_prediction(models[name], name)
            if name in ("L", "Q") and models[name]["status"] == "issued":
                log_ratio = models[name].get("logRatio")
                p0 = _positive(models[name].get("p0"))
                p0_date = _date(models[name].get("p0Date"))
                if (not isinstance(log_ratio, (int, float)) or isinstance(log_ratio, bool)
                        or not math.isfinite(log_ratio) or p0 is None or p0_date is None
                        or p0_date > origin):
                    raise ValueError(f"{name} issued forecast lacks a causal log-price decomposition")
        p = models["P"]
        raw_status = p.get("rawModelStatus")
        raw_prediction = _positive(p.get("rawModelPrediction"))
        if raw_status not in ("issued", "not_issued") or ((raw_status == "issued") != (raw_prediction is not None)):
            raise ValueError("P raw model status/prediction mismatch")
        used = p.get("modelUsed")
        if not isinstance(used, bool) or used != (raw_status == "issued"):
            raise ValueError("P modelUsed does not match raw model status")
        if used:
            raw_log_ratio = p.get("rawModelLogRatio")
            p0 = _positive(p.get("p0"))
            p0_date = _date(p.get("p0Date"))
            if (not isinstance(raw_log_ratio, (int, float)) or isinstance(raw_log_ratio, bool)
                    or not math.isfinite(raw_log_ratio) or p0 is None or p0_date is None or p0_date > origin):
                raise ValueError("P issued model lacks a causal log-price decomposition")
            if p["status"] != "issued" or not math.isclose(_positive(p["prediction"]), raw_prediction, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("P issued policy differs from direct model output")
        elif baseline["E106_RHS"]["status"] == "issued":
            if p["status"] != "issued" or not math.isclose(_positive(p["prediction"]), _positive(baseline["E106_RHS"]["prediction"]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("P fallback does not reproduce stored E106-RHS")
            if not p.get("fallbackReason"):
                raise ValueError("P fallback reason is missing")
        elif p["status"] != "not_issued" or not p.get("withheldReason"):
            raise ValueError("P must preserve withheld state when B is unavailable")
        if any(key in row for key in ("actual", "actualPrice", "targetValue")):
            raise ValueError("answer value found in forecast-only manifest")
    # Eligibility uses the phase's frozen fit-origin boundary, which differs from
    # T's later first target date.
    expected = {
        (product, target - dt.timedelta(days=horizon), target, horizon)
        for product in PRODUCTS for horizon in HORIZONS
        for offset in range(max(0, (origin_start + dt.timedelta(days=horizon)-start).days),
                            (end-start).days + 1)
        for target in (start + dt.timedelta(days=offset),)
    }
    if seen != expected:
        raise ValueError(f"forecast grid mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}")
    return phase


def _loss(actual: float, prediction: float, metric: str) -> float:
    absolute = abs(prediction - actual)
    return absolute if metric == "ae" else 100.0 * absolute / actual


def _model_prediction(row: dict[str, Any], model: str) -> float | None:
    if model == "P_raw":
        return _positive(row["models"]["P"].get("rawModelPrediction"))
    if model == "E106_RHS":
        return _positive(row["B"][model].get("prediction"))
    return _positive(row["models"][model].get("prediction"))


def _pair_rows(rows: list[dict[str, Any]], actuals: dict[tuple[str, dt.date], float],
               left: str, right: str) -> list[dict[str, Any]]:
    paired = []
    for row in rows:
        target = _date(row["target"])
        actual = actuals.get((row["product"], target))
        a, b = _model_prediction(row, left), _model_prediction(row, right)
        if actual is not None and a is not None and b is not None:
            paired.append({"product": row["product"], "horizon": row["horizon"],
                           "target": target.isoformat(), "actual": actual, "a": a, "b": b})
    return paired


def _cell_metrics(rows: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["product"], row["horizon"])].append(row)
    result = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            values = by_cell.get((product, horizon), [])
            if values:
                mae_a = math.fsum(abs(v["a"]-v["actual"]) for v in values) / len(values)
                mae_b = math.fsum(abs(v["b"]-v["actual"]) for v in values) / len(values)
                mape_a = 100.0 * math.fsum(abs(v["a"]-v["actual"])/v["actual"] for v in values) / len(values)
                mape_b = 100.0 * math.fsum(abs(v["b"]-v["actual"])/v["actual"] for v in values) / len(values)
            else:
                mae_a = mae_b = mape_a = mape_b = None
            result.append({"product": product, "horizon": horizon, "comparison": comparison, "n": len(values),
                           "aMae": mae_a, "bMae": mae_b, "aMape": mape_a, "bMape": mape_b})
    return result


def _macro(cells: list[dict[str, Any]]) -> dict[str, Any]:
    complete = len(cells) == 40 and all(row["n"] > 0 for row in cells)
    out: dict[str, Any] = {"cells": len(cells), "supportedCells": sum(row["n"] > 0 for row in cells),
                           "pairedRows": sum(row["n"] for row in cells), "status": "complete" if complete else "incomplete"}
    for key in ("aMae", "bMae", "aMape", "bMape"):
        out[key] = math.fsum(row[key] for row in cells) / 40 if complete else None
    if complete:
        out["maeDifference"] = out["aMae"] - out["bMae"]
        out["mapeDifferencePctPoints"] = out["aMape"] - out["bMape"]
    return out


def _dm_family(rows: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["product"], row["horizon"])].append(row)
    result = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            values = by_cell.get((product, horizon), [])
            for loss in LOSSES:
                if loss == "ape":
                    diffs = {dt.date.fromisoformat(row["target"]):
                             100.0*(abs(row["a"]-row["actual"])-abs(row["b"]-row["actual"]))/row["actual"]
                             for row in values}
                else:
                    diffs = {dt.date.fromisoformat(row["target"]):
                             abs(row["a"]-row["actual"])-abs(row["b"]-row["actual"])
                             for row in values}
                for bandwidth in BANDWIDTHS:
                    stat = shared_stats._dm(diffs, bandwidth)
                    result.append({"product": product, "horizon": horizon, "comparison": comparison, "loss": loss,
                                   "bandwidth": bandwidth, "status": stat["status"], "reason": stat.get("reason"),
                                   "n": stat["n"], "statistic": stat["statistic"],
                                   "meanLossDifference": stat["meanLossDifference"], "pValue": stat["pValue"]})
    return result


def _price_contributions(rows: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["product"], row["horizon"])].append(row)
    groups: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"maeGain": 0.0, "maeLoss": 0.0, "mapeGain": 0.0, "mapeLoss": 0.0,
                 "maeNet": 0.0, "mapeNet": 0.0, "n": 0})
    for (product, horizon), values in cells.items():
        denom = 40 * len(values)
        for row in values:
            group = groups[(product, horizon, row["target"][:7])]
            for metric, key in (("ae", "mae"), ("ape", "mape")):
                delta = (_loss(row["actual"], row["a"], metric) -
                         _loss(row["actual"], row["b"], metric)) / denom
                group[key+"Net"] += delta
                group[key+("Loss" if delta > 0 else "Gain")] += abs(delta)
            group["n"] += 1
    items = [{"product": key[0], "horizon": key[1], "month": key[2], **value}
             for key, value in sorted(groups.items())]
    cells_metrics = _cell_metrics(rows, comparison)
    macro = _macro(cells_metrics)
    expected = {"mae": macro.get("maeDifference"), "mape": macro.get("mapeDifferencePctPoints")}
    sums = {metric: math.fsum(row[metric+"Net"] for row in items) for metric in ("mae", "mape")}
    checks = {metric: (expected[metric] is not None and
                       math.isclose(sums[metric], expected[metric], rel_tol=1e-10, abs_tol=1e-10))
              for metric in sums}
    if any(expected[key] is not None and not checks[key] for key in checks):
        raise ValueError("product/horizon/month price contributions do not reconcile to 40-cell macro")
    return {"groups": items, "reconciledMacroDifference": expected, "groupNetSums": sums,
            "accountingVerified": checks}


def _log_decomposition(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    """Compute SSE(right)-SSE(left) as delta^2 - 2*e_left*delta."""
    sums = {"n": 0, "deltaSquared": 0.0, "minusTwoResidualDelta": 0.0, "squaredErrorDifference": 0.0}
    for row in rows:
        models = row["models"]
        a_obj = models["P"] if left == "P_raw" else models[left]
        b_obj = models["P"] if right == "P_raw" else models[right]
        a_log = a_obj.get("rawModelLogRatio") if left == "P_raw" else a_obj.get("logRatio")
        b_log = b_obj.get("rawModelLogRatio") if right == "P_raw" else b_obj.get("logRatio")
        p0, actual = _positive(a_obj.get("p0")), row.get("actual")
        try:
            a_log, b_log = float(a_log), float(b_log)
        except (TypeError, ValueError, OverflowError):
            continue
        if p0 is None or actual is None or not math.isfinite(a_log) or not math.isfinite(b_log):
            continue
        if not math.isclose(p0, _positive(b_obj.get("p0")) or float("nan"), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("log decomposition requires the same representative base price")
        actual_log_ratio = math.log(actual / p0)
        e_left, delta = actual_log_ratio - a_log, b_log - a_log
        part_a, part_b = delta*delta, -2.0*e_left*delta
        diff = (actual_log_ratio-b_log)**2 - (actual_log_ratio-a_log)**2
        sums["n"] += 1
        sums["deltaSquared"] += part_a
        sums["minusTwoResidualDelta"] += part_b
        sums["squaredErrorDifference"] += diff
    identity = sums["deltaSquared"] + sums["minusTwoResidualDelta"]
    sums["identityResidual"] = identity - sums["squaredErrorDifference"]
    sums["identityVerified"] = math.isclose(identity, sums["squaredErrorDifference"], rel_tol=1e-9, abs_tol=1e-10)
    if not sums["identityVerified"]:
        raise ValueError("log-square change decomposition failed its identity")
    return {"leftModel": left, "rightModel": right,
            "differenceDefinition": "right minus left squared log error",
            **sums}


def _train_log_decomposition(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for fit in fits:
        if fit.get("status") in ("not_run", "not_applicable"):
            continue
        for contrast, payload in (fit.get("representativeTrainLogDecomposition") or {}).items():
            if payload is None:
                raise ValueError("fitted model has a null training log decomposition")
            if not isinstance(payload, dict):
                raise ValueError("malformed training log decomposition")
            d2, cross, diff = (payload.get(key) for key in
                               ("deltaSquaredSum", "crossTermSum", "squaredErrorDifferenceSum"))
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (d2, cross, diff)):
                raise ValueError("non-finite training log decomposition")
            n = payload.get("n")
            identity_error = payload.get("identityMaxAbsError")
            if (not isinstance(n, int) or isinstance(n, bool) or n < 1
                    or not isinstance(identity_error, (int, float)) or not math.isfinite(identity_error)
                    or identity_error < 0 or identity_error > 1e-8):
                raise ValueError("training log decomposition lacks common-row support or identity evidence")
            residual = float(d2) + float(cross) - float(diff)
            if not math.isclose(residual, 0.0, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("training log-square change decomposition failed its identity")
            output.append({"product": fit.get("product"), "horizon": fit.get("horizon"),
                           "fitMonth": fit.get("fitMonth"), "contrast": contrast, **payload,
                           "identityResidual": residual, "identityVerified": True})
    return output


def _origin_weight_integrity(fits: list[dict[str, Any]]) -> bool:
    for fit in fits:
        if fit.get("status") in ("not_run", "not_applicable"):
            continue
        variant = fit.get("variant")
        rows = fit.get("originWeightDiagnostics")
        if variant not in ("L", "P", "Q") or not isinstance(rows, list) or not rows:
            return False
        seen_origins = set()
        for row in rows:
            origin, k = row.get("origin"), row.get("K")
            if not isinstance(origin, str) or origin in seen_origins or not isinstance(k, int) or isinstance(k, bool) or k < 0:
                return False
            seen_origins.add(origin)
            values = [row.get(key) for key in ("totalWeight", "representativeWeight", "auxiliaryWeight")]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
                return False
            total, representative, auxiliary = map(float, values)
            expected_rep = 0.5 if variant in ("P", "Q") and k > 0 else 1.0
            expected_aux = 0.5 if variant in ("P", "Q") and k > 0 else 0.0
            if not all(math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10)
                       for observed, expected in ((total, 1.0), (representative, expected_rep),
                                                  (auxiliary, expected_aux))):
                return False
    return True


def _representative_key_identity(fits: list[dict[str, Any]], phase: str) -> bool:
    variants = ("L", "P", "Q") if phase == "V" else ("L", "P")
    digests: dict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    for fit in fits:
        if fit.get("variant") not in variants:
            continue
        if fit.get("status") in ("not_run", "not_applicable"):
            continue
        key = (fit.get("product"), fit.get("horizon"), fit.get("fitMonth"))
        digest = fit.get("representativeKeyDigest")
        if not isinstance(digest, str) or not digest:
            return False
        digests[key][fit["variant"]] = digest
    return bool(digests) and all(set(values) == set(variants) and len(set(values.values())) == 1
                                 for values in digests.values())


def _fit_diagnostic_integrity(fits: list[dict[str, Any]], phase: str) -> tuple[bool, bool, bool]:
    required_train = ("n", "logMse", "priceMAE", "priceMAPE")
    required_variants = ("L", "P", "Q") if phase == "V" else ("L", "P")
    diagnostics_ok = True
    permutation_ok = True
    decomposition_ok = True
    for fit in fits:
        variant, status = fit.get("variant"), fit.get("status")
        if status in ("not_run", "not_applicable"):
            if not (phase == "T" and variant == "Q"):
                diagnostics_ok = False
            continue
        if variant not in required_variants or status not in ("fit", "reused", "complete", "aliased_no_support"):
            diagnostics_ok = False
            continue
        train = fit.get("representativeTrainDiagnostics")
        if (not isinstance(train, dict) or any(key not in train for key in required_train)
                or not isinstance(train.get("n"), int) or isinstance(train.get("n"), bool) or train["n"] < 1
                or any(not isinstance(train.get(key), (int, float)) or not math.isfinite(train[key])
                       or train[key] < 0 for key in required_train[1:])):
            diagnostics_ok = False
        support = fit.get("support")
        if not isinstance(support, list) or not support:
            diagnostics_ok = False
        else:
            rep = [row for row in support if isinstance(row, dict) and row.get("seriesId") == 0]
            if not rep or any(not isinstance(row.get("rows"), int) or row["rows"] < 0
                              or not isinstance(row.get("distinctTargets"), int) or row["distinctTargets"] < 0
                              or not isinstance(row.get("weightSum"), (int, float))
                              or not math.isfinite(row["weightSum"]) or row["weightSum"] < 0 for row in support):
                diagnostics_ok = False
        if phase == "V" and variant == "Q":
            permutation = fit.get("permutationDiagnostics")
            if not isinstance(permutation, dict):
                permutation_ok = False
            else:
                counts = [permutation.get(key) for key in ("blockCount", "unchangedBlocks", "changedRows", "auxiliaryRows")]
                share = permutation.get("changedWeightedShare")
                changed_weight, auxiliary_weight = (permutation.get("changedAuxiliaryWeight"),
                                                    permutation.get("auxiliaryWeight"))
                if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
                        or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 <= share <= 1
                        or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                               for value in (changed_weight, auxiliary_weight))
                        or not math.isclose(share, changed_weight/auxiliary_weight if auxiliary_weight else 0.0,
                                            rel_tol=1e-10, abs_tol=1e-10)
                        or permutation["changedRows"] > permutation["auxiliaryRows"]):
                    permutation_ok = False
        if variant in ("P", "Q"):
            decomposition = fit.get("representativeTrainLogDecomposition")
            expected = ("P-L",) if variant == "P" else ("P-Q",)
            if phase == "T" and variant == "Q":
                expected = ()
            if not isinstance(decomposition, dict) or any(key not in decomposition for key in expected):
                if expected:
                    decomposition_ok = False
    return diagnostics_ok, permutation_ok, decomposition_ok


def _fit_diagnostics(doc: dict[str, Any], phase: str,
                     external_checks: dict[str, bool] | None = None) -> dict[str, Any]:
    fits = doc.get("fits")
    if not isinstance(fits, list):
        raise ValueError("fit diagnostics are missing")
    technical = doc.get("technicalCompletion")
    if not isinstance(technical, dict):
        raise ValueError("technical completion by product is missing")
    complete = all(isinstance(technical.get(product), dict) and technical[product].get("status") == "complete"
                   for product in PRODUCTS)
    used_fits = [fit for fit in fits if fit.get("status") not in ("not_run", "not_applicable")]
    time_ok = all(isinstance(fit.get("timeIntegrity"), dict)
                  and fit["timeIntegrity"].get("allTrainingTargetsBeforeFitOrigin") is True
                  and fit["timeIntegrity"].get("allFeatureDatesAtOrBeforeCutoff") is True for fit in used_fits)
    q_fits = [fit for fit in fits if phase == "V" and fit.get("variant") == "Q"]
    q_ok = all(isinstance(fit.get("pqIdentity"), dict)
               and all(fit["pqIdentity"].get(key) is True for key in
                       ("sameRepresentativeKeyDigest", "sameInputWeightDigest", "sameBinningIdentity"))
               for fit in q_fits)
    origin_weights_ok = _origin_weight_integrity(fits)
    representative_keys_ok = _representative_key_identity(fits, phase)
    train_diag_ok, permutation_ok, train_decomposition_present = _fit_diagnostic_integrity(fits, phase)
    # Manifest claims are diagnostic only. The four checks below are populated
    # exclusively by independently opening and hashing the referenced artifacts.
    required_integrity = ("sourceCacheHashesVerified", "baselineKeysAndActualsVerified",
                          "forecastReplayVerified", "resolvedConfigVerified")
    integrity_checks = {key: external_checks is not None and external_checks.get(key) is True
                        for key in required_integrity}
    supports, origin_weights, representative_train, permutations = [], [], [], []
    for fit in fits:
        if fit.get("status") == "failed":
            complete = False
        meta = {key: fit.get(key) for key in ("product", "horizon", "fitMonth", "variant")}
        if isinstance(fit.get("representativeTrainDiagnostics"), dict):
            representative_train.append({**meta, **fit["representativeTrainDiagnostics"]})
        for support in fit.get("support", fit.get("seriesSupport", [])) or []:
            supports.append({**meta, **support})
        for check in fit.get("originWeightDiagnostics", []) or []:
            origin_weights.append({**meta, **check})
        if phase == "V" and fit.get("variant") == "Q":
            permutations.append({**meta, **(fit.get("permutationDiagnostics") or {})})
    return {"technicalCompletion": technical, "allProductsComplete": complete, "timeIntegrityVerified": time_ok,
            "pqIdentityVerified": q_ok, "supportBySeries": supports,
            "originWeightDiagnostics": origin_weights,
            "informationIntegrityChecks": integrity_checks,
            "informationIntegrityVerified": all(integrity_checks.values()),
            "originWeightIntegrityVerified": origin_weights_ok,
            "representativeTrainKeysIdentical": representative_keys_ok,
            "representativeTrainDiagnosticsComplete": train_diag_ok,
            "permutationDiagnosticsComplete": permutation_ok,
            "trainLogDecompositionComplete": train_decomposition_present,
            "representativeTrainDiagnostics": representative_train,
            "permutationDiagnostics": permutations,
            "representativeTrainLogDecomposition": _train_log_decomposition(fits)}


def _expected_phase_actual_coverage(phase: str, forecasts: list[dict[str, Any]],
                                    actuals: dict[tuple[str, dt.date], float]) -> dict[str, Any]:
    phase_start, phase_end = PERIODS[phase]
    origin_floor = ORIGIN_STARTS[phase]
    first_legal = max(phase_start, origin_floor + dt.timedelta(days=1))
    expected_targets = sorted((product, (first_legal + dt.timedelta(days=offset)).isoformat())
                              for product in PRODUCTS
                              for offset in range((phase_end-first_legal).days + 1))
    forecast_targets = sorted({(row["product"], row["target"]) for row in forecasts})
    phase_positive = [(product, day.isoformat()) for (product, day), value in actuals.items()
                      if phase_start <= day <= phase_end and _positive(value) is not None]
    covered_positive = [(product, day) for product, day in phase_positive if day >= first_legal.isoformat()]
    excluded_positive = sorted((product, day) for product, day in phase_positive
                               if day < first_legal.isoformat())
    return {
        "status": "pass", "phaseStart": phase_start.isoformat(), "phaseEnd": phase_end.isoformat(),
        "originFloor": origin_floor.isoformat(), "firstLegalTarget": first_legal.isoformat(),
        "expectedLegalTargetPairCount": len(expected_targets), "forecastTargetPairCount": len(forecast_targets),
        "positiveActualKeyCountInPhase": len(phase_positive),
        "coveredPositiveActualKeyCount": len(covered_positive),
        "excludedPositiveActualKeyCount": len(excluded_positive),
        "excludedPositiveActualKeys": [list(key) for key in excluded_positive],
        "legalTargetGridSha256": _sha(_canonical(expected_targets)),
        "_gridMatches": forecast_targets == expected_targets,
    }


def _verify_external_artifacts(manifest: dict[str, Any], prices: dict[str, Any], *,
                               manifest_path: Path, prices_path: Path) -> dict[str, bool]:
    """Verify the phase evidence chain using the pinned project artifacts."""
    from . import chronos2_market_context as rhs
    from . import external_reference_v1 as source
    from . import intra_crop_panel as producer

    manifest_path, prices_path = Path(manifest_path), Path(prices_path)
    phase = manifest["phase"]
    output_dir = manifest_path.parent.parent
    if manifest_path.read_bytes() != _canonical(manifest):
        raise ValueError("manifest file bytes do not match the supplied manifest")
    if _sha(prices_path.read_bytes()) != price_helper.PRICES_SHA256:
        raise ValueError("pinned prices file hash mismatch")
    if json.loads(prices_path.read_text(encoding="utf-8")) != prices:
        raise ValueError("supplied prices differ from the pinned snapshot file")
    snapshot_sha = _sha(prices_path.read_bytes())
    if manifest.get("targetSnapshotSha256") != snapshot_sha:
        raise ValueError("manifest target snapshot identity mismatch")
    actuals = source._actuals_from_snapshot(prices)
    positive_keys = sorted((product, day.isoformat()) for (product, day), value in actuals.items()
                           if _positive(value) is not None)
    if (manifest.get("positiveActualKeyCount") != len(positive_keys)
            or manifest.get("positiveActualKeysSha256") != _sha(_canonical(positive_keys))):
        raise ValueError("manifest positive target-key summary differs from pinned snapshot")
    expected_coverage = _expected_phase_actual_coverage(phase, manifest["forecasts"], actuals)
    coverage = manifest.get("phaseActualCoverage")
    if (not expected_coverage.pop("_gridMatches") or not isinstance(coverage, dict)
            or any(_canonical(coverage.get(key)) != _canonical(value)
                   for key, value in expected_coverage.items())):
        raise ValueError("phase actual-coverage evidence differs from the legal target grid or pinned snapshot")

    baseline_path = rhs.BASELINE_PATHS[phase]
    baseline_sha = _sha(baseline_path.read_bytes())
    if baseline_sha != rhs.BASELINE_SHA256[phase] or manifest.get("baselineManifestSha256") != baseline_sha:
        raise ValueError("manifest baseline is not the pinned phase baseline")
    baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = source._baseline_index(baseline_doc, phase)
    old_predictions = {}
    if phase == "V":
        old_manifest_path = producer.SOURCE_OUT / "V/forecast-manifest.json"
        if _sha(old_manifest_path.read_bytes()) != producer.SOURCE_V_MANIFEST_SHA256:
            raise ValueError("saved A_old V manifest is not the pinned artifact")
        old_predictions = producer._a_old_index()
    forecasts = manifest["forecasts"]
    if len(forecasts) != len(baseline):
        raise ValueError("forecast rows do not cover the pinned baseline grid")
    for row in forecasts:
        key = (row["product"], _date(row["target"]), row["horizon"])
        expected = baseline.get(key)
        if expected is None or expected["origin"] != _date(row["origin"]):
            raise ValueError("forecast row key differs from pinned baseline")
        for name, status_key in (("E106_RHS", "rhsStatus"), ("E106", "e106Status")):
            wanted = producer._baseline_prediction(expected[name], expected[status_key])
            stored = row["B"][name]
            if (stored.get("status"), stored.get("prediction"), stored.get("sourceStatus")) != (
                    wanted.get("status"), wanted.get("prediction"), wanted.get("sourceStatus")):
                raise ValueError(f"stored {name} baseline differs from pinned source")
        if phase == "V" and row.get("A_old") != old_predictions.get(key):
            raise ValueError("stored A_old predictions differ from pinned saved V manifest")

    pre_rel = manifest.get("preflightArtifactPath")
    if pre_rel != "preflight.json":
        raise ValueError("unexpected preflight artifact path")
    pre_path = output_dir / pre_rel
    preflight = json.loads(pre_path.read_text(encoding="utf-8"))
    pre_hash = preflight.pop("preflightSha256", None)
    if pre_hash != manifest.get("preflightSha256") or pre_hash != _sha(_canonical(preflight)):
        raise ValueError("preflight self-hash or manifest binding mismatch")
    cache_path = producer.SOURCE_CACHE
    cache_sha = _sha(cache_path.read_bytes())
    cache_doc = json.loads(cache_path.read_text(encoding="utf-8"))
    if (cache_sha != manifest.get("sourceCacheFileSha256")
            or cache_sha != preflight.get("sourceCacheFileSha256")
            or cache_doc.get("cacheSha256") != producer.SOURCE_CACHE_INTERNAL_SHA256
            or cache_doc.get("cacheSha256") != manifest.get("sourceCacheSha256")
            or cache_doc.get("cacheSha256") != preflight.get("sourceCacheInternalSha256")):
        raise ValueError("source-cache identity mismatch")
    if preflight.get("status") != "preflight_complete" or preflight.get("targetSnapshotSha256") != snapshot_sha:
        raise ValueError("preflight does not bind the pinned target snapshot")
    if preflight.get("baselineHashes") != rhs.BASELINE_SHA256:
        raise ValueError("preflight baseline identities differ from pinned baselines")
    expected_key_counts = {name: len(source._baseline_index(
        json.loads(path.read_text(encoding="utf-8")), name)) for name, path in rhs.BASELINE_PATHS.items()}
    if (preflight.get("baselineKeysByPhase") != expected_key_counts
            or preflight.get("targetSnapshotVerified") is not True
            or preflight.get("baselineKeysVerified") is not True
            or preflight.get("representativeKeyInheritanceVerified") is not True):
        raise ValueError("preflight target, baseline-key, or A_old inheritance check failed")
    config_sha = preflight.get("resolvedConfigSha256")
    config = producer._load_resolved_config(output_dir)
    execution = producer._execution_identity(config, cache_sha)
    execution_sha = _sha(_canonical(execution))
    if (config_sha != manifest.get("resolvedConfigSha256") or config_sha != config.get("configSha256")
            or preflight.get("resolvedConfigVerified") is not True
            or preflight.get("sourceCacheHashesVerified") is not True
            or manifest.get("executionIdentity") != execution
            or manifest.get("executionIdentitySha256") != execution_sha
            or preflight.get("executionIdentity") != execution):
        raise ValueError("resolved configuration or source-cache preflight check failed")

    replay_rel = manifest.get("forecastReplayAuditPath")
    if replay_rel != f"{phase}/replay-verification.json":
        raise ValueError("unexpected replay audit path")
    replay_path = output_dir / replay_rel
    replay_raw = replay_path.read_bytes()
    replay = json.loads(replay_raw)
    replay_hash = replay.get("replayVerificationSha256")
    replay_unsigned = dict(replay)
    replay_unsigned.pop("replayVerificationSha256", None)
    if (_sha(replay_raw) != manifest.get("forecastReplayAuditSha256")
            or replay_hash != manifest.get("replayVerificationSha256")
            or replay_hash != _sha(_canonical(replay_unsigned))):
        raise ValueError("replay audit hash binding failed")
    if (replay.get("phase") != phase or replay.get("status") != "pass"
            or replay.get("forecastRowsSha256") != _sha(_canonical(forecasts))
            or replay.get("sourceCacheFileSha256") != cache_sha
            or replay.get("baselineManifestSha256") != baseline_sha
            or replay.get("resolvedConfigSha256") != config_sha
            or replay.get("executionIdentitySha256") != manifest.get("executionIdentitySha256")
            or replay.get("preReplayManifestSha256") is None
            or replay.get("baselineCopiedValuesVerified") is not True
            or replay.get("AOldCopiedValuesVerified") is not True):
        raise ValueError("replay audit does not verify this forecast/evidence set")
    evidence = replay.get("checkpointEvidence")
    if not isinstance(evidence, list):
        raise ValueError("replay checkpoint evidence is missing")
    evidence_by_key = {(item.get("product"), item.get("horizon"), item.get("fitMonth"), item.get("variant")): item
                       for item in evidence if isinstance(item, dict)}
    if len(evidence_by_key) != len(evidence):
        raise ValueError("duplicate or malformed checkpoint evidence")
    for fit in manifest.get("fits", []):
        key = (fit.get("product"), fit.get("horizon"), fit.get("fitMonth"), fit.get("variant"))
        item = evidence_by_key.get(key)
        if item is None or item.get("status") != fit.get("status"):
            raise ValueError("replay checkpoint evidence differs from fit manifest")
        if item.get("modelFileSha256") != fit.get("modelFileSha256"):
            raise ValueError("replay checkpoint file hash differs from fit manifest")
        if item.get("checkpointIdentitySha256") != fit.get("checkpointIdentitySha256"):
            raise ValueError("replay checkpoint identity differs from fit manifest")
    if len(evidence_by_key) != len(manifest.get("fits", [])):
        raise ValueError("replay checkpoint set differs from fit manifest")
    return {"sourceCacheHashesVerified": True, "baselineKeysAndActualsVerified": True,
            "forecastReplayVerified": True, "resolvedConfigVerified": True}


def score(manifest: dict[str, Any], prices: dict[str, Any], *, manifest_path: Path | None = None,
          prices_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase = _validate_manifest(manifest)
    if (manifest_path is None) != (prices_path is None):
        raise ValueError("manifest_path and prices_path must be supplied together")
    external_checks = (_verify_external_artifacts(manifest, prices, manifest_path=manifest_path,
                                                  prices_path=prices_path)
                       if manifest_path is not None else None)
    actual_series = price_helper._series(prices)
    actuals = {(product, day): value for product, series in actual_series.items()
               for day, value in series.items()}
    start, end = PERIODS[phase]
    origin_start = ORIGIN_STARTS[phase]
    rows = [row for row in manifest["forecasts"]
            if start <= _date(row["target"]) <= end and _date(row["origin"]) >= origin_start]
    diagnostics = _fit_diagnostics(manifest, phase, external_checks)
    comparisons, all_dm, bootstraps, contributions, eval_log = {}, [], {}, {}, {}
    for label, left, right in CONTRASTS[phase]:
        paired = _pair_rows(rows, actuals, left, right)
        cells = _cell_metrics(paired, label)
        macro = _macro(cells)
        comparisons[label] = {"cells": cells, "macro": macro}
        all_dm.extend(_dm_family(paired, label))
        contributions[label] = _price_contributions(paired, label)
        if label != "P_vs_B":
            log_rows = [{**row, "actual": actuals[(row["product"], _date(row["target"]))]}
                        for row in rows if (row["product"], _date(row["target"])) in actuals
                        and _model_prediction(row, left) is not None and _model_prediction(row, right) is not None]
            eval_log[label] = _log_decomposition(log_rows, right, left)
        else:
            eval_log[label] = None
        bootstrap_rows = [{**row, "target": row["target"]} for row in paired]
        if macro["status"] != "complete":
            bootstraps[label] = [
                {"metric": metric, "blockDays": block, "status": "unsupported_missing_frozen_cell",
                 "reason": "paired comparison does not cover all 40 fixed product-horizon cells",
                 "requested": 2000, "valid": 0, "invalid": 2000, "seed": 0, "interval95": None}
                for metric in ("ae", "ape") for block in (14, 28)]
        else:
            bootstraps[label] = [
                {"metric": metric, "status": "completed", **shared_diagnostics.circular_block_bootstrap(
                    bootstrap_rows, a="a", b="b", metric=metric, block=block, replicates=2000,
                    seed=0, calendar_start=start, calendar_end=end)}
                for metric in ("ae", "ape") for block in (14, 28)]
    legacy_comparisons = {}
    if phase == "V":
        for name in ("CTRL_LGB", "CAND_A"):
            paired = []
            for row in rows:
                actual = actuals.get((row["product"], _date(row["target"])))
                candidate = _model_prediction(row, "P")
                legacy = _positive(row["A_old"][name].get("prediction"))
                if actual is not None and candidate is not None and legacy is not None:
                    paired.append({"product": row["product"], "horizon": row["horizon"],
                                   "target": row["target"], "actual": actual, "a": candidate, "b": legacy})
            cells = _cell_metrics(paired, f"P_vs_{name}")
            legacy_comparisons[f"P_vs_{name}"] = {
                "cells": cells, "macro": _macro(cells),
                "interpretation": "descriptive saved A_old comparison; outside fixed inference family"}
    model_only_pairs = _pair_rows(rows, actuals, "P_raw", "E106_RHS")
    model_only_cells = _cell_metrics(model_only_pairs, "P_modelOnly_vs_B")
    model_only = {"cells": model_only_cells, "macro": _macro(model_only_cells),
                  "interpretation": "direct model-issued P rows vs B; descriptive subset, outside fixed inference family"}
    for bandwidth in BANDWIDTHS:
        family = [row for row in all_dm if row["bandwidth"] == bandwidth]
        expected = 240 if phase == "V" else 160
        if len(family) != expected:
            raise ValueError(f"{phase} frozen HAC{bandwidth} family size mismatch")
    # The monthly expanding folds are fixed for all products and horizons.
    expected_folds = {(product, horizon, month)
                      for product in PRODUCTS for horizon in HORIZONS
                      for month in (("2026-04", "2026-05", "2026-06") if phase == "V"
                                    else ("2026-07", "2026-08", "2026-09"))}
    fit_fold_variants = {(fit.get("product"), fit.get("horizon"), fit.get("fitMonth"), fit.get("variant"))
                         for fit in manifest["fits"]}
    observed_folds = {(product, horizon, month)
                      for product, horizon, month, _ in fit_fold_variants}
    required_variants = {"L", "P", "Q"} if phase == "V" else {"L", "P"}
    folds_complete = observed_folds == expected_folds and all(
        (product, horizon, month, variant) in fit_fold_variants
        for product, horizon, month in expected_folds for variant in required_variants)
    integrity_ok = (diagnostics["allProductsComplete"] and diagnostics["timeIntegrityVerified"]
                    and diagnostics["pqIdentityVerified"] and diagnostics["informationIntegrityVerified"]
                    and diagnostics["originWeightIntegrityVerified"]
                    and diagnostics["representativeTrainKeysIdentical"]
                    and diagnostics["representativeTrainDiagnosticsComplete"]
                    and diagnostics["permutationDiagnosticsComplete"]
                    and diagnostics["trainLogDecompositionComplete"] and folds_complete)
    # Holm correction spans all fixed contrasts for each bandwidth.
    for bandwidth in BANDWIDTHS:
        family = [row for row in all_dm if row["bandwidth"] == bandwidth]
        shared_stats._holm(family)
        for row in family:
            row["holmAdjustedPValue"] = row.pop("holmPValue")
    dm_conclusions = []
    for label, _, _ in CONTRASTS[phase]:
        for bandwidth in BANDWIDTHS:
            family = [row for row in all_dm if row["comparison"] == label and row["bandwidth"] == bandwidth]
            improved = sum(row.get("holmSignificant") is True and row["meanLossDifference"] is not None
                           and row["meanLossDifference"] < 0 for row in family)
            worsened = sum(row.get("holmSignificant") is True and row["meanLossDifference"] is not None
                           and row["meanLossDifference"] > 0 for row in family)
            unavailable = sum(row["status"] != "tested" for row in family)
            if improved and worsened:
                conclusion = "mixed_significant_directions"
            elif improved:
                conclusion = "significant_improvement_detected"
            elif worsened:
                conclusion = "significant_worsening_detected"
            else:
                conclusion = "inconclusive_no_family_significant_direction"
            dm_conclusions.append({"comparison": label, "bandwidth": bandwidth,
                                   "familyTests": len(family), "inferenceUnavailable": unavailable,
                                   "significantImprovement": improved, "significantWorsening": worsened,
                                   "testedWithoutSignificantDirection": len(family)-unavailable-improved-worsened,
                                   "conclusion": conclusion})
    coverage = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            group = [row for row in rows if row["product"] == product and row["horizon"] == horizon]
            actual_group = [row for row in group if (product, _date(row["target"])) in actuals]
            bissued = [row for row in group if row["B"]["E106_RHS"]["status"] == "issued"]
            coverage[f"{product}/{horizon}"] = {
                "forecastRows": len(group), "positiveActualRows": len(actual_group), "baselineIssued": len(bissued),
                "candidateModelIssued": sum(row["models"]["P"]["modelUsed"] for row in group),
                "candidateFallback": sum(not row["models"]["P"]["modelUsed"] and row["models"]["P"]["status"] == "issued" for row in group),
                "candidateWithheld": sum(row["models"]["P"]["status"] != "issued" for row in group),
                "baselineCoverageLossKeys": [row["target"] for row in bissued if row["models"]["P"]["status"] != "issued"],
                "LModelIssued": sum(row["models"]["L"]["status"] == "issued" for row in group),
                "QModelIssued": sum(row["models"].get("Q", {}).get("status") == "issued" for row in group),
            }
    primary = comparisons["P_vs_B"]["macro"]
    coverage_loss = any(value["baselineCoverageLossKeys"] for value in coverage.values())
    all40 = primary["status"] == "complete"
    price_pass = all40 and primary["aMape"] < primary["bMape"] and primary["aMae"] <= primary["bMae"]
    passed = price_pass and not coverage_loss and integrity_ok
    score_report = {
        "schemaVersion": 1, "phase": phase, "status": "scored", "manifestSha256": manifest["manifestSha256"],
        "comparisons": comparisons, "dm": all_dm,
        "descriptiveLegacyComparisons": legacy_comparisons,
        "directModelOnlyPvsB": model_only,
        "dmConclusionByComparison": dm_conclusions,
        "dmSummaryByBandwidth": [{
            "bandwidth": bw, "familySize": sum(row["bandwidth"] == bw for row in all_dm),
            "significantImprovement": sum(row["bandwidth"] == bw and row.get("holmSignificant")
                                          and row["meanLossDifference"] is not None and row["meanLossDifference"] < 0 for row in all_dm),
            "significantWorsening": sum(row["bandwidth"] == bw and row.get("holmSignificant")
                                        and row["meanLossDifference"] is not None and row["meanLossDifference"] > 0 for row in all_dm),
            "inferenceUnavailable": sum(row["bandwidth"] == bw and row["status"] != "tested" for row in all_dm)}
            for bw in BANDWIDTHS],
        "jointCalendarBlockBootstrap": bootstraps, "coverage": coverage,
        "priceGainLossContributions": contributions, "evalLogDeltaDecomposition": eval_log,
        "limitations": ["V/T are previously inspected retrospective periods, not independent future validation",
                        "nonsignificant tests do not establish equivalence",
                        "P−Q is one fixed response permutation and does not isolate all shared information"],
    }
    diagnostic_report = {"schemaVersion": 1, "phase": phase, "manifestSha256": manifest["manifestSha256"],
                         **diagnostics, "evalLogDeltaDecomposition": eval_log,
                         "priceGainLossContributions": contributions, "coverage": coverage}
    decision = {"schemaVersion": 1, "phase": phase,
                "status": "pass" if passed else ("technical_or_integrity_block" if not integrity_ok else "price_or_coverage_fail"),
                "gate": {"all40CellsSupported": all40,
                         "PMapeStrictlyLowerThanB": bool(all40 and primary["aMape"] < primary["bMape"]),
                         "PMaeNoHigherThanB": bool(all40 and primary["aMae"] <= primary["bMae"]),
                         "noBaselineCoverageLoss": not coverage_loss,
                         "allFourProductsTechnicallyComplete": diagnostics["allProductsComplete"],
                         "timeIntegrityVerified": diagnostics["timeIntegrityVerified"],
                         "pqIdentityVerified": diagnostics["pqIdentityVerified"],
                         "originWeightIntegrityVerified": diagnostics["originWeightIntegrityVerified"],
                         "representativeTrainKeysIdentical": diagnostics["representativeTrainKeysIdentical"],
                         "representativeTrainDiagnosticsComplete": diagnostics["representativeTrainDiagnosticsComplete"],
                         "permutationDiagnosticsComplete": diagnostics["permutationDiagnosticsComplete"],
                         "trainLogDecompositionComplete": diagnostics["trainLogDecompositionComplete"],
                         "informationIntegrityVerified": diagnostics["informationIntegrityVerified"],
                         "informationIntegrityChecks": diagnostics["informationIntegrityChecks"],
                         "fitFoldSetComplete": folds_complete, "passed": passed},
                "primaryMetrics": primary, "TExecutionAllowed": bool(phase == "V" and passed),
                "nextStep": "run_T" if phase == "V" and passed else
                            ("retain_E106_RHS" if not passed else "retain_candidate_for_retrospective_record")}
    return score_report, diagnostic_report, decision


def write_exclusive(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prices", type=Path)
    parser.add_argument("score_output", type=Path)
    parser.add_argument("diagnostic_output", type=Path)
    parser.add_argument("decision_output", type=Path)
    args = parser.parse_args(argv)
    if price_helper._sha(args.prices) != price_helper.PRICES_SHA256:
        raise SystemExit("pinned prices file hash mismatch")
    reports = score(json.loads(args.manifest.read_text(encoding="utf-8")),
                    json.loads(args.prices.read_text(encoding="utf-8")),
                    manifest_path=args.manifest, prices_path=args.prices)
    for path, value in zip((args.score_output, args.diagnostic_output, args.decision_output), reports):
        write_exclusive(path, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
