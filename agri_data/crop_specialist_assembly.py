"""Frozen V selection and selected-only T assembly for CROP_SPECIALIST_ASSEMBLY_V1.

This module reads previously saved CAND_A/CAND_B and E106-RHS forecasts.  The
V selector performs no fitting; the T runner is intentionally added only after
the V selection contract has passed review.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import external_reference_v1 as source_a
from . import intra_crop_panel as source_b
from . import chronos2_market_context as rhs_source
from . import equal_info_online_forecasts as price_source
from .chronos2_market_context import BASELINE_PATHS, BASELINE_SHA256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/crop-specialist-assembly-v1-20261013"
PRICE_PATH = price_source.PRICES_PATH
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
TIE_ORDER = {name: i for i, name in enumerate(("B", "L", "A", "P"))}
THREAD_CAP_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


class _ExpectedFitFailure(RuntimeError):
    """A declared support shortfall that permits the documented B fallback."""


def _enforce_thread_limits() -> None:
    bad_caps = {name: os.environ.get(name) for name in THREAD_CAP_ENV
                if os.environ.get(name) != "1"}
    if bad_caps:
        raise RuntimeError(f"T requires all six frozen thread caps set to 1: {bad_caps}")
POLICY_RULE = "per-product-10-horizon-equal-weight-MAPE;eligible iff MAPE<B, MAE<=B, coverage>=B;minimum-unrounded-MAPE-then-MAE-then-B-L-A-P"
PINNED_A_V_MANIFEST_SHA256 = "1257f6c49c48b8dbc0f8467d91d025b8dbf9f5ea24db6127b3d2f7916a87138d"
PINNED_CAND_B_V_MANIFEST_SHA256 = "b5c340eca95ed5bf9e31372552fb009bce21a28e366710e2a9fba7fee5e14077"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"refusing to overwrite frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def _positive(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) and n > 0 else None


def _date(value: Any) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date {value!r}") from exc


def _month_key(value: Any) -> dt.date:
    """Return the first day of a parsed date's calendar month."""
    return _date(value).replace(day=1)


def _source_rows(document: Mapping[str, Any], source: str,
                 target_version: str) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    """Normalize one saved V source without attaching or looking up labels."""
    if document.get("phase") != "V" or document.get("targetValuesAttached") is not False:
        raise ValueError(f"{source} is not a saved answer-free V forecast")
    rows = document.get("forecasts")
    if not isinstance(rows, list):
        raise ValueError(f"{source} has no forecast rows")
    result: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} contains a malformed forecast row")
        product = str(row.get("product"))
        origin = _date(row.get("origin")).isoformat()
        target = _date(row.get("target")).isoformat()
        horizon = int(row.get("horizon"))
        if product not in PRODUCTS or horizon not in HORIZONS or _date(target) - _date(origin) != dt.timedelta(days=horizon):
            raise ValueError(f"{source} has an invalid product/horizon/origin-target row")
        key = (product, origin, target, horizon, target_version)
        if key in result:
            raise ValueError(f"{source} contains a duplicate exact key {key}")
        if source == "B":
            payload = row.get("baseline", row.get("B", {}))
            payload = payload.get("E106_RHS", {}) if isinstance(payload, Mapping) else {}
        elif source == "A":
            payload = row.get("CAND_A", {})
        elif source == "L":
            models = row.get("models", {})
            payload = models.get("L", {}) if isinstance(models, Mapping) else {}
        elif source == "P":
            models = row.get("models", {})
            payload = models.get("P", {}) if isinstance(models, Mapping) else {}
            # A technical RHS fallback is not a direct P observation.
            if isinstance(payload, Mapping) and payload.get("modelUsed") is False:
                payload = {**payload, "prediction": None, "status": "not_issued"}
            elif isinstance(payload, Mapping) and payload.get("rawModelStatus") is not None:
                payload = {**payload, "prediction": payload.get("rawModelPrediction"),
                           "status": payload.get("rawModelStatus")}
        else:
            raise ValueError(f"unknown source {source}")
        payload = payload if isinstance(payload, Mapping) else {}
        prediction = _positive(payload.get("prediction")) if payload.get("status") == "issued" else None
        result[key] = {"prediction": prediction,
                       "status": "issued" if prediction is not None else "not_issued"}
    return result


def _verify_manifest_self_hash(document: Mapping[str, Any], source: str) -> None:
    copied = document.get("manifestSha256")
    unsigned = dict(document)
    unsigned.pop("manifestSha256", None)
    if not copied or copied != _sha(_canonical(unsigned)):
        raise ValueError(f"{source} manifest self-hash mismatch")


def _score_policy(rows: Mapping[tuple[str, str, str, int, str], Mapping[str, Any]],
                  actuals: Mapping[tuple[str, str], float], product: str) -> dict[str, Any]:
    by_horizon: dict[int, list[tuple[float, float | None]]] = defaultdict(list)
    for (p, _origin, target, horizon, _version), forecast in rows.items():
        if p != product:
            continue
        actual = actuals.get((p, target))
        if actual is not None:
            by_horizon[horizon].append((actual, _positive(forecast.get("prediction"))))
    if set(by_horizon) != set(HORIZONS):
        raise ValueError(f"{product}: V forecast sources do not cover all 10 horizons")
    horizon_stats = {}
    for horizon in HORIZONS:
        pairs = by_horizon[horizon]
        if not pairs:
            raise ValueError(f"{product}/h{horizon}: no positive V targets")
        valid = [(actual, pred) for actual, pred in pairs if pred is not None]
        issue_rows = [(row.get("prediction"), key) for key, row in rows.items()
                      if key[0] == product and key[3] == horizon]
        issued_keys = [key for prediction, key in issue_rows if _positive(prediction) is not None]
        horizon_stats[str(horizon)] = {
            "targetCount": len(pairs), "issuedCount": len(valid),
            "coverage": len(valid) / len(pairs),
            "issuanceTargetCount": len(issue_rows), "issuanceCount": len(issued_keys),
            "issuanceCoverage": len(issued_keys) / len(issue_rows) if issue_rows else 0.0,
            "issuedKeysSha256": _sha(_canonical(sorted([list(key) for key in issued_keys]))),
            "mape": math.fsum(abs(actual - pred) / actual for actual, pred in valid) / len(valid) * 100 if valid else None,
            "mae": math.fsum(abs(actual - pred) for actual, pred in valid) / len(valid) if valid else None,
        }
    valid_horizons = [horizon_stats[str(h)] for h in HORIZONS]
    return {
        "horizonCount": len(HORIZONS),
        "targetCountPerHorizon": {str(h): horizon_stats[str(h)]["targetCount"] for h in HORIZONS},
        "issuedCountPerHorizon": {str(h): horizon_stats[str(h)]["issuedCount"] for h in HORIZONS},
        "coveragePerHorizon": {str(h): horizon_stats[str(h)]["coverage"] for h in HORIZONS},
        "issuanceCoveragePerHorizon": {str(h): horizon_stats[str(h)]["issuanceCoverage"] for h in HORIZONS},
        "issuanceCountPerHorizon": {str(h): horizon_stats[str(h)]["issuanceCount"] for h in HORIZONS},
        "issuanceTargetCountPerHorizon": {str(h): horizon_stats[str(h)]["issuanceTargetCount"] for h in HORIZONS},
        "mapePerHorizon": {str(h): horizon_stats[str(h)]["mape"] for h in HORIZONS},
        "maePerHorizon": {str(h): horizon_stats[str(h)]["mae"] for h in HORIZONS},
        "mape": math.fsum(x["mape"] for x in valid_horizons) / len(HORIZONS)
                if all(x["mape"] is not None for x in valid_horizons) else None,
        "mae": math.fsum(x["mae"] for x in valid_horizons) / len(HORIZONS)
               if all(x["mae"] is not None for x in valid_horizons) else None,
        "coverage": math.fsum(x["coverage"] for x in valid_horizons) / len(HORIZONS),
        "horizonStats": horizon_stats,
    }


def _score_pair(left: Mapping[tuple[str, str, str, int, str], Mapping[str, Any]],
                right: Mapping[tuple[str, str, str, int, str], Mapping[str, Any]],
                actuals: Mapping[tuple[str, str], float], product: str) -> dict[str, Any]:
    """Compare two candidates on exact shared issued keys, equally by horizon."""
    by_horizon: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for key in sorted(set(left) & set(right)):
        p, _origin, target, horizon, _version = key
        if p != product or (p, target) not in actuals:
            continue
        a, b = (_positive(left[key].get("prediction")), _positive(right[key].get("prediction")))
        if a is not None and b is not None:
            by_horizon[horizon].append((actuals[(p, target)], a, b))
    if set(by_horizon) != set(HORIZONS) or any(not by_horizon[h] for h in HORIZONS):
        return {"pairReady": False,
                "commonIssuedCountPerHorizon": {str(h): len(by_horizon[h]) for h in HORIZONS},
                "leftMAPE": None, "rightMAPE": None, "leftMAE": None, "rightMAE": None}
    left_mape, right_mape, left_mae, right_mae = [], [], [], []
    for h in HORIZONS:
        triples = by_horizon[h]
        left_mape.append(math.fsum(abs(y - a) / y for y, a, _ in triples) / len(triples) * 100)
        right_mape.append(math.fsum(abs(y - b) / y for y, _, b in triples) / len(triples) * 100)
        left_mae.append(math.fsum(abs(y - a) for y, a, _ in triples) / len(triples))
        right_mae.append(math.fsum(abs(y - b) for y, _, b in triples) / len(triples))
    return {"pairReady": True,
            "commonIssuedCountPerHorizon": {str(h): len(by_horizon[h]) for h in HORIZONS},
            "leftMAPE": math.fsum(left_mape) / len(HORIZONS),
            "rightMAPE": math.fsum(right_mape) / len(HORIZONS),
            "leftMAE": math.fsum(left_mae) / len(HORIZONS),
            "rightMAE": math.fsum(right_mae) / len(HORIZONS),
            "leftMAPEPerHorizon": {str(h): left_mape[i] for i, h in enumerate(HORIZONS)},
            "rightMAPEPerHorizon": {str(h): right_mape[i] for i, h in enumerate(HORIZONS)},
            "leftMAEPerHorizon": {str(h): left_mae[i] for i, h in enumerate(HORIZONS)},
            "rightMAEPerHorizon": {str(h): right_mae[i] for i, h in enumerate(HORIZONS)}}


def build_selection(*, source_paths: Mapping[str, Path], prices_path: Path,
                    source_hashes: Mapping[str, str] | None = None,
                    expected_price_sha256: str | None = None,
                    rhs_baseline_path: Path | None = None,
                    rhs_baseline_sha256: str | None = None) -> dict[str, Any]:
    """Build immutable V map from frozen forecasts and target snapshot.

    Labels are attached only here, after all four source key sets and target
    version bindings have been checked.  This function must be reviewed before
    invocation on the live saved artifacts.
    """
    if set(source_paths) != {"A", "B", "L", "P"}:
        raise ValueError("source_paths must identify exactly A, B, L and P")
    price_hash = _file_sha(prices_path)
    if expected_price_sha256 is not None and price_hash != expected_price_sha256:
        raise ValueError("pinned price snapshot file hash mismatch")
    prices = _read_json(prices_path)
    products = {str(p.get("id")): p for p in prices.get("products", []) if isinstance(p, Mapping)}
    actuals: dict[tuple[str, str], float] = {}
    for product in PRODUCTS:
        for row in products.get(product, {}).get("observations", []):
            if isinstance(row, Mapping):
                value = _positive(row.get("actual"))
                if value is not None:
                    actuals[(product, _date(row.get("date")).isoformat())] = value
    if not set(PRODUCTS).issubset(products):
        raise ValueError("pinned target snapshot lacks one or more products")

    docs = {name: _read_json(path) for name, path in source_paths.items()}
    actual_source_hashes = {name: _file_sha(source_paths[name]) for name in source_paths}
    if source_hashes is not None and any(source_hashes.get(name) != digest
                                         for name, digest in actual_source_hashes.items()):
        raise ValueError("saved V source manifest file hash differs from pinned provenance")
    for name, doc in docs.items():
        _verify_manifest_self_hash(doc, name)
        version = (doc.get("priceSnapshotSha256") if name == "A" else
                   doc.get("targetSnapshotSha256"))
        if version != price_hash:
            raise ValueError(f"{name} saved forecasts use a different target snapshot version")
    rhs_path = rhs_baseline_path or BASELINE_PATHS["V"]
    rhs_sha = rhs_baseline_sha256 or BASELINE_SHA256["V"]
    for name in ("A", "B"):
        if docs[name].get("baselineManifestSha256") != rhs_sha:
            raise ValueError(f"{name} is not bound to the independently pinned V RHS baseline")
    version_set = {((doc.get("priceSnapshotSha256") if name == "A" else doc.get("targetSnapshotSha256")))
                   for name, doc in docs.items()}
    if version_set != {price_hash}:
        raise ValueError("A/B/L/P target version provenance differs")
    normalized = {name: _source_rows(docs[name], name, price_hash) for name in ("B", "L", "A", "P")}
    baseline_keys = set(normalized["B"])
    for name in ("L", "A", "P"):
        if set(normalized[name]) != baseline_keys:
            raise ValueError(f"{name} key set differs from B: missing={len(baseline_keys-set(normalized[name]))}, unexpected={len(set(normalized[name])-baseline_keys)}")
    if _file_sha(rhs_path) != rhs_sha:
        raise ValueError("raw pinned V RHS baseline file hash mismatch")
    raw_rhs = source_a._baseline_index(_read_json(rhs_path), "V")
    raw_rhs_rows = {}
    for (product, target, horizon), row in raw_rhs.items():
        key = (product, row["origin"].isoformat(), target.isoformat(), horizon, price_hash)
        raw_rhs_rows[key] = {"prediction": _positive(row.get("E106_RHS")),
                             "status": "issued" if _positive(row.get("E106_RHS")) is not None else "not_issued"}
    if set(raw_rhs_rows) != baseline_keys:
        raise ValueError("copied CAND_B B forecast keys differ from pinned raw V RHS baseline")
    for key, expected in raw_rhs_rows.items():
        copied = normalized["B"][key]
        if copied["prediction"] != expected["prediction"] or copied["status"] != expected["status"]:
            raise ValueError(f"copied CAND_B B forecast differs from pinned raw RHS row: {key}")
    v_targets = {(key[0], key[2]) for key in baseline_keys}
    actuals = {key: value for key, value in actuals.items() if key in v_targets}
    scored = {product: {model: _score_policy(normalized[model], actuals, product)
                        for model in ("B", "L", "A", "P")} for product in PRODUCTS}
    choices = {}
    for product in PRODUCTS:
        baseline = scored[product]["B"]
        eligible = []
        reasons = {}
        paired_metrics = {}
        for model in ("A", "L", "P"):
            candidate = scored[product][model]
            paired = _score_pair(normalized[model], normalized["B"], actuals, product)
            baseline_issued = {key for key, row in normalized["B"].items()
                               if key[0] == product and _positive(row.get("prediction")) is not None}
            candidate_issued = {key for key, row in normalized[model].items()
                                if key[0] == product and _positive(row.get("prediction")) is not None}
            no_issue_loss = baseline_issued.issubset(candidate_issued)
            passes = (paired["pairReady"]
                      and paired["leftMAPE"] < paired["rightMAPE"]
                      and paired["leftMAE"] <= paired["rightMAE"]
                      and candidate["mape"] is not None
                      and baseline["mape"] is not None
                      and candidate["coverage"] >= baseline["coverage"]
                      and no_issue_loss)
            reasons[model] = {"eligible": passes,
                              "mapeStrictlyBelowB": (paired["leftMAPE"] < paired["rightMAPE"]
                                                      if paired["pairReady"] else None),
                              "maeNoHigherThanB": (paired["leftMAE"] <= paired["rightMAE"]
                                                   if paired["pairReady"] else None),
                              "coverageNoLowerThanB": candidate["coverage"] >= baseline["coverage"],
                              "noBIssuedKeyLoss": no_issue_loss,
                              "pairedMetricsVsB": paired}
            paired_metrics[model] = paired
            if passes:
                eligible.append(model)
        if baseline["mape"] is None or baseline["mae"] is None:
            raise ValueError(f"{product}: B has no complete 10-horizon V score")
        selected = min(eligible, key=lambda model: (scored[product][model]["mape"],
                                                     scored[product][model]["mae"],
                                                     TIE_ORDER[model])) if eligible else "B"
        choices[product] = {"selectedModelId": selected,
                            "eligibleAlternatives": eligible,
                            "eligibility": reasons,
                            "metrics": scored[product],
                            "pairedMetricsVsB": paired_metrics,
                            "selectionReason": "lowest-unrounded-MAPE-then-MAE-fixed-order" if eligible else "no-eligible-alternative-B-retained"}
    paths = {name: {"path": str(source_paths[name]),
                    "sha256": actual_source_hashes[name],
                    "manifestSha256": docs[name].get("manifestSha256"),
                    "forecastCount": len(docs[name]["forecasts"])} for name in ("B", "A", "L", "P")}
    selection = {"schemaVersion": 1, "experiment": "CROP_SPECIALIST_ASSEMBLY_V1",
                 "stage": "V", "policyRule": POLICY_RULE,
                 "selectionUsesPreviouslyInspectedValidation": True,
                 "targetVersion": price_hash,
                 "targetSnapshot": {"path": str(prices_path), "sha256": price_hash},
                 "selectorCodeSha256": _file_sha(Path(__file__)),
                 "pinnedRhsBaseline": {"path": str(rhs_path), "sha256": rhs_sha},
                 "sources": paths,
                 "exactJoinKey": ["product", "origin", "target", "horizon", "target_version"],
                 "sourceKeyCount": len(baseline_keys),
                 "joinedKeyCount": len(baseline_keys),
                 "actualCoverage": {"positiveActualTargetCount": len(actuals),
                                     "positiveActualKeysSha256": _sha(_canonical(sorted([list(k) for k in actuals])))},
                 "products": choices}
    cell_values = []
    for product in PRODUCTS:
        selected = choices[product]["selectedModelId"]
        pair = ({"leftMAPEPerHorizon": {str(h): scored[product]["B"]["mapePerHorizon"][str(h)] for h in HORIZONS},
                                              "rightMAPEPerHorizon": {str(h): scored[product]["B"]["mapePerHorizon"][str(h)] for h in HORIZONS},
                                              "leftMAEPerHorizon": {str(h): scored[product]["B"]["maePerHorizon"][str(h)] for h in HORIZONS},
                                              "rightMAEPerHorizon": {str(h): scored[product]["B"]["maePerHorizon"][str(h)] for h in HORIZONS}}
                                             if selected == "B" else choices[product]["pairedMetricsVsB"][selected])
        for h in HORIZONS:
            cell_values.append({"product": product, "horizon": h,
                                "G_MAPE": pair["leftMAPEPerHorizon"][str(h)],
                                "B_MAPE": pair["rightMAPEPerHorizon"][str(h)],
                                "G_MAE": pair["leftMAEPerHorizon"][str(h)],
                                "B_MAE": pair["rightMAEPerHorizon"][str(h)]})
    selection["vRecomposition40CellMetrics"] = {
        "cellCount": len(cell_values), "cellsSha256": _sha(_canonical(cell_values)),
        "cells": cell_values,
        "G_MAPE": math.fsum(x["G_MAPE"] for x in cell_values) / len(cell_values),
        "B_MAPE": math.fsum(x["B_MAPE"] for x in cell_values) / len(cell_values),
        "G_MAE": math.fsum(x["G_MAE"] for x in cell_values) / len(cell_values),
        "B_MAE": math.fsum(x["B_MAE"] for x in cell_values) / len(cell_values)}
    folds_per_policy = 30
    fit_models = {product: set() for product in PRODUCTS}
    for product in PRODUCTS:
        selected = choices[product]["selectedModelId"]
        if selected != "B":
            fit_models[product].add(selected)
    fit_models["radish"].update(("P", "L"))
    fit_models["onion"].update(("A", "CTRL"))
    selection["plannedFitCount"] = sum(len(models) * folds_per_policy for models in fit_models.values())
    selection["plannedFitsByProduct"] = {product: {"models": sorted(fit_models[product]),
                                                   "fitCount": len(fit_models[product]) * folds_per_policy}
                                          for product in PRODUCTS}
    selection["selectionSha256"] = _sha(_canonical(selection))
    return selection


def select_v(*, output_dir: Path = OUT) -> dict[str, Any]:
    """Resolve V sources and freeze selection.json once; performs no fitting."""
    paths = {"A": source_a.OUT / "V/forecast-manifest.json",
             "B": source_b.OUT / "V/forecast-manifest.json",
             "L": source_b.OUT / "V/forecast-manifest.json",
             "P": source_b.OUT / "V/forecast-manifest.json"}
    # B/L/P come from the same answer-free CAND_B manifest. Its B values are
    # independently bound to the frozen RHS baseline by baselineManifestSha256.
    expected_hashes = {"A": PINNED_A_V_MANIFEST_SHA256,
                       "B": PINNED_CAND_B_V_MANIFEST_SHA256,
                       "L": PINNED_CAND_B_V_MANIFEST_SHA256,
                       "P": PINNED_CAND_B_V_MANIFEST_SHA256}
    selection = build_selection(source_paths=paths, prices_path=PRICE_PATH,
                                source_hashes=expected_hashes,
                                expected_price_sha256=price_source.PRICES_SHA256)
    path = output_dir / "selection.json"
    _write_once(path, selection)
    return {"status": "frozen", "selectionPath": str(path),
            "selectionSha256": selection["selectionSha256"],
            "selected": {p: v["selectedModelId"] for p, v in selection["products"].items()}}


def _source_sha_map() -> dict[str, str]:
    files = {"producer": Path(__file__), "candidateA": Path(source_a.__file__),
             "candidateB": Path(source_b.__file__), "modeling": Path(source_b.modeling.__file__),
             "dashboard": Path(source_b.dashboard.__file__),
             "priceSource": Path(price_source.__file__),
             "rhsSource": Path(rhs_source.__file__)}
    result = {name: _file_sha(path) for name, path in files.items()}
    scorer_path = Path(__file__).with_name("crop_specialist_assembly_score.py")
    if scorer_path.is_file():
        result["scorer"] = _file_sha(scorer_path)
    return result


def _execution_identity(config: Mapping[str, Any], cache_file_sha: str,
                        selection_sha: str) -> dict[str, Any]:
    packages = ("lightgbm", "numpy", "scipy", "pandas", "joblib", "threadpoolctl")
    runtime = {name: importlib.metadata.version(name) for name in packages}
    source_hashes = _source_sha_map()
    identity = {"experiment": "CROP_SPECIALIST_ASSEMBLY_V1", "stage": "T",
                "sourceHashes": source_hashes,
                "sourceHashesSha256": _sha(_canonical(source_hashes)),
                "sourceCacheFileSha256": cache_file_sha,
                "sourceCacheInternalSha256": source_b.SOURCE_CACHE_INTERNAL_SHA256,
                "resolvedConfigSha256": config["configSha256"],
                "selectionSha256": selection_sha,
                "effectiveFitParams": source_b._fit_parameters(config),
                "effectiveDatasetParams": source_b._dataset_params(config),
                "CAND_AConfig": dict(source_a.LIGHTGBM_CONFIG),
                "python": platform.python_version(), "runtimeDependencies": runtime,
                "threadLimits": {name: os.environ.get(name) for name in THREAD_CAP_ENV},
                "threadLimitEnforcement": "launch-time environment caps; model parameters set one thread"}
    identity["executionIdentitySha256"] = _sha(_canonical(identity))
    return identity


def _training_target_digest(rows: list[Mapping[str, Any]]) -> str:
    material = [[r["origin"].isoformat(), r["target"].isoformat(), float(r["label"])]
                for r in rows]
    return _sha(_canonical(material))


def _temporal_fit_evidence(rows: list[Mapping[str, Any]], fit_origin: dt.date,
                           *, feature_date_key: str = "sourceDate") -> dict[str, Any]:
    """Record the label and feature boundaries used by one monthly fit."""
    targets = [r["target"] for r in rows]
    target_max = max(targets) if targets else None
    feature_dates = [r.get(feature_date_key, r.get("p0Date")) for r in rows]
    cutoffs = [r.get("cutoff") for r in rows]
    feature_checks = []
    for feature_day, cutoff in zip(feature_dates, cutoffs):
        if feature_day is None or cutoff is None:
            feature_checks.append(False)
            continue
        feature_day = feature_day if isinstance(feature_day, dt.date) else _date(feature_day)
        cutoff = cutoff if isinstance(cutoff, dt.date) else _date(cutoff)
        feature_checks.append(feature_day <= cutoff)
    return {"fitTargetMax": target_max.isoformat() if target_max else None,
            "trainingTargetsBeforeFitOrigin": all(day < fit_origin for day in targets),
            "allFeatureDatesAtOrBeforeCutoff": all(feature_checks)}


def _train_l_only(product: str, horizon: int, fit_month: dt.date,
                  rep_rows: list[dict[str, Any]], config: Mapping[str, Any],
                  phase_dir: Path, execution_sha: str) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Persist/reuse only L; deliberately does not construct or fit P."""
    x = source_b._x_matrix(rep_rows)
    keys_sha = source_b._key_digest(rep_rows)
    target_sha = _training_target_digest(rep_rows)
    identity = {"product": product, "horizon": horizon,
                "fitMonth": fit_month.isoformat(), "fitOrigin": fit_month.isoformat(),
                "variant": "L", "representativeKeysSha256": keys_sha,
                "representativeInputWeightSha256": source_b._input_weight_digest(rep_rows),
                "trainingRowsSha256": _sha(_canonical([[r["origin"].isoformat(),
                                                          r["target"].isoformat(),
                                                          float(r["label"]), 1.0]
                                                         for r in rep_rows])),
                "configSha256": config["configSha256"],
                "featureNames": list(source_b.FEATURE_NAMES),
                "sourceCacheFileSha256": _file_sha(source_b.SOURCE_CACHE),
                "executionIdentitySha256": execution_sha}
    path = phase_dir / "models" / f"{product}-h{horizon:02d}-{fit_month:%Y-%m}-L.txt"
    sidecar = path.with_suffix(".identity.json")
    if path.exists() != sidecar.exists():
        raise ValueError("L checkpoint file and identity sidecar are incomplete")
    started = time.perf_counter()
    if path.exists():
        model, model_sha, identity_sha = source_b._load_booster(path, sidecar, identity)
        state = "reused"
    else:
        model, _ = source_b._train_native(x, [float(r["label"]) for r in rep_rows],
                                          [1.0] * len(rep_rows), config)
        model_sha, identity_sha = source_b._save_booster(model, path, identity)
        state = "fit"
        model, model_sha, identity_sha = source_b._load_booster(path, sidecar, identity)
    elapsed = time.perf_counter() - started
    predictions = np.asarray(model.predict(x), dtype=np.float64)
    if predictions.shape != (len(rep_rows),) or not np.isfinite(predictions).all():
        raise ValueError("L training replay produced invalid representative predictions")
    diagnostics = _representative_diagnostics(rep_rows, predictions)
    record = {"status": state, "modelFile": str(path.relative_to(phase_dir.parent)),
              "modelFileSha256": model_sha, "checkpointIdentitySha256": identity_sha,
              "fitSeconds": elapsed, "trainingRows": len(rep_rows),
              "fitTargetCount": len({r["target"] for r in rep_rows}),
              "fitTargetSha256": target_sha, "representativeKeysSha256": keys_sha,
              "representativeTrainDiagnostics": diagnostics}
    return record, model, identity


def _representative_diagnostics(rows: list[Mapping[str, Any]], log_ratio_predictions: Any) -> dict[str, Any]:
    ratios = np.asarray(log_ratio_predictions, dtype=np.float64)
    labels = np.asarray([float(r["label"]) for r in rows], dtype=np.float64)
    p0 = np.asarray([float(r["p0"]) for r in rows], dtype=np.float64)
    actual = np.asarray([float(r["targetPrice"]) for r in rows], dtype=np.float64)
    predicted = np.asarray([source_a._price_from_log_ratio(float(start), float(value))
                            for start, value in zip(p0, ratios)], dtype=np.float64)
    errors = labels - ratios
    return {"n": len(rows), "logMse": float(np.mean(errors ** 2)),
            "logMae": float(np.mean(np.abs(errors))),
            "priceMAE": float(np.mean(np.abs(actual - predicted))),
            "priceMAPE": float(np.mean(np.abs(actual - predicted) / actual) * 100)}


def _log_loss_decomposition(left_labels: np.ndarray, left_predictions: np.ndarray,
                            right_predictions: np.ndarray) -> dict[str, Any]:
    delta = right_predictions - left_predictions
    left_error = left_labels - left_predictions
    moved = delta ** 2
    cross = -2 * left_error * delta
    squared_difference = ((left_labels - right_predictions) ** 2 -
                          (left_labels - left_predictions) ** 2)
    identity_error = squared_difference - (moved + cross)
    return {"n": int(len(left_labels)), "deltaSquaredSum": float(np.sum(moved)),
            "crossTermSum": float(np.sum(cross)),
            "squaredErrorDifferenceSum": float(np.sum(squared_difference)),
            "identityMaxAbsError": float(np.max(np.abs(identity_error), initial=0.0))}


def _save_pair_checkpoint(train_rows: list[dict[str, Any]], path: Path,
                          identity: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import joblib
    full_targets_sha = source_a._sha(source_a._canonical([
        {"target": r["target"].isoformat(), "label": float(r["label"]),
         "basic40": r["basic40"], "candidate": r["candidate"]} for r in train_rows]))
    checkpoint_identity = {**dict(identity), "trainingRows": len(train_rows),
                           "trainingTargetsSha256": full_targets_sha,
                           "modelConfig": dict(source_a.LIGHTGBM_CONFIG)}
    identity_sha = _sha(_canonical(checkpoint_identity))
    sidecar = path.with_suffix(".identity.json")
    if path.exists() != sidecar.exists():
        raise ValueError("A/CTRL checkpoint file and identity sidecar are incomplete")
    started = time.perf_counter()
    if path.exists():
        bundle = joblib.load(path)
        side = _read_json(sidecar)
        if (bundle.get("checkpointIdentity") != checkpoint_identity
                or side.get("identity") != checkpoint_identity
                or side.get("identitySha256") != identity_sha
                or side.get("checkpointFileSha256") != _file_sha(path)):
            raise ValueError("saved A/CTRL checkpoint identity/hash mismatch")
        state = "reused"
    else:
        ctrl, cand = source_a._fit_pair(train_rows)
        bundle = {"checkpointIdentity": checkpoint_identity,
                  "CTRL_LGB": ctrl, "CAND_A": cand}
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(bundle, temp)
        if path.exists():
            temp.unlink(missing_ok=True)
            raise ValueError("A/CTRL checkpoint appeared during save")
        temp.replace(path)
        _write_once(sidecar, {"identity": checkpoint_identity,
                              "identitySha256": identity_sha,
                              "checkpointFileSha256": _file_sha(path)})
        state = "fit"
    elapsed = time.perf_counter() - started
    # Reload after save so inference always uses the persisted checkpoint.
    reloaded = joblib.load(path)
    if reloaded.get("checkpointIdentity") != checkpoint_identity:
        raise ValueError("reloaded A/CTRL checkpoint identity mismatch")
    return reloaded["CTRL_LGB"], reloaded["CAND_A"], {
        "status": state, "path": path, "fileSha256": _file_sha(path),
        "identity": checkpoint_identity, "identitySha256": identity_sha,
        "fitSeconds": elapsed, "fitTargetSha256": full_targets_sha,
        "fitTargetCount": len({r["target"] for r in train_rows}),
        "trainingRows": len(train_rows)}


def _forecast_a_model(model: Any, row: Mapping[str, Any], *, feature_name: str) -> dict[str, Any]:
    values = row.get(feature_name)
    if not isinstance(values, (list, tuple)) or not values:
        return {"prediction": None, "status": "not_issued", "reason": "feature_unavailable"}
    try:
        ratio = float(np.asarray(model.predict([list(values)]), dtype=np.float64)[0])
        prediction = source_a._price_from_log_ratio(float(row["p0"]), ratio)
    except (KeyError, ValueError, OverflowError, FloatingPointError, IndexError) as exc:
        return {"prediction": None, "status": "not_issued",
                "reason": f"model_output_unusable:{type(exc).__name__}"}
    return {"prediction": prediction, "status": "issued", "logRatio": ratio,
            "p0": float(row["p0"]), "p0Date": row["p0Date"]}


def run_t(*, output_dir: Path = OUT) -> dict[str, Any]:
    """Train selected T specialists/controls, save answer-free forecasts, replay.

    This stage must only be invoked after Astra approves the final T producer
    and scorer hashes. It never scores or joins target labels.
    """
    forecasts_path = output_dir / "T/forecasts.json"
    if forecasts_path.is_file():
        replay = verify_saved_t_replay(output_dir=output_dir,
                                       forecasts_path=forecasts_path)
        doc = _read_json(forecasts_path)
        return {"status": "saved_forecasts_replayed", "forecastsPath": str(forecasts_path),
                "manifestSha256": doc["manifestSha256"],
                "forecastRowsSha256": doc["forecastRowsSha256"],
                "replayVerificationSha256": replay["replayVerificationSha256"],
                "forecastKeyCount": doc["forecastKeyCount"],
                "plannedFitCount": doc["plannedFitCount"],
                "actualEstimatorFitCount": doc["actualEstimatorFitCount"],
                "newEstimatorFitCount": doc["newEstimatorFitCount"]}
    selection_path = output_dir / "selection.json"
    selection = _read_json(selection_path)
    selection_sha = selection.get("selectionSha256")
    if selection.get("experiment") != "CROP_SPECIALIST_ASSEMBLY_V1" or selection.get("stage") != "V":
        raise ValueError("T requires the frozen V selection artifact")
    unsigned_selection = dict(selection)
    unsigned_selection.pop("selectionSha256", None)
    if selection_sha != _sha(_canonical(unsigned_selection)):
        raise ValueError("frozen selection self-hash mismatch")
    if _file_sha(PRICE_PATH) != price_source.PRICES_SHA256 or selection.get("targetVersion") != price_source.PRICES_SHA256:
        raise ValueError("T price snapshot differs from the frozen V target version")
    if _file_sha(source_a.OUT / "V/forecast-manifest.json") != PINNED_A_V_MANIFEST_SHA256:
        raise ValueError("pinned CAND_A V source changed after selection")
    if _file_sha(source_b.OUT / "V/forecast-manifest.json") != PINNED_CAND_B_V_MANIFEST_SHA256:
        raise ValueError("pinned CAND_B V source changed after selection")
    if _file_sha(BASELINE_PATHS["T"]) != BASELINE_SHA256["T"]:
        raise ValueError("pinned raw T RHS baseline file hash mismatch")
    _enforce_thread_limits()
    import lightgbm as lgb

    # Input-only verification. This reuses saved local source artifacts; it does
    # not extract data, fit models, or inspect actual targets for scoring.
    preflight = source_b.preflight(output_dir=source_b.OUT)
    config = source_b._load_resolved_config(source_b.OUT)
    cache_doc = source_b._read_json(source_b.SOURCE_CACHE)
    channels, channel_selection = source_a.load_source_cache(source_b.SOURCE_CACHE)
    prices_raw = PRICE_PATH.read_bytes()
    prices = json.loads(prices_raw)
    if _sha(prices_raw) != selection["targetVersion"]:
        raise ValueError("frozen target snapshot bytes changed")
    baseline_doc = _read_json(BASELINE_PATHS["T"])
    baseline = source_a._baseline_index(baseline_doc, "T")
    if len(baseline) != len(set(baseline)) or not baseline:
        raise ValueError("invalid or empty T RHS baseline key set")
    months = sorted({_month_key(item["origin"]) for item in baseline.values()})
    expected_months = [dt.date(2026, month, 1) for month in (7, 8, 9)]
    if months != expected_months:
        raise ValueError(f"T baseline origins do not cover the frozen monthly fit grid: {months}")
    selected_map = {p: selection["products"][p]["selectedModelId"] for p in PRODUCTS}
    planned_models = {p: set() for p in PRODUCTS}
    for product, selected in selected_map.items():
        if selected != "B":
            planned_models[product].add(selected)
    for product in PRODUCTS:
        if "P" in planned_models[product]:
            planned_models[product].add("L")
        if "A" in planned_models[product]:
            planned_models[product].add("CTRL")
    planned_models["radish"].update(("P", "L"))
    planned_models["onion"].update(("A", "CTRL"))
    planned_count = sum(30 * len(models) for models in planned_models.values())
    if planned_count != selection.get("plannedFitCount"):
        raise ValueError("T model fit plan differs from immutable V selection artifact")
    stored_plan = {p: sorted(selection["plannedFitsByProduct"][p]["models"]) for p in PRODUCTS}
    if stored_plan != {p: sorted(planned_models[p]) for p in PRODUCTS}:
        raise ValueError("T per-product model plan differs from immutable V selection artifact")

    cache_file_sha = _file_sha(source_b.SOURCE_CACHE)
    execution = _execution_identity(config, cache_file_sha, selection_sha)
    execution_sha = execution["executionIdentitySha256"]
    phase_dir = output_dir / "T"
    phase_dir.mkdir(parents=True, exist_ok=True)
    product_records = {str(p.get("id")): p for p in prices.get("products", [])
                       if isinstance(p, Mapping)}
    fit_records: list[dict[str, Any]] = []
    artifact_map: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    model_registry: dict[tuple[str, int, str, str], Any] = {}
    series_info: dict[str, dict[str, Any]] = {}
    training_cache_by_h: dict[tuple[str, int], dict[int, dict[Any, Any]]] = {}
    for product in PRODUCTS:
        product_record = product_records.get(product)
        if not isinstance(product_record, Mapping):
            raise ValueError(f"frozen target snapshot lacks product {product}")
        representative = source_b.modeling._observed_series(product_record)
        series_records, series_by_id = source_b._series_map(channels, channel_selection,
                                                             product, representative)
        series_info[product] = {"representative": representative,
                                "seriesRecords": series_records,
                                "seriesById": series_by_id}
        for horizon in HORIZONS:
            key_cache = training_cache_by_h.setdefault((product, horizon),
                                                       {sid: {} for sid in series_by_id})
            for fit_month in months:
                needed = sorted(planned_models[product])
                if not needed:
                    continue
                fold_started = time.perf_counter()
                fold_failure = None
                rep_rows, panel_rows, per_origin = [], [], {}
                fold_record = None
                loaded: dict[str, Any] = {}
                panel_records = {}
                if any(model_id in ("L", "P") for model_id in needed):
                    try:
                        rep_rows, panel_rows, per_origin = source_b._training_rows(
                            series_by_id, fit_month, horizon,
                            feature_caches=key_cache)
                        if len(rep_rows) < source_b.MIN_TRAIN_ROWS:
                            raise _ExpectedFitFailure(
                                f"representative support {len(rep_rows)} < {source_b.MIN_TRAIN_ROWS}")
                        if "P" in needed:
                            fold_record, panel_models = source_b._train_fold(
                                product, horizon, fit_month, fit_month, rep_rows, panel_rows,
                                per_origin, series_records, config, phase_dir,
                                include_q=False, execution_identity_sha=execution_sha)
                            panel_records = fold_record["modelVariants"]
                            for model_id in ("L", "P"):
                                rec = panel_records.get(model_id)
                                if model_id in needed and rec and rec.get("status") not in ("failed", "not_run"):
                                    rel = rec.get("modelFile")
                                    model_path = output_dir / rel
                                    sidecar = model_path.with_suffix(".identity.json")
                                    identity = _read_json(sidecar).get("identity")
                                    loaded[model_id], model_sha, identity_sha = source_b._load_booster(
                                        model_path, sidecar, identity)
                                    artifact_map[(product, horizon, fit_month.strftime("%Y-%m"), model_id)] = {
                                        "modelId": model_id, "product": product,
                                        "horizon": horizon, "fitMonth": fit_month.strftime("%Y-%m"),
                                        "path": str(rel), "fileSha256": model_sha,
                                        "identityPath": str(sidecar.relative_to(output_dir)),
                                        "identityFileSha256": _file_sha(sidecar),
                                        "identitySha256": identity_sha,
                                        "checkpointIdentity": identity}
                                    panel_models[model_id] = loaded.get(model_id, panel_models.get(model_id))
                            # The native helper always fits L+P; when only L is
                            # scheduled use the dedicated L-only path below.
                            if "L" in needed:
                                loaded["L"] = panel_models.get("L", loaded.get("L"))
                            if "P" in needed:
                                loaded["P"] = panel_models.get("P", loaded.get("P"))
                        else:
                            rec, model, identity = _train_l_only(
                                product, horizon, fit_month, rep_rows, config,
                                phase_dir, execution_sha)
                            rel = rec["modelFile"]
                            path = output_dir / rel
                            sidecar = path.with_suffix(".identity.json")
                            artifact_map[(product, horizon, fit_month.strftime("%Y-%m"), "L")] = {
                                "modelId": "L", "product": product, "horizon": horizon,
                                "fitMonth": fit_month.strftime("%Y-%m"), "path": rel,
                                "fileSha256": rec["modelFileSha256"],
                                "identityPath": str(sidecar.relative_to(output_dir)),
                                "identityFileSha256": _file_sha(sidecar),
                                "identitySha256": rec["checkpointIdentitySha256"],
                                "checkpointIdentity": identity}
                            loaded["L"] = model
                            panel_records["L"] = rec
                    except (lgb.basic.LightGBMError, _ExpectedFitFailure) as exc:
                        fold_failure = f"{type(exc).__name__}: {exc}"
                if any(model_id in ("A", "CTRL") for model_id in needed):
                    try:
                        selected_series = source_a._channel_series(channels, channel_selection, product)
                        train_rows = []
                        feature_cache: dict[Any, Any] = {}
                        for target, (target_price, _) in sorted(representative.items()):
                            if target < source_a.TRAIN_START or target >= fit_month:
                                continue
                            origin = target - dt.timedelta(days=horizon)
                            feature = source_a._feature_row(representative, selected_series,
                                                           origin, target, horizon,
                                                           feature_cache,
                                                           include_path_audit=False)
                            if feature is not None:
                                train_rows.append({**feature, "target": target,
                                                   "origin": origin,
                                                   "label": math.log(target_price / feature["p0"]),
                                                   "targetPrice": target_price})
                        if len(train_rows) < source_a.MIN_TRAIN_ROWS:
                            raise _ExpectedFitFailure(
                                f"A/CTRL support {len(train_rows)} < {source_a.MIN_TRAIN_ROWS}")
                        a_identity = {"product": product, "horizon": horizon,
                                      "fitOrigin": fit_month.isoformat(),
                                      "fitMonth": fit_month.isoformat(),
                                      "sourceCacheSha256": cache_doc.get("cacheSha256"),
                                      "sourceCacheFileSha256": cache_file_sha,
                                      "channelSelectionSha256": _sha(_canonical(channel_selection)),
                                      "targetVersion": selection["targetVersion"],
                                      "executionIdentitySha256": execution_sha}
                        pair_path = phase_dir / "models" / f"{product}-h{horizon:02d}-{fit_month:%Y-%m}-A-CTRL.joblib"
                        ctrl, cand, pair_info = _save_pair_checkpoint(train_rows, pair_path, a_identity)
                        loaded.update({"CTRL": ctrl, "A": cand})
                        pair_info["path"] = str(pair_path.relative_to(output_dir))
                        pair_info["identityPath"] = str(pair_path.with_suffix(".identity.json").relative_to(output_dir))
                        pair_info["identityFileSha256"] = _file_sha(pair_path.with_suffix(".identity.json"))
                        artifact_map[(product, horizon, fit_month.strftime("%Y-%m"), "A/CTRL")] = {
                            "modelIds": ["A", "CTRL"], "product": product, "horizon": horizon,
                            "fitMonth": fit_month.strftime("%Y-%m"),
                            "path": pair_info["path"], "fileSha256": pair_info["fileSha256"],
                            "identityPath": pair_info["identityPath"],
                            "identityFileSha256": pair_info["identityFileSha256"],
                            "identitySha256": pair_info["identitySha256"],
                            "checkpointIdentity": pair_info["identity"]}
                        labels = np.asarray([float(r["label"]) for r in train_rows], dtype=np.float64)
                        pred_ctrl = np.asarray(ctrl.predict([r["basic40"] for r in train_rows]), dtype=np.float64)
                        pred_a = np.asarray(cand.predict([r["candidate"] for r in train_rows]), dtype=np.float64)
                        if not np.isfinite(pred_ctrl).all() or not np.isfinite(pred_a).all():
                            raise ValueError("A/CTRL representative training predictions are nonfinite")
                        diag_ctrl = _representative_diagnostics(train_rows, pred_ctrl)
                        diag_a = _representative_diagnostics(train_rows, pred_a)
                        decomposition = _log_loss_decomposition(labels, pred_ctrl, pred_a)
                        a_train_rows = train_rows
                        for model_id in ("A", "CTRL"):
                            train_info = {"modelId": model_id,
                                          "fitStatus": pair_info["status"],
                                          "status": pair_info["status"],
                                          "fitSeconds": pair_info["fitSeconds"],
                                          "fitTargetCount": pair_info["fitTargetCount"],
                                          "fitTargetSha256": pair_info["fitTargetSha256"],
                                          "fitTrainingRowCount": pair_info["trainingRows"],
                                          "checkpointPath": pair_info["path"],
                                          "checkpointFileSha256": pair_info["fileSha256"],
                                          "checkpointIdentitySha256": pair_info["identitySha256"],
                                          "representativeTrainDiagnostics": (diag_a if model_id == "A" else diag_ctrl),
                                          "representativeTrainLogDecomposition": ({"A-CTRL": decomposition}
                                                                                   if model_id == "A" else {})}
                            fit_records.append({"product": product, "modelId": model_id,
                                                "horizon": horizon,
                                                "fitMonth": fit_month.strftime("%Y-%m"),
                                                "fitOrigin": fit_month.isoformat(),
                                                **_temporal_fit_evidence(train_rows, fit_month,
                                                                         feature_date_key="p0Date"),
                                                **train_info})
                    except (lgb.basic.LightGBMError, _ExpectedFitFailure) as exc:
                        fold_failure = f"{type(exc).__name__}: {exc}"
                        for model_id in ("A", "CTRL"):
                            if model_id in needed:
                                fit_records.append({"product": product, "modelId": model_id,
                                                    "horizon": horizon,
                                                    "fitMonth": fit_month.strftime("%Y-%m"),
                                                    "fitOrigin": fit_month.isoformat(),
                                                    **_temporal_fit_evidence([], fit_month),
                                                    "fitTargetCount": 0,
                                                    "fitTargetSha256": None,
                                                    "fitTrainingRowCount": 0,
                                                    "fitStatus": "failed",
                                                    "fitSeconds": time.perf_counter() - fold_started,
                                                    "failureReason": fold_failure,
                                                    "checkpointPath": None,
                                                    "checkpointFileSha256": None,
                                                    "checkpointIdentitySha256": None,
                                                    "representativeTrainDiagnostics": None,
                                                    "representativeTrainLogDecomposition": {}})
                elapsed_fold = time.perf_counter() - fold_started
                # Persist panel fit diagnostics, including explicit failures.
                if any(model_id in ("L", "P") for model_id in needed):
                    for model_id in sorted(model for model in needed if model in ("L", "P")):
                        panel_record = panel_records.get(model_id, {})
                        status = panel_record.get("status", "failed" if fold_failure else "failed")
                        model_record = panel_record
                        model_path = output_dir / model_record["modelFile"] if model_record.get("modelFile") else None
                        identity_path = model_path.with_suffix(".identity.json") if model_path else None
                        target_rows = panel_rows if model_id == "P" else rep_rows
                        target_digest = _training_target_digest(target_rows) if target_rows else None
                        decomposition = (fold_record or {}).get("representativeTrainLogDecomposition", {})
                        diag = (fold_record or {}).get("representativeTrainDiagnostics", {}).get(model_id)
                        if diag is None:
                            diag = model_record.get("representativeTrainDiagnostics")
                        fit_records.append({"product": product, "modelId": model_id,
                                            "horizon": horizon,
                                            "fitMonth": fit_month.strftime("%Y-%m"),
                                            "fitOrigin": fit_month.isoformat(),
                                            **_temporal_fit_evidence(target_rows, fit_month),
                                            "fitTargetCount": len({r["target"] for r in target_rows}),
                                            "fitTargetSha256": target_digest,
                                            "fitTrainingRowCount": len(target_rows),
                                            "fitStatus": status,
                                            "fitSeconds": elapsed_fold,
                                            "failureReason": fold_failure,
                                            "checkpointPath": model_record.get("modelFile"),
                                            "checkpointFileSha256": model_record.get("modelFileSha256"),
                                            "checkpointIdentitySha256": model_record.get("checkpointIdentitySha256"),
                                            "representativeTrainDiagnostics": diag,
                                            "representativeTrainLogDecomposition": {
                                                "P-L": decomposition.get("P-L") if model_id == "P" else None}})
                        model = loaded.get(model_id)
                        if model is not None:
                            model_registry[(product, horizon, fit_month.strftime("%Y-%m"), model_id)] = model

    # Reopen every saved checkpoint before generating predictions.
    model_registry = _reload_model_artifacts(output_dir, artifact_map)
    selection_by_product = selected_map
    rows = []
    eval_feature_cache: dict[tuple[str, int], dict[Any, Any]] = {}
    a_feature_cache: dict[tuple[str, int], dict[Any, Any]] = {}
    for key, base in sorted(baseline.items(), key=lambda kv: (kv[0][0], kv[1]["origin"], kv[0][2])):
        product, target, horizon = key
        origin = base["origin"]
        month = _month_key(origin).strftime("%Y-%m")
        info = series_info[product]
        representative = info["representative"]
        feature_cache = eval_feature_cache.setdefault((product, horizon), {})
        panel_feature = source_b._series_feature(representative, origin, target, horizon, feature_cache)
        selected_series = source_a._channel_series(channels, channel_selection, product)
        a_cache = a_feature_cache.setdefault((product, horizon), {})
        a_feature = source_a._feature_row(representative, selected_series, origin, target,
                                          horizon, a_cache, include_path_audit=False)
        model_rows: dict[str, dict[str, Any]] = {}
        for model_id in sorted(planned_models[product]):
            model = model_registry.get((product, horizon, month, model_id))
            if model_id in ("L", "P"):
                if panel_feature is None or model is None:
                    result = {"prediction": None, "status": "not_issued",
                              "reason": "model_or_feature_unavailable"}
                else:
                    try:
                        pred = source_b._candidate_prediction(model, panel_feature["features"],
                                                             panel_feature["p0"],
                                                             panel_feature["sourceDate"])
                        result = {"prediction": pred["prediction"], "status": "issued",
                                  "logRatio": pred["logRatio"], "p0": pred["p0"],
                                  "p0Date": pred["p0Date"]}
                    except (ValueError, OverflowError, FloatingPointError) as exc:
                        result = {"prediction": None, "status": "not_issued",
                                  "reason": f"model_output_unusable:{type(exc).__name__}"}
            else:
                a_row = {**a_feature} if a_feature is not None else {}
                if a_feature is not None:
                    a_row["p0Date"] = a_feature["p0Date"]
                feature_name = "candidate" if model_id == "A" else "basic40"
                result = (_forecast_a_model(model, a_row, feature_name=feature_name)
                          if model is not None and a_feature is not None else
                          {"prediction": None, "status": "not_issued",
                           "reason": "model_or_feature_unavailable"})
            model_rows[model_id] = result
        b_value = source_a._finite_positive(base.get("E106_RHS"))
        b_status = "issued" if b_value is not None else "not_issued"
        b = {"prediction": b_value, "status": b_status,
             "sourceStatus": base.get("rhsStatus")}
        selected = selection_by_product[product]
        if selected == "B":
            g = {"prediction": b_value, "status": b_status,
                 "selectedModelId": "B", "rawModelStatus": "not_applicable",
                 "rawModelPrediction": None, "fallbackTo": None,
                 "fallbackReason": None, "modelUsed": False}
        else:
            raw = model_rows.get(selected, {"prediction": None, "status": "not_issued",
                                            "reason": "selected_model_not_fitted"})
            if raw.get("status") == "issued" and _positive(raw.get("prediction")) is not None:
                g = {"prediction": raw["prediction"], "status": "issued",
                     "selectedModelId": selected, "rawModelStatus": "issued",
                     "rawModelPrediction": raw["prediction"], "fallbackTo": None,
                     "fallbackReason": None, "modelUsed": True,
                     "p0": raw.get("p0"), "p0_date": raw.get("p0Date"),
                     "log_ratio": raw.get("logRatio")}
            elif b_status == "issued":
                g = {"prediction": b_value, "status": "fallback",
                     "selectedModelId": selected, "rawModelStatus": "not_issued",
                     "rawModelPrediction": None, "fallbackTo": "B",
                     "fallbackReason": raw.get("reason", "selected_model_unavailable"),
                     "modelUsed": False, "p0": raw.get("p0"),
                     "p0_date": raw.get("p0Date"), "log_ratio": None}
            else:
                g = {"prediction": None, "status": "not_issued",
                     "selectedModelId": selected, "rawModelStatus": "not_issued",
                     "rawModelPrediction": None, "fallbackTo": None,
                     "fallbackReason": raw.get("reason", "selected_model_and_B_unavailable"),
                     "modelUsed": False, "p0": raw.get("p0"),
                     "p0_date": raw.get("p0Date"), "log_ratio": None}
        rows.append({"product": product, "origin": origin.isoformat(),
                     "target": target.isoformat(), "horizon": horizon,
                     "target_version": selection["targetVersion"],
                     "fit_month": month, "cutoff": (origin - dt.timedelta(days=1)).isoformat(),
                     "selected_model_id": selected, "selected_policy_id": f"G:{product}:{selected}",
                     "B": b, "models": model_rows, "G": g})
    rows.sort(key=lambda r: (r["product"], r["origin"], r["horizon"]))
    rows_sha = _sha(_canonical(rows))
    model_artifacts = sorted(artifact_map.values(),
                             key=lambda a: (a["product"], a["fitMonth"], a["horizon"], a.get("modelId", "/".join(a.get("modelIds", [])))))
    source_paths = {"A": source_a.OUT / "V/forecast-manifest.json",
                    "B": source_b.OUT / "V/forecast-manifest.json",
                    "L": source_b.OUT / "V/forecast-manifest.json",
                    "P": source_b.OUT / "V/forecast-manifest.json"}
    sources = {name: {"path": str(path), "fileSha256": _file_sha(path),
                      "manifestSha256": _read_json(path).get("manifestSha256")}
               for name, path in source_paths.items()}
    doc = {"schemaVersion": 1, "experiment": "CROP_SPECIALIST_ASSEMBLY_V1",
           "stage": "T", "phase": "T", "targetValuesAttached": False,
           "targetVersion": selection["targetVersion"],
           "selection": {"path": str(selection_path), "fileSha256": _file_sha(selection_path),
                         "selectionSha256": selection_sha,
                         "selectorCodeSha256": selection.get("selectorCodeSha256")},
           "targetSnapshot": {"path": str(PRICE_PATH), "fileSha256": _file_sha(PRICE_PATH)},
           "pinnedRhsBaseline": {"path": str(BASELINE_PATHS["T"]),
                                 "fileSha256": _file_sha(BASELINE_PATHS["T"]),
                                 "pinnedSha256": BASELINE_SHA256["T"]},
           "sources": sources,
           "resolvedConfig": {"path": str(source_b.OUT / "resolved-config.json"),
                              "fileSha256": _file_sha(source_b.OUT / "resolved-config.json"),
                              "configSha256": config["configSha256"]},
           "sourceCache": {"path": str(source_b.SOURCE_CACHE),
                           "fileSha256": cache_file_sha,
                           "cacheSha256": cache_doc.get("cacheSha256")},
           "preflight": {"path": str(source_b.OUT / "preflight.json"),
                         "fileSha256": _file_sha(source_b.OUT / "preflight.json"),
                         "preflightSha256": preflight.get("preflightSha256")},
           "executionIdentity": execution,
           "executionIdentitySha256": execution_sha,
           "selectionMap": selected_map,
           "plannedModelIdsByProduct": {p: sorted(planned_models[p]) for p in PRODUCTS},
           "plannedFitCount": planned_count,
           "actualEstimatorFitCount": sum(1 for record in fit_records
                                          if record.get("fitStatus") in ("fit", "reused")),
           "newEstimatorFitCount": sum(1 for record in fit_records
                                       if record.get("fitStatus") == "fit"),
           "fitDiagnostics": fit_records, "modelArtifacts": model_artifacts,
           "forecasts": rows, "forecastRowsSha256": rows_sha,
           "forecastKeyCount": len(rows),
           "replayVerificationPath": "T/replay-verification.json"}
    doc["manifestSha256"] = _sha(_canonical(doc))
    forecasts_path = phase_dir / "forecasts.json"
    _write_once(forecasts_path, doc)
    replay = verify_saved_t_replay(output_dir=output_dir, forecasts_path=forecasts_path)
    return {"status": "forecasts_saved_and_replayed", "forecastsPath": str(forecasts_path),
            "manifestSha256": doc["manifestSha256"], "forecastRowsSha256": rows_sha,
            "replayVerificationSha256": replay["replayVerificationSha256"],
            "forecastKeyCount": len(rows), "plannedFitCount": planned_count,
            "actualEstimatorFitCount": doc["actualEstimatorFitCount"],
            "newEstimatorFitCount": doc["newEstimatorFitCount"]}


def _reload_model_artifacts(output_dir: Path, artifacts: Mapping[Any, Mapping[str, Any]]) -> dict[tuple[str, int, str, str], Any]:
    import joblib
    registry = {}
    for artifact in artifacts.values():
        path = output_dir / artifact["path"]
        if not path.is_file() or _file_sha(path) != artifact["fileSha256"]:
            raise ValueError(f"checkpoint artifact file hash mismatch: {path}")
        sidecar = output_dir / artifact["identityPath"]
        if not sidecar.is_file() or _file_sha(sidecar) != artifact["identityFileSha256"]:
            raise ValueError(f"checkpoint identity sidecar hash mismatch: {sidecar}")
        side = _read_json(sidecar)
        if (side.get("identity") != artifact["checkpointIdentity"]
                or side.get("identitySha256") != artifact["identitySha256"]):
            raise ValueError(f"checkpoint identity mismatch: {sidecar}")
        product, horizon, fit_month = artifact["product"], artifact["horizon"], artifact["fitMonth"]
        if artifact.get("modelIds"):
            bundle = joblib.load(path)
            if bundle.get("checkpointIdentity") != artifact["checkpointIdentity"]:
                raise ValueError(f"pair checkpoint embedded identity mismatch: {path}")
            for model_id in artifact["modelIds"]:
                registry[(product, horizon, fit_month, model_id)] = bundle["CAND_A" if model_id == "A" else "CTRL_LGB"]
        else:
            model_id = artifact["modelId"]
            model, _, _ = source_b._load_booster(path, sidecar,
                                                  artifact["checkpointIdentity"])
            registry[(product, horizon, fit_month, model_id)] = model
    return registry


def verify_saved_t_replay(*, output_dir: Path = OUT,
                          forecasts_path: Path | None = None) -> dict[str, Any]:
    """Reload the saved forecast/checkpoint artifacts and replay without labels."""
    _enforce_thread_limits()
    path = forecasts_path or output_dir / "T/forecasts.json"
    doc = _read_json(path)
    unsigned = dict(doc)
    copied_manifest_sha = unsigned.pop("manifestSha256", None)
    if copied_manifest_sha != _sha(_canonical(unsigned)):
        raise ValueError("T forecast manifest self-hash mismatch")
    rows = doc.get("forecasts")
    if not isinstance(rows, list) or doc.get("forecastRowsSha256") != _sha(_canonical(rows)):
        raise ValueError("T forecast rows hash mismatch")
    if doc.get("targetValuesAttached") is not False:
        raise ValueError("T replay input improperly contains target labels")
    execution = dict(doc.get("executionIdentity", {}))
    execution_sha = execution.pop("executionIdentitySha256", None)
    if (execution_sha != doc.get("executionIdentitySha256")
            or execution_sha != _sha(_canonical(execution))):
        raise ValueError("T execution identity self-hash mismatch")
    if execution.get("sourceHashes") != _source_sha_map():
        raise ValueError("T replay source code/dependency hashes differ from the frozen execution")
    selection_path = Path(doc["selection"]["path"])
    selection = _read_json(selection_path)
    if (_file_sha(selection_path) != doc["selection"]["fileSha256"]
            or selection.get("selectionSha256") != doc["selection"]["selectionSha256"]):
        raise ValueError("T replay selection artifact changed")
    cache_path = Path(doc["sourceCache"]["path"])
    if _file_sha(cache_path) != doc["sourceCache"]["fileSha256"]:
        raise ValueError("T replay source cache changed")
    config_path = Path(doc["resolvedConfig"]["path"])
    if _file_sha(config_path) != doc["resolvedConfig"]["fileSha256"]:
        raise ValueError("T replay resolved config changed")
    baseline_path = Path(doc["pinnedRhsBaseline"]["path"])
    if _file_sha(baseline_path) != doc["pinnedRhsBaseline"]["fileSha256"]:
        raise ValueError("T replay RHS baseline changed")
    for source in doc["sources"].values():
        source_path = Path(source["path"])
        source_doc = _read_json(source_path)
        _verify_manifest_self_hash(source_doc, "T replay V source")
        if (_file_sha(source_path) != source["fileSha256"]
                or source_doc.get("manifestSha256") != source["manifestSha256"]):
            raise ValueError("T replay V source changed")
    prices = json.loads(Path(doc["targetSnapshot"]["path"]).read_text(encoding="utf-8"))
    if _file_sha(Path(doc["targetSnapshot"]["path"])) != doc["targetSnapshot"]["fileSha256"]:
        raise ValueError("T replay target snapshot changed")
    baseline = source_a._baseline_index(_read_json(baseline_path), "T")
    channels, channel_selection = source_a.load_source_cache(cache_path)
    price_records = {str(p.get("id")): p for p in prices.get("products", [])}
    registry = _reload_model_artifacts(output_dir, {
        (i,): item for i, item in enumerate(doc["modelArtifacts"])})
    selection_map = doc["selectionMap"]
    seen = set()
    replayed = 0
    for row in rows:
        product = row["product"]
        origin, target = _date(row["origin"]), _date(row["target"])
        horizon = int(row["horizon"])
        month = row["fit_month"]
        key = (product, target, horizon)
        if key in seen or key not in baseline:
            raise ValueError(f"T replay found duplicate or unexpected forecast key: {key}")
        seen.add(key)
        pinned = baseline[key]
        if pinned["origin"] != origin or row["target_version"] != doc["targetVersion"]:
            raise ValueError(f"T forecast exact key or target version mismatch: {key}")
        if (month != origin.strftime("%Y-%m")
                or row.get("cutoff") != (origin - dt.timedelta(days=1)).isoformat()):
            raise ValueError(f"T forecast fit month or feature cutoff differs from frozen monthly rule: {key}")
        expected_b = source_a._finite_positive(pinned.get("E106_RHS"))
        stored_b = row["B"]
        if stored_b.get("prediction") != expected_b:
            raise ValueError(f"T copied B prediction differs from pinned raw RHS: {key}")
        model_feature_cache = {}
        product_record = price_records[product]
        representative = source_b.modeling._observed_series(product_record)
        panel_feature = source_b._series_feature(representative, origin, target, horizon,
                                                model_feature_cache)
        selected_series = source_a._channel_series(channels, channel_selection, product)
        a_feature = source_a._feature_row(representative, selected_series, origin, target,
                                          horizon, {}, include_path_audit=False)
        for model_id, stored in row["models"].items():
            if stored.get("status") != "issued":
                if stored.get("prediction") is not None:
                    raise ValueError(f"unissued model retains a prediction: {key}/{model_id}")
                continue
            model = registry.get((product, horizon, month, model_id))
            if model is None:
                raise ValueError(f"issued T direct prediction has no saved checkpoint: {key}/{model_id}")
            if model_id in ("L", "P"):
                if panel_feature is None:
                    raise ValueError(f"issued panel prediction lacks replay feature: {key}/{model_id}")
                direct = source_b._candidate_prediction(model, panel_feature["features"],
                                                       panel_feature["p0"], panel_feature["sourceDate"])
                prediction, ratio = direct["prediction"], direct["logRatio"]
            else:
                if a_feature is None:
                    raise ValueError(f"issued A/CTRL prediction lacks replay feature: {key}/{model_id}")
                feature_name = "candidate" if model_id == "A" else "basic40"
                result = _forecast_a_model(model, a_feature, feature_name=feature_name)
                prediction, ratio = result.get("prediction"), result.get("logRatio")
            if prediction is None or not math.isclose(float(stored["prediction"]),
                                                       float(prediction), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"T direct checkpoint replay differs: {key}/{model_id}")
            replayed += 1
        selected = selection_map[product]
        g = row["G"]
        if selected == "B":
            if g.get("prediction") != expected_b or g.get("status") != stored_b.get("status"):
                raise ValueError(f"T B-selected policy differs from pinned RHS: {key}")
        elif g.get("modelUsed") is True:
            direct = row["models"].get(selected, {})
            if direct.get("status") != "issued" or direct.get("prediction") != g.get("prediction"):
                raise ValueError(f"T G direct prediction differs from selected model: {key}")
        elif g.get("fallbackTo") == "B":
            if g.get("prediction") != expected_b or g.get("status") != "fallback":
                raise ValueError(f"T G fallback differs from pinned RHS: {key}")
        elif g.get("status") != "not_issued":
            raise ValueError(f"T G has invalid technical status: {key}")
    if seen != set(baseline):
        raise ValueError("T replay forecast grid differs from pinned RHS key set")
    output_files = [{"path": item["path"], "fileSha256": item["fileSha256"],
                     "identityPath": item["identityPath"],
                     "identityFileSha256": item["identityFileSha256"],
                     "identitySha256": item["identitySha256"]}
                    for item in doc["modelArtifacts"]]
    result = {"schemaVersion": 1, "experiment": "CROP_SPECIALIST_ASSEMBLY_V1",
              "phase": "T", "status": "pass",
              "forecastsFileSha256": _file_sha(path),
              "manifestSha256": copied_manifest_sha,
              "forecastRowsSha256": doc["forecastRowsSha256"],
              "selectionFileSha256": _file_sha(selection_path),
              "selectionSha256": selection["selectionSha256"],
              "executionIdentitySha256": doc["executionIdentitySha256"],
              "sourceCacheFileSha256": doc["sourceCache"]["fileSha256"],
              "resolvedConfigFileSha256": doc["resolvedConfig"]["fileSha256"],
              "baselineFileSha256": doc["pinnedRhsBaseline"]["fileSha256"],
              "targetSnapshotFileSha256": doc["targetSnapshot"]["fileSha256"],
              "sourceManifestHashes": {name: source["fileSha256"] for name, source in doc["sources"].items()},
              "checkpointArtifacts": output_files,
              "forecastKeysVerified": len(seen), "directPredictionsReplayed": replayed,
              "answerLabelsJoined": False}
    result["replayVerificationSha256"] = _sha(_canonical(result))
    _write_once(output_dir / "T/replay-verification.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select-V", "run-T", "replay-T"), default="select-V")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.stage == "select-V":
        result = select_v(output_dir=args.output_dir)
    elif args.stage == "run-T":
        result = run_t(output_dir=args.output_dir)
    else:
        result = verify_saved_t_replay(output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
