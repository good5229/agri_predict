"""Conditional residual experiment over frozen REPAIR_V2 forecasts.

The workflow has three explicit stages. ``fit-select`` fits on C labels and
uses D6 labels through June 20 only for selection; ``forecast-T`` only reads
that frozen state and answer-free forecasts; ``score-T`` is the only stage that
opens the target snapshot for July-September labels. No baseline is fitted or
invoked by this module.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
import sklearn
from sklearn.tree import DecisionTreeRegressor

from . import crop_specialist_assembly_score as inference
from . import full_history_return_eval as price_helper
from . import research_progress_diagnostic as diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/conditional-baseline-residual-v1-20261013"
PROTOCOL_SHA256 = "17f5f3fc6da0bff53668812e0295a86ef989bb0d810919f0e71514bb4e6a087d"
V2_FORECAST_PATH = "outputs/experiments/anchored-increment-repair-v2-20261013/forecasts.json"
V2_FORECAST_SHA256 = "ba052658a8eab70c8c52b989f7c938a28455e92b8b73dd54c82753d8a7d3da88"
V2_MANIFEST_SHA256 = "58e9af7ed6b07ddce8134425f301b3d443c71af9f4e6b7bd0d9299cdcd45d429"
K1_PATHS = {
    "V": "outputs/experiments/anchored-increment-diagnostic-v1-20261013/V/forecasts.json",
    "T": "outputs/experiments/anchored-increment-diagnostic-v1-20261013/T/forecasts.json",
}
K1_FILE_SHA256 = {
    "V": "c0af6b4a20b4354046f9c02b102563244928e9929ae01e3ae43be89304576b09",
    "T": "6ca9daf6f0b1c2fe787c202bd2f03b697603bb1abc38a6ebded894db202d6afd",
}
PRICE_PATH = "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
MODEL_PRODUCTS = ("radish", "onion")
HORIZONS = tuple(range(1, 11))
C_START, C_END = dt.date(2026, 4, 2), dt.date(2026, 5, 31)
D6_START, D6_END, D6_ORIGIN = dt.date(2026, 6, 2), dt.date(2026, 6, 20), dt.date(2026, 6, 1)
T_START, T_END, T_ORIGIN = dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1)
PRICE_FLOOR_RATIO = 1e-8
SOLVER_METHOD = "highs"
SOLVER_OPTIONS = {"presolve": True, "primal_feasibility_tolerance": 1e-9,
                  "dual_feasibility_tolerance": 1e-9, "ipm_optimality_tolerance": 1e-10}
OBJECTIVE_FACE_TOL = 1e-10
VERIFY_ABS_TOL = 1e-6
REPLAY_TOL = 1e-12
TREE_CONFIG = {"criterion": "absolute_error", "splitter": "best", "max_depth": 1,
               "min_samples_leaf": 150, "max_features": None, "random_state": 0,
               "ccp_alpha": 0.0}
FEATURES = {"R0": ("z1_horizon_over_10", "z2_D_minus_B_over_B"),
            "Rdelta": ("z1_horizon_over_10", "z2_D_minus_B_over_B", "z3_delta_over_B")}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _doc_sha(doc: Mapping[str, Any], field: str) -> str:
    unsigned = dict(doc)
    expected = unsigned.pop(field, None)
    actual = _sha(_canonical(unsigned))
    if expected != actual:
        raise ValueError(f"{field} mismatch")
    return actual


def _path(value: str | Path) -> Path:
    p = Path(value)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}") from None
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"expected JSON object: {path}")
    return doc


def _num(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} cannot be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        product = row["product"]
        origin = dt.date.fromisoformat(row["origin"])
        target = dt.date.fromisoformat(row["target"])
        horizon = row["horizon"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed forecast key") from exc
    if product not in PRODUCTS or horizon not in HORIZONS or origin != target - dt.timedelta(days=horizon):
        raise ValueError("forecast key origin/target/horizon mismatch")
    return product, origin.isoformat(), target.isoformat(), horizon


def _expected_grid(start: dt.date, end: dt.date, origin_start: dt.date) -> set[tuple]:
    return {(p, (target - dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
            for p in PRODUCTS for h in HORIZONS
            for i in range((end - start).days + 1)
            for target in (start + dt.timedelta(days=i),)
            if target - dt.timedelta(days=h) >= origin_start}


def _index_rows(rows: Any, label: str) -> dict[tuple, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows missing")
    indexed: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"malformed {label} row")
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate {label} forecast key")
        if any(k in row for k in ("actual", "actualPrice", "targetValue")):
            raise ValueError(f"{label} source unexpectedly contains a target value")
        indexed[key] = row
    return indexed


def _load_sources() -> tuple[dict[str, Any], dict[str, dict[tuple, dict[str, Any]]], dict[str, Any]]:
    protocol = _path("outputs/experiments/conditional-baseline-residual-v1-20261013/protocol.md")
    if _sha(protocol.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("frozen protocol hash changed")
    v2_path = _path(V2_FORECAST_PATH)
    v2_raw = v2_path.read_bytes()
    if _sha(v2_raw) != V2_FORECAST_SHA256:
        raise ValueError("pinned REPAIR_V2 forecast file hash mismatch")
    v2 = _read(v2_path)
    if _doc_sha(v2, "manifestSha256") != V2_MANIFEST_SHA256:
        raise ValueError("REPAIR_V2 manifest identity mismatch")
    if (v2.get("experiment") != "ANCHORED_INCREMENT_REPAIR_V2"
            or v2.get("targetValuesAttached") is not False
            or v2.get("targetSnapshotSha256") != PRICE_SHA256):
        raise ValueError("expected immutable, answer-free REPAIR_V2 forecasts")
    phases: dict[str, dict[tuple, dict[str, Any]]] = {}
    source_docs: dict[str, Any] = {}
    for phase in ("V", "T"):
        path = _path(K1_PATHS[phase])
        raw = path.read_bytes()
        if _sha(raw) != K1_FILE_SHA256[phase]:
            raise ValueError(f"pinned underlying K1/{phase} forecast file hash mismatch")
        doc = _read(path)
        _doc_sha(doc, "manifestSha256")
        if (doc.get("experiment") != "ANCHORED_INCREMENT_DIAGNOSTIC_V1"
                or doc.get("phase") != phase or doc.get("targetValuesAttached") is not False
                or doc.get("targetSnapshotSha256") != PRICE_SHA256):
            raise ValueError(f"unexpected underlying K1/{phase} source")
        rows = doc.get("forecasts")
        if _sha(_canonical(rows)) != doc.get("forecastRowsSha256"):
            raise ValueError(f"K1/{phase} row digest mismatch")
        indexed = _index_rows(rows, f"K1/{phase}")
        if phase == "V":
            expected = _expected_grid(C_START, D6_END, dt.date(2026, 4, 1))
            # V source spans April through June 20 and includes C and D6.
            if set(indexed) != expected:
                raise ValueError("underlying K1/V forecast grid differs from C+D6 interval")
        phases[phase] = indexed
        source_docs[phase] = {"path": K1_PATHS[phase], "fileSha256": K1_FILE_SHA256[phase],
                              "manifestSha256": doc["manifestSha256"],
                              "forecastRowsSha256": doc["forecastRowsSha256"]}
    for name, phase_key in (("C", "C_training"), ("D6", "June_development"), ("T", "T_final_evaluation")):
        obj = v2.get("phaseRows", {}).get(phase_key)
        if not isinstance(obj, dict):
            raise ValueError(f"REPAIR_V2 {phase_key} rows missing")
        v2_rows = _index_rows(obj.get("forecasts"), f"REPAIR_V2/{name}")
        if obj.get("count") != len(v2_rows):
            raise ValueError(f"REPAIR_V2/{name} row count mismatch")
        underlying = phases["T" if name == "T" else "V"]
        expected = _expected_grid(*( (C_START, C_END, dt.date(2026, 4, 1)) if name == "C"
                                     else (D6_START, D6_END, D6_ORIGIN) if name == "D6"
                                     else (T_START, T_END, T_ORIGIN) ))
        if set(v2_rows) != expected:
            raise ValueError(f"REPAIR_V2/{name} exact forecast grid mismatch")
        for key, row in v2_rows.items():
            original = underlying.get(key)
            if original is None:
                raise ValueError(f"REPAIR_V2/{name} key absent from underlying K1 forecast")
            prov = row.get("sourceProvenance")
            if (not isinstance(prov, Mapping) or prov.get("sourceRowSha256") != _sha(_canonical(original))
                    or prov.get("sourceKey") != {"product": key[0], "origin": key[1], "target": key[2], "horizon": key[3]}):
                raise ValueError(f"REPAIR_V2/{name} source row provenance mismatch")
            cutoff = (dt.date.fromisoformat(key[1]) - dt.timedelta(days=1)).isoformat()
            if prov.get("informationBoundary", {}).get("requiredAvailableThrough") != cutoff:
                raise ValueError(f"REPAIR_V2/{name} origin cutoff provenance mismatch")
            b = _num(row.get("B", {}).get("prediction"), f"{name}/B", positive=True)
            if row.get("B", {}).get("status") != "issued":
                raise ValueError(f"REPAIR_V2/{name} baseline status is not issued")
            # Only radish and onion consume D. The immutable source has no
            # directD field for the fixed-B crops, so do not invent one.
            d = (_num(original.get("directD"), f"{name}/D", positive=True)
                 if key[0] in MODEL_PRODUCTS else None)
            delta = _num(row.get("delta"), f"{name}/delta")
            if not math.isclose(b, _num(original.get("baselineB"), "source B", positive=True), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"REPAIR_V2/{name} B differs from pinned source")
            if not math.isclose(delta, _num(original.get("delta"), "source delta"), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"REPAIR_V2/{name} delta differs from pinned source")
            want_definition = "P-L" if key[0] == "radish" else "A-CTRL" if key[0] == "onion" else None
            if want_definition and original.get("deltaDefinition") != want_definition:
                raise ValueError(f"{name}/{key[0]} direct delta definition changed")
            if key[0] == "radish" and not math.isclose(d, _num(original["sourcePredictions"]["L"], "L"), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("radish D is not L")
            if key[0] == "onion" and not math.isclose(d, _num(original["sourcePredictions"]["CTRL"], "CTRL"), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("onion D is not CTRL")
            row["_B"], row["_D"], row["_delta"], row["_key"] = b, d, delta, key
        phases[name] = v2_rows
    sources = {"REPAIR_V2": {"path": V2_FORECAST_PATH, "fileSha256": V2_FORECAST_SHA256,
                             "manifestSha256": v2["manifestSha256"],
                             "forecastsSha256": v2["forecastsSha256"]},
               **{f"K1_{k}": v for k, v in source_docs.items()}}
    return v2, phases, sources


def _limited_actuals(prices_path: str | Path, through: dt.date) -> tuple[dict[tuple[str, str], float], str]:
    path = _path(prices_path)
    raw = path.read_bytes()
    if _sha(raw) != PRICE_SHA256:
        raise ValueError("pinned target snapshot file hash mismatch")
    series = price_helper._series(json.loads(raw))
    # Only expose protocol-authorized dates to the caller. T labels are never returned here.
    return {(p, day.isoformat()): float(value)
            for p, values in series.items() for day, value in values.items()
            if day <= through and value > 0}, PRICE_SHA256


def _stage_join(rows: Mapping[tuple, dict[str, Any]], actuals: Mapping[tuple[str, str], float],
                start: dt.date, end: dt.date, origin_start: dt.date) -> list[dict[str, Any]]:
    out = []
    for key, row in sorted(rows.items()):
        product, origin, target, horizon = key
        day = dt.date.fromisoformat(target)
        if day < start or day > end or dt.date.fromisoformat(origin) < origin_start:
            continue
        actual = actuals.get((product, target))
        if actual is None:
            continue
        out.append({"product": product, "origin": origin, "target": target, "horizon": horizon,
                    "actual": actual, "B": row["_B"], "D": row["_D"], "delta": row["_delta"],
                    "sourceRowSha256": row["sourceProvenance"]["sourceRowSha256"]})
    return out


def _feature(row: Mapping[str, Any], model_name: str) -> list[float]:
    b = _num(row["B"], "B", positive=True)
    d = _num(row["D"], "D", positive=True)
    delta = _num(row["delta"], "delta")
    base = [int(row["horizon"]) / 10.0, (d - b) / b]
    if model_name == "Rdelta":
        base.append(delta / b)
    elif model_name != "R0":
        raise ValueError("unknown residual model")
    if not all(math.isfinite(x) for x in base):
        raise ValueError("nonfinite residual feature")
    return base


def _training_checks(records: list[dict[str, Any]], product: str) -> dict[str, Any]:
    if not records:
        raise ValueError(f"no positive C training rows for {product}")
    per_target = Counter(r["target"] for r in records)
    if max(per_target.values()) > 10:
        raise ValueError(f"C has more than 10 forecast rows per target day for {product}")
    by_horizon = Counter(r["horizon"] for r in records)
    if set(by_horizon) != set(HORIZONS):
        raise ValueError(f"C lacks positive support for a horizon in {product}")
    return {"rows": len(records), "uniqueTargetDays": len(per_target),
            "maxRowsPerTargetDay": max(per_target.values()),
            "horizonCounts": {str(h): by_horizon[h] for h in HORIZONS},
            "weightsByHorizon": {str(h): 1.0 / (10 * by_horizon[h]) for h in HORIZONS}}


def _native_leaf(model: Mapping[str, Any], values: list[float]) -> int:
    x = np.asarray(values, dtype=np.float32)
    left, right, feature, threshold = (model[k] for k in ("childrenLeft", "childrenRight", "feature", "threshold"))
    node = 0
    while left[node] != -1:
        col = int(feature[node])
        node = int(left[node] if x[col] <= float(threshold[node]) else right[node])
    return node


def _restore_native_model(model: Mapping[str, Any]) -> DecisionTreeRegressor:
    """Restore the sklearn Cython tree state, then use its native float32 apply."""
    from sklearn.tree import _tree

    native = model.get("nativeTreeState")
    if not isinstance(native, Mapping):
        raise ValueError("saved native sklearn tree state is missing")
    n_features = len(model.get("features", ()))
    if not n_features:
        raise ValueError("saved model has no features")
    nodes = np.zeros(int(native["nodeCount"]), dtype=_tree.NODE_DTYPE)
    for i, obj in enumerate(native["nodes"]):
        for field in _tree.NODE_DTYPE.names:
            nodes[field][i] = obj[field]
    state = {"max_depth": int(native["maxDepth"]), "node_count": int(native["nodeCount"]),
             "nodes": nodes, "values": np.asarray(native["values"], dtype=np.float64)}
    tree = _tree.Tree(n_features, np.ones(n_features, dtype=np.intp), 1)
    tree.__setstate__(state)
    estimator = DecisionTreeRegressor(**TREE_CONFIG)
    estimator.tree_ = tree
    estimator.n_features_in_ = n_features
    estimator.n_outputs_ = 1
    estimator.n_classes_ = 1
    return estimator


def _model_g(model: Mapping[str, Any], feature_row: Mapping[str, Any],
             native: DecisionTreeRegressor | None = None) -> tuple[float, int]:
    x = _feature(feature_row, model["modelId"])
    estimator = native or _restore_native_model(model)
    leaf = int(estimator.apply(np.asarray([x], dtype=np.float64))[0])
    b = _num(model.get("leafCorrections", {}).get(str(leaf)), "leaf correction")
    return b, leaf


def _fit_one(records: list[dict[str, Any]], product: str, model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from scipy.optimize import linprog

    checks = _training_checks(records, product)
    x = np.asarray([_feature({**r, "B": r["B"], "D": r["D"], "delta": r["delta"]}, model_name) for r in records], dtype=np.float64)
    b = np.asarray([r["B"] for r in records], dtype=np.float64)
    y = np.asarray([r["actual"] for r in records], dtype=np.float64)
    residual = (y - b) / b
    w = np.asarray([1.0 / (10 * checks["horizonCounts"][str(r["horizon"])]) for r in records], dtype=np.float64)
    sample_weight = w * b / y
    estimator = DecisionTreeRegressor(**TREE_CONFIG)
    estimator.fit(x, residual, sample_weight=sample_weight)
    train_leaves = estimator.apply(x)
    n_nodes = int(estimator.tree_.node_count)
    leaf_ids = sorted(int(i) for i in np.unique(train_leaves))
    by_leaf: dict[int, list[int]] = {leaf: [i for i, value in enumerate(train_leaves) if int(value) == leaf] for leaf in leaf_ids}
    leaf_days = {leaf: sorted({records[i]["target"] for i in indices}) for leaf, indices in by_leaf.items()}
    for leaf in leaf_ids:
        if len(by_leaf[leaf]) < TREE_CONFIG["min_samples_leaf"]:
            raise ValueError(f"{product}/{model_name} leaf violates min_samples_leaf")
        if len(leaf_days[leaf]) < 15:
            raise ValueError(f"{product}/{model_name} leaf has fewer than 15 unique target days")
    children_left = [int(v) for v in estimator.tree_.children_left]
    children_right = [int(v) for v in estimator.tree_.children_right]
    feature = [int(v) for v in estimator.tree_.feature]
    thresholds = [float(v) if math.isfinite(float(v)) else None for v in estimator.tree_.threshold]
    for i in range(n_nodes):
        if (children_left[i] == -1) != (children_right[i] == -1):
            raise ValueError("malformed sklearn tree structure")
    raw_tree = {"modelId": model_name, "product": product, "features": list(FEATURES[model_name]),
                "childrenLeft": children_left, "childrenRight": children_right, "feature": feature,
                "threshold": thresholds, "nNodeSamples": [int(v) for v in estimator.tree_.n_node_samples],
                "weightedNodeSamples": [float(v) for v in estimator.tree_.weighted_n_node_samples],
                "splitter": TREE_CONFIG["splitter"], "maxDepth": TREE_CONFIG["max_depth"],
                "minSamplesLeaf": TREE_CONFIG["min_samples_leaf"], "randomState": TREE_CONFIG["random_state"],
                "criterion": TREE_CONFIG["criterion"], "ccpAlpha": TREE_CONFIG["ccp_alpha"],
                "sklearnVersion": sklearn.__version__}
    native_state = estimator.tree_.__getstate__()
    raw_tree["nativeTreeState"] = {
        "maxDepth": int(native_state["max_depth"]), "nodeCount": int(native_state["node_count"]),
        "nodes": [{name: (int(native_state["nodes"][i][name]) if name in
                            ("left_child", "right_child", "feature", "n_node_samples", "missing_go_to_left")
                            else float(native_state["nodes"][i][name]))
                   for name in native_state["nodes"].dtype.names}
                  for i in range(int(native_state["node_count"]))],
        "values": native_state["values"].tolist()}
    for i, values in enumerate(x):
        saved_leaf = _native_leaf(raw_tree, values.tolist())
        if saved_leaf != int(train_leaves[i]):
            raise ValueError("saved float32 tree routing differs from sklearn apply on C")
    active_leaves = leaf_ids
    leaf_slot = {leaf: j for j, leaf in enumerate(active_leaves)}
    n = len(records)
    k = len(active_leaves)
    # Variables [b_leaf..., t_i...]. The objective is exact cell-balanced MAPE up to 100.
    objective = np.zeros(k + n, dtype=np.float64)
    for i in range(n):
        objective[k + i] = w[i] * b[i] / y[i]
    aub: list[np.ndarray] = []
    bub: list[float] = []
    base_mae = math.fsum(float(wi) * abs(float(yi - bi)) for wi, yi, bi in zip(w, y, b))
    for i, (leaf, r) in enumerate(zip(train_leaves, residual)):
        j = leaf_slot[int(leaf)]
        first = np.zeros(k + n); first[j] = -1.; first[k+i] = -1.
        aub.append(first); bub.append(-float(r))       # t >= r - correction
        second = np.zeros(k + n); second[j] = 1.; second[k+i] = -1.
        aub.append(second); bub.append(float(r))       # t >= correction - r
    mae_row = np.zeros(k + n)
    for i in range(n):
        mae_row[k+i] = w[i] * b[i]
    aub.append(mae_row); bub.append(base_mae)
    bounds = [(-1. + PRICE_FLOOR_RATIO, None)] * k + [(0., None)] * n
    primary = linprog(objective, A_ub=np.asarray(aub), b_ub=np.asarray(bub), bounds=bounds,
                      method=SOLVER_METHOD, options=dict(SOLVER_OPTIONS))
    if not primary.success or primary.x is None:
        raise RuntimeError(f"{product}/{model_name} primary LP failed: {primary.status} {primary.message}")
    primary_obj = float(primary.fun)
    # Preserve the inherited optimal-face tie rule: minimum sum |b_leaf| within tolerance.
    m2 = k + n + k
    obj2 = np.zeros(m2); obj2[k+n:] = 1.
    aub2, bub2 = [], []
    for row, rhs in zip(aub, bub):
        new = np.zeros(m2); new[:k+n] = row; aub2.append(new); bub2.append(rhs)
    face = np.zeros(m2); face[k:k+n] = objective[k:]
    aub2.append(face); bub2.append(primary_obj + OBJECTIVE_FACE_TOL)
    for j in range(k):
        pos = np.zeros(m2); pos[j] = 1.; pos[k+n+j] = -1.
        neg = np.zeros(m2); neg[j] = -1.; neg[k+n+j] = -1.
        aub2.extend((pos, neg)); bub2.extend((0., 0.))
    bounds2 = [(-1. + PRICE_FLOOR_RATIO, None)] * k + [(0., None)] * n + [(0., None)] * k
    tie = linprog(obj2, A_ub=np.asarray(aub2), b_ub=np.asarray(bub2), bounds=bounds2,
                  method=SOLVER_METHOD, options=dict(SOLVER_OPTIONS))
    if not tie.success or tie.x is None:
        raise RuntimeError(f"{product}/{model_name} tie-break LP failed: {tie.status} {tie.message}")
    correction = [float(x) for x in tie.x[:k]]
    new_mae = math.fsum(float(wi) * abs(float(yi - bi * (1. + correction[leaf_slot[int(leaf)]])))
                        for wi, yi, bi, leaf in zip(w, y, b, train_leaves))
    tie_obj = math.fsum(float(wi) * float(bi) / float(yi)
                        * abs(float(ri) - correction[leaf_slot[int(leaf)]])
                        for wi, bi, yi, ri, leaf in zip(w, b, y, residual, train_leaves))
    objective_gap = tie_obj - primary_obj
    if new_mae - base_mae > VERIFY_ABS_TOL:
        raise RuntimeError(f"{product}/{model_name} fitted weighted MAE exceeds B constraint")
    if objective_gap < -1e-9 or objective_gap > OBJECTIVE_FACE_TOL + 1e-9:
        raise RuntimeError(f"{product}/{model_name} tie solution left the recorded primary objective face")
    if min(1. + v for v in correction) < PRICE_FLOOR_RATIO - 1e-9:
        raise RuntimeError(f"{product}/{model_name} fitted correction violates price floor")
    raw_tree["leafCorrections"] = {str(leaf): correction[leaf_slot[leaf]] for leaf in active_leaves}
    raw_tree["leafDiagnostics"] = {str(leaf): {"rows": len(by_leaf[leaf]), "uniqueTargetDays": len(leaf_days[leaf]),
                                                  "targetDaySha256": _sha(_canonical(leaf_days[leaf]))}
                                   for leaf in active_leaves}
    raw_tree["trainingReplay"] = [{"key": [records[i]["product"], records[i]["origin"], records[i]["target"],
                                                records[i]["horizon"]],
                                   "features": x[i].tolist(), "leafId": int(train_leaves[i])}
                                  for i in range(n)]
    restored = _restore_native_model(raw_tree)
    replay_leaves = restored.apply(x)
    if not np.array_equal(replay_leaves, train_leaves):
        raise ValueError("serialized sklearn native tree does not replay C routing")
    diagnostics = {"product": product, "modelId": model_name, "logicalTreeFits": 1, "logicalLpFits": 1,
                  "trainingRows": n, "uniqueTargetDays": len({r["target"] for r in records}),
                  "maxRowsPerTargetDay": checks["maxRowsPerTargetDay"],
                  "horizonCounts": checks["horizonCounts"], "weightsByHorizon": checks["weightsByHorizon"],
                  "targetValuesSha256": _sha(_canonical([[r["target"], r["actual"]] for r in records])),
                  "features": list(FEATURES[model_name]), "featureCount": len(FEATURES[model_name]),
                  "split": {"nodeCount": n_nodes, "leafIds": active_leaves,
                            "threshold": raw_tree["threshold"], "feature": raw_tree["feature"]},
                  "lp": {"method": SOLVER_METHOD, "options": SOLVER_OPTIONS,
                         "primarySuccess": bool(primary.success), "primaryStatus": int(primary.status),
                         "primaryMessage": str(primary.message), "tieSuccess": bool(tie.success),
                         "tieStatus": int(tie.status), "tieMessage": str(tie.message),
                         "primaryMapeFraction": primary_obj, "tieMapeFraction": tie_obj,
                         "objectiveGap": objective_gap,
                         "objectiveFaceTolerance": OBJECTIVE_FACE_TOL,
                         "maeConstraintBaseline": base_mae, "maeConstraintFitted": new_mae,
                         "maeResidual": new_mae-base_mae, "priceFloorRatio": PRICE_FLOOR_RATIO,
                         "scipyVersion": scipy.__version__}}
    return raw_tree, diagnostics


def _verify_frozen_model(model: Mapping[str, Any]) -> None:
    if model.get("criterion") != TREE_CONFIG["criterion"] or model.get("sklearnVersion") != sklearn.__version__:
        raise ValueError("saved tree config or sklearn version differs from the reviewed runtime")
    n = len(model["childrenLeft"])
    if not all(len(model[k]) == n for k in ("childrenRight", "feature", "threshold", "nNodeSamples", "weightedNodeSamples")):
        raise ValueError("saved tree arrays have inconsistent lengths")
    leaves = {str(i) for i, left in enumerate(model["childrenLeft"]) if int(left) == -1}
    if leaves != set(model.get("leafCorrections", {})):
        raise ValueError("saved leaf correction map does not cover exactly the leaves")
    for leaf, value in model["leafCorrections"].items():
        _num(value, f"leaf {leaf} correction")
    replay = model.get("trainingReplay")
    if not isinstance(replay, list):
        raise ValueError("saved C routing replay rows missing")
    estimator = _restore_native_model(model)
    for item in replay:
        got = int(estimator.apply(np.asarray([item["features"]], dtype=np.float64))[0])
        if got != int(item["leafId"]):
            raise ValueError("reloaded native sklearn tree changed C leaf routing")


def _metrics(records: list[dict[str, Any]], models: tuple[str, ...], products: tuple[str, ...] = PRODUCTS) -> dict[str, Any]:
    cells = []
    for product in products:
        for h in HORIZONS:
            group = [r for r in records if r["product"] == product and r["horizon"] == h]
            cell: dict[str, Any] = {"product": product, "horizon": h, "n": len(group)}
            for name in ("B",) + models:
                if group:
                    cell[name+"Mae"] = math.fsum(abs(r[name]-r["actual"]) for r in group) / len(group)
                    cell[name+"Mape"] = 100 * math.fsum(abs(r[name]-r["actual"])/r["actual"] for r in group) / len(group)
                else:
                    cell[name+"Mae"] = cell[name+"Mape"] = None
            cells.append(cell)
    complete = len(cells) == len(products)*10 and all(c["n"] for c in cells)
    sums = {}
    for name in ("B",) + models:
        available = complete and all(c[name+"Mae"] is not None for c in cells)
        sums[name+"Mae"] = math.fsum(c[name+"Mae"] for c in cells)/len(cells) if available else None
        sums[name+"Mape"] = math.fsum(c[name+"Mape"] for c in cells)/len(cells) if available else None
    return {"cells": cells, "completeCells": complete, "cellCount": len(cells),
            "pairedRows": sum(c["n"] for c in cells), "metrics": sums}


def fit_select(prices_path: str | Path = PRICE_PATH, models_output: str | Path | None = None,
               selection_output: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit four C trees/LPs, choose B or Rdelta from D6, then freeze both files."""
    if sklearn.__version__ != "1.3.0" or scipy.__version__ != "1.10.1":
        raise RuntimeError("reviewed runtime requires sklearn 1.3.0 and scipy 1.10.1; do not upgrade")
    v2, phases, sources = _load_sources()
    actuals, target_sha = _limited_actuals(prices_path, D6_END)
    c = _stage_join(phases["C"], actuals, C_START, C_END, dt.date(2026, 4, 1))
    d6 = _stage_join(phases["D6"], actuals, D6_START, D6_END, D6_ORIGIN)
    if not c or not d6:
        raise ValueError("C fit or D6 selection has no eligible positive targets")
    models: dict[str, Any] = {}
    fit_diagnostics: list[dict[str, Any]] = []
    for product in MODEL_PRODUCTS:
        rows = [r for r in c if r["product"] == product]
        for model_name in ("R0", "Rdelta"):
            fitted, diag = _fit_one(rows, product, model_name)
            _verify_frozen_model(fitted)
            models[f"{product}:{model_name}"] = fitted
            fit_diagnostics.append(diag)
    # Persist the exact C-fitted arrays before any D6 candidate evaluation. D6
    # predictions below are made only from these reloaded arrays.
    planned_d6 = Counter(key[0] for key in phases["D6"] if key[0] in MODEL_PRODUCTS)
    fit_rows = sum(d["trainingRows"] for d in fit_diagnostics)
    native_models = {name: _restore_native_model(model) for name, model in models.items()}
    c_forecasts = []
    for key, src in sorted(phases["C"].items()):
        p, o, t, h = key
        row = {"product":p,"origin":o,"target":t,"horizon":h,"B":src["_B"],
               "D":src["_D"],"delta":src["_delta"]}
        if p in MODEL_PRODUCTS:
            for model_name in ("R0","Rdelta"):
                correction, leaf = _model_g(models[f"{p}:{model_name}"],row,native_models[f"{p}:{model_name}"])
                raw = row["B"]*(1.+correction)
                reason = "raw_nonfinite" if not math.isfinite(raw) else "raw_nonpositive" if raw <= 0 else None
                row[model_name] = {"prediction":row["B"] if reason else raw,
                                   "rawPrediction":raw if math.isfinite(raw) else "nonfinite",
                                   "correction":correction,"leafId":leaf,"status":"issued",
                                   "fallbackReason":reason}
        else:
            row["R0"] = row["Rdelta"] = {"prediction":row["B"],"rawPrediction":row["B"],
                                          "correction":0.,"leafId":None,"status":"not_applicable",
                                          "fallbackReason":None}
        c_forecasts.append(row)
    c_prediction_index = {_key(r):r for r in c_forecasts}
    c_scored = []
    for r in c:
        forecast = c_prediction_index[(r["product"],r["origin"],r["target"],r["horizon"])]
        c_scored.append({**r,"R0":forecast["R0"]["prediction"],"Rdelta":forecast["Rdelta"]["prediction"]})
    c_metrics = _metrics(c_scored,("R0","Rdelta"),MODEL_PRODUCTS)
    model_doc: dict[str, Any] = {"schemaVersion": 1, "experiment": "CONDITIONAL_BASELINE_RESIDUAL_V1",
        "stage": "C_models_frozen", "protocolSha256": PROTOCOL_SHA256,
        "targetSnapshotSha256": target_sha, "targetUse": {"trainingThrough": C_END.isoformat(),
        "selectionThrough": D6_END.isoformat(), "TLabelsUsed": False},
        "sourceArtifacts": sources, "treeConfig": TREE_CONFIG,
        "libraries": {"python": __import__("sys").version.split()[0], "numpy": np.__version__,
                       "scipy": scipy.__version__, "scikitLearn": sklearn.__version__},
        "fitCounts": {"treeFits": 4, "lpFits": 4, "baselineFits": 0, "baselineInference": 0},
        "newModelInference": {"D6LogicalModelBatches": 4,
             "D6RowPredictionsByProductModel": {f"{p}:{m}": planned_d6[p]
                                                 for p in MODEL_PRODUCTS for m in ("R0", "Rdelta")},
             "CTrainingLeafAssignmentRows": fit_rows,
             "CNativeRoutingReplayRows": fit_rows,
             "baselineModelCalls": 0, "baselineRowPredictions": 0},
        "fitDiagnostics": fit_diagnostics, "trainingForecasts": c_forecasts,
        "trainingForecastsSha256": _sha(_canonical(c_forecasts)),
        "trainingGridRows": len(c_forecasts), "trainingMetrics20Cells": c_metrics,
        "models": models}
    model_doc["modelSha256"] = _sha(_canonical(model_doc))
    model_dest = _path(models_output or OUT / "models.json")
    selection_dest = _path(selection_output or OUT / "selection.json")
    _write_once(model_dest, _canonical(model_doc))
    model_file_sha = _sha(model_dest.read_bytes())
    reloaded_models = _read(model_dest)
    _doc_sha(reloaded_models, "modelSha256")
    if model_file_sha != _sha(_canonical(reloaded_models)):
        raise ValueError("reloaded C model serialization changed")
    for fitted in reloaded_models.get("models", {}).values():
        _verify_frozen_model(fitted)
    reloaded_native = {name:_restore_native_model(model) for name,model in reloaded_models["models"].items()}
    c_replay = 0
    if _sha(_canonical(reloaded_models["trainingForecasts"])) != reloaded_models.get("trainingForecastsSha256"):
        raise ValueError("saved C training forecast digest mismatch")
    for key, src in sorted(phases["C"].items()):
        if key[0] not in MODEL_PRODUCTS:
            continue
        expected_row = c_prediction_index[key]
        input_row = {"B":src["_B"],"D":src["_D"],"delta":src["_delta"],"horizon":key[3]}
        for model_name in ("R0","Rdelta"):
            correction, leaf = _model_g(reloaded_models["models"][f"{key[0]}:{model_name}"],
                                        input_row,reloaded_native[f"{key[0]}:{model_name}"])
            observed = expected_row[model_name]
            raw = src["_B"]*(1.+correction)
            replay_prediction = src["_B"] if not math.isfinite(raw) or raw <= 0 else raw
            if (leaf != observed["leafId"] or correction != observed["correction"]
                    or not math.isclose(replay_prediction,observed["prediction"],rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL)):
                raise ValueError("reloaded native model changed C training forecast")
            c_replay += 1
    # Issue candidates across the full D6 key grid. Labels are joined only to
    # these saved prediction values when creating the aggregate selection stats.
    d6_forecasts: list[dict[str, Any]] = []
    d6_pairs: list[dict[str, Any]] = []
    call_counts = Counter()
    d6_issued = Counter()
    for key, src in sorted(phases["D6"].items()):
        p, o, t, h = key
        row = {"product":p,"origin":o,"target":t,"horizon":h,"B":src["_B"],
               "D":src["_D"],"delta":src["_delta"]}
        for model_name in ("R0","Rdelta"):
            d6_issued[(p,h,model_name)] += 1
            if p not in MODEL_PRODUCTS:
                row[model_name] = {"prediction":src["_B"],"rawPrediction":src["_B"],
                                   "correction":0.,"leafId":None,"status":"not_applicable","fallbackReason":None}
                continue
            call_counts[(p,model_name)] += 1
            fitted = reloaded_models["models"][f"{p}:{model_name}"]
            correction, leaf = _model_g(fitted,row,reloaded_native[f"{p}:{model_name}"])
            raw = src["_B"]*(1.+correction)
            reason = "raw_nonfinite" if not math.isfinite(raw) else "raw_nonpositive" if raw <= 0 else None
            row[model_name] = {"prediction":src["_B"] if reason else raw,
                               "rawPrediction":raw if math.isfinite(raw) else "nonfinite",
                               "correction":correction,"leafId":leaf,"status":"issued",
                               "fallbackReason":reason}
        actual = actuals.get((p,t))
        if actual is not None:
            d6_pairs.append({**row,"actual":actual,"R0":row["R0"]["prediction"],
                             "Rdelta":row["Rdelta"]["prediction"]})
        d6_forecasts.append(row)
    selections: dict[str, Any] = {}
    for product in MODEL_PRODUCTS:
        product_rows = [r for r in d6_pairs if r["product"] == product]
        b_metrics = _metrics(product_rows, (), (product,))
        all_metrics = _metrics(product_rows, ("R0","Rdelta"), (product,))
        # Exact issued keys are shared; target availability only changes the paired denominator.
        if not b_metrics["completeCells"] or not all_metrics["completeCells"]:
            raise ValueError(f"D6 selection for {product} lacks complete positive-target horizon coverage")
        r_mape, r_mae = all_metrics["metrics"]["RdeltaMape"], all_metrics["metrics"]["RdeltaMae"]
        eligible = (r_mape < b_metrics["metrics"]["BMape"]
                    and r_mae <= b_metrics["metrics"]["BMae"]
                    and len(product_rows) >= 1)
        issued_rows = sum(1 for r in d6_forecasts if r["product"] == product)
        issued_model_rows = [r for r in d6_forecasts if r["product"] == product]
        coverage_no_loss = all(r["R0"].get("status") == "issued"
                               and r["Rdelta"].get("status") == "issued" for r in issued_model_rows)
        if not coverage_no_loss:
            raise ValueError(f"D6 candidate issuance coverage differs for {product}")
        selections[product] = {"selectedModel": "Rdelta" if eligible else "B",
                               "rule": "Rdelta MAPE strictly lower, MAE no higher, exact paired coverage; otherwise B",
                               "coverageNoLoss": coverage_no_loss,
                               "issuedGridRows": issued_rows,"positiveTargetPairs":len(product_rows),
                               "unavailableTargetKeys":issued_rows-len(product_rows),
                               "metrics20Cells": {"B": b_metrics["metrics"],
                                                  "R0":{"mae":all_metrics["metrics"]["R0Mae"],
                                                       "mape":all_metrics["metrics"]["R0Mape"]},
                                                  "Rdelta":{"mae":r_mae,"mape":r_mape}},
                               "metricsByHorizon": all_metrics["cells"]}
    # Attach selected-policy outputs only after D6 model-vs-B selection is frozen.
    for row in d6_forecasts:
        selected = selections.get(row["product"],{}).get("selectedModel","B")
        row["selectedModel"] = selected
        row["selectedPolicy"] = (row["Rdelta"] if selected == "Rdelta" else
                                  {"prediction":row["B"],"rawPrediction":row["B"],"correction":0.,
                                   "leafId":None,
                                   "status":"fixed_B" if row["product"] not in MODEL_PRODUCTS else "selected_B",
                                   "fallbackReason":None})
    for product in MODEL_PRODUCTS:
        product_rows = [r for r in d6_pairs if r["product"] == product]
        selected_model = selections[product]["selectedModel"]
        for row in product_rows:
            row["selectedPolicy"] = row[selected_model] if selected_model == "Rdelta" else row["B"]
        selected_metrics = _metrics(product_rows,("selectedPolicy",),(product,))
        selections[product]["metrics20Cells"]["selectedPolicy"] = {
            "mae":selected_metrics["metrics"]["selectedPolicyMae"],
            "mape":selected_metrics["metrics"]["selectedPolicyMape"]}
    selection_doc: dict[str, Any] = {"schemaVersion": 1, "experiment": "CONDITIONAL_BASELINE_RESIDUAL_V1",
        "stage": "D6_selection_frozen", "protocolSha256": PROTOCOL_SHA256,
        "modelsSha256": model_doc["modelSha256"], "modelFileSha256": model_file_sha,
        "targetSnapshotSha256": target_sha, "targetUse": {"selectionDates": [D6_START.isoformat(), D6_END.isoformat()],
        "originAtLeast": D6_ORIGIN.isoformat(), "CRefitAfterD6": False, "TLabelsUsed": False},
        "selectionByProduct": selections,
        "allCandidatesComputedOnD6": True, "R0Selectable": False,
        "d6IssuedGridRows":len(d6_forecasts),"d6PositiveActualPairs":len(d6_pairs),
        "d6ForecastsSha256":_sha(_canonical(d6_forecasts)),"d6Forecasts":d6_forecasts}
    selection_doc["D6LogicalModelBatches"] = 4
    selection_doc["D6RowPredictionCount"] = sum(call_counts.values())
    selection_doc["CNativeForecastReplayRows"] = c_replay
    selection_doc["D6RowPredictionsByProductModel"] = model_doc["newModelInference"]["D6RowPredictionsByProductModel"]
    selection_doc["selectionSha256"] = _sha(_canonical(selection_doc))
    _write_once(selection_dest, _canonical(selection_doc))
    return model_doc, selection_doc


def _load_frozen(path: str | Path, expected_file_sha: str, field: str, expected_stage: str) -> tuple[dict[str, Any], str]:
    p = _path(path); raw = p.read_bytes(); digest = _sha(raw)
    if digest != expected_file_sha:
        raise ValueError(f"caller-pinned {expected_stage} file hash mismatch")
    doc = _read(p)
    _doc_sha(doc, field)
    if doc.get("stage") != expected_stage or doc.get("experiment") != "CONDITIONAL_BASELINE_RESIDUAL_V1":
        raise ValueError(f"expected frozen {expected_stage} artifact")
    return doc, digest


def forecast_t(models_path: str | Path, models_sha256: str, selection_path: str | Path,
               selection_sha256: str, output: str | Path | None = None) -> dict[str, Any]:
    """Apply frozen C functions to answer-free T rows; this never opens prices."""
    model_doc, model_file_sha = _load_frozen(models_path, models_sha256, "modelSha256", "C_models_frozen")
    selection, selection_file_sha = _load_frozen(selection_path, selection_sha256, "selectionSha256", "D6_selection_frozen")
    if selection.get("modelsSha256") != model_doc.get("modelSha256") or selection.get("modelFileSha256") != model_file_sha:
        raise ValueError("D6 selection is not bound to this exact frozen model file")
    if model_doc.get("protocolSha256") != PROTOCOL_SHA256 or selection.get("protocolSha256") != PROTOCOL_SHA256:
        raise ValueError("frozen protocol hash changed")
    if model_doc.get("targetSnapshotSha256") != PRICE_SHA256 or selection.get("targetSnapshotSha256") != PRICE_SHA256:
        raise ValueError("model/selection target snapshot mismatch")
    selection_map = selection.get("selectionByProduct")
    if (not isinstance(selection_map, Mapping) or set(selection_map) != set(MODEL_PRODUCTS)
            or any(v.get("selectedModel") not in ("B","Rdelta") for v in selection_map.values())):
        raise ValueError("D6 frozen policy map is incomplete or contains an ineligible candidate")
    v2, phases, sources = _load_sources()
    source_rows = phases["T"]
    out_rows = []
    row_calls = Counter()
    native_models = {name:_restore_native_model(model) for name,model in model_doc["models"].items()}
    for key, src in sorted(source_rows.items()):
        p, o, t, h = key
        b = src["_B"]
        pred_objs: dict[str, Any] = {}
        for model_name in ("R0", "Rdelta"):
            if p not in MODEL_PRODUCTS:
                pred_objs[model_name] = {"prediction": b, "rawPrediction": b,
                                         "correction": 0., "leafId": None,
                                         "status": "not_applicable", "fallbackReason": None}
                continue
            fitted = model_doc["models"][f"{p}:{model_name}"]
            row_calls[(p, model_name)] += 1
            correction, leaf = _model_g(fitted, {"B": b, "D": src["_D"], "delta": src["_delta"], "horizon": h},
                                        native_models[f"{p}:{model_name}"])
            raw = b * (1. + correction)
            reason = None
            if not math.isfinite(raw):
                reason, prediction = "raw_nonfinite", b
            elif raw <= 0:
                reason, prediction = "raw_nonpositive", b
            else:
                prediction = raw
            pred_objs[model_name] = {"prediction": prediction, "rawPrediction": raw if math.isfinite(raw) else "nonfinite",
                                     "correction": correction, "leafId": leaf, "status": "issued",
                                     "fallbackReason": reason}
        selected = selection_map.get(p, {}).get("selectedModel", "B")
        selected_obj = (pred_objs[selected] if selected in ("R0", "Rdelta") else
                        {"prediction": b, "rawPrediction": b, "correction": 0., "leafId": None,
                         "status": "fixed_B" if p not in MODEL_PRODUCTS else "selected_B",
                         "fallbackReason": None})
        out_rows.append({"product": p, "origin": o, "target": t, "horizon": h,
                         "B": b, "D": src["_D"], "delta": src["_delta"],
                         "directDeltaDefinition": "P-L" if p == "radish" else "A-CTRL" if p == "onion" else "not_applicable",
                         "R0": pred_objs["R0"], "Rdelta": pred_objs["Rdelta"],
                         "selectedModel": selected, "selectedPolicy": selected_obj,
                         "sourceProvenance": {"REPAIR_V2ManifestSha256": sources["REPAIR_V2"]["manifestSha256"],
                                               "K1SourceRowSha256": src["sourceProvenance"]["sourceRowSha256"]}})
    # Replay every T row from the reloaded arrays and require exact outputs.
    replay_count = 0
    for row in out_rows:
        if row["product"] not in MODEL_PRODUCTS:
            continue
        for model_name in ("R0", "Rdelta"):
            fitted = model_doc["models"][f"{row['product']}:{model_name}"]
            correction, leaf = _model_g(fitted, {"B":row["B"],"D":row["D"],"delta":row["delta"],"horizon":row["horizon"]},
                                        native_models[f"{row['product']}:{model_name}"])
            stored = row[model_name]
            replay_value = row["B"] * (1. + correction)
            replay_reason = "raw_nonfinite" if not math.isfinite(replay_value) else "raw_nonpositive" if replay_value <= 0 else None
            replay_prediction = row["B"] if replay_reason else replay_value
            if (leaf != stored["leafId"] or correction != stored["correction"]
                    or replay_reason != stored["fallbackReason"]
                    or not math.isclose(replay_prediction, stored["prediction"], rel_tol=REPLAY_TOL, abs_tol=REPLAY_TOL)):
                raise ValueError("T saved-model inference replay mismatch")
            replay_count += 1
    doc = {"schemaVersion": 1, "experiment": "CONDITIONAL_BASELINE_RESIDUAL_V1", "phase": "T",
           "stage": "T_answer_free_forecasts", "targetValuesAttached": False,
           "protocolSha256": PROTOCOL_SHA256, "modelFileSha256": model_file_sha,
           "modelSha256": model_doc["modelSha256"], "selectionFileSha256": selection_file_sha,
           "selectionSha256": selection["selectionSha256"], "targetSnapshotSha256": PRICE_SHA256,
           "selectionByProduct": selection["selectionByProduct"],
           "sourceArtifacts": sources, "forecastCounts": {"rows": len(out_rows),
             "expectedRows": len(_expected_grid(T_START, T_END, T_ORIGIN))},
           "inferenceCounts": {"baselineModelCalls": 0, "baselineRowPredictions": 0,
             "newModelLogicalModelBatches": 4,
             "newModelRowPredictionsByProductModel": {f"{p}:{m}": row_calls[(p,m)]
                                                        for p in MODEL_PRODUCTS for m in ("R0", "Rdelta")},
             "newModelRowPredictionCount": sum(row_calls.values()),
             "replayRowPredictions": replay_count,
             "replayToleranceRelAbs": REPLAY_TOL,
             "selectionPolicyLookups": len(out_rows), "TLabelsLoaded": False},
           "forecastsSha256": _sha(_canonical(out_rows)), "forecasts": out_rows}
    doc["manifestSha256"] = _sha(_canonical(doc))
    if output is not None:
        _write_once(_path(output), _canonical(doc))
    return doc


def _load_t_prices(prices_path: str | Path) -> tuple[dict[tuple[str, str], float], str]:
    path = _path(prices_path); raw = path.read_bytes()
    if _sha(raw) != PRICE_SHA256:
        raise ValueError("pinned T target snapshot file hash mismatch")
    series = price_helper._series(json.loads(raw))
    actuals = {(p, day.isoformat()): float(value) for p, values in series.items()
               for day, value in values.items() if T_START <= day <= T_END and value > 0}
    return actuals, PRICE_SHA256


def _contributions20(records: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, Any]:
    cells = Counter((r["product"], r["horizon"]) for r in records)
    groups: dict[tuple[str, int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in records:
        w = 1. / (20 * cells[(r["product"], r["horizon"])])
        mae = abs(r[candidate]-r["actual"]) - abs(r[baseline]-r["actual"])
        mape = 100 * mae / r["actual"]
        g = groups[(r["product"], r["horizon"], r["target"][:7])]
        g["n"] += 1
        for name, diff in (("mae", mae), ("mape", mape)):
            g[name+"Net"] += diff*w
            g[name+"Gain"] += max(-diff, 0.)*w
            g[name+"Loss"] += max(diff, 0.)*w
    return {"productHorizonMonth": [{"product":k[0],"horizon":k[1],"month":k[2],**v}
                                    for k,v in sorted(groups.items())],
            "netMae": math.fsum(v["maeNet"] for v in groups.values()),
            "netMapePercentagePoints": math.fsum(v["mapeNet"] for v in groups.values()),
            "weighting": "20 changed-product cells, equal horizon weight"}


def _loss_direction(actual: float, baseline: float, prediction: float) -> str:
    if math.isclose(prediction,baseline,rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL):
        return "unchanged_B"
    residual = actual - baseline
    correction = prediction - baseline
    if math.isclose(residual,0.,rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL):
        return "changed_when_B_exact"
    if correction * residual < 0 and not math.isclose(correction,0.,rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL):
        return "opposite_direction"
    if math.isclose(prediction,actual,rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL):
        return "exact_correction"
    if abs(correction) > abs(residual):
        return "overshot_actual"
    return "toward_actual"


def _selected_policy_identical_to_baseline(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check exact identity over the complete issued forecast grid, including unavailable targets."""
    return all(row["selectedPolicy"]["prediction"] == row["B"] for row in rows)


def _direction_report(records: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    delta_relation: dict[tuple[str,str,str], dict[str,float]] = defaultdict(lambda: defaultdict(float))
    action_classes: dict[str,dict[str,float]] = defaultdict(lambda: defaultdict(float))
    cells = Counter((r["product"], r["horizon"]) for r in records)
    for r in records:
        direction = _loss_direction(r["actual"], r["B"], r[candidate])
        g = counts[direction]
        w = 1. / (40 * cells[(r["product"], r["horizon"])])
        mae = abs(r[candidate]-r["actual"]) - abs(r["B"]-r["actual"])
        mape = 100 * mae/r["actual"]
        g["n"] += 1
        g["maeNet"] += mae*w; g["mapeNet"] += mape*w
        g["maeGain"] += max(-mae, 0.)*w; g["maeLoss"] += max(mae, 0.)*w
        g["mapeGain"] += max(-mape, 0.)*w; g["mapeLoss"] += max(mape, 0.)*w
        delta = r.get("delta",0.)
        correction = r[candidate]-r["B"]
        residual = r["actual"]-r["B"]
        delta_state = ("zero_delta" if math.isclose(delta,0.,rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL)
                       else "correction_opposes_delta" if correction*delta < 0
                       else "correction_aligns_delta" if correction*delta > 0 else "zero_correction")
        residual_state = ("baseline_exact" if math.isclose(residual,0.,rel_tol=REPLAY_TOL,abs_tol=REPLAY_TOL)
                          else "correction_matches_residual" if correction*residual > 0
                          else "correction_opposes_residual" if correction*residual < 0 else "zero_correction")
        old_delta_state = ("delta_opposed_residual" if delta*residual < 0 else
                           "delta_aligned_residual" if delta*residual > 0 else "zero_delta_or_residual")
        dg = delta_relation[(delta_state,residual_state,old_delta_state)]
        dg["n"] += 1
        dg["maeNet"] += mae*w; dg["mapeNet"] += mape*w
        dg["maeGain"] += max(-mae,0.)*w; dg["maeLoss"] += max(mae,0.)*w
        dg["mapeGain"] += max(-mape,0.)*w; dg["mapeLoss"] += max(mape,0.)*w
        if old_delta_state == "delta_opposed_residual" and residual_state == "correction_matches_residual":
            action = "direction_fix_against_old_delta"
        elif old_delta_state == "delta_opposed_residual" and residual_state == "correction_opposes_residual":
            action = ("old_wrong_direction_shrunk" if abs(correction) < abs(delta)
                      else "old_wrong_direction_not_shrunk")
        elif old_delta_state == "delta_aligned_residual" and residual_state == "correction_opposes_residual":
            action = "new_correction_flipped_correct_old_direction"
        elif old_delta_state == "delta_aligned_residual" and residual_state == "correction_matches_residual":
            action = "both_directions_align_with_residual"
        else:
            action = "zero_or_exact_baseline_case"
        ag = action_classes[action]
        ag["n"] += 1
        ag["maeNet"] += mae*w; ag["mapeNet"] += mape*w
        ag["maeGain"] += max(-mae,0.)*w; ag["maeLoss"] += max(mae,0.)*w
        ag["mapeGain"] += max(-mape,0.)*w; ag["mapeLoss"] += max(mape,0.)*w
    return {"byResidualDirection": [{"direction":k,**v} for k,v in sorted(counts.items())],
            "byDeltaAndResidualDirection": [{"deltaCorrectionRelation":k[0],
                "correctionResidualRelation":k[1],"oldDeltaVsResidual":k[2],**v}
                for k,v in sorted(delta_relation.items())],
            "directionFixVersusShrinkage": [{"classification":k,**v} for k,v in sorted(action_classes.items())],
            "classification": "realized residual versus applied correction direction; not a causal market diagnosis"}


def score_t(forecast_path: str | Path, expected_forecast_sha256: str,
            prices_path: str | Path = PRICE_PATH, output: str | Path | None = None) -> dict[str, Any]:
    """Join T targets only after verifying the exact saved answer-free forecast file."""
    path = _path(forecast_path); raw = path.read_bytes(); file_sha = _sha(raw)
    if file_sha != expected_forecast_sha256:
        raise ValueError("caller-pinned T forecast file hash mismatch")
    doc = _read(path); _doc_sha(doc, "manifestSha256")
    if (doc.get("experiment") != "CONDITIONAL_BASELINE_RESIDUAL_V1" or doc.get("phase") != "T"
            or doc.get("stage") != "T_answer_free_forecasts" or doc.get("targetValuesAttached") is not False):
        raise ValueError("expected frozen answer-free T forecasts")
    if doc.get("protocolSha256") != PROTOCOL_SHA256 or doc.get("targetSnapshotSha256") != PRICE_SHA256:
        raise ValueError("T forecast protocol or target identity differs")
    rows = doc.get("forecasts")
    if not isinstance(rows, list) or _sha(_canonical(rows)) != doc.get("forecastsSha256"):
        raise ValueError("T forecast rows digest mismatch")
    keyed = _index_rows(rows, "conditional T")
    if set(keyed) != _expected_grid(T_START, T_END, T_ORIGIN):
        raise ValueError("T forecast grid incomplete; refusing intersection-based scoring")
    # Only now read the frozen target snapshot, after forecast hash and full grid checks.
    actuals, target_sha = _load_t_prices(prices_path)
    joined = []
    for key, row in sorted(keyed.items()):
        actual = actuals.get((key[0], key[2]))
        if actual is None:
            continue
        b = _num(row.get("B"), "T/B", positive=True)
        pair = {"product":key[0],"origin":key[1],"target":key[2],"horizon":key[3],"actual":actual,"B":b}
        for name in ("R0", "Rdelta", "selectedPolicy"):
            obj = row.get(name)
            if not isinstance(obj, Mapping):
                raise ValueError(f"T {name} output is missing")
            prediction = _num(obj.get("prediction"), f"T/{name}", positive=True)
            pair[name] = prediction
        pair["delta"] = _num(row.get("delta"), "T/delta")
        pair["D"] = (_num(row.get("D"), "T/D", positive=True)
                     if key[0] in MODEL_PRODUCTS else None)
        joined.append(pair)
    if len(joined) != 2470:
        raise ValueError(f"T positive-target pair count expected 2470, found {len(joined)}")
    primary = _metrics(joined, ("selectedPolicy",))
    candidate = _metrics(joined, ("Rdelta",))
    changed = [r for r in joined if r["product"] in MODEL_PRODUCTS]
    secondary = _metrics(changed, ("Rdelta", "R0"), MODEL_PRODUCTS)
    if not primary["completeCells"] or not secondary["completeCells"]:
        raise ValueError("T inference family lacks full prespecified horizon cells")
    primary_pairs = [{**r, "a":r["selectedPolicy"], "b":r["B"]} for r in joined]
    secondary_pairs = [{**r, "a":r["Rdelta"], "b":r["R0"]} for r in changed]
    dm_primary = inference._dm(primary_pairs, PRODUCTS, "selected_policy_vs_B")
    dm_secondary = inference._dm(secondary_pairs, MODEL_PRODUCTS, "Rdelta_vs_R0_changed_products")
    # The existing contribution helper uses a 40-cell denominator; secondary has 20 cells.
    contributions = {"selectedPolicyVsB": inference._contributions(primary_pairs),
                     "RdeltaVsB": inference._contributions([{**r,"a":r["Rdelta"],"b":r["B"]} for r in joined]),
                     "RdeltaVsR0": _contributions20(secondary_pairs, "Rdelta", "R0")}
    bootstraps = []
    for metric in ("ae", "ape"):
        for block in (14, 28):
            bootstraps.append({"family":"selected_policy_vs_B","metric":metric,"blockDays":block,
                **diagnostics.circular_block_bootstrap(primary_pairs,a="a",b="b",metric=metric,block=block,
                      replicates=2000,seed=0,calendar_start=T_START,calendar_end=T_END)})
    m = primary["metrics"]
    selected_all_b = all(doc["selectionByProduct"][p]["selectedModel"] == "B" for p in MODEL_PRODUCTS)
    selected_identical = _selected_policy_identical_to_baseline(rows)
    decision = ("completed_no_selected_correction" if selected_all_b
                else "completed_no_change" if selected_identical
                else "completed_exploratory_gain" if m["selectedPolicyMape"] < m["BMape"]
                     and m["selectedPolicyMae"] <= m["BMae"]
                else "completed_no_incremental_gain")
    result = {"schemaVersion":1,"experiment":"CONDITIONAL_BASELINE_RESIDUAL_V1","phase":"T_score",
        "forecastFileSha256":file_sha,"forecastManifestSha256":doc["manifestSha256"],
        "modelFileSha256":doc["modelFileSha256"],"selectionFileSha256":doc["selectionFileSha256"],
        "targetSnapshotSha256":target_sha,"keyCount":len(keyed),"positiveActualPairs":len(joined),
        "unavailableTargetKeys":len(keyed)-len(joined),"metrics40Cell":primary,
        "candidateRdeltaMetrics40Cell":candidate,"secondaryRdeltaVsR0Metrics20Cell":secondary,
        "inference":{"primary80TestsPerBandwidth":dm_primary,
                     "secondary40TestsPerBandwidth":dm_secondary,
                     "familiesSeparate":True,"statistics":"DM HAC 9/28 with Holm within family/bandwidth"},
        "jointCalendarBlockBootstrap":{"seed":0,"replicates":2000,"blocksDays":[14,28],"results":bootstraps},
        "contributions":contributions,
        "residualDirection":{"selectedPolicyVsB":_direction_report(joined,"selectedPolicy"),
                              "RdeltaVsB":_direction_report(joined,"Rdelta")},
        "selectionByProduct":doc["selectionByProduct"],"inferenceCounts":doc["inferenceCounts"],
        "decisionGate":decision,
        "interpretation":"retrospective evaluation on previously inspected V/T; not independent confirmation or causal evidence"}
    result["reportSha256"] = _sha(_canonical(result))
    if output is not None:
        _write_once(_path(output), _canonical(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-select")
    fit.add_argument("--prices", default=PRICE_PATH); fit.add_argument("--models", default=str(OUT/"models.json"))
    fit.add_argument("--selection", default=str(OUT/"selection.json"))
    forecast = sub.add_parser("forecast-T")
    forecast.add_argument("--models", required=True); forecast.add_argument("--models-sha256", required=True)
    forecast.add_argument("--selection", required=True); forecast.add_argument("--selection-sha256", required=True)
    forecast.add_argument("--output", required=True)
    score = sub.add_parser("score-T")
    score.add_argument("--forecasts", required=True); score.add_argument("--forecast-sha256", required=True)
    score.add_argument("--prices", default=PRICE_PATH); score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "fit-select":
        models, selection = fit_select(args.prices, args.models, args.selection)
        result = {"stage":"fit-select","modelSha256":models["modelSha256"],"selectionSha256":selection["selectionSha256"]}
    elif args.command == "forecast-T":
        doc = forecast_t(args.models,args.models_sha256,args.selection,args.selection_sha256,args.output)
        result = {"stage":doc["stage"],"manifestSha256":doc["manifestSha256"],"forecastsSha256":doc["forecastsSha256"]}
    else:
        doc = score_t(args.forecasts,args.forecast_sha256,args.prices,args.output)
        result = {"stage":"T_score","reportSha256":doc["reportSha256"]}
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
