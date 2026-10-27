"""Within-crop panel training experiment (CAND_B), without data collection."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import dashboard, modeling, external_reference_v1 as source_exp
from . import equal_info_online_forecasts as online_source
from .chronos2_market_context import BASELINE_PATHS, BASELINE_SHA256, PHASES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/intra-crop-panel-cand-b-v1-20261013"
SOURCE_OUT = ROOT / "outputs/experiments/external-reference-v1-20261012"
SOURCE_CACHE = SOURCE_OUT / "source-cache.json"
SOURCE_CACHE_INTERNAL_SHA256 = "d6d0212a7feecad7914ab41c5655f134df15cee259002497a4684f310b58ce8a"
SOURCE_V_MANIFEST_SHA256 = "1257f6c49c48b8dbc0f8467d91d025b8dbf9f5ea24db6127b3d2f7916a87138d"
SOURCE_V_MANIFEST_CANONICAL_SHA256 = "fac1f0f8bf62fab5879453eb866e151c5b6b123cda3cabd1aaf352d757224a8a"
PRICES_PATH = source_exp.PRICES_PATH
PRICE_SNAPSHOT_SHA256 = online_source.PRICES_SHA256
PRODUCTS = source_exp.PRODUCTS
HORIZONS = source_exp.HORIZONS
VARIANTS = ("L", "P", "Q")
MODEL_VARIANTS = ("L", "P", "Q")
MIN_TRAIN_ROWS = source_exp.MIN_TRAIN_ROWS
MASTER_SEED = 0
BASE_FEATURE_NAMES = tuple(modeling.FEATURE_NAMES)
SERIES_FEATURE_NAME = "series_id"
FEATURE_NAMES = BASE_FEATURE_NAMES + (SERIES_FEATURE_NAME,)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive(value: Any) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0 else None


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _write_once(path: Path, data: Mapping[str, Any]) -> None:
    raw = _canonical(data)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"refusing to overwrite immutable artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def _required_saved_parameter(booster_params: Mapping[str, Any], model_params: Mapping[str, Any],
                              names: tuple[str, ...], field: str) -> Any:
    candidates = [booster_params[name] for name in names if name in booster_params]
    candidates.extend(model_params[name] for name in names if name in model_params)
    if not candidates:
        raise ValueError(f"saved CTRL booster does not serialize required parameter {field}; aliases={names}")
    first = candidates[0]
    def equivalent(value: Any, reference: Any) -> bool:
        if isinstance(value, bool) or isinstance(reference, bool):
            return isinstance(value, bool) and isinstance(reference, bool) and value is reference
        try:
            return math.isclose(float(value), float(reference), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError, OverflowError):
            return str(value) == str(reference)
    if any(not equivalent(value, first) for value in candidates[1:]):
        raise ValueError(f"saved CTRL booster has conflicting aliases for {field}: {candidates}")
    return first


_LIGHTGBM_BOOLEAN_PARAMETERS = {
    "use_missing", "zero_as_missing", "deterministic", "force_col_wise", "force_row_wise",
    "is_enable_sparse", "enable_bundle", "feature_pre_filter", "is_unbalance",
    "boost_from_average", "reg_sqrt", "is_provide_training_metric", "linear_tree",
}


def _parse_booster_parameter_block(model_text: str) -> tuple[str, dict[str, Any]]:
    lines = model_text.splitlines()
    try:
        start = lines.index("parameters:") + 1
    except ValueError as exc:
        raise ValueError("saved LightGBM model text lacks a parameters block") from exc
    block_lines = []
    parsed: dict[str, Any] = {}
    for line in lines[start:]:
        if not line:
            break
        if not (line.startswith("[") and line.endswith("]") and ": " in line):
            raise ValueError(f"unrecognized LightGBM parameter serialization: {line!r}")
        key, raw_value = line[1:-1].split(": ", 1)
        if key in parsed:
            raise ValueError(f"duplicate serialized LightGBM parameter: {key}")
        block_lines.append(line)
        if raw_value == "":
            value: Any = ""
        elif key in _LIGHTGBM_BOOLEAN_PARAMETERS and raw_value in ("0", "1"):
            value = raw_value == "1"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        parsed[key] = value
    if not parsed:
        raise ValueError("saved LightGBM parameters block is empty")
    return "parameters:\n" + "\n".join(block_lines) + "\n", parsed


def extract_saved_ctrl_config(*, source_manifest_path: Path | None = None,
                              output_path: Path | None = None) -> dict[str, Any]:
    """Read serialized CTRL boosters; never fit or infer during config resolution."""
    import joblib

    manifest_path = source_manifest_path or SOURCE_OUT / "V/forecast-manifest.json"
    manifest = _read_json(manifest_path)
    source_digest = _file_sha(manifest_path)
    if source_digest != SOURCE_V_MANIFEST_SHA256:
        raise ValueError("saved CAND_A V manifest differs from the independently pinned artifact")
    if manifest.get("phase") != "V" or manifest.get("targetValuesAttached") is not False:
        raise ValueError("saved CAND_A V forecast manifest is not answer-free")
    unsigned_manifest = dict(manifest)
    copied_self_hash = unsigned_manifest.pop("manifestSha256", None)
    if (copied_self_hash != SOURCE_V_MANIFEST_CANONICAL_SHA256 or
            copied_self_hash != _sha(_canonical(unsigned_manifest))):
        raise ValueError("saved CAND_A V manifest self-hash mismatch")
    if manifest.get("sourceCodeSha256") != source_exp._file_sha(Path(source_exp.__file__)):
        raise ValueError("saved CAND_A manifest source-code identity mismatch")
    fit_records = manifest.get("fitRecords")
    if not isinstance(fit_records, list):
        raise ValueError("saved CAND_A manifest lacks fit records")
    models = []
    for record in fit_records:
        if record.get("modelStatus") not in ("fit", "reused") or not record.get("modelFile"):
            continue
        path = SOURCE_OUT / record["modelFile"]
        if not path.is_file() or _file_sha(path) != record.get("modelFileSha256"):
            raise ValueError(f"saved CTRL checkpoint missing or hash mismatch: {path}")
        bundle = joblib.load(path)
        model = bundle.get("CTRL_LGB")
        booster = getattr(model, "booster_", None)
        if booster is None:
            raise ValueError(f"saved CTRL checkpoint lacks a fitted LightGBM booster: {path}")
        params = dict(getattr(booster, "params", {}) or {})
        sklearn_params = dict(model.get_params(deep=False))
        raw_model = booster.model_to_string()
        raw_parameter_block, serialized_parameters = _parse_booster_parameter_block(raw_model)
        models.append({"identity": {"product": record.get("product"), "horizon": record.get("horizon"),
                                    "fitMonth": record.get("fitMonth"), "checkpointSha256": record["modelFileSha256"],
                                    "boosterSha256": _sha(raw_model.encode())},
                       "params": params, "sklearnParams": sklearn_params,
                       "serializedParameters": serialized_parameters,
                       "rawParameterBlock": raw_parameter_block,
                       "iterations": int(booster.current_iteration()),
                       "featureNames": list(booster.feature_name())})
    if not models:
        raise ValueError("no fitted saved CTRL_LGB boosters were available for config extraction")
    param_signatures = {_sha(_canonical({"params": x["params"], "sklearnParams": x["sklearnParams"],
                                        "serializedParameters": x["serializedParameters"],
                                        "iterations": x["iterations"], "featureNames": x["featureNames"]}))
                        for x in models}
    if len(param_signatures) != 1:
        raise ValueError("saved CTRL boosters have nonidentical serialized configurations")
    representative = models[0]
    bp, sp, serialized = (representative["params"], representative["sklearnParams"],
                           representative["serializedParameters"])
    parameter_sources = {**bp, **sp}
    inherited = {
        "num_boost_round": _required_saved_parameter(serialized, parameter_sources, ("num_iterations", "num_iteration", "n_estimators"), "num_boost_round"),
        "learning_rate": _required_saved_parameter(serialized, parameter_sources, ("learning_rate", "eta"), "learning_rate"),
        "num_leaves": _required_saved_parameter(serialized, parameter_sources, ("num_leaves",), "num_leaves"),
        "max_depth": _required_saved_parameter(serialized, parameter_sources, ("max_depth",), "max_depth"),
        "min_data_in_leaf": _required_saved_parameter(serialized, parameter_sources, ("min_data_in_leaf", "min_child_samples"), "min_data_in_leaf"),
        "min_sum_hessian_in_leaf": _required_saved_parameter(serialized, parameter_sources, ("min_sum_hessian_in_leaf", "min_child_weight"), "min_sum_hessian_in_leaf"),
        "lambda_l1": _required_saved_parameter(serialized, parameter_sources, ("lambda_l1", "reg_alpha"), "lambda_l1"),
        "lambda_l2": _required_saved_parameter(serialized, parameter_sources, ("lambda_l2", "reg_lambda"), "lambda_l2"),
        "min_gain_to_split": _required_saved_parameter(serialized, parameter_sources, ("min_gain_to_split", "min_split_gain"), "min_gain_to_split"),
        "max_bin": _required_saved_parameter(serialized, parameter_sources, ("max_bin",), "max_bin"),
        "use_missing": _required_saved_parameter(serialized, parameter_sources, ("use_missing",), "use_missing"),
        "zero_as_missing": _required_saved_parameter(serialized, parameter_sources, ("zero_as_missing",), "zero_as_missing"),
    }
    iterations = {int(x["iterations"]) for x in models}
    if len(iterations) != 1 or int(float(inherited["num_boost_round"])) != next(iter(iterations)):
        raise ValueError("serialized CTRL rounds differ from booster iteration count")
    seed_params = {key: value for key, value in serialized.items()
                   if "seed" in key.lower() or "random_state" in key.lower()}
    if not seed_params:
        raise ValueError("saved CTRL booster does not serialize seed-related parameters")
    overrides = {"objective": "regression_l2", "boosting_type": "gbdt",
                 "feature_fraction": 1.0, "bagging_fraction": 1.0,
                 "bagging_freq": 0, "num_threads": 1}
    aliases_to_drop = {"num_iteration", "n_estimators", "eta", "min_child_samples", "min_child_weight",
                       "reg_alpha", "reg_lambda", "min_split_gain"}
    fit_params = {key: value for key, value in bp.items() if key not in aliases_to_drop}
    fit_params.update({"num_iterations": int(float(inherited["num_boost_round"])),
                       "learning_rate": float(inherited["learning_rate"]),
                       "num_leaves": int(inherited["num_leaves"]),
                       "max_depth": int(inherited["max_depth"]),
                       "min_data_in_leaf": int(inherited["min_data_in_leaf"]),
                       "min_sum_hessian_in_leaf": float(inherited["min_sum_hessian_in_leaf"]),
                       "lambda_l1": float(inherited["lambda_l1"]),
                       "lambda_l2": float(inherited["lambda_l2"]),
                       "min_gain_to_split": float(inherited["min_gain_to_split"]),
                       "max_bin": int(inherited["max_bin"]),
                       "use_missing": bool(inherited["use_missing"]),
                       "zero_as_missing": bool(inherited["zero_as_missing"])})
    fit_params.update(seed_params)
    fit_params.update(overrides)
    fit_params["num_iterations"] = next(iter(iterations))
    config = {
        "schemaVersion": 1, "experiment": "INTRA_CROP_PANEL_CAND_B_V1",
        "status": "resolved", "sourceManifestPath": str(manifest_path),
        "sourceManifestSha256": source_digest, "savedCtrlModelCount": len(models),
        "savedCtrlModels": [x["identity"] for x in models],
        "serializedBoosterParameters": bp, "serializedSklearnParameters": sp,
        "serializedBoosterParameterText": representative["rawParameterBlock"],
        "serializedCanonicalParameters": serialized,
        "inherited": inherited, "seedParameters": seed_params,
        "featureNames": representative["featureNames"],
        "commonOverrides": {**overrides, "early_stopping": False},
        "fitParams": fit_params,
        "environment": {"python": platform.python_version(),
                        "lightgbm": importlib.metadata.version("lightgbm"),
                        "numpy": importlib.metadata.version("numpy"),
                        "joblib": importlib.metadata.version("joblib")},
        "configSha256": None,
    }
    config["configSha256"] = _sha(_canonical({k: v for k, v in config.items() if k != "configSha256"}))
    if output_path is not None:
        _write_once(output_path, config)
    return config


def _transform_basic40(vector: tuple[float, ...] | list[float]) -> list[float]:
    """Express price/volume scale features relative to own-series current state."""
    values = [float(x) for x in vector]
    if len(values) != len(BASE_FEATURE_NAMES) or not all(math.isfinite(x) for x in values):
        raise ValueError("expected finite basic40 feature vector")
    last = values[BASE_FEATURE_NAMES.index("lastPriceLog")]
    for lag in modeling.LAGS:
        available_idx = BASE_FEATURE_NAMES.index(f"lag{lag}Available")
        price_idx = BASE_FEATURE_NAMES.index(f"lag{lag}PriceLog")
        if values[available_idx] == 1.0:
            values[price_idx] -= last
        elif values[available_idx] != 0.0:
            raise ValueError("lag availability mask is not binary")
    for window in modeling.WINDOWS:
        observed_idx = BASE_FEATURE_NAMES.index(f"window{window}ObservedDays")
        if values[observed_idx] < 0 or values[observed_idx] > window:
            raise ValueError("window observation count outside valid range")
        if values[observed_idx] > 0:
            for stat in ("MeanLogPrice", "MinLogPrice", "MaxLogPrice"):
                idx = BASE_FEATURE_NAMES.index(f"window{window}{stat}")
                values[idx] -= last
    for window in (7, 28):
        observed_idx = BASE_FEATURE_NAMES.index(f"window{window}ObservedDays")
        volume_idx = BASE_FEATURE_NAMES.index(f"window{window}LogVolumeKg")
        ref_observed = values[BASE_FEATURE_NAMES.index("window90ObservedDays")]
        if values[observed_idx] > 0 and ref_observed > 0:
            ref_volume = values[BASE_FEATURE_NAMES.index("window90LogVolumeKg")]
            values[volume_idx] -= ref_volume
    if not all(math.isfinite(x) for x in values):
        raise ValueError("scale transformation generated a nonfinite feature")
    return values


_INHERITED_CATEGORICAL_PARAMS = (
    "cat_l2", "cat_smooth", "max_cat_threshold", "max_cat_to_onehot", "min_data_per_group",
)


def _fit_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply saved native categorical defaults needed by the new series-id feature."""
    result = dict(config["fitParams"])
    serialized = config["serializedCanonicalParameters"]
    for key in _INHERITED_CATEGORICAL_PARAMS:
        if key not in serialized:
            raise ValueError(f"saved CTRL model did not serialize categorical parameter {key}")
        result[key] = serialized[key]
    return result


def _series_feature(series: Mapping[dt.date, tuple[float, float]], origin: dt.date,
                    target: dt.date, horizon: int, cache: dict[Any, Any]) -> dict[str, Any] | None:
    cutoff = origin - dt.timedelta(days=1)
    found = modeling._feature_row(dict(series), cutoff, target, horizon, cache)
    if found is None:
        return None
    source_day, raw = found
    age_from_origin = (origin - source_day).days
    if age_from_origin < 0 or age_from_origin > dashboard.MAX_SOURCE_AGE_DAYS:
        return None
    transformed = _transform_basic40(raw)
    days = cache.setdefault(("series_days",), tuple(series))
    source_idx = bisect_right(days, cutoff) - 1
    max_source = days[source_idx] if source_idx >= 0 else None
    if max_source is None or max_source > cutoff:
        raise ValueError("feature source date exceeds row cutoff")
    return {"sourceDate": source_day, "cutoff": cutoff,
            "ageFromOrigin": age_from_origin, "features": transformed,
            "p0": series[source_day][0]}


def _series_map(channels: Mapping[str, Any], selection: Mapping[str, Any],
                product: str, representative: Mapping[dt.date, tuple[float, float]]) -> tuple[list[dict[str, Any]], dict[int, dict[dt.date, tuple[float, float]]]]:
    selected = selection.get(product, {}).get("selected", [])
    records = [{"seriesId": 0, "channelId": ["representative"], "kind": "representative"}]
    series_by_id = {0: dict(representative)}
    for series_id, item in enumerate(selected, start=1):
        key = tuple(item["channelId"])
        observations = channels.get(product, {}).get(key, {})
        series = {}
        raw_counts = Counter()
        for day, record in observations.items():
            raw_counts["rawRows"] += 1
            if not isinstance(record, Mapping):
                raw_counts["invalidRecordRows"] += 1
                continue
            price = _positive(record.get("price"))
            volume = _positive(record.get("volumeKgSigned"))
            if price is None:
                raw_counts["nonpositiveOrInvalidPriceRows"] += 1
            if volume is None:
                raw_counts["nonpositiveOrInvalidVolumeRows"] += 1
            if price is None or volume is None:
                raw_counts["excludedModelRows"] += 1
                continue
            series[day] = (price, volume)
        records.append({"seriesId": series_id, "channelId": list(key), "kind": "detail",
                        "grain": item.get("grain"), "selectionCutoff": item.get("selectionCutoff"),
                        "selectionStatus": item.get("selectionStatus"), **dict(raw_counts)})
        series_by_id[series_id] = series
    return records, series_by_id


def _training_rows(series_by_id: Mapping[int, Mapping[dt.date, tuple[float, float]]],
                   fit_origin: dt.date, horizon: int, *,
                   feature_caches: dict[int, dict[Any, Any]] | None = None
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                               dict[tuple[dt.date, dt.date], dict[str, Any]]]:
    representative = series_by_id[0]
    caches = feature_caches if feature_caches is not None else {sid: {} for sid in series_by_id}
    if set(caches) != set(series_by_id):
        raise ValueError("feature cache series ids differ from source series")
    rep_rows = []
    panel_rows = []
    per_origin: dict[tuple[dt.date, dt.date], dict[str, Any]] = {}
    for target, (price, _) in sorted(representative.items()):
        if target < source_exp.TRAIN_START or target >= fit_origin:
            continue
        origin = target - dt.timedelta(days=horizon)
        rep_feat = _series_feature(representative, origin, target, horizon, caches[0])
        if rep_feat is None:
            continue
        rep_label = math.log(price / rep_feat["p0"])
        rep_row = {"seriesId": 0, "target": target, "origin": origin,
                   "features": rep_feat["features"], "label": rep_label,
                   "sourceDate": rep_feat["sourceDate"], "cutoff": rep_feat["cutoff"],
                   "p0": rep_feat["p0"], "targetPrice": price, "weight": 1.0}
        rep_rows.append(rep_row)
        panel_rep_row = dict(rep_row)
        aux_rows = []
        for series_id in sorted(sid for sid in series_by_id if sid != 0):
            own = series_by_id[series_id]
            target_value = own.get(target)
            if target_value is None:
                continue
            feature = _series_feature(own, origin, target, horizon, caches[series_id])
            if feature is None:
                continue
            label = math.log(target_value[0] / feature["p0"])
            aux_rows.append({"seriesId": series_id, "target": target, "origin": origin,
                             "features": feature["features"], "label": label,
                             "sourceDate": feature["sourceDate"], "cutoff": feature["cutoff"]})
        count = len(aux_rows)
        rep_weight = 0.5 if count else 1.0
        panel_rep_row["weight"] = rep_weight
        panel_rows.append(panel_rep_row)
        for row in aux_rows:
            row["weight"] = 0.5 / count
            panel_rows.append(row)
        per_origin[(origin, target)] = {"K": count, "representativeWeight": rep_weight,
                                        "auxiliaryWeightSum": math.fsum(r["weight"] for r in aux_rows),
                                        "totalWeight": rep_weight + math.fsum(r["weight"] for r in aux_rows),
                                        "auxiliarySeriesIds": [r["seriesId"] for r in aux_rows]}
    rep_rows.sort(key=lambda r: (r["target"], r["origin"], r["seriesId"]))
    panel_rows.sort(key=lambda r: (r["target"], r["origin"], r["seriesId"]))
    return rep_rows, panel_rows, per_origin


def _x_matrix(rows: list[Mapping[str, Any]]) -> np.ndarray:
    values = np.asarray([list(row["features"]) + [int(row["seriesId"])] for row in rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("panel feature matrix width/nonfinite check failed")
    return values


def _key_digest(rows: list[Mapping[str, Any]]) -> str:
    return _sha(_canonical([[r["seriesId"], r["origin"].isoformat(), r["target"].isoformat(), r["label"]]
                            for r in rows]))


def _input_weight_digest(rows: list[Mapping[str, Any]]) -> str:
    return _sha(_canonical([[r["seriesId"], r["origin"].isoformat(), r["target"].isoformat(),
                             r["features"], r["weight"]] for r in rows]))


def _per_origin_report(per_origin: Mapping[tuple[dt.date, dt.date], Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [{"origin": origin.isoformat(), "target": target.isoformat(), **values}
               for (origin, target), values in sorted(per_origin.items())]
    weights = [float(x["totalWeight"]) for x in ordered]
    counts = Counter(str(x["K"]) for x in ordered)
    return {"originCount": len(ordered), "KCounts": dict(sorted(counts.items(), key=lambda p: int(p[0]))),
            "representativeWeightSum": math.fsum(float(x["representativeWeight"]) for x in ordered),
            "auxiliaryWeightSum": math.fsum(float(x["auxiliaryWeightSum"]) for x in ordered),
            "totalWeightSum": math.fsum(weights),
            "allOriginWeightsEqualOne": all(math.isclose(x, 1.0, abs_tol=1e-12, rel_tol=0) for x in weights),
            "maxAbsOriginWeightError": max((abs(x - 1.0) for x in weights), default=0.0),
            "perOriginDigestSha256": _sha(_canonical(ordered))}


def _origin_weight_diagnostics(per_origin: Mapping[tuple[dt.date, dt.date], Mapping[str, Any]],
                               variant: str) -> list[dict[str, Any]]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown model variant for origin weights: {variant}")
    return [{"origin": origin.isoformat(), "target": target.isoformat(),
             "K": int(values["K"]), "totalWeight": 1.0,
             "representativeWeight": (1.0 if variant == "L" else float(values["representativeWeight"])),
             "auxiliaryWeight": (0.0 if variant == "L" else float(values["auxiliaryWeightSum"]))}
            for (origin, target), values in sorted(per_origin.items())]


def _series_support(rows: list[Mapping[str, Any]], series_records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in series_records:
        sid = record["seriesId"]
        selected = [r for r in rows if r["seriesId"] == sid]
        result.append({"seriesId": sid, "channelId": list(record["channelId"]),
                       "rows": len(selected),
                       "distinctTargets": len({r["target"] for r in selected}),
                       "weightSum": math.fsum(float(r["weight"]) for r in selected),
                       **{key: value for key, value in record.items()
                          if key not in ("seriesId", "channelId", "kind")}})
    return result


def _permute_auxiliary_labels(panel_rows: list[dict[str, Any]], per_origin: Mapping[tuple[dt.date, dt.date], Mapping[str, Any]],
                              *, product: str, horizon: int, fit_month: dt.date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q_rows = [dict(row) for row in panel_rows]
    blocks: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(q_rows):
        if row["seriesId"] == 0:
            continue
        target = row["target"]
        k_count = int(per_origin[(row["origin"], target)]["K"])
        blocks[(row["seriesId"], target.year, (target.month - 1) // 3 + 1, k_count)].append(i)
    changed_rows = 0
    changed_weight = 0.0
    unchanged_blocks = 0
    records = []
    for block_key, indices in sorted(blocks.items()):
        labels = [q_rows[i]["label"] for i in indices]
        seed_material = [MASTER_SEED, product, horizon, fit_month.isoformat(), list(block_key)]
        seed = int(_sha(_canonical(seed_material))[:16], 16)
        permutation = np.random.default_rng(seed).permutation(len(indices))
        permuted = [labels[int(j)] for j in permutation]
        changed = [a != b for a, b in zip(labels, permuted)]
        if not any(changed):
            unchanged_blocks += 1
        for i, label, is_changed in zip(indices, permuted, changed):
            q_rows[i]["label"] = label
            if is_changed:
                changed_rows += 1
                changed_weight += float(q_rows[i]["weight"])
        records.append({"block": list(block_key), "rows": len(indices), "seed": seed,
                        "changedRows": sum(changed),
                        "changedWeight": math.fsum(float(q_rows[i]["weight"]) for i, v in zip(indices, changed) if v)})
    aux_weight = math.fsum(float(r["weight"]) for r in q_rows if r["seriesId"] != 0)
    return q_rows, {"blockCount": len(blocks), "unchangedBlocks": unchanged_blocks,
                    "changedRows": changed_rows, "auxiliaryRows": sum(len(v) for v in blocks.values()),
                    "changedAuxiliaryWeight": changed_weight, "auxiliaryWeight": aux_weight,
                    "changedWeightedShare": changed_weight / aux_weight if aux_weight else 0.0,
                    "permutationBlocksSha256": _sha(_canonical(records))}


def _dataset_params(config: Mapping[str, Any]) -> dict[str, Any]:
    serialized = config["serializedCanonicalParameters"]
    dataset_keys = ("max_bin", "max_bin_by_feature", "min_data_in_bin",
                    "bin_construct_sample_cnt", "data_random_seed", "feature_pre_filter",
                    "is_enable_sparse", "enable_bundle", "use_missing", "zero_as_missing",
                    *_INHERITED_CATEGORICAL_PARAMS)
    result = {key: serialized[key] for key in dataset_keys if key in serialized}
    result["max_bin"] = int(config["inherited"]["max_bin"])
    return result


def _native_dataset(data: np.ndarray, labels: list[float], weights: list[float],
                    config: Mapping[str, Any], *, reference: Any = None):
    import lightgbm as lgb
    dataset = lgb.Dataset(data, label=np.asarray(labels, dtype=np.float64),
                          weight=np.asarray(weights, dtype=np.float64),
                          feature_name=list(FEATURE_NAMES), categorical_feature=[40],
                          params=_dataset_params(config),
                          reference=reference, free_raw_data=False)
    dataset.construct()
    return dataset


def _train_native(data: np.ndarray, labels: list[float], weights: list[float],
                  config: Mapping[str, Any], *, reference: Any = None):
    import lightgbm as lgb
    params = _fit_parameters(config)
    params["verbosity"] = -1
    dataset = _native_dataset(data, labels, weights, config, reference=reference)
    model = lgb.train(params=params, train_set=dataset,
                      num_boost_round=int(config["inherited"]["num_boost_round"]))
    return model, dataset


def _save_booster(model: Any, path: Path, identity: Mapping[str, Any]) -> tuple[str, str]:
    model_text = model.model_to_string()
    identity_sha = _sha(_canonical(identity))
    model_raw = model_text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != model_raw:
            raise ValueError(f"refusing to overwrite immutable model checkpoint {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(model_raw)
        temp.replace(path)
    _write_once(path.with_suffix(".identity.json"), {
        "identity": dict(identity), "identitySha256": identity_sha,
        "modelTextSha256": _sha(model_raw),
    })
    raw = path.read_bytes()
    return _sha(raw), identity_sha


def _load_booster(model_path: Path, sidecar_path: Path, identity: Mapping[str, Any]):
    import lightgbm as lgb
    sidecar = _read_json(sidecar_path)
    if sidecar.get("identitySha256") != _sha(_canonical(identity)) or sidecar.get("identity") != dict(identity):
        raise ValueError(f"checkpoint identity mismatch: {model_path}")
    text = model_path.read_text(encoding="utf-8")
    if sidecar.get("modelTextSha256") != _sha(text.encode()):
        raise ValueError(f"checkpoint model text hash mismatch: {model_path}")
    return lgb.Booster(model_str=text), _file_sha(model_path), sidecar["identitySha256"]


def _train_fold(product: str, horizon: int, fit_month: dt.date, fit_origin: dt.date,
                rep_rows: list[dict[str, Any]], panel_rows: list[dict[str, Any]],
                per_origin: Mapping[tuple[dt.date, dt.date], Mapping[str, Any]],
                series_records: list[Mapping[str, Any]], config: Mapping[str, Any],
                phase_dir: Path, *, include_q: bool,
                execution_identity_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = phase_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rep_x = _x_matrix(rep_rows)
    panel_x = _x_matrix(panel_rows)
    rep_keys = _key_digest(rep_rows)
    panel_weights = [float(r["weight"]) for r in panel_rows]
    identity_base = {"product": product, "horizon": horizon, "fitMonth": fit_month.isoformat(),
                     "fitOrigin": fit_origin.isoformat(), "representativeKeysSha256": rep_keys,
                     "representativeInputWeightSha256": _input_weight_digest(rep_rows),
                     "panelInputWeightSha256": _input_weight_digest(panel_rows),
                     "configSha256": config["configSha256"], "featureNames": list(FEATURE_NAMES),
                     "sourceCacheFileSha256": _file_sha(SOURCE_CACHE),
                     "executionIdentitySha256": execution_identity_sha}
    data_support = _series_support(panel_rows, series_records)
    train_models = {}
    model_records = {}
    l_path = model_dir / f"{product}-h{horizon:02d}-{fit_month:%Y-%m}-L.txt"
    l_sidecar = l_path.with_suffix(".identity.json")
    l_identity = {**identity_base, "variant": "L", "trainingRowsSha256": _sha(_canonical([
        [r["origin"].isoformat(), r["target"].isoformat(), r["label"], 1.0] for r in rep_rows]))}
    if l_path.exists() != l_sidecar.exists():
        raise ValueError("L checkpoint or identity sidecar is incomplete")
    if l_path.exists():
        l_model, l_hash, l_identity_sha = _load_booster(l_path, l_sidecar, l_identity)
        l_state = "reused"
    else:
        l_model, _ = _train_native(rep_x, [r["label"] for r in rep_rows], [1.0] * len(rep_rows), config)
        l_hash, l_identity_sha = _save_booster(l_model, l_path, l_identity)
        _write_once(l_sidecar, {"identity": l_identity, "identitySha256": l_identity_sha,
                                "modelTextSha256": _sha(l_model.model_to_string().encode())})
        l_state = "fit"
    train_models["L"] = l_model
    model_records["L"] = {"status": l_state, "modelFile": str(l_path.relative_to(phase_dir.parent)),
                          "modelFileSha256": l_hash, "checkpointIdentitySha256": l_identity_sha,
                          "representativeKeysSha256": rep_keys}
    variant_info = {"L": {"trainingRows": len(rep_rows), "representativeKeysSha256": rep_keys}}

    aux_rows = [row for row in panel_rows if row["seriesId"] != 0]
    if not aux_rows:
        for variant in ("P", "Q"):
            if variant == "Q" and not include_q:
                model_records[variant] = {"status": "not_run", "reason": "Q is validation-only"}
                continue
            train_models[variant] = l_model
            model_records[variant] = {"status": "aliased_no_support", "modelFile": model_records["L"]["modelFile"],
                                      "modelFileSha256": l_hash, "checkpointIdentitySha256": l_identity_sha,
                                      "representativeKeysSha256": rep_keys}
            variant_info[variant] = {"trainingRows": len(panel_rows), "auxiliaryRows": 0,
                                     "sameInputWeightAsL": True}
        q_diagnostics = {"blockCount": 0, "unchangedBlocks": 0, "changedRows": 0,
                         "auxiliaryRows": 0, "changedWeightedShare": 0.0,
                         "changedAuxiliaryWeight": 0.0, "auxiliaryWeight": 0.0}
        pq_identity = {"sameRepresentativeKeyDigest": True, "sameInputWeightDigest": True,
                       "sameBinningIdentity": True, "reason": "no_auxiliary_rows_alias_L"}
    else:
        # P and Q share this constructed Dataset binning; no construction setting changes after bins exist.
        p_dataset = _native_dataset(panel_x, [r["label"] for r in panel_rows],
                                    panel_weights, config)
        p_identity = {**identity_base, "variant": "P", "trainingLabelsSha256": _sha(_canonical([r["label"] for r in panel_rows]))}
        p_path = model_dir / f"{product}-h{horizon:02d}-{fit_month:%Y-%m}-P.txt"
        p_sidecar = p_path.with_suffix(".identity.json")
        if p_path.exists() != p_sidecar.exists():
            raise ValueError("P checkpoint or identity sidecar is incomplete")
        if p_path.exists():
            p_model, p_hash, p_identity_sha = _load_booster(p_path, p_sidecar, p_identity)
            p_state = "reused"
        else:
            import lightgbm as lgb
            p_model = lgb.train(params={**_fit_parameters(config), "verbosity": -1},
                                train_set=p_dataset,
                                num_boost_round=int(config["inherited"]["num_boost_round"]))
            p_hash, p_identity_sha = _save_booster(p_model, p_path, p_identity)
            p_state = "fit"
        train_models["P"] = p_model
        model_records["P"] = {"status": p_state, "modelFile": str(p_path.relative_to(phase_dir.parent)),
                               "modelFileSha256": p_hash, "checkpointIdentitySha256": p_identity_sha,
                               "representativeKeysSha256": rep_keys}
        variant_info["P"] = {"trainingRows": len(panel_rows), "auxiliaryRows": len(aux_rows),
                              "inputWeightDigestSha256": _input_weight_digest(panel_rows)}
        q_path = model_dir / f"{product}-h{horizon:02d}-{fit_month:%Y-%m}-Q.txt"
        q_sidecar = q_path.with_suffix(".identity.json")
        if include_q:
            q_rows, q_diagnostics = _permute_auxiliary_labels(panel_rows, per_origin, product=product,
                                                               horizon=horizon, fit_month=fit_month)
            q_labels = [r["label"] for r in q_rows]
            q_dataset = _native_dataset(_x_matrix(q_rows), q_labels, panel_weights, config,
                                        reference=p_dataset)
            q_identity = {**identity_base, "variant": "Q", "trainingLabelsSha256": _sha(_canonical(q_labels)),
                          "binReferenceVariant": "P", "binReferenceInputWeightSha256": _input_weight_digest(panel_rows)}
            if q_path.exists() != q_sidecar.exists():
                raise ValueError("Q checkpoint or identity sidecar is incomplete")
            if q_path.exists():
                q_model, q_hash, q_identity_sha = _load_booster(q_path, q_sidecar, q_identity)
                q_state = "reused"
            else:
                import lightgbm as lgb
                q_model = lgb.train(params={**_fit_parameters(config), "verbosity": -1},
                                    train_set=q_dataset,
                                    num_boost_round=int(config["inherited"]["num_boost_round"]))
                q_hash, q_identity_sha = _save_booster(q_model, q_path, q_identity)
                q_state = "fit"
            train_models["Q"] = q_model
            model_records["Q"] = {"status": q_state, "modelFile": str(q_path.relative_to(phase_dir.parent)),
                                    "modelFileSha256": q_hash, "checkpointIdentitySha256": q_identity_sha,
                                    "representativeKeysSha256": rep_keys}
            variant_info["Q"] = {"trainingRows": len(q_rows), "auxiliaryRows": len(aux_rows),
                                  "inputWeightDigestSha256": _input_weight_digest(q_rows),
                                  "permutedLabelsSha256": _sha(_canonical(q_labels))}
            pq_identity = {"sameRepresentativeKeyDigest": rep_keys == model_records["P"]["representativeKeysSha256"],
                           "sameInputWeightDigest": _input_weight_digest(q_rows) == _input_weight_digest(panel_rows),
                           "sameBinningIdentity": p_dataset in q_dataset.get_ref_chain(),
                           "binReferenceVariant": "P",
                           "binReferenceInputWeightSha256": _input_weight_digest(panel_rows)}
        else:
            model_records["Q"] = {"status": "not_run", "reason": "Q is validation-only"}
            q_diagnostics = {"status": "not_run", "blockCount": 0, "unchangedBlocks": 0,
                             "changedRows": 0, "auxiliaryRows": 0, "changedWeightedShare": 0.0}
            pq_identity = {"sameRepresentativeKeyDigest": True,
                           "sameInputWeightDigest": True, "sameBinningIdentity": True,
                           "reason": "Q is validation-only"}

    representative_train = {}
    rep_labels = np.asarray([r["label"] for r in rep_rows], dtype=np.float64)
    rep_price = np.asarray([r["p0"] for r in rep_rows], dtype=np.float64)
    rep_actual = np.asarray([float(row["targetPrice"]) for row in rep_rows], dtype=np.float64)
    rep_predictions = {}
    for variant, model in train_models.items():
        pred = np.asarray(model.predict(rep_x), dtype=np.float64)
        if len(pred) != len(rep_rows) or not np.isfinite(pred).all():
            raise ValueError(f"{variant} representative training predictions are nonfinite")
        rep_predictions[variant] = pred
        pred_price = np.asarray([source_exp._price_from_log_ratio(float(p0), float(ratio))
                                 for p0, ratio in zip(rep_price, pred)], dtype=np.float64)
        errors = rep_labels - pred
        price_mape = float(np.mean(np.abs(rep_actual - pred_price) / rep_actual) * 100)
        representative_train[variant] = {"n": len(rep_rows),
            "logMse": float(np.mean(errors ** 2)),
            "logMae": float(np.mean(np.abs(errors))),
            "priceMAE": float(np.mean(np.abs(rep_actual - pred_price))),
            "priceMAPE": price_mape}
    decomposition = {}
    for left, right in (("L", "P"), ("Q", "P")):
        if left not in rep_predictions or right not in rep_predictions:
            continue
        delta = rep_predictions[right] - rep_predictions[left]
        e_left = rep_labels - rep_predictions[left]
        terms_move = delta ** 2
        terms_cross = -2 * e_left * delta
        squared_diff = ((rep_labels - rep_predictions[right]) ** 2 -
                       (rep_labels - rep_predictions[left]) ** 2)
        identity_error = squared_diff - (terms_move + terms_cross)
        decomposition[f"{right}-{left}"] = {"n": len(rep_rows),
            "deltaSquaredSum": float(np.sum(terms_move)), "crossTermSum": float(np.sum(terms_cross)),
            "squaredErrorDifferenceSum": float(np.sum(squared_diff)),
            "identityMaxAbsError": float(np.max(np.abs(identity_error), initial=0.0))}
    record = {"product": product, "horizon": horizon, "fitMonth": fit_month.strftime("%Y-%m"),
              "fitOrigin": fit_origin.isoformat(), "modelVariants": model_records,
              "representativeKeyDigestSha256": rep_keys,
              "trainingRowsRepresentative": len(rep_rows), "trainingRowsPanel": len(panel_rows),
              "trainingFirstTarget": rep_rows[0]["target"].isoformat() if rep_rows else None,
              "trainingLastTarget": rep_rows[-1]["target"].isoformat() if rep_rows else None,
              "timeIntegrity": {"allTrainingTargetsBeforeFitOrigin": all(r["target"] < fit_origin for r in panel_rows),
                                "allFeatureDatesAtOrBeforeCutoff": all(r["sourceDate"] <= r["cutoff"] for r in panel_rows)},
              "seriesSupport": data_support, "perOriginSupportWeightSummary": _per_origin_report(per_origin),
              "representativeTrainDiagnostics": representative_train,
              "representativeTrainLogDecomposition": decomposition,
              "permutationDiagnostics": q_diagnostics,
              "pqIdentity": pq_identity, "variants": variant_info,
              "fitStatus": "complete"}
    return record, train_models


def _execution_identity(config: Mapping[str, Any], source_cache_file_sha: str) -> dict[str, Any]:
    from . import (chronos2_market_context, equal_info_online_forecasts,
                   equal_info_online_score,
                   full_history_return_eval, intra_crop_panel_score,
                   research_progress_diagnostic)
    modules = {
        "panel": Path(__file__), "externalReference": Path(source_exp.__file__),
        "modeling": Path(modeling.__file__), "dashboard": Path(dashboard.__file__),
        "baselineBuilder": Path(chronos2_market_context.__file__),
        "priceSnapshotOwner": Path(equal_info_online_forecasts.__file__),
        "scorer": Path(intra_crop_panel_score.__file__),
        "dmHelpers": Path(equal_info_online_score.__file__),
        "priceHelpers": Path(full_history_return_eval.__file__),
        "bootstrapHelpers": Path(research_progress_diagnostic.__file__),
    }
    source_hashes = {name: _file_sha(path) for name, path in sorted(modules.items())}
    runtime = {name: importlib.metadata.version(name)
               for name in ("lightgbm", "numpy", "scipy", "pandas", "joblib")}
    return {"sourceHashes": source_hashes, "sourceHashesSha256": _sha(_canonical(source_hashes)),
            "sourceCacheFileSha256": source_cache_file_sha,
            "sourceCacheInternalSha256": SOURCE_CACHE_INTERNAL_SHA256,
            "resolvedConfigSha256": config["configSha256"],
            "effectiveFitParams": _fit_parameters(config),
            "effectiveDatasetParams": _dataset_params(config),
            "python": platform.python_version(), "runtimeDependencies": runtime}


def _load_resolved_config(output_dir: Path) -> dict[str, Any]:
    config = _read_json(output_dir / "resolved-config.json")
    digest = config.get("configSha256")
    if digest != _sha(_canonical({key: value for key, value in config.items() if key != "configSha256"})):
        raise ValueError("resolved saved-CTRL config hash mismatch")
    if config.get("status") != "resolved" or config.get("sourceManifestSha256") != SOURCE_V_MANIFEST_SHA256:
        raise ValueError("resolved config is not bound to the pinned CAND_A V artifact")
    expected_environment = {"python": platform.python_version(),
                            **{name: importlib.metadata.version(name)
                               for name in ("lightgbm", "numpy", "joblib")}}
    if config.get("environment") != expected_environment:
        raise ValueError("resolved CTRL config runtime differs from the frozen execution environment")
    return config


def preflight(*, output_dir: Path = OUT) -> dict[str, Any]:
    """Verify reusable frozen inputs; perform no database extraction or model fitting."""
    report_path = output_dir / "preflight.json"
    if report_path.is_file():
        report = _read_json(report_path)
        unsigned = dict(report)
        digest = unsigned.pop("preflightSha256", None)
        config = _load_resolved_config(output_dir)
        current_cache_sha = _file_sha(SOURCE_CACHE)
        current_price_sha = _file_sha(PRICES_PATH)
        current_execution = _execution_identity(config, current_cache_sha)
        pinned_ok = (
            digest == _sha(_canonical(unsigned))
            and report.get("status") == "preflight_complete"
            and report.get("sourceCacheInternalSha256") == SOURCE_CACHE_INTERNAL_SHA256
            and report.get("sourceCacheFileSha256") == current_cache_sha
            and report.get("resolvedConfigSha256") == config.get("configSha256")
            and report.get("targetSnapshotSha256") == current_price_sha
            and report.get("targetSnapshotSha256") == PRICE_SNAPSHOT_SHA256
            and report.get("baselineHashes") == BASELINE_SHA256
            and all(_file_sha(BASELINE_PATHS[name]) == BASELINE_SHA256[name] for name in BASELINE_PATHS)
            and report.get("aOldVManifestSha256") == SOURCE_V_MANIFEST_SHA256
            and _file_sha(SOURCE_OUT / "V/forecast-manifest.json") == SOURCE_V_MANIFEST_SHA256
            and report.get("executionIdentity") == current_execution
            and report.get("sourceCacheHashesVerified") is True
            and report.get("resolvedConfigVerified") is True
            and report.get("targetSnapshotVerified") is True
            and report.get("baselineKeysVerified") is True
            and report.get("representativeKeyInheritanceVerified") is True
            and report.get("representativeKeyInheritance", {}).get("allMatched") is True
            and report.get("representativeKeyInheritance", {}).get("foldCount") == 120
        )
        if not pinned_ok:
            raise ValueError("existing immutable preflight artifact does not match current frozen inputs")
        return report
    cache_doc = _read_json(SOURCE_CACHE)
    if cache_doc.get("cacheSha256") != SOURCE_CACHE_INTERNAL_SHA256:
        raise ValueError("reusable CAND_A source cache differs from the pinned cache identity")
    channels, selection = source_exp.load_source_cache(SOURCE_CACHE)
    if not channels or set(selection) != set(PRODUCTS):
        raise ValueError("reusable CAND_A source cache lacks product channel support")
    config = _load_resolved_config(output_dir)
    price_raw = PRICES_PATH.read_bytes()
    if _sha(price_raw) != PRICE_SNAPSHOT_SHA256:
        raise ValueError("representative target snapshot differs from its pinned hash")
    prices = json.loads(price_raw)
    source_exp.validate_snapshot_range(prices)
    phase_keys = {}
    for phase in ("V", "T"):
        baseline = _read_json(BASELINE_PATHS[phase])
        if _file_sha(BASELINE_PATHS[phase]) != BASELINE_SHA256[phase]:
            raise ValueError(f"pinned E106-RHS {phase} baseline hash mismatch")
        phase_keys[phase] = len(source_exp._baseline_index(baseline, phase))
    old_manifest_path = SOURCE_OUT / "V/forecast-manifest.json"
    old_manifest = _read_json(old_manifest_path)
    if _file_sha(old_manifest_path) != SOURCE_V_MANIFEST_SHA256:
        raise ValueError("saved A_old V manifest differs from its independently pinned artifact")
    source_exp.validate_forecast_manifest(old_manifest, "V")
    if old_manifest.get("manifestSha256") != SOURCE_V_MANIFEST_CANONICAL_SHA256:
        raise ValueError("saved A_old V manifest canonical identity mismatch")
    key_inheritance = verify_representative_key_inheritance(old_manifest, prices, channels, selection)
    if not key_inheritance["allMatched"]:
        raise ValueError("representative training targets differ from frozen CAND_A checkpoint evidence")
    identity = _execution_identity(config, _file_sha(SOURCE_CACHE))
    report = {"schemaVersion": 1, "experiment": "INTRA_CROP_PANEL_CAND_B_V1",
              "status": "preflight_complete", "sourceCacheInternalSha256": cache_doc["cacheSha256"],
              "sourceCacheFileSha256": _file_sha(SOURCE_CACHE),
              "selectedChannelsByProduct": {product: len(selection[product].get("selected", []))
                                             for product in PRODUCTS},
              "resolvedConfigSha256": config["configSha256"],
              "baselineHashes": dict(BASELINE_SHA256), "baselineKeysByPhase": phase_keys,
              "targetSnapshotSha256": _sha(price_raw), "aOldVManifestSha256": SOURCE_V_MANIFEST_SHA256,
              "representativeKeyInheritance": key_inheritance,
              "executionIdentity": identity,
              "sourceCacheHashesVerified": True, "resolvedConfigVerified": True,
              "representativeKeyInheritanceVerified": key_inheritance["allMatched"],
              "baselineKeysVerified": all(count > 0 for count in phase_keys.values()),
              "targetSnapshotVerified": True}
    report["preflightSha256"] = _sha(_canonical(report))
    _write_once(report_path, report)
    return report


def verify_representative_key_inheritance(old_manifest: Mapping[str, Any], prices: Mapping[str, Any],
                                          channels: Mapping[str, Any],
                                          selection: Mapping[str, Any]) -> dict[str, Any]:
    """Compare exact representative keys/labels to pinned CAND_A checkpoint digests."""
    import joblib

    old_records = {}
    for record in old_manifest.get("fitRecords", []):
        fit_month = record.get("fitMonth")
        try:
            fit_month_day = dt.date.fromisoformat(fit_month)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid saved CAND_A fit month: {fit_month!r}") from exc
        if fit_month_day.day != 1:
            raise ValueError(f"saved CAND_A fit month is not a month boundary: {fit_month!r}")
        key = (record.get("product"), record.get("horizon"), fit_month_day.strftime("%Y-%m"))
        if key in old_records:
            raise ValueError(f"duplicate frozen CAND_A representative fold evidence: {key}")
        old_records[key] = record
    expected = {(product, horizon, month.strftime("%Y-%m"))
                for product in PRODUCTS for horizon in HORIZONS
                for month in (dt.date(2026, 4, 1), dt.date(2026, 5, 1), dt.date(2026, 6, 1))}
    if set(old_records) != expected:
        raise ValueError("saved CAND_A fit evidence does not cover all 120 V folds")
    product_records = {row.get("id"): row for row in prices.get("products", [])}
    checks = []
    feature_caches: dict[tuple[str, int], dict[int, dict[Any, Any]]] = {}
    legacy_feature_caches: dict[tuple[str, int], dict[Any, Any]] = {}
    selections_sha = source_exp._sha(source_exp._canonical(selection))
    for product, horizon, month_text in sorted(expected):
        representative = modeling._observed_series(product_records[product])
        fold_origin = dt.date.fromisoformat(month_text + "-01")
        cache = feature_caches.setdefault((product, horizon), {0: {}})
        rows, _, _ = _training_rows({0: representative}, dt.date.fromisoformat(month_text + "-01"),
                                    horizon, feature_caches=cache)
        prior = old_records[(product, horizon, month_text)]
        checkpoint_path = SOURCE_OUT / prior.get("modelFile", "")
        if not prior.get("modelFile") or _file_sha(checkpoint_path) != prior.get("modelFileSha256"):
            raise ValueError(f"saved CAND_A checkpoint file hash mismatch for {(product, horizon, month_text)}")
        bundle = joblib.load(checkpoint_path)
        checkpoint_identity = bundle.get("checkpointIdentity") if isinstance(bundle, Mapping) else None
        if not isinstance(checkpoint_identity, Mapping):
            raise ValueError(f"saved CAND_A checkpoint lacks identity for {(product, horizon, month_text)}")
        if source_exp._sha(source_exp._canonical(checkpoint_identity)) != prior.get("checkpointIdentitySha256"):
            raise ValueError(f"saved CAND_A checkpoint identity hash mismatch for {(product, horizon, month_text)}")
        identity_ok = (checkpoint_identity.get("product") == product
                       and checkpoint_identity.get("horizon") == horizon
                       and checkpoint_identity.get("fitOrigin") == fold_origin.isoformat()
                       and checkpoint_identity.get("sourceCacheSha256") == SOURCE_CACHE_INTERNAL_SHA256
                       and checkpoint_identity.get("channelSelectionSha256") == selections_sha
                       and checkpoint_identity.get("modelConfig") == source_exp.LIGHTGBM_CONFIG)
        selected_series = source_exp._channel_series(channels, selection, product)
        legacy_cache = legacy_feature_caches.setdefault((product, horizon), {})
        legacy_rows = []
        for target in sorted(representative):
            if target < source_exp.TRAIN_START or target >= fold_origin:
                continue
            origin = target - dt.timedelta(days=horizon)
            feature = source_exp._feature_row(representative, selected_series, origin, target, horizon,
                                              legacy_cache, include_path_audit=False)
            if feature is not None:
                legacy_rows.append({**feature, "target": target,
                                    "label": math.log(representative[target][0] / feature["p0"])})
        legacy_targets_sha = source_exp._sha(source_exp._canonical([
            {"target": row["target"].isoformat(), "label": row["label"],
             "basic40": row["basic40"], "candidate": row["candidate"]}
            for row in legacy_rows]))
        checkpoint_targets_sha = checkpoint_identity.get("trainingTargetsSha256")
        exact_saved_keys = (checkpoint_targets_sha == legacy_targets_sha
                            and len(legacy_rows) == checkpoint_identity.get("trainingRows")
                            and len(legacy_rows) == prior.get("trainingRows"))
        new_targets = [(row["target"], row["label"]) for row in rows]
        old_targets = [(row["target"], row["label"]) for row in legacy_rows]
        exact_cross_variant_keys = new_targets == old_targets
        first = rows[0]["target"].isoformat() if rows else None
        last = rows[-1]["target"].isoformat() if rows else None
        matched = (identity_ok and exact_saved_keys and exact_cross_variant_keys
                   and len(rows) == prior.get("trainingRows") and
                   first == prior.get("trainingFirstTarget") and
                   last == prior.get("trainingLastTarget") and
                   prior.get("trainingTargetsStrictlyBeforeFitOrigin") is True and
                   checkpoint_identity.get("trainingTargetsSha256") == legacy_targets_sha)
        checks.append({"product": product, "horizon": horizon, "fitMonth": month_text,
                       "representativeTrainingRows": len(rows), "firstTarget": first, "lastTarget": last,
                       "regeneratedKeyDigestSha256": _key_digest(rows),
                       "reconstructedCANDATargetsSha256": legacy_targets_sha,
                       "savedCANDATargetsSha256": checkpoint_targets_sha,
                       "exactSavedKeysAndLabelsMatch": exact_saved_keys,
                       "exactCANDAToCANDBKeysAndLabelsMatch": exact_cross_variant_keys,
                       "matchedSavedCheckpointEvidence": matched})
    return {"foldCount": len(checks), "allMatched": all(item["matchedSavedCheckpointEvidence"] for item in checks),
            "checksSha256": _sha(_canonical(checks)), "checks": checks}


def _baseline_prediction(value: Any, original_status: Any) -> dict[str, Any]:
    prediction = _positive(value)
    if prediction is None:
        return {"status": "not_issued", "prediction": None, "sourceStatus": original_status}
    return {"status": "issued", "prediction": prediction, "sourceStatus": original_status}


def _a_old_index() -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    path = SOURCE_OUT / "V/forecast-manifest.json"
    doc = _read_json(path)
    if _file_sha(path) != SOURCE_V_MANIFEST_SHA256:
        raise ValueError("saved A_old V manifest hash mismatch")
    source_exp.validate_forecast_manifest(doc, "V")
    result = {}
    for row in doc["forecasts"]:
        key = (row["product"], dt.date.fromisoformat(row["target"]), row["horizon"])
        if key in result:
            raise ValueError(f"duplicate saved A_old forecast key: {key}")
        result[key] = {name: _baseline_prediction(row.get(name, {}).get("prediction"),
                                                  row.get(name, {}).get("status"))
                       for name in ("CTRL_LGB", "CAND_A")}
    return result


def _candidate_not_issued(reason: str, *, p0: float | None = None,
                          p0_date: dt.date | None = None) -> dict[str, Any]:
    return {"status": "not_issued", "prediction": None,
            "p0": p0, "p0Date": p0_date.isoformat() if p0_date else None,
            "logRatio": None, "reason": reason}


def _candidate_prediction(model: Any, features: list[float], p0: float,
                          p0_date: dt.date) -> dict[str, Any]:
    x = np.asarray([features + [0]], dtype=np.float64)
    values = np.asarray(model.predict(x), dtype=np.float64)
    if values.shape != (1,) or not math.isfinite(float(values[0])):
        raise ValueError("model returned nonfinite or malformed log-ratio prediction")
    log_ratio = float(values[0])
    prediction = source_exp._price_from_log_ratio(p0, log_ratio)
    return {"status": "issued", "prediction": prediction,
            "p0": p0, "p0Date": p0_date.isoformat(), "logRatio": log_ratio}


def _forecast_p(model: Any, features: list[float] | None, p0: float | None,
                p0_date: dt.date | None, baseline_rhs: Mapping[str, Any]) -> dict[str, Any]:
    reason = "model_or_feature_unavailable"
    raw = _candidate_not_issued(reason, p0=p0, p0_date=p0_date)
    if model is not None and features is not None and p0 is not None and p0_date is not None:
        try:
            raw = _candidate_prediction(model, features, p0, p0_date)
            reason = ""
        except (ValueError, OverflowError, FloatingPointError) as exc:
            reason = f"model_output_unusable:{type(exc).__name__}"
    if raw["status"] == "issued":
        return {**raw, "modelUsed": True, "rawModelStatus": "issued",
                "rawModelPrediction": raw["prediction"], "rawModelLogRatio": raw["logRatio"],
                "fallbackReason": None, "withheldReason": None}
    rhs = _positive(baseline_rhs.get("prediction"))
    if baseline_rhs.get("status") == "issued" and rhs is not None:
        return {"status": "issued", "prediction": rhs, "modelUsed": False,
                "rawModelStatus": "not_issued", "rawModelPrediction": None,
                "rawModelLogRatio": None, "p0": p0,
                "p0Date": p0_date.isoformat() if p0_date else None,
                "logRatio": None, "fallbackReason": reason, "withheldReason": None}
    return {"status": "not_issued", "prediction": None, "modelUsed": False,
            "rawModelStatus": "not_issued", "rawModelPrediction": None,
            "rawModelLogRatio": None, "p0": p0,
            "p0Date": p0_date.isoformat() if p0_date else None,
            "logRatio": None, "fallbackReason": reason,
            "withheldReason": "model_unavailable_and_E106_RHS_not_issued"}


def _month_key(day: dt.date) -> dt.date:
    return day.replace(day=1)




def build_forecasts(phase: str, *, output_dir: Path = OUT) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    cache_doc = _read_json(SOURCE_CACHE)
    channels, selection = source_exp.load_source_cache(SOURCE_CACHE)
    config = _load_resolved_config(output_dir)
    prices_raw = PRICES_PATH.read_bytes()
    prices = json.loads(prices_raw)
    baseline_path = BASELINE_PATHS[phase]
    baseline_doc = _read_json(baseline_path)
    baseline = source_exp._baseline_index(baseline_doc, phase)
    execution = _execution_identity(config, _file_sha(SOURCE_CACHE))
    execution_sha = _sha(_canonical(execution))
    target_start, target_end, _ = PHASES[phase]
    months = sorted({_month_key(row["origin"]) for row in baseline.values()
                     if target_start <= row["target"] <= target_end})
    legacy = _a_old_index() if phase == "V" else {}
    if phase == "V" and set(legacy) != set(baseline):
        raise ValueError("saved A_old forecast key set differs from V RHS key set")

    forecast_by_key: dict[tuple[str, dt.date, int], dict[str, Any]] = {}
    fits: list[dict[str, Any]] = []
    technical: dict[str, dict[str, Any]] = {}
    for product in PRODUCTS:
        product_record = next((item for item in prices.get("products", [])
                               if item.get("id") == product), None)
        if not isinstance(product_record, Mapping):
            raise ValueError(f"pinned target snapshot lacks product {product}")
        representative = modeling._observed_series(product_record)
        series_records, series_by_id = _series_map(channels, selection, product, representative)
        eval_feature_cache: dict[Any, Any] = {}
        product_fold_ok, failure_reasons = True, []
        for horizon in HORIZONS:
            training_feature_caches = {sid: {} for sid in series_by_id}
            horizon_keys = sorted((key for key in baseline if key[0] == product and key[2] == horizon),
                                  key=lambda key: (baseline[key]["origin"], key[1]))
            feature_by_key = {key: _series_feature(representative, baseline[key]["origin"],
                                                    baseline[key]["target"], horizon, eval_feature_cache)
                              for key in horizon_keys}
            for fit_month in months:
                month_keys = [key for key in horizon_keys
                              if _month_key(baseline[key]["origin"]) == fit_month]
                if not month_keys:
                    continue
                fit_base = {"product": product, "horizon": horizon,
                            "fitMonth": fit_month.strftime("%Y-%m"),
                            "fitOrigin": fit_month.isoformat()}
                try:
                    rep_rows, panel_rows, per_origin = _training_rows(
                        series_by_id, fit_month, horizon, feature_caches=training_feature_caches)
                    if len(rep_rows) < MIN_TRAIN_ROWS:
                        raise ValueError(f"representative support {len(rep_rows)} < {MIN_TRAIN_ROWS}")
                    fold, models = _train_fold(
                        product, horizon, fit_month, fit_month, rep_rows, panel_rows,
                        per_origin, series_records, config, output_dir / phase,
                        include_q=(phase == "V"), execution_identity_sha=execution_sha)
                    model_records = fold["modelVariants"]
                    for variant in ("L", "P", "Q"):
                        model_record = model_records.get(variant, {"status": "not_run"})
                        model_status = model_record.get("status", "not_run")
                        fit_status = ("not_run" if model_status == "not_run" else
                                      "aliased_no_support" if model_status == "aliased_no_support" else "complete")
                        fit_item = {**fit_base, "variant": variant, "status": fit_status,
                                    "modelStatus": fit_status,
                                    "representativeKeyDigest": fold["representativeKeyDigestSha256"],
                                    "support": (_series_support(rep_rows, series_records)
                                                if variant == "L" else fold["seriesSupport"]),
                                    "originWeightDiagnostics": _origin_weight_diagnostics(per_origin, variant),
                                    "perOriginSupportWeightSummary": fold["perOriginSupportWeightSummary"],
                                    "timeIntegrity": fold["timeIntegrity"],
                                    "representativeTrainDiagnostics": fold["representativeTrainDiagnostics"].get(variant),
                                    "modelFile": model_record.get("modelFile"),
                                    "modelFileSha256": model_record.get("modelFileSha256"),
                                    "checkpointIdentitySha256": model_record.get("checkpointIdentitySha256")}
                        if variant == "P":
                            fit_item["representativeTrainLogDecomposition"] = {
                                "P-L": fold["representativeTrainLogDecomposition"].get("P-L")}
                        if variant == "Q" and phase == "V":
                            fit_item["permutationDiagnostics"] = fold["permutationDiagnostics"]
                            fit_item["pqIdentity"] = fold["pqIdentity"]
                            fit_item["representativeTrainLogDecomposition"] = {
                                "P-Q": fold["representativeTrainLogDecomposition"].get("P-Q")}
                        fits.append(fit_item)
                except Exception as exc:
                    product_fold_ok = False
                    reason = f"{type(exc).__name__}: {exc}"
                    failure_reasons.append({**fit_base, "reason": reason})
                    models = {}
                    for variant in ("L", "P", "Q"):
                        fit_status = "failed" if variant != "Q" or phase == "V" else "not_run"
                        fits.append({**fit_base, "variant": variant, "status": fit_status,
                                     "reason": reason if fit_status == "failed" else "Q is validation-only",
                                     "representativeKeyDigest": None, "support": [],
                                     "originWeightDiagnostics": [],
                                     "timeIntegrity": {"allTrainingTargetsBeforeFitOrigin": False,
                                                       "allFeatureDatesAtOrBeforeCutoff": False},
                                     "pqIdentity": {"sameRepresentativeKeyDigest": False,
                                                    "sameInputWeightDigest": False,
                                                    "sameBinningIdentity": False}
                                     if variant == "Q" and phase == "V" else None})
                for key in month_keys:
                    base, feature = baseline[key], feature_by_key[key]
                    values = feature["features"] if feature is not None else None
                    p0 = feature["p0"] if feature is not None else None
                    p0_date = feature["sourceDate"] if feature is not None else None
                    rhs = _baseline_prediction(base["E106_RHS"], base["rhsStatus"])
                    e106 = _baseline_prediction(base["E106"], base["e106Status"])
                    candidate = {}
                    for variant in ("L", "P", "Q"):
                        model = models.get(variant)
                        if variant == "P":
                            candidate[variant] = _forecast_p(model, values, p0, p0_date, rhs)
                        elif variant == "Q" and phase != "V":
                            candidate[variant] = _candidate_not_issued("Q is validation-only")
                        elif model is None or feature is None:
                            candidate[variant] = _candidate_not_issued("model_or_feature_unavailable",
                                                                        p0=p0, p0_date=p0_date)
                        else:
                            try:
                                candidate[variant] = _candidate_prediction(model, values, p0, p0_date)
                            except (ValueError, OverflowError, FloatingPointError) as exc:
                                candidate[variant] = _candidate_not_issued(
                                    f"model_output_unusable:{type(exc).__name__}", p0=p0, p0_date=p0_date)
                    row = {"product": product, "origin": base["origin"].isoformat(),
                           "target": base["target"].isoformat(), "horizon": horizon,
                           "fitMonth": fit_month.strftime("%Y-%m"),
                           "cutoff": (base["origin"] - dt.timedelta(days=1)).isoformat(),
                           "B": {"E106_RHS": rhs, "E106": e106}, "models": candidate}
                    if phase == "V":
                        row["A_old"] = legacy[key]
                    forecast_by_key[key] = row
        technical[product] = {"status": "complete" if product_fold_ok else "failed",
                              "reason": None if product_fold_ok else failure_reasons}
    if set(forecast_by_key) != set(baseline):
        raise ValueError("candidate forecast grid differs from frozen RHS baseline keys")
    forecasts = [forecast_by_key[key] for key in sorted(forecast_by_key)]
    doc = {"schemaVersion": 1, "experiment": "INTRA_CROP_PANEL_CAND_B_V1",
           "phase": phase, "targetValuesAttached": False, "protocol": protocol(),
           "sourceCacheSha256": cache_doc["cacheSha256"],
           "sourceCacheFileSha256": _file_sha(SOURCE_CACHE),
           "baselineManifestSha256": BASELINE_SHA256[phase],
           "targetSnapshotSha256": _sha(prices_raw),
           "resolvedConfigSha256": config["configSha256"],
           "executionIdentity": execution, "executionIdentitySha256": execution_sha,
           "forecasts": forecasts, "fits": fits, "technicalCompletion": technical,
           "forecastKeyCount": len(forecasts), "fitVariantCount": len(fits)}
    doc["manifestSha256"] = _sha(_canonical(doc))
    _write_once(output_dir / phase / "forecast-manifest.pre-replay.json", doc)
    return doc


def verify_saved_model_replay(phase: str, manifest: Mapping[str, Any], *, output_dir: Path = OUT) -> dict[str, Any]:
    if manifest.get("phase") != phase or manifest.get("targetValuesAttached") is not False:
        raise ValueError("replay requires an answer-free phase manifest")
    unsigned = dict(manifest)
    pre_sha = unsigned.pop("manifestSha256", None)
    if pre_sha != _sha(_canonical(unsigned)):
        raise ValueError("pre-replay manifest self-hash mismatch")
    config = _load_resolved_config(output_dir)
    current_execution = _execution_identity(config, _file_sha(SOURCE_CACHE))
    if manifest.get("executionIdentity") != current_execution:
        raise ValueError("replay code/runtime/source identity mismatch")
    if _file_sha(SOURCE_CACHE) != manifest.get("sourceCacheFileSha256"):
        raise ValueError("replay source cache file hash mismatch")
    if _file_sha(BASELINE_PATHS[phase]) != manifest.get("baselineManifestSha256"):
        raise ValueError("replay baseline manifest hash mismatch")
    channels, selection = source_exp.load_source_cache(SOURCE_CACHE)
    baseline_doc = _read_json(BASELINE_PATHS[phase])
    baseline = source_exp._baseline_index(baseline_doc, phase)
    records_by_fold: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    loaded_models = {}
    checkpoint_identities = []
    checkpoint_evidence = []
    for fit in manifest.get("fits", []):
        if fit.get("status") in ("failed", "not_run"):
            checkpoint_evidence.append({"product": fit.get("product"), "horizon": fit.get("horizon"),
                                        "fitMonth": fit.get("fitMonth"), "variant": fit.get("variant"),
                                        "status": fit.get("status"), "modelFileSha256": None,
                                        "checkpointIdentitySha256": None})
            continue
        fold_key = (fit["product"], fit["horizon"], fit["fitMonth"])
        records_by_fold[fold_key][fit["variant"]] = fit
        rel = fit.get("modelFile")
        if not rel:
            raise ValueError("complete fit lacks a persisted checkpoint reference")
        path = output_dir / rel
        sidecar_path = path.with_suffix(".identity.json")
        if not path.is_file() or _file_sha(path) != fit.get("modelFileSha256"):
            raise ValueError(f"saved checkpoint hash mismatch: {path}")
        sidecar = _read_json(sidecar_path)
        identity = sidecar.get("identity")
        identity_sha = _sha(_canonical(identity))
        if (identity_sha != fit.get("checkpointIdentitySha256") or
                sidecar.get("identitySha256") != identity_sha or
                sidecar.get("modelTextSha256") != _file_sha(path)):
            raise ValueError(f"saved checkpoint identity mismatch: {path}")
        model_key = str(path.resolve())
        if model_key not in loaded_models:
            booster, model_hash, _ = _load_booster(path, sidecar_path, identity)
            if model_hash != fit["modelFileSha256"] or booster.feature_name() != list(FEATURE_NAMES):
                raise ValueError(f"saved checkpoint features/hash mismatch: {path}")
            loaded_models[model_key] = booster
        fit = dict(fit)
        fit["_loadedModel"] = loaded_models[model_key]
        records_by_fold[fold_key][fit["variant"]] = fit
        checkpoint_identities.append([fit["product"], fit["horizon"], fit["fitMonth"],
                                      fit["variant"], fit["modelFileSha256"], identity_sha])
        checkpoint_evidence.append({"product": fit["product"], "horizon": fit["horizon"],
                                    "fitMonth": fit["fitMonth"], "variant": fit["variant"],
                                    "status": fit["status"], "modelFileSha256": fit["modelFileSha256"],
                                    "checkpointIdentitySha256": identity_sha})

    old = _a_old_index() if phase == "V" else {}
    prices = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    product_records = {item["id"]: item for item in prices.get("products", [])}
    representative_series = {product: modeling._observed_series(product_records[product])
                             for product in PRODUCTS}
    feature_cache = {product: {} for product in PRODUCTS}
    seen = set()
    replay_count = 0
    for row in manifest.get("forecasts", []):
        product = row["product"]
        target, origin, horizon = (dt.date.fromisoformat(row["target"]),
                                   dt.date.fromisoformat(row["origin"]), row["horizon"])
        key = (product, target, horizon)
        if key in seen or key not in baseline:
            raise ValueError("replay found duplicate or unexpected forecast key")
        seen.add(key)
        pinned = baseline[key]
        if origin != pinned["origin"]:
            raise ValueError("replay forecast origin differs from pinned baseline")
        for name, value_field, status_field in (("E106_RHS", "E106_RHS", "rhsStatus"),
                                                 ("E106", "E106", "e106Status")):
            stored = row["B"][name]
            expected = _baseline_prediction(pinned[value_field], pinned[status_field])
            if (stored.get("status"), stored.get("prediction"), stored.get("sourceStatus")) != (
                    expected["status"], expected["prediction"], expected["sourceStatus"]):
                raise ValueError(f"stored baseline forecast changed: {key}/{name}")
        if phase == "V" and row.get("A_old") != old.get(key):
            raise ValueError(f"stored A_old forecast changed: {key}")
        feature = _series_feature(representative_series[product], origin, target, horizon,
                                  feature_cache[product])
        fold_key = (product, horizon, row["fitMonth"])
        fold_fits = records_by_fold.get(fold_key, {})
        for variant in (("L", "P", "Q") if phase == "V" else ("L", "P")):
            stored = row["models"][variant]
            if variant == "P":
                issued = stored.get("rawModelStatus") == "issued"
            else:
                issued = stored.get("status") == "issued"
            fit = fold_fits.get(variant)
            if fit is None and variant == "P":
                # An explicit no-support alias reuses the exact L checkpoint.
                fit = fold_fits.get("L")
            model = fit.get("_loadedModel") if fit else None
            if not issued:
                continue
            if model is None or feature is None:
                raise ValueError("issued forecast cannot be replayed from its saved model and causal features")
            replayed = _candidate_prediction(model, feature["features"], feature["p0"], feature["sourceDate"])
            if (not math.isclose(replayed["prediction"], float(stored.get("rawModelPrediction")
                                                                  if variant == "P" else stored["prediction"]),
                                 rel_tol=1e-12, abs_tol=1e-12) or
                    not math.isclose(replayed["logRatio"], float(stored.get("rawModelLogRatio")
                                                                  if variant == "P" else stored["logRatio"]),
                                     rel_tol=1e-12, abs_tol=1e-12) or
                    not math.isclose(replayed["p0"], float(stored["p0"]), rel_tol=1e-12, abs_tol=1e-12) or
                    replayed["p0Date"] != stored.get("p0Date")):
                raise ValueError(f"saved model forecast replay mismatch: {key}/{variant}")
            replay_count += 1
    if seen != set(baseline):
        raise ValueError("replay forecast keys differ from pinned baseline keys")
    result = {"schemaVersion": 1, "phase": phase, "status": "pass",
              "preReplayManifestSha256": pre_sha,
              "forecastRowsSha256": _sha(_canonical(manifest.get("forecasts", []))),
              "sourceCacheFileSha256": _file_sha(SOURCE_CACHE),
              "baselineManifestSha256": _file_sha(BASELINE_PATHS[phase]),
              "resolvedConfigSha256": config["configSha256"],
              "executionIdentitySha256": manifest["executionIdentitySha256"],
              "checkpointIdentitySetSha256": _sha(_canonical(sorted(checkpoint_identities))),
              "checkpointEvidence": checkpoint_evidence,
              "checkpointFilesVerified": len(loaded_models),
              "forecastKeysVerified": len(seen), "directPredictionsReplayed": replay_count,
              "baselineCopiedValuesVerified": True,
              "AOldCopiedValuesVerified": True}
    result["replayVerificationSha256"] = _sha(_canonical(result))
    _write_once(output_dir / phase / "replay-verification.json", result)
    return result


def _verify_phase_actual_coverage(phase: str, forecasts: list[Mapping[str, Any]],
                                 actuals: Mapping[tuple[str, dt.date], Any]) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase for actual coverage: {phase}")
    start, end, origin_floor = PHASES[phase]
    first_legal_target = max(start, origin_floor + dt.timedelta(days=1))
    expected_targets = {(product, day.isoformat()) for product in PRODUCTS
                        for day in (first_legal_target + dt.timedelta(days=offset)
                                    for offset in range(max(0, (end - first_legal_target).days + 1)))}
    forecast_targets = {(row.get("product"), row.get("target")) for row in forecasts}
    if forecast_targets != expected_targets:
        missing_grid = sorted(expected_targets - forecast_targets)[:5]
        unexpected_grid = sorted(forecast_targets - expected_targets)[:5]
        raise ValueError(f"forecast target grid differs from legal phase keys; missing={missing_grid}, unexpected={unexpected_grid}")
    phase_positive = sorted((product, day.isoformat()) for (product, day), value in actuals.items()
                            if start <= day <= end and _positive(value) is not None)
    excluded = [(product, day) for product, day in phase_positive if day < first_legal_target.isoformat()]
    covered = [(product, day) for product, day in phase_positive if day >= first_legal_target.isoformat()]
    return {"status": "pass", "phaseStart": start.isoformat(), "phaseEnd": end.isoformat(),
            "originFloor": origin_floor.isoformat(), "firstLegalTarget": first_legal_target.isoformat(),
            "expectedLegalTargetPairCount": len(expected_targets),
            "forecastTargetPairCount": len(forecast_targets),
            "positiveActualKeyCountInPhase": len(phase_positive),
            "coveredPositiveActualKeyCount": len(covered),
            "excludedPositiveActualKeyCount": len(excluded),
            "excludedPositiveActualKeys": excluded,
            "legalTargetGridSha256": _sha(_canonical(sorted(expected_targets)))}


def run_phase(phase: str, *, output_dir: Path = OUT) -> dict[str, Any]:
    preflight(output_dir=output_dir)
    if phase == "T":
        decision_path = output_dir / "V/decision.json"
        if not decision_path.is_file():
            raise ValueError("T requires a completed V decision")
        decision = _read_json(decision_path)
        if decision.get("status") != "pass" or decision.get("TExecutionAllowed") is not True:
            raise ValueError("V failed its frozen gate; T is not permitted")
    preliminary = build_forecasts(phase, output_dir=output_dir)
    replay = verify_saved_model_replay(phase, preliminary, output_dir=output_dir)
    prices = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    actuals = source_exp._actuals_from_snapshot(prices)
    positive_actual_keys = sorted((product, day.isoformat()) for (product, day), value in actuals.items()
                                  if _positive(value) is not None)
    phase_actual_coverage = _verify_phase_actual_coverage(phase, preliminary["forecasts"], actuals)
    integrity_checks = {"sourceCacheHashesVerified": True,
                        "baselineKeysAndActualsVerified": phase_actual_coverage["status"] == "pass",
                        "forecastReplayVerified": replay.get("status") == "pass",
                        "resolvedConfigVerified": True}
    if not all(integrity_checks.values()):
        raise ValueError("information integrity checks did not all pass")
    final = {key: value for key, value in preliminary.items()
             if key != "manifestSha256"}
    final["integrityChecks"] = integrity_checks
    preflight_doc = _read_json(output_dir / "preflight.json")
    preflight_unsigned = dict(preflight_doc)
    preflight_sha = preflight_unsigned.pop("preflightSha256", None)
    if preflight_sha != _sha(_canonical(preflight_unsigned)):
        raise ValueError("preflight artifact self-hash mismatch")
    final["preflightArtifactPath"] = "preflight.json"
    final["preflightSha256"] = preflight_sha
    final["forecastReplayAuditPath"] = f"{phase}/replay-verification.json"
    final["forecastReplayAuditSha256"] = _file_sha(output_dir / phase / "replay-verification.json")
    final["replayVerificationSha256"] = replay["replayVerificationSha256"]
    final["positiveActualKeyCount"] = len(positive_actual_keys)
    final["positiveActualKeysSha256"] = _sha(_canonical(positive_actual_keys))
    final["phaseActualCoverage"] = phase_actual_coverage
    final["manifestSha256"] = _sha(_canonical(final))
    final_path = output_dir / phase / "forecast-manifest.json"
    _write_once(final_path, final)
    # No target labels are joined until after replay and final manifest persistence.
    from . import intra_crop_panel_score as scorer
    score_report, diagnostics, decision = scorer.score(
        final, prices, manifest_path=final_path, prices_path=PRICES_PATH)
    for path, artifact in ((output_dir / phase / "score-report.json", score_report),
                           (output_dir / phase / "diagnostics.json", diagnostics),
                           (output_dir / phase / "decision.json", decision)):
        _write_once(path, artifact)
    return {"phase": phase, "manifestSha256": final["manifestSha256"],
            "replayVerificationSha256": replay["replayVerificationSha256"],
            "decision": decision}


def protocol() -> dict[str, Any]:
    return {"id": "INTRA_CROP_PANEL_CAND_B_V1", "products": list(PRODUCTS),
            "horizons": list(HORIZONS), "phases": {k: [x.isoformat() for x in v] for k, v in PHASES.items()},
            "variants": list(VARIANTS), "featureTransform": "basic40 series-relative per frozen plan plus categorical series_id",
            "panelRows": "exact representative L eligible training keys; supplemental series require same origin/target observed target and causal own-series features",
            "weights": {"L": "representative=1", "P_Q": "K=0 representative=1; K>0 representative=0.5; each supplemental=0.5/K"},
            "qPermutation": {"blocks": ["series_id", "target_year_quarter", "K(origin)"], "seed": MASTER_SEED},
            "fitBoundary": "all training target dates < calendar-month fit origin",
            "fallback": "P policy falls back to saved E106_RHS; no clipping; technical product failure prevents complete status",
            "statistics": {"module": "intra_crop_panel_score", "owned_by": "panel_scoring"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("resolve-config", "preflight", "run-V", "run-T", "all"),
                        default="resolve-config")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--config-output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "resolve-config":
        config_path = args.config_output or args.output_dir / "resolved-config.json"
        config = extract_saved_ctrl_config(output_path=config_path)
        print(json.dumps({"status": "resolved", "configPath": str(config_path),
                          "configSha256": config["configSha256"],
                          "savedCtrlModelCount": config["savedCtrlModelCount"],
                          "parameters": config["inherited"]}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage == "preflight":
        result = preflight(output_dir=args.output_dir)
        print(json.dumps({"status": result["status"], "preflightSha256": result["preflightSha256"],
                          "sourceCacheSha256": result["sourceCacheInternalSha256"],
                          "baselineKeysByPhase": result["baselineKeysByPhase"]},
                         ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage in ("run-V", "run-T"):
        result = run_phase(args.stage[-1], output_dir=args.output_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    validation = run_phase("V", output_dir=args.output_dir)
    if validation["decision"].get("status") != "pass":
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 0
    test = run_phase("T", output_dir=args.output_dir)
    print(json.dumps({"V": validation, "T": test}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

