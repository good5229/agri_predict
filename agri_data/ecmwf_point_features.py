"""Decode frozen ECMWF collection checkpoints to deterministic station features.

Dry-run is the default and reads no collected artifacts. Execution is offline:
it validates local checkpoint provenance and decodes the existing single-field
GRIB ranges; it makes no network requests and performs no model fitting.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import ecmwf_field_collection as collection
from .ecmwf_field_compare import _finite_number, _get, _load_eccodes, _one_grib_handle

DEFAULT_RAW_DIR = Path("data/raw/ecmwf_field_collection")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/ecmwf-point-features-20261007")
STATIONS = (
    {"id": 100, "name": "daegwallyeong", "latitude": 37.67713, "longitude": 128.71834},
    {"id": 108, "name": "seoul", "latitude": 37.57142, "longitude": 126.9658},
    {"id": 261, "name": "haenam", "latitude": 34.55375, "longitude": 126.56907},
)
STEPS = (24, 72, 168)
PARAMS = ("2t", "tp")
FEATURE_ORDER = tuple(
    f"asos_{s['id']}_{param}_{step}h_{'c' if param == '2t' else 'mm'}"
    for s in STATIONS for step in STEPS for param in PARAMS
)
EARTH_RADIUS_KM = 6371.0088


class FeatureError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _run_dates() -> list[str]:
    return collection.expected_run_dates()


def _verify_checkpoint(root: Path, run_date: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    paths = sorted(root.glob(f"day_{run_date}_checkpoint_*.json"))
    if not paths:
        return None, None, "checkpoint_missing"
    path = paths[-1]
    try:
        payload = path.read_bytes()
        state = json.loads(payload)
    except (OSError, json.JSONDecodeError):
        return None, path.name, "checkpoint_unreadable"
    if not path.name.endswith(f"_{_sha256(payload)[:12]}.json"):
        return None, path.name, "checkpoint_hash_mismatch"
    if (state.get("schema_version") != 1 or state.get("source_scope") != collection.SOURCE_SCOPE
            or state.get("source_scope_hash") != collection.SCOPE_HASH
            or state.get("run_date") != run_date
            or not isinstance(state.get("captures"), dict)):
        return None, path.name, "checkpoint_scope_or_schema_mismatch"
    return state, path.name, None


def _verified_attempt(root: Path, attempt: Any, *, expected_kind: str,
                      expected_scope_hash: str) -> tuple[dict[str, Any] | None, bytes | None, str | None]:
    if not isinstance(attempt, dict) or attempt.get("complete") is not True:
        return None, None, "capture_incomplete"
    meta_name, raw_name = attempt.get("metadata_file"), attempt.get("raw_file")
    meta_hash, raw_hash = attempt.get("metadata_sha256"), attempt.get("sha256")
    if not all(isinstance(x, str) and Path(x).name == x for x in (meta_name, raw_name)):
        return None, None, "capture_artifact_path_invalid"
    try:
        metadata_bytes = (root / meta_name).read_bytes()
        raw = (root / raw_name).read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, json.JSONDecodeError):
        return None, None, "capture_artifact_missing_or_unreadable"
    if _sha256(metadata_bytes) != meta_hash or _sha256(raw) != raw_hash:
        return None, None, "capture_artifact_hash_mismatch"
    if (metadata.get("complete") is not True or metadata.get("sha256") != raw_hash
            or metadata.get("source_scope_hash") != expected_scope_hash
            or metadata.get("kind") != expected_kind):
        return None, None, "capture_metadata_scope_mismatch"
    return metadata, raw, None


def _decode_field(data: bytes, eccodes: Any, *, run_date: str, step: int,
                  param: str, stations: tuple[dict[str, Any], ...] = STATIONS) -> dict[int, dict[str, Any]]:
    handle = _one_grib_handle(data, eccodes)
    try:
        keys = ("shortName", "paramId", "date", "time", "step", "validityDate",
                "validityTime", "units", "typeOfLevel", "level", "totalLength", "missingValue")
        metadata = {key: _get(eccodes, handle, key) for key in keys}
        valid = dt.datetime.strptime(run_date, "%Y%m%d").replace(
            hour=12, tzinfo=dt.timezone.utc) + dt.timedelta(hours=step)
        expect_date = int(run_date)
        param_id = 167 if param == "2t" else 228
        unit = "K" if param == "2t" else "m"
        if (metadata["shortName"] != param or int(metadata["paramId"]) != param_id
                or int(metadata["date"]) != expect_date or int(metadata["time"]) != 1200
                or int(metadata["step"]) != step or int(metadata["validityDate"]) != int(valid.strftime("%Y%m%d"))
                or int(metadata["validityTime"]) != int(valid.strftime("%H%M"))
                or metadata["units"] != unit or int(metadata["totalLength"]) != len(data)):
            raise FeatureError("grib_metadata_mismatch")
        if param == "2t" and (metadata["typeOfLevel"] != "heightAboveGround" or int(metadata["level"]) != 2):
            raise FeatureError("grib_temperature_level_mismatch")
        if param == "tp" and metadata["typeOfLevel"] != "surface":
            raise FeatureError("grib_precipitation_level_mismatch")
        try:
            nearest = eccodes.codes_grib_find_nearest_multiple(
                handle, False,
                [float(station["latitude"]) for station in stations],
                [float(station["longitude"]) for station in stations])
        except Exception as exc:
            raise FeatureError("grib_nearest_grid_lookup_failed") from exc
        if len(nearest) != len(stations):
            raise FeatureError("grib_nearest_grid_result_count_mismatch")
        missing_value = metadata["missingValue"]
        result = {}
        for station, point in zip(stations, nearest):
            lat, lon = point.lat, point.lon
            raw_value, distance, index = point.value, point.distance, point.index
            if not all(_finite_number(item) for item in (lat, lon, raw_value, distance)):
                result[station["id"]] = {"error": f"nearest_grid_result_non_finite_station_{station['id']}"}
                continue
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                result[station["id"]] = {"error": f"nearest_grid_index_invalid_station_{station['id']}"}
                continue
            if _finite_number(missing_value) and float(raw_value) == float(missing_value):
                result[station["id"]] = {"error": f"nearest_grid_value_missing_station_{station['id']}"}
                continue
            converted = float(raw_value) - 273.15 if param == "2t" else float(raw_value) * 1000.0
            if not math.isfinite(converted):
                result[station["id"]] = {"error": f"converted_value_non_finite_station_{station['id']}"}
                continue
            result[station["id"]] = {
                "grib_metadata": metadata, "grid_point_index": int(index),
                "grid_latitude": float(lat), "grid_longitude": float(lon),
                "distance_km": float(distance), "raw_value": float(raw_value), "raw_unit": unit,
                "value": converted, "unit": "°C" if param == "2t" else "mm"}
        return result
    finally:
        eccodes.codes_release(handle)


def _parse_utc(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _feature_for(root: Path, state: dict[str, Any] | None, checkpoint: str | None,
                 run_date: str, stations: tuple[dict[str, Any], ...], step: int, param: str,
                 eccodes: Any, checkpoint_error: str | None) -> dict[int, dict[str, Any]]:
    unit = "°C" if param == "2t" else "mm"
    records = {station["id"]: {"value": None, "unit": unit,
                               "state": "unavailable", "reason": None}
               for station in stations}

    def fail(reason: str, *, state_name: str | None = None,
             extras: dict[str, Any] | None = None) -> dict[int, dict[str, Any]]:
        for record in records.values():
            record["reason"] = reason
            if state_name:
                record["state"] = state_name
            if extras:
                record.update(extras)
        return records

    if checkpoint_error:
        return fail(checkpoint_error)
    captures = state["captures"]
    index_entry = captures.get("indices", {}).get(f"{run_date}/{step}")
    index_attempt = index_entry.get("latest") if isinstance(index_entry, dict) else None
    index_meta, index_raw, error = _verified_attempt(root, index_attempt,
                                                      expected_kind="index", expected_scope_hash=collection.SCOPE_HASH)
    if error:
        return fail(f"index_{error}")
    field_entry = captures.get("fields", {}).get(f"{run_date}/{step}/{param}")
    field_attempt = field_entry.get("latest") if isinstance(field_entry, dict) else None
    field_meta, raw, error = _verified_attempt(root, field_attempt,
                                                expected_kind="field", expected_scope_hash=collection.SCOPE_HASH)
    if error:
        return fail(f"field_{error}")
    if field_meta.get("index_sha256") != index_meta.get("sha256") or field_meta.get("index_sha256") != _sha256(index_raw):
        return fail("field_index_provenance_mismatch")
    selected = index_meta.get("selected_records")
    selected_param = selected.get(param) if isinstance(selected, dict) else None
    if (not isinstance(selected_param, dict)
            or selected_param.get("_offset") != field_meta.get("requested_offset")
            or selected_param.get("_length") != field_meta.get("requested_length")):
        return fail("field_range_index_mismatch")
    if (field_meta.get("run_date") != run_date or field_meta.get("step_hours") != step
            or field_meta.get("parameter") != param):
        return fail("field_capture_identity_mismatch")
    last_modified = _parse_utc(field_meta.get("last_modified_utc"))
    origin_date = dt.datetime.strptime(run_date, "%Y%m%d").date() + dt.timedelta(days=1)
    origin = dt.datetime.combine(origin_date, dt.time(1), tzinfo=dt.timezone.utc)
    if last_modified is None:
        return fail("last_modified_missing_or_invalid",
                    extras={"last_modified_raw": field_meta.get("last_modified_raw")})
    lm_context = {"last_modified_utc": last_modified.isoformat(),
                  "last_modified_raw": field_meta.get("last_modified_raw"),
                  "last_modified_before_origin": last_modified < origin,
                  "origin_deadline_utc": origin.isoformat()}
    if not last_modified < origin:
        return fail("last_modified_not_before_origin", state_name="late", extras=lm_context)
    offset, length = field_meta.get("requested_offset"), field_meta.get("requested_length")
    if (isinstance(offset, bool) or not isinstance(offset, int) or isinstance(length, bool)
            or not isinstance(length, int) or offset < 0 or length <= 0 or len(raw) != length):
        return fail("field_range_metadata_invalid", extras=lm_context)
    try:
        decoded = _decode_field(raw, eccodes, run_date=run_date, step=step, param=param, stations=stations)
    except (FeatureError, TypeError, ValueError, OverflowError) as exc:
        return fail(str(exc) or "grib_decode_invalid", extras=lm_context)
    for station in stations:
        item = records[station["id"]]
        decoded_item = decoded[station["id"]]
        if "error" in decoded_item:
            item.update({"reason": decoded_item["error"], **lm_context})
            continue
        item.update({"state": "valid", "reason": None, **lm_context, **decoded_item,
                     "provenance": {"run_date": run_date, "step_hours": step, "parameter": param,
                                    "checkpoint_file": checkpoint,
                                    "index_sha256": field_meta.get("index_sha256"),
                                    "raw_file": field_attempt.get("raw_file"),
                                    "raw_sha256": field_attempt.get("sha256"),
                                    "metadata_file": field_attempt.get("metadata_file"),
                                    "metadata_sha256": field_attempt.get("metadata_sha256"),
                                    "requested_offset": offset, "requested_length": length,
                                    "captured_at_utc": field_meta.get("captured_at_utc")}})
    return records


def build_features(raw_dir: str | Path = DEFAULT_RAW_DIR, *, eccodes: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(raw_dir)
    decoder = eccodes or _load_eccodes()
    rows, checkpoint_hashes = [], {}
    for run_date in _run_dates():
        state, checkpoint_name, checkpoint_error = _verify_checkpoint(root, run_date)
        if checkpoint_name and state is not None:
            checkpoint_hashes[run_date] = _sha256((root / checkpoint_name).read_bytes())
        origin = (dt.datetime.strptime(run_date, "%Y%m%d").date() + dt.timedelta(days=1)).isoformat()
        features, provenance = {}, {}
        by_field = {}
        for step in STEPS:
            for param in PARAMS:
                by_field[(step, param)] = _feature_for(root, state, checkpoint_name, run_date,
                                                       STATIONS, step, param, decoder, checkpoint_error)
        for station in STATIONS:
            for step in STEPS:
                for param in PARAMS:
                    key = f"asos_{station['id']}_{param}_{step}h_{'c' if param == '2t' else 'mm'}"
                    item = by_field[(step, param)][station["id"]]
                    features[key] = item["value"]
                    provenance[key] = item
        valid = [key for key, item in provenance.items() if item["state"] == "valid"]
        rows.append({"origin_date": origin, "run_date": run_date,
                     "candidate_valid": len(valid) == len(FEATURE_ORDER),
                     "available_feature_count": len(valid), "features": features,
                     "feature_states": provenance})
    manifest = {"schema_version": 1, "status": "offline_ecmwf_point_features",
                "source_scope": collection.SOURCE_SCOPE, "source_scope_hash": collection.SCOPE_HASH,
                "source_checkpoint_hashes": checkpoint_hashes,
                "run_date_start": _run_dates()[0], "run_date_end": _run_dates()[-1],
                "origin_date_rule": "origin date is the calendar day after the 12UTC run date; eligible Last-Modified < origin date 01:00 UTC (10:00 KST)",
                "station_order": list(STATIONS), "steps_hours": list(STEPS), "parameters": list(PARAMS),
                "feature_order": list(FEATURE_ORDER), "feature_count": len(FEATURE_ORDER),
                "row_count": len(rows), "candidate_valid_rows": sum(row["candidate_valid"] for row in rows),
                "field_value_count": sum(row["available_feature_count"] for row in rows),
                "ecmwf_decoding": "2t Kelvin to Celsius; tp metres to millimetres without deaccumulation",
                "nearest_lookup": "ecCodes codes_grib_find_nearest_multiple; one C-level lookup for all three stations per field; exact-grid ties follow ecCodes returned grid index",
                "missing_value_policy": "null with explicit feature state/reason; no imputation",
                "claim_limit": "decoded nearest 0.25-degree grid values are not station observations or proven commodity-origin weather",
                "eccodes_version": str(decoder.codes_get_api_version())}
    return {"schema_version": 1, "manifest": manifest, "rows": rows}, manifest


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise
        finally:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="read and decode existing local collector artifacts")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "raw_dir": args.raw_dir,
                          "output_dir": args.output_dir, "inputs_read": False,
                          "network_requests": 0, "model_training": False,
                          "rows": len(_run_dates()), "features_per_row": len(FEATURE_ORDER)}, ensure_ascii=False))
        return 0
    try:
        result, manifest = build_features(args.raw_dir)
        output = Path(args.output_dir)
        feature_doc = {"schema_version": 1, "rows": result["rows"]}
        rows_bytes = _canonical(feature_doc)
        rows_hash = _sha256(rows_bytes)
        feature_name = f"features_{rows_hash}.json"
        manifest = {**manifest, "feature_file": feature_name, "feature_sha256": rows_hash}
        manifest_bytes = _canonical(manifest)
        manifest_hash = _sha256(manifest_bytes)
        manifest_name = f"manifest_{manifest_hash}.json"
        _write_immutable(output / feature_name, rows_bytes)
        _write_immutable(output / manifest_name, manifest_bytes)
        print(json.dumps({"output_dir": str(output), "feature_file": feature_name,
                          "rows_sha256": rows_hash, "manifest_file": manifest_name,
                          "manifest_sha256": manifest_hash, **manifest}, ensure_ascii=False))
        return 0
    except (FeatureError, OSError, ValueError, ImportError) as exc:
        print(f"feature decoding failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
