"""Offline, one-point comparison of a pinned ECMWF field and Open-Meteo capture.

The command is dry-run by default. ``--execute`` reads only the fixed local
artifacts, checks their pinned hashes and metadata, then reports one signed
temperature difference. It does not request data or establish equivalence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

GRIB_PATH = Path("data/raw/ecmwf_operational_probe/ecmwf_ifs_20250831T1200_step24_20261007T085738951102Z_809f14048504_2t.grib2")
METEO_PATH = Path("data/raw/open_meteo_single_runs/single_run_20250831T1200_20261007T083320487134Z_53e424fa5f2e.json")
GRIB_SHA256 = "0ea3631f06bc8d7f48e6814f5b51ed57402998a406eb1c2b8b9774f9a16c6579"
METEO_SHA256 = "53e424fa5f2e9abb23b48f6e845e3570cff70379a48336ab4486ca05561abef1"
ECCODES_TARGET = Path("/private/tmp/agri-ecmwf-eccodes")
LATITUDE = 34.57
LONGITUDE = 126.60
VALID_TIME = "2025-09-01T12:00"
RUN_DATE = 20250831
RUN_TIME = 1200
STEP_HOURS = 24
VALID_DATE = 20250901
VALID_HOUR = 1200
EARTH_RADIUS_KM = 6371.0088
DEFAULT_OUTPUT = Path("outputs/experiments/ecmwf-field-compare-one-point/report.json")


class ComparisonError(ValueError):
    """Raised when pinned inputs or their provenance do not meet the protocol."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _load_eccodes() -> Any:
    if str(ECCODES_TARGET) not in sys.path:
        sys.path.insert(0, str(ECCODES_TARGET))
    try:
        return importlib.import_module("eccodes")
    except ImportError as exc:
        raise ComparisonError(f"ecCodes Python bindings unavailable under {ECCODES_TARGET}") from exc


def _one_grib_handle(data: bytes, eccodes: Any) -> Any:
    # ecCodes' Python binding passes this object through to a C FILE* API;
    # BytesIO has no file descriptor and is not accepted by codes_grib_new_from_file.
    with tempfile.TemporaryFile(mode="w+b") as stream:
        stream.write(data)
        stream.flush()
        stream.seek(0)
        first = eccodes.codes_grib_new_from_file(stream)
        if first is None:
            raise ComparisonError("GRIB contains no message")
        second = eccodes.codes_grib_new_from_file(stream)
        if second is not None:
            eccodes.codes_release(second)
            eccodes.codes_release(first)
            raise ComparisonError("GRIB must contain exactly one message")
        try:
            declared_length = int(eccodes.codes_get(first, "totalLength"))
        except Exception as exc:
            eccodes.codes_release(first)
            raise ComparisonError("GRIB declared message length is unavailable") from exc
        if declared_length != len(data):
            eccodes.codes_release(first)
            raise ComparisonError("GRIB declared message length does not match input bytes")
        return first


def _get(eccodes: Any, handle: Any, key: str) -> Any:
    try:
        return eccodes.codes_get(handle, key)
    except Exception as exc:
        raise ComparisonError(f"GRIB metadata missing or unreadable: {key}") from exc


def _grib_temperature(data: bytes, eccodes: Any) -> dict[str, Any]:
    handle = _one_grib_handle(data, eccodes)
    try:
        keys = ("shortName", "paramId", "date", "time", "step", "validityDate",
                "validityTime", "units", "typeOfLevel", "level")
        metadata = {key: _get(eccodes, handle, key) for key in keys}
        if metadata["shortName"] != "2t" or int(metadata["paramId"]) != 167:
            raise ComparisonError("GRIB parameter is not 2m temperature (2t / 167)")
        if int(metadata["date"]) != RUN_DATE or int(metadata["time"]) != RUN_TIME:
            raise ComparisonError("GRIB reference date/time does not match the fixed run")
        if int(metadata["step"]) != STEP_HOURS:
            raise ComparisonError("GRIB forecast step does not match 24 hours")
        if int(metadata["validityDate"]) != VALID_DATE or int(metadata["validityTime"]) != VALID_HOUR:
            raise ComparisonError("GRIB validity time does not match the fixed verification time")
        if metadata["units"] != "K":
            raise ComparisonError("GRIB temperature unit is not Kelvin")
        if metadata["typeOfLevel"] != "heightAboveGround" or int(metadata["level"]) != 2:
            raise ComparisonError("GRIB level is not 2 metres above ground")
        try:
            latitudes = eccodes.codes_get_array(handle, "latitudes")
            longitudes = eccodes.codes_get_array(handle, "longitudes")
            values = eccodes.codes_get_values(handle)
        except Exception as exc:
            raise ComparisonError("could not decode GRIB grid coordinates or values") from exc
        if not (len(latitudes) == len(longitudes) == len(values)) or not len(values):
            raise ComparisonError("GRIB grid coordinates and values have inconsistent lengths")
        candidates = []
        for index, (lat, lon, value) in enumerate(zip(latitudes, longitudes, values)):
            if not (_finite_number(lat) and _finite_number(lon)):
                continue
            distance = _haversine_km(LATITUDE, LONGITUDE, float(lat), float(lon))
            candidates.append((distance, index, float(lat), float(lon), value))
        if not candidates:
            raise ComparisonError("GRIB has no valid grid coordinates")
        distance, index, lat, lon, kelvin = min(candidates, key=lambda item: item[0])
        if not _finite_number(kelvin):
            raise ComparisonError("nearest GRIB gridpoint temperature is null or non-finite")
        kelvin = float(kelvin)
        celsius = kelvin - 273.15
        if not math.isfinite(celsius):
            raise ComparisonError("decoded GRIB temperature is not finite")
        return {"metadata": metadata, "grid_point_index": index,
                "grid_latitude": lat, "grid_longitude": lon,
                "distance_km": distance, "temperature_kelvin": kelvin,
                "temperature_celsius": celsius}
    finally:
        eccodes.codes_release(handle)


def _meteo_temperature(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError("Open-Meteo source is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ComparisonError("Open-Meteo response root is not an object")
    tz = payload.get("timezone")
    offset = payload.get("utc_offset_seconds")
    if tz not in {"UTC", "GMT"} or offset != 0:
        raise ComparisonError("Open-Meteo source timezone is not UTC-equivalent")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ComparisonError("Open-Meteo hourly object is missing")
    times, values = hourly.get("time"), hourly.get("temperature_2m")
    if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
        raise ComparisonError("Open-Meteo time and temperature arrays are missing or mismatched")
    matches = [i for i, value in enumerate(times) if value == VALID_TIME]
    if len(matches) != 1:
        raise ComparisonError("Open-Meteo must contain exactly one requested valid timestamp")
    index = matches[0]
    value = values[index]
    if not _finite_number(value):
        raise ComparisonError("Open-Meteo temperature at the requested time is null or non-finite")
    units = payload.get("hourly_units", {}).get("temperature_2m")
    if units != "°C":
        raise ComparisonError("Open-Meteo temperature unit is not Celsius")
    lat, lon = payload.get("latitude"), payload.get("longitude")
    if not (_finite_number(lat) and _finite_number(lon)):
        raise ComparisonError("Open-Meteo response grid coordinates are missing or invalid")
    if abs(float(lat) - LATITUDE) > 0.5 or abs(float(lon) - LONGITUDE) > 0.5:
        raise ComparisonError("Open-Meteo response grid is outside the expected coordinate tolerance")
    return {"timestamp": VALID_TIME, "array_index": index,
            "temperature_celsius": float(value), "temperature_unit": units,
            "response_grid_latitude": float(lat), "response_grid_longitude": float(lon),
            "source_null_counts": {name: sum(item is None for item in items)
                                    for name, items in hourly.items() if isinstance(items, list)}}


def compare_bytes(grib_bytes: bytes, meteo_bytes: bytes, *, eccodes: Any,
                  expected_grib_sha256: str = GRIB_SHA256,
                  expected_meteo_sha256: str = METEO_SHA256) -> dict[str, Any]:
    grib_hash, meteo_hash = _sha256(grib_bytes), _sha256(meteo_bytes)
    if grib_hash != expected_grib_sha256:
        raise ComparisonError("ECMWF GRIB SHA-256 does not match the frozen source")
    if meteo_hash != expected_meteo_sha256:
        raise ComparisonError("Open-Meteo response SHA-256 does not match the frozen source")
    grib = _grib_temperature(grib_bytes, eccodes)
    meteo = _meteo_temperature(meteo_bytes)
    return {"schema_version": 1, "status": "one_point_comparison_complete",
            "claim_scope": "one coordinate and one valid time; no equivalence or model-performance claim",
            "protocol": {"latitude": LATITUDE, "longitude": LONGITUDE,
                         "valid_time_utc": VALID_TIME, "difference_sign": "ECMWF minus Open-Meteo"},
            "sources": {"ecmwf": {"path": str(GRIB_PATH), "sha256": grib_hash,
                                  "eccodes_version": eccodes.codes_get_api_version(), **grib},
                        "open_meteo": {"path": str(METEO_PATH), "sha256": meteo_hash, **meteo}},
            "temperature_difference_celsius": grib["temperature_celsius"] - meteo["temperature_celsius"]}


def compare_files(grib_path: str | Path = GRIB_PATH, meteo_path: str | Path = METEO_PATH,
                  *, eccodes: Any | None = None) -> dict[str, Any]:
    grib_path, meteo_path = Path(grib_path), Path(meteo_path)
    try:
        grib_bytes, meteo_bytes = grib_path.read_bytes(), meteo_path.read_bytes()
    except OSError as exc:
        raise ComparisonError("a pinned source artifact could not be read") from exc
    return compare_bytes(grib_bytes, meteo_bytes, eccodes=eccodes or _load_eccodes())


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="read and validate the pinned local artifacts")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="immutable comparison report path")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "grib_path": str(GRIB_PATH),
                          "grib_sha256": GRIB_SHA256, "meteo_path": str(METEO_PATH),
                          "meteo_sha256": METEO_SHA256, "valid_time_utc": VALID_TIME,
                          "inputs_read": False}, ensure_ascii=False))
        return 0
    try:
        report = compare_files()
        encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        _write_immutable(Path(args.output), encoded)
        print(json.dumps({"report": str(Path(args.output)), "sha256": _sha256(encoded), **report},
                         ensure_ascii=False, allow_nan=False))
        return 0
    except (ComparisonError, OSError, ValueError) as exc:
        print(f"comparison failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
