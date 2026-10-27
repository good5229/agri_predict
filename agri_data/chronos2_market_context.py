"""Frozen, answer-free Chronos-2 market-context retrospective.

This adapter only builds U/M inputs from the preserved daily snapshot and runs
one isolated model task per product/origin.  Statistical evaluation is
delegated to :mod:`chronos2_market_context_score` after forecasts are sealed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from . import dashboard, direct_horizon as dh, market_extension as me
from . import radish_horizon_sharing as rhs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/chronos2-market-context-v1-20261012"
MARKET_DB_PATH = ROOT / "data/agri.sqlite"
MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "e19c427c73837c00b5e932da5fbfede2929677ce"
MODEL_WEIGHT_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
MODEL_WEIGHT_SIZE = 477_930_472
MODEL_CONFIG_SHA256 = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
PACKAGE_VERSION = "2.1.0"
PACKAGE_WHEEL_SHA256 = "b266c7bd5f1dc3d835ce8a6a03c1552aa3f8cb64bb0c59a10e441df9029eaa8e"
SCORER_SHA256 = "6207dc2709a62db6a2935ea4a43a490f7134827b8aa69454739ec2024642be14"
PRICE_SHA256 = dh.PRICES_SHA256
PRODUCTS = tuple(dashboard.PRODUCTS)[:4]
MARKETS = tuple(dashboard.MARKETS)
HORIZONS = tuple(range(1, 11))
CONTEXT_DAYS = 730
PREDICTION_LENGTH = 11
PHASES = {"V": (dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1)),
          "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1))}
MIN_MARKET_DATE = PHASES["V"][2] - dt.timedelta(days=CONTEXT_DAYS)
MAX_MARKET_DATE = PHASES["T"][1] - dt.timedelta(days=2)
BASELINE_PATHS = {"V": rhs.OUT / "validation-manifest.json", "T": rhs.OUT / "test-manifest.json"}
BASELINE_SHA256 = {
    "V": "632e4968123715612738145cfab35f679cbf02ee9f733c2d470c9b991423beb6",
    "T": "bc4b453df2ded47b8968fc7907fecad024e7da7fa0e2a5e730099a233848cc76",
}
SCHEMA_VERSION = 1


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _typed_provenance(value: Any) -> Any:
    """Return JSON-safe provenance with map key types explicit and collision-free."""
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            typed_key = _typed_provenance_key(key)
            entries.append(( _canonical(typed_key), typed_key, _typed_provenance(item)))
        entries.sort(key=lambda row: row[0])
        return {"__mappingEntries__": [{"key": key, "value": item}
                                      for _, key, item in entries]}
    if isinstance(value, (list, tuple)):
        return [_typed_provenance(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__float__": repr(value)}
        return value
    raise TypeError(f"unsupported provenance value type: {type(value).__name__}")


def _typed_provenance_key(key: Any) -> dict[str, Any]:
    if key is None:
        return {"type": "null", "value": None}
    if isinstance(key, bool):
        return {"type": "bool", "value": key}
    if isinstance(key, int):
        return {"type": "int", "value": key}
    if isinstance(key, float):
        return {"type": "float", "value": repr(key)}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    if isinstance(key, tuple):
        return {"type": "tuple", "value": [_typed_provenance_key(item) for item in key]}
    raise TypeError(f"unsupported provenance key type: {type(key).__name__}")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_or_match(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"existing derived artifact differs from deterministic reconstruction: {path.name}")
        return
    _write_once(path, raw)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime() -> dict[str, Any]:
    names = ("chronos-forecasting", "torch", "transformers", "accelerate", "numpy")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "device": "mps", "dtype": "float32", "seed": 0,
            "evalMode": True, "predictBatchesJointly": False,
            "deterministicAlgorithmsEnabled": bool(torch.are_deterministic_algorithms_enabled()),
            "mpsBuilt": bool(torch.backends.mps.is_built()),
            "mpsAvailable": bool(torch.backends.mps.is_available())}


def _protocol() -> dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, "experiment": "CHRONOS2_MARKET_CONTEXT_V1",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                      "weightSha256": MODEL_WEIGHT_SHA256, "weightSize": MODEL_WEIGHT_SIZE,
                      "configSha256": MODEL_CONFIG_SHA256,
                      "publicHistory": {"revisionIntroduced": "95a9710e2596287d08352589f42634fa5abdf0a7",
                          "weightsFirstObservedAtUtc": "2025-10-30",
                          "pinnedRevisionCreatedAtUtc": "2025-11-05T10:32:11Z",
                          "pinnedRevisionHistoryUrl":
                              f"https://huggingface.co/{MODEL_ID}/commits/{MODEL_REVISION}",
                          "pinnedRevisionCommitUrl":
                              f"https://huggingface.co/{MODEL_ID}/commit/{MODEL_REVISION}",
                          "cutoffUtc": "2026-01-01T00:00:00Z"}},
            "rulesAssessment": {"status": "eligibility_unverified",
                "pretrainedWeightsAllowedExplicitly": False, "explicitBanFound": False,
                "offlinePageSha256": "accc34ed0db42eac658566280d3b3e8b650405038edbc7ad92b0b1922809e507",
                "reviewedAt": "2026-10-12"},
            "package": {"name": "chronos-forecasting", "version": PACKAGE_VERSION,
                        "wheelSha256": PACKAGE_WHEEL_SHA256},
            "input": {"contextCalendarDays": CONTEXT_DAYS, "priceUnit": "KRW_per_kg",
                      "missing": "NaN; no interpolation/fill", "U": ["representativePrice"],
                      "M": ["representativePrice", *MARKETS],
                      "marketAuxiliaryOrder": list(MARKETS)},
            "prediction": {"length": PREDICTION_LENGTH, "quantile": 0.5,
                           "evaluatedIndices": {str(h): h for h in HORIZONS},
                           "device": "mps", "dtype": "float32", "seed": 0,
                           "modelEval": True, "predictBatchesJointly": False,
                           "sampling": False, "deterministicAlgorithmsEnabled": False,
                           "calls": "one model task per product/origin/condition"},
            "phases": {k: {"targetStart": a.isoformat(), "targetEnd": b.isoformat(),
                           "originStart": c.isoformat()} for k, (a, b, c) in PHASES.items()},
            "baseline": {k: {"path": str(v.relative_to(ROOT)), "sha256": BASELINE_SHA256[k]}
                         for k, v in BASELINE_PATHS.items()},
            "runtime": {"python": "3.10.2", "machine": "arm64", "torch": "2.5.1",
                        "transformers": "4.49.0", "accelerate": "1.1.1",
                        "chronos-forecasting": PACKAGE_VERSION, "device": "mps",
                        "dtype": "float32", "seed": 0, "modelEval": True,
                        "deterministicAlgorithmsEnabled": False},
            "sourceSha256": {name: _file_sha(ROOT / name) for name in (
                "agri_data/chronos2_market_context.py", "agri_data/chronos2_market_context_score.py",
                "agri_data/radish_horizon_sharing.py", "agri_data/market_extension.py",
                "agri_data/coverage.py", "agri_data/dashboard.py", "agri_data/direct_horizon.py",
                "agri_data/equal_info_online_score.py", "agri_data/research_progress_diagnostic.py")},
            "eligibility": "eligibility_unverified"}


def freeze_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    protocol = _protocol()
    protocol["protocolSha256"] = _sha(_canonical(protocol))
    path = output_dir / "config.yaml"
    _write_once(path, _canonical(protocol))
    return protocol


def _load_protocol(output_dir: Path) -> dict[str, Any]:
    doc = _read_json(output_dir / "config.yaml")
    sha = doc.pop("protocolSha256", None)
    if sha != _sha(_canonical(doc)) or doc != _protocol():
        raise ValueError("frozen Chronos-2 protocol mismatch")
    doc["protocolSha256"] = sha
    return doc


def _phase_rows(phase: str, baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    if baseline.get("phase") != phase or not isinstance(baseline.get("forecasts"), list):
        raise ValueError("saved RHS baseline phase/forecast rows are invalid")
    target_start, target_end, origin_start = PHASES[phase]
    rows = []
    seen = set()
    for source in baseline["forecasts"]:
        try:
            product = source["product"]
            origin = dt.date.fromisoformat(source["originDate"] if "originDate" in source else source["origin"])
            target = dt.date.fromisoformat(source["targetDate"] if "targetDate" in source else source["target"])
            horizon = source["horizon"]
            item = source["conditions"]["C"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("RHS baseline row lacks its saved E106-RHS C price forecast") from exc
        key = (product, origin, target, horizon)
        if product not in PRODUCTS or type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError("baseline key outside frozen product/horizon domain")
        if target - origin != dt.timedelta(days=horizon):
            raise ValueError("baseline origin/target/horizon mismatch")
        if not (target_start <= target <= target_end and origin >= origin_start):
            continue
        if key in seen:
            raise ValueError(f"duplicate baseline issue key: {key}")
        seen.add(key)
        source_status = item.get("status")
        if source_status not in ("issued", "fallback"):
            continue
        try:
            b = float(item["blendPrediction"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"issued baseline forecast has no numeric price: {key}") from exc
        if not math.isfinite(b) or b <= 0:
            raise ValueError(f"issued baseline forecast has nonpositive/nonfinite price: {key}")
        rows.append({"product": product, "origin": origin, "target": target,
                     "horizon": horizon, "B": b,
                     "baselineFallbackReason": item.get("fallbackReason"),
                     "baselineSourceStatus": source_status})
    if not rows:
        raise ValueError("no B-eligible keys in frozen phase")
    return sorted(rows, key=lambda r: (r["product"], r["origin"], r["horizon"]))


def _check_runtime() -> dict[str, Any]:
    runtime = _runtime()
    expected = _protocol()["runtime"]
    packages = runtime["packages"]
    actual = {"python": platform.python_version(), "machine": platform.machine(),
              "torch": packages["torch"], "transformers": packages["transformers"],
              "accelerate": packages["accelerate"], "chronos-forecasting": packages["chronos-forecasting"],
              "deterministicAlgorithmsEnabled": runtime["deterministicAlgorithmsEnabled"]}
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError(f"frozen runtime mismatch for {key}: {value!r}")
    if runtime["mpsBuilt"] is not True or runtime["mpsAvailable"] is not True:
        raise RuntimeError("frozen MPS device is unavailable")
    return runtime


def _build_context(series: Mapping[dt.date, Any], markets: Mapping[str, Mapping[dt.date, Any]],
                   origin: dt.date,
                   market_states: Mapping[tuple[dt.date, str], Mapping[str, Any]] | None = None
                   ) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff = origin - dt.timedelta(days=1)
    first = cutoff - dt.timedelta(days=CONTEXT_DAYS - 1)
    dates = [first + dt.timedelta(days=i) for i in range(CONTEXT_DAYS)]
    representative = np.full(CONTEXT_DAYS, np.nan, dtype=np.float32)
    auxiliary = [np.full(CONTEXT_DAYS, np.nan, dtype=np.float32) for _ in MARKETS]
    for i, day in enumerate(dates):
        if day in series:
            value = float(series[day][0])
            if math.isfinite(value) and value > 0:
                representative[i] = np.float32(value)
        for j, market in enumerate(MARKETS):
            item = markets.get(market, {}).get(day)
            if item is not None:
                value = float(item["price"])
                if math.isfinite(value) and value > 0:
                    auxiliary[j][i] = np.float32(value)
    matrix = np.vstack([representative, *auxiliary]).astype(np.float32, copy=False)
    if matrix.shape != (1 + len(MARKETS), CONTEXT_DAYS) or np.isinf(matrix).any():
        raise ValueError("context must be fixed length and contain finite values or NaN only")
    if not np.array_equal(matrix[0], representative, equal_nan=True):
        raise AssertionError("representative input changed while forming U/M context")
    counts = {"representative": int(np.isfinite(representative).sum())}
    counts.update({m: int(np.isfinite(auxiliary[i]).sum()) for i, m in enumerate(MARKETS)})
    state_counts = {}
    for market in MARKETS:
        labels = {}
        for day in dates:
            if day in markets.get(market, {}):
                label = "positive_price_observed"
            else:
                state = (market_states or {}).get((day, market), {})
                selected = state.get("selectedState")
                latest = state.get("latestState")
                label = (str(selected) if selected in ("complete", "empty") else
                         f"unavailable:{latest or 'unqueried'}")
                if selected == "complete":
                    label = "complete_no_positive_price"
            labels[label] = labels.get(label, 0) + 1
        state_counts[market] = dict(sorted(labels.items()))
    return matrix, {"cutoff": cutoff.isoformat(), "firstDate": first.isoformat(),
                    "lastDate": cutoff.isoformat(), "observedCounts": counts,
                    "allMissingAuxiliary": [m for m in MARKETS if counts[m] == 0],
                    "marketStateCounts": state_counts}


def _market_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = _read_json(dh.INPUT_PRICES)
    if _file_sha(dh.INPUT_PRICES) != PRICE_SHA256:
        raise ValueError("frozen representative price snapshot hash mismatch")
    core = dh._observations(raw)
    db_path = MARKET_DB_PATH
    uri = "file:" + str(db_path.resolve()) + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        states, broad = me._core_broad_query_states(db, MIN_MARKET_DATE)
        prices, selected, diagnostics = me._extract_core_market_prices(
            db, MIN_MARKET_DATE, MAX_MARKET_DATE, include_diagnostics=True)
        db.rollback()
    finally:
        db.close()
    markets = {product: {market: {} for market in MARKETS} for product in PRODUCTS}
    for product in PRODUCTS:
        for market in MARKETS:
            for day, price in prices[product][market].items():
                diag = diagnostics.get((product, day, market), {})
                markets[product][market][day] = {"price": price,
                    "weight": float(diag.get("signedNetWeight", 0)),
                    "rows": int(diag.get("eligibleAuctionKgRows", 0))}
    return raw, core, {"markets": markets, "states": states, "diagnostics": diagnostics,
                       "provenance": {"broad": broad, "selected": selected}}


def _load_pipeline():
    import torch
    from chronos import Chronos2Pipeline
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("frozen MPS device is unavailable")
    torch.manual_seed(0)
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=False, torch_dtype=torch.float32)
    pipeline.model.to(torch.device("mps"))
    pipeline.model.eval()
    if pipeline.model.device.type != "mps" or pipeline.model.dtype != torch.float32:
        raise RuntimeError("loaded model does not match frozen MPS float32 execution")
    _verify_cached_weights()
    return pipeline


def _verify_cached_weights() -> None:
    from huggingface_hub import try_to_load_from_cache
    weight = try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    config = try_to_load_from_cache(MODEL_ID, "config.json", revision=MODEL_REVISION)
    for path, expected_sha, expected_size, name in (
        (weight, MODEL_WEIGHT_SHA256, MODEL_WEIGHT_SIZE, "model.safetensors"),
        (config, MODEL_CONFIG_SHA256, None, "config.json")):
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError(f"pinned model file not cached: {name}")
        if expected_size is not None and Path(path).stat().st_size != expected_size:
            raise ValueError(f"pinned model file size mismatch: {name}")
        if _file_sha(Path(path)) != expected_sha:
            raise ValueError(f"pinned model file checksum mismatch: {name}")


def _task_id(product: str, origin: dt.date, condition: str) -> str:
    return f"{product}-{origin.isoformat()}-{condition}"


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if sys.platform == "darwin" else 1024))


def _task_checkpoint(path: Path, product: str, origin: dt.date, condition: str,
                     matrix: np.ndarray, pipeline) -> dict[str, Any]:
    input_hash = _sha(np.ascontiguousarray(matrix).tobytes())
    task = matrix[:1] if condition == "U" else matrix
    shape = list(task.shape)
    channel_order = ["representativePrice"] if condition == "U" else ["representativePrice", *MARKETS]
    if path.exists():
        saved = _read_json(path)
        saved_sha = saved.pop("taskSha256", None)
        if saved_sha != _sha(_canonical(saved)):
            raise ValueError(f"task checkpoint self-hash mismatch: {path.name}")
        if (saved.get("inputSha256") != input_hash or saved.get("modelRevision") != MODEL_REVISION
                or saved.get("condition") != condition or saved.get("product") != product
                or saved.get("origin") != origin.isoformat() or saved.get("taskShape") != shape
                or saved.get("channelOrder") != channel_order
                or saved.get("predictionLength") != PREDICTION_LENGTH
                or len(saved.get("representativePredictions", [])) != PREDICTION_LENGTH
                or saved.get("status") not in ("complete", "failed")
                or type(saved.get("processMaxRssBytesAfterTask")) is not int
                or saved["processMaxRssBytesAfterTask"] <= 0
                or (saved.get("status") == "failed" and
                    (not isinstance(saved.get("failure"), Mapping)
                     or any(x is not None for x in saved["representativePredictions"])))
                or (saved.get("status") == "complete" and saved.get("failure") is not None)):
            raise ValueError(f"task checkpoint does not match frozen input: {path.name}")
        saved["taskSha256"] = saved_sha
        return saved
    started = time.monotonic()
    try:
        quantiles, _ = pipeline.predict_quantiles(
            [task], prediction_length=PREDICTION_LENGTH, context_length=CONTEXT_DAYS,
            quantile_levels=[0.5], predict_batches_jointly=False)
        q = quantiles[0]
        if tuple(q.shape) != (task.shape[0], PREDICTION_LENGTH, 1):
            raise ValueError(f"unexpected Chronos quantile output shape: {tuple(q.shape)}")
        raw_values = [float(x) for x in q[0, :, 0].detach().cpu().tolist()]
        if len(raw_values) != PREDICTION_LENGTH:
            raise ValueError("model output length differs from frozen prediction length")
        values = [x if math.isfinite(x) else None for x in raw_values]
        status, failure = "complete", None
    except Exception as exc:
        # U and M are separate calls. Preserve a failed call as an explicit
        # checkpoint so the other condition can still run without fallback.
        values = [None] * PREDICTION_LENGTH
        status, failure = "failed", {"type": type(exc).__name__, "message": str(exc)[:500]}
    elapsed = time.monotonic() - started
    doc = {"schemaVersion": 1, "product": product, "origin": origin.isoformat(),
           "condition": condition, "inputSha256": input_hash, "modelRevision": MODEL_REVISION,
           "taskShape": shape, "channelOrder": channel_order,
           "groupPolicy": "one product-origin per predict call; all channels in this M task share group 0",
           "quantile": 0.5, "predictionLength": PREDICTION_LENGTH,
           "representativePredictions": values, "elapsedSeconds": elapsed,
           "processMaxRssBytesAfterTask": _max_rss_bytes(),
           "status": status, "failure": failure, "taskId": _task_id(product, origin, condition)}
    doc["taskSha256"] = _sha(_canonical(doc))
    _write_once(path, _canonical(doc))
    return doc


def _forecast_document(phase: str, rows: list[dict[str, Any]], task_results: Mapping[tuple[str, dt.date, str], Any],
                       contexts: Mapping[tuple[str, dt.date], Any], inputs: Mapping[str, Any],
                       output_dir: Path) -> dict[str, Any]:
    forecasts = []
    for source in rows:
        product, origin, target, horizon = (source[x] for x in ("product", "origin", "target", "horizon"))
        u_task = task_results[(product, origin, "U")]
        m_task = task_results[(product, origin, "M")]
        u = u_task["representativePredictions"][horizon]
        m = m_task["representativePredictions"][horizon]
        valid_u = u_task["status"] == "complete" and u is not None and math.isfinite(u) and u > 0
        valid_m = m_task["status"] == "complete" and m is not None and math.isfinite(m) and m > 0
        h = 0.5 * source["B"] + 0.5 * m if valid_m else None
        states = {
            "B": {"status": "issued", "prediction": source["B"],
                  "reason": source.get("baselineFallbackReason") or
                           ("saved_baseline_status_fallback" if source.get("baselineSourceStatus") == "fallback" else None),
                  "baselineSourceStatus": source.get("baselineSourceStatus", "issued")},
            "U": {"status": "issued" if valid_u else "inference_error" if u_task["status"] == "failed" else "invalid_prediction",
                  "prediction": u if valid_u else None,
                  "reason": (None if valid_u else
                             f"{u_task['failure']['type']}: {u_task['failure']['message']}" if u_task["status"] == "failed"
                             else "q50_not_finite_positive")},
            "M": {"status": "issued" if valid_m else "inference_error" if m_task["status"] == "failed" else "invalid_prediction",
                  "prediction": m if valid_m else None,
                  "reason": (None if valid_m else
                             f"{m_task['failure']['type']}: {m_task['failure']['message']}" if m_task["status"] == "failed"
                             else "q50_not_finite_positive")},
            "H": {"status": "issued" if valid_m else "not_issued", "prediction": h,
                  "reason": None if valid_m else "M_inference_error" if m_task["status"] == "failed"
                           else "M_not_finite_positive"}}
        ctx = contexts[(product, origin)]
        forecasts.append({"key": {"product": product, "originDate": origin.isoformat(),
                                   "targetDate": target.isoformat(), "horizon": horizon},
                          "conditions": states, "context": ctx})
    return {"schemaVersion": 1, "phase": phase, "protocolSha256": _load_protocol(output_dir)["protocolSha256"],
            "inputSha256": inputs["inputSha256"], "modelRevision": MODEL_REVISION,
            "eligibility": "eligibility_unverified", "forecasts": forecasts}


def _csv_bytes(forecasts: list[dict[str, Any]], protocol_sha: str, input_sha: str) -> bytes:
    columns = ("product", "originDate", "cutoff", "targetDate", "horizon", "B", "U", "M", "H",
               "UStatus", "MStatus", "HStatus", "representativeObservedDays",
               f"{MARKETS[0]}ObservedDays", f"{MARKETS[1]}ObservedDays", "modelRevision",
               "protocolSha256", "inputSha256")
    import io
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in forecasts:
        key, cond, support = row["key"], row["conditions"], row["context"]
        counts = support["observedCounts"]
        writer.writerow([key["product"], key["originDate"], support["cutoff"], key["targetDate"],
                         key["horizon"], cond["B"]["prediction"], cond["U"]["prediction"],
                         cond["M"]["prediction"], cond["H"]["prediction"], cond["U"]["status"],
                         cond["M"]["status"], cond["H"]["status"], counts["representative"],
                         counts.get(MARKETS[0], 0), counts.get(MARKETS[1], 0), MODEL_REVISION,
                         protocol_sha, input_sha])
    return buffer.getvalue().encode("utf-8")


def _primary_gate(report: Mapping[str, Any]) -> bool:
    gate = report.get("validationGate")
    return isinstance(gate, Mapping) and gate.get("passed") is True


def preflight(phase: str, output_dir: Path = OUT, *, _retain_sources: bool = False) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    _load_protocol(output_dir)
    runtime = _check_runtime()
    baseline_path = BASELINE_PATHS[phase]
    if _file_sha(baseline_path) != BASELINE_SHA256[phase]:
        raise ValueError("saved RHS baseline manifest hash mismatch")
    baseline = _read_json(baseline_path)
    rows = _phase_rows(phase, baseline)
    raw, core, market_bundle = _market_inputs()
    key_sha = _sha(_canonical([[r["product"], r["origin"].isoformat(), r["target"].isoformat(), r["horizon"]]
                               for r in rows]))
    market_digest = _sha(_canonical([
        [product, market, day.isoformat(), value["price"], value["weight"]]
        for product in sorted(market_bundle["markets"])
        for market in sorted(market_bundle["markets"][product])
        for day, value in sorted(market_bundle["markets"][product][market].items())]))
    market_state_digest = _sha(_canonical([
        [product, day.isoformat(), market, state.get("selectedState"), state.get("latestState")]
        for (product, day, market), state in sorted(market_bundle["states"].items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2]))]))
    typed_provenance = _typed_provenance(market_bundle["provenance"])
    inputs = {"priceSnapshotSha256": _file_sha(dh.INPUT_PRICES),
              "baselineManifestSha256": _file_sha(baseline_path),
              "BKeyUniverseSha256": key_sha,
              "marketPricesSha256": market_digest,
              "marketStateSha256": market_state_digest,
              "databasePath": str(MARKET_DB_PATH.relative_to(ROOT)),
              "databaseSize": MARKET_DB_PATH.stat().st_size,
              "marketProvenance": typed_provenance,
              "inputSha256": _sha(_canonical({"price": _file_sha(dh.INPUT_PRICES),
                                              "baseline": _file_sha(baseline_path), "BKeys": key_sha,
                                              "marketPrices": market_digest, "marketStates": market_state_digest,
                                              "marketProvenance": typed_provenance}))}
    result = {"phase": phase, "eligibleBKeys": len(rows),
              "uniqueProductOrigins": len({(r["product"], r["origin"]) for r in rows}),
              "runtime": runtime, "inputs": inputs,
              "contextChecks": {"contextDays": CONTEXT_DAYS, "allInputsFiniteOrNaN": True,
                                "futureTargetsUsed": False}}
    if _retain_sources:
        result["_raw"] = raw
        result["_core"] = core
        result["_marketData"] = market_bundle["markets"]
        result["_marketStates"] = market_bundle["states"]
    return result


def build_phase(phase: str, output_dir: Path = OUT) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(output_dir)
    gate = None
    if phase == "T":
        gate = _verified_validation_gate(output_dir)
        if gate is not True:
            raise ValueError("T is gated on a verified passing V report")
    baseline_path = BASELINE_PATHS[phase]
    if _file_sha(baseline_path) != BASELINE_SHA256[phase]:
        raise ValueError("saved RHS baseline manifest hash mismatch")
    baseline = _read_json(baseline_path)
    rows = _phase_rows(phase, baseline)
    pre = preflight(phase, output_dir, _retain_sources=True)
    raw_inputs = pre.pop("_raw")
    core = pre.pop("_core")
    market_data = pre.pop("_marketData")
    market_states = pre.pop("_marketStates")
    contexts: dict[tuple[str, dt.date], Any] = {}
    task_matrices: dict[tuple[str, dt.date, str], np.ndarray] = {}
    origins = sorted({(r["product"], r["origin"]) for r in rows})
    for product, origin in origins:
        local_states = {(day, market): state for (p, day, market), state in market_states.items()
                        if p == product}
        matrix, support = _build_context(core[product], market_data[product], origin, local_states)
        contexts[(product, origin)] = support
        task_matrices[(product, origin, "U")] = matrix[:1]
        task_matrices[(product, origin, "M")] = matrix
    claim_path = output_dir / f"{phase}-run-claim.json"
    claim = {"phase": phase, "protocolSha256": protocol["protocolSha256"],
             "inputSha256": pre["inputs"]["inputSha256"], "BKeys": len(rows),
             "startedAtKst": dt.datetime.now().astimezone().isoformat(), "status": "running"}
    terminal_path = output_dir / f"{phase}-run-terminal.json"
    if terminal_path.is_file():
        terminal = _read_json(terminal_path)
        terminal_sha = terminal.pop("terminalSha256", None)
        manifest_path = output_dir / ("validation-manifest.json" if phase == "V" else "test-manifest.json")
        if terminal_sha != _sha(_canonical(terminal)) or terminal.get("status") != "complete":
            raise ValueError("existing phase terminal record is invalid or incomplete")
        manifest = _read_json(manifest_path)
        if (terminal.get("manifestSha256") != manifest.get("manifestSha256")
                or manifest.get("inputSha256") != pre["inputs"]["inputSha256"]
                or manifest.get("manifestSha256") != _sha(_canonical({k: v for k, v in manifest.items()
                                                                        if k != "manifestSha256"}))):
            raise ValueError("completed phase artifacts do not match terminal record/current inputs")
        if _file_sha(output_dir / manifest["forecastFile"]) != manifest["forecastFileSha256"]:
            raise ValueError("completed forecast file hash mismatch")
        return manifest
    if claim_path.exists():
        old = _read_json(claim_path)
        old_sha = old.pop("claimSha256", None)
        if old_sha != _sha(_canonical(old)) or (old.get("phase"), old.get("protocolSha256"),
                old.get("inputSha256"), old.get("BKeys")) != (
                phase, claim["protocolSha256"], claim["inputSha256"], claim["BKeys"]):
            raise ValueError("existing phase claim is invalid or bound to different inputs")
        old["claimSha256"] = old_sha
        claim = old
    else:
        claim["claimSha256"] = _sha(_canonical(claim))
        _write_once(claim_path, _canonical(claim))
    task_dir = output_dir / "task-checkpoints" / phase
    task_dir.mkdir(parents=True, exist_ok=True)
    pipeline = None
    model_load_seconds = 0.0
    task_results = {}
    planned_tasks = len(task_matrices)
    first_origin = origins[0]
    for key, matrix in task_matrices.items():
        product, origin, condition = key
        filename = _task_id(product, origin, condition) + ".json"
        checkpoint_path = task_dir / filename
        if not checkpoint_path.exists() and pipeline is None:
            load_started = time.monotonic()
            pipeline = _load_pipeline()
            model_load_seconds = time.monotonic() - load_started
        result = _task_checkpoint(checkpoint_path, product, origin, condition, matrix, pipeline)
        task_results[key] = result
        if (product, origin) == first_origin and condition == "M":
            pair_seconds = sum(task_results[(product, origin, cond)]["elapsedSeconds"] for cond in ("U", "M"))
            pair_seconds = max(pair_seconds, 1e-9)
            estimate = model_load_seconds + pair_seconds * len(origins)
            now_kst = dt.datetime.now(ZoneInfo("Asia/Seoul"))
            deadline = dt.datetime(2026, 10, 13, 23, 59, 59, tzinfo=ZoneInfo("Asia/Seoul"))
            remaining = max(0.0, (deadline - now_kst).total_seconds())
            required = estimate * (2 if phase == "V" else 1)
            budget = {"phase": phase,
                      "status": "BLOCKED_BUDGET" if required > remaining else
                               "over_two_hour_checkpoint" if estimate > 7200 else "within_two_hour_checkpoint",
                      "plannedTasks": planned_tasks, "plannedProductOrigins": len(origins),
                          "firstProductOriginUSeconds": task_results[(product, origin, "U")]["elapsedSeconds"],
                          "firstProductOriginMSeconds": task_results[(product, origin, "M")]["elapsedSeconds"],
                          "modelLoadSeconds": model_load_seconds,
                          "firstPairMaxRssBytes": max(task_results[(product, origin, cond)]["processMaxRssBytesAfterTask"]
                                                      for cond in ("U", "M")),
                          "estimatedSeconds": estimate, "remainingUntilDeadlineSeconds": remaining,
                          "estimatedPhasesIncludingT": required,
                          "checkpoint": result["taskId"], "checkpointScope": "answer_free_first_product_origin_U_and_M"}
            budget_path = output_dir / f"{phase}-budget-status.json"
            if budget_path.exists():
                saved_budget = _read_json(budget_path)
                if (saved_budget.get("plannedTasks"), saved_budget.get("checkpoint")) != (
                        planned_tasks, result["taskId"]):
                    raise ValueError("budget checkpoint does not match resumed task plan")
            else:
                _write_once(budget_path, _canonical(budget))
            if budget["status"] == "BLOCKED_BUDGET":
                raise RuntimeError("BLOCKED_BUDGET: first-origin throughput estimate exceeds time remaining to Oct 13")
    budget_path = output_dir / f"{phase}-budget-status.json"
    budget_saved = _read_json(budget_path) if budget_path.exists() else {}
    elapsed = sum(float(result["elapsedSeconds"]) for result in task_results.values())
    forecast_doc = _forecast_document(phase, rows, task_results, contexts, pre["inputs"], output_dir)
    raw = _canonical(forecast_doc)
    forecast_path = output_dir / ("validation-manifest.json" if phase == "V" else "test-manifest.json")
    forecast_file = output_dir / ("validation-forecasts.json" if phase == "V" else "test-forecasts.json")
    _write_or_match(forecast_file, raw)
    csv_path = output_dir / ("validation-forecasts.csv" if phase == "V" else "test-forecasts.csv")
    _write_or_match(csv_path, _csv_bytes(forecast_doc["forecasts"], protocol["protocolSha256"],
                                         pre["inputs"]["inputSha256"]))
    manifest = {"schemaVersion": 1, "phase": phase, "protocolSha256": protocol["protocolSha256"],
                "inputSha256": pre["inputs"]["inputSha256"], "forecastFile": forecast_file.name,
                "forecastFileSha256": _file_sha(forecast_file), "csvFile": csv_path.name,
                "csvFileSha256": _file_sha(csv_path), "eligibleBKeys": len(rows),
                "taskCount": len(task_results), "elapsedSeconds": elapsed,
                "taskIds": sorted(v["taskId"] for v in task_results.values()),
                "taskSha256": {v["taskId"]: v["taskSha256"] for v in task_results.values()},
                "modelLoadSeconds": budget_saved.get("modelLoadSeconds", model_load_seconds),
                "processMaxRssBytes": max((int(v["processMaxRssBytesAfterTask"]) for v in task_results.values()),
                                          default=0),
                "taskFailureCounts": {condition: sum(v.get("status") == "failed"
                                                       for (p, o, c), v in task_results.items()
                                                       if c == condition) for condition in ("U", "M")},
                "runtime": pre["runtime"], "model": protocol["model"],
                "baselineManifestSha256": _file_sha(baseline_path),
                "inputProvenance": pre["inputs"], "vGate": gate,
                "createdAtKst": claim["startedAtKst"]}
    manifest["manifestSha256"] = _sha(_canonical(manifest))
    _write_or_match(forecast_path, _canonical(manifest))
    terminal = {**claim, "status": "complete", "finishedAtKst": dt.datetime.now().astimezone().isoformat(),
                "manifestSha256": manifest["manifestSha256"]}
    terminal["terminalSha256"] = _sha(_canonical(terminal))
    _write_or_match(terminal_path, _canonical(terminal))
    return manifest


def score_phase(phase: str, output_dir: Path = OUT) -> dict[str, Any]:
    from . import chronos2_market_context_score as scorer
    manifest_path = output_dir / ("validation-manifest.json" if phase == "V" else "test-manifest.json")
    manifest = _read_json(manifest_path)
    pre = preflight(phase, output_dir, _retain_sources=True)
    raw = pre.pop("_raw")
    _verify_saved_phase(phase, output_dir, manifest, pre)
    forecast_path = output_dir / manifest["forecastFile"]
    forecast = _read_json(forecast_path)
    actuals = {(p, day): value[0] for p, series in dh._observations(raw).items()
               for day, value in series.items()}
    # The report filename is phase-specific so a test run never overwrites V.
    report = scorer.score_manifest(forecast, actuals=actuals, phase=phase)
    task_dir = output_dir / "task-checkpoints" / phase
    task_failures = {condition: sum(
        _read_json(task_dir / f"{task_id}.json").get("status") == "failed"
        for task_id in manifest.get("taskIds", []) if task_id.endswith(f"-{condition}"))
        for condition in ("U", "M")}
    report["taskFailureCounts"] = task_failures
    report["manifestSha256"] = manifest["manifestSha256"]
    report["forecastFileSha256"] = manifest["forecastFileSha256"]
    report["phase"] = phase
    report["primaryDecision"] = _primary_gate(report)
    path = output_dir / ("validation-report.json" if phase == "V" else "test-report.json")
    raw_report = _canonical(report)
    if path.exists():
        if path.read_bytes() != raw_report:
            raise ValueError("existing score report differs from deterministic score")
    else:
        _write_once(path, raw_report)
    return report


def _verify_saved_phase(phase: str, output_dir: Path, manifest: Mapping[str, Any],
                        preflight_doc: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _load_protocol(output_dir)
    if (manifest.get("phase") != phase or manifest.get("protocolSha256") != protocol["protocolSha256"]
            or manifest.get("inputSha256") != preflight_doc["inputs"]["inputSha256"]
            or manifest.get("baselineManifestSha256") != BASELINE_SHA256[phase]
            or manifest.get("model", {}).get("revision") != MODEL_REVISION):
        raise ValueError("saved phase protocol/runtime/current-input bindings differ")
    if manifest.get("manifestSha256") != _sha(_canonical({k: v for k, v in manifest.items()
                                                            if k != "manifestSha256"})):
        raise ValueError("forecast manifest self-hash mismatch")
    if _file_sha(ROOT / "agri_data/chronos2_market_context_score.py") != SCORER_SHA256:
        raise ValueError("frozen scoring helper hash mismatch")
    claim_path = output_dir / f"{phase}-run-claim.json"
    terminal_path = output_dir / f"{phase}-run-terminal.json"
    claim = _read_json(claim_path)
    claim_sha = claim.pop("claimSha256", None)
    if claim_sha != _sha(_canonical(claim)):
        raise ValueError("phase run claim self-hash mismatch")
    terminal = _read_json(terminal_path)
    terminal_sha = terminal.pop("terminalSha256", None)
    if terminal_sha != _sha(_canonical(terminal)) or terminal.get("status") != "complete":
        raise ValueError("phase terminal record invalid or incomplete")
    if (claim.get("inputSha256") != manifest.get("inputSha256")
            or claim.get("protocolSha256") != manifest.get("protocolSha256")
            or claim.get("phase") != phase
            or terminal.get("claimSha256") != claim_sha
            or terminal.get("inputSha256") != manifest.get("inputSha256")
            or terminal.get("protocolSha256") != manifest.get("protocolSha256")
            or terminal.get("phase") != phase
            or terminal.get("manifestSha256") != manifest.get("manifestSha256")):
        raise ValueError("phase run claim/terminal not bound to current manifest")
    if manifest.get("runtime") != preflight_doc.get("runtime"):
        raise ValueError("saved runtime differs from current frozen runtime")
    forecast_path = output_dir / manifest["forecastFile"]
    csv_path = output_dir / manifest["csvFile"]
    if (_file_sha(forecast_path) != manifest.get("forecastFileSha256")
            or _file_sha(csv_path) != manifest.get("csvFileSha256")):
        raise ValueError("answer-free forecast/CSV hash mismatch")
    forecast = _read_json(forecast_path)
    baseline_path = BASELINE_PATHS[phase]
    if _file_sha(baseline_path) != BASELINE_SHA256[phase]:
        raise ValueError("saved RHS baseline manifest hash mismatch")
    rows = _phase_rows(phase, _read_json(baseline_path))
    if manifest.get("eligibleBKeys") != len(rows):
        raise ValueError("saved B key count differs from pinned source")
    core, markets, states = (preflight_doc["_core"], preflight_doc["_marketData"],
                             preflight_doc["_marketStates"])
    origins = sorted({(r["product"], r["origin"]) for r in rows})
    contexts, tasks = {}, {}
    for product, origin in origins:
        local_states = {(day, market): state for (p, day, market), state in states.items() if p == product}
        matrix, support = _build_context(core[product], markets[product], origin, local_states)
        contexts[(product, origin)] = support
        tasks[(product, origin, "U")] = matrix[:1]
        tasks[(product, origin, "M")] = matrix
    task_dir = output_dir / "task-checkpoints" / phase
    task_results = {}
    for (product, origin, condition), matrix in tasks.items():
        path = task_dir / (_task_id(product, origin, condition) + ".json")
        if not path.is_file():
            raise ValueError(f"missing saved model task checkpoint: {path.name}")
        result = _task_checkpoint(path, product, origin, condition, matrix, None)
        task_results[(product, origin, condition)] = result
    expected_task_hashes = {v["taskId"]: v["taskSha256"] for v in task_results.values()}
    if expected_task_hashes != manifest.get("taskSha256") or len(task_results) != manifest.get("taskCount"):
        raise ValueError("saved task checkpoint hashes/count differ from phase manifest")
    failure_counts = {condition: sum(v.get("status") == "failed" for (p, o, c), v in task_results.items()
                                     if c == condition) for condition in ("U", "M")}
    max_rss = max((int(v["processMaxRssBytesAfterTask"]) for v in task_results.values()), default=0)
    if manifest.get("taskFailureCounts") != failure_counts or manifest.get("processMaxRssBytes") != max_rss:
        raise ValueError("saved task failure/RSS diagnostics do not replay from checkpoints")
    reconstructed = _forecast_document(phase, rows, task_results, contexts,
                                       preflight_doc["inputs"], output_dir)
    if _canonical(reconstructed) != _canonical(forecast):
        raise ValueError("saved forecast rows do not replay from pinned B and task checkpoints")
    csv_raw = _csv_bytes(reconstructed["forecasts"], protocol["protocolSha256"],
                         preflight_doc["inputs"]["inputSha256"])
    if csv_path.read_bytes() != csv_raw:
        raise ValueError("saved answer-free CSV does not replay from task checkpoints")
    return forecast


def _verified_validation_gate(output_dir: Path) -> bool:
    report_path = output_dir / "validation-report.json"
    manifest_path = output_dir / "validation-manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise ValueError("T requires completed V manifest and score report")
    saved_report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if manifest.get("manifestSha256") != _sha(_canonical({k: v for k, v in manifest.items() if k != "manifestSha256"})):
        raise ValueError("V manifest self-hash mismatch")
    if saved_report.get("manifestSha256") != manifest.get("manifestSha256"):
        raise ValueError("V score report is not bound to the saved V manifest")
    reproduced = score_phase("V", output_dir)
    if _canonical(reproduced) != _canonical(saved_report):
        raise ValueError("V report differs from recomputed score")
    return _primary_gate(reproduced)


def run(output_dir: Path = OUT) -> dict[str, Any]:
    if not (output_dir / "config.yaml").exists():
        freeze_protocol(output_dir)
    validation = build_phase("V", output_dir)
    report = score_phase("V", output_dir)
    if report.get("primaryDecision") is True:
        test = build_phase("T", output_dir)
        test_report = score_phase("T", output_dir)
    else:
        test = {"phase": "T", "status": "not_run_validation_failed"}
        test_report = None
        _write_or_match(output_dir / "test-report.json", _canonical(test))
    if test_report is None:
        failures = report.get("taskFailureCounts", {})
        reason = (f"V had {failures.get('M', 0)} M inference-task failures, so the H performance assessment "
                  "is incomplete and is not evidence of performance.") if failures.get("M", 0) else (
                  "The completed V execution did not meet the frozen performance gate.")
        if failures.get("U", 0):
            reason += f" U diagnostic tasks failed: {failures['U']}."
        decision = ("# Chronos-2 market-context retrospective\n\n"
                    f"V primary decision: {report.get('primaryDecision')!r}. T status: not_run_validation_failed.\n\n"
                    f"{reason} T was not run. Preserve E106-RHS. Competition eligibility remains unverified.\n")
    else:
        passed = report.get("primaryDecision") is True and test_report.get("primaryDecision") is True
        test_failures = test_report.get("taskFailureCounts", {})
        failure_note = (f" T had {test_failures.get('M', 0)} M inference-task failures, leaving its H assessment "
                        "incomplete; this is not evidence about performance."
                        if test_failures.get("M", 0) else "")
        decision = ("# Chronos-2 market-context retrospective\n\n"
                    f"V primary decision: {report.get('primaryDecision')!r}. "
                    f"T primary decision: {test_report.get('primaryDecision')!r}.\n\n"
                    + ("Both historical periods passed the frozen gate; preserve E106-C2 as a retrospective candidate. "
                       "No operational replacement is made. " if passed else
                       "The frozen candidate did not pass in both periods; preserve E106-RHS. ")
                    + failure_note
                    + "Competition eligibility remains unverified; these previously viewed periods do not establish "
                    "independent future performance.\n")
    decision_path = output_dir / "decision.md"
    if decision_path.exists():
        if decision_path.read_text(encoding="utf-8") != decision:
            raise ValueError("existing decision document differs from scored results")
    else:
        _write_once(decision_path, decision.encode("utf-8"))
    return {"validationManifestSha256": validation["manifestSha256"],
            "validationDecision": report.get("primaryDecision"),
            "testStatus": "complete" if test_report else test["status"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "preflight", "build", "score", "run"))
    parser.add_argument("--phase", choices=("V", "T"), default="V")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.command == "freeze": result = freeze_protocol(args.output_dir)
    elif args.command == "preflight": result = preflight(args.phase, args.output_dir)
    elif args.command == "build": result = build_phase(args.phase, args.output_dir)
    elif args.command == "score": result = score_phase(args.phase, args.output_dir)
    else: result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
