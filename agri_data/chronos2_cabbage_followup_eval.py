"""Stage B: frozen cabbage-only Chronos-2 follow-up on saved RHS T keys.

Forecast creation and truth scoring are separate commands. The prediction artifact
is sealed before target observations are converted to numeric values.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from . import chronos2_cabbage_followup as stage_a
from . import chronos2_market_context as c2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/chronos2-cabbage-followup-v1-20261012"
RHS_T = ROOT / "outputs/experiments/radish-horizon-sharing-v1-20261011/test-manifest.json"
RHS_T_SHA256 = "bc4b453df2ded47b8968fc7907fecad024e7da7fa0e2a5e730099a233848cc76"
SELECTION_SHA256 = "00f8e7147f501f54fadfe33a833decc81cb4c1042cbfd6418f0ddf6e660f75c4"
SELECTION_PROTOCOL_FILE_SHA256 = "69bf636299d50f24045ea97a95a254e7f952d30a6d0503f4ad954b654e8a0ff9"
SELECTION_PROTOCOL_SHA256 = "e0767c027ff53393ba4a1a2136acaa85a88d710cb04ebe56039845c98ac37f54"
C2_SOURCE_SHA256 = "d3a476a0d2825c6e54d756a40a7edeb771ed2aef3a755d6fe14dab622c87ebbc"
V_MANIFEST_SHA256 = "9d8ebeb0752bfbc438504b98c2336f9b94492908e345199a2a09b37a96a51bf5"
V_FORECAST_SHA256 = "af0e0ecf3949889d1b9bc0d5c394d97e82e24f5f21f3063fc761ab939623e2ce"
TARGET_START, TARGET_END, ORIGIN_START = dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1)
EXPECTED_KEYS, EXPECTED_CABBAGE_KEYS, EXPECTED_ORIGINS = 3160, 790, 88
PRODUCTS = tuple(c2.PRODUCTS)
HORIZONS = tuple(range(1, 11))
ALPHA, HAC_BANDWIDTHS, BLOCKS, REPS, SEED = 0.05, (9, 28), (14, 28), 2000, 0


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            raise ValueError(f"existing deterministic artifact differs: {path.name}")
        return
    _write_once(path, raw)


def _source_files() -> dict[str, Path]:
    return {
        "stageB": ROOT / "agri_data/chronos2_cabbage_followup_eval.py",
        "stageA": ROOT / "agri_data/chronos2_cabbage_followup.py",
        "adapter": ROOT / "agri_data/chronos2_market_context.py",
        "scorer": ROOT / "agri_data/chronos2_market_context_score.py",
        "directHorizon": ROOT / "agri_data/direct_horizon.py",
        "marketExtension": ROOT / "agri_data/market_extension.py",
        "rhs": ROOT / "agri_data/radish_horizon_sharing.py",
        "equalInfoScore": ROOT / "agri_data/equal_info_online_score.py",
        "bootstrap": ROOT / "agri_data/research_progress_diagnostic.py",
        "modeling": ROOT / "agri_data/modeling.py",
        "plan": ROOT / "docs/CHRONOS2_CABBAGE_FOLLOWUP_V1_2026-10-12.md",
    }


def _verify_selection() -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = stage_a.OUT / "selection.json"
    protocol_path = stage_a.OUT / "protocol.json"
    if _file_sha(selection_path) != SELECTION_SHA256:
        raise ValueError("Stage A selection hash mismatch")
    if _file_sha(protocol_path) != SELECTION_PROTOCOL_FILE_SHA256:
        raise ValueError("Stage A protocol file hash mismatch")
    selection, protocol = _read(selection_path), _read(protocol_path)
    if (selection.get("selected") != "HM" or selection.get("status") != "selected"
            or selection.get("protocolSha256") != SELECTION_PROTOCOL_SHA256
            or protocol.get("protocolSha256") != SELECTION_PROTOCOL_SHA256):
        raise ValueError("Stage A did not immutably select HM")
    if protocol.get("selectionCodeSha256") != _file_sha(Path(stage_a.__file__)):
        raise ValueError("Stage A selection code changed")
    return selection, protocol


def _base_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, dt.date, dt.date, int], dict[str, Any]]]:
    if _file_sha(RHS_T) != RHS_T_SHA256:
        raise ValueError("pinned RHS T baseline hash mismatch")
    doc = _read(RHS_T)
    if doc.get("phase") != "T":
        raise ValueError("RHS source is not T")
    rows = c2._phase_rows("T", doc)
    result = {}
    for row in rows:
        key = (row["product"], row["origin"], row["target"], row["horizon"])
        if key in result:
            raise ValueError("duplicate RHS C source key")
        result[key] = row
    cabbage = [r for r in rows if r["product"] == "cabbage"]
    origins = {r["origin"] for r in cabbage}
    if (len(rows), len(cabbage), len(origins)) != (EXPECTED_KEYS, EXPECTED_CABBAGE_KEYS, EXPECTED_ORIGINS):
        raise ValueError("RHS T key counts differ from frozen 3160/790/88 universe")
    if min(origins) != ORIGIN_START or max(origins) != dt.date(2026, 9, 26):
        raise ValueError("cabbage issue-origin bounds differ from frozen T window")
    return rows, result


def _market_bundle() -> dict[str, Any]:
    from . import market_extension as me
    db_path = ROOT / "data/agri.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    db = sqlite3.connect("file:" + str(db_path.resolve()) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    start = c2.MIN_MARKET_DATE
    end = TARGET_END - dt.timedelta(days=1)
    try:
        db.execute("BEGIN")
        states, broad = me._core_broad_query_states(db, start)
        prices, selected, diagnostics = me._extract_core_market_prices(db, start, end, include_diagnostics=True)
        db.rollback()
    finally:
        db.close()
    markets = {p: {m: {} for m in c2.MARKETS} for p in PRODUCTS}
    for product in PRODUCTS:
        for market in c2.MARKETS:
            for day, price in prices[product][market].items():
                diag = diagnostics.get((product, day, market), {})
                markets[product][market][day] = {"price": price,
                    "weight": float(diag.get("signedNetWeight", 0)),
                    "rows": int(diag.get("eligibleAuctionKgRows", 0))}
    return {"markets": markets, "states": states,
            "provenance": c2._typed_provenance({"broad": broad, "selected": selected}),
            "databaseSize": db_path.stat().st_size}


def _cabbage_series_as_of(raw: Mapping[str, Any], cutoff: dt.date) -> dict[dt.date, tuple[float, float]]:
    products = raw.get("products")
    if raw.get("schemaVersion") != 1 or not isinstance(products, list):
        raise ValueError("frozen price input must use dashboard schemaVersion 1")
    filtered, seen = [], set()
    for product in products:
        pid = product.get("id")
        if pid not in c2.dashboard.PRODUCTS or pid in seen:
            raise ValueError("invalid or duplicate product in frozen prices")
        seen.add(pid)
        observations = []
        if pid == "cabbage":
            for row in product.get("observations", []):
                day = row.get("date")
                if isinstance(day, str) and day <= cutoff.isoformat():
                    observations.append(dict(row))
        filtered.append({"id": pid, "observations": observations})
    from . import direct_horizon as dh
    out = dh._observations({"schemaVersion": 1, "products": filtered})
    return out["cabbage"]


def _prepare() -> dict[str, Any]:
    selection, _ = _verify_selection()
    c2._load_protocol(c2.OUT)
    if selection.get("selected") != "HM":
        raise ValueError("the frozen Stage A candidate is not HM")
    rows, base = _base_rows()
    price_path = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
    source_config = _read(c2.OUT / "config.yaml")
    source_manifest = _read(c2.OUT / "validation-manifest.json")
    if (_file_sha(c2.OUT / "validation-manifest.json") != V_MANIFEST_SHA256
            or source_manifest.get("manifestSha256") != _sha(_canonical(
                {k:v for k,v in source_manifest.items() if k != "manifestSha256"}))
            or source_manifest.get("forecastFileSha256") != V_FORECAST_SHA256
            or _file_sha(c2.OUT / "validation-forecasts.json") != V_FORECAST_SHA256
            or _file_sha(price_path) != source_manifest["inputProvenance"]["priceSnapshotSha256"]
            or source_config.get("sourceSha256", {}).get("agri_data/chronos2_market_context.py") != C2_SOURCE_SHA256):
        raise ValueError("original price snapshot/source adapter pins changed")
    raw_prices = _read(price_path)
    market_bundle = _market_bundle()
    input_hashes, context_support_hashes, contexts = {}, {}, {}
    cabbage_origins = sorted({r["origin"] for r in rows if r["product"] == "cabbage"})
    for origin in cabbage_origins:
        cutoff = origin - dt.timedelta(days=1)
        series = _cabbage_series_as_of(raw_prices, cutoff)
        states = {(day, market): state for (product, day, market), state in market_bundle["states"].items()
                  if product == "cabbage"}
        matrix, support = c2._build_context(series, market_bundle["markets"]["cabbage"], origin, states)
        contexts[origin] = {"matrix": matrix, "support": support}
        context_support_hashes[origin.isoformat()] = _sha(_canonical(support))
        for cond, task_matrix in (("U", matrix[:1]), ("M", matrix)):
            input_hashes[c2._task_id("cabbage", origin, cond)] = _sha(task_matrix.tobytes())
    market_hash = _sha(_canonical({"markets": [[p,m,d.isoformat(),v["price"],v["weight"]]
        for p in sorted(market_bundle["markets"]) for m in sorted(market_bundle["markets"][p])
        for d,v in sorted(market_bundle["markets"][p][m].items())], "states": market_bundle["provenance"]}))
    input_doc = {"rhsTManifestSha256": RHS_T_SHA256, "stageASelectionSha256": SELECTION_SHA256,
                 "stageAProtocolSha256": SELECTION_PROTOCOL_SHA256,
                 "priceSnapshotSha256": _file_sha(price_path), "marketDataSha256": market_hash,
                 "modelTaskInputs": input_hashes, "contextSupportSha256": context_support_hashes,
                 "BKeyUniverseSha256": _sha(_canonical([
                     [p,o.isoformat(),t.isoformat(),h,base[(p,o,t,h)]["B"]]
                     for p,o,t,h in sorted(base)])),
                 "taskCount": len(input_hashes), "cabbageKeyCount": sum(r["product"] == "cabbage" for r in rows),
                 "cabbageOriginCount": len(cabbage_origins), "databaseSize": market_bundle["databaseSize"]}
    input_doc["inputSha256"] = _sha(_canonical(input_doc))
    return {"selection": selection, "rows": rows, "base": base, "contexts": contexts,
            "input": input_doc, "marketBundle": market_bundle, "rawPrices": raw_prices,
            "pricePath": price_path}


def freeze_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    """Read only issue-time covariates and baseline keys; do not convert target actuals."""
    env = c2._check_runtime()
    c2._verify_cached_weights()
    prepared = _prepare()
    source_hashes = {name: _file_sha(path) for name, path in _source_files().items()}
    protocol = {"schemaVersion": 1, "experiment": "CHRONOS2_CABBAGE_FOLLOWUP_V1",
        "stage": "B_frozen_cabbage_T_forecast_then_score",
        "stageASelectionSha256": SELECTION_SHA256, "stageAProtocolSha256": SELECTION_PROTOCOL_SHA256,
        "selectedCandidate": "HM", "sourceHashes": source_hashes,
        "sourceRun": {"rhsTManifestSha256": RHS_T_SHA256,
                      "originalChronosConfigSha256": _file_sha(c2.OUT / "config.yaml"),
                      "originalChronosVManifestSha256": _file_sha(c2.OUT / "validation-manifest.json"),
                      "representativePriceSnapshotSha256": _file_sha(prepared["pricePath"]),
                      "marketDatabasePath": "data/agri.sqlite",
                      "marketDatabaseSize": prepared["input"]["databaseSize"]},
        "model": {"id": c2.MODEL_ID, "revision": c2.MODEL_REVISION,
                  "weightSha256": c2.MODEL_WEIGHT_SHA256, "weightSize": c2.MODEL_WEIGHT_SIZE,
                  "configSha256": c2.MODEL_CONFIG_SHA256,
                  "cachedWeightsVerified": True, "runtime": env},
        "input": prepared["input"],
        "T": {"targetStart": TARGET_START.isoformat(), "targetEnd": TARGET_END.isoformat(),
              "originStart": ORIGIN_START.isoformat(), "priceUnit": "KRW_per_kg",
              "contextCalendarDays": 730, "cutoff": "origin minus 1 calendar day",
              "inputColumns": {"U": ["representativePrice"], "M": ["representativePrice", *c2.MARKETS]},
              "predictionLength": 11, "quantile": 0.5, "predictionIndex": "horizon 1..10 indexes 1..10",
              "independentTaskCount": 176, "cabbageOriginCount": 88,
              "allBKeys": EXPECTED_KEYS, "cabbageBKeys": EXPECTED_CABBAGE_KEYS},
        "blends": {"HU": "0.5*B+0.5*U", "HM": "0.5*B+0.5*M", "S": "HM (Stage A locked)"},
        "truthSeparation": {"forecastArtifactMustSealFirst": True,
            "T numeric actuals materialized only in score command after forecast manifest hash validation": True,
            "noFallback": True},
        "evaluation": {"primary": "S versus B", "secondary": "HM versus HU",
            "gate": {"equalWeightCells": 40, "mape": "S < B", "mae": "S <= B", "coverageLoss": 0},
            "dm": {"hypothesesPerBandwidth": 40, "comparisons": ["S_vs_B", "HM_vs_HU"],
                   "horizons": list(HORIZONS), "losses": ["ae", "ape"],
                   "bandwidths": list(HAC_BANDWIDTHS), "holmAlpha": ALPHA},
            "bootstrap": {"blocks": list(BLOCKS), "replicates": REPS, "seed": SEED,
                          "S_vs_B": "all 40 crop-horizon cells", "HM_vs_HU": "cabbage 10 horizon cells"}},
        "expectedForecastRows": EXPECTED_KEYS, "expectedPredictionTasks": 176}
    protocol["protocolSha256"] = _sha(_canonical(protocol))
    _write_once(output_dir / "test-protocol.json", _canonical(protocol))
    return protocol


def _load_frozen(output_dir: Path) -> dict[str, Any]:
    doc = _read(output_dir / "test-protocol.json")
    sha = doc.pop("protocolSha256", None)
    if sha != _sha(_canonical(doc)):
        raise ValueError("Stage B protocol self-hash mismatch")
    doc["protocolSha256"] = sha
    if (doc.get("stageASelectionSha256") != SELECTION_SHA256 or doc.get("selectedCandidate") != "HM"
            or doc.get("input", {}).get("inputSha256") != _prepare()["input"]["inputSha256"]):
        raise ValueError("frozen Stage B selection/input bindings changed")
    current = {name: _file_sha(path) for name, path in _source_files().items()}
    if current != doc.get("sourceHashes"):
        raise ValueError("Stage B source code/plan changed after freeze")
    env = c2._check_runtime()
    c2._verify_cached_weights()
    model = doc.get("model", {})
    if (env != model.get("runtime") or model.get("id") != c2.MODEL_ID
            or model.get("revision") != c2.MODEL_REVISION
            or model.get("weightSha256") != c2.MODEL_WEIGHT_SHA256
            or model.get("weightSize") != c2.MODEL_WEIGHT_SIZE
            or model.get("configSha256") != c2.MODEL_CONFIG_SHA256
            or model.get("cachedWeightsVerified") is not True):
        raise ValueError("current runtime/model cache differs from frozen protocol")
    return doc


def _task_matrix(prepared: Mapping[str, Any], origin: dt.date, condition: str):
    return prepared["contexts"][origin]["matrix"][:1] if condition == "U" else prepared["contexts"][origin]["matrix"]


def _forecast_doc(prepared: Mapping[str, Any], task_results: Mapping[tuple[dt.date, str], Mapping[str, Any]],
                  protocol: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    forecasts = []
    for source in prepared["rows"]:
        p, origin, target, h = source["product"], source["origin"], source["target"], source["horizon"]
        b = source["B"]
        conditions: dict[str, Any] = {"B": {"status": "issued", "prediction": b,
            "sourceStatus": source["baselineSourceStatus"], "fallbackReason": source["baselineFallbackReason"]}}
        if p == "cabbage":
            tasks = {cond: task_results[(origin, cond)] for cond in ("U", "M")}
            preds = {cond: tasks[cond]["representativePredictions"][h] for cond in ("U", "M")}
            for cond in ("U", "M"):
                result, pred = tasks[cond], preds[cond]
                valid = result["status"] == "complete" and pred is not None and math.isfinite(pred) and pred > 0
                conditions[cond] = {"status": "issued" if valid else
                                    "inference_error" if result["status"] == "failed" else "invalid_prediction",
                                    "prediction": pred if valid else None,
                                    "reason": result.get("failure") if result["status"] == "failed" else
                                              None if valid else "q50_not_finite_positive",
                                    "taskId": result["taskId"]}
            for name, cond in (("HU", "U"), ("HM", "M")):
                value = conditions[cond]["prediction"]
                valid = value is not None
                conditions[name] = {"status": "issued" if valid else "not_issued",
                    "prediction": 0.5*b+0.5*value if valid else None,
                    "reason": None if valid else f"{cond}_no_valid_prediction"}
            conditions["S"] = dict(conditions["HM"])
            conditions["S"]["selectedCandidate"] = "HM"
            support = prepared["contexts"][origin]["support"]
        else:
            for name in ("U", "M", "HU", "HM"):
                conditions[name] = {"status": "not_applicable", "prediction": None,
                                    "reason": "new_inference_restricted_to_cabbage"}
            conditions["S"] = {"status": "issued", "prediction": b, "reason": "unchanged_RHS_C"}
            support = {"cutoff": (origin-dt.timedelta(days=1)).isoformat(), "firstDate": None,
                       "lastDate": (origin-dt.timedelta(days=1)).isoformat(), "observedCounts": {},
                       "marketStateCounts": {}}
        forecasts.append({"key": {"product": p, "originDate": origin.isoformat(),
                                  "targetDate": target.isoformat(), "horizon": h},
                          "conditions": conditions, "context": support})
    return {"schemaVersion": 1, "phase": "T", "protocolSha256": protocol["protocolSha256"],
            "inputSha256": protocol["input"]["inputSha256"], "stageASelectionSha256": SELECTION_SHA256,
            "selectedCandidate": "HM", "targetValuesAttached": False, "forecasts": forecasts}


def _verify_saved_forecasts(output_dir: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions_path = output_dir / "test-predictions.json"
    manifest_path = output_dir / "test-predictions.manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("manifestSha256") != _sha(_canonical({k:v for k,v in manifest.items() if k != "manifestSha256"})):
        raise ValueError("test prediction manifest self-hash mismatch")
    if manifest.get("protocolSha256") != protocol["protocolSha256"]:
        raise ValueError("test predictions were not made under frozen protocol")
    if _file_sha(predictions_path) != manifest.get("predictionsSha256"):
        raise ValueError("test prediction file hash mismatch")
    doc = _read(predictions_path)
    if doc.get("targetValuesAttached") is not False or doc.get("phase") != "T":
        raise ValueError("test prediction artifact is not answer-free T output")
    if len(doc.get("forecasts", [])) != EXPECTED_KEYS:
        raise ValueError("test prediction key count differs from frozen RHS universe")
    return manifest, doc


def _replay_task_outputs(output_dir: Path, prepared: Mapping[str, Any],
                         protocol: Mapping[str, Any], manifest: Mapping[str, Any],
                         predictions: Mapping[str, Any]) -> None:
    task_dir = output_dir / "task-checkpoints"
    results = {}
    for origin in sorted(prepared["contexts"]):
        for condition in ("U", "M"):
            task_id = c2._task_id("cabbage", origin, condition)
            path = task_dir / f"{task_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing saved T task checkpoint: {path.name}")
            result = c2._task_checkpoint(path, "cabbage", origin, condition,
                         _task_matrix(prepared, origin, condition), None)
            results[(origin, condition)] = result
    hashes = {v["taskId"]: v["taskSha256"] for v in results.values()}
    if (hashes != manifest.get("taskSha256") or len(results) != manifest.get("newTaskCount")
            or len(results) != EXPECTED_ORIGINS*2):
        raise ValueError("saved T task checkpoint hashes/count differ from manifest")
    rebuilt = _forecast_doc(prepared, results, protocol, OUT)
    if _canonical(rebuilt) != _canonical(predictions):
        raise ValueError("saved answer-free T rows fail baseline/task replay")


def _claim_sha(output_dir: Path, protocol: Mapping[str, Any]) -> str:
    claim = _read(output_dir / "test-run-claim.json")
    claim_sha = claim.pop("claimSha256", None)
    if (claim_sha != _sha(_canonical(claim)) or claim.get("protocolSha256") != protocol["protocolSha256"]
            or claim.get("inputSha256") != protocol["input"]["inputSha256"]):
        raise ValueError("T run claim invalid or bound to different frozen inputs")
    return claim_sha


def _ensure_terminal(output_dir: Path, protocol: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    claim_sha = _claim_sha(output_dir, protocol)
    path = output_dir / "test-run-terminal.json"
    if path.exists():
        terminal = _read(path)
        saved_sha = terminal.pop("terminalSha256", None)
        if (saved_sha != _sha(_canonical(terminal)) or terminal.get("status") != "complete"
                or terminal.get("claimSha256") != claim_sha
                or terminal.get("protocolSha256") != protocol["protocolSha256"]
                or terminal.get("manifestSha256") != manifest["manifestSha256"]):
            raise ValueError("T forecast terminal does not bind saved manifest")
        terminal["terminalSha256"] = saved_sha
        return terminal
    terminal = {"status": "complete", "claimSha256": claim_sha,
                "protocolSha256": protocol["protocolSha256"], "manifestSha256": manifest["manifestSha256"],
                "finishedAtKst": dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
    terminal["terminalSha256"] = _sha(_canonical(terminal))
    _write_or_match(path, _canonical(terminal))
    return terminal


def build_predictions(output_dir: Path = OUT) -> dict[str, Any]:
    protocol = _load_frozen(output_dir)
    prepared = _prepare()
    if prepared["input"]["inputSha256"] != protocol["input"]["inputSha256"]:
        raise ValueError("T forecast inputs differ from frozen protocol")
    prediction_path, manifest_path = output_dir / "test-predictions.json", output_dir / "test-predictions.manifest.json"
    if manifest_path.exists():
        manifest, saved = _verify_saved_forecasts(output_dir, protocol)
        _replay_task_outputs(output_dir, prepared, protocol, manifest, saved)
        _ensure_terminal(output_dir, protocol, manifest)
        return manifest
    claim_path = output_dir / "test-run-claim.json"
    claim = {"protocolSha256": protocol["protocolSha256"], "inputSha256": protocol["input"]["inputSha256"],
             "selectionSha256": SELECTION_SHA256, "status": "running",
             "startedAtKst": dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
    if claim_path.exists():
        saved = _read(claim_path)
        claim_sha = saved.pop("claimSha256", None)
        if claim_sha != _sha(_canonical(saved)) or saved.get("inputSha256") != claim["inputSha256"]:
            raise ValueError("existing T run claim invalid or bound to other inputs")
        claim = saved
        claim["claimSha256"] = claim_sha
    else:
        claim["claimSha256"] = _sha(_canonical(claim))
        _write_once(claim_path, _canonical(claim))
    origins = sorted(prepared["contexts"])
    task_dir = output_dir / "task-checkpoints"
    task_results = {}
    pipeline = None
    for origin in origins:
        for cond in ("U", "M"):
            task_id = c2._task_id("cabbage", origin, cond)
            path = task_dir / f"{task_id}.json"
            if not path.exists() and pipeline is None:
                pipeline = c2._load_pipeline()
            task_results[(origin, cond)] = c2._task_checkpoint(path, "cabbage", origin, cond,
                                      _task_matrix(prepared, origin, cond), pipeline)
    forecast_doc = _forecast_doc(prepared, task_results, protocol, output_dir)
    raw = _canonical(forecast_doc)
    _write_or_match(prediction_path, raw)
    task_hashes = {result["taskId"]: result["taskSha256"] for result in task_results.values()}
    task_status = {cond: {status: sum(result["status"] == status for (origin, task_cond), result in task_results.items()
                                      if task_cond == cond) for status in ("complete", "failed")}
                   for cond in ("U", "M")}
    manifest = {"schemaVersion": 1, "phase": "T", "protocolSha256": protocol["protocolSha256"],
                "inputSha256": protocol["input"]["inputSha256"], "selectionSha256": SELECTION_SHA256,
                "selectedCandidate": "HM", "predictionsFile": prediction_path.name,
                "predictionsSha256": _sha(raw), "predictionRows": len(forecast_doc["forecasts"]),
                "newTaskCount": len(task_results), "taskSha256": task_hashes,
                "taskIds": sorted(task_hashes), "cabbageBKeys": EXPECTED_CABBAGE_KEYS,
                "taskStatus": task_status,
                "cabbageOriginCount": len(origins), "targetValuesAttached": False,
                "createdAtKst": claim["startedAtKst"]}
    manifest["manifestSha256"] = _sha(_canonical(manifest))
    _write_or_match(manifest_path, _canonical(manifest))
    _ensure_terminal(output_dir, protocol, manifest)
    return manifest


def _stage_b_actuals(raw: Mapping[str, Any]) -> dict[tuple[str, dt.date], float]:
    products = raw.get("products")
    if raw.get("schemaVersion") != 1 or not isinstance(products, list):
        raise ValueError("pinned price snapshot schema mismatch")
    filtered = []
    for item in products:
        pid = item.get("id")
        observations = []
        if pid in PRODUCTS:
            for row in item.get("observations", []):
                day = row.get("date")
                if isinstance(day, str) and TARGET_START.isoformat() <= day <= TARGET_END.isoformat():
                    observations.append(dict(row))
        filtered.append({"id": pid, "observations": observations})
    from . import direct_horizon as dh
    observed = dh._observations({"schemaVersion": 1, "products": filtered})
    return {(product, day): values[0] for product, series in observed.items() for day, values in series.items()}


def _metrics(rows: list[Mapping[str, Any]], a: str, b: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "aMae": None, "bMae": None, "maeDifference": None,
                "aMape": None, "bMape": None, "mapeDifference": None}
    ae_a = [abs(r[a]-r["actual"]) for r in rows]
    ae_b = [abs(r[b]-r["actual"]) for r in rows]
    mae_a, mae_b = math.fsum(ae_a)/len(rows), math.fsum(ae_b)/len(rows)
    mape_a = math.fsum(e/r["actual"]*100 for e,r in zip(ae_a,rows))/len(rows)
    mape_b = math.fsum(e/r["actual"]*100 for e,r in zip(ae_b,rows))/len(rows)
    return {"n": len(rows), "aMae": mae_a, "bMae": mae_b, "maeDifference": mae_a-mae_b,
            "aMape": mape_a, "bMape": mape_b, "mapeDifference": mape_a-mape_b}


def _fixed_cells(rows: list[Mapping[str, Any]], left: str, right: str, products: tuple[str, ...]) -> list[dict[str, Any]]:
    cells = []
    for product in products:
        for h in HORIZONS:
            cell = [r for r in rows if r["product"] == product and r["horizon"] == h
                    and r.get(left) is not None and r.get(right) is not None]
            metrics = _metrics(cell, left, right)
            cells.append({"product": product, "horizon": h, **metrics})
    return cells


def _equal_cell_summary(rows: list[Mapping[str, Any]], left: str, right: str,
                        products: tuple[str, ...]) -> dict[str, Any]:
    cells = _fixed_cells(rows, left, right, products)
    expected = len(products) * len(HORIZONS)
    supported = [cell for cell in cells if cell["n"] > 0]
    complete = len(supported) == expected
    metrics = {key: (math.fsum(cell[key] for cell in supported) / len(supported)
                     if complete else None)
               for key in ("aMae", "bMae", "maeDifference", "aMape", "bMape", "mapeDifference")}
    return {"supportedCells": len(supported), "expectedCells": expected,
            "allCellsSupported": complete, **metrics}


def _dm_family(rows: list[Mapping[str, Any]], comparison: str, left: str, right: str,
               products: tuple[str, ...]) -> list[dict[str, Any]]:
    from . import equal_info_online_score as inference
    result = []
    for h in HORIZONS:
        relevant = [r for r in rows if r["horizon"] == h and r["product"] in products
                    and r.get(left) is not None and r.get(right) is not None]
        for loss in ("ae", "ape"):
            daily: dict[dt.date, list[float]] = defaultdict(list)
            for row in relevant:
                actual = row["actual"]
                e1, e2 = abs(row[left]-actual), abs(row[right]-actual)
                daily[row["target"]].append((e1-e2) if loss == "ae" else 100*(e1-e2)/actual)
            diff = {day: math.fsum(values)/len(values) for day, values in daily.items()}
            for bandwidth in HAC_BANDWIDTHS:
                identical = bool(relevant) and all(r[left] == r[right] for r in relevant)
                out = inference._dm(diff, bandwidth)
                if identical:
                    out = {"status": "identical_predictions", "reason": "same_saved_price_for_every_pair",
                           "n": len(diff), "statistic": None, "meanLossDifference": 0.0, "pValue": 1.0}
                result.append({"comparison": comparison, "horizon": h, "loss": loss,
                               "bandwidth": bandwidth, **out})
    return result


def score_report(predictions: Mapping[str, Any], actuals: Mapping[tuple[str, dt.date], float],
                 manifest: Mapping[str, Any]) -> dict[str, Any]:
    from . import research_progress_diagnostic as diagnostic
    paired = []
    coverage = {p: {"B": 0, "S": 0, "actual": 0, "evaluation": 0} for p in PRODUCTS}
    for item in predictions["forecasts"]:
        key = item["key"]
        product, target, horizon = key["product"], dt.date.fromisoformat(key["targetDate"]), key["horizon"]
        cond = item["conditions"]
        if cond["B"]["status"] == "issued": coverage[product]["B"] += 1
        if cond["S"]["status"] == "issued": coverage[product]["S"] += 1
        actual = actuals.get((product, target))
        if actual is not None: coverage[product]["actual"] += 1
        if actual is not None and cond["B"]["status"] == "issued" and cond["S"]["status"] == "issued":
            coverage[product]["evaluation"] += 1
        row = {"product": product, "origin": dt.date.fromisoformat(key["originDate"]),
               "target": target, "horizon": horizon, "actual": actual,
               "B": cond["B"]["prediction"], "S": cond["S"]["prediction"],
               "HU": cond["HU"]["prediction"], "HM": cond["HM"]["prediction"]}
        paired.append(row)
    positive = [r for r in paired if r["actual"] is not None and r["B"] is not None]
    primary_rows = [r for r in positive if r["S"] is not None]
    secondary_rows = [r for r in positive if r["HU"] is not None and r["HM"] is not None]
    primary_cells = _fixed_cells(primary_rows, "S", "B", PRODUCTS)
    secondary_cells = _fixed_cells(secondary_rows, "HM", "HU", ("cabbage",))
    cabbage_rows = [r for r in positive if r["product"] == "cabbage"]
    cabbage_forecasts = [r for r in paired if r["product"] == "cabbage"]
    cabbage_s_b = _equal_cell_summary(cabbage_rows, "S", "B", ("cabbage",))
    cabbage_hu_b = _equal_cell_summary(cabbage_rows, "HU", "B", ("cabbage",))
    monthly_equal_cells = []
    for month in ("2026-07", "2026-08", "2026-09"):
        month_primary = [r for r in primary_rows if r["target"].strftime("%Y-%m") == month]
        month_secondary = [r for r in secondary_rows if r["target"].strftime("%Y-%m") == month]
        monthly_equal_cells.append({"month": month,
            "primary_S_vs_B": _equal_cell_summary(month_primary, "S", "B", PRODUCTS),
            "secondary_HM_vs_HU": _equal_cell_summary(month_secondary, "HM", "HU", ("cabbage",))})
    all_primary = len(primary_cells) == 40 and all(c["n"] > 0 for c in primary_cells)
    macro = None if not all_primary else {name: math.fsum(c[name] for c in primary_cells)/40
        for name in ("aMae", "bMae", "maeDifference", "aMape", "bMape", "mapeDifference")}
    no_loss = all(coverage[p]["B"] == coverage[p]["S"] and
                  coverage[p]["actual"] == coverage[p]["evaluation"] for p in PRODUCTS)
    gate = {"all40CellsSupported": all_primary, "noCoverageLoss": no_loss,
            "mapeStrictlyLower": bool(all_primary and macro["aMape"] < macro["bMape"]),
            "maeNoHigher": bool(all_primary and macro["aMae"] <= macro["bMae"]),
            "passed": bool(all_primary and no_loss and macro["aMape"] < macro["bMape"] and macro["aMae"] <= macro["bMae"])}
    dm_primary = _dm_family(primary_rows, "S_vs_B", "S", "B", ("cabbage",))
    dm_secondary = _dm_family(secondary_rows, "HM_vs_HU", "HM", "HU", ("cabbage",))
    dm_all = dm_primary + dm_secondary
    from . import equal_info_online_score as inference
    for bandwidth in HAC_BANDWIDTHS:
        family = [r for r in dm_all if r["bandwidth"] == bandwidth]
        if len(family) != 40:
            raise AssertionError("fixed 40-test family was not preserved")
        inference._holm(family)
    boot = []
    for block in BLOCKS:
        comparisons = {}
        for name, sample, a, b in (("S_vs_B", primary_rows, "S", "B"),
                                    ("HM_vs_HU", secondary_rows, "HM", "HU")):
            product_set = set(PRODUCTS) if name == "S_vs_B" else {"cabbage"}
            fixed = {(p,h) for p in product_set for h in HORIZONS}
            supported = {(r["product"],r["horizon"]) for r in sample}
            if supported != fixed:
                comparisons[name] = {"blockDays": block, "requested": REPS, "valid": 0,
                                     "invalid": REPS, "interval95": None,
                                     "status": "unavailable_missing_fixed_cell"}
            else:
                samples = [dict(r, target=r["target"].isoformat()) for r in sample]
                metric_results = {}
                for metric in ("ae", "ape"):
                    raw = diagnostic.circular_block_bootstrap(samples, a=a, b=b, metric=metric, block=block,
                        replicates=REPS, seed=SEED, calendar_start=TARGET_START, calendar_end=TARGET_END)
                    metric_results[metric] = {**raw, "status": "available" if raw.get("interval95") is not None
                                              else "all_replicates_invalid"}
                comparisons[name] = metric_results
        boot.append({"blockDays": block, "seed": SEED, "replicates": REPS, "comparisons": comparisons})
    def detail(rows, left, right):
        return {"byHorizon": [{"horizon":h, **_metrics([r for r in rows if r["horizon"] == h], left, right)} for h in HORIZONS],
                "byMonth": [{"month":month, **_metrics([r for r in rows if r["target"].strftime('%Y-%m') == month], left, right)}
                            for month in ("2026-07", "2026-08", "2026-09")],
                "overall": _metrics(rows, left, right)}
    task_status = manifest.get("taskStatus", {})
    m_failed, u_failed = task_status.get("M", {}).get("failed", 0), task_status.get("U", {}).get("failed", 0)
    decision_reason = ("candidate_passed_frozen_price_gate" if gate["passed"] else
                       "technical_execution_incomplete" if m_failed else "frozen_price_gate_not_passed")
    report = {"schemaVersion": 1, "phase": "T", "protocolSha256": manifest["protocolSha256"],
        "manifestSha256": manifest["manifestSha256"], "primaryComparison": "S_vs_B",
        "secondaryComparison": "HM_vs_HU", "selectedCandidate": "HM", "gate": gate,
        "fortyCellEqualWeight": macro, "primaryCells": primary_cells,
        "secondaryCabbageCells": secondary_cells,
        "cabbageEqual10Cell": {"S_vs_B": cabbage_s_b, "HU_vs_B_diagnostic": cabbage_hu_b,
            "diagnosticCoverage": {
                "HUIssued": sum(r["HU"] is not None for r in cabbage_forecasts),
                "HMIssued": sum(r["HM"] is not None for r in cabbage_forecasts),
                "HUHMPositiveActualPairs": len(secondary_rows)}},
        "monthlyEqualCellSummaries": monthly_equal_cells,
        "descriptiveRowPooled": {"primary": detail(primary_rows, "S", "B"),
                                  "secondary": detail(secondary_rows, "HM", "HU")},
        "coverage": coverage,
        "technicalExecution": {"taskStatus": task_status, "MFailuresBlockPrimaryInference": m_failed > 0,
                               "UFailuresAffectOnlySecondaryDiagnostic": u_failed > 0},
        "fixedDMFamily": {"hypothesesPerBandwidth": 40, "alpha": ALPHA,
                          "tests": dm_all,
                          "primary": dm_primary, "secondary": dm_secondary},
        "bootstrap": boot,
        "limitations": ["This is a post-selected retrospective follow-up on a repeatedly inspected period, not independent or future validation.",
                        "HU/HM component comparisons do not establish causal market effects or general pretrained-model performance."],
        "decision": "E106-C2-CAB" if gate["passed"] else "E106-RHS",
        "decisionReason": decision_reason}
    return report


def score_phase(output_dir: Path = OUT) -> dict[str, Any]:
    protocol = _load_frozen(output_dir)
    manifest, predictions = _verify_saved_forecasts(output_dir, protocol)
    prepared = _prepare()
    if prepared["input"]["inputSha256"] != manifest["inputSha256"]:
        raise ValueError("T forecast inputs no longer match saved manifest")
    _replay_task_outputs(output_dir, prepared, protocol, manifest, predictions)
    # Recover a terminal lost after manifest publication; truth remains unread until this validates.
    _ensure_terminal(output_dir, protocol, manifest)
    actuals = _stage_b_actuals(prepared["rawPrices"])
    report = score_report(predictions, actuals, manifest)
    report["predictionFileSha256"] = manifest["predictionsSha256"]
    report["actualSnapshotSha256"] = _file_sha(prepared["pricePath"])
    raw = _canonical(report)
    _write_or_match(output_dir / "test-report.json", raw)
    decision = _decision_text(report)
    _write_or_match(output_dir / "decision.md", decision.encode("utf-8"))
    return report


def _decision_text(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    reason = report.get("decisionReason")
    if reason == "technical_execution_incomplete":
        outcome = ("Technical execution was incomplete because one or more selected M tasks failed; the frozen price gate is therefore not inferentially complete. "
                   "Preserve E106-RHS and make no price-performance conclusion for this follow-up.")
    elif report["decision"] == "E106-C2-CAB":
        outcome = "The fixed S=HM candidate passed the frozen equal-cell MAE/MAPE and coverage gate. Preserve E106-C2-CAB as a post-selected retrospective candidate."
    else:
        outcome = "The fixed S=HM candidate did not pass the frozen gate or lacked support. Preserve E106-RHS and end this cabbage follow-up."
    return ("# Chronos-2 cabbage retrospective follow-up\n\n" + outcome + "\n\n"
            + f"Gate: {json.dumps(gate, sort_keys=True)}\n\n"
            + f"Technical tasks: {json.dumps(report.get('technicalExecution', {}).get('taskStatus', {}), sort_keys=True)}\n\n"
            + "Target-month summaries are equal-cell-weighted; `descriptiveRowPooled` is unweighted row-pooled context only.\n\n"
            + "This is a post-selected historical follow-up on a repeatedly inspected period, not independent confirmation or future-performance evidence. HM-HU is diagnostic and cannot replace the preselected S. No operational forecast was changed.\n")


def run(output_dir: Path = OUT) -> dict[str, Any]:
    manifest = build_predictions(output_dir)
    report = score_phase(output_dir)
    return {"manifestSha256": manifest["manifestSha256"], "decision": report["decision"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "build", "score", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.command == "freeze": result = freeze_protocol(args.output_dir)
    elif args.command == "build": result = build_predictions(args.output_dir)
    elif args.command == "score": result = score_phase(args.output_dir)
    else: result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
