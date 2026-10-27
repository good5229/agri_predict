"""Monthly D/F training for FINAL_ENSEMBLE_LOSS_ONION_V2.

This module deliberately imports the frozen direct-path implementation for its
model and causal feature builders.  It contains no API client and never loads
target answers for forecast-only records.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import time
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import direct_price_path as frozen
from . import direct_price_path_data as data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/final-ensemble-loss-onion-v2-20261014"
BASELINE = ROOT / "agri_data/final_ensemble_onion_baseline.py"
PRODUCT = "onion"
CONTEXT_DAYS = 84
INPUT_TYPE = "M"
REPAIR = "base"
SEED = 42
LAMBDA_BASE = 1.0
LAMBDA_REPAIR = 3.0
MAX_EPOCHS = 200
EXTENDED_EPOCHS = 400
PATIENCE = 25
TIE_TOL = 1e-8
HORIZONS = tuple(range(1, 11))
PROTOCOL_SHA256 = "a7077463d787766f0f7196a16adf690511db53eee26da03c500dd9293bf39a89"
CONFIG_PATH = OUT / "resolved-config.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_body() -> dict[str, Any]:
    old = json.loads((frozen.OUT / "resolved-config.json").read_text(encoding="utf-8"))
    # The frozen model module resolves this input through its data helper;
    # the cache path is defined there, not on direct_price_path itself.
    source_path = Path(data.SOURCE_CACHE_PATH)
    if (str(source_path.relative_to(ROOT)) != old.get("sourceCachePath")
            or old.get("sourceCacheInternalShaField") != "sourceCacheSha256"):
        raise ValueError("source-cache path/schema differs from the approved prior resolved config")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    files = {
        "agri_data/final_ensemble_onion.py": file_sha256(Path(__file__)),
        "agri_data/final_ensemble_onion_baseline.py": file_sha256(BASELINE),
        "agri_data/final_ensemble_onion_export.py": file_sha256(ROOT/"agri_data/final_ensemble_onion_export.py"),
        "agri_data/final_ensemble_onion_score.py": file_sha256(ROOT/"agri_data/final_ensemble_onion_score.py"),
        "agri_data/direct_price_path.py": file_sha256(Path(frozen.__file__)),
        "agri_data/direct_price_path_data.py": file_sha256(Path(data.__file__)),
    }
    return {
        "schemaVersion": 1, "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
        "protocolPath": str((OUT / "protocol.md").relative_to(ROOT)),
        "protocolSha256": PROTOCOL_SHA256,
        "producerCodeSha256": files,
        "priorResolvedConfigPath": str((frozen.OUT / "resolved-config.json").relative_to(ROOT)),
        "priorResolvedConfigSha256": frozen._file_sha(frozen.OUT / "resolved-config.json"),
        "sourceCachePath": str(source_path.relative_to(ROOT)),
        "sourceCacheFileSha256": file_sha256(source_path),
        "sourceCacheInternalSha256": frozen._read_json(source_path)["sourceCacheSha256"],
        "priceSnapshotPath": str(frozen.PRICE_PATH.relative_to(ROOT)),
        "priceSnapshotSha256": file_sha256(frozen.PRICE_PATH),
        "model": {"contextDays": CONTEXT_DAYS, "inputType": INPUT_TYPE, "repair": REPAIR,
                  "seed": SEED, "dtype": "float32", "optimizer": "AdamW",
                  "learningRate": frozen.INITIAL_LR, "weightDecay": frozen.INITIAL_WEIGHT_DECAY,
                  "batchSize": frozen.BATCH_SIZE, "clipNorm": frozen.CLIP_NORM,
                  "devicePolicy": "native prior CPython/Torch/MPS or CPU fallback only"},
        "loss": {"lambdaBase": LAMBDA_BASE, "lambdaRepair": LAMBDA_REPAIR,
                 "weight": "1/y + lambda/q", "directObjective": "mean_h mean_valid(w*abs(ND-y))",
                 "finalObjective": "mean_h mean_valid(w*abs(.5*stop_gradient(B)+.5*NF-y))",
                 "optimizerMultiplier": {"D": 1.0, "F": 2.0},
                 "horizonAggregation": "equal horizon; global n_h denominators; partial labels masked"},
        "selection": {"innerDays": 90, "innerFitLabels": "target < g", "innerValidationOrigins": "origin >= g and target < f",
            "eligibility": "MAPE_H < MAPE_B and MAE_H <= MAE_B",
            "eligibleRank": ["MAPE", "MAE", "earliest_epoch"],
            "fallbackRank": ["max(MAPE/MAPE_B, MAE/MAE_B)", "mean_ratios", "earliest_epoch"],
            "ratioZeroDenominator": "0/0=1; positive/0=+infinity", "tieTolerance": TIE_TOL,
            "q": "median unique representative target-day prices among paired training labels; inner target<g, full target<f; shared D/F within split",
            "patience": PATIENCE, "maxEpochs": MAX_EPOCHS,
            "extension": "to400 only when best176-200 meaningfully outranks bestthrough175",
            "D25LambdaChoice": {"conditionForLambda3": "base H_F MAE > B",
                "eligibleChoice": "minimum raw H_F MAPE; lambda1 on ties",
                "fallbackChoice": "minimum worst relative ratio; lambda1 on ties"}},
        "schedule": {"phases": {"D25": ["2025-04-01", "2025-06-20"],
                                  "E25": ["2025-07-11", "2025-09-27"],
                                  "V26": ["2026-04-01", "2026-06-20"],
                                  "T26": ["2026-07-11", "2026-09-27"]},
                     "fitMonths": "first eligible baseline origin per origin month; six V/T months only",
                     "D25": "lambda1 always; lambda3 only if base H_F D25 MAE > B",
                     "refit": "same selected inner epoch; fresh paired D/F full training prefixes"},
        "answers": "forecasts are persisted answer-free; scoring joins pinned target snapshot later",
        "priorFrozenArchitecture": old["architecture"],
        "runtime": old["runtime"],
    }


def freeze_config() -> dict[str, Any]:
    body = _config_body()
    doc = {**body, "configSha256": sha256(canonical(body))}
    if CONFIG_PATH.exists():
        existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if existing != doc:
            raise FileExistsError("resolved config already exists with different inputs/settings")
        return existing
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_PATH.with_suffix(".json.tmp")
    temp.write_bytes(canonical(doc)); temp.replace(CONFIG_PATH)
    return doc


def validate_config() -> dict[str, Any]:
    doc = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    body = dict(doc); got = body.pop("configSha256", None)
    if got != sha256(canonical(body)) or body != _config_body():
        raise ValueError("resolved configuration self-hash/upstream/code pin mismatch")
    if file_sha256(OUT/"protocol.md") != PROTOCOL_SHA256:
        raise ValueError("frozen protocol file hash mismatch")
    return doc


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+infinity" if value > 0 else "-infinity" if value < 0 else "NaN"
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    raw = canonical(_json_safe(value))
    if path.exists():
        if immutable:
            if path.read_bytes() != raw:
                raise FileExistsError(f"refusing to replace immutable artifact {path}")
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_bytes(raw); tmp.replace(path)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def rank_metrics(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[float, float, int]:
    """Dimensionless B-relative rank; lower is better, raw eligibility is separate."""
    ratios = (_ratio(float(candidate["MAPE"]), float(baseline["MAPE"])),
              _ratio(float(candidate["MAE"]), float(baseline["MAE"])))
    return max(ratios), sum(ratios) / 2.0, 0


def _cmp_rank(a: tuple[float, float, int], b: tuple[float, float, int], tol: float = TIE_TOL) -> int:
    """Return -1 if a is better, +1 if worse, 0 for protocol ties."""
    for x, y in zip(a[:2], b[:2]):
        if math.isinf(x) or math.isinf(y):
            if x == y:
                continue
            return -1 if x < y else 1
        if abs(x - y) > tol:
            return -1 if x < y else 1
    return 0


def epoch_rank(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[float, float, int]:
    """Protocol order: eligible epochs first by MAPE/MAE; otherwise B-relative rank."""
    if (candidate["MAPE"] < baseline["MAPE"] and candidate["MAE"] <= baseline["MAE"]):
        return (_ratio(float(candidate["MAPE"]), float(baseline["MAPE"])),
                _ratio(float(candidate["MAE"]), float(baseline["MAE"])),
                int(candidate["epoch"]))
    return (*rank_metrics(candidate, baseline)[:2], int(candidate["epoch"]))


def _rank_cmp(a: Mapping[str, Any], b: Mapping[str, Any], baseline: Mapping[str, Any]) -> int:
    ae = a["MAPE"] < baseline["MAPE"] and a["MAE"] <= baseline["MAE"]
    be = b["MAPE"] < baseline["MAPE"] and b["MAE"] <= baseline["MAE"]
    if ae != be:
        return -1 if ae else 1
    ar, br = rank_metrics(a, baseline), rank_metrics(b, baseline)
    # Eligible epochs are ordered by relative MAPE then relative MAE; fallback
    # epochs use worst relative error then the mean ratio. Both are dimensionless.
    left, right = ((_ratio(float(a["MAPE"]), float(baseline["MAPE"])),
                    _ratio(float(a["MAE"]), float(baseline["MAE"]))),
                   (_ratio(float(b["MAPE"]), float(baseline["MAPE"])),
                    _ratio(float(b["MAE"]), float(baseline["MAE"])))) if ae else (ar, br)
    for x, y in zip(left[:2], right[:2]):
        if math.isinf(x) or math.isinf(y):
            if x == y:
                continue
            return -1 if x < y else 1
        if abs(x-y) > TIE_TOL:
            return -1 if x < y else 1
    return (a["epoch"] > b["epoch"]) - (a["epoch"] < b["epoch"])


def _metrics(records: Sequence[Mapping[str, Any]], prediction: Sequence[Sequence[float]],
             baseline: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Horizon-balanced raw-price metrics on a shared finite prediction support."""
    import numpy as np
    y = np.asarray([r["targets"] for r in records], dtype=np.float64)
    mask = np.asarray([r["targetMask"] for r in records], dtype=bool)
    p = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    per = {}
    for name, x in (("candidate", p), ("baseline", b)):
        maes, mapes, counts = [], [], []
        for h in range(10):
            ok = mask[:, h] & np.isfinite(x[:, h]) & (x[:, h] > 0) & (y[:, h] > 0)
            counts.append(int(ok.sum()))
            maes.append(float(np.mean(np.abs(x[ok, h]-y[ok, h]))) if ok.any() else None)
            mapes.append(float(np.mean(np.abs(x[ok, h]-y[ok, h])/y[ok, h])) if ok.any() else None)
        per[name] = {"MAE": float(np.mean(maes)) if all(v is not None for v in maes) else None,
                     "MAPE": float(np.mean(mapes)) if all(v is not None for v in mapes) else None,
                     "horizonMAE": maes, "horizonMAPE": mapes, "horizonN": counts}
    if per["candidate"]["horizonN"] != per["baseline"]["horizonN"]:
        raise ValueError("model and B metrics lack shared horizon support")
    return per


def _restore(model: Any, records: Sequence[Mapping[str, Any]], torch: Any) -> list[list[float]]:
    import numpy as np
    device = frozen._device(torch)
    return frozen._predict_records(model, records, device, torch)


def _rng_snapshot(torch: Any) -> dict[str, Any]:
    import numpy as np
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):
            state["mps"] = None
    return state


def _rng_restore(state: Mapping[str, Any], torch: Any) -> None:
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("mps") is not None and hasattr(torch, "mps"):
        torch.mps.set_rng_state(state["mps"])


def _with_b(records: Sequence[Mapping[str, Any]], b_rows: Mapping[tuple[dt.date, int], float]
            ) -> list[dict[str, Any]]:
    out = []
    for record in records:
        row = dict(record)
        b = []
        for h in HORIZONS:
            target = row["origin"] + dt.timedelta(days=h)
            value = b_rows.get((row["origin"], h))
            valid = row["targetMask"][h-1]
            if valid and (value is None or not _finite(value) or float(value) <= 0):
                raise ValueError(f"missing eligible B for paired onion label {row['origin']}/{target}/h{h}")
            b.append(float(value) if value is not None else math.nan)
        row["B"] = b
        out.append(row)
    return out


def q_for_records(records: Sequence[Mapping[str, Any]]) -> float:
    """Median unique representative target-day prices covered by this split."""
    prices: dict[dt.date, float] = {}
    for row in records:
        for i, (day, value, valid) in enumerate(zip(row["targetDays"], row["targets"], row["targetMask"])):
            if valid:
                target_day = dt.date.fromisoformat(day)
                prior = prices.setdefault(target_day, float(value))
                if prior != float(value):
                    raise ValueError(f"conflicting representative label for {target_day}")
    if not prices:
        raise ValueError("no supported representative target-day labels for q")
    q = float(statistics.median(prices.values()))
    if not math.isfinite(q) or q <= 0:
        raise ValueError("split q must be finite and positive")
    return q


def label_support(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prices: dict[dt.date, float] = {}
    for row in records:
        for day, value, valid in zip(row["targetDays"], row["targets"], row["targetMask"]):
            if valid:
                prices.setdefault(dt.date.fromisoformat(day), float(value))
    vals = list(prices.values())
    if not vals:
        raise ValueError("label support is empty")
    return {"uniqueTargetDays": len(prices), "priceMin": min(vals), "priceMax": max(vals),
            "medianPriceQ": float(statistics.median(vals))}


def load_baseline_rows(path: Path = OUT / "baseline-cache.json") -> dict[tuple[dt.date, int], float]:
    """Load the panel scorer's answer-free causal50 cache, rejecting unusable states."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("experiment") != "FINAL_ENSEMBLE_LOSS_ONION_V2"
            or doc.get("phase") != "answer_free" or doc.get("targetValuesAttached") is not False):
        raise ValueError("baseline cache is not the pinned answer-free experiment cache")
    rows: dict[tuple[dt.date, int], float] = {}
    for row in doc.get("rows", []):
        if row.get("product") != PRODUCT:
            continue
        origin = dt.date.fromisoformat(row["origin"])
        target = dt.date.fromisoformat(row["target"])
        horizon = int(row["horizon"])
        if not 1 <= horizon <= 10 or target != origin + dt.timedelta(days=horizon):
            raise ValueError("malformed onion baseline target key")
        key = (origin, horizon)
        if key in rows:
            raise ValueError(f"duplicate onion baseline key {key}")
        status = row.get("BSourceStatus")
        value = row.get("B")
        if status in ("issued", "fallback"):
            if not _finite(value) or float(value) <= 0:
                raise ValueError("issued/fallback B row needs a positive finite prediction")
            rows[key] = float(value)
        elif status != "not_issued" or value is not None:
            raise ValueError("baseline cache has an unknown/unresolved status")
    return rows


def month_split(fit_origin: dt.date, representatives: Mapping[dt.date, float],
                channels: Mapping[tuple[str, ...], Mapping[dt.date, Any]]) -> dict[str, Any]:
    """Construct fixed causal inner/full rows and one shared inner-selected channel set."""
    if fit_origin.day != 1 or fit_origin.year not in (2025, 2026):
        raise ValueError("fit origin must be the pinned first day of a 2025/2026 month")
    g = fit_origin - dt.timedelta(days=90)
    selected = data.select_channels_for_cutoff(channels, g-dt.timedelta(days=1))
    train_start = dt.date(2021, 1, 1)
    inner_train = frozen._records_for_period(PRODUCT, representatives, channels, CONTEXT_DAYS,
        INPUT_TYPE, selected, train_start, g, label_before=g)
    inner_validation = frozen._records_for_period(PRODUCT, representatives, channels, CONTEXT_DAYS,
        INPUT_TYPE, selected, max(train_start, g), fit_origin, label_before=fit_origin)
    full_train = frozen._records_for_period(PRODUCT, representatives, channels, CONTEXT_DAYS,
        INPUT_TYPE, selected, train_start, fit_origin, label_before=fit_origin)
    if not inner_train or not inner_validation or not full_train:
        raise ValueError(f"empty causal data split for fit origin {fit_origin}")
    q_inner, q_full = q_for_records(inner_train), q_for_records(full_train)
    return {"fitOrigin": fit_origin, "innerBoundary": g, "selectionCutoff": g-dt.timedelta(days=1),
            "selectedChannels": selected, "innerTrain": inner_train,
            "innerValidation": inner_validation, "fullTrain": full_train,
            "qInner": q_inner, "qFull": q_full}


def weighted_price_loss(prediction: Sequence[Sequence[float]], records: Sequence[Mapping[str, Any]],
                        q: float, penalty: float = 1.0, *, blend: Sequence[Sequence[float]] | None = None) -> float:
    """Exact equal-horizon mean of (1/y+lambda/q)|P-y| on valid labels."""
    import numpy as np
    p = np.asarray(blend if blend is not None else prediction, dtype=np.float64)
    y = np.asarray([r["targets"] for r in records], dtype=np.float64)
    mask = np.asarray([r["targetMask"] for r in records], dtype=bool)
    pieces = []
    for h in range(10):
        ok = mask[:, h]
        if ok.any():
            err = np.abs(p[ok, h] - y[ok, h])
            pieces.append(float(np.mean((1/y[ok, h] + penalty/q) * err)))
    if len(pieces) != 10:
        raise ValueError("each horizon needs at least one valid training label")
    return float(np.mean(pieces))


def cross_loss_diagnostics(records: Sequence[Mapping[str, Any]],
                           prediction_d: Sequence[Sequence[float]],
                           prediction_f: Sequence[Sequence[float]], q: float,
                           penalty: float) -> dict[str, Any]:
    """Same-row D/F cross-evaluation under both objectives and deployed prices."""
    import numpy as np
    paired = records
    nd = np.asarray(prediction_d, dtype=np.float64)
    nf = np.asarray(prediction_f, dtype=np.float64)
    b = np.asarray([r["B"] for r in paired], dtype=np.float64)
    hd, hf = .5*b+.5*nd, .5*b+.5*nf
    return {"D": {"LD": weighted_price_loss(nd, paired, q, penalty),
                  "LF": weighted_price_loss(nd, paired, q, penalty, blend=.5*b+.5*nd),
                  "price": _metrics(paired, hd, b)["candidate"]},
            "F": {"LD": weighted_price_loss(nf, paired, q, penalty),
                  "LF": weighted_price_loss(nf, paired, q, penalty, blend=hf),
                  "price": _metrics(paired, hf, b)["candidate"]},
            "B": _metrics(paired, b, b)["candidate"]}


def representation_floor(records: Sequence[Mapping[str, Any]],
                         prediction_f: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Equal-horizon bound induced by positive neural N in the deployed blend."""
    import numpy as np
    n = np.asarray(prediction_f, dtype=np.float64)
    b = np.asarray([r["B"] for r in records], dtype=np.float64)
    y = np.asarray([r["targets"] for r in records], dtype=np.float64)
    mask = np.asarray([r["targetMask"] for r in records], dtype=bool)
    if not np.isfinite(n[mask]).all() or (n[mask] <= 0).any():
        raise ValueError("representation floor requires finite positive N on scored support")
    by_horizon = []
    for h in range(10):
        valid = mask[:, h]
        count = int(valid.sum())
        if not count:
            raise ValueError("representation floor requires target support at each horizon")
        impossible = valid & (y[:, h] < .5*b[:, h])
        by_horizon.append({"horizon": h+1, "n": count,
            "strictlyAboveHalfBCount": int(np.sum(valid & (n[:, h] > 0)),),
            "targetBelowHalfBCount": int(impossible.sum()),
            "sumTargetBelowHalfBShortfall": float(np.maximum(.5*b[impossible, h]-y[impossible, h], 0).sum())})
    return {"weighting": "equal-horizon mean; within-horizon arithmetic mean",
            "allHorizons": by_horizon,
            "equalHorizonMeanShortfall": statistics.mean(r["sumTargetBelowHalfBShortfall"]/r["n"] for r in by_horizon),
            "equalHorizonMeanBelowBoundRate": statistics.mean(r["targetBelowHalfBCount"]/r["n"] for r in by_horizon),
            "strictlyAboveHalfBCount": sum(r["strictlyAboveHalfBCount"] for r in by_horizon),
            "targetBelowHalfBCount": sum(r["targetBelowHalfBCount"] for r in by_horizon),
            "sumTargetBelowHalfBShortfall": sum(r["sumTargetBelowHalfBShortfall"] for r in by_horizon)}


def _evaluate(model: Any, records: Sequence[Mapping[str, Any]], torch: Any,
              b_rows: Mapping[tuple[dt.date, int], float], q: float, penalty: float) -> dict[str, Any]:
    import numpy as np
    pred = _restore(model, records, torch)
    b = np.asarray([r["B"] for r in records], dtype=np.float64)
    h = .5*b + .5*np.asarray(pred, dtype=np.float64)
    score = _metrics(records, h, b)
    score["epoch"] = 0
    score["LD"] = weighted_price_loss(pred, records, q, penalty)
    score["LF"] = weighted_price_loss(pred, records, q, penalty, blend=h)
    return score


def train_one(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]],
              *, b_train: Mapping[tuple[dt.date, int], float],
              b_validation: Mapping[tuple[dt.date, int], float], q: float, penalty: float,
              fit_id: str, checkpoint_path: Path, optimizer_scale: float = 1.0,
              max_epochs: int = MAX_EPOCHS, baseline_validation: Mapping[str, Any] | None = None,
              provenance: Mapping[str, str] | None = None, phase: str = "unknown",
              objective: str = "inner-F", penalty_value: float | None = None,
              full_refit: bool = False
              ) -> dict[str, Any]:
    """Fit from scratch, selecting only on paired horizon-balanced price metrics.

    Checkpoints are atomically rewritten after each epoch and bind the complete
    row, q, and loss identity.  A resume path is intentionally not inferred.
    """
    import numpy as np
    np_runtime, torch, (nn, F) = frozen._torch_runtime()
    if not train_records or not validation_records or q <= 0 or penalty <= 0:
        raise ValueError("training requires rows, positive q, and positive lambda")
    train_records = _with_b(train_records, b_train)
    validation_records = _with_b(validation_records, b_validation)
    masks = np.asarray([r["targetMask"] for r in train_records], dtype=bool)
    nh = masks.sum(axis=0).astype(int).tolist()
    if any(n <= 0 for n in nh):
        raise ValueError("training support must be positive for all ten horizons")
    if baseline_validation is None:
        baseline_validation = {"MAE": 1.0, "MAPE": 1.0}
    frozen._seed_everything(torch, SEED)
    device = frozen._device(torch)
    model = frozen.build_model(CONTEXT_DAYS, REPAIR, torch=torch, nn=nn, F=F)
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=frozen.INITIAL_LR,
                                  weight_decay=frozen.INITIAL_WEIGHT_DECAY)
    identity = sha256(canonical({"fitId": fit_id, "trainOrigins": [r["origin"].isoformat() for r in train_records],
                                 "validationOrigins": [r["origin"].isoformat() for r in validation_records],
                                 "trainLabels": _json_safe([[r["targets"], r["targetMask"], r["B"]] for r in train_records]),
                                 "validationLabels": _json_safe([[r["targets"], r["targetMask"], r["B"]] for r in validation_records]),
                                 "nH": nh, "q": q, "lambda": penalty,
                                 "optimizerScale": optimizer_scale, "seed": SEED,
                                 "provenance": dict(provenance or {}),
                                 "contextDays": CONTEXT_DAYS, "inputType": INPUT_TYPE,
                                 "repair": REPAIR, "baseline": _json_safe([r["B"] for r in validation_records])}))
    history, best_state, best, patience = [], None, None, 0
    effective_limit = max_epochs
    epoch = 0; updates = 0; elapsed_prior = 0.0; started = time.monotonic()
    progress_path = OUT / "training-progress.json"
    def progress(status: str, current_epoch: int, best_epoch: int | None) -> None:
        _atomic_json(progress_path, {"schemaVersion": 1, "phase": phase,
            "lambda": penalty_value if penalty_value is not None else penalty,
            "fitId": fit_id, "objective": objective, "fullRefit": full_refit,
            "status": status, "epoch": current_epoch, "bestEpoch": best_epoch,
            "updates": updates, "elapsedSeconds": elapsed_prior + time.monotonic() - started})
    progress("fit_started", epoch, None)
    if checkpoint_path.is_file():
        try:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # Torch versions without the weights_only keyword
            saved = torch.load(checkpoint_path, map_location="cpu")
        if saved.get("fitIdentity") != identity:
            raise ValueError(f"checkpoint identity mismatch for {fit_id}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best_state, best = saved["history"], saved["bestModel"], saved["best"]
        patience, epoch = int(saved["patience"]), int(saved["epoch"])
        updates=int(saved.get("updates",0)); elapsed_prior=float(saved.get("elapsedSeconds",0.0))
        effective_limit = int(saved.get("effectiveLimit", max_epochs))
        _rng_restore(saved["rngState"], torch)
        if patience >= PATIENCE:
            effective_limit = epoch
    # Recheck the epoch-200 extension decision on resume in case the process
    # stopped after its checkpoint write but before persisting the extension.
    if epoch == MAX_EPOCHS and effective_limit == MAX_EPOCHS:
        early = [r for r in history if r["epoch"] <= 175]
        late = [r for r in history if 176 <= r["epoch"] <= 200]
        if early and late:
            early_best = early[0]
            for row in early[1:]:
                if _rank_cmp(row["candidate"], early_best["candidate"], row["baseline"]) < 0:
                    early_best = row
            late_best = late[0]
            for row in late[1:]:
                if _rank_cmp(row["candidate"], late_best["candidate"], row["baseline"]) < 0:
                    late_best = row
            if _rank_cmp(late_best["candidate"], early_best["candidate"], late_best["baseline"]) < 0:
                effective_limit = EXTENDED_EPOCHS
                temp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                torch.save({"fitIdentity": identity, "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best": best, "bestModel": best_state,
                    "history": history, "patience": patience, "effectiveLimit": effective_limit,
                    "updates": updates, "elapsedSeconds": elapsed_prior,
                    "rngState": _rng_snapshot(torch)}, temp)
                temp.replace(checkpoint_path)
    while epoch < effective_limit:
        epoch += 1
        model.train()
        order = list(range(len(train_records)))
        random.Random(SEED + epoch).shuffle(order)
        for start in range(0, len(order), frozen.BATCH_SIZE):
            inds = order[start:start+frozen.BATCH_SIZE]
            batch = [train_records[i] for i in inds]
            z, obs, active, roles, meta, cal, y_cpu, mask_cpu = frozen._as_tensor_batch(
                batch, [r["targets"] for r in batch], [r["targetMask"] for r in batch], device, torch)
            delta = model(z, obs, active, roles, meta, cal).to(device="cpu").to(dtype=torch.float64)
            p0 = torch.as_tensor([r["window"]["p0"] for r in batch], dtype=torch.float64)
            sigma = torch.as_tensor([r["window"]["representativeSigma"] for r in batch], dtype=torch.float64)
            pred = torch.exp(torch.log(p0).unsqueeze(1) + sigma.unsqueeze(1)*delta)
            b = torch.as_tensor([r["B"] for r in batch], dtype=torch.float64)
            h = .5*b + .5*pred
            actual, valid = y_cpu, mask_cpu
            pieces = []
            for j in range(10):
                sel = valid[:, j]
                if bool(sel.any()):
                    err = (h[sel, j]-actual[sel, j]).abs()
                    pieces.append(((1/actual[sel, j] + penalty/q)*err).sum()/nh[j])
            loss = torch.stack(pieces).sum() * (len(train_records)/len(batch)/10) * optimizer_scale
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite training loss in {fit_id} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), frozen.CLIP_NORM)
            optimizer.step()
            updates += 1
        current = _evaluate(model, validation_records, torch, b_validation, q, penalty)
        current["epoch"] = epoch
        current["candidate"]["epoch"] = epoch
        current["updates"] = updates
        current["elapsedSeconds"] = elapsed_prior + time.monotonic() - started
        rank = epoch_rank(current["candidate"], current["baseline"])
        # Use validation baseline metrics from exactly the paired rows.
        current["rank"] = list(rank)
        history.append(current)
        if best is None or _rank_cmp(current["candidate"], best["candidate"], current["baseline"]) < 0:
            best, best_state, patience = current, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if patience >= PATIENCE:
            effective_limit = epoch
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save({"fitIdentity": identity, "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best": best, "bestModel": best_state,
                    "history": history, "patience": patience, "effectiveLimit": effective_limit,
                    "updates": updates, "elapsedSeconds": current["elapsedSeconds"],
                    "rngState": _rng_snapshot(torch)}, temp)
        temp.replace(checkpoint_path)
        progress("epoch_completed", epoch, int(best["epoch"]))
        if patience >= PATIENCE:
            break
        if epoch == MAX_EPOCHS and max_epochs == MAX_EPOCHS:
            early = [r for r in history if r["epoch"] <= 175]
            late = [r for r in history if 176 <= r["epoch"] <= 200]
            if early and late:
                early_best = early[0]
                for row in early[1:]:
                    if _rank_cmp(row["candidate"], early_best["candidate"], row["baseline"]) < 0:
                        early_best = row
                late_best = late[0]
                for row in late[1:]:
                    if _rank_cmp(row["candidate"], late_best["candidate"], row["baseline"]) < 0:
                        late_best = row
                if _rank_cmp(late_best["candidate"], early_best["candidate"], late_best["baseline"]) < 0:
                    effective_limit = EXTENDED_EPOCHS
                    temp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                    torch.save({"fitIdentity": identity, "epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "best": best, "bestModel": best_state,
                        "history": history, "patience": patience, "effectiveLimit": effective_limit,
                        "updates": updates, "elapsedSeconds": elapsed_prior + time.monotonic()-started,
                        "rngState": _rng_snapshot(torch)}, temp)
                    temp.replace(checkpoint_path)
    if best is None or best_state is None:
        raise RuntimeError("no valid internal validation epoch")
    model.load_state_dict(best_state)
    # A fully completed checkpoint must keep its saved elapsed time so that a
    # phase restart reproduces the existing immutable diagnostics exactly.
    elapsed = float(history[-1]["elapsedSeconds"])
    progress("fit_completed", epoch, int(best["epoch"]))
    return {"model": model, "selectedEpoch": best["epoch"], "best": best,
            "history": history, "nH": nh, "fitIdentity": identity,
            "validationBaseline": best["baseline"],"updates":updates,
            "elapsedSeconds":elapsed}


def fit_fixed(records: Sequence[Mapping[str, Any]], b_rows: Mapping[tuple[dt.date, int], float],
              *, q: float, penalty: float, epochs: int, fit_id: str,
              checkpoint_path: Path, objective: str,
              provenance: Mapping[str, str] | None = None, phase: str = "unknown") -> dict[str, Any]:
    """Fresh full-prefix D or F refit at an already selected epoch count."""
    import numpy as np
    if objective not in ("D", "F") or epochs < 1:
        raise ValueError("fixed fit requires objective D/F and positive selected epoch")
    _, torch, (nn, F) = frozen._torch_runtime()
    records = _with_b(records, b_rows)
    masks = np.asarray([r["targetMask"] for r in records], dtype=bool)
    nh = masks.sum(axis=0).astype(int).tolist()
    if any(n <= 0 for n in nh):
        raise ValueError("full refit has an unsupported horizon")
    frozen._seed_everything(torch, SEED)
    device = frozen._device(torch)
    model = frozen.build_model(CONTEXT_DAYS, REPAIR, torch=torch, nn=nn, F=F)
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=frozen.INITIAL_LR,
                                  weight_decay=frozen.INITIAL_WEIGHT_DECAY)
    scale_obj = 1.0 if objective == "D" else 2.0
    ident = sha256(canonical({"fitId": fit_id, "objective": objective, "epochs": epochs,
        "q": q, "lambda": penalty, "nH": nh, "seed": SEED, "optimizerScale": scale_obj,
        "provenance": dict(provenance or {}),
        "records": _json_safe([[r["origin"].isoformat(), r["targets"], r["targetMask"], r["B"]] for r in records])}))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    first_epoch = 1
    updates=0; elapsed_prior=0.0; elapsed_saved=0.0; started=time.monotonic()
    def progress(status: str, current_epoch: int) -> None:
        _atomic_json(OUT / "training-progress.json", {"schemaVersion": 1, "phase": phase,
            "lambda": penalty, "fitId": fit_id, "objective": objective, "fullRefit": True,
            "status": status, "epoch": current_epoch, "bestEpoch": epochs,
            "updates": updates, "elapsedSeconds": elapsed_prior + time.monotonic() - started})
    progress("fit_started", first_epoch - 1)
    if checkpoint_path.is_file():
        try:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(checkpoint_path, map_location="cpu")
        if saved.get("fitIdentity") != ident:
            raise ValueError(f"checkpoint identity mismatch for {fit_id}")
        first_epoch = int(saved["epoch"]) + 1
        updates=int(saved.get("updates",0)); elapsed_prior=float(saved.get("elapsedSeconds",0.0))
        elapsed_saved=elapsed_prior
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if first_epoch > epochs:
            progress("fit_completed", epochs)
            return {"model": model, "fitIdentity": ident, "selectedEpoch": epochs,
                    "nH": nh, "q": q, "lambda": penalty, "objective": objective,
                    "updates":updates,"elapsedSeconds":elapsed_prior}
        _rng_restore(saved["rngState"], torch)
    for epoch in range(first_epoch, epochs+1):
        model.train()
        order = list(range(len(records)))
        random.Random(SEED + epoch).shuffle(order)
        for start in range(0, len(order), frozen.BATCH_SIZE):
            batch = [records[i] for i in order[start:start+frozen.BATCH_SIZE]]
            z, obs, active, roles, meta, cal, y, mask = frozen._as_tensor_batch(
                batch, [r["targets"] for r in batch], [r["targetMask"] for r in batch], device, torch)
            delta = model(z, obs, active, roles, meta, cal).to(device="cpu").to(dtype=torch.float64)
            p0 = torch.as_tensor([r["window"]["p0"] for r in batch], dtype=torch.float64)
            sigma = torch.as_tensor([r["window"]["representativeSigma"] for r in batch], dtype=torch.float64)
            pred = torch.exp(torch.log(p0).unsqueeze(1)+sigma.unsqueeze(1)*delta)
            b = torch.as_tensor([r["B"] for r in batch], dtype=torch.float64)
            use = .5*b + .5*pred if objective == "F" else pred
            pieces = []
            for h in range(10):
                valid = mask[:, h]
                if bool(valid.any()):
                    err = (use[valid, h]-y[valid, h]).abs()
                    pieces.append(((1/y[valid, h]+penalty/q)*err).sum()/nh[h])
            loss = torch.stack(pieces).sum() * (len(records)/len(batch)/10) * scale_obj
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite {objective} loss in {fit_id} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), frozen.CLIP_NORM)
            optimizer.step()
            updates += 1
        elapsed_saved=elapsed_prior+time.monotonic()-started
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save({"fitIdentity": ident, "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "objective": objective,
                    "selectedEpoch": epochs, "nH": nh, "q": q, "lambda": penalty,
                    "updates": updates, "elapsedSeconds":elapsed_saved,
                    "rngState": _rng_snapshot(torch)}, tmp)
        tmp.replace(checkpoint_path)
        progress("epoch_completed", epoch)
    progress("fit_completed", epochs)
    return {"model": model, "fitIdentity": ident, "selectedEpoch": epochs,
            "nH": nh, "q": q, "lambda": penalty, "objective": objective,
            "updates":updates,"elapsedSeconds":elapsed_saved}


def fit_month(split: Mapping[str, Any], *, b_rows: Mapping[tuple[dt.date, int], float],
              penalty: float, month_id: str, checkpoint_dir: Path = OUT / "checkpoints",
              provenance: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run inner F selection, then paired full-prefix D and F refits."""
    b_train = _with_b(split["innerTrain"], b_rows)
    b_val = _with_b(split["innerValidation"], b_rows)
    # Establish B's validation metrics from the exact paired rows.
    b_pred = [[float(v) for v in r["B"]] for r in b_val]
    baseline_metrics = _metrics(b_val, b_pred, b_pred)["baseline"]
    inner = train_one(split["innerTrain"], split["innerValidation"], b_train=b_rows,
        b_validation=b_rows, q=split["qInner"], penalty=penalty,
        fit_id=f"{month_id}-inner-F-L{penalty:g}", checkpoint_path=checkpoint_dir/f"{month_id}-inner-F-L{penalty:g}.pt",
        optimizer_scale=2.0, baseline_validation=baseline_metrics, provenance=provenance,
        phase=month_id.split("-", 1)[0], objective="inner-F", penalty_value=penalty)
    epochs = int(inner["selectedEpoch"])
    d = fit_fixed(split["fullTrain"], b_rows, q=split["qFull"], penalty=penalty,
        epochs=epochs, fit_id=f"{month_id}-full-D-L{penalty:g}",
        checkpoint_path=checkpoint_dir/f"{month_id}-full-D-L{penalty:g}.pt", objective="D", provenance=provenance,
        phase=month_id.split("-",1)[0])
    f = fit_fixed(split["fullTrain"], b_rows, q=split["qFull"], penalty=penalty,
        epochs=epochs, fit_id=f"{month_id}-full-F-L{penalty:g}",
        checkpoint_path=checkpoint_dir/f"{month_id}-full-F-L{penalty:g}.pt", objective="F", provenance=provenance,
        phase=month_id.split("-",1)[0])
    return {"monthId": month_id, "lambda": penalty, "selectedEpoch": epochs,
            "inner": inner, "D": d, "F": f}


PHASE_DATES = {
    "D25": (dt.date(2025, 4, 1), dt.date(2025, 6, 20)),
    "E25": (dt.date(2025, 7, 11), dt.date(2025, 9, 27)),
    "V26": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
    "T26": (dt.date(2026, 7, 11), dt.date(2026, 9, 27)),
}
PHASE_FIRST_ORIGIN = {
    "D25": dt.date(2025, 4, 1), "E25": dt.date(2025, 7, 1),
    "V26": dt.date(2026, 4, 1), "T26": dt.date(2026, 7, 1),
}


def _phase_baseline_rows(phase: str) -> list[dict[str, Any]]:
    start, end = PHASE_DATES[phase]
    doc = json.loads((OUT / "baseline-cache.json").read_text(encoding="utf-8"))
    rows = []
    for row in doc.get("rows", []):
        if row.get("product") != PRODUCT:
            continue
        origin, target = dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"])
        origin_start = PHASE_FIRST_ORIGIN[phase]
        if origin < origin_start or not start <= target <= end:
            continue
        if row.get("BSourceStatus") not in ("issued", "fallback"):
            raise ValueError(f"eligible {phase} onion key lacks causal B: {row['origin']}/{row['target']}/h{row['horizon']}")
        rows.append({**row, "originDate": origin, "targetDate": target})
    if not rows:
        raise ValueError(f"no onion baseline rows for {phase}")
    return sorted(rows, key=lambda r: (r["originDate"], r["horizon"]))


def _predict_records(model: Any, records: Sequence[Mapping[str, Any]], torch: Any) -> list[list[float]]:
    return frozen._predict_records(model, records, frozen._device(torch), torch)


def run_phase(phase: str, penalty: float = LAMBDA_BASE) -> dict[str, Any]:
    """Execute an authorized phase from its pinned answer-free B cache."""
    import numpy as np
    config = validate_config()
    preflight()
    runtime = frozen._runtime_identity(include_device=True)
    for field in ("python", "platform", "machine", "implementation", "torch", "mpsAvailable", "device", "dtype", "threadCount"):
        expected = config["runtime"].get(field)
        if expected is not None and runtime.get(field) != expected:
            raise RuntimeError(f"runtime differs from frozen prior environment for {field}: {runtime.get(field)!r} != {expected!r}")
    if phase not in PHASE_DATES or penalty not in (LAMBDA_BASE, LAMBDA_REPAIR):
        raise ValueError("phase/lambda outside the frozen schedule")
    if phase == "D25" and penalty == LAMBDA_REPAIR:
        base_score = json.loads((OUT/"score-D25-L1.json").read_text(encoding="utf-8"))
        if (base_score.get("configSha256") != config["configSha256"]
                or not base_score.get("pairedPriceCoverage")):
            raise ValueError("lambda3 requires a config-bound paired D25 base score")
        if file_sha256(ROOT/base_score["forecastPath"]) != base_score.get("forecastFileSha256"):
            raise ValueError("lambda3 base-score forecast binding is stale")
        if base_score["metrics"]["H_F_MAE"] <= base_score["metrics"]["B_MAE"]:
            raise ValueError("lambda3 is unauthorized: base D25 H_F MAE does not exceed B")
    elif phase != "D25":
        selection = json.loads((OUT/"D25-selection.json").read_text(encoding="utf-8"))
        if selection.get("configSha256") != config["configSha256"]:
            raise ValueError("D25 selection belongs to a different resolved config")
        for level, digest in selection.get("inputScoreSha256", {}).items():
            score_path = OUT/f"score-D25-{level}.json"
            score_doc = json.loads(score_path.read_text(encoding="utf-8"))
            if file_sha256(score_path) != digest:
                raise ValueError(f"D25 selection source score changed: L{level}")
            if file_sha256(ROOT/score_doc["forecastPath"]) != score_doc.get("forecastFileSha256"):
                raise ValueError(f"D25 selection source forecast changed: L{level}")
        if penalty != selection["selectedLambda"]:
            raise ValueError("later phase lambda differs from frozen D25 selection")
    cache_path = OUT / "baseline-cache.json"
    manifest_path = OUT / "baseline-cache-manifest.json"
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("baseline cache and manifest must be completed first")
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(cache_path) != cache_manifest.get("cacheFileSha256"):
        raise ValueError("baseline cache file hash differs from its manifest")
    source, source_doc = data.load_source_cache()
    representatives = data.representative_series(frozen._read_json(frozen.PRICE_PATH))[PRODUCT]
    channels = source[PRODUCT]
    rows = _phase_baseline_rows(phase)
    b_rows = load_baseline_rows()
    grouped: dict[dt.date, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["originDate"].replace(day=1), []).append(row)
    selected = {}
    for month, month_rows in sorted(grouped.items()):
        fit_origin = min(r["originDate"] for r in month_rows)
        if fit_origin.day != 1:
            raise ValueError(f"first eligible origin is not month-first: {phase}/{fit_origin}")
        split = month_split(fit_origin, representatives, channels)
        month_id = f"{phase}-{month:%Y%m}-onion-M"
        fitted = fit_month(split, b_rows=b_rows, penalty=penalty, month_id=month_id,
            provenance={"configSha256": config["configSha256"],
                        "baselineCacheManifestSha256": file_sha256(manifest_path)})
        train_scored = _with_b(split["fullTrain"], b_rows)
        torch = frozen._torch_runtime()[1]
        train_d = _predict_records(fitted["D"]["model"], train_scored, torch)
        train_f = _predict_records(fitted["F"]["model"], train_scored, torch)
        inner_val = _with_b(split["innerValidation"], b_rows)
        inner_f = _predict_records(fitted["inner"]["model"], inner_val, torch)
        month_diagnostics = {"fitMonth": month.isoformat(), "fitOrigin": fit_origin.isoformat(),
            "phase": phase, "fitKey": f"{phase}-{month:%Y%m}-L{penalty:g}",
            "innerBoundary": split["innerBoundary"].isoformat(), "qInner": split["qInner"],
            "qFull": split["qFull"], "lambda": penalty, "selectedEpoch": fitted["selectedEpoch"],
            "innerTrainKeyDigest": sha256(canonical([[r["origin"].isoformat(), r["targetDays"], r["targetMask"]]
                for r in split["innerTrain"]])),
            "fullTrainKeyDigest": sha256(canonical([[r["origin"].isoformat(), r["targetDays"], r["targetMask"]]
                for r in split["fullTrain"]])),
            "qInnerSupport": label_support(split["innerTrain"]),
            "qFullSupport": label_support(split["fullTrain"]),
            "trainRows": len(train_scored), "trainKeyDigest": sha256(canonical(_json_safe([
                [r["origin"].isoformat(), r["targetDays"], r["targets"], r["targetMask"], r["B"]]
                for r in train_scored]))),
            "nH": fitted["D"]["nH"], "trainingCrossEvaluation": cross_loss_diagnostics(
                train_scored, train_d, train_f, split["qFull"], penalty),
            "representationFloorTraining": representation_floor(train_scored, train_f),
            "representationFloorInnerValidation": representation_floor(inner_val, inner_f),
            "innerValidationBest": fitted["inner"]["best"],
            "innerEpochHistory": fitted["inner"]["history"],
            "fitStats": {name:{"fitIdentity":fitted[name]["fitIdentity"],
                "selectedEpoch":fitted[name]["selectedEpoch"],"updates":fitted[name]["updates"],
                "elapsedSeconds":fitted[name]["elapsedSeconds"]}
                for name in ("inner","D","F")},
            "selectedChannelIds": [c.get("channelId") for c in split["selectedChannels"]],
            "selectedChannelMetadata": [{k: v for k, v in c.items() if k in
                ("channelId", "channelType", "source", "seriesKey", "product", "family")}
                for c in split["selectedChannels"]]}
        diag_path = OUT / "training-diagnostics.json"
        diagnostics_doc = {"schemaVersion": 1, "configSha256": config["configSha256"],
                           "baselineCacheManifestSha256": file_sha256(manifest_path), "fits": []}
        if diag_path.exists():
            diagnostics_doc = json.loads(diag_path.read_text(encoding="utf-8"))
            if diagnostics_doc.get("configSha256") != config["configSha256"]:
                raise ValueError("training diagnostics belong to a different frozen config")
        existing_fit = {r.get("fitKey",r["fitMonth"]): r for r in diagnostics_doc["fits"]}
        safe_diag = _json_safe(month_diagnostics)
        diag_key=month_diagnostics["fitKey"]
        prior_diag = existing_fit.get(diag_key, {})
        prior_base = {k:v for k,v in prior_diag.items()
            if k not in ("modelArtifacts", "inputContract", "replayInputDigest")}
        if diag_key in existing_fit and prior_base != safe_diag:
            raise ValueError(f"stored diagnostics differ on restart for {month_id}")
        existing_fit[diag_key] = safe_diag
        diagnostics_doc["fits"] = [existing_fit[k] for k in sorted(existing_fit)]
        _atomic_json(diag_path, diagnostics_doc)
        # Save separate immutable model snapshots and compare predictions after
        # a genuine native-device reload before writing answer-free forecasts.
        infer = []
        for origin in sorted({r["originDate"] for r in month_rows}):
            window = data.build_input_window(PRODUCT, origin, CONTEXT_DAYS, INPUT_TYPE,
                representatives, channels, split["selectedChannels"])
            if window is None:
                raise ValueError(f"forecast origin has no causal M-input window: {origin}")
            target_days = [origin + dt.timedelta(days=h) for h in HORIZONS]
            infer.append({"product": PRODUCT, "origin": origin, "window": window,
                "calendar": frozen.calendar_features(target_days), "targets": [0.0]*10,
                "targetMask": [False]*10, "targetDays": [d.isoformat() for d in target_days]})
        fit_paths = {}
        predictions = {}
        for name in ("D", "F"):
            model = fitted[name]["model"]
            original = _predict_records(model, infer, torch)
            path = OUT / "weights" / f"{month_id}-{name}-L{penalty:g}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), tmp)
            if path.exists() and file_sha256(path) != file_sha256(tmp):
                tmp.unlink(); raise FileExistsError(f"refusing to replace immutable model {path}")
            if path.exists(): tmp.unlink()
            else: tmp.replace(path)
            fresh = frozen.build_model(CONTEXT_DAYS, REPAIR, torch=torch,
                nn=torch.nn, F=torch.nn.functional)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            fresh.load_state_dict(payload)
            fresh.to(device=frozen._device(torch), dtype=torch.float32)
            reloaded = _predict_records(fresh, infer, torch)
            if not np.allclose(np.asarray(original), np.asarray(reloaded), rtol=1e-6, atol=1e-6, equal_nan=True):
                raise ValueError(f"saved/reloaded prediction mismatch for {month_id}/{name}")
            fit_paths[name] = (path, file_sha256(path))
            predictions[name] = reloaded
        safe_diag["inputContract"] = {"contextDays": CONTEXT_DAYS, "inputType": INPUT_TYPE,
            "selectedChannelIds": [c.get("channelId") for c in split["selectedChannels"]],
            "calendarShape": [len(infer), len(HORIZONS), 4],
            "windowColumnCount": len(infer[0]["window"]["columns"])}
        safe_diag["replayInputDigest"] = sha256(canonical([{
            "origin": r["origin"].isoformat(), "window": r["window"], "calendar": r["calendar"]}
            for r in infer]))
        safe_diag["modelArtifacts"] = {name: {"weightPath": str(fit_paths[name][0].relative_to(ROOT)),
            "weightSha256": fit_paths[name][1], "checkpointPath": str(
                (OUT/"checkpoints"/f"{month_id}-full-{name}-L{penalty:g}.pt").relative_to(ROOT)),
            "checkpointSha256": file_sha256(OUT/"checkpoints"/f"{month_id}-full-{name}-L{penalty:g}.pt"),
            "fitIdentity": fitted[name]["fitIdentity"], "replayAgreement": True}
            for name in ("D", "F")}
        existing_fit[diag_key] = safe_diag
        diagnostics_doc["fits"] = [existing_fit[k] for k in sorted(existing_fit)]
        _atomic_json(diag_path, diagnostics_doc)
        selected[month.isoformat()] = {"fit": fitted, "infer": infer,
            "predictions": predictions, "fitPaths": fit_paths, "split": split,
            "monthRows": month_rows}
    out_rows = []
    for month, part in selected.items():
        for i, inf in enumerate(part["infer"]):
            origin = inf["origin"]
            for row in part["monthRows"]:
                if row["originDate"] != origin:
                    continue
                h = int(row["horizon"])-1
                nd = float(part["predictions"]["D"][i][h])
                nf = float(part["predictions"]["F"][i][h])
                b = float(row["B"])
                if (math.isfinite(nd) and nd <= 0) or (math.isfinite(nf) and nf <= 0):
                    raise ValueError("finite nonpositive restored neural price indicates numeric underflow/integrity failure")
                good_d = math.isfinite(nd)
                good_f = math.isfinite(nf)
                hd = .5*b+.5*nd if good_d else b
                hf = .5*b+.5*nf if good_f else b
                out_rows.append({"product": PRODUCT, "phase": phase,
                    "origin": origin.isoformat(), "target": row["targetDate"].isoformat(),
                    "horizon": int(row["horizon"]), "B": b,
                    "BSourceStatus": row["BSourceStatus"],
                    "ND": nd if good_d else None, "NF": nf if good_f else None,
                    "HD": hd, "HF": hf,
                    "statusD": "issued" if good_d else "technical_fallback",
                    "statusF": "issued" if good_f else "technical_fallback",
                    "statusHD": "issued" if good_d else "technical_fallback",
                    "statusHF": "issued" if good_f else "technical_fallback",
                    "reasonD": None if good_d else "nonfinite_neural_prediction",
                    "reasonF": None if good_f else "nonfinite_neural_prediction",
                    "reasonHD": None if good_d else "nonfinite_neural_prediction",
                    "reasonHF": None if good_f else "nonfinite_neural_prediction",
                    "fitIdD": f"{phase}-{dt.date.fromisoformat(month):%Y%m}-onion-M-full-D-L{penalty:g}",
                    "fitIdF": f"{phase}-{dt.date.fromisoformat(month):%Y%m}-onion-M-full-F-L{penalty:g}",
                    "fitOrigin": part["split"]["fitOrigin"].isoformat(),
                    "innerBoundary": part["split"]["innerBoundary"].isoformat(),
                    "cutoff": (origin-dt.timedelta(days=1)).isoformat(),
                    "selectedEpoch": part["fit"]["selectedEpoch"], "lambda": penalty,
                    "q": part["split"]["qFull"],
                    "weightPathD": str(part["fitPaths"]["D"][0].relative_to(ROOT)),
                    "weightPathF": str(part["fitPaths"]["F"][0].relative_to(ROOT)),
                    "weightSha256D": part["fitPaths"]["D"][1],
                    "weightSha256F": part["fitPaths"]["F"][1],
                    "replayAgreement": True})
    out_rows.sort(key=lambda r: (r["product"], r["origin"], r["horizon"]))
    keys = [(r["product"], r["origin"], r["target"], r["horizon"]) for r in out_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate phase forecast key")
    answer_free = {"schemaVersion": 1, "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
        "phase": phase, "targetValuesAttached": False,
        "configSha256": config["configSha256"],
        "d25SelectionSha256": file_sha256(OUT/"D25-selection.json") if phase != "D25" else None,
        "baselineCacheManifestSha256": file_sha256(manifest_path), "rows": out_rows}
    forecast_path = OUT / f"forecast-{phase}-L{penalty:g}-targetfree.json"
    if forecast_path.exists():
        old = json.loads(forecast_path.read_text(encoding="utf-8"))
        if old != answer_free:
            raise FileExistsError(f"refusing to replace existing immutable forecast: {forecast_path}")
    else:
        temp = forecast_path.with_suffix(".json.tmp")
        temp.write_bytes(canonical(answer_free)); temp.replace(forecast_path)
    manifest_path = OUT / "forecast-manifest.json"
    manifest = {"schemaVersion": 1, "configSha256": config["configSha256"], "forecasts": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configSha256") != config["configSha256"]:
            raise ValueError("forecast manifest belongs to a different resolved config")
    entry = {"phase": phase, "lambda": penalty,
        "path": str(forecast_path.relative_to(ROOT)), "fileSha256": file_sha256(forecast_path),
        "rows": len(out_rows), "keyDigest": sha256(canonical(keys)),
        "targetValuesAttached": False, "baselineCacheManifestSha256": file_sha256(OUT/"baseline-cache-manifest.json")}
    existing = [r for r in manifest["forecasts"] if (r["phase"],r["lambda"]) != (phase,penalty)]
    existing.append(entry); manifest["forecasts"] = sorted(existing,key=lambda r:(r["phase"],r["lambda"]))
    _atomic_json(manifest_path, manifest)
    ledger_path = OUT / "training-ledger.json"
    ledger = {"schemaVersion":1,"configSha256":config["configSha256"],"completedPhases":[]}
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("configSha256") != config["configSha256"]:
            raise ValueError("training ledger belongs to a different frozen config")
    fit_summaries=[]
    elapsed_cost=0.0
    for month_key,part in sorted(selected.items()):
        month_id=f"{phase}-{dt.date.fromisoformat(month_key):%Y%m}-onion-M"
        for role in ("inner","D","F"):
            fit=part["fit"][role]
            fit_id=(f"{month_id}-inner-F-L{penalty:g}" if role=="inner"
                    else f"{month_id}-full-{role}-L{penalty:g}")
            checkpoint=OUT/"checkpoints"/(fit_id+".pt")
            executed=(len(fit["history"]) if role=="inner" else int(fit["selectedEpoch"]))
            elapsed=float(fit["elapsedSeconds"]); elapsed_cost+=elapsed
            fit_summaries.append({"fitId":fit_id,"fitIdentity":fit["fitIdentity"],
                "role":role,"objective":"F" if role=="inner" else role,
                "selectedEpoch":int(fit["selectedEpoch"]),"executedEpochs":executed,
                "updates":int(fit["updates"]),"elapsedSeconds":elapsed,
                "checkpointPath":str(checkpoint.relative_to(ROOT)),
                "checkpointSha256":file_sha256(checkpoint)})
    phase_entry={"phase":phase,"lambda":penalty,"status":"completed",
        "rows":len(out_rows),"forecastPath":str(forecast_path.relative_to(ROOT)),
        "forecastSha256":file_sha256(forecast_path),
        "fitIds":[f"{phase}-{dt.date.fromisoformat(m):%Y%m}-onion-M" for m in selected],
        "fitSummaries":fit_summaries,"updates":sum(r["updates"] for r in fit_summaries),
        "elapsedSeconds":elapsed_cost,
        "trainingDiagnosticsPath":str((OUT/"training-diagnostics.json").relative_to(ROOT)),
        "trainingDiagnosticsSnapshotSha256":file_sha256(OUT/"training-diagnostics.json")}
    done=[r for r in ledger["completedPhases"] if (r["phase"],r["lambda"])!=(phase,penalty)]
    done.append(phase_entry); ledger["completedPhases"]=sorted(done,key=lambda r:(r["phase"],r["lambda"]))
    _atomic_json(ledger_path, ledger)
    return {"phase": phase, "rows": len(out_rows), "forecastPath": str(forecast_path),
            "forecastFileSha256": file_sha256(forecast_path),
            "baselineManifestSha256": file_sha256(OUT/"baseline-cache-manifest.json")}


def score_phase(phase: str, penalty: float) -> dict[str, Any]:
    """Join saved answer-free forecasts to the pinned price snapshot after saving."""
    config = validate_config()
    preflight()
    forecast_path = OUT / f"forecast-{phase}-L{penalty:g}-targetfree.json"
    doc = json.loads(forecast_path.read_text(encoding="utf-8"))
    if (doc.get("targetValuesAttached") is not False or doc.get("phase") != phase
            or doc.get("configSha256") != config["configSha256"]
            or doc.get("baselineCacheManifestSha256") != file_sha256(OUT/"baseline-cache-manifest.json")):
        raise ValueError("source forecast has stale phase/config/baseline bindings")
    producer_manifest = json.loads((OUT/"forecast-manifest.json").read_text(encoding="utf-8"))
    if producer_manifest.get("configSha256") != config["configSha256"]:
        raise ValueError("forecast producer manifest belongs to a different configuration")
    entries = [r for r in producer_manifest.get("forecasts", [])
               if r.get("phase") == phase and float(r.get("lambda", -1)) == penalty]
    if (len(entries) != 1 or entries[0].get("fileSha256") != file_sha256(forecast_path)
            or entries[0].get("targetValuesAttached") is not False):
        raise ValueError("forecast source file is not bound by exactly one answer-free producer entry")
    prices = frozen._read_json(frozen.PRICE_PATH)
    actual = data.representative_series(prices)[PRODUCT]
    scored = []; missing=[]
    for row in doc["rows"]:
        y = actual.get(dt.date.fromisoformat(row["target"]))
        if y is None:
            missing.append({"origin":row["origin"],"target":row["target"],"horizon":row["horizon"],
                            "targetStatus":"not_in_pinned_representative_snapshot"})
            continue
        scored.append({**row, "actual": float(y)})
    if not scored:
        raise ValueError("no forecast rows could be joined to pinned target prices")
    horizons = []
    for h in HORIZONS:
        subset = [r for r in scored if r["horizon"] == h]
        if not subset:
            raise ValueError(f"no paired score keys for horizon {h}")
        err_f = [abs(r["HF"]-r["actual"]) for r in subset]
        err_b = [abs(r["B"]-r["actual"]) for r in subset]
        err_d = [abs((.5*r["B"]+.5*r["ND"] if r["ND"] is not None else r["B"])-r["actual"]) for r in subset]
        horizons.append({"horizon": h, "n": len(subset),
            "H_F_MAE": statistics.mean(err_f), "H_F_MAPE": statistics.mean(e/r["actual"] for e,r in zip(err_f,subset)),
            "B_MAE": statistics.mean(err_b), "B_MAPE": statistics.mean(e/r["actual"] for e,r in zip(err_b,subset)),
            "H_D_MAE": statistics.mean(err_d), "H_D_MAPE": statistics.mean(e/r["actual"] for e,r in zip(err_d,subset))})
    mean = lambda key: statistics.mean(r[key] for r in horizons)
    score = {"schemaVersion": 1, "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
        "configSha256": json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["configSha256"],
        "phase": phase, "lambda": penalty,
        "forecastPath": str(forecast_path.relative_to(ROOT)),
        "forecastFileSha256": file_sha256(forecast_path),
        "priceSnapshotSha256": file_sha256(frozen.PRICE_PATH), "pairedRows": len(scored),
        "completeTargetCoverage": not missing,
        "pairedPriceCoverage": all(any(r["horizon"] == h for r in scored) for h in HORIZONS),
        "targetJoinCoverage": {"joinedRows": len(scored), "forecastRows": len(doc["rows"]),
                               "missingTargetRows": len(missing)}, "missingTargets": missing,
        "uniqueForecastKeys": len({(r['origin'],r['target'],r['horizon']) for r in scored}),
        "metrics": {k: mean(k) for k in ("H_F_MAE", "H_F_MAPE", "B_MAE", "B_MAPE", "H_D_MAE", "H_D_MAPE")},
        "horizons": horizons}
    score_path = OUT / f"score-{phase}-L{penalty:g}.json"
    if score_path.exists():
        if json.loads(score_path.read_text(encoding="utf-8")) != score:
            raise FileExistsError(f"refusing to replace immutable score {score_path}")
    else:
        tmp=score_path.with_suffix(".json.tmp"); tmp.write_bytes(canonical(score)); tmp.replace(score_path)
    return score


def select_d25_lambda() -> dict[str, Any]:
    base = json.loads((OUT / "score-D25-L1.json").read_text(encoding="utf-8"))
    config = validate_config()
    if base.get("configSha256") != config["configSha256"] or not base.get("pairedPriceCoverage"):
        raise ValueError("D25 lambda selection requires a config-bound shared paired price support")
    if file_sha256(ROOT/base["forecastPath"]) != base.get("forecastFileSha256"):
        raise ValueError("D25 lambda=1 score does not bind its forecast source")
    s1 = base["metrics"]
    need_repair = s1["H_F_MAE"] > s1["B_MAE"]
    candidates = [base]
    if need_repair:
        candidates.append(json.loads((OUT / "score-D25-L3.json").read_text(encoding="utf-8")))
        if (candidates[1].get("configSha256") != config["configSha256"]
                or not candidates[1].get("pairedPriceCoverage")):
            raise ValueError("lambda3 selection requires config-bound paired price support")
        if file_sha256(ROOT/candidates[1]["forecastPath"]) != candidates[1].get("forecastFileSha256"):
            raise ValueError("D25 lambda=3 score does not bind its forecast source")
    eligible = [s for s in candidates if s["metrics"]["H_F_MAPE"] < s["metrics"]["B_MAPE"]
                and s["metrics"]["H_F_MAE"] <= s["metrics"]["B_MAE"]]
    pool = eligible or candidates
    if eligible:
        # Eligible choices use exact raw MAPE ordering; exact MAPE ties retain
        # the base lambda, as in the shared policy scorer.
        chosen = min(pool, key=lambda row: (row["metrics"]["H_F_MAPE"], int(row["lambda"])))
    else:
        chosen = pool[0]
        for candidate in pool[1:]:
            cm, bm = candidate["metrics"], chosen["metrics"]
            x = max(_ratio(cm["H_F_MAPE"], cm["B_MAPE"]), _ratio(cm["H_F_MAE"], cm["B_MAE"]))
            y = max(_ratio(bm["H_F_MAPE"], bm["B_MAPE"]), _ratio(bm["H_F_MAE"], bm["B_MAE"]))
            tied=(x==y) if (math.isinf(x) or math.isinf(y)) else abs(x-y)<=TIE_TOL
            better=(not math.isinf(x) and math.isinf(y)) or (not math.isinf(x) and not math.isinf(y) and x<y-TIE_TOL)
            if better or (tied and int(candidate["lambda"]) < int(chosen["lambda"])):
                chosen = candidate
    result = {"schemaVersion":1,"configSha256":config["configSha256"],"phase":"D25","baseLambda1":base,
        "lambda3Required":need_repair,"lambda3": candidates[1] if len(candidates)>1 else None,
        "selectedLambda":chosen["lambda"],"selectionBasis":"eligible_minimum_MAPE_then_lambda1" if eligible else "minimum_max_relative_ratio_then_lambda1",
        "inputScoreSha256":{"L1":file_sha256(OUT/"score-D25-L1.json"),
            **({"L3":file_sha256(OUT/"score-D25-L3.json")} if len(candidates)>1 else {})},
        "downstreamLambda":chosen["lambda"],"noFurtherAdjustment":True}
    path=OUT/"D25-selection.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != result:
        raise FileExistsError("refusing to replace frozen D25 selection")
    if not path.exists():
        tmp=path.with_suffix(".json.tmp");tmp.write_bytes(canonical(result));tmp.replace(path)
    return result


def preflight() -> dict[str, Any]:
    """Read-only integrity checks; never trains or attaches target values."""
    config = validate_config()
    manifest_path, cache_path = OUT/"baseline-cache-manifest.json", OUT/"baseline-cache.json"
    if not manifest_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("complete baseline cache and manifest first")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned=dict(manifest); mh=unsigned.pop("manifestSha256",None)
    if mh != sha256(canonical(unsigned)) or manifest.get("complete") is not True:
        raise ValueError("baseline manifest is incomplete or fails its self-hash")
    if (manifest.get("experiment") != "FINAL_ENSEMBLE_LOSS_ONION_V2"
            or manifest.get("phase") != "answer_free"
            or manifest.get("targetValuesAttached") is not False):
        raise ValueError("baseline manifest has the wrong experiment/answer state")
    expected_baseline_code = config["producerCodeSha256"]["agri_data/final_ensemble_onion_baseline.py"]
    if manifest.get("sourceCodeSha256", {}).get("baselineCacheModule") != expected_baseline_code:
        raise ValueError("baseline cache was produced by a different reviewed helper")
    if file_sha256(cache_path) != manifest.get("cacheFileSha256"):
        raise ValueError("baseline cache bytes differ from manifest")
    cache=json.loads(cache_path.read_text(encoding="utf-8"))
    if (cache.get("experiment") != "FINAL_ENSEMBLE_LOSS_ONION_V2"
            or cache.get("phase") != "answer_free" or cache.get("targetValuesAttached") is not False):
        raise ValueError("baseline cache has the wrong experiment/answer state")
    if sha256(canonical(cache.get("rows",[]))) != manifest.get("cacheRowsSha256"):
        raise ValueError("baseline row digest differs from its manifest")
    keys=[]; onion_keys=set()
    for row in cache.get("rows",[]):
        key=(row["product"],row["origin"],row["target"],int(row["horizon"]))
        if key in keys:
            raise ValueError(f"duplicate B cache key {key}")
        keys.append(key)
        if row["product"]==PRODUCT:
            onion_keys.add((dt.date.fromisoformat(row["origin"]),int(row["horizon"])))
    if len(keys) != len(cache.get("rows",[])):
        raise ValueError("baseline cache key uniqueness failed")
    # All 2021+ historical daily-origin rows must be explicit, including genuine
    # not-issued outcomes; this prevents silent training-support shrinkage.
    from .final_ensemble_onion_baseline import expected_origin_keys
    expected=expected_origin_keys()
    if not expected.issubset({(p,o,t,h) for p,o,t,h in keys}):
        raise ValueError("baseline cache omits expected historical onion origin/horizon keys")
    valid=[]
    for phase in ("V26","T26"):
        valid.extend(_phase_baseline_rows(phase))
    distinct_months={r["originDate"].replace(day=1) for r in valid}
    if len(distinct_months) != 6:
        raise ValueError(f"V/T baseline origins do not form the frozen six-month fit schedule: {distinct_months}")
    return {"status":"preflight_only","configSha256":config["configSha256"],
        "baselineManifestSha256":file_sha256(manifest_path),"baselineCacheSha256":file_sha256(cache_path),
        "baselineRows":len(keys),"historicalOnionKeys":len(expected),
        "eligibleV26Keys":sum(1 for r in valid if r["originDate"] < dt.date(2026,7,1)),
        "eligibleT26Keys":sum(1 for r in valid if r["originDate"] >= dt.date(2026,7,1)),
        "vtFitMonths":[d.isoformat() for d in sorted(distinct_months)],"modelFitsPerformed":0,
        "targetValuesAttached":False}


def export_forecasts() -> dict[str, Any]:
    """Join frozen phase forecast keys and create one host-side Parquet artifact."""
    import pandas as pd
    config=validate_config()
    selection=json.loads((OUT/"D25-selection.json").read_text(encoding="utf-8"))
    if selection.get("configSha256") != config["configSha256"]:
        raise ValueError("D25 selection belongs to a different frozen config")
    for level, digest in selection.get("inputScoreSha256", {}).items():
        score_path=OUT/f"score-D25-{level}.json"
        score_doc=json.loads(score_path.read_text(encoding="utf-8"))
        if file_sha256(score_path)!=digest or file_sha256(ROOT/score_doc["forecastPath"])!=score_doc.get("forecastFileSha256"):
            raise ValueError(f"D25 selection provenance is stale for {level}")
    penalty=float(selection["selectedLambda"])
    cache=json.loads((OUT/"baseline-cache.json").read_text(encoding="utf-8"))
    base={(r["product"],r["origin"],r["target"],int(r["horizon"])):r for r in cache["rows"]}
    rows=[]
    for phase in ("D25","E25","V26","T26"):
        phase_lambda=penalty
        fpath=OUT/f"forecast-{phase}-L{phase_lambda:g}-targetfree.json"
        fdoc=json.loads(fpath.read_text(encoding="utf-8"))
        if fdoc.get("targetValuesAttached") is not False or fdoc.get("configSha256") != config["configSha256"]:
            raise ValueError(f"invalid answer-free source forecast {phase}")
        onion={(r["product"],r["origin"],r["target"],int(r["horizon"])):r for r in fdoc["rows"]}
        if phase in ("D25","E25"):
            keys=sorted(onion)
        else:
            target_start,target_end=PHASE_DATES[phase]
            origin_start=dt.date(2026,4,1) if phase=="V26" else dt.date(2026,7,1)
            keys=sorted(k for k,r in base.items() if dt.date.fromisoformat(k[1])>=origin_start
                and target_start<=dt.date.fromisoformat(k[2])<=target_end
                and r.get("BSourceStatus") in ("issued","fallback"))
            expected_onion={k for k in keys if k[0]==PRODUCT}
            if set(onion) != expected_onion:
                raise ValueError(f"onion V/T forecast key grid differs from the pinned B key grid for {phase}")
        for key in keys:
            b_row=base.get(key)
            if b_row is None or b_row.get("BSourceStatus") not in ("issued","fallback"):
                raise ValueError(f"forecast key lacks issued/fallback baseline {key}")
            b=float(b_row["B"])
            if key[0]==PRODUCT:
                row=onion.get(key)
                if row is None:
                    raise ValueError(f"missing onion neural forecast for expected baseline key {key}")
                rows.append({**row,"phaseLambda":phase_lambda,"finalH":float(row["HF"]),
                    "finalStatus":row["statusF"],"baselineReason":b_row.get("fallbackReason") or b_row.get("sourceReason")})
            else:
                if phase not in ("V26","T26"):
                    continue
                rows.append({"product":key[0],"phase":phase,"origin":key[1],"target":key[2],
                    "horizon":key[3],"B":b,"BSourceStatus":b_row["BSourceStatus"],
                    "ND":None,"NF":None,"HF":b,"statusD":"baseline_copy",
                    "HD":b,"statusF":"baseline_copy","statusHD":"baseline_copy",
                    "statusHF":"baseline_copy","reasonD":None,"reasonF":None,
                    "reasonHD":None,"reasonHF":None,"finalH":b,"finalStatus":"baseline_copy",
                    "phaseLambda":phase_lambda,"fitIdD":None,"fitIdF":None,
                    "fitOrigin":None,"innerBoundary":None,"cutoff":(dt.date.fromisoformat(key[1])-dt.timedelta(days=1)).isoformat(),
                    "selectedEpoch":None,"lambda":None,"q":None,"weightPathD":None,
                    "weightPathF":None,"weightSha256D":None,"weightSha256F":None,
                    "replayAgreement":True,"baselineReason":b_row.get("fallbackReason") or b_row.get("sourceReason")})
    rows.sort(key=lambda r:(r["phase"],r["product"],r["origin"],r["horizon"]))
    keys=[(r["phase"],r["product"],r["origin"],r["target"],int(r["horizon"])) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError("final Parquet has duplicate phase/product/origin/target/horizon keys")
    frame=pd.DataFrame(rows)
    path=OUT/"forecasts.parquet"
    temp=path.with_suffix(".parquet.tmp")
    frame.to_parquet(temp,index=False)
    restored=pd.read_parquet(temp)
    if len(restored)!=len(frame) or restored[["phase","product","origin","target","horizon"]].duplicated().any():
        raise ValueError("Parquet round-trip row/key validation failed")
    verify_fields=("phase","product","origin","target","horizon","B","ND","NF","HD","HF",
        "BSourceStatus","statusD","statusF","statusHD","statusHF","reasonD","reasonF",
        "finalH","finalStatus","phaseLambda","weightSha256D","weightSha256F")
    for field in verify_fields:
        left=frame[field].tolist() if field in frame else [None]*len(frame)
        right=restored[field].tolist() if field in restored else [None]*len(restored)
        for i,(a,b) in enumerate(zip(left,right)):
            if pd.isna(a) and pd.isna(b): continue
            if isinstance(a,(float,int)) and isinstance(b,(float,int)):
                if float(a)!=float(b): raise ValueError(f"Parquet round-trip value mismatch {field} row {i}")
            elif a!=b:
                raise ValueError(f"Parquet round-trip value mismatch {field} row {i}")
    if path.exists():
        if file_sha256(path)!=file_sha256(temp):
            temp.unlink(); raise FileExistsError(f"refusing to replace immutable Parquet output {path}")
        temp.unlink()
    else:
        temp.replace(path)
    manifest={"schemaVersion":1,"configSha256":config["configSha256"],
        "baselineCacheManifestSha256":file_sha256(OUT/"baseline-cache-manifest.json"),
        "D25SelectionSha256":file_sha256(OUT/"D25-selection.json"),
        "forecastSourceFiles":{phase:{"path":f"forecast-{phase}-L{phase_lambda:g}-targetfree.json",
            "sha256":file_sha256(OUT/f"forecast-{phase}-L{phase_lambda:g}-targetfree.json")} for phase in ("D25","E25","V26","T26")},
        "path":str(path.relative_to(ROOT)),"fileSha256":file_sha256(path),"rows":len(frame),
        "keyDigest":sha256(canonical(keys)),"countsByPhaseProduct":{
            f"{p}/{c}":int(n) for (p,c),n in frame.groupby(["phase","product"]).size().items()},
        "roundTripRows":len(restored),"targetValuesAttached":False}
    _atomic_json(OUT/"forecasts-parquet-manifest.json",manifest,immutable=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("freeze",help="freeze resolved config from reviewed code/upstream artifacts")
    sub.add_parser("preflight",help="read-only cache/config integrity checks")
    run=sub.add_parser("run-phase",help="fit and emit answer-free forecast rows")
    run.add_argument("phase",choices=tuple(PHASE_DATES))
    run.add_argument("--lambda",dest="penalty",type=float,choices=(1.0,3.0),default=1.0)
    score=sub.add_parser("score-phase",help="join an already persisted answer-free forecast to pinned prices")
    score.add_argument("phase",choices=tuple(PHASE_DATES))
    score.add_argument("--lambda",dest="penalty",type=float,choices=(1.0,3.0),default=1.0)
    sub.add_parser("select-d25",help="freeze the D25 conditional lambda choice")
    sub.add_parser("export",help="host-side final forecasts.parquet export; does not import/run Torch")
    args=parser.parse_args(argv)
    if args.command=="freeze": result=freeze_config()
    elif args.command=="preflight": result=preflight()
    elif args.command=="run-phase": result=run_phase(args.phase,args.penalty)
    elif args.command=="score-phase": result=score_phase(args.phase,args.penalty)
    elif args.command=="select-d25": result=select_d25_lambda()
    else: result=export_forecasts()
    print(json.dumps(_json_safe(result),ensure_ascii=False,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

