"""Pinned local source extraction and causal channel windows for the direct path model.

This module reads only the already-reviewed local SQLite snapshot.  It does not
make API requests, copy source databases, or inspect future raw market rows from
inside any fit: callers provide a fit cutoff and candidate selection is bounded
by that cutoff.
"""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import dashboard, external_reference_v1 as ext, modeling

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "DIRECT_PRICE_PATH_ATTENTION_V1"
OUT = ROOT / "outputs/experiments/direct-price-path-attention-v1-20261014"
SOURCE_CACHE_PATH = OUT / "source-cache.json"
DB_PATH = ROOT / "data/agri.sqlite"
PRICE_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
PRIOR_CACHE_PATH = ROOT / "outputs/experiments/external-reference-v1-20261012/source-cache.json"
PRIOR_CACHE_SHA256 = "d6d0212a7feecad7914ab41c5655f134df15cee259002497a4684f310b58ce8a"
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
TRAIN_START = dt.date(2014, 1, 1)
SOURCE_END = dt.date(2026, 9, 27)
PRODUCTS = tuple(dashboard.PRODUCTS)[:4]
MARKETS = tuple(dashboard.MARKETS)
CONTEXTS = (84, 168)
MAX_DETAIL_CHANNELS = 8
MIN_DETAIL_DAYS = 30
DETAIL_LOOKBACK_DAYS = 365
DETAIL_MAX_AGE_DAYS = 90
MIN_AUX_WINDOW_DAYS = 5
MAX_AUX_AGE_DAYS = 14
MAX_REP_AGE_DAYS = 14


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _check_self_hash(doc: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(doc)
    got = unsigned.pop(field, None)
    if got != sha256(canonical(unsigned)):
        raise ValueError(f"{label} self-hash mismatch")


def _serialize_full_channels(
    channels: Mapping[str, Mapping[tuple[str, ...], Mapping[dt.date, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    serialized: dict[str, list[dict[str, Any]]] = {}
    for product in PRODUCTS:
        records = []
        for key, series in sorted(channels.get(product, {}).items()):
            observations = []
            for day, value in sorted(series.items()):
                cell = dict(value) if isinstance(value, Mapping) else {"price": value}
                observations.append({"date": day.isoformat(), **cell})
            records.append({"channelId": list(key), "grain": {
                1: "market", 2: "market_subclass", 3: "market_subclass_grade"}.get(len(key), "unknown"),
                "observations": observations})
        serialized[product] = records
    return serialized


def _deserialize_full_channels(raw: Mapping[str, Any]) -> dict[str, dict[tuple[str, ...], dict[dt.date, dict[str, Any]]]]:
    result: dict[str, dict[tuple[str, ...], dict[dt.date, dict[str, Any]]]] = {}
    for product in PRODUCTS:
        by_key: dict[tuple[str, ...], dict[dt.date, dict[str, Any]]] = {}
        entries = raw.get(product, [])
        if not isinstance(entries, list):
            raise ValueError(f"source cache channels for {product} must be a list")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("channelId"), list):
                raise ValueError("malformed channel entry")
            key = tuple(str(v) for v in entry["channelId"])
            if len(key) not in (1, 2, 3) or key in by_key:
                raise ValueError(f"invalid or duplicate channel ID for {product}: {key}")
            series: dict[dt.date, dict[str, Any]] = {}
            for row in entry.get("observations", []):
                if not isinstance(row, Mapping):
                    raise ValueError("malformed channel observation")
                try:
                    day = dt.date.fromisoformat(str(row["date"]))
                    price = float(row["price"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("channel observation needs an ISO date and numeric price") from exc
                if not math.isfinite(price) or price <= 0 or day in series:
                    raise ValueError("channel observation must have a unique date and positive finite price")
                series[day] = {k: v for k, v in row.items() if k != "date"}
            by_key[key] = dict(sorted(series.items()))
        result[product] = by_key
    return result


def extract_source_cache(
    *, db_path: Path = DB_PATH, prices_path: Path = PRICE_PATH,
    prior_cache_path: Path = PRIOR_CACHE_PATH, output_path: Path = SOURCE_CACHE_PATH,
    start: dt.date = TRAIN_START, end: dt.date = SOURCE_END,
) -> dict[str, Any]:
    """Extract all existing market/subclass/grade channels once from the local DB.

    The prior 30-channel cache is only a provenance/checkpoint reference.  This
    extraction regenerates its selected-row digest and target totals once, then
    retains every positive supported channel rather than its selected subset.
    """
    if start != TRAIN_START or end > SOURCE_END or start > end:
        raise ValueError("source extraction range must stay within the pinned 2014-01-01..2026-09-27 range")
    if not db_path.is_file() or db_path.resolve() != DB_PATH.resolve():
        raise ValueError("only the pinned local read-only agri.sqlite database is allowed")
    prior = _read_json(prior_cache_path)
    _check_self_hash(prior, "cacheSha256", "prior external-reference cache")
    if prior.get("cacheSha256") != PRIOR_CACHE_SHA256:
        raise ValueError("prior selected-channel source cache hash differs from the frozen protocol")
    helper_path = ROOT / "agri_data/external_reference_v1.py"
    if prior.get("sourceCodeSha256") != file_sha256(helper_path):
        raise ValueError("validated extraction-helper version differs from the prior cache")
    price_raw = prices_path.read_bytes()
    if sha256(price_raw) != PRICE_SHA256:
        raise ValueError("pinned price snapshot hash mismatch")
    prior_source = prior.get("source", {})
    if (prior_source.get("snapshotSha256") != PRICE_SHA256
            or prior_source.get("extractedRange") != [start.isoformat(), end.isoformat()]):
        raise ValueError("prior source cache range/snapshot does not match the requested extraction")
    if output_path.exists():
        frozen = _read_json(output_path)
        _check_self_hash(frozen, "sourceCacheSha256", "full direct-price source cache")
        if (frozen.get("source", {}).get("priorCacheSha256") != PRIOR_CACHE_SHA256
                or frozen.get("source", {}).get("selectedRowsSha256") != prior_source.get("selectedRowsSha256")):
            raise ValueError("existing full source cache is bound to a different prior extraction")
        if frozen.get("source", {}).get("dataCodeSha256") != file_sha256(Path(__file__)):
            raise ValueError("existing full source cache was created by a different data-extraction implementation")
        _deserialize_full_channels(frozen.get("channels", {}))
        return frozen

    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        channels, audit = ext.extract_channels(db, start, end)
    prior_audit = prior.get("sourceAudit", {})
    if audit.get("selectedRowsSha256") != prior_source.get("selectedRowsSha256"):
        raise ValueError("re-extracted validated raw-row digest differs from the prior verified cache")
    if audit.get("rawDailyTargets") != prior_audit.get("rawDailyTargets"):
        raise ValueError("re-extracted representative-price raw totals differ from the prior verified cache")
    if audit.get("queryRange") != [start.isoformat(), end.isoformat()]:
        raise ValueError("source extraction returned an unexpected date range")
    full_channels = _serialize_full_channels(channels)
    # Round-trip validation catches key collisions and malformed numeric cells
    # before the cache becomes a frozen training input.
    decoded = _deserialize_full_channels(full_channels)
    channel_counts = {p: len(decoded[p]) for p in PRODUCTS}
    observation_counts = {p: sum(len(series) for series in decoded[p].values()) for p in PRODUCTS}
    doc = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "source": {
            "databasePath": "data/agri.sqlite", "readOnly": True,
            "query": "external_reference_v1.extract_channels / validated dashboard.PRICE_SQL rows",
            "range": [start.isoformat(), end.isoformat()],
            "sourceCodePath": "agri_data/external_reference_v1.py",
            "sourceCodeSha256": file_sha256(helper_path),
            "dataCodePath": "agri_data/direct_price_path_data.py",
            "dataCodeSha256": file_sha256(Path(__file__)),
            "priceSnapshotPath": str(prices_path.relative_to(ROOT)),
            "priceSnapshotSha256": PRICE_SHA256,
            "priorCachePath": str(prior_cache_path.relative_to(ROOT)),
            "priorCacheSha256": PRIOR_CACHE_SHA256,
            "selectedRowsSha256": audit["selectedRowsSha256"],
            "targetRawTotalsMatchedPriorVerifiedCache": True,
            "targetReconciliation": prior.get("targetReconciliation"),
        },
        "channelCounts": channel_counts,
        "positiveCellObservationCounts": observation_counts,
        "qualityAndQueryAudit": audit,
        "channelSelectionRule": {
            "representative": "separate frozen representative price snapshot",
            "markets": list(MARKETS),
            "detailWidth": 3,
            "lookbackCalendarDays": DETAIL_LOOKBACK_DAYS,
            "minimumPositiveObservedDates": MIN_DETAIL_DAYS,
            "maximumLastObservationAgeDays": DETAIL_MAX_AGE_DAYS,
            "maximumDetails": MAX_DETAIL_CHANNELS,
            "order": ["positive observed dates descending", "channel ID lexicographic ascending"],
            "usesOnlyTrainingCutoff": True,
        },
        "channels": full_channels,
    }
    doc["sourceCacheSha256"] = sha256(canonical(doc))
    raw = canonical(doc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_bytes(raw)
    temp_path.replace(output_path)
    return doc


def load_source_cache(path: Path = SOURCE_CACHE_PATH) -> tuple[dict[str, dict[tuple[str, ...], dict[dt.date, dict[str, Any]]]], dict[str, Any]]:
    doc = _read_json(path)
    _check_self_hash(doc, "sourceCacheSha256", "full direct-price source cache")
    if doc.get("experiment") != EXPERIMENT_ID:
        raise ValueError("source cache belongs to another experiment")
    if doc.get("source", {}).get("priorCacheSha256") != PRIOR_CACHE_SHA256:
        raise ValueError("source cache is not linked to the frozen verified prior cache")
    if doc.get("source", {}).get("priceSnapshotSha256") != PRICE_SHA256:
        raise ValueError("source cache uses another target snapshot")
    if doc.get("source", {}).get("dataCodeSha256") != file_sha256(Path(__file__)):
        raise ValueError("source cache was created by a different data-extraction implementation")
    return _deserialize_full_channels(doc.get("channels", {})), doc


def representative_series(prices: Mapping[str, Any]) -> dict[str, dict[dt.date, float]]:
    if prices.get("schemaVersion") != 1 or not isinstance(prices.get("products"), list):
        raise ValueError("pinned representative price snapshot has an unsupported schema")
    result: dict[str, dict[dt.date, float]] = {}
    seen_products = set()
    for item in prices["products"]:
        if not isinstance(item, Mapping):
            raise ValueError("price snapshot product entry must be an object")
        product = item.get("id")
        # The pinned shared snapshot also carries pepper; this experiment is
        # explicitly scoped to four target crops, so unrelated products are
        # ignored before validating the target-only domain.
        if product not in PRODUCTS:
            continue
        if product in seen_products:
            raise ValueError("price snapshot has duplicate direct-price product")
        seen_products.add(product)
        series: dict[dt.date, float] = {}
        for row in item.get("observations", []):
            try:
                day = dt.date.fromisoformat(str(row["date"]))
                value = float(row["actual"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"malformed representative observation for {product}") from exc
            if day in series or not math.isfinite(value) or value <= 0:
                raise ValueError(f"representative observation must be unique, positive and finite: {product}/{day}")
            series[day] = value
        result[product] = dict(sorted(series.items()))
    if seen_products != set(PRODUCTS):
        raise ValueError("price snapshot does not cover all fixed direct-price products")
    return result


def select_channels_for_cutoff(
    product_channels: Mapping[tuple[str, ...], Mapping[dt.date, Any]],
    training_end: dt.date,
) -> list[dict[str, Any]]:
    """Select up to eight detailed paths using only observations through training_end."""
    lower = training_end - dt.timedelta(days=DETAIL_LOOKBACK_DAYS - 1)
    details = []
    for key, series in product_channels.items():
        if len(key) != 3:
            continue
        days = sorted(day for day, value in series.items()
                      if lower <= day <= training_end
                      and _positive(value.get("price") if isinstance(value, Mapping) else value))
        past_days = [day for day in series if day <= training_end]
        if len(days) < MIN_DETAIL_DAYS or not past_days:
            continue
        last = max(past_days)
        if (training_end - last).days > DETAIL_MAX_AGE_DAYS:
            continue
        details.append((len(days), key, days[0], days[-1], last))
    details.sort(key=lambda x: (-x[0], x[1]))
    return [{"channelId": list(key), "grain": "market_subclass_grade",
             "supportDaysInLookback": count, "supportStart": first.isoformat(),
             "supportEnd": last_support.isoformat(), "lastObservedAtTrainingEnd": last.isoformat(),
             "selectionCutoff": training_end.isoformat(), "selectionStatus": "selected"}
            for count, key, first, last_support, last in details[:MAX_DETAIL_CHANNELS]]


def build_input_window(
    product: str, origin: dt.date, context_days: int, input_type: str,
    representative: Mapping[dt.date, float],
    product_channels: Mapping[tuple[str, ...], Mapping[dt.date, Any]],
    selected_details: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Build one origin-only, padded 11-channel window and metadata record."""
    if product not in PRODUCTS or context_days not in CONTEXTS or input_type not in ("U", "M"):
        raise ValueError("invalid product/context/input type")
    cutoff = origin - dt.timedelta(days=1)
    rep_before = [day for day in representative if day <= cutoff]
    if not rep_before:
        return None
    rep_day = max(rep_before)
    if (origin - rep_day).days > MAX_REP_AGE_DAYS:
        return None
    p0 = float(representative[rep_day])
    start = cutoff - dt.timedelta(days=context_days - 1)
    days = [start + dt.timedelta(days=i) for i in range(context_days)]

    channel_specs: list[tuple[tuple[str, ...] | None, int, str]] = [(None, 0, "representative")]
    for market_i, market in enumerate(MARKETS):
        channel_specs.append(((market,), 1 + market_i, "market"))
    for detail in list(selected_details)[:MAX_DETAIL_CHANNELS]:
        key = tuple(str(v) for v in detail["channelId"])
        if len(key) != 3:
            raise ValueError("selected detail channel must contain market, subclass and grade")
        channel_specs.append((key, 3, "detail"))
    while len(channel_specs) < 1 + len(MARKETS) + MAX_DETAIL_CHANNELS:
        channel_specs.append((None, 3, "detail_padding"))
    if len(channel_specs) != 11:
        raise AssertionError("the direct-price network must always receive eleven channels")

    columns: list[dict[str, Any]] = []
    for index, (key, role, kind) in enumerate(channel_specs):
        if index == 0:
            series: Mapping[dt.date, Any] = representative
        else:
            series = product_channels.get(key, {}) if key is not None else {}
        values: list[float] = []
        mask: list[int] = []
        for day in days:
            value = series.get(day)
            raw = value.get("price") if isinstance(value, Mapping) else value
            positive = _positive(raw)
            if positive is None:
                values.append(0.0)
                mask.append(0)
            else:
                values.append(float(positive))
                mask.append(1)
        observed_days = [day for day, observed in zip(days, mask) if observed]
        latest = max(observed_days) if observed_days else None
        if index == 0:
            active = True
            inactive_reason = None
        elif input_type == "U":
            active = False
            inactive_reason = "U_input_masks_auxiliary_channels"
        elif key is None:
            active = False
            inactive_reason = "padded_channel_slot"
        elif len(observed_days) < MIN_AUX_WINDOW_DAYS:
            active = False
            inactive_reason = "fewer_than_five_input_window_observations"
        elif latest is None or (origin - latest).days > MAX_AUX_AGE_DAYS:
            active = False
            inactive_reason = "auxiliary_channel_stale_at_origin"
        else:
            active = True
            inactive_reason = None
        if not active:
            values = [0.0] * context_days
            mask = [0] * context_days

        if active:
            logs = [math.log(v) for v, observed in zip(values, mask) if observed]
            mu = math.fsum(logs) / len(logs)
            variance = math.fsum((v - mu) ** 2 for v in logs) / len(logs)
            sigma = max(math.sqrt(variance), 0.01)
            normalized = [(math.log(v) - mu) / sigma if observed else 0.0
                          for v, observed in zip(values, mask)]
        else:
            mu, sigma, normalized = 0.0, 1.0, [0.0] * context_days
        last_any = max((day for day in series if day <= cutoff), default=None)
        age = (origin - last_any).days if last_any is not None else None
        if index == 0:
            metadata = [0.0, 0.0, sum(mask) / context_days,
                        min(age, MAX_AUX_AGE_DAYS) / MAX_AUX_AGE_DAYS if age is not None else 0.0]
        elif active:
            latest_value = series[last_any]
            latest_price = float(latest_value.get("price") if isinstance(latest_value, Mapping) else latest_value)
            metadata = [math.log(latest_price / p0), math.log(sigma / columns[0]["sigma"]),
                        sum(mask) / context_days,
                        min(age, MAX_AUX_AGE_DAYS) / MAX_AUX_AGE_DAYS if age is not None else 0.0]
        else:
            metadata = [0.0, 0.0, 0.0, 0.0]
        columns.append({"channelId": list(key) if key is not None else None,
                        "role": int(role), "kind": kind, "active": bool(active),
                        "inactiveReason": inactive_reason, "rawValues": values,
                        "observedMask": mask, "normalized": normalized,
                        "mu": mu, "sigma": sigma, "metadata": metadata,
                        "windowObservationCount": int(sum(mask)),
                        "latestWindowObservation": latest.isoformat() if latest else None,
                        "lastObservedAtCutoff": last_any.isoformat() if last_any else None,
                        "ageDaysAtOrigin": age})
    if not columns[0]["active"] or columns[0]["sigma"] < 0.01:
        raise AssertionError("representative channel normalization invariant failed")
    aux_count = sum(bool(c["active"]) for c in columns[1:])
    return {"product": product, "origin": origin.isoformat(), "cutoff": cutoff.isoformat(),
            "contextDays": context_days, "inputType": input_type,
            "targetVersion": PRICE_SHA256, "p0": p0, "p0Date": rep_day.isoformat(),
            "representativeSigma": columns[0]["sigma"],
            "selectedInputChannels": [dict(x) for x in selected_details] if input_type == "M" else [],
            "columns": columns, "activeAuxCount": aux_count}


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct-price path local source extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("extract", help="read the pinned local raw DB once and freeze all price channels")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        doc = extract_source_cache()
        print(json.dumps({"path": str(SOURCE_CACHE_PATH.relative_to(ROOT)),
                          "sourceCacheSha256": doc["sourceCacheSha256"],
                          "selectedRowsSha256": doc["source"]["selectedRowsSha256"],
                          "channelCounts": doc["channelCounts"],
                          "positiveCellObservationCounts": doc["positiveCellObservationCounts"],
                          "range": doc["source"]["range"]}, sort_keys=True))
        return 0
    raise ValueError("unsupported source extraction command")


if __name__ == "__main__":
    raise SystemExit(main())
