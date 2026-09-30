"""As-of lagged transaction signals and strictly issued-time-verified forecast adapter.

This is a retrospective exploratory runner. Forecast observations are never replaced with
ASOS observations; absent historical forecast archives leave that candidate unavailable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import coverage, dashboard, experiments, modeling, significance

PRODUCTS = ("cabbage", "radish")
CANDIDATES = ("lagged_transactions", "lagged_transactions_plus_forecast")
DEV_START, DEV_END = dt.date(2017, 1, 1), dt.date(2018, 12, 31)
EVAL_START, EVAL_END = dt.date(2019, 1, 1), dt.date(2020, 2, 18)
CONTEXT_START = dt.date(2014, 1, 1)
MIN_TRAIN_TARGET_DAYS = 90
MIN_INFERENCE_PAIRS = 30
EXPECTED_FROZEN_SHA256 = "9d6ac4d6a00c581821e97b4c4de074d129af6f06b8c8b38fe7ecbbf398c277f7"
HORIZONS = tuple(range(1, 11))
LOSSES = ("absolute_error", "absolute_percentage_error")
FORECAST_ELEMENTS = ("temperature_c", "rain_mm")
FORECAST_TZ = ZoneInfo("Asia/Seoul")
MODEL_RUNS = ("strict_asof_ridge", *CANDIDATES)
PAIRINGS = (("strict_asof_ridge", "asof_persistence"),
            ("lagged_transactions", "asof_persistence"),
            ("lagged_transactions", "strict_asof_ridge"),
            ("lagged_transactions_plus_forecast", "lagged_transactions"))
COMPARATORS = ("asof_persistence", "strict_asof_ridge", "lagged_transactions")
FAMILY_SIZE = len(PRODUCTS) * len(PAIRINGS) * len(LOSSES)
MARKET_CODES = tuple(dashboard.MARKETS)
DIMENSIONS = ("subclass", "grade")
VOLUME_WINDOWS_DAYS = (3, 7, 28)


def _day(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _aware_time(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return result


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


class ForecastArchive:
    """Validated in-memory index of a source archive and an evidenced location crosswalk."""
    def __init__(self, raw: dict[str, Any], source_bytes: bytes):
        if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
            raise ValueError("forecast archive must use schemaVersion 1")
        self.sha256 = hashlib.sha256(source_bytes).hexdigest()
        docs = {}
        for doc in raw.get("sourceDocuments", []):
            if not isinstance(doc, dict):
                raise ValueError("forecast source document must be an object")
            doc_id = str(doc.get("documentId", ""))
            if not doc_id or doc_id in docs:
                raise ValueError("forecast documentId is missing or duplicated")
            source_hash = _sha(doc.get("sourceSha256"), "sourceSha256")
            if not doc.get("sourceUrl"):
                raise ValueError("forecast sourceUrl is required")
            issue_at = _aware_time(doc.get("issuedAt"), "issuedAt")
            published_at = _aware_time(doc.get("publishedAt"), "publishedAt")
            available_at = _aware_time(doc.get("availableAt"), "availableAt")
            retrieved_at = _aware_time(doc.get("retrievedAt"), "retrievedAt")
            if published_at < issue_at or available_at < published_at or retrieved_at < available_at:
                raise ValueError("forecast issue/publication/availability/retrieval timestamps are not chronological")
            evidence = doc.get("issueTimeEvidence")
            if not isinstance(evidence, dict) or evidence.get("verified") is not True:
                raise ValueError("issueTimeEvidence.verified must be true")
            if _aware_time(evidence.get("verifiedIssuedAt"), "verifiedIssuedAt") != issue_at:
                raise ValueError("verifiedIssuedAt does not match issuedAt")
            if not evidence.get("sourceUrl"):
                raise ValueError("issue-time evidence sourceUrl is required")
            evidence_hash = _sha(evidence.get("sha256"), "issueTimeEvidence.sha256")
            publication_evidence = doc.get("publicationTimeEvidence")
            availability_evidence = doc.get("availabilityTimeEvidence")
            if not isinstance(publication_evidence, dict) or publication_evidence.get("verified") is not True:
                raise ValueError("publicationTimeEvidence.verified must be true")
            if _aware_time(publication_evidence.get("verifiedPublishedAt"), "verifiedPublishedAt") != published_at:
                raise ValueError("verifiedPublishedAt does not match publishedAt")
            if not publication_evidence.get("sourceUrl"):
                raise ValueError("publication-time evidence sourceUrl is required")
            publication_hash = _sha(publication_evidence.get("sha256"), "publicationTimeEvidence.sha256")
            if not isinstance(availability_evidence, dict) or availability_evidence.get("verified") is not True:
                raise ValueError("availabilityTimeEvidence.verified must be true")
            if _aware_time(availability_evidence.get("verifiedAvailableAt"), "verifiedAvailableAt") != available_at:
                raise ValueError("verifiedAvailableAt does not match availableAt")
            if not availability_evidence.get("sourceUrl"):
                raise ValueError("availability-time evidence sourceUrl is required")
            availability_hash = _sha(availability_evidence.get("sha256"), "availabilityTimeEvidence.sha256")
            docs[doc_id] = {**doc, "sourceSha256": source_hash,
                "issueTimeEvidence": {**evidence, "sha256": evidence_hash},
                "publicationTimeEvidence": {**publication_evidence, "sha256": publication_hash},
                "availabilityTimeEvidence": {**availability_evidence, "sha256": availability_hash},
                "issueAtParsed": issue_at, "publishedAtParsed": published_at,
                "availableAtParsed": available_at, "retrievedAtParsed": retrieved_at}
        if not docs:
            raise ValueError("forecast archive contains no issue-time-evidenced source documents")
        locations = {}
        for item in raw.get("originLocationCrosswalk", []):
            if not isinstance(item, dict) or item.get("verified") is not True or item.get("codeValidityVerified") is not True:
                continue
            code, location = str(item.get("originCode", "")), str(item.get("locationKey", ""))
            evidence = item.get("evidence")
            if not code or not location or not isinstance(evidence, dict) or not evidence.get("sourceUrl"):
                continue
            _sha(evidence.get("sha256"), "crosswalk evidence sha256")
            if code in locations and locations[code]["locationKey"] != location:
                raise ValueError(f"origin code has conflicting forecast locations: {code}")
            locations[code] = {**item, "originCode": code, "locationKey": location}
        rows: dict[tuple[str, dt.date, str], list[dict[str, Any]]] = defaultdict(list)
        for row in raw.get("forecasts", []):
            if not isinstance(row, dict):
                raise ValueError("forecast record must be an object")
            doc_id = str(row.get("documentId", ""))
            if doc_id not in docs:
                raise ValueError("forecast record references an unknown document")
            element = str(row.get("element", ""))
            if element not in FORECAST_ELEMENTS:
                raise ValueError(f"unsupported forecast element: {element}")
            location = str(row.get("locationKey", ""))
            if not location:
                raise ValueError("forecast locationKey is required")
            valid_date = _day(row.get("validDate"), "forecast validDate")
            value = _finite(row.get("value"), "forecast value")
            if element == "rain_mm" and value < 0:
                raise ValueError("rain_mm forecast cannot be negative")
            lead_hours = row.get("leadHours")
            if isinstance(lead_hours, bool) or not isinstance(lead_hours, int) or lead_hours < 0:
                raise ValueError("leadHours must be a nonnegative integer")
            source_doc = docs[doc_id]
            rows[(location, valid_date, element)].append({**row, "documentId": doc_id, "value": value,
                "issuedAtParsed": source_doc["issueAtParsed"], "retrievedAt": source_doc["retrievedAt"],
                "publishedAtParsed": source_doc["publishedAtParsed"],
                "availableAtParsed": source_doc["availableAtParsed"],
                "sourceUrl": source_doc["sourceUrl"], "sourceSha256": source_doc["sourceSha256"],
                "issueEvidenceUrl": source_doc["issueTimeEvidence"]["sourceUrl"],
                "issueEvidenceSha256": source_doc["issueTimeEvidence"]["sha256"],
                "publicationEvidenceUrl": source_doc["publicationTimeEvidence"]["sourceUrl"],
                "publicationEvidenceSha256": source_doc["publicationTimeEvidence"]["sha256"],
                "availabilityEvidenceUrl": source_doc["availabilityTimeEvidence"]["sourceUrl"],
                "availabilityEvidenceSha256": source_doc["availabilityTimeEvidence"]["sha256"]})
        self.documents, self.locations, self.rows = docs, locations, dict(rows)
        self.provenance = {"archiveSha256": self.sha256, "documentCount": len(docs),
            "evidencedDocumentCount": len(docs), "crosswalkCount": len(locations),
            "forecastRecordCount": sum(len(v) for v in rows.values()),
            "sourceDocuments": [{"documentId": d["documentId"], "sourceUrl": d["sourceUrl"],
                "sourceSha256": d["sourceSha256"], "issuedAt": d["issuedAt"], "retrievedAt": d["retrievedAt"],
                "publishedAt": d["publishedAt"], "availableAt": d["availableAt"],
                "issueTimeEvidenceUrl": d["issueTimeEvidence"]["sourceUrl"],
                "issueTimeEvidenceSha256": d["issueTimeEvidence"]["sha256"],
                "publicationTimeEvidenceUrl": d["publicationTimeEvidence"]["sourceUrl"],
                "publicationTimeEvidenceSha256": d["publicationTimeEvidence"]["sha256"],
                "availabilityTimeEvidenceUrl": d["availabilityTimeEvidence"]["sourceUrl"],
                "availabilityTimeEvidenceSha256": d["availabilityTimeEvidence"]["sha256"]}
                for d in sorted(docs.values(), key=lambda x: x["documentId"])]}

    def issued_before(self, origin_at: dt.datetime, target: dt.date, location_key: str, element: str):
        if origin_at.tzinfo is None or origin_at.utcoffset() is None:
            raise ValueError("origin_at must be timezone-aware")
        candidates = [row for row in self.rows.get((location_key, target, element), [])
                      if row["issuedAtParsed"] < origin_at
                      and row["publishedAtParsed"] < origin_at
                      and row["availableAtParsed"] < origin_at]
        if not candidates:
            return None
        latest = max(row["issuedAtParsed"] for row in candidates)
        winners = [row for row in candidates if row["issuedAtParsed"] == latest]
        values = {row["value"] for row in winners}
        if len(values) != 1:
            raise ValueError("conflicting forecasts with the same issue timestamp")
        return max(winners, key=lambda x: (x["retrievedAt"], x["documentId"]))


def load_forecast_archive(path: str | Path | None):
    if path is None or not Path(path).is_file():
        return None, {"status": "unavailable", "reason": "검증된 역사적 예보 archive가 없습니다."}
    raw_bytes = Path(path).read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("forecast archive is not valid JSON") from exc
    archive = ForecastArchive(raw, raw_bytes)
    return archive, {"status": "available", **archive.provenance}


def _origin_cutoff(origin: dt.date) -> dt.datetime:
    """Fixed 09:00 KST forecast issuance cutoff on origin day."""
    return dt.datetime.combine(origin, dt.time(9, 0), FORECAST_TZ)


def lagged_trade_feature_names() -> tuple[str, ...]:
    names = ["source_age_days", "log_total_volume_kg", "total_row_count_log"]
    for window in VOLUME_WINDOWS_DAYS:
        names.extend((f"volume_window_{window}d_observed_days", f"volume_window_{window}d_has_pair",
                      f"volume_window_{window}d_log_change", f"volume_window_{window}d_log_volatility"))
    for market in MARKET_CODES:
        names.extend((f"{market}_available", f"{market}_price_relative_log", f"{market}_volume_share",
                      f"{market}_log_volume_kg", f"{market}_log_row_count", f"{market}_quality_flag_count"))
        for dimension in DIMENSIONS:
            names.extend((f"{market}_{dimension}_distinct", f"{market}_{dimension}_largest_share",
                          f"{market}_{dimension}_hhi", *(f"{market}_{dimension}_top_share_{i}" for i in (1, 2, 3))))
            names.extend(f"{market}_{dimension}_code_hash_{i:02d}" for i in range(16))
    return tuple(names)


def lagged_trade_features(product: dict[str, Any], origin: dt.date):
    """Build fixed-width features from the latest completed selected trade date <= origin."""
    observations = {}
    for row in product.get("observations", []):
        day = _day(row.get("date"), "panel observation date")
        # A forecast issued at 09:00 cannot use the origin day's market aggregate,
        # which is only complete after trading. Use the latest earlier date.
        if day < origin:
            if day in observations:
                raise ValueError(f"duplicate selected panel date: {day}")
            observations[day] = row
    if not observations:
        return None
    source_day = max(observations)
    age = (origin - source_day).days
    if age < 0 or age > dashboard.MAX_SOURCE_AGE_DAYS:
        return None
    source = observations[source_day]
    total_volume = _finite(source.get("volumeKg"), "source volumeKg")
    if total_volume <= 0:
        return None
    aggregate_price = _finite(source.get("actual"), "source actual")
    if aggregate_price <= 0:
        return None
    market_data = source.get("marketData")
    if not isinstance(market_data, dict):
        market_data = {}
    positive_market_volume = sum(max(0.0, _finite(value.get("volumeKg", 0), "market volume"))
                                 for market in MARKET_CODES
                                 if isinstance((value := market_data.get(market)), dict))
    features = [float(age), math.log1p(total_volume), math.log1p(max(0, int(source.get("rowCount", 0))))]
    for window in VOLUME_WINDOWS_DAYS:
        window_start = source_day-dt.timedelta(days=window-1)
        selected = [(day, _finite(row.get("volumeKg"), "window volumeKg"))
                    for day, row in observations.items() if window_start <= day <= source_day]
        logged = [math.log1p(value) for _, value in sorted(selected)]
        enough = len(logged) >= 2
        if enough:
            mean = sum(logged)/len(logged)
            volatility = math.sqrt(sum((value-mean)**2 for value in logged)/len(logged))
            change = logged[-1]-logged[0]
        else:
            change = volatility = 0.0
        features.extend((float(len(logged)), float(enough), change, volatility))
    for market in MARKET_CODES:
        cell = market_data.get(market)
        if not isinstance(cell, dict):
            features.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            for _ in DIMENSIONS:
                features.extend((0.0,) * 22)
            continue
        price, volume = _finite(cell.get("price"), "market price"), _finite(cell.get("volumeKg"), "market volume")
        if price <= 0 or volume <= 0:
            raise ValueError("marketData must contain positive net price and volume")
        features.extend((1.0, math.log(price / aggregate_price), volume / positive_market_volume if positive_market_volume else 0.0,
                         math.log1p(volume), math.log1p(max(0, int(cell.get("rowCount", 0)))),
                         math.log1p(max(0, int(cell.get("qualityFlagCount", 0))))))
        composition = cell.get("composition", {})
        for dimension in DIMENSIONS:
            stats = composition.get(dimension, {}) if isinstance(composition, dict) else {}
            shares_map = stats.get("signedVolumeShares", {}) if isinstance(stats, dict) else {}
            if not isinstance(shares_map, dict):
                shares_map = {}
            shares = sorted((_finite(v, f"{dimension} share") for v in shares_map.values()), key=lambda x: (-abs(x), x))
            hashed = [0.0] * 16
            for code, value in shares_map.items():
                slot = int.from_bytes(hashlib.sha256(str(code).encode("utf-8")).digest()[:4], "big") % 16
                hashed[slot] += _finite(value, f"{dimension} share")
            features.extend((float(stats.get("distinctCategories", len(shares))) if stats else 0.0,
                             _finite(stats.get("largestSignedVolumeShare", max(shares, default=0.0)), f"{dimension} largest share") if stats else 0.0,
                             _finite(stats.get("concentrationHhi", sum(x*x for x in shares)), f"{dimension} HHI") if stats else 0.0,
                             *(shares[i] if i < len(shares) else 0.0 for i in range(3)), *hashed))
    if len(features) != len(lagged_trade_feature_names()) or not all(math.isfinite(v) for v in features):
        raise ValueError("lagged feature vector has an invalid shape or value")
    return {"sourceDate": source_day, "sourceAgeDays": age, "features": tuple(features),
            "featureNames": lagged_trade_feature_names()}


def _positive_origin_shares(product: dict[str, Any], origin: dt.date):
    prior = [(day, row) for row in product.get("observations", [])
             if (day := _day(row["date"], "date")) < origin]
    if not prior:
        return {}, {"status": "no_observation_on_or_before_origin"}
    source_day, row = max(prior, key=lambda item: item[0])
    source_age = (origin-source_day).days
    if source_age > dashboard.MAX_SOURCE_AGE_DAYS:
        return {}, {"status": "origin_composition_stale", "sourceDate": source_day.isoformat(), "sourceAgeDays": source_age}
    weights = defaultdict(float)
    negative = 0.0
    for market_cell in row.get("marketData", {}).values():
        market_volume = _finite(market_cell.get("volumeKg", 0), "origin market volume")
        shares = market_cell.get("compositionFieldShares", {}).get("plor_cd", {})
        for code, value in shares.items():
            signed_kg = _finite(value, "origin share") * market_volume
            if signed_kg > 0:
                weights[str(code)] += signed_kg
            elif signed_kg < 0:
                negative += abs(signed_kg)
    total = sum(weights.values())
    normalized = {code: value/total for code, value in weights.items()} if total > 0 else {}
    return normalized, {"status": "available" if normalized else "origin_shares_unavailable",
                        "sourceDate": source_day.isoformat(), "sourceAgeDays": source_age,
                        "negativeOriginShareKg": negative, "positiveOriginShareKg": total}


def forecast_feature_values(product: dict[str, Any], origin: dt.date, target: dt.date,
                            archive: ForecastArchive | None):
    names = ("temperature_c", "temperature_coverage", "temperature_issue_age_hours", "temperature_lead_hours",
             "rain_mm", "rain_coverage", "rain_issue_age_hours", "rain_lead_hours")
    if archive is None:
        return {"featureNames": names, "features": (0.0,)*len(names),
                "status": "unavailable_no_verified_forecast_archive", "provenance": []}
    origin_shares, share_meta = _positive_origin_shares(product, origin)
    if not origin_shares:
        return {"featureNames": names, "features": (0.0,)*len(names),
                "status": "unavailable_origin_composition", "shareDiagnostics": share_meta, "provenance": []}
    cutoff = _origin_cutoff(origin)
    locations = [(code, archive.locations.get(code)) for code in sorted(origin_shares)]
    outputs, used = [], []
    for element in FORECAST_ELEMENTS:
        weighted, matched_weight, issue_ages, leads, values_found = 0.0, 0.0, [], [], []
        for code, mapping in locations:
            if mapping is None:
                continue
            row = archive.issued_before(cutoff, target, mapping["locationKey"], element)
            if row is None:
                continue
            share = origin_shares[code]
            weighted += share * row["value"]
            matched_weight += share
            issue_ages.append((cutoff - row["issuedAtParsed"]).total_seconds()/3600)
            leads.append(float(row["leadHours"]))
            values_found.append(row["value"])
            used.append({"originCode": code, "locationKey": mapping["locationKey"], "element": element,
                "validDate": target.isoformat(), "issuedAt": row["issuedAtParsed"].isoformat(),
                "publishedAt": row["publishedAtParsed"].isoformat(),
                "availableAt": row["availableAtParsed"].isoformat(),
                "sourceUrl": row["sourceUrl"], "sourceSha256": row["sourceSha256"],
                "issueTimeEvidenceUrl": row["issueEvidenceUrl"], "issueTimeEvidenceSha256": row["issueEvidenceSha256"],
                "publicationTimeEvidenceUrl": row["publicationEvidenceUrl"],
                "publicationTimeEvidenceSha256": row["publicationEvidenceSha256"],
                "availabilityTimeEvidenceUrl": row["availabilityEvidenceUrl"],
                "availabilityTimeEvidenceSha256": row["availabilityEvidenceSha256"],
                "documentId": row["documentId"], "leadHours": row["leadHours"]})
        outputs.extend((weighted/matched_weight if matched_weight else 0.0,
                        matched_weight, max(issue_ages) if issue_ages else 0.0,
                        sum(leads)/len(leads) if leads else 0.0))
    return {"featureNames": names, "features": tuple(outputs),
        "status": "available" if used else "unavailable_no_asof_forecast_for_origin_regions",
        "originCutoffAt": cutoff.isoformat(), "validDate": target.isoformat(),
        "shareDiagnostics": share_meta, "provenance": used}


def _feature_row(product, series, origin, target, horizon, mode, archive, cache):
    # This candidate is evaluated as a 09:00 forecast on origin. Daily origin
    # prices and transaction aggregates are not complete yet, so its model
    # history ends on the prior calendar date.
    feature_origin = origin - dt.timedelta(days=1)
    base = modeling._feature_row(series, feature_origin, target, horizon, cache)
    if base is not None:
        source, core = base
        if (origin-source).days > dashboard.MAX_SOURCE_AGE_DAYS:
            return None
        # The strict price-history control excludes volume fields from the
        # frozen E0 feature set; transaction-added runs carry volume separately.
        keep = [i for i, name in enumerate(modeling.FEATURE_NAMES) if not name.endswith("LogVolumeKg")]
        base = source, tuple(core[i] for i in keep)
    if mode == "strict_asof_ridge":
        if base is None:
            return None
        source, core = base
        return source, tuple(core), {"sourceDate": source.isoformat(),
            "sourceAgeDays": (origin-source).days, "forecast": None}
    trade = lagged_trade_features(product, origin)
    if base is None or trade is None:
        return None
    source, core = base
    extra = trade["features"]
    weather_meta = None
    if mode == "trade_forecast":
        weather_meta = forecast_feature_values(product, origin, target, archive)
        extra = extra + weather_meta["features"]
    return source, tuple(core) + tuple(extra), {"sourceDate": trade["sourceDate"].isoformat(),
        "sourceAgeDays": trade["sourceAgeDays"], "forecast": weather_meta}


def _fit_candidate(product: dict[str, Any], start: dt.date, end: dt.date, mode: str,
                   archive: ForecastArchive | None, min_train_days: int = MIN_TRAIN_TARGET_DAYS):
    """Monthly expanding Ridge; labels must be strictly earlier than each fit origin."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    if mode not in ("strict_asof_ridge", "trade_only", "trade_forecast"):
        raise ValueError("unknown lagged-signal mode")
    if mode == "trade_forecast" and archive is None:
        return {"status": "unavailable_forecast_archive", "reason": "verified historical forecast archive unavailable",
                "predictions": [], "fits": [], "diagnostics": {"forecastUsableTrainingSamples": 0}}
    series = experiments._series(product)
    if not series:
        return {"status": "insufficient_data", "predictions": [], "fits": [], "diagnostics": {"issued": 0}}
    cache, samples, forecast_usable = {}, [], 0
    for target, (target_price, _) in sorted(series.items()):
        for horizon in HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            row = _feature_row(product, series, origin, target, horizon, mode, archive, cache)
            if row is None:
                continue
            source, vector, meta = row
            if mode == "trade_forecast" and meta["forecast"]["status"] == "available":
                forecast_usable += 1
            source_price = series[source][0]
            label = math.log(target_price / source_price)
            if math.isfinite(label) and all(math.isfinite(v) for v in vector):
                samples.append((target, origin, horizon, vector, label))
    samples.sort(key=lambda x: (x[0], x[1], x[2]))
    if mode == "trade_forecast" and forecast_usable == 0:
        return {"status": "unavailable_no_asof_forecast_for_training_targets",
                "reason": "no verified issued forecast joined to historical origin regions and valid dates",
                "predictions": [], "fits": [],
                "diagnostics": {"forecastUsableTrainingSamples": 0, "featureSamples": len(samples)}}
    preds, fits, estimator, fit_cutoff = [], [], None, None
    last_month = None
    fit_count = 0
    with threadpool_limits(limits=1):
        origin = start - dt.timedelta(days=max(HORIZONS))
        while origin < end:
            month = (origin.year, origin.month)
            if month != last_month:
                last_month = month
                # Strictly before the cutoff: same-day target labels are not treated as known.
                training = [sample for sample in samples if sample[0] < origin]
                target_dates = sorted({sample[0] for sample in training})
                if len(target_dates) < min_train_days:
                    estimator, fit_cutoff = None, None
                    fits.append({"cutoff": origin.isoformat(), "targetDays": len(target_dates),
                                 "samples": len(training), "status": "insufficient_data"})
                else:
                    estimator = modeling._new_estimator("ridge")
                    estimator.fit(np.asarray([sample[3] for sample in training], dtype=np.float64),
                                  np.asarray([sample[4] for sample in training], dtype=np.float64))
                    fit_cutoff = origin
                    fit_count += 1
                    fits.append({"cutoff": origin.isoformat(), "targetDays": len(target_dates),
                        "samples": len(training), "trainStart": target_dates[0].isoformat(),
                        "trainEnd": target_dates[-1].isoformat(), "labelsStrictlyBeforeCutoff": True,
                        "status": "fitted"})
            if estimator is not None and fit_cutoff is not None:
                target = origin + dt.timedelta(days=1)
                row = _feature_row(product, series, origin, target, 1, mode, archive, cache)
                if row is not None:
                    source, vector, meta = row
                    predicted = series[source][0] * math.exp(max(-700, min(700, float(estimator.predict([vector])[0]))))
                    if math.isfinite(predicted) and predicted > 0:
                        # Issue all calendar targets in scope; actual availability is joined afterwards.
                        if start <= target <= end:
                            forecast_meta = meta.get("forecast")
                            preds.append({"origin": origin.isoformat(), "target": target.isoformat(), "horizon": 1,
                                "predicted": predicted, "sourceDate": source.isoformat(),
                                "tradeSourceDate": meta["sourceDate"], "tradeSourceAgeDays": meta["sourceAgeDays"],
                                "fitCutoff": fit_cutoff.isoformat(), "fitTrainLabelEndExclusive": fit_cutoff.isoformat(),
                                "forecastStatus": forecast_meta["status"] if forecast_meta else "not_applicable",
                                "forecastDocumentIds": sorted({item["documentId"] for item in forecast_meta["provenance"]}) if forecast_meta else []})
            origin += dt.timedelta(days=1)
    return {"status": "available" if preds and fit_count else "insufficient_data",
        "predictions": preds, "fits": fits,
        "diagnostics": {"issued": len(preds), "successfulFits": fit_count,
            "featureTrainingSamples": len(samples), "forecastUsableTrainingSamples": forecast_usable}}


def _validate_panel(panel: dict[str, Any], frozen: dict[str, Any]):
    if _day(panel.get("panelStart"), "panelStart") > CONTEXT_START or _day(panel.get("panelEnd"), "panelEnd") < EVAL_END:
        raise ValueError("selected-price panel must cover 2014-01-01 through 2020-02-18")
    products, _validated_predictions, eval_start, eval_end = significance._validate_input(frozen)
    if eval_start > DEV_START or eval_end < EVAL_END:
        raise ValueError("frozen dashboard must cover development and exploratory evaluation periods")
    panel_products = {p.get("id"): p for p in panel.get("products", [])}
    if set(panel_products) != set(dashboard.PRODUCTS):
        raise ValueError("selected-price panel must contain all five product scopes")
    for product_id, product in panel_products.items():
        frozen_rows = {row["date"]: row for row in products[product_id].get("observations", [])
                       if CONTEXT_START.isoformat() <= row["date"] <= EVAL_END.isoformat()}
        panel_rows = {row["date"]: row for row in product.get("observations", [])
                      if CONTEXT_START.isoformat() <= row["date"] <= EVAL_END.isoformat()}
        if set(frozen_rows) != set(panel_rows):
            raise ValueError(f"frozen dashboard/panel observed dates differ for {product_id}")
        for day, row in frozen_rows.items():
            for field, panel_field in (("actual", "actual"), ("volumeKg", "volumeKg")):
                if not math.isclose(_finite(row[field], field), _finite(panel_rows[day][panel_field], panel_field),
                                    rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError(f"frozen dashboard/panel {field} mismatch for {product_id}/{day}")
            if int(row.get("rowCount", -1)) != int(panel_rows[day].get("rowCount", -2)):
                raise ValueError(f"frozen dashboard/panel rowCount mismatch for {product_id}/{day}")
    actuals = {product_id: {row["date"]: _finite(row["actual"], "frozen actual")
                           for row in product.get("observations", [])}
               for product_id, product in products.items()}
    return products, actuals, {"evaluationStart": eval_start.isoformat(), "evaluationEnd": eval_end.isoformat()}


def _metrics(rows, actual_by_day):
    errors = [float(row["predicted"]) - actual_by_day[row["target"]] for row in rows if row["target"] in actual_by_day]
    if not errors:
        return {"n": 0, "mae": None, "mape": None, "rmse": None}
    actuals_for_rows = [actual_by_day[row["target"]] for row in rows if row["target"] in actual_by_day]
    n = len(errors)
    return {"n": n, "mae": sum(abs(x) for x in errors)/n,
        "mape": 100*sum(abs(x)/a for x, a in zip(errors, actuals_for_rows))/n,
        "rmse": math.sqrt(sum(x*x for x in errors)/n)}


def _dm_checked(candidate, parent, actual_map, loss, max_lag):
    common = {(row["target"], int(row["horizon"])) for row in candidate} & \
             {(row["target"], int(row["horizon"])) for row in parent} & \
             set(actual_map)
    n = len(common)
    if n < MIN_INFERENCE_PAIRS:
        return {"status": "unavailable_insufficient_pairs", "n": n, "pValue": 1.0,
                "meanLossDifference": None, "statistic": None, "calendarHacMaxLagDays": max_lag}
    dm = experiments._dm(candidate, parent, actual_map, 1, loss, max_lag=max_lag)
    if dm.get("status") != "available":
        return {**dm, "status": dm.get("status", "unavailable"), "pValue": 1.0,
                "calendarHacMaxLagDays": max_lag}
    return dm


def _holm(records):
    if len(records) != FAMILY_SIZE:
        raise ValueError(f"each inference family must contain exactly {FAMILY_SIZE} records")
    order = sorted(range(len(records)), key=lambda i: (records[i]["test"]["pValue"], i))
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(records)-rank) * records[index]["test"]["pValue"])
        running = max(running, adjusted)
        records[index]["test"]["holmAdjustedPValue"] = running


def _asof_persistence(product, start: dt.date, end: dt.date):
    observations = {_day(row["date"], "observation date"): row for row in product.get("observations", [])}
    predictions = []
    day = start
    while day <= end:
        origin = day - dt.timedelta(days=1)
        eligible = [source for source in observations if source < origin and (origin-source).days <= dashboard.MAX_SOURCE_AGE_DAYS]
        if eligible:
            source = max(eligible)
            price = _finite(observations[source].get("actual"), "persistence source actual")
            if price > 0:
                predictions.append({"origin": origin.isoformat(), "target": day.isoformat(), "horizon": 1,
                    "predicted": price, "sourceDate": source.isoformat(), "fitCutoff": origin.isoformat(),
                    "fitTrainLabelEndExclusive": origin.isoformat()})
        day += dt.timedelta(days=1)
    return {"status": "available" if predictions else "insufficient_data", "predictions": predictions,
            "diagnostics": {"issued": len(predictions), "kind": "last_completed_observation_persistence"}}


def _archived_source_cutoff_audit(rows, evaluation_start: dt.date, evaluation_end: dt.date):
    h1 = [row for row in rows if row.get("horizon") == 1]
    evaluated = [row for row in h1 if evaluation_start.isoformat() <= row["target"] <= evaluation_end.isoformat()]
    eval_targets = {row["target"] for row in evaluated}
    compliant = eval_compliant = 0
    for row in h1:
        origin = _day(row["target"], "archived prediction target") - dt.timedelta(days=1)
        source = _day(row.get("sourceDate"), "archived prediction sourceDate")
        is_compliant = source < origin
        compliant += is_compliant
        if row["target"] in eval_targets:
            eval_compliant += is_compliant
    return {"allH1Predictions": len(h1), "allSourceDateBefore09OriginCount": compliant,
        "allSourceDateAtOrAfterOriginCount": len(h1)-compliant,
        "sourceCutoffComplianceShare": compliant/len(h1) if h1 else None,
        "exploratoryEvaluationH1Predictions": len(evaluated),
        "exploratorySourceDateBefore09OriginCount": eval_compliant,
        "exploratorySourceDateAtOrAfterOriginCount": len(evaluated)-eval_compliant,
        "exploratorySourceCutoffComplianceShare": eval_compliant/len(evaluated) if evaluated else None,
        "excludedFromInference": True}


def _comparison_records(model_runs, actuals, period):
    records = []
    for product_id in PRODUCTS:
        actual_map = {(day, 1): value for day, value in actuals[product_id].items()
                      if period[0] <= _day(day, "actual date") <= period[1]}
        actual_day_map = {day: value for (day, _), value in actual_map.items()}
        for candidate_id, comparator_id in PAIRINGS:
            candidate_run = model_runs.get(product_id, {}).get(candidate_id, {})
            comparator_run = model_runs.get(product_id, {}).get(comparator_id, {})
            candidate = [row for row in candidate_run.get("predictions", [])
                         if period[0].isoformat() <= row["target"] <= period[1].isoformat()]
            parent_rows = [row for row in comparator_run.get("predictions", [])
                           if period[0].isoformat() <= row["target"] <= period[1].isoformat() and row["horizon"] == 1]
            common = {(r["target"], 1) for r in candidate} & {(r["target"], 1) for r in parent_rows} & set(actual_map)
            c_rows = [r for r in candidate if (r["target"], 1) in common]
            p_rows = [r for r in parent_rows if (r["target"], 1) in common]
            c_metrics, p_metrics = _metrics(c_rows, actual_day_map), _metrics(p_rows, actual_day_map)
            pair_meta = {"candidateCoverage": len({r["target"] for r in candidate if r["target"] in actual_day_map}),
                "comparatorCoverage": len({r["target"] for r in parent_rows if r["target"] in actual_day_map}),
                "eligibleActualDays": len(actual_map), "commonPairs": len(common)}
            for loss in LOSSES:
                for lag, label in ((9, "primary"), (28, "sensitivity")):
                    if candidate_run.get("status") != "available" or comparator_run.get("status") != "available":
                        failed_status = candidate_run.get("status", "unavailable") if candidate_run.get("status") != "available" else comparator_run.get("status", "unavailable")
                        test = {"status": failed_status, "n": len(common), "pValue": 1.0,
                                "meanLossDifference": None, "statistic": None, "calendarHacMaxLagDays": lag}
                    else:
                        test = _dm_checked(c_rows, p_rows, actual_map, loss, lag)
                    records.append({"period": label, "hacMaxLagDays": lag, "productId": product_id,
                        "candidateId": candidate_id, "comparatorId": comparator_id, "loss": loss,
                        "metrics": {"candidate": c_metrics, "comparator": p_metrics},
                        "pairing": pair_meta, "test": test})
    for period_label in ("primary", "sensitivity"):
        _holm([row for row in records if row["period"] == period_label])
    conclusions = []
    for key in sorted({(r["productId"], r["candidateId"], r["comparatorId"], r["hacMaxLagDays"])
                       for r in records}):
        subset = [r for r in records if (r["productId"], r["candidateId"], r["comparatorId"], r["hacMaxLagDays"]) == key]
        tests = {r["loss"]: r["test"] for r in subset}
        if any(tests[l]["status"] != "available" for l in LOSSES):
            status = "unavailable"
        elif all(tests[l]["meanLossDifference"] < 0 and tests[l]["holmAdjustedPValue"] < .05 for l in LOSSES):
            status = "improved"
        elif any(tests[l]["meanLossDifference"] > 0 and tests[l]["holmAdjustedPValue"] < .05 for l in LOSSES):
            status = "worsened_or_mixed"
        else:
            status = "no_evidence"
        conclusions.append({"productId": key[0], "candidateId": key[1], "comparatorId": key[2],
                            "hacMaxLagDays": key[3], "status": status})
    return records, conclusions


def build_report(panel: dict[str, Any], frozen: dict[str, Any], frozen_bytes: bytes,
                 archive: ForecastArchive | None = None, archive_status: dict[str, Any] | None = None):
    frozen_products, actual_maps, frozen_period = _validate_panel(panel, frozen)
    panel_products = {p["id"]: p for p in panel["products"]}
    candidates = {}
    persistence = {}
    historical_ridge_reference = {}
    historical_baseline_reference = {}
    for product_id in PRODUCTS:
        product = panel_products[product_id]
        candidates[product_id] = {
            "strict_asof_ridge": _fit_candidate(product, DEV_START, EVAL_END, "strict_asof_ridge", None),
            CANDIDATES[0]: _fit_candidate(product, DEV_START, EVAL_END, "trade_only", None),
            CANDIDATES[1]: _fit_candidate(product, DEV_START, EVAL_END, "trade_forecast", archive),
        }
        persistence[product_id] = _asof_persistence(product, DEV_START, EVAL_END)
    actuals = {p: actual_maps[p] for p in PRODUCTS}
    for product_id in PRODUCTS:
        product = frozen_products[product_id]
        run = next((r for r in product["runs"] if r["modelId"] == "ridge" and r["cadence"] == "monthly"), None)
        if run is None:
            raise ValueError(f"frozen monthly Ridge run is missing for {product_id}")
        all_ref_rows = [row for row in run["predictions"] if row["horizon"] == 1]
        eval_ref_rows = [row for row in all_ref_rows
                         if EVAL_START.isoformat() <= row["target"] <= EVAL_END.isoformat()]
        actual_day_map = actuals[product_id]
        historical_ridge_reference[product_id] = {"inputEvaluationRange": frozen_period,
            **_archived_source_cutoff_audit(run["predictions"], EVAL_START, EVAL_END),
            "metricsExploratoryHistoricalOnly": _metrics(eval_ref_rows, actual_day_map),
            "excludedFromInference": True}
        historical_baseline_reference[product_id] = {"inputEvaluationRange": frozen_period,
            **_archived_source_cutoff_audit(product["predictions"], EVAL_START, EVAL_END)}
    split_records = {}
    for label, period in (("development", (DEV_START, DEV_END)), ("exploratoryEvaluation", (EVAL_START, EVAL_END))):
        model_runs = {product_id: {**candidates[product_id], "asof_persistence": persistence[product_id]}
                      for product_id in PRODUCTS}
        records, conclusions = _comparison_records(model_runs, actuals, period)
        split_records[label] = {"start": period[0].isoformat(), "end": period[1].isoformat(),
            "fixedFamilySizePerHac": FAMILY_SIZE, "comparisons": records, "conclusions": conclusions}
    coverage_report = {}
    for product in panel["products"]:
        if product["id"] not in PRODUCTS:
            continue
        obs = product["observations"]
        coverage_report[product["id"]] = {
            "developmentObservedTargetDays": sum(DEV_START.isoformat() <= row["date"] <= DEV_END.isoformat() for row in obs),
            "evaluationObservedTargetDays": sum(EVAL_START.isoformat() <= row["date"] <= EVAL_END.isoformat() for row in obs),
            "incompleteBroadMarketDaysInPanelRange": product.get("diagnostics", {}).get("incompleteBroadMarketDays"),
            "queryStateCounts": product.get("diagnostics", {}).get("queryStates", {}),
            "selectedRows": product.get("diagnostics", {}).get("includedRows")}
    return {"schemaVersion": 1, "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": {"name": "as-of lagged transaction features and historical issued-forecast adapter",
            "frozenInputSha256": hashlib.sha256(frozen_bytes).hexdigest(),
            "selectedPricePanelHash": experiments._panel_hash(panel),
            "contextStart": panel["panelStart"], "contextEnd": panel["panelEnd"],
            "development": {"start": DEV_START.isoformat(), "end": DEV_END.isoformat()},
            "evaluation": {"start": EVAL_START.isoformat(), "end": EVAL_END.isoformat(), "scope": "exploratory; previously inspected events; not confirmatory"},
            "model": {"id": "ridge", "parameters": modeling._model_parameters("ridge"),
                "cadence": "monthly", "target": "log(price_target/source_price)",
                "predictionTiming": "origin date D at 09:00 KST; target is calendar D+1, not next trading day",
                "trainingLabelCutoff": "target date strictly before fit origin", "minimumTargetDays": MIN_TRAIN_TARGET_DAYS},
            "featurePolicy": {"tradeSource": "latest selected observation date strictly before 09:00 origin date; maximum age 14 calendar days",
                "transactionFeatureNames": list(lagged_trade_feature_names()),
                "volumeWindows": {"calendarDays": list(VOLUME_WINDOWS_DAYS),
                    "windowEnd": "latest completed sourceDate, which may precede origin by multiple days",
                    "change": "log1p(latest observed net daily kg)-log1p(earliest observed net daily kg) in the trailing calendar window ending at latest source date",
                    "volatility": "population standard deviation of log1p(observed net daily kg) levels within the window, not return volatility",
                    "sparseRule": "no forward fill; emit observed-day count and has-pair indicator; change/volatility are zero when fewer than two observed dates"},
                "marketDataPolicy": "signed daily total target preserved; only available positive-net market cells expose market detail; missing markets are explicit zero+indicator",
                "compositionPolicy": "per-market subclass/grade signed share summaries and top-three absolute-share ranks; no future categories are used as vocabulary"},
            "candidateDefinitions": {"strict_asof_ridge": "기존 Ridge price-history features, 09:00 origin 전일까지만 사용.",
                CANDIDATES[0]: "Strict-as-of Ridge price-history features plus lagged transaction volume, market, subclass and grade features.",
                CANDIDATES[1]: "The lagged-transaction candidate plus evidenced, as-of-issued historical forecast features."},
            "forecastPolicy": {"status": archive_status or {"status": "unavailable", "reason": "no archive supplied"},
                "originCutoff": "09:00 Asia/Seoul on origin date; issuedAt, publishedAt and availableAt must each be strictly earlier and independently evidenced",
                "validDate": "forecast validDate must exactly equal target date",
                "crosswalk": "only rows with verified=true, codeValidityVerified=true, and evidence URL+SHA are usable; positive origin shares weight locations; negative origin mass is reported, not normalized into positive weights",
                "noObservedSubstitution": True,
                "unavailableDoesNotImplyNoForecast": True},
            "inference": {"losses": list(LOSSES), "hacCalendarLags": [9, 28], "minimumPairs": MIN_INFERENCE_PAIRS,
                "familySizePerPeriodAndLag": FAMILY_SIZE,
                "family": "2 products × 4 fixed paired contrasts × 2 losses; Holm separately for HAC9 and HAC28, unavailable tests retained at p=1",
                "fixedContrasts": [{"candidate": c, "comparator": p} for c, p in PAIRINGS],
                "improvement": "both AE and APE mean differences <0 and Holm-adjusted p<0.05",
                "interpretation": "retrospective exploratory; no-evidence is not equivalence; prior inspection and possible publication-vintage uncertainty remain"}},
        "coverage": coverage_report,
        "historicalRidgeReference": historical_ridge_reference,
        "historicalBaselineReference": historical_baseline_reference,
        "asOfPersistence": persistence,
        "forecastArchiveStatus": archive_status or {"status": "unavailable", "reason": "no archive supplied"},
        "products": [{"productId": p, "candidates": candidates[p]} for p in PRODUCTS],
        "splits": split_records,
        "limitations": ["Price and forecast publication times may be unknown; the experiment uses corrected current-retrieved price vintages.",
            "Frozen historical Ridge and persistence comparators retain their archived predictions; their original intraday information cutoff is not independently reconstructed here.",
            "Forecast rows require source document and issue-time evidence plus an independently evidenced origin-code crosswalk.",
            "Forecast/transaction associations remain noncausal; no forecast candidate result exists when its verified archive is unavailable."]}


def markdown_report(report):
    lines = ["# Lagged transaction and issued-forecast exploratory report", "",
        f"Development: {DEV_START}..{DEV_END}; exploratory evaluation: {EVAL_START}..{EVAL_END}.",
        f"Frozen input SHA-256: `{report['protocol']['frozenInputSha256']}`.",
        f"Forecast archive: **{report['forecastArchiveStatus']['status']}** — {report['forecastArchiveStatus'].get('reason') or 'verified archive loaded' }.",
        "Forecast output is not available when historical issue-time evidence or a verified origin crosswalk is missing. No ASOS observations substitute for forecasts.", "",
        "All comparisons are same-target-date, h1, and exploratory. Month-level rolling Ridge labels are strictly before each fit origin. First-publication times for corrected prices remain unknown.", ""]
    for split_name, split in report["splits"].items():
        lines.extend([f"## {split_name}: {split['start']}..{split['end']}", "",
            "| 품목 | 후보 | 비교 기준 | 후보 MAE | 기준 MAE | 후보 MAPE | 기준 MAPE | 공통 n | HAC9 AE/APE Holm p | HAC28 AE/APE Holm p | 결론 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
        rows = [r for r in split["comparisons"] if r["hacMaxLagDays"] == 9 and r["loss"] == "absolute_error"]
        by_key = {(r["productId"], r["candidateId"], r["comparatorId"], r["loss"], r["hacMaxLagDays"]): r
                  for r in split["comparisons"]}
        conclusions = {(r["productId"], r["candidateId"], r["comparatorId"], r["hacMaxLagDays"]): r["status"] for r in split["conclusions"]}
        for row in rows:
            key = (row["productId"], row["candidateId"], row["comparatorId"])
            ae9 = by_key[(*key, "absolute_error", 9)]
            ape9 = by_key[(*key, "absolute_percentage_error", 9)]
            ae28 = by_key[(*key, "absolute_error", 28)]
            ape28 = by_key[(*key, "absolute_percentage_error", 28)]
            m = row["metrics"]
            def metric(v, k): return "—" if v[k] is None else f"{v[k]:.3f}"
            pshow = lambda r: "—" if r["test"]["status"] != "available" else f"{r['test']['holmAdjustedPValue']:.4g}"
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {metric(m['candidate'],'mae')} | {metric(m['comparator'],'mae')} | "
                f"{metric(m['candidate'],'mape')} | {metric(m['comparator'],'mape')} | {row['pairing']['commonPairs']} | "
                f"{pshow(ae9)}/{pshow(ape9)} | {pshow(ae28)}/{pshow(ape28)} | {conclusions[(key[0],key[1],key[2],9)]} |")
        lines.append("")
    return "\n".join(lines)


def _write_exclusive(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # atomic no-overwrite publication
    finally:
        try: os.unlink(temp)
        except FileNotFoundError: pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--frozen", default="outputs/daily/2026-09-30/20260930T122342-844989ce/predictions.json")
    parser.add_argument("--forecast-archive", default=None)
    parser.add_argument("--output", default="outputs/lagged-signal-20260930.json")
    parser.add_argument("--markdown", default="outputs/lagged-signal-20260930.md")
    args = parser.parse_args(argv)
    try:
        frozen_bytes = Path(args.frozen).read_bytes()
        actual_sha = hashlib.sha256(frozen_bytes).hexdigest()
        if actual_sha != EXPECTED_FROZEN_SHA256:
            raise ValueError(f"frozen prediction input SHA-256 mismatch: {actual_sha}")
        frozen = json.loads(frozen_bytes)
        archive, archive_status = load_forecast_archive(args.forecast_archive)
        start, end = CONTEXT_START, EVAL_END
        db = coverage.open_readonly(args.db)
        try:
            db.execute("BEGIN")
            panel = experiments.build_price_panel(db, start, end)
            report = build_report(panel, frozen, frozen_bytes, archive, archive_status)
            db.rollback()
        finally:
            db.close()
        _write_exclusive(Path(args.output), json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        _write_exclusive(Path(args.markdown), markdown_report(report))
        return 0
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"lagged signal experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
