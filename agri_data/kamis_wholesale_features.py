"""Offline validation and causal features for KAMIS wholesale survey prices.

This module does not train or score models and never makes network requests.
The survey remains an external input signal in its original unit; it does not
replace or rewrite auction-price targets.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from agri_data import direct_horizon, full_history_xgb_rf_eval as rich
from agri_data import robust_price_features as baseline
from agri_data.kamis_wholesale_export import parse_export
from agri_data.kamis_wholesale_history import CROPS, build_query_plan

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "data/raw/kamis_wholesale_history_2014_202609_20261010"
EXPECTED_PLAN_SHA256 = "11e5655adcf51ef49400cf00abb150e96b962a505325308af5090f77ea41d26a"
EXPECTED_QUERIES = 255
ARCHIVE_START = dt.date(2014, 1, 1)
ARCHIVE_END = dt.date(2026, 9, 30)
DIAGNOSTIC_START = dt.date(2014, 1, 1)
DIAGNOSTIC_END = dt.date(2024, 12, 1)
DIAGNOSTIC_MAX_SOURCE_DATE = dt.date(2024, 11, 30)
HORIZON = 1
KG_DIVISORS = {"cabbage": 10, "radish": 20, "onion": 15, "garlic": 20, "pepper": 30}
FEATURE_NAMES = (
    "survey_available0",
    "survey_log_q0_over_auction_p0",
    "survey_q0_age_over_14",
    "survey_available0_q7_pair",
    "survey_log_q0_over_q7",
    "survey_available0_q28_pair",
    "survey_log_q0_over_q28",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[Any, bytes]:
    data = path.read_bytes()
    return json.loads(data), data


def _write_exclusive(path: Path, value: Any) -> None:
    """Atomically publish one create-only JSON artifact in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    finally:
        try: temp_path.unlink()
        except FileNotFoundError: pass


def _profile_for_query(query: dict[str, Any]) -> dict[str, Any]:
    crop = query.get("crop")
    if crop not in CROPS:
        raise ValueError(f"unknown KAMIS crop in query: {crop!r}")
    profile = CROPS[crop]
    expected = {"item": profile["item"], "variety": profile["variety"],
                "grade": "상품", "unit": profile["unit"]}
    for key, value in expected.items():
        if query.get(key) != value:
            raise ValueError(f"query {query.get('id')} {key} mismatch for {crop}")
    return profile


def _expected_queries(korean_language: bool = True) -> list[dict[str, Any]]:
    return build_query_plan(ARCHIVE_START, ARCHIVE_END, list(CROPS), korean_language=korean_language)


def _quality_status(parsed: dict[str, Any]) -> str:
    cells = [price for row in parsed["rawRows"] for price in row["prices"].values()]
    return "review_needed" if any(p["status"] == "invalid" or p["flags"] for p in cells) else "clear"


def _eligible_numeric(value: Decimal, divisor: int) -> bool:
    if not value.is_finite() or value <= 0: return False
    try: per_kg = float(value / Decimal(divisor))
    except (OverflowError, InvalidOperation, ValueError): return False
    return math.isfinite(per_kg) and per_kg > 0


def validate_query_checkpoint(directory: Path, query: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one frozen KAMIS query, every response body, and a reparse of its source."""
    profile = _profile_for_query(query)
    meta_path = directory / f"query-{query['id']}.json"
    meta, meta_bytes = _read_json(meta_path)
    key_pairs = (("id", "id"), ("crop", "crop"), ("start", "start"), ("end", "end"),
                 ("method", "method"), ("url", "url"), ("requestBody", "requestBody"),
                 ("requestSha256", "requestSha256"), ("fields", "formFields"),
                 ("requestHeaders", "requestHeaders"))
    for metadata_key, query_key in key_pairs:
        if meta.get(metadata_key) != query.get(query_key):
            raise ValueError(f"query metadata {metadata_key} mismatch for {query['id']}")
    if meta.get("state") != "completed" or meta.get("httpStatus") != 200:
        raise ValueError(f"query {query['id']} is not a completed HTTP 200 capture")
    responses = meta.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"query {query['id']} has no saved response attempts")
    response_hashes = []
    for response in responses:
        if response.get("requestUrl") != query["url"] or response.get("responseUrl") != query["url"]:
            raise ValueError(f"request/response URL provenance mismatch for {query['id']}")
        name = response.get("bodyFile")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe/missing body filename for {query['id']}")
        body = (directory / name).read_bytes()
        digest = _sha_bytes(body)
        if len(body) != response.get("bodyBytesCaptured") or digest != response.get("bodySha256"):
            raise ValueError(f"response body length/hash mismatch for {query['id']} attempt {response.get('attempt')}")
        if response.get("httpStatus") != 200 or response.get("outcome") != "http_success" or not response.get("framingComplete"):
            raise ValueError(f"non-success or incomplete response in query {query['id']}")
        response_hashes.append({"attempt": response.get("attempt"), "sha256": digest,
                                "bytes": len(body), "bodyFile": name})
    final = responses[-1]
    if (meta.get("sourceBodyFile") != final.get("bodyFile")
            or meta.get("sourceSha256") != final.get("bodySha256")):
        raise ValueError(f"final source reference mismatch for {query['id']}")
    for key in ("sourceBodyFile", "rowsFile"):
        if not isinstance(meta.get(key), str) or Path(meta[key]).name != meta[key]:
            raise ValueError(f"unsafe/missing {key} for {query['id']}")
    source = (directory / meta["sourceBodyFile"]).read_bytes()
    start, end = dt.date.fromisoformat(query["start"]), dt.date.fromisoformat(query["end"])
    parsed = parse_export(source, start=start, end=end, item=profile["item"], variety=profile["variety"],
                          grade="상품", unit=profile["unit"], table_id="itemTable_1")
    parsed["requested"]["crop"] = query["crop"]
    parsed["qualityStatus"] = _quality_status(parsed)
    rows_path = directory / meta["rowsFile"]
    rows_bytes = rows_path.read_bytes()
    if _sha_bytes(rows_bytes) != meta.get("rowsSha256"):
        raise ValueError(f"parsed rows SHA mismatch for {query['id']}")
    saved_rows = json.loads(rows_bytes)
    if _json_bytes(saved_rows) != _json_bytes(parsed):
        raise ValueError(f"parsed rows differ from reparsed source for {query['id']}")
    if saved_rows.get("sourceSha256") != meta.get("sourceSha256"):
        raise ValueError(f"parsed rows source hash mismatch for {query['id']}")
    return parsed, {"queryId": query["id"], "crop": query["crop"],
                    "metadataSha256": _sha_bytes(meta_bytes), "metadataBytes": len(meta_bytes),
                    "sourceSha256": meta["sourceSha256"], "sourceBytes": len(source),
                    "rowsSha256": meta["rowsSha256"], "rowsBytes": len(rows_bytes),
                    "responseBodies": response_hashes}


def validate_archive(collection_dir: Path | str = ARCHIVE_DIR, *,
                     expected_plan_sha256: str | None = EXPECTED_PLAN_SHA256,
                     expected_query_count: int = EXPECTED_QUERIES) -> dict[str, Any]:
    """Validate the pinned 255-query Korean-language archive, then reparse sources."""
    directory = Path(collection_dir)
    plan, plan_bytes = _read_json(directory / "plan.json")
    plan_sha = _sha_bytes(plan_bytes)
    if expected_plan_sha256 is not None and plan_sha != expected_plan_sha256:
        raise ValueError("frozen KAMIS collection plan SHA-256 mismatch")
    queries = plan.get("queries") if isinstance(plan, dict) else None
    expected = _expected_queries(korean_language=True)
    if not isinstance(queries, list) or len(queries) != expected_query_count:
        raise ValueError(f"expected exactly {expected_query_count} frozen KAMIS queries")
    if _json_bytes(queries) != _json_bytes(expected):
        raise ValueError("frozen KAMIS plan does not match exact crop/quarter/header request schema")
    if len({q.get("id") for q in queries}) != len(queries):
        raise ValueError("duplicate query id in frozen plan")
    expected_meta_files = {f"query-{q['id']}.json" for q in queries}
    expected_row_files = {f"query-{q['id']}.rows.json" for q in queries}
    actual_meta_files = {p.name for p in directory.glob("query-*.json") if not p.name.endswith(".rows.json")}
    actual_row_files = {p.name for p in directory.glob("query-*.rows.json")}
    if actual_meta_files != expected_meta_files or actual_row_files != expected_row_files:
        raise ValueError("archive contains missing or unexpected query metadata/parsed-row files")
    by_crop: dict[str, list[dict[str, Any]]] = {crop: [] for crop in CROPS}
    records: dict[str, dict[dt.date, dict[str, Any]]] = {crop: {} for crop in CROPS}
    audit: dict[str, dict[str, int]] = {crop: {"queryCount": 0, "observedRows": 0, "eligiblePositiveMean": 0,
                                                  "missingMean": 0, "invalidMean": 0, "flaggedMean": 0,
                                                  "nonPositiveOrNonfiniteMean": 0}
                                         for crop in CROPS}
    manifest_entries = []
    for query in queries:
        parsed, evidence = validate_query_checkpoint(directory, query)
        crop = query["crop"]
        by_crop[crop].append(query)
        audit[crop]["queryCount"] += 1
        for row in parsed["rawRows"]:
            day = dt.date.fromisoformat(row["date"])
            if day in records[crop]:
                raise ValueError(f"duplicate source observation date {crop}/{day}")
            mean = row["prices"]["평균"]
            audit[crop]["observedRows"] += 1
            flags = mean.get("flags", [])
            if mean["status"] == "missing": audit[crop]["missingMean"] += 1
            if mean["status"] == "invalid": audit[crop]["invalidMean"] += 1
            if flags: audit[crop]["flaggedMean"] += 1
            try:
                numeric = Decimal(mean["raw"].replace(",", "").strip()) if isinstance(mean.get("raw"), str) else Decimal("NaN")
            except (InvalidOperation, ValueError):
                numeric = Decimal("NaN")
            numeric_domain = _eligible_numeric(numeric, KG_DIVISORS[crop])
            eligible = mean["status"] == "observed" and not flags and numeric_domain
            if eligible: audit[crop]["eligiblePositiveMean"] += 1
            elif mean["status"] == "observed" and not numeric_domain:
                audit[crop]["nonPositiveOrNonfiniteMean"] += 1
            records[crop][day] = {"date": day.isoformat(), "crop": crop,
                                  "meanStatus": mean["status"], "meanFlags": list(flags),
                                  "meanRaw": mean.get("raw"), "meanDecimal": mean.get("decimal"),
                                  "originalUnit": query["unit"], "kgDivisor": KG_DIVISORS[crop],
                                  "queryId": query["id"], "sourceSha256": parsed["sourceSha256"],
                                  "eligiblePositiveMean": bool(eligible)}
        manifest_entries.append(evidence)
    if any(len(by_crop[crop]) != 51 for crop in CROPS):
        raise ValueError("expected exactly 51 quarter queries per crop")
    referenced_bodies = {body["bodyFile"] for entry in manifest_entries for body in entry["responseBodies"]}
    actual_body_files = {p.name for p in directory.glob("query-*.body")}
    if actual_body_files != referenced_bodies:
        raise ValueError("archive contains missing or unreferenced response body files")
    manifest = {"schemaVersion": 1, "collectionPlanSha256": plan_sha,
                "queries": manifest_entries}
    manifest_sha = _sha_bytes(_json_bytes(manifest))
    return {"collectionDir": str(directory), "collectionPlanSha256": plan_sha,
            "sourceManifestSha256": manifest_sha, "sourceManifest": manifest,
            "recordsByCrop": records, "auditByCrop": audit,
            "planQueryCount": len(queries), "archiveValidationStatus": "passed"}


def _mean_per_kg(record: dict[str, Any], crop: str) -> float | None:
    """Convert one validated eligible raw mean to float/kg without changing provenance."""
    if record.get("crop") != crop:
        raise ValueError("survey record crop mismatch")
    if record.get("originalUnit") != CROPS[crop]["unit"] or record.get("kgDivisor") != KG_DIVISORS[crop]:
        raise ValueError("survey record unit/divisor mismatch")
    if not record.get("eligiblePositiveMean") or record.get("meanStatus") != "observed" or record.get("meanFlags"):
        return None
    raw = record.get("meanRaw")
    if not isinstance(raw, str):
        return None
    try:
        value = Decimal(raw.replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite() or value <= 0:
        return None
    declared = record.get("meanDecimal")
    try:
        if not isinstance(declared, str) or Decimal(declared) != value:
            raise ValueError("survey raw mean/decimal mismatch")
    except InvalidOperation as exc:
        raise ValueError("invalid parsed survey decimal") from exc
    per_kg = float(value / Decimal(KG_DIVISORS[crop]))
    return per_kg if math.isfinite(per_kg) and per_kg > 0 else None


def _anchor(crop: str, surveys: dict[dt.date, dict[str, Any]], cutoff: dt.date) -> tuple[dict[str, Any] | None, int | None]:
    # Date filtering precedes all inspection/conversion of survey values.
    candidates = [(day, surveys[day]) for day in surveys if day <= cutoff]
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for day, record in candidates:
        value = _mean_per_kg(record, crop)
        if value is not None:
            age = (cutoff - day).days
            if age < 0: raise AssertionError("survey anchor is after its cutoff")
            return {"date": day, "valuePerKg": value, "record": record}, age
    return None, None


def features(crop: str, records: dict[dt.date, dict[str, Any]],
             direct_series: dict[dt.date, tuple[float, float]],
             survey_records: dict[dt.date, dict[str, Any]], target: dt.date, horizon: int
             ) -> tuple[list[float] | None, float | None, dict[str, Any]]:
    """Return the frozen 106 auction features plus seven causal KAMIS inputs."""
    if crop not in CROPS: raise ValueError("unknown crop")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=1)
    # Keep the numerical helpers from seeing post-cutoff values at all.
    causal_records = {day: value for day, value in records.items() if day <= cutoff}
    causal_direct = {day: value for day, value in direct_series.items() if day <= cutoff}
    base, p0, base_audit = baseline.features(causal_records, causal_direct, target, horizon)
    if base is None or p0 is None:
        return None, None, {"status": base_audit.get("status", "base106_unavailable"),
                            "featureCutoff": cutoff.isoformat(), "baseAudit": base_audit}
    if len(base) < 106 or not all(math.isfinite(float(x)) for x in base[:106]):
        raise ValueError("robust_price_features did not provide a finite frozen 106-value prefix")
    if not math.isfinite(float(p0)) or float(p0) <= 0:
        raise ValueError("auction p0 must be positive and finite")
    survey_cutoffs = {0: cutoff, 7: cutoff - dt.timedelta(days=7), 28: cutoff - dt.timedelta(days=28)}
    anchors: dict[int, dict[str, Any] | None] = {}
    ages: dict[int, int | None] = {}
    # Survey lookup also filters by date before parsing raw numeric strings.
    for lag, anchor_cutoff in survey_cutoffs.items():
        candidates = {day: row for day, row in survey_records.items() if day <= anchor_cutoff}
        anchor, age = _anchor(crop, candidates, anchor_cutoff)
        if age is not None and age > 14: anchor, age = None, age
        anchors[lag], ages[lag] = anchor, age
    q0, q7, q28 = (anchors[lag] for lag in (0, 7, 28))
    available0 = q0 is not None
    available7pair = q0 is not None and q7 is not None
    available28pair = q0 is not None and q28 is not None
    appended = [float(available0), math.log(q0["valuePerKg"] / float(p0)) if available0 else 0.0,
                float(ages[0] / 14.0) if available0 else 0.0,
                float(available7pair), math.log(q0["valuePerKg"] / q7["valuePerKg"]) if available7pair else 0.0,
                float(available28pair), math.log(q0["valuePerKg"] / q28["valuePerKg"]) if available28pair else 0.0]
    if not all(math.isfinite(x) for x in appended): raise ValueError("nonfinite KAMIS feature")
    vector = [float(x) for x in base[:106]] + appended
    anchor_audit = {}
    for lag, anchor in anchors.items():
        if anchor is None:
            anchor_audit[str(lag)] = {"available": False, "ageDays": ages[lag]}
            continue
        source = anchor["record"]
        anchor_audit[str(lag)] = {"available": True, "date": anchor["date"].isoformat(),
                                  "ageDays": ages[lag], "meanRaw": source["meanRaw"],
                                  "meanDecimal": source["meanDecimal"], "originalUnit": source["originalUnit"],
                                  "kgDivisor": source["kgDivisor"], "valuePerKg": anchor["valuePerKg"],
                                  "queryId": source["queryId"], "sourceSha256": source["sourceSha256"]}
    return vector, float(p0), {"status": "available", "featureCutoff": cutoff.isoformat(),
                               "baseFeatureCount": 106, "addedFeatureCount": 7,
                               "surveyCutoffs": {str(k): v.isoformat() for k, v in survey_cutoffs.items()},
                               "anchors": anchor_audit, "baseAudit": base_audit}


def _first_of_month_origins() -> list[dt.date]:
    return [dt.date(year, month, 1) for year in range(DIAGNOSTIC_START.year, DIAGNOSTIC_END.year + 1)
            for month in range(1, 13) if dt.date(year, month, 1) <= DIAGNOSTIC_END]


def _entry_gate(joint_log_q0_p0: list[float]) -> dict[str, Any]:
    nonconstant = len(set(joint_log_q0_p0)) > 1
    reasons = []
    if len(joint_log_q0_p0) < 30: reasons.append("fewer_than_30_joint_base_q0_monthly_origins")
    if not nonconstant: reasons.append("log_q0_over_p0_is_constant_or_unavailable")
    return {"eligibleForLaterReview": not reasons, "jointBaseAndQ0Origins": len(joint_log_q0_p0),
            "minimumJointOrigins": 30, "logQ0OverP0Nonconstant": nonconstant, "reasons": reasons}


def _load_pinned_prices(path: Path) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    sha = _sha_bytes(raw_bytes)
    if sha != baseline.PRICES_SHA256:
        raise ValueError("pinned auction prices SHA-256 mismatch")
    raw = json.loads(raw_bytes)
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("auction price snapshot must use dashboard schemaVersion 1")
    clipped_products = []
    for product in raw["products"]:
        if not isinstance(product, dict) or not isinstance(product.get("observations"), list):
            raise ValueError("malformed auction price product")
        clipped = []
        for row in product["observations"]:
            if not isinstance(row, dict): continue
            try: day = dt.date.fromisoformat(row.get("date", ""))
            except (TypeError, ValueError): continue
            if day <= DIAGNOSTIC_MAX_SOURCE_DATE:
                clipped.append(row)
        clipped_products.append({**product, "observations": clipped})
    return {**raw, "products": clipped_products}, sha


def run_diagnostic(collection_dir: Path | str = ARCHIVE_DIR, prices_path: Path = baseline.PRICES_PATH,
                   output_path: Path | None = None) -> dict[str, Any]:
    archive = validate_archive(collection_dir)
    auction, auction_sha = _load_pinned_prices(prices_path)
    direct_by_crop = direct_horizon._observations(auction)
    raw_by_crop, raw_audit = rich._raw_maps(auction)
    origins = _first_of_month_origins()
    per_crop = {}
    for crop in CROPS:
        survey_all = archive["recordsByCrop"][crop]
        survey = {day: rec for day, rec in survey_all.items() if day <= DIAGNOSTIC_MAX_SOURCE_DATE}
        direct_all = direct_by_crop[crop]
        records_all = raw_by_crop[crop]
        auction_dates = set()
        for row in next(p for p in auction["products"] if p["id"] == crop)["observations"]:
            try: auction_dates.add(dt.date.fromisoformat(row["date"]))
            except (KeyError, TypeError, ValueError): pass
        survey_dates = set(survey)
        raw_row_counts = {name: [] for name in FEATURE_NAMES}
        joint_q0_log_ratios = []
        q0_ages, q7_ages, q28_ages = [], [], []
        avail0 = pair7 = pair28 = base_unavailable = 0
        exact_prefix = True
        for origin in origins:
            target = origin + dt.timedelta(days=HORIZON)
            cutoff = origin - dt.timedelta(days=1)
            # Trim all numerical feature inputs to this origin before calling helpers.
            local_records = {d: r for d, r in records_all.items() if d <= cutoff}
            local_direct = {d: x for d, x in direct_all.items() if d <= cutoff}
            local_survey = {d: r for d, r in survey.items() if d <= cutoff}
            base, _p0, _ba = baseline.features(local_records, local_direct, target, HORIZON)
            vector, _p, audit = features(crop, local_records, local_direct, local_survey, target, HORIZON)
            if vector is None:
                base_unavailable += 1
                continue
            if base is None or vector[:106] != [float(x) for x in base[:106]]:
                exact_prefix = False
                raise ValueError("KAMIS feature helper changed the robust baseline 106-value prefix")
            for name, value in zip(FEATURE_NAMES, vector[106:]): raw_row_counts[name].append(value)
            anchors = audit["anchors"]
            if anchors["0"]["available"]:
                avail0 += 1; q0_ages.append(anchors["0"]["ageDays"])
                joint_q0_log_ratios.append(vector[107])
            if anchors["0"]["available"] and anchors["7"]["available"]:
                pair7 += 1; q7_ages.append(anchors["7"]["ageDays"])
            if anchors["0"]["available"] and anchors["28"]["available"]:
                pair28 += 1; q28_ages.append(anchors["28"]["ageDays"])
        feature_summary = {}
        for name, values in raw_row_counts.items():
            feature_summary[name] = {"samples": len(values), "min": min(values) if values else None,
                                     "max": max(values) if values else None,
                                     "nonconstant": len(set(values)) > 1 if values else False,
                                     "nonzero": sum(v != 0 for v in values)}
        per_crop[crop] = {"plannedOrigins": len(origins), "baseUnavailable": base_unavailable,
                          "q0Available": avail0, "q0AgeDays": {"min": min(q0_ages) if q0_ages else None,
                                                                  "max": max(q0_ages) if q0_ages else None},
                          "q0q7PairAvailable": pair7,
                          "q7AgeDays": {"min": min(q7_ages) if q7_ages else None, "max": max(q7_ages) if q7_ages else None},
                          "q0q28PairAvailable": pair28,
                          "q28AgeDays": {"min": min(q28_ages) if q28_ages else None, "max": max(q28_ages) if q28_ages else None},
                          "auctionDateCountThroughCutoff": len(auction_dates),
                          "surveyDateCountThroughCutoff": len(survey_dates),
                          "overlappingAuctionSurveyDatesThroughCutoff": len(auction_dates & survey_dates),
                          "sourceMeanAudit": archive["auditByCrop"][crop],
                          "auctionRawMapAudit": raw_audit[crop],
                          "featureSummary": feature_summary,
                          "entryGate": _entry_gate(joint_q0_log_ratios),
                          "baseline106PrefixExact": exact_prefix}
    result = {"schemaVersion": 1, "status": "read_only_kamis_survey_feature_diagnostic",
              "noFit": True, "noScoring": True, "noNetwork": True,
              "targetValuesOrLossesAttached": False,
              "protocol": {"sample": "first day of each month, 2014-01 through 2024-12, five crops",
                           "plannedSamples": len(origins) * len(CROPS), "horizonDays": HORIZON,
                           "surveyFeatureCutoff": "origin minus one day; anchor cutoffs are cutoff, cutoff-7d, cutoff-28d",
                           "maximumDiagnosticAuctionAndSurveySourceDate": DIAGNOSTIC_MAX_SOURCE_DATE.isoformat(),
                           "maximumSurveyAnchorAgeDays": 14,
                           "featureNames": list(FEATURE_NAMES), "frozenAuctionFeaturePrefix": 106,
                           "surveyPriceConversion": "original unit divisor; auction target unchanged",
                           "publicationTiming": "not established; survey date is not a verified first-publication timestamp",
                           "missingDates": "not filled; only bounded latest observed anchors are used"},
              "archiveValidation": {"status": archive["archiveValidationStatus"],
                                    "validatedQueries": archive["planQueryCount"],
                                    "collectionPlanSha256": archive["collectionPlanSha256"],
                                    "sourceManifestSha256": archive["sourceManifestSha256"],
                                    "maximumArchiveDate": ARCHIVE_END.isoformat(),
                                    "separateDiagnosticCutoff": DIAGNOSTIC_MAX_SOURCE_DATE.isoformat()},
              "sourceManifest": archive["sourceManifest"],
              "auctionSource": {"path": str(prices_path), "sha256": auction_sha},
              "perCrop": per_crop,
              "featureCodeSha256": _sha_bytes(Path(__file__).read_bytes()),
              "baselineFeatureCodeSha256": _sha_bytes(Path(baseline.__file__).read_bytes())}
    if output_path is not None: _write_exclusive(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    validate = subs.add_parser("validate", help="verify and reparse the pinned 255-query archive")
    validate.add_argument("--collection-dir", type=Path, default=ARCHIVE_DIR)
    validate.add_argument("--output", type=Path, required=True)
    diagnostic = subs.add_parser("diagnostic", help="run the cutoff-bounded no-fit availability/feature diagnostic")
    diagnostic.add_argument("--collection-dir", type=Path, default=ARCHIVE_DIR)
    diagnostic.add_argument("--prices", type=Path, default=baseline.PRICES_PATH)
    diagnostic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_archive(args.collection_dir)
        _write_exclusive(args.output, {k: result[k] for k in
            ("archiveValidationStatus", "planQueryCount", "collectionPlanSha256", "sourceManifestSha256", "auditByCrop")})
        print(json.dumps({"status": "passed", "queries": result["planQueryCount"],
                          "sourceManifestSha256": result["sourceManifestSha256"]}))
    else:
        result = run_diagnostic(args.collection_dir, args.prices, args.output)
        print(json.dumps({"status": result["status"], "plannedSamples": result["protocol"]["plannedSamples"],
                          "sourceManifestSha256": result["archiveValidation"]["sourceManifestSha256"],
                          "output": str(args.output)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
