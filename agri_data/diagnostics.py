"""Read-only historical diagnostics for price, weather, composition and onion errors."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import collector, coverage, experiments, significance

CONTEXT_START = dt.date(2014, 1, 1)
CONTEXT_END = dt.date(2018, 12, 20)
EVALUATION_START = dt.date(2017, 1, 1)
EVALUATION_END = dt.date(2018, 12, 20)
STATIONS = ("100", "108", "261")
WEATHER_FEATURES = ("avgTa_7day_mean", "daily_range_7day_mean", "sumRn_7day_sum")
WEATHER_FIELDS = {"avgTa_7day_mean": ("avgTa",),
                  "daily_range_7day_mean": ("maxTa", "minTa"),
                  "sumRn_7day_sum": ("sumRn",)}
TARGET_PRODUCTS = ("cabbage", "radish")
INVENTORY_PRODUCTS = tuple(experiments.PRODUCT_IDS)
SEASONS = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
COMPOSITION_FIELDS = {"market": None, "subclass": "gds_sclsf_cd", "grade": "grd_cd", "origin": "plor_cd"}
TVD_MISSING = "__unaccounted_share__"
FAMILY_SIZE = len(TARGET_PRODUCTS) * (len(STATIONS) * len(WEATHER_FEATURES) + 1)
ALPHA = 0.05
HAC_LAGS = (9, 28)


def _date(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {value}") from exc


def _number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _source_scope(dataset: str, url: str, query_json: str) -> tuple[str, dict[str, str]] | None:
    try:
        params = json.loads(query_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(params, dict):
        return None
    base = {str(key): str(value) for key, value in params.items()
            if key not in ("pageNo", "numOfRows", "returnType")}
    return collector.task_key(dataset, url, base), base


def _page_counts(db, request_id: int) -> tuple[int, int]:
    raw_count = db.execute("SELECT COUNT(*) FROM raw_records WHERE request_id=?", (request_id,)).fetchone()[0]
    norm_count = db.execute("SELECT COUNT(*) FROM normalized_records WHERE request_id=?", (request_id,)).fetchone()[0]
    return int(raw_count), int(norm_count)


def _valid_epoch(pages: list[dict[str, Any]], plan: dict[str, Any]) -> tuple[bool, str | None]:
    if not pages:
        return False, "no_response_pages"
    if plan["epochId"] == pages[0]["epochId"] and plan["state"] not in ("complete", "empty"):
        return False, "current_manifest_epoch_not_complete"
    page_numbers = sorted(page["pageNo"] for page in pages if page["pageNo"] is not None)
    if len(page_numbers) != len(pages) or len(set(page_numbers)) != len(pages):
        return False, "duplicate_or_missing_page_number"
    if page_numbers != list(range(1, len(pages) + 1)):
        return False, "page_gap"
    totals = {page["expectedCount"] for page in pages}
    page_sizes = {page["pageSize"] for page in pages}
    if len(totals) != 1 or None in totals or len(page_sizes) != 1 or None in page_sizes:
        return False, "unstable_total_or_page_size"
    total, page_size = next(iter(totals)), next(iter(page_sizes))
    if page_size <= 0 or total < 0:
        return False, "invalid_total_or_page_size"
    expected_pages = max(1, (total + page_size - 1) // page_size)
    if len(pages) != expected_pages:
        return False, "incomplete_pagination"
    returned = 0
    for page in pages:
        expected_page_rows = max(0, min(page_size, total - (page["pageNo"] - 1) * page_size))
        if page["state"] != "complete":
            return False, "response_not_complete"
        if page["returnedCount"] != expected_page_rows:
            return False, "page_count_mismatch"
        if page["rawCount"] != page["returnedCount"] or page["normalizedCount"] != page["returnedCount"]:
            return False, "record_link_count_mismatch"
        returned += page["returnedCount"]
    if returned != total:
        return False, "epoch_total_mismatch"
    if plan["state"] in ("complete", "empty") and plan["epochId"] == pages[0]["epochId"]:
        if plan["rowCount"] != total:
            return False, "manifest_count_mismatch"
    return True, None


def _weather_plans(db, start: dt.date, end: dt.date) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    plan_rows = db.execute("""SELECT p.job_id,p.task_key,p.source_url,p.query_json,p.state,p.row_count,p.epoch_id,
                                     j.started_at,j.state AS job_state
                              FROM planned_queries p JOIN jobs j USING(job_id)
                              WHERE p.dataset='weather' ORDER BY p.job_id,p.task_key""").fetchall()
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relevant = {}
    for row in plan_rows:
        if row["source_url"] != collector.ASOS:
            continue
        try:
            params = json.loads(row["query_json"])
            query_start = dt.datetime.strptime(str(params["startDt"]), "%Y%m%d").date()
            query_end = dt.datetime.strptime(str(params["endDt"]), "%Y%m%d").date()
            station = str(params["stnIds"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise ValueError(f"invalid weather query manifest for job {row['job_id']}")
        if station not in STATIONS or params.get("dataCd") != "ASOS" or params.get("dateCd") != "DAY":
            continue
        if query_end < start or query_start > end:
            continue
        task = collector.task_key("weather", row["source_url"],
                                  {str(key): str(value) for key, value in params.items()})
        if task != row["task_key"]:
            raise ValueError(f"weather task key mismatch for job {row['job_id']}")
        scope_key = (row["source_url"], row["task_key"])
        plan = {"jobId": int(row["job_id"]), "taskKey": row["task_key"], "sourceUrl": row["source_url"],
                "start": query_start, "end": query_end, "station": station,
                "state": row["state"], "rowCount": row["row_count"], "epochId": row["epoch_id"],
                "jobState": row["job_state"], "jobStartedAt": row["started_at"], "query": params}
        scopes[scope_key].append(plan)
        relevant[scope_key] = plan

    request_rows = db.execute("""SELECT r.request_id,r.job_id,r.epoch_id,r.source_url,r.query_json,r.retrieved_at,
                                       r.response_sha256,r.http_status,r.api_status,r.expected_count,r.returned_count,
                                       r.page_no,r.state,r.published_at,r.version_published_at,
                                       (SELECT COUNT(*) FROM raw_records rr WHERE rr.request_id=r.request_id) AS raw_count,
                                       (SELECT COUNT(*) FROM normalized_records nn WHERE nn.request_id=r.request_id) AS normalized_count
                                FROM source_requests r WHERE r.dataset='weather' AND r.source_url=?
                                ORDER BY r.request_id""", (collector.ASOS,)).fetchall()
    page_groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        scope = _source_scope("weather", row["source_url"], row["query_json"])
        if scope is None or (row["source_url"], scope[0]) not in relevant:
            continue
        scope_key = (row["source_url"], scope[0])
        params = scope[1]
        try:
            request_start = dt.datetime.strptime(params["startDt"], "%Y%m%d").date()
            request_end = dt.datetime.strptime(params["endDt"], "%Y%m%d").date()
            page_no, page_size = int(json.loads(row["query_json"]).get("pageNo")), int(json.loads(row["query_json"]).get("numOfRows"))
            expected, returned = int(row["expected_count"]), int(row["returned_count"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if request_end < start or request_start > end:
            continue
        plan_candidates = [plan for plan in scopes[scope_key] if plan["jobId"] == row["job_id"]]
        if not plan_candidates:
            continue
        plan = plan_candidates[0]
        for key in ("startDt", "endDt", "stnIds", "dataCd", "dateCd", "dataType"):
            if str(params.get(key)) != str(plan["query"].get(key)):
                raise ValueError(f"weather request scope mismatch for request {row['request_id']}")
        page = {"requestId": int(row["request_id"]), "jobId": int(row["job_id"]),
                "epochId": row["epoch_id"], "retrievedAt": row["retrieved_at"],
                "responseSha256": row["response_sha256"], "httpStatus": row["http_status"],
                "apiStatus": row["api_status"], "expectedCount": expected, "returnedCount": returned,
                "pageNo": page_no, "pageSize": page_size, "state": row["state"],
                "rawCount": int(row["raw_count"]), "normalizedCount": int(row["normalized_count"]),
                "publishedAt": row["published_at"], "versionPublishedAt": row["version_published_at"]}
        group_key = (row["source_url"], scope[0], int(row["job_id"]), str(row["epoch_id"]))
        page_groups[group_key].append(page)

    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    scope_report = []
    for scope_key, plan_versions in scopes.items():
        candidates = []
        failures = []
        for plan in plan_versions:
            job_epochs = [(group_key, pages) for group_key, pages in page_groups.items()
                          if group_key[:2] == scope_key and group_key[2] == plan["jobId"]]
            if not job_epochs:
                failures.append({"jobId": plan["jobId"], "epochId": plan["epochId"], "state": plan["state"],
                                 "reason": "no_response_pages"})
                continue
            for group_key, pages in job_epochs:
                epoch_plan = {**plan, "epochId": group_key[3]}
                valid, reason = _valid_epoch(pages, plan)
                if valid:
                    candidates.append({"plan": epoch_plan, "pages": pages,
                                       "retrievedAt": max(p["retrievedAt"] for p in pages),
                                       "lastRequestId": max(p["requestId"] for p in pages)})
                else:
                    failures.append({"jobId": plan["jobId"], "epochId": group_key[3],
                                     "state": plan["state"], "reason": reason})
        if candidates:
            candidates.sort(key=lambda item: (item["retrievedAt"], item["lastRequestId"]))
            selected = candidates[-1]
            latest_plan = max(plan_versions, key=lambda item: (item["jobStartedAt"], item["jobId"]))
            selected["fallback"] = not (latest_plan["state"] in ("complete", "empty")
                                         and latest_plan["jobId"] == selected["plan"]["jobId"]
                                         and latest_plan["epochId"] == selected["plan"]["epochId"])
            chosen[scope_key] = selected
            selected_total = selected["pages"][0]["expectedCount"]
            state = "empty" if selected_total == 0 else "complete"
        else:
            latest_plan = max(plan_versions, key=lambda item: (item["jobStartedAt"], item["jobId"]))
            state = latest_plan["state"]
        plan = latest_plan if scope_key not in chosen else chosen[scope_key]["plan"]
        scope_report.append({"station": plan["station"], "start": plan["start"].isoformat(),
                             "end": plan["end"].isoformat(), "state": state,
                             "selectedJobId": chosen[scope_key]["plan"]["jobId"] if scope_key in chosen else None,
                             "selectedEpochId": chosen[scope_key]["plan"]["epochId"] if scope_key in chosen else None,
                             "selectedRequestIds": sorted(p["requestId"] for p in chosen[scope_key]["pages"]) if scope_key in chosen else [],
                             "usedPreviousCompleteEpoch": bool(chosen.get(scope_key, {}).get("fallback")),
                             "excludedEpochs": failures})
    return chosen, {"queryScopes": sorted(scope_report, key=lambda row: (row["station"], row["start"])),
                    "plannedQueryCount": len(scopes),
                    "selectedQueryCount": len(chosen),
                    "unselectedQueryCount": len(scopes) - len(chosen),
                    "fallbackQueryCount": sum(bool(item.get("usedPreviousCompleteEpoch")) for item in scope_report)}


def _weather_records(db, selected: dict[tuple[str, str], dict[str, Any]],
                     start: dt.date, end: dt.date) -> tuple[dict[str, dict[dt.date, dict[str, Any]]], dict[str, Any]]:
    stations: dict[str, dict[dt.date, dict[str, Any]]] = {station: {} for station in STATIONS}
    provenance: dict[str, dict[dt.date, dict[str, Any]]] = {station: {} for station in STATIONS}
    conflicts: list[dict[str, Any]] = []
    source_request_rows = []
    retrieved_values, published_unknown = [], 0
    for chosen in selected.values():
        for page in chosen["pages"]:
            source_request_rows.append(page)
            retrieved_values.append(page["retrievedAt"])
            if page["publishedAt"] is None or page["versionPublishedAt"] is None:
                published_unknown += 1
    request_ids = sorted({row["requestId"] for row in source_request_rows})
    if not request_ids:
        return stations, {"records": 0, "duplicateRows": 0, "conflictRows": 0}
    # Weather windows are a small, bounded extract; payload rows are streamed in provenance order.
    marks = ",".join("?" for _ in request_ids)
    sql = f"""SELECT r.request_id,r.job_id,r.epoch_id,r.retrieved_at,r.response_sha256,r.published_at,
                      n.row_no,n.reference_date,n.payload_json
               FROM source_requests r JOIN normalized_records n USING(request_id)
               WHERE r.request_id IN ({marks}) ORDER BY n.reference_date,r.retrieved_at,r.request_id,n.row_no"""
    page_lookup = {page["requestId"]: page for page in source_request_rows}
    duplicate_rows = 0
    record_count = 0
    for row in db.execute(sql, request_ids):
        try:
            payload = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"invalid weather JSON in request {row['request_id']}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"weather row is not an object in request {row['request_id']}")
        station = str(payload.get("stnId", ""))
        if station not in STATIONS:
            raise ValueError(f"weather row has unknown station in request {row['request_id']}")
        day = _date(payload.get("tm"), "weather tm")
        if row["reference_date"] not in (None, "") and _date(str(row["reference_date"]), "weather reference_date") != day:
            raise ValueError(f"weather source date and normalized reference date differ in request {row['request_id']}")
        page = page_lookup[row["request_id"]]
        selected_plan = next(plan for key, item in selected.items() if page in item["pages"]
                             for plan in (item["plan"],))
        if station != selected_plan["station"] or not selected_plan["start"] <= day <= selected_plan["end"]:
            raise ValueError(f"weather source station/date does not match query in request {row['request_id']}")
        if not start <= day <= end:
            continue
        record_count += 1
        values = {field: _number(payload.get(field)) for field in ("avgTa", "maxTa", "minTa", "sumRn")}
        normalized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        candidate = {"values": values, "payload": normalized_payload,
                     "retrievedAt": row["retrieved_at"], "requestId": int(row["request_id"]),
                     "jobId": int(row["job_id"]), "epochId": row["epoch_id"],
                     "responseSha256": row["response_sha256"], "rowNo": int(row["row_no"]),
                     "publishedAt": row["published_at"]}
        old = stations[station].get(day)
        if old is None:
            stations[station][day] = values
            provenance[station][day] = candidate
            continue
        duplicate_rows += 1
        old_provenance = provenance[station][day]
        if candidate["payload"] == old_provenance["payload"]:
            if (candidate["retrievedAt"], candidate["requestId"]) > (old_provenance["retrievedAt"], old_provenance["requestId"]):
                provenance[station][day] = candidate
            continue
        if (candidate["retrievedAt"], candidate["requestId"]) == (old_provenance["retrievedAt"], old_provenance["requestId"]):
            raise ValueError(f"conflicting weather values with identical priority for station {station} date {day}")
        winner, discarded = ((candidate, old_provenance) if
                             (candidate["retrievedAt"], candidate["requestId"]) >
                             (old_provenance["retrievedAt"], old_provenance["requestId"])
                             else (old_provenance, candidate))
        stations[station][day] = winner["values"]
        provenance[station][day] = winner
        conflicts.append({"station": station, "date": day.isoformat(),
                          "winnerRequestId": winner["requestId"],
                          "discardedRequestId": discarded["requestId"],
                          "winnerRetrievedAt": winner["retrievedAt"]})
    return stations, {"recordCount": record_count, "uniqueStationDates": sum(len(rows) for rows in stations.values()),
                      "duplicateRows": duplicate_rows, "conflictRows": len(conflicts), "conflicts": conflicts,
                      "selectedRequestIds": request_ids,
                      "responseHashes": [{"requestId": row["requestId"], "sha256": row["responseSha256"]}
                                         for row in sorted(source_request_rows, key=lambda item: item["requestId"])],
                      "retrievedAtStart": min(retrieved_values) if retrieved_values else None,
                      "retrievedAtEnd": max(retrieved_values) if retrieved_values else None,
                      "publicationTimeUnknownRequestCount": published_unknown}


def _price_series(product: dict[str, Any]) -> dict[dt.date, dict[str, Any]]:
    result = {}
    for row in product.get("observations", []):
        day = _date(row.get("date"), "price date")
        price, volume = _number(row.get("actual")), _number(row.get("volumeKg"))
        if price is None or price <= 0 or volume is None or volume <= 0:
            raise ValueError(f"invalid price-panel observation for {product.get('id')} on {day}")
        if day in result:
            raise ValueError(f"duplicate price-panel observation for {product.get('id')} on {day}")
        result[day] = {**row, "actual": price, "volumeKg": volume, "rowCount": int(row.get("rowCount", 0))}
    return dict(sorted(result.items()))


def _abs_log_pct_change(current: float, previous: float) -> float:
    return 100.0 * abs(math.log(current / previous))


def _consecutive_changes(series: dict[dt.date, dict[str, Any]], start: dt.date, end: dt.date) -> list[dict[str, Any]]:
    result = []
    for day, current in series.items():
        if not start <= day <= end:
            continue
        prior = series.get(day - dt.timedelta(days=1))
        if prior is None:
            continue
        price_change = _abs_log_pct_change(current["actual"], prior["actual"])
        volume_change = _abs_log_pct_change(current["volumeKg"], prior["volumeKg"])
        result.append({"date": day, "priceChangeAbsLogPct": price_change,
                       "volumeChangeAbsLogPct": volume_change,
                       "price": current["actual"], "volumeKg": current["volumeKg"],
                       "previousPrice": prior["actual"], "previousVolumeKg": prior["volumeKg"]})
    return result


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def _period_summary(series: dict[dt.date, dict[str, Any]], start: dt.date, end: dt.date) -> dict[str, Any]:
    rows = {day: row for day, row in series.items() if start <= day <= end}
    changes = _consecutive_changes(series, start, end)
    context_changes = _consecutive_changes(series, min(series, default=start), max(series, default=end))
    month_groups: dict[str, list[tuple[dt.date, dict[str, Any]]]] = defaultdict(list)
    season_groups: dict[str, list[tuple[dt.date, dict[str, Any]]]] = defaultdict(list)
    for day, row in series.items():
        month_groups[day.strftime("%Y-%m")].append((day, row))
        season = next(label for label, months in SEASONS.items() if day.month in months)
        season_groups[season].append((day, row))
    eval_change_by_day = {row["date"]: row["priceChangeAbsLogPct"] for row in changes}
    context_change_by_day = {row["date"]: row["priceChangeAbsLogPct"] for row in context_changes}

    def summarize(group: list[tuple[dt.date, dict[str, Any]]]) -> dict[str, Any]:
        days = {day for day, _ in group}
        group_changes = [value for day, value in context_change_by_day.items() if day in days]
        price_values = [row["actual"] for _, row in group]
        return {"observedDays": len(group), "rowCount": sum(row["rowCount"] for _, row in group),
                "auctionVolumeKg": sum(row["volumeKg"] for _, row in group),
                "meanPrice": statistics.fmean(price_values) if price_values else None,
                "priceStdDev": statistics.pstdev(price_values) if len(price_values) > 1 else (0.0 if price_values else None),
                "meanAbsLogPriceChangePct": statistics.fmean(group_changes) if group_changes else None,
                "priceChangePairs": len(group_changes)}

    months = {key: summarize(value) for key, value in sorted(month_groups.items())}
    seasons = {key: summarize(value) for key, value in sorted(season_groups.items())}
    periods = {}
    for label, a, b in (("2014-2016", CONTEXT_START, dt.date(2016, 12, 31)),
                        ("2017-2018", EVALUATION_START, EVALUATION_END)):
        period_rows = [(day, row) for day, row in series.items() if a <= day <= b]
        periods[label] = summarize(period_rows)
    shock_n = sum(item["priceChangeAbsLogPct"] > 20.0 for item in changes)
    return {"context": {"observedDays": len(series), "firstObservedDate": min(series).isoformat() if series else None,
                        "lastObservedDate": max(series).isoformat() if series else None,
                        "rowCount": sum(row["rowCount"] for row in series.values()),
                        "auctionVolumeKg": sum(row["volumeKg"] for row in series.values())},
            "evaluation": {"observedDays": len(rows), "firstObservedDate": min(rows).isoformat() if rows else None,
                           "lastObservedDate": max(rows).isoformat() if rows else None,
                           "rowCount": sum(row["rowCount"] for row in rows.values()),
                           "auctionVolumeKg": sum(row["volumeKg"] for row in rows.values()),
                           "consecutiveCalendarPriceChangePairs": len(changes),
                           "fixedAbsLogChangeOver20PctCount": shock_n,
                           "fixedAbsLogChangeOver20PctRate": shock_n / len(changes) if changes else None},
            "periodComparison": periods, "byYearMonth": months, "bySeason": seasons}


def _onion_error_diagnostics(source: dict[str, Any], source_hash: str,
                             significance_report: dict[str, Any] | None) -> dict[str, Any]:
    products, _, eval_start, eval_end = significance._validate_input(source)
    output = {"evaluationStart": eval_start.isoformat(), "evaluationEnd": eval_end.isoformat(),
              "primaryDailyHorizon1": [], "significanceLink": {"status": "not_provided"}}
    if significance_report is not None:
        protocol = significance_report.get("protocol", {})
        if protocol.get("inputSha256") == source_hash:
            if (protocol.get("evaluationStart") == eval_start.isoformat()
                    and protocol.get("evaluationEnd") == eval_end.isoformat()):
                output["significanceLink"] = {"status": "matched_input_hash_and_period",
                                               "inputSha256": source_hash}
            else:
                output["significanceLink"] = {"status": "period_mismatch",
                                               "inputSha256": source_hash,
                                               "reportEvaluationStart": protocol.get("evaluationStart"),
                                               "reportEvaluationEnd": protocol.get("evaluationEnd")}
        else:
            output["significanceLink"] = {"status": "input_hash_mismatch",
                                           "reportInputSha256": protocol.get("inputSha256"),
                                           "predictionsSha256": source_hash}
    for product_id, product in products.items():
        if product_id != "onion":
            continue
        actuals = {row["date"]: float(row["actual"]) for row in product["observations"]}
        baseline = {(row["target"], int(row["horizon"])): row for row in product["predictions"]}
        for run in product["runs"]:
            if run.get("cadence") != "daily":
                continue
            model_id = run["modelId"]
            predictions = {(row["target"], int(row["horizon"])): row for row in run["predictions"]}
            pairs = []
            for key in sorted(set(baseline) & set(predictions)):
                target, horizon = key
                if horizon != 1 or target not in actuals:
                    continue
                base, model, actual = (float(baseline[key]["predicted"]),
                                       float(predictions[key]["predicted"]), actuals[target])
                delta, movement = model - base, actual - base
                model_error, base_error = abs(model - actual), abs(base - actual)
                pairs.append({"target": target, "baselinePrediction": base, "modelPrediction": model,
                              "actual": actual, "predictionDelta": delta, "actualMovement": movement,
                              "baselineAbsoluteError": base_error, "modelAbsoluteError": model_error,
                              "mseDifferenceContribution": delta * delta - 2.0 * delta * movement,
                              "classification": "zero_delta" if delta == 0 else "zero_movement" if movement == 0
                                  else "wrong_direction" if delta * movement < 0
                                  else "harmful_overshoot" if abs(delta) > 2.0 * abs(movement)
                                  else "same_direction_nonharmful"})
            if pairs:
                terms = Counter(row["classification"] for row in pairs)
                output["primaryDailyHorizon1"].append({"modelId": model_id, "matchedPairs": len(pairs),
                    "matchedDates": [row["target"] for row in pairs],
                    "meanMseDifferenceContribution": statistics.fmean(row["mseDifferenceContribution"] for row in pairs),
                    "observedMseDifference": statistics.fmean(row["modelAbsoluteError"] ** 2 -
                                                                 row["baselineAbsoluteError"] ** 2 for row in pairs),
                    "identityDifference": statistics.fmean(row["mseDifferenceContribution"] for row in pairs) -
                        statistics.fmean(row["modelAbsoluteError"] ** 2 - row["baselineAbsoluteError"] ** 2 for row in pairs),
                    "classifications": dict(sorted(terms.items()),), "pairs": pairs})
    return output


def build_report(db, prices_source: dict[str, Any], predictions_source: dict[str, Any],
                 prices_hash: str, predictions_hash: str,
                 significance_report: dict[str, Any] | None = None) -> dict[str, Any]:
    panel = experiments.build_price_panel(db, CONTEXT_START, CONTEXT_END)
    target_guard = experiments.verify_target_equivalence(panel, prices_source, CONTEXT_START, CONTEXT_END)
    panel_products = {item["id"]: item for item in panel["products"]}
    price_series = {product: _price_series(panel_products[product]) for product in INVENTORY_PRODUCTS}
    product_map, _, evaluation_start, evaluation_end = significance._validate_input(predictions_source)
    if evaluation_start != EVALUATION_START or evaluation_end != EVALUATION_END:
        raise ValueError("prediction evaluation period differs from fixed diagnostic period")
    # Ensure the model export uses the same target context and daily values as the DB panel.
    for product_id in INVENTORY_PRODUCTS:
        exported = _price_series(product_map[product_id])
        if set(price_series[product_id]) != set(exported):
            raise ValueError(f"prediction export target dates differ from DB panel for {product_id}")
        for day, row in price_series[product_id].items():
            if (not math.isclose(row["actual"], exported[day]["actual"], rel_tol=1e-10, abs_tol=1e-7)
                    or not math.isclose(row["volumeKg"], exported[day]["volumeKg"], rel_tol=1e-10, abs_tol=1e-7)):
                raise ValueError(f"prediction export target differs from DB panel for {product_id} on {day}")
    selected, query_report = _weather_plans(db, CONTEXT_START, CONTEXT_END)
    weather, weather_provenance = _weather_records(db, selected, CONTEXT_START, CONTEXT_END)
    associations = _run_associations(price_series, weather)
    products = {}
    for product_id in INVENTORY_PRODUCTS:
        products[product_id] = {"name": panel_products[product_id].get("name", product_id),
                                "inventory": _period_summary(price_series[product_id], EVALUATION_START, EVALUATION_END),
                                "composition": _composition_diagnostics(price_series[product_id], EVALUATION_START, EVALUATION_END)}
    return {"schemaVersion": 1, "generatedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "protocol": {"contextStart": CONTEXT_START.isoformat(), "contextEnd": CONTEXT_END.isoformat(),
                         "evaluationStart": EVALUATION_START.isoformat(), "evaluationEnd": EVALUATION_END.isoformat(),
                         "pricesSha256": prices_hash, "predictionsSha256": predictions_hash,
                         "significanceSha256": None if significance_report is None else _hash_json(significance_report),
                         "targetDefinition": "가락·강서 경매 kg; 부호 보존 총액/총중량; 최신 검증 vintage, 최초 공개시각 unknown",
                         "weatherRegressionFamilySize": FAMILY_SIZE, "fixedHolmFamilySize": FAMILY_SIZE,
                         "alpha": ALPHA, "primaryCalendarBartlettLagDays": 9,
                         "sensitivityCalendarBartlettLagDays": 28,
                         "olsGate": "rank full and n >= max(60, 5*k); unavailable tests remain in Holm family with p=1",
                         "publicationTimeKnown": False,
                         "limitations": ["회고적 연관이며 인과관계를 나타내지 않습니다.",
                                        "현재 수정 vintage이며 최초 공개시점은 확인되지 않았습니다.",
                                        "결측 날짜는 보간하지 않았고 날씨 강수 결측은 창 전체를 제외했습니다.",
                                        "HAC 정규근사는 점근적이며 최소 표본수 조건이 보장을 뜻하지 않습니다."]},
            "targetEquivalence": target_guard,
            "pricePanelProvenance": {key: panel.get(key) for key in ("selectedJobIds", "selectedRequestCount", "sourceUrls", "panelStart", "panelEnd")},
            "weatherCoverage": _weather_field_coverage(weather, CONTEXT_START, CONTEXT_END),
            "weatherQuerySelection": query_report, "weatherProvenance": weather_provenance,
            "associations": associations, "products": products,
            "onionPredictionError": _onion_error_diagnostics(predictions_source, predictions_hash, significance_report)}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Historical diagnostics", "", f"Evaluation: {report['protocol']['evaluationStart']} to {report['protocol']['evaluationEnd']}",
             "", "## Fixed association family", "", "| Test | n | Primary p | Holm p | Sensitivity p | Status |", "|---|---:|---:|---:|---:|---|"]
    for row in report["associations"]:
        primary, sensitive = row["primary"], row["sensitivity"]
        def fmt(obj, field):
            value = obj.get(field)
            return f"{value:.4g}" if obj.get("status") == "available" and value is not None else "—"
        lines.append(f"| {row['testId']} | {primary['n']} | {fmt(primary, 'pValue')} | {fmt(primary, 'holmAdjustedPValue')} | {fmt(sensitive, 'pValue')} / {fmt(sensitive, 'holmAdjustedPValue')} | {primary['status']} |")
    lines += ["", "## Onion daily h=1 paired diagnostics", "", "| Model | n | Mean ΔMSE | Harmful overshoot | Wrong direction |", "|---|---:|---:|---:|---:|"]
    for row in report["onionPredictionError"]["primaryDailyHorizon1"]:
        counts = row["classifications"]
        lines.append(f"| {row['modelId']} | {row['matchedPairs']} | {row['meanMseDifferenceContribution']:.6g} | {counts.get('harmful_overshoot', 0)} | {counts.get('wrong_direction', 0)} |")
    lines.append("")
    return "\n".join(lines)


def _atomic_new_file(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--prices", default="outputs/prices-expanded-20260929.json")
    parser.add_argument("--predictions", default="outputs/dashboard-expanded-20260929.json")
    parser.add_argument("--significance")
    parser.add_argument("--output", default="outputs/diagnostics-20260929.json")
    parser.add_argument("--markdown", default="outputs/diagnostics-20260929.md")
    args = parser.parse_args(argv)
    paths = [Path(args.output), Path(args.markdown)]
    if Path(args.db).resolve() in {path.resolve() for path in paths}:
        parser.error("output cannot replace the database")
    if any(path.exists() for path in paths):
        parser.error("refusing to overwrite an existing diagnostics output")
    try:
        price_bytes = Path(args.prices).read_bytes()
        prediction_bytes = Path(args.predictions).read_bytes()
        prices = json.loads(price_bytes)
        predictions = json.loads(prediction_bytes)
        significance_obj = None
        if args.significance:
            significance_obj = json.loads(Path(args.significance).read_bytes())
        db = coverage.open_readonly(args.db)
        try:
            db.execute("BEGIN")
            report = build_report(db, prices, predictions, hashlib.sha256(price_bytes).hexdigest(),
                                  hashlib.sha256(prediction_bytes).hexdigest(), significance_obj)
            db.rollback()
        finally:
            db.close()
        json_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        _atomic_new_file(paths[0], json_bytes)
        if args.markdown:
            _atomic_new_file(paths[1], markdown_report(report).encode())
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        print(f"diagnostics failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _market_reconciliation(observation: dict[str, Any]) -> dict[str, Any]:
    market_data = observation.get("marketData") or {}
    known_volume, known_amount = 0.0, 0.0
    for market in market_data.values():
        volume = _number(market.get("volumeKg"))
        price = _number(market.get("price"))
        if volume is None or price is None:
            continue
        known_volume += volume
        known_amount += price * volume
    overall_volume = float(observation["volumeKg"])
    overall_amount = float(observation["actual"]) * overall_volume
    delta_volume = overall_volume - known_volume
    delta_amount = overall_amount - known_amount
    return {"date": observation["date"], "overallVolumeKg": overall_volume,
            "marketDataVolumeKg": known_volume, "unrepresentedVolumeKg": delta_volume,
            "unrepresentedVolumeShare": delta_volume / overall_volume if overall_volume > 0 else None,
            "overallAmountApprox": overall_amount, "marketDataAmountApprox": known_amount,
            "amountDifferenceApprox": delta_amount,
            "matchesWithinFloatTolerance": math.isclose(overall_volume, known_volume, rel_tol=1e-8, abs_tol=1e-7)
                and math.isclose(overall_amount, known_amount, rel_tol=1e-8, abs_tol=1e-6)}


def _shares(observation: dict[str, Any], dimension: str) -> tuple[dict[str, float] | None, bool, float | None]:
    total = _number(observation.get("volumeKg"))
    if total is None or total <= 0:
        return None, False, None
    market_data = observation.get("marketData") or {}
    shares: dict[str, float] = defaultdict(float)
    if dimension == "market":
        for code, values in market_data.items():
            volume = _number(values.get("volumeKg"))
            if volume is not None:
                shares[str(code)] += volume / total
    else:
        field = COMPOSITION_FIELDS[dimension]
        for values in market_data.values():
            volume = _number(values.get("volumeKg"))
            if volume is None:
                continue
            field_values = (values.get("compositionFieldShares") or {}).get(field) or {}
            for code, share_raw in field_values.items():
                share = _number(share_raw)
                if share is not None:
                    shares[str(code)] += volume * share / total
    known_share = sum(shares.values())
    missing_share = 1.0 - known_share
    shares[TVD_MISSING] = missing_share
    invalid = any(value < -1e-9 for value in shares.values()) or known_share > 1.0 + 1e-8
    return dict(shares), invalid, missing_share


def _composition_diagnostics(series: dict[dt.date, dict[str, Any]],
                             start: dt.date, end: dt.date) -> dict[str, Any]:
    evaluation_rows = {day: row for day, row in series.items() if start <= day <= end}
    day_gaps = Counter()
    ordered = sorted(evaluation_rows)
    for prior, current in zip(ordered, ordered[1:]):
        gap = (current - prior).days
        day_gaps["1" if gap == 1 else "2-7" if gap <= 7 else ">7"] += 1
    reconciliations = {day: _market_reconciliation(row) for day, row in evaluation_rows.items()}
    dimensions = {}
    gap_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prior, current in zip(ordered, ordered[1:]):
        gap = (current - prior).days
        label = "1" if gap == 1 else "2-7" if gap <= 7 else ">7"
        left, right = evaluation_rows[prior], evaluation_rows[current]
        pair = {"from": prior.isoformat(), "to": current.isoformat(), "calendarGapDays": gap,
                "endpointAbsLogPriceChangePct": _abs_log_pct_change(right["actual"], left["actual"]),
                "fromRowCount": left["rowCount"], "toRowCount": right["rowCount"],
                "fromAuctionVolumeKg": left["volumeKg"], "toAuctionVolumeKg": right["volumeKg"],
                "compositionTvd": {}, "compositionExclusionReasons": {}, "reconciliationValid":
                    reconciliations[prior]["matchesWithinFloatTolerance"] and
                    reconciliations[current]["matchesWithinFloatTolerance"]}
        for dimension in COMPOSITION_FIELDS:
            prior_share, prior_invalid, _ = _shares(left, dimension)
            current_share, current_invalid, _ = _shares(right, dimension)
            if not pair["reconciliationValid"]:
                pair["compositionTvd"][dimension] = None
                pair["compositionExclusionReasons"][dimension] = "market_reconciliation_mismatch"
            elif prior_share is None or current_share is None:
                pair["compositionTvd"][dimension] = None
                pair["compositionExclusionReasons"][dimension] = "nonpositive_or_missing_volume"
            elif prior_invalid or current_invalid:
                pair["compositionTvd"][dimension] = None
                pair["compositionExclusionReasons"][dimension] = "negative_or_overfull_share"
            else:
                keys = set(prior_share) | set(current_share)
                pair["compositionTvd"][dimension] = 0.5 * sum(
                    abs(current_share.get(key, 0.0) - prior_share.get(key, 0.0)) for key in keys)
        gap_pairs[label].append(pair)
    for dimension in COMPOSITION_FIELDS:
        transitions, invalid_transitions, missing_mass, neg_share_days = [], 0, [], 0
        for pair in sum((values for values in gap_pairs.values()), []):
            if not pair["reconciliationValid"]:
                invalid_transitions += 1
                continue
            tvd = pair["compositionTvd"].get(dimension)
            if tvd is None:
                invalid_transitions += 1
                if pair["compositionExclusionReasons"].get(dimension) == "negative_or_overfull_share":
                    neg_share_days += 1
                continue
            transitions.append(tvd)
            for day in (dt.date.fromisoformat(pair["from"]), dt.date.fromisoformat(pair["to"])):
                shares, invalid, missing = _shares(evaluation_rows[day], dimension)
                if invalid:
                    neg_share_days += 1
                elif missing is not None:
                    missing_mass.append(missing)
        dimensions[dimension] = {"transitionCount": len(transitions),
                                 "invalidTransitionCount": invalid_transitions,
                                 "negativeShareTransitionCount": neg_share_days,
                                 "totalVariationDistance": _distribution(transitions),
                                 "meanUnaccountedShare": statistics.fmean(missing_mass) if missing_mass else None,
                                 "renormalized": False,
                                 "interpretation": "Transition-level TVD including explicit unaccounted mass; reconciliation failures and negative/overfull shares are excluded, not renormalized."}
    mismatch = sum(not row["matchesWithinFloatTolerance"] for row in reconciliations.values())
    return {"priceObservationTransitionsByCalendarGap": dict(sorted(day_gaps.items())),
            "jointPairDiagnosticsByCalendarGap": {label: {
                "pairCount": len(pairs),
                "reconciliationValidCount": sum(pair["reconciliationValid"] for pair in pairs),
                "meanEndpointAbsLogPriceChangePct": _distribution([p["endpointAbsLogPriceChangePct"] for p in pairs]),
                "meanRowCountChange": _distribution([p["toRowCount"] - p["fromRowCount"] for p in pairs]),
                "meanAuctionVolumeChangeKg": _distribution([p["toAuctionVolumeKg"] - p["fromAuctionVolumeKg"] for p in pairs]),
                "meanTvdByDimension": {dimension: _distribution([p["compositionTvd"][dimension]
                    for p in pairs if p["compositionTvd"].get(dimension) is not None])
                    for dimension in COMPOSITION_FIELDS},
                "pairs": pairs} for label, pairs in sorted(gap_pairs.items())},
            "compositionShareChanges": dimensions,
            "marketDataReconciliation": {"n": len(reconciliations), "mismatchCount": mismatch,
                                         "maxAbsVolumeDifferenceKg": max((abs(r["unrepresentedVolumeKg"]) for r in reconciliations.values()), default=None),
                                         "maxAbsAmountDifferenceApprox": max((abs(r["amountDifferenceApprox"]) for r in reconciliations.values()), default=None),
                                         "unrepresentedVolumeShareMean": statistics.fmean(
                                         [r["unrepresentedVolumeShare"] for r in reconciliations.values()
                                              if r["unrepresentedVolumeShare"] is not None]) if reconciliations else None}}


def _weather_field_coverage(weather: dict[str, dict[dt.date, dict[str, Any]]],
                            start: dt.date, end: dt.date) -> dict[str, Any]:
    output = {}
    expected_days = (end - start).days + 1
    for station in STATIONS:
        dates = weather[station]
        fields = {}
        for field in ("avgTa", "maxTa", "minTa", "sumRn"):
            valid = sum(_number(row.get(field)) is not None for row in dates.values())
            fields[field] = {"validValues": valid, "blankOrInvalidValues": len(dates) - valid}
        output[station] = {"availableStationDays": len(dates), "firstDate": min(dates).isoformat() if dates else None,
                           "lastDate": max(dates).isoformat() if dates else None,
                           "missingStationDaysWithinContext": expected_days - sum(start <= day <= end for day in dates),
                           "fields": fields}
    return output


def _weather_window_value(weather: dict[dt.date, dict[str, Any]], target: dt.date,
                          feature: str) -> tuple[float | None, str]:
    dates = [target - dt.timedelta(days=offset) for offset in range(7, 0, -1)]
    if any(day not in weather for day in dates):
        return None, "missing_calendar_day"
    values = [weather[day] for day in dates]
    fields = WEATHER_FIELDS[feature]
    if any(any(_number(row.get(field)) is None for field in fields) for row in values):
        return None, "blank_or_invalid_field"
    if feature == "avgTa_7day_mean":
        return statistics.fmean(float(row["avgTa"]) for row in values), "complete"
    if feature == "daily_range_7day_mean":
        ranges = [float(row["maxTa"]) - float(row["minTa"]) for row in values]
        if any(value < 0 for value in ranges):
            return None, "invalid_max_below_min"
        return statistics.fmean(ranges), "complete"
    rainfall = [float(row["sumRn"]) for row in values]
    if any(value < 0 for value in rainfall):
        return None, "negative_rainfall"
    return sum(rainfall), "complete"


def _design_matrix(samples: list[dict[str, Any]], np):
    months = sorted({sample["date"].month for sample in samples})
    years = sorted({sample["date"].year for sample in samples})
    weekdays = sorted({sample["date"].weekday() for sample in samples})
    names = ["intercept"]
    columns = [[1.0] * len(samples)]
    for field, levels in (("month", months), ("year", years), ("weekday", weekdays)):
        for level in levels[1:]:
            names.append(f"{field}_{level:02d}")
            columns.append([float((sample["date"].month if field == "month" else
                                   sample["date"].year if field == "year" else
                                   sample["date"].weekday()) == level) for sample in samples])
    names.append("feature")
    columns.append([sample["x"] for sample in samples])
    return np.column_stack(columns).astype(np.float64), names


def _regression(samples: list[dict[str, Any]], max_lag: int) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("diagnostic OLS requires numpy; install the existing modeling dependencies") from exc
    n = len(samples)
    if n == 0:
        return {"status": "unavailable_no_pairs", "n": 0, "parameterCount": None, "rank": None,
                "coefficient": None, "standardError": None, "statistic": None, "pValue": None,
                "maxLagDays": max_lag}
    matrix, names = _design_matrix(samples, np)
    y = np.asarray([sample["y"] for sample in samples], dtype=np.float64)
    k = matrix.shape[1]
    beta, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if int(rank) < k:
        return {"status": "unavailable_rank_deficient", "n": n, "parameterCount": k, "rank": int(rank),
                "coefficient": float(beta[-1]) if len(beta) else None, "standardError": None,
                "statistic": None, "pValue": None, "maxLagDays": max_lag,
                "coefficientUnit": "change in 100×absolute-log-price-change per one feature unit"}
    coefficient = float(beta[-1])
    residuals = y - matrix @ beta
    if n < max(60, 5 * k):
        return {"status": "unavailable_insufficient_n", "n": n, "parameterCount": k, "rank": int(rank),
                "minimumPairs": max(60, 5 * k), "coefficient": coefficient,
                "standardError": None, "statistic": None, "pValue": None, "maxLagDays": max_lag,
                "coefficientUnit": "change in 100×absolute-log-price-change per one feature unit"}
    bread = np.linalg.inv(matrix.T @ matrix)
    scores = matrix * residuals[:, None]
    meat = scores.T @ scores
    index = {sample["date"]: i for i, sample in enumerate(samples)}
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = np.zeros((k, k), dtype=np.float64)
        for i, sample in enumerate(samples):
            previous = index.get(sample["date"] - dt.timedelta(days=lag))
            if previous is not None:
                cross += np.outer(scores[i], scores[previous])
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    variance = float(covariance[-1, -1])
    if not math.isfinite(variance) or variance <= 0:
        return {"status": "unavailable_nonpositive_hac_variance", "n": n,
                "parameterCount": k, "rank": int(rank), "coefficient": coefficient,
                "standardError": None, "statistic": None, "pValue": None, "maxLagDays": max_lag,
                "coefficientUnit": "change in 100×absolute-log-price-change per one feature unit"}
    standard_error = math.sqrt(variance)
    statistic = coefficient / standard_error
    return {"status": "available", "n": n, "parameterCount": k, "rank": int(rank),
            "coefficient": coefficient, "standardError": standard_error, "statistic": statistic,
            "pValue": math.erfc(abs(statistic) / math.sqrt(2)), "maxLagDays": max_lag,
            "coefficientUnit": "change in 100×absolute-log-price-change per one feature unit"}


def _weather_samples(prices: dict[dt.date, dict[str, Any]], weather: dict[dt.date, dict[str, Any]],
                     feature: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    changes = _consecutive_changes(prices, EVALUATION_START, EVALUATION_END)
    samples, reasons = [], Counter()
    for row in changes:
        day = row["date"]
        x, reason = _weather_window_value(weather, day, feature)
        if x is not None:
            samples.append({"date": day, "y": row["priceChangeAbsLogPct"], "x": x})
        else:
            reasons[reason] += 1
    return samples, dict(sorted(reasons.items()))


def _volume_samples(prices: dict[dt.date, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"date": row["date"], "y": row["priceChangeAbsLogPct"],
             "x": row["volumeChangeAbsLogPct"]} for row in
            _consecutive_changes(prices, EVALUATION_START, EVALUATION_END)]


def _apply_holm_family(results: list[dict[str, Any]], key: str) -> None:
    rows = [{"pValue": item[key]["pValue"] if item[key]["pValue"] is not None else 1.0,
             "result": item} for item in results]
    if len(rows) != FAMILY_SIZE:
        raise ValueError(f"fixed association family must contain {FAMILY_SIZE} tests")
    experiments._apply_holm(rows)
    for row in rows:
        result = row["result"][key]
        result["holmAdjustedPValue"] = row["holmAdjustedPValue"]
        result["holmSignificantAt005"] = bool(result["status"] == "available"
                                              and row["holmAdjustedPValue"] < ALPHA)


def _run_associations(price_series: dict[str, dict[dt.date, dict[str, Any]]],
                      weather: dict[str, dict[dt.date, dict[str, Any]]]) -> list[dict[str, Any]]:
    results = []
    for product_id in TARGET_PRODUCTS:
        for station in STATIONS:
            for feature in WEATHER_FEATURES:
                samples, reasons = _weather_samples(price_series[product_id], weather[station], feature)
                primary = _regression(samples, 9)
                sensitivity = _regression(samples, 28)
                fieldset = WEATHER_FIELDS[feature]
                results.append({"testId": f"{product_id}.{station}.{feature}", "productId": product_id,
                                "kind": "weather", "station": station, "feature": feature,
                                "featureFields": list(fieldset), "featureUnit": "°C" if "Ta" in "".join(fieldset)
                                    else "mm / prior seven calendar days",
                                "outcome": "100×absolute-log-price-change (%)",
                                "controls": "intercept + month FE + year FE + weekday FE (drop-first over observed levels)",
                                "window": "exact calendar dates t-7..t-1; all seven rows and required values must be present",
                                "windowCounts": {"consecutivePricePairs": sum(1 for _ in _consecutive_changes(
                                    price_series[product_id], EVALUATION_START, EVALUATION_END)),
                                    "eligibleWindows": len(samples), "excludedWindowsByReason": reasons},
                                "primary": primary, "sensitivity": sensitivity})
    for product_id in TARGET_PRODUCTS:
        samples = _volume_samples(price_series[product_id])
        results.append({"testId": f"{product_id}.auction_volume_change", "productId": product_id,
                        "kind": "same_day_volume_change", "station": None,
                        "feature": "100×absolute-log-auction-volume-change (%)", "featureFields": ["unit_tot_qty"],
                        "featureUnit": "100×absolute log ratio of consecutive daily auction kg",
                        "outcome": "100×absolute-log-price-change (%)",
                        "controls": "intercept + month FE + year FE + weekday FE (drop-first over observed levels)",
                        "window": "price and signed-auction-kg observations on consecutive calendar dates only",
                        "windowCounts": {"consecutivePricePairs": len(samples), "eligibleWindows": len(samples),
                                         "excludedWindowsByReason": {}},
                        "primary": _regression(samples, 9), "sensitivity": _regression(samples, 28)})
    _apply_holm_family(results, "primary")
    _apply_holm_family(results, "sensitivity")
    return results


if __name__ == "__main__":
    raise SystemExit(main())
