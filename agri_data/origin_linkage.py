"""Read-only transaction-origin audit for three preselected price episodes."""
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
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import collector, dashboard, diagnostics, significance, spike_attribution

FROZEN_SHA256 = spike_attribution.FROZEN_SHA256
CONTEXT_START, CONTEXT_END = dt.date(2018, 12, 1), dt.date(2020, 2, 18)
WINDOW = 7
CASES = (("radish", dt.date(2019, 7, 12)), ("cabbage", dt.date(2019, 7, 27)),
         ("radish", dt.date(2019, 10, 3)))
MISSING = "(missing)"
WEATHER_STATIONS = ("100", "108", "212", "243", "251", "261")


def _station_match(code: str, raw_name: str) -> dict[str, Any] | None:
    """Conservative county-level correspondence; never infer a farm location."""
    checks = {
        "250000": ("홍천", "212", "홍천읍", 140.2),
        "579000": ("부안", "243", "부안군", None),
        "585000": ("고창", "251", "고창군", None),
    }
    if code in checks:
        token, station, station_place, elevation = checks[code]
        if token in raw_name:
            return {"stationId": station, "stationPlace": station_place,
                    "stationElevationM": elevation,
                    "mappingBasis": "conditional raw-label correspondence; 2019 origin-code validity unverified; representative station only"}
        return None
    # Station 100 is Daegwallyeong. Do not map all Pyeongchang-origin rows to it.
    if code == "232000" and "평창" in raw_name and "대관령" in raw_name:
        return {"stationId": "100", "stationPlace": "대관령",
                "stationElevationM": None,
                "mappingBasis": "conditional raw-label correspondence; 2019 origin-code validity unverified; exact Pyeongchang-Daegwallyeong label only"}
    return None


def _date(value: Any) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ISO date") from exc


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _origin(code: Any, name: Any) -> tuple[str, str, str]:
    code = str(code) if code not in (None, "") else MISSING
    name = str(name) if name not in (None, "") else MISSING
    if code == name == MISSING:
        kind = "unknown"
    elif any(token in name for token in ("주소", "번지", "도로명", "시장")):
        kind = "address_or_facility_text_unverified"
    elif name.endswith(("도", "시", "군", "구", "읍", "면", "동")):
        kind = "administrative_name_text_unverified"
    else:
        kind = "raw_place_label_unverified"
    return code, name, kind


def _frozen(source: dict[str, Any], source_bytes: bytes):
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != FROZEN_SHA256:
        raise ValueError("frozen input SHA-256 mismatch")
    products, _, start, end = significance._validate_input(source)
    if start > min(day-WINDOW*dt.timedelta(days=1) for _, day in CASES) or end < max(day+dt.timedelta(days=WINDOW) for _, day in CASES):
        raise ValueError("frozen export does not cover all event windows")
    values = {}
    for product in ("cabbage", "radish"):
        values[product] = {}
        for row in products[product]["observations"]:
            day = _date(row["date"])
            if CONTEXT_START <= day <= CONTEXT_END:
                if "rowCount" not in row:
                    raise ValueError(f"frozen export missing rowCount for {product}/{day}")
                values[product][day] = {"price": _number(row["actual"], "actual"),
                    "weightKg": _number(row["volumeKg"], "volumeKg"), "rowCount": int(row["rowCount"])}
    return digest, values, {"start": start.isoformat(), "end": end.isoformat()}


def _extract(db: sqlite3.Connection, start: dt.date, end: dt.date):
    rows = {p: defaultdict(list) for p in ("cabbage", "radish")}
    states = {p: {} for p in rows}
    requests = {p: {} for p in rows}
    query_requests = {p: defaultdict(set) for p in rows}
    excluded = {p: Counter() for p in rows}
    for rec in db.execute(dashboard.PRICE_SQL, (start.isoformat(), end.isoformat(), *dashboard.MARKETS.keys())):
        query = dashboard._json_object(rec["query_json"])
        if not query:
            continue
        product = dashboard._query_product(query)
        if product not in rows:
            continue
        day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
        market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
        # Broad market/day scope is selected once; ignore overlapping narrower queries.
        if day is None or market not in dashboard.MARKETS or query.get("cond[gds_sclsf_cd::EQ]") not in (None, ""):
            continue
        states[product][(day.isoformat(), market)] = str(rec["selected_state"] if rec["selected_state"] in ("complete", "empty") else rec["latest_state"] or "unavailable")
        request_id = rec["request_id"]
        if request_id is None:
            continue
        if rec["selected_state"] not in ("complete", "empty"):
            continue
        request_id = int(request_id)
        if request_id not in requests[product]:
            digest = db.execute("SELECT response_sha256 FROM source_requests WHERE request_id=?", (request_id,)).fetchone()
            requests[product][request_id] = {"requestId": request_id, "jobId": int(rec["selected_job_id"]),
                "epochId": str(rec["selected_epoch_id"]), "responseSha256": digest[0] if digest else None,
                "retrievedAt": str(rec["retrieved_at"]), "publishedAt": rec["published_at"],
                "versionPublishedAt": rec["version_published_at"],
                "publicationTimeUnknown": rec["published_at"] is None or rec["version_published_at"] is None}
        query_requests[product][day].add(request_id)
        if rec["payload_json"] is None:
            continue
        row = dashboard._json_object(rec["payload_json"])
        if row is None:
            excluded[product]["invalid_json"] += 1
            continue
        if dashboard._product_id(row) != product:
            excluded[product]["product_mismatch"] += 1
            continue
        if row.get("trd_se") != "경매":
            excluded[product]["not_auction"] += 1
            continue
        if row.get("unit_nm") != "kg":
            excluded[product]["unit_not_kg"] += 1
            continue
        if str(row.get("whsl_mrkt_cd", "")) != market or dashboard._parse_date(row.get("trd_clcln_ymd")) != day:
            excluded[product]["query_scope_mismatch"] += 1
            continue
        amount, weight = dashboard._as_decimal(row.get("totprc")), dashboard._as_decimal(row.get("unit_tot_qty"))
        if amount is None or weight is None:
            excluded[product]["invalid_amount_or_weight"] += 1
            continue
        code, name, label_kind = _origin(row.get("plor_cd"), row.get("plor_nm"))
        rows[product][day].append({"date": day.isoformat(), "marketCode": market,
            "marketName": str(row.get("whsl_mrkt_nm") or dashboard.MARKETS[market]),
            "originCode": code, "originNameRaw": name, "originLabelKind": label_kind,
            "subclassCode": str(row.get("gds_sclsf_cd") or MISSING),
            "subclassNameRaw": str(row.get("gds_sclsf_nm") or MISSING),
            "gradeCode": str(row.get("grd_cd") or MISSING), "gradeNameRaw": str(row.get("grd_nm") or MISSING),
            "amount": amount, "weightKg": weight, "requestId": request_id,
            "qualityFlags": json.loads(rec["quality_flags"] or "[]") if rec["quality_flags"] else []})
    return rows, states, requests, excluded, query_requests


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    amount = sum((row["amount"] for row in rows), Decimal(0))
    weight = sum((row["weightKg"] for row in rows), Decimal(0))
    price = amount / weight if weight > 0 else None
    return {"amount": float(amount), "signedWeightKg": float(weight), "rowCount": len(rows),
        "price": float(price) if price is not None else None,
        "status": "available" if price is not None and price > 0 else "unavailable_nonpositive_net"}


def _origin_groups(rows: list[dict[str, Any]]):
    groups = {}
    for row in rows:
        key = (row["originCode"], row["originNameRaw"])
        bucket = groups.setdefault(key, {"amount": Decimal(0), "weightKg": Decimal(0), "rowCount": 0,
            "labelKind": row["originLabelKind"], "markets": Counter(), "subclasses": Counter(), "grades": Counter(),
            "negativeWeightRows": 0, "zeroWeightRows": 0, "negativeAmountRows": 0})
        bucket["amount"] += row["amount"]
        bucket["weightKg"] += row["weightKg"]
        bucket["rowCount"] += 1
        bucket["markets"][row["marketCode"]] += 1
        bucket["subclasses"][row["subclassCode"]] += 1
        bucket["grades"][row["gradeCode"]] += 1
        bucket["negativeWeightRows"] += row["weightKg"] < 0
        bucket["zeroWeightRows"] += row["weightKg"] == 0
        bucket["negativeAmountRows"] += row["amount"] < 0
    return groups


def _period(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = _aggregate(rows)
    groups = _origin_groups(rows)
    denominator = Decimal(str(total["signedWeightKg"]))
    label_totals = defaultdict(lambda: {"amount": Decimal(0), "weightKg": Decimal(0), "rowCount": 0})
    for row in rows:
        bucket = label_totals[row["originLabelKind"]]
        bucket["amount"] += row["amount"]
        bucket["weightKg"] += row["weightKg"]
        bucket["rowCount"] += 1
    origins = []
    for (code, name), value in sorted(groups.items()):
        origins.append({"originCode": code, "originNameRaw": name, "labelClassification": value["labelKind"],
            "amount": float(value["amount"]), "signedWeightKg": float(value["weightKg"]),
            "signedWeightShare": float(value["weightKg"] / denominator) if denominator > 0 else None,
            "rowCount": value["rowCount"], "negativeWeightRows": value["negativeWeightRows"],
            "zeroWeightRows": value["zeroWeightRows"], "negativeAmountRows": value["negativeAmountRows"],
            "marketRowCounts": dict(sorted(value["markets"].items())),
            "subclassRowCounts": dict(sorted(value["subclasses"].items())),
            "gradeRowCounts": dict(sorted(value["grades"].items()))})
    return {**total, "originCount": len(origins),
        "unknownOrAddressLikeRows": sum(row["rowCount"] for row in origins
            if row["labelClassification"] in ("unknown", "address_or_facility_text_unverified")),
        "negativeWeightOriginCount": sum(row["signedWeightKg"] < 0 for row in origins), "origins": origins,
        "originLabelClassTotals": {kind: {"amount": float(value["amount"]), "signedWeightKg": float(value["weightKg"]),
            "signedWeightShare": float(value["weightKg"] / denominator) if denominator > 0 else None,
            "rowCount": value["rowCount"]} for kind, value in sorted(label_totals.items())},
        "negativeWeightRows": sum(row["weightKg"] < 0 for row in rows),
        "negativeWeightSignedKg": float(sum((row["weightKg"] for row in rows if row["weightKg"] < 0), Decimal(0))),
        "negativeWeightShareOfNet": float(sum((row["weightKg"] for row in rows if row["weightKg"] < 0), Decimal(0))/denominator) if denominator > 0 else None}


def _weight_contributions(before: list[dict[str, Any]], after: list[dict[str, Any]]):
    old, new = _origin_groups(before), _origin_groups(after)
    total_delta = sum((x["weightKg"] for x in new.values()), Decimal(0))-sum((x["weightKg"] for x in old.values()), Decimal(0))
    result = []
    for code, name in set(old) | set(new):
        key = (code, name)
        template = old[key] if key in old else new[key]
        empty = {"weightKg": Decimal(0), "amount": Decimal(0), "rowCount": 0,
                 "labelKind": template["labelKind"]}
        a = old[key] if key in old else empty
        b = new[key] if key in new else empty
        delta = b["weightKg"] - a["weightKg"]
        result.append({"originCode": code, "originNameRaw": name, "labelClassification": b["labelKind"],
            "beforeWeightKg": float(a["weightKg"]), "afterWeightKg": float(b["weightKg"]),
            "weightDeltaKg": float(delta), "shareOfNetWeightDelta": float(delta/total_delta) if total_delta else None,
            "beforeAmount": float(a["amount"]), "afterAmount": float(b["amount"]),
            "beforeRowCount": a["rowCount"], "afterRowCount": b["rowCount"]})
    return {"netWeightDeltaKg": float(total_delta), "origins": sorted(result,
        key=lambda r: (-abs(r["weightDeltaKg"]), r["originCode"], r["originNameRaw"]))}


def _one_day_origin_snapshot(rows: list[dict[str, Any]]):
    """Same-date origin × market signed totals, reconciled to selected whole-day totals."""
    total = _aggregate(rows)
    grouped = {}
    for row in rows:
        key = (row["originCode"], row["originNameRaw"])
        bucket = grouped.setdefault(key, {"labelClassification": row["originLabelKind"],
            "amount": Decimal(0), "signedWeightKg": Decimal(0), "rowCount": 0, "markets": {}})
        bucket["amount"] += row["amount"]
        bucket["signedWeightKg"] += row["weightKg"]
        bucket["rowCount"] += 1
        market = bucket["markets"].setdefault(row["marketCode"],
            {"amount": Decimal(0), "signedWeightKg": Decimal(0), "rowCount": 0})
        market["amount"] += row["amount"]
        market["signedWeightKg"] += row["weightKg"]
        market["rowCount"] += 1
    markets = {}
    for code, name in dashboard.MARKETS.items():
        market_rows = [row for row in rows if row["marketCode"] == code]
        market_total = _aggregate(market_rows)
        origin_amount = sum((value["markets"].get(code, {}).get("amount", Decimal(0))
                             for value in grouped.values()), Decimal(0))
        origin_weight = sum((value["markets"].get(code, {}).get("signedWeightKg", Decimal(0))
                             for value in grouped.values()), Decimal(0))
        origin_rows = sum(value["markets"].get(code, {}).get("rowCount", 0) for value in grouped.values())
        if (origin_amount != sum((row["amount"] for row in market_rows), Decimal(0))
                or origin_weight != sum((row["weightKg"] for row in market_rows), Decimal(0))
                or origin_rows != len(market_rows)):
            raise ValueError(f"origin detail does not reconcile to selected market totals: {code}")
        markets[code] = {"marketName": name, **market_total}
    market_amount = sum((sum((row["amount"] for row in rows if row["marketCode"] == code), Decimal(0))
                         for code in dashboard.MARKETS), Decimal(0))
    market_weight = sum((sum((row["weightKg"] for row in rows if row["marketCode"] == code), Decimal(0))
                         for code in dashboard.MARKETS), Decimal(0))
    if (market_amount != sum((row["amount"] for row in rows), Decimal(0))
            or market_weight != sum((row["weightKg"] for row in rows), Decimal(0))
            or sum(value["rowCount"] for value in markets.values()) != len(rows)):
        raise ValueError("market totals do not reconcile to selected whole-day total")
    origins = []
    for (code, name), value in sorted(grouped.items()):
        origin_weight = value["signedWeightKg"]
        markets_detail = {}
        for market, cell in sorted(value["markets"].items()):
            markets_detail[market] = {"amount": float(cell["amount"]),
                "signedWeightKg": float(cell["signedWeightKg"]), "rowCount": cell["rowCount"],
                "price": float(cell["amount"]/cell["signedWeightKg"]) if cell["signedWeightKg"] > 0 else None}
        origins.append({"originCode": code, "originNameRaw": name,
            "labelClassification": value["labelClassification"], "amount": float(value["amount"]),
            "signedWeightKg": float(origin_weight), "rowCount": value["rowCount"],
            "price": float(value["amount"]/origin_weight) if origin_weight > 0 else None,
            "markets": markets_detail})
    return {"aggregate": total, "marketTotals": markets, "origins": origins,
            "reconciliation": {"originToMarket": True, "marketToWholeDay": True}}


def _one_day_origin_change(previous_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]],
                           previous_day: dt.date, event_day: dt.date):
    before = _one_day_origin_snapshot(previous_rows)
    after = _one_day_origin_snapshot(event_rows)
    old = {(r["originCode"], r["originNameRaw"]): r for r in before["origins"]}
    new = {(r["originCode"], r["originNameRaw"]): r for r in after["origins"]}
    changes = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        template = b or a
        before_kg, after_kg = (a["signedWeightKg"] if a else 0.0), (b["signedWeightKg"] if b else 0.0)
        market_changes = {}
        for market in dashboard.MARKETS:
            ac = (a or {}).get("markets", {}).get(market, {})
            bc = (b or {}).get("markets", {}).get(market, {})
            before_market, after_market = ac.get("signedWeightKg", 0.0), bc.get("signedWeightKg", 0.0)
            market_changes[market] = {"beforeWeightKg": before_market, "afterWeightKg": after_market,
                "weightDeltaKg": after_market-before_market,
                "beforeRowCount": ac.get("rowCount", 0), "afterRowCount": bc.get("rowCount", 0),
                "beforePrice": ac.get("price"), "afterPrice": bc.get("price")}
        changes.append({"originCode": key[0], "originNameRaw": key[1],
            "labelClassification": template["labelClassification"],
            "beforeWeightKg": before_kg, "afterWeightKg": after_kg,
            "weightDeltaKg": after_kg-before_kg,
            "beforeRowCount": a["rowCount"] if a else 0, "afterRowCount": b["rowCount"] if b else 0,
            "beforePrice": a["price"] if a else None, "afterPrice": b["price"] if b else None,
            "markets": market_changes})
    declines = sorted((r for r in changes if r["weightDeltaKg"] < 0),
                      key=lambda r: (r["weightDeltaKg"], r["originCode"], r["originNameRaw"]))
    increases = sorted((r for r in changes if r["weightDeltaKg"] > 0),
                       key=lambda r: (-r["weightDeltaKg"], r["originCode"], r["originNameRaw"]))
    return {"status": "available", "comparisonBasis": "one prior completed trading date vs one event date; equal one-day duration",
        "previousTradingDate": previous_day.isoformat(), "eventDate": event_day.isoformat(),
        "periods": {"previousTradingDay": before, "eventDay": after},
        "wholeDayWeightDeltaKg": after["aggregate"]["signedWeightKg"]-before["aggregate"]["signedWeightKg"],
        "originChanges": changes, "topDecreases": declines[:10], "topIncreases": increases[:10]}


def _cells(rows: list[dict[str, Any]]):
    grouped = {}
    for row in rows:
        key = (row["marketCode"], row["subclassCode"], row["gradeCode"], row["originCode"], row["originNameRaw"])
        cell = grouped.setdefault(key, {"amount": Decimal(0), "weightKg": Decimal(0), "rowCount": 0,
                                        "negativeAmountRows": 0, "negativeWeightRows": 0})
        cell["amount"] += row["amount"]
        cell["weightKg"] += row["weightKg"]
        cell["rowCount"] += 1
        cell["negativeAmountRows"] += row["amount"] < 0
        cell["negativeWeightRows"] += row["weightKg"] < 0
    return [{"marketCode": key[0], "subclassCode": key[1], "gradeCode": key[2],
        "originCode": key[3], "originNameRaw": key[4], "amount": float(cell["amount"]),
        "signedWeightKg": float(cell["weightKg"]), "rowCount": cell["rowCount"],
        "negativeAmountRows": cell["negativeAmountRows"], "negativeWeightRows": cell["negativeWeightRows"]}
        for key, cell in sorted(grouped.items())]


def _weather_lags(weather: dict[str, dict[dt.date, dict[str, Any]]], day: dt.date):
    output = {}
    for station, records in weather.items():
        windows = {}
        for n in (1, 3, 7):
            dates = [day-dt.timedelta(days=i) for i in range(n, 0, -1)]
            items = [records[d] for d in dates if d in records]
            temps = [x["avgTa"] for x in items if x.get("avgTa") is not None]
            rains = [x["sumRn"] for x in items if x.get("sumRn") is not None]
            windows[str(n)] = {"calendarDays": n, "temperatureValidN": len(temps),
                "meanAvgTaC": sum(temps)/len(temps) if temps else None,
                "rainValidN": len(rains), "rainExpectedN": n,
                "sumRainMm": sum(rains) if len(rains) == n else None,
                "rainStatus": "complete" if len(rains) == n else "incomplete_unknown_not_zero"}
        output[station] = {"mappingStatus": "unverified_no_origin_station_crosswalk",
                           "lagBeforeEvent": windows}
    return output


def _daily_weather(weather: dict[str, dict[dt.date, dict[str, Any]]], center: dt.date):
    dates = [center+dt.timedelta(days=i) for i in range(-WINDOW, WINDOW+1)]
    return {station: {"mappingStatus": "unverified_no_origin_station_crosswalk",
        "daily": [{"date": day.isoformat(), "values": records.get(day),
                   "status": "observed" if day in records else "missing_station_day"} for day in dates]}
        for station, records in weather.items()}


def _matched_station_weather(station: str, weather, weather_provenance, center: dt.date):
    values = weather.get(station, {})
    prov = weather_provenance.get(station, {})
    lag = _weather_lags({station: values}, center)[station]["lagBeforeEvent"]
    daily = []
    for day in (center+dt.timedelta(days=i) for i in range(-WINDOW, WINDOW+1)):
        source = prov.get(day)
        daily.append({"date": day.isoformat(), "values": values.get(day),
            "status": "observed" if day in values else "missing_station_day",
            "requestId": source.get("requestId") if source else None,
            "responseSha256": source.get("responseSha256") if source else None,
            "retrievedAt": source.get("retrievedAt") if source else None,
            "publishedAt": source.get("publishedAt") if source else None,
            "versionPublishedAt": source.get("versionPublishedAt") if source else None,
            "publicationTimeUnknown": bool(source and (source.get("publishedAt") is None or source.get("versionPublishedAt") is None)) if source else None})
    return {"lagBeforeEvent": lag, "daily": daily,
        "distanceKm": None, "stationElevationM": None,
        "stationMetadataStatus": "distance unavailable; station coordinates not compared"}


def _matched_origins(origin_names, weather, weather_provenance, center):
    matches = []
    for code, name, _label_kind in origin_names:
        mapping = _station_match(code, name)
        if mapping is None:
            continue
        matches.append({"originCode": code, "originNameRaw": name,
                    "mappingStatus": "conditional_raw_label_correspondence_code_validity_unverified_not_farm_level",
            **mapping, "stationWeather": _matched_station_weather(mapping["stationId"], weather,
                                                                      weather_provenance, center)})
    return matches


def _select_weather(db: sqlite3.Connection, start: dt.date, end: dt.date):
    """Select complete ASOS epochs for a small allowlist without changing shared policies."""
    plan_rows = db.execute("""SELECT p.job_id,p.task_key,p.source_url,p.query_json,p.state,p.row_count,p.epoch_id,
                                    j.started_at,j.state AS job_state
                             FROM planned_queries p JOIN jobs j USING(job_id)
                             WHERE p.dataset='weather' AND p.source_url=? ORDER BY p.job_id,p.task_key""",
                           (collector.ASOS,)).fetchall()
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        try:
            params = json.loads(row["query_json"])
            station = str(params["stnIds"])
            qstart = dt.datetime.strptime(str(params["startDt"]), "%Y%m%d").date()
            qend = dt.datetime.strptime(str(params["endDt"]), "%Y%m%d").date()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid weather query manifest for job {row['job_id']}") from exc
        if station not in WEATHER_STATIONS or params.get("dataCd") != "ASOS" or params.get("dateCd") != "DAY":
            continue
        if qend < start or qstart > end:
            continue
        if collector.task_key("weather", row["source_url"], {str(k): str(v) for k, v in params.items()}) != row["task_key"]:
            raise ValueError(f"weather task key mismatch for job {row['job_id']}")
        scopes[(row["source_url"], row["task_key"])].append({
            "jobId": int(row["job_id"]), "taskKey": row["task_key"], "sourceUrl": row["source_url"],
            "start": qstart, "end": qend, "station": station, "state": row["state"],
            "rowCount": row["row_count"], "epochId": row["epoch_id"],
            "jobStartedAt": row["started_at"], "jobState": row["job_state"], "query": params})

    page_groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    relevant_keys = set(scopes)
    request_rows = db.execute("""SELECT r.request_id,r.job_id,r.epoch_id,r.source_url,r.query_json,r.retrieved_at,
                                      r.response_sha256,r.http_status,r.api_status,r.expected_count,r.returned_count,
                                      r.page_no,r.state,r.published_at,r.version_published_at,
                                      (SELECT COUNT(*) FROM raw_records rr WHERE rr.request_id=r.request_id) raw_count,
                                      (SELECT COUNT(*) FROM normalized_records nn WHERE nn.request_id=r.request_id) normalized_count
                               FROM source_requests r WHERE r.dataset='weather' AND r.source_url=?
                               ORDER BY r.request_id""", (collector.ASOS,)).fetchall()
    for row in request_rows:
        scope = diagnostics._source_scope("weather", row["source_url"], row["query_json"])
        if scope is None or (row["source_url"], scope[0]) not in relevant_keys:
            continue
        scope_key = (row["source_url"], scope[0])
        try:
            params = json.loads(row["query_json"])
            page_no, page_size = int(params["pageNo"]), int(params["numOfRows"])
            total, returned = int(row["expected_count"]), int(row["returned_count"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        plans = [p for p in scopes[scope_key] if p["jobId"] == int(row["job_id"])]
        if not plans:
            continue
        plan = plans[0]
        for key in ("startDt", "endDt", "stnIds", "dataCd", "dateCd", "dataType"):
            if str(params.get(key)) != str(plan["query"].get(key)):
                raise ValueError(f"weather request scope mismatch in request {row['request_id']}")
        page = {"requestId": int(row["request_id"]), "jobId": int(row["job_id"]),
            "epochId": row["epoch_id"], "retrievedAt": row["retrieved_at"],
            "responseSha256": row["response_sha256"], "httpStatus": row["http_status"],
            "apiStatus": row["api_status"], "expectedCount": total, "returnedCount": returned,
            "pageNo": page_no, "pageSize": page_size, "state": row["state"],
            "rawCount": int(row["raw_count"]), "normalizedCount": int(row["normalized_count"]),
            "publishedAt": row["published_at"], "versionPublishedAt": row["version_published_at"]}
        page_groups[(row["source_url"], scope[0], int(row["job_id"]), str(row["epoch_id"]))].append(page)

    selected = {}
    reports = []
    for scope_key, versions in scopes.items():
        candidates, rejected = [], []
        for plan in versions:
            groups = [(key, pages) for key, pages in page_groups.items()
                      if key[:2] == scope_key and key[2] == plan["jobId"]]
            for key, pages in groups:
                valid, reason = diagnostics._valid_epoch(pages, plan)
                if valid:
                    candidates.append({"plan": {**plan, "epochId": key[3]}, "pages": pages,
                        "retrievedAt": max(p["retrievedAt"] for p in pages),
                        "lastRequestId": max(p["requestId"] for p in pages)})
                else:
                    rejected.append({"jobId": plan["jobId"], "epochId": key[3], "reason": reason})
            if not groups:
                rejected.append({"jobId": plan["jobId"], "epochId": plan["epochId"], "reason": "no_response_pages"})
        if candidates:
            candidates.sort(key=lambda x: (x["retrievedAt"], x["lastRequestId"]))
            choice = candidates[-1]
            latest_plan = max(versions, key=lambda x: (x["jobStartedAt"], x["jobId"]))
            choice["fallback"] = not (latest_plan["state"] in ("complete", "empty")
                and latest_plan["jobId"] == choice["plan"]["jobId"]
                and latest_plan["epochId"] == choice["plan"]["epochId"])
            selected[scope_key] = choice
            state = "empty" if choice["pages"][0]["expectedCount"] == 0 else "complete"
        else:
            latest_plan = max(versions, key=lambda x: (x["jobStartedAt"], x["jobId"]))
            state = latest_plan["state"]
        chosen_plan = selected.get(scope_key, {}).get("plan", latest_plan)
        reports.append({"station": chosen_plan["station"], "start": chosen_plan["start"].isoformat(),
            "end": chosen_plan["end"].isoformat(), "state": state,
            "selectedJobId": selected.get(scope_key, {}).get("plan", {}).get("jobId"),
            "selectedEpochId": selected.get(scope_key, {}).get("plan", {}).get("epochId"),
            "selectedRequestIds": sorted(p["requestId"] for p in selected.get(scope_key, {}).get("pages", [])),
            "usedPreviousCompleteEpoch": bool(selected.get(scope_key, {}).get("fallback")),
            "excludedEpochs": rejected})

    values: dict[str, dict[dt.date, dict[str, Any]]] = {s: {} for s in WEATHER_STATIONS}
    provenance: dict[str, dict[dt.date, dict[str, Any]]] = {s: {} for s in WEATHER_STATIONS}
    pages = [p for chosen in selected.values() for p in chosen["pages"]]
    request_ids = sorted({p["requestId"] for p in pages})
    if request_ids:
        marks = ",".join("?" for _ in request_ids)
        query = f"""SELECT r.request_id,r.job_id,r.epoch_id,r.retrieved_at,r.response_sha256,r.published_at,
                           r.version_published_at,n.row_no,n.reference_date,n.payload_json
                    FROM source_requests r JOIN normalized_records n USING(request_id)
                    WHERE r.request_id IN ({marks}) ORDER BY n.reference_date,r.retrieved_at,r.request_id,n.row_no"""
        page_lookup = {p["requestId"]: p for p in pages}
        plan_lookup = {p["requestId"]: chosen["plan"] for chosen in selected.values() for p in chosen["pages"]}
        for row in db.execute(query, request_ids):
            payload = json.loads(row["payload_json"])
            station, day = str(payload.get("stnId", "")), _date(payload.get("tm"))
            plan = plan_lookup[int(row["request_id"])]
            if station != plan["station"]:
                raise ValueError(f"weather station/query mismatch in request {row['request_id']}")
            if row["reference_date"] not in (None, "") and _date(row["reference_date"]) != day:
                raise ValueError(f"weather source/reference date mismatch in request {row['request_id']}")
            if not plan["start"] <= day <= plan["end"]:
                raise ValueError(f"weather date/query mismatch in request {row['request_id']}")
            if not start <= day <= end:
                continue
            parsed = {field: diagnostics._number(payload.get(field)) for field in ("avgTa", "maxTa", "minTa", "sumRn")}
            signature = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            candidate = {"values": parsed, "signature": signature, "retrievedAt": row["retrieved_at"],
                "requestId": int(row["request_id"]), "jobId": int(row["job_id"]), "epochId": row["epoch_id"],
                "responseSha256": row["response_sha256"], "publishedAt": row["published_at"],
                "versionPublishedAt": row["version_published_at"], "rowNo": int(row["row_no"])}
            old = provenance[station].get(day)
            if old and (candidate["retrievedAt"], candidate["requestId"]) == (old["retrievedAt"], old["requestId"]) and candidate["signature"] != old["signature"]:
                raise ValueError(f"conflicting weather values with identical priority: {station}/{day}")
            if old is None or (candidate["retrievedAt"], candidate["requestId"]) > (old["retrievedAt"], old["requestId"]):
                provenance[station][day] = candidate
                values[station][day] = parsed
    report = {"queryScopes": sorted(reports, key=lambda x: (x["station"], x["start"])),
        "selectedQueryCount": len(selected), "selectedRequestIds": request_ids,
        "selectedRequests": [{"requestId": p["requestId"], "jobId": p["jobId"], "epochId": p["epochId"],
            "stationId": next(chosen["plan"]["station"] for chosen in selected.values() if p in chosen["pages"]),
            "retrievedAt": p["retrievedAt"], "responseSha256": p["responseSha256"],
            "httpStatus": p["httpStatus"], "apiStatus": p["apiStatus"],
            "publishedAt": p["publishedAt"], "versionPublishedAt": p["versionPublishedAt"],
            "publicationTimeUnknown": p["publishedAt"] is None or p["versionPublishedAt"] is None}
            for p in sorted(pages, key=lambda x: x["requestId"])],
        "publicationTimeUnknownRequestCount": sum(p["publishedAt"] is None or p["versionPublishedAt"] is None for p in pages),
        "recordCount": sum(len(x) for x in values.values()),
        "fallbackQueryCount": sum(bool(x.get("fallback")) for x in selected.values())}
    return values, provenance, report


def _day_audit(product: str, day: dt.date, rows: list[dict[str, Any]], states, frozen):
    market_state = {market: states[product].get((day.isoformat(), market), "unplanned") for market in dashboard.MARKETS}
    complete = all(state in ("complete", "empty") for state in market_state.values())
    total = _aggregate(rows)
    markets = {}
    for code, name in dashboard.MARKETS.items():
        markets[code] = {"marketName": name, **_aggregate([r for r in rows if r["marketCode"] == code])}
    if not complete:
        check = {"status": "not_checked_incomplete_or_unplanned_query", "matches": None}
    elif total["status"] != "available":
        check = {"status": total["status"], "matches": None}
    elif frozen is None:
        check = {"status": "frozen_export_missing_observation", "matches": False}
    else:
        matches = math.isclose(total["price"], frozen["price"], rel_tol=1e-10, abs_tol=1e-7) and math.isclose(
            total["signedWeightKg"], frozen["weightKg"], rel_tol=1e-10, abs_tol=1e-7) and total["rowCount"] == frozen["rowCount"]
        check = {"status": "verified" if matches else "mismatch", "matches": matches,
            "frozenPrice": frozen["price"], "frozenWeightKg": frozen["weightKg"], "frozenRowCount": frozen["rowCount"],
            "priceDifference": total["price"]-frozen["price"], "weightDifferenceKg": total["signedWeightKg"]-frozen["weightKg"],
            "rowCountDifference": total["rowCount"]-frozen["rowCount"]}
    empty_all = all(value == "empty" for value in market_state.values())
    raw_status = "confirmed_empty_query" if complete and empty_all and not rows else (
        "complete_queries_no_selected_auction_kg_rows" if complete and not rows else
        "selected_rows_present" if rows else "unresolved_coverage")
    return {"date": day.isoformat(), "queryStatesByMarket": market_state,
        "coverageStatus": "complete_or_empty" if complete else "incomplete_or_unplanned",
        "rawRowsStatus": raw_status,
        "aggregate": total, "marketTotals": markets, "frozenReconciliation": check}


def build_report(db: sqlite3.Connection, source: dict[str, Any], source_bytes: bytes) -> dict[str, Any]:
    source_hash, frozen, frozen_period = _frozen(source, source_bytes)
    start = min(day-dt.timedelta(days=WINDOW) for _, day in CASES)
    end = max(day+dt.timedelta(days=WINDOW) for _, day in CASES)
    # Shared extractor guarantees same validated-epoch, signed auction/kg totals as the spike audit.
    selected = spike_attribution.extract_prices(db, start, end)
    rows_by_day, states, requests, excluded, query_requests = _extract(db, start, end)
    for product in ("cabbage", "radish"):
        frozen_days = {d: row for d, row in frozen[product].items() if start <= d <= end}
        selected_days = {d: row for d, row in selected["series"][product].items() if start <= d <= end}
        if set(frozen_days) != set(selected_days):
            raise ValueError(f"selected price dates do not match frozen archive for {product}")
        for day, row in selected_days.items():
            baseline = frozen_days[day]
            if not (math.isclose(row["actual"], baseline["price"], rel_tol=1e-10, abs_tol=1e-7)
                    and math.isclose(row["volumeKg"], baseline["weightKg"], rel_tol=1e-10, abs_tol=1e-7)
                    and row["rowCount"] == baseline["rowCount"]):
                raise ValueError(f"selected source differs from frozen target on {product}/{day}")
    weather_start, weather_end = start, end
    weather, weather_row_provenance, weather_plan_meta = _select_weather(db, weather_start, weather_end)
    events = []
    for product, center in CASES:
        ranges = {"before7": [center-dt.timedelta(days=i) for i in range(7, 0, -1)],
                  "eventDay": [center], "after7": [center+dt.timedelta(days=i) for i in range(1, 8)]}
        period_rows = {label: [row for day in dates for row in rows_by_day[product].get(day, [])]
                       for label, dates in ranges.items()}
        daily = {}
        for day in (d for days in ranges.values() for d in days):
            day_rows = rows_by_day[product].get(day, [])
            daily[day.isoformat()] = _day_audit(product, day, day_rows, states,
                                                frozen[product].get(day))
        coverage = {label: {"calendarDays": len(dates),
            "completeOrEmptyBothMarkets": sum(all(states[product].get((d.isoformat(), m)) in ("complete", "empty")
                for m in dashboard.MARKETS) for d in dates),
            "incompleteDates": [d.isoformat() for d in dates if not all(
                states[product].get((d.isoformat(), m)) in ("complete", "empty") for m in dashboard.MARKETS)]}
            for label, dates in ranges.items()}
        before, event_day, after = (_period(period_rows[x]) for x in ("before7", "eventDay", "after7"))
        weight_steps = {
            "before7_to_eventDay": _weight_contributions(period_rows["before7"], period_rows["eventDay"]),
            "eventDay_to_after7": _weight_contributions(period_rows["eventDay"], period_rows["after7"]),
            "before7_to_after7": _weight_contributions(period_rows["before7"], period_rows["after7"]),
        }
        previous_candidates = [day for day in ranges["before7"]
            if frozen[product].get(day) is not None
            and daily[day.isoformat()]["coverageStatus"] == "complete_or_empty"
            and daily[day.isoformat()]["frozenReconciliation"]["status"] == "verified"]
        event_audit = daily[center.isoformat()]
        if not previous_candidates:
            one_day_comparison = {"status": "unavailable_no_prior_complete_trade_day_with_frozen_match",
                "comparisonBasis": "one prior completed trading date vs one event date; equal one-day duration"}
        elif (event_audit["coverageStatus"] != "complete_or_empty"
              or event_audit["frozenReconciliation"]["status"] != "verified"):
            one_day_comparison = {"status": "unavailable_event_day_query_or_frozen_reconciliation_not_complete",
                "comparisonBasis": "one prior completed trading date vs one event date; equal one-day duration"}
        else:
            previous_day = max(previous_candidates)
            one_day_comparison = _one_day_origin_change(
                rows_by_day[product].get(previous_day, []), rows_by_day[product].get(center, []), previous_day, center)
        used_request_ids = sorted({request_id for day in (d for dates in ranges.values() for d in dates)
                                   for request_id in query_requests[product].get(day, set())})
        price_requests = [requests[product][rid] for rid in used_request_ids if rid in requests[product]]
        origin_names = sorted({(row["originCode"], row["originNameRaw"], row["originLabelKind"])
            for label in period_rows for row in period_rows[label]})
        limited_weather = _matched_origins(origin_names, weather, weather_row_provenance, center)
        events.append({"productId": product, "eventDate": center.isoformat(),
            "window": "before D-7..D-1; event D; after D+1..D+7",
            "coverage": coverage, "dailyReconciliation": daily,
            "periods": {"before7": before, "eventDay": event_day, "after7": after},
            "originWeightChangeContribution": weight_steps,
            "oneDayPriorTradeVsEvent": one_day_comparison,
            "topOriginWeightChanges": weight_steps["before7_to_after7"]["origins"][:10],
            "marketSubclassGradeOriginCells": {label: _cells(period_rows[label]) for label in ranges},
            "selectedPriceRequests": price_requests,
            "rowQuality": {"excluded": dict(sorted(excluded[product].items())),
                "negativeAmountRowsRetained": sum(r["amount"] < 0 for label in ranges for r in period_rows[label]),
                "negativeWeightRowsRetained": sum(r["weightKg"] < 0 for label in ranges for r in period_rows[label]),
                "zeroWeightRowsRetained": sum(r["weightKg"] == 0 for label in ranges for r in period_rows[label])},
            "weather": {"matchingStatus": "only county/code matches and explicit Pyeongchang-Daegwallyeong raw labels are linked to representative stations; no farm-level claim",
                "preEventLag1_3_7": _weather_lags(weather, center),
                "observedWeatherDailyAroundEvent": _daily_weather(weather, center),
                "limitedExactPlaceMatches": limited_weather,
                "unmappedOriginCount": len(origin_names)-len(limited_weather),
                "retrospectiveObservedWeather": True,
                "sameDayPreHarvestStatus": "unknown; harvest/dispatch time unavailable"}})
    return {"schemaVersion": 1,
        "protocol": {"frozenPredictionSha256": source_hash, "frozenPredictionEvaluationPeriod": frozen_period,
            "contextStart": start.isoformat(), "contextEnd": end.isoformat(),
            "events": [{"productId": p, "date": d.isoformat()} for p, d in CASES],
            "windowDays": WINDOW,
            "oneDayComparison": "closest earlier date with selected frozen actual and both-market complete/empty queries versus event date; each side is one calendar-day aggregate, never seven-day versus one-day",
            "priceDefinition": "aT katOrigin selected validated query rows; auction/kg; signed totprc / signed unit_tot_qty",
            "originFields": ["plor_cd", "plor_nm"], "missingLabel": MISSING,
            "labelCaveat": "source place labels do not verify the farm or actual cultivation location; current admin lookup not performed",
            "weatherMatch": "conditional raw-label correspondences only: 250000 홍천→ASOS212, 579000 부안→243, 585000 고창→251; 232000→100 only when raw label explicitly includes 평창 and 대관령. 2019 origin-code validity is unverified, so these are not confirmed source-location links. Representative stations, not farms. Other labels stay unmapped.",
            "weatherPublicationTime": "unknown; observed daily weather is retrospective/current-vintage and not forecast input",
            "weatherStations": {"212": {"stationPlace": "홍천읍", "elevationM": 140.2},
                "243": {"stationPlace": "부안군", "elevationM": None},
                "251": {"stationPlace": "고창군", "elevationM": None},
                "100": {"stationPlace": "대관령", "elevationM": None}},
            "weatherStationMetadataBasis": "KMA station metadata supplied by project owner; only the 212 elevation was provided; no distances calculated",
            "harvestOrDispatchTime": "unknown; event-day rain is not asserted to precede harvest or transport",
            "interpretation": "selected-event descriptive source audit; not causal evidence or an independent evaluation"},
        "provenance": {"databaseReadOnly": True, "priceSource": f"{collector.PORTAL}/B552845/katOrigin/trades",
            "weatherSource": collector.ASOS,
            "selection": "dashboard.PRICE_SQL/coverage policy via shared spike_attribution extractor",
            "weatherQuerySelection": weather_plan_meta,
            "publicationTimeUnknown": True}, "events": events}


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# 출하지역·기상 원자료 감사", "",
        f"Frozen archive SHA-256: `{report['protocol']['frozenPredictionSha256']}`.",
        "산지코드의 2019년 유효성은 검증되지 않았습니다. 코드와 원문 군명이 맞는 일부만 조건부 대표 관측소 대응으로 표시하며 농장·재배지 증거가 아닙니다. ASOS는 후향 일관측이고 최초 공개·수확 시각은 확인되지 않았습니다.", ""]
    for event in report["events"]:
        lines.extend([f"## {event['productId']} · {event['eventDate']}", "",
            "| 구간 | 중량 kg | 금액 | 행수 | 산지 그룹 | 음수 중량 그룹 | 음수 중량 행 | 음수 중량/net kg | 미상/주소 kg 비중 | 양시장 정상조회 일수 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for label, pretty in (("before7", "전 7일"), ("eventDay", "사건일"), ("after7", "후 7일")):
            row = event["periods"][label]
            classes = row["originLabelClassTotals"]
            unknown_address_share = sum((value["signedWeightShare"] or 0.0) for kind, value in classes.items()
                                        if kind in ("unknown", "address_or_facility_text_unverified"))
            lines.append(f"| {pretty} | {row['signedWeightKg']:.2f} | {row['amount']:.1f} | {row['rowCount']} | {row['originCount']} | "
                         f"{row['negativeWeightOriginCount']} | {row['negativeWeightRows']} | {_fmt(row['negativeWeightShareOfNet'])} | {unknown_address_share:.3f} | "
                         f"{event['coverage'][label]['completeOrEmptyBothMarkets']}/{event['coverage'][label]['calendarDays']} |")
        lines.extend(["", "| 산지코드 | 원문명 | 표기 분류 | 전 kg | 후 kg | 변화 kg | 순변화 기여 |",
                      "|---|---|---|---:|---:|---:|---:|"])
        for row in event["topOriginWeightChanges"]:
            lines.append(f"| {row['originCode']} | {row['originNameRaw']} | {row['labelClassification']} | "
                f"{row['beforeWeightKg']:.2f} | {row['afterWeightKg']:.2f} | {row['weightDeltaKg']:+.2f} | {_fmt(row['shareOfNetWeightDelta'])} |")
        one_day = event["oneDayPriorTradeVsEvent"]
        lines.extend(["", "### 직전 완료 거래일 대 사건일: 1일 대 1일", "",
            "7일 합계와 사건일 1일을 직접 비교하지 않도록, 사건일 전 가장 가까운 양시장 정상조회·동결가격 일치 거래일만 골랐습니다."])
        if one_day["status"] != "available":
            lines.extend(["", f"- 비교 불가: `{one_day['status']}`."])
        else:
            previous = one_day["periods"]["previousTradingDay"]
            current = one_day["periods"]["eventDay"]
            lines.extend(["", f"- 직전 거래일 {one_day['previousTradingDate']} → 사건일 {one_day['eventDate']}; "
                f"전체 순중량 변화 {one_day['wholeDayWeightDeltaKg']:+.2f} kg.", "",
                "| 구간 | 금액 | signed kg | 행수 | 가중 가격 원/kg |", "|---|---:|---:|---:|---:|"])
            for label, snapshot in (("직전 1일", previous), ("사건일 1일", current)):
                agg = snapshot["aggregate"]
                lines.append(f"| {label} | {agg['amount']:.1f} | {agg['signedWeightKg']:.2f} | {agg['rowCount']} | {_fmt(agg['price'])} |")
            lines.extend(["", "| 시장 | 직전 kg / 행수 / 원/kg | 사건일 kg / 행수 / 원/kg |", "|---|---:|---:|"])
            for market, market_name in dashboard.MARKETS.items():
                before_market, after_market = previous["marketTotals"][market], current["marketTotals"][market]
                lines.append(f"| {market_name} ({market}) | {before_market['signedWeightKg']:.2f} / {before_market['rowCount']} / {_fmt(before_market['price'])} | "
                    f"{after_market['signedWeightKg']:.2f} / {after_market['rowCount']} / {_fmt(after_market['price'])} |")
            for heading, rows in (("중량 감소 상위", one_day["topDecreases"][:5]),
                                  ("중량 증가 상위", one_day["topIncreases"][:5])):
                lines.extend(["", f"**{heading}**", "",
                    "| 산지 원문 라벨 | 전 kg | 사건일 kg | 변화 kg | 전/후 행수 | 가락 Δ kg | 강서 Δ kg |",
                    "|---|---:|---:|---:|---:|---:|---:|"])
                if not rows:
                    lines.append("해당 산지 변화 없음.")
                    continue
                for row in rows:
                    market_delta = row["markets"]
                    lines.append(f"| {row['originCode']} {row['originNameRaw']} ({row['labelClassification']}) | "
                        f"{row['beforeWeightKg']:.2f} | {row['afterWeightKg']:.2f} | {row['weightDeltaKg']:+.2f} | "
                        f"{row['beforeRowCount']}/{row['afterRowCount']} | "
                        f"{market_delta.get('110001',{}).get('weightDeltaKg',0.0):+.2f} | "
                        f"{market_delta.get('110008',{}).get('weightDeltaKg',0.0):+.2f} |")
            lines.extend(["", f"시장별 산지 합계와 전체 거래는 reconciliation={previous['reconciliation']['originToMarket'] and current['reconciliation']['originToMarket'] and previous['reconciliation']['marketToWholeDay'] and current['reconciliation']['marketToWholeDay']}로 확인했습니다. 음수 조정은 원부호로 합산했습니다."])
        lines.extend(["", "대표 관측소 연결은 코드와 원문 군명칭이 일치할 때만 제한적으로 표시합니다. 농장 위치나 사건 전 수확·운송 조건을 뜻하지 않습니다."])
        for match in event["weather"]["limitedExactPlaceMatches"]:
            rain7 = match["stationWeather"]["lagBeforeEvent"]["7"]["sumRainMm"]
            lines.append(f"- {match['originCode']} {match['originNameRaw']} → ASOS {match['stationId']} ({match.get('stationPlace') or '장소 미상'}): "
                f"사전 7일 강수 합계 {('미완전/알 수 없음' if rain7 is None else f'{rain7:.1f} mm')}; 거리·고도 미상.")
        lines.extend(["", "날짜별 가격·중량 reconciliation과 시장×소분류×등급×산지 cell, 요청 해시·수집시각은 JSON에 포함했습니다. 기상은 후향 일관측이며 최초 공개·수확·운송 시각은 알 수 없습니다.", ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--prices", default="outputs/daily/2026-09-30/20260930T122342-844989ce/predictions.json")
    parser.add_argument("--output-dir", default="outputs/diagnostics/origin-linkage-20260930")
    args = parser.parse_args(argv)
    try:
        raw = Path(args.prices).read_bytes()
        source = json.loads(raw)
        db = diagnostics.coverage.open_readonly(args.db)
        try:
            db.execute("BEGIN")
            report = build_report(db, source, raw)
            db.rollback()
        finally:
            db.close()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        for filename, content in (("report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                                  ("report.md", to_markdown(report))):
            fd, temp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=output)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, output/filename)
            finally:
                try: os.unlink(temp_name)
                except FileNotFoundError: pass
        return 0
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"origin linkage audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
