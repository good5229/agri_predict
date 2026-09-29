"""Export a small, explicitly retrospective persistence-baseline dashboard."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import coverage


START_DATE = dt.date(2014, 1, 1)
CONTEXT_END = dt.date(2015, 12, 31)
EVALUATION_START = dt.date(2015, 1, 1)
EVALUATION_END = dt.date(2015, 12, 31)
MARKETS = {"110001": "서울가락", "110008": "서울강서"}
PRODUCTS = {
    "cabbage": {"name": "배추", "codes": ("10", "01")},
    "radish": {"name": "무", "codes": ("11", "01")},
    "onion": {"name": "양파", "codes": ("12", "01")},
    "garlic": {"name": "깐마늘", "codes": ("12", "09")},
    "pepper": {"name": "건고추", "codes": ("12", "07")},
}
# These labels are documented by the observed aT code responses. Code 06 has
# appeared historically with a null row label, so its verified code mapping is
# retained here; other non-listed garlic classes are excluded.
PEELED_GARLIC_LABELS = {
    "06": "깐마늘",
    "16": "깐마늘(수입)",
    "41": "깐마늘 한지",
    "42": "깐마늘 대서",
    "43": "깐마늘 남도",
}
HORIZONS = tuple(range(1, 11))
MAX_SOURCE_AGE_DAYS = 14


PRICE_SQL = coverage.SELECTION_CTE + """
SELECT p.dataset,p.task_key,p.query_json,p.latest_job_id,p.latest_state,p.latest_epoch_id,
       p.latest_has_attempt,p.selected_job_id,p.selected_epoch_id,p.selected_state,
       p.selected_row_count,p.selected_response_at,p.selected_previous,
       p.excluded_epoch_candidates,p.source_url,
       r.request_id,r.page_no,r.retrieved_at,r.api_status,r.published_at,r.version_published_at,
       n.row_no,n.reference_date,n.payload_json,n.quality_flags
FROM picked p
LEFT JOIN request_pages r
  ON r.job_id=p.selected_job_id AND r.dataset=p.dataset AND r.source_url=p.source_url
 AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id AND r.state='complete'
LEFT JOIN normalized_records n ON n.request_id=r.request_id
WHERE p.dataset='prices_origin'
  AND json_extract(p.query_json,'$."cond[trd_clcln_ymd::EQ]"') BETWEEN ? AND ?
  AND json_extract(p.query_json,'$."cond[whsl_mrkt_cd::EQ]"') IN (?,?)
  AND (
       (json_extract(p.query_json,'$."cond[gds_lclsf_cd::EQ]"')='10'
        AND json_extract(p.query_json,'$."cond[gds_mclsf_cd::EQ]"')='01')
    OR (json_extract(p.query_json,'$."cond[gds_lclsf_cd::EQ]"')='11'
        AND json_extract(p.query_json,'$."cond[gds_mclsf_cd::EQ]"')='01')
    OR (json_extract(p.query_json,'$."cond[gds_lclsf_cd::EQ]"')='12'
        AND json_extract(p.query_json,'$."cond[gds_mclsf_cd::EQ]"') IN ('01','07','09'))
  )
ORDER BY json_extract(p.query_json,'$."cond[trd_clcln_ymd::EQ]"'),
         json_extract(p.query_json,'$."cond[whsl_mrkt_cd::EQ]"'),
         json_extract(p.query_json,'$."cond[gds_lclsf_cd::EQ]"'),
         json_extract(p.query_json,'$."cond[gds_mclsf_cd::EQ]"'),
         CASE WHEN COALESCE(json_extract(p.query_json,'$."cond[gds_sclsf_cd::EQ]"'),'')='' THEN 1 ELSE 0 END,
         p.task_key,r.request_id,n.row_no
"""


def _blank_product(product_id: str, evaluation_start: dt.date = EVALUATION_START,
                   evaluation_end: dt.date = EVALUATION_END) -> dict[str, Any]:
    product = PRODUCTS[product_id]
    notes = [
        "수정본이 반영된 회고적 데이터입니다. API 응답의 최초 공개 시점은 확인할 수 없습니다.",
        "aT 원천에서 매매구분이 '경매'이고 단위명이 'kg'인 행만 사용합니다. 원산지 국산·수입은 필터링하지 않습니다.",
        "일별 가격은 부호를 유지한 총금액 합계(totprc) ÷ 총중량 합계(unit_tot_qty)로 계산합니다. 음수 조정도 반영합니다.",
        f"거래가 없는 날짜는 보간하지 않습니다. {evaluation_start.isoformat()}~{evaluation_end.isoformat()} 실제 관측일만 평가 대상으로 삼습니다.",
        "학습하지 않은 지속성 기준선이며, 미래 운영용 예측은 아닙니다.",
    ]
    if product_id == "garlic":
        notes.append("검증된 깐마늘 소분류 06·16·41·42·43만 포함합니다. 16은 '깐마늘(수입)'으로 확인되며 원산지 필터는 적용하지 않습니다.")
    if product_id == "pepper":
        notes.append("건고추는 원천에서 거래가 드문 품목입니다. 실제 경매 거래일만 표시합니다.")
    return {
        "id": product_id,
        "name": product["name"],
        "unit": "원/kg",
        "status": "no_data",
        "notes": notes,
        "observations": [],
        "predictions": [],
        "diagnostics": {
            "plannedQueryCount": 0,
            "latestAttemptStates": {},
            "selectedResponseStates": {},
            "validatedResponseQueryCount": 0,
            "emptyResponseQueryCount": 0,
            "selectedRequestCount": 0,
            "selectedJobIds": [],
            "selectedSourceUrls": [],
            "excludedEpochCandidates": 0,
            "sourceRowCount": 0,
            "includedRowCount": 0,
            "excludedRows": {},
            "negativeAmountRowsRetained": 0,
            "negativeVolumeRowsRetained": 0,
            "zeroVolumeRowsRetained": 0,
            "unitQuantityMismatchRowsRetained": 0,
            "amountMismatchRowsRetained": 0,
            "sourceDateMismatchRows": 0,
            "nonpositiveNetVolumeDays": 0,
            "nonpositiveNetPriceDays": 0,
            "incompleteMarketDays": 0,
            "missingOrIncompleteMarketQueries": 0,
            "marketQueryStatusCounts": {},
            "evaluationTargetDays": 0,
            "predictionCountByHorizon": {},
            "missingPriorObservationByHorizon": {},
            "stalePriorObservationByHorizon": {},
        },
    }


def _as_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _number(value: Decimal) -> int | float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("numeric value is outside the JSON finite range")
    if value != 0 and converted == 0:
        raise ValueError("numeric value underflows the JSON number representation")
    if value == value.to_integral_value() and abs(value) <= Decimal(2**53):
        return int(value)
    return converted


def _product_id(row: dict[str, Any]) -> str | None:
    codes = (str(row.get("gds_lclsf_cd", "")), str(row.get("gds_mclsf_cd", "")))
    for product_id, config in PRODUCTS.items():
        if config["codes"] == codes:
            return product_id
    return None


def _garlic_class_is_verified(row: dict[str, Any]) -> bool:
    code = str(row.get("gds_sclsf_cd", ""))
    expected = PEELED_GARLIC_LABELS.get(code)
    if expected is None:
        return False
    label = row.get("gds_sclsf_nm")
    return label in (None, "", expected)


def _query_product(query: dict[str, Any]) -> str | None:
    large = str(query.get("cond[gds_lclsf_cd::EQ]", ""))
    medium = str(query.get("cond[gds_mclsf_cd::EQ]", ""))
    for product_id, config in PRODUCTS.items():
        if config["codes"] == (large, medium):
            return product_id
    return None


def _json_object(text: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_date(value: Any) -> dt.date | None:
    parsed = coverage.canonical_day(value)
    if parsed is None:
        return None
    try:
        return dt.date.fromisoformat(parsed)
    except ValueError:
        return None


def _validate_date_ranges(context_start: dt.date, context_end: dt.date,
                          evaluation_start: dt.date, evaluation_end: dt.date) -> None:
    if context_start > context_end:
        raise ValueError("context_start must not be after context_end")
    if evaluation_start > evaluation_end:
        raise ValueError("evaluation_start must not be after evaluation_end")
    if evaluation_start < context_start or evaluation_end > context_end:
        raise ValueError("evaluation range must be inside the exported context range")
    if context_end > dt.date.today():
        raise ValueError("context_end must not be a future date")


def build_dashboard(db: sqlite3.Connection, db_path: str, generated_at: str | None = None, *,
                    context_start: dt.date = START_DATE, context_end: dt.date = CONTEXT_END,
                    evaluation_start: dt.date = EVALUATION_START,
                    evaluation_end: dt.date = EVALUATION_END) -> dict[str, Any]:
    """Build a bounded retrospective dashboard from a read-only connection."""
    _validate_date_ranges(context_start, context_end, evaluation_start, evaluation_end)
    products = {product_id: _blank_product(product_id, evaluation_start, evaluation_end) for product_id in PRODUCTS}
    row_buckets: dict[str, dict[dt.date, dict[str, Any]]] = {key: defaultdict(lambda: {"amount": Decimal(0), "volume": Decimal(0), "row_count": 0, "markets": set()}) for key in PRODUCTS}
    metadata: dict[str, dict[str, Any]] = {
        key: {"query_keys": set(), "request_ids": set(), "job_ids": set(), "source_urls": set(),
              "attempt_states": Counter(), "selected_states": Counter(), "excluded_rows": Counter(),
              "excluded_epoch_candidates": set(), "contributing_request_ids": set()}
        for key in PRODUCTS
    }
    market_day_states: dict[str, dict[tuple[str, str], str]] = {key: {} for key in PRODUCTS}
    market_names = dict(MARKETS)
    source_urls: set[str] = set()
    preferred_garlic_scopes: set[tuple[str, str, str]] = set()

    for record in db.execute(PRICE_SQL, (context_start.isoformat(), context_end.isoformat(), *MARKETS.keys())):
        query = _json_object(record["query_json"])
        product_id = _query_product(query or {})
        if product_id is None:
            continue
        meta = metadata[product_id]
        query_subclass = str(query.get("cond[gds_sclsf_cd::EQ]", "")) if query else ""
        query_day_text = str(query.get("cond[trd_clcln_ymd::EQ]", "")) if query else ""
        query_market_text = str(query.get("cond[whsl_mrkt_cd::EQ]", "")) if query else ""
        if not query_subclass and query_day_text and query_market_text in MARKETS:
            market_day_states[product_id][(query_day_text, query_market_text)] = (
                str(record["selected_state"]) if record["selected_state"] in ("complete", "empty")
                else str(record["latest_state"] or "unavailable")
            )
        if (product_id == "garlic" and query_subclass in PEELED_GARLIC_LABELS
                and record["selected_state"] in ("complete", "empty")):
            preferred_garlic_scopes.add((query_day_text, query_market_text, query_subclass))
        task_identity = (record["dataset"], record["source_url"], record["task_key"])
        if task_identity not in meta["query_keys"]:
            meta["query_keys"].add(task_identity)
            meta["attempt_states"][str(record["latest_state"] or "unavailable")] += 1
            meta["selected_states"][str(record["selected_state"] or "unavailable")] += 1
            if record["excluded_epoch_candidates"]:
                meta["excluded_epoch_candidates"].add(task_identity)
            if record["selected_job_id"] is not None:
                meta["job_ids"].add(int(record["selected_job_id"]))
            if record["source_url"]:
                meta["source_urls"].add(record["source_url"])
                source_urls.add(record["source_url"])
        if record["request_id"] is None:
            continue
        meta["request_ids"].add(int(record["request_id"]))
        if record["payload_json"] is None:
            continue

        product = products[product_id]
        product["diagnostics"]["sourceRowCount"] += 1
        row = _json_object(record["payload_json"])
        if row is None:
            meta["excluded_rows"]["invalid_payload_json"] += 1
            continue
        if _product_id(row) != product_id:
            meta["excluded_rows"]["payload_category_mismatch"] += 1
            continue
        if product_id == "garlic" and not _garlic_class_is_verified(row):
            meta["excluded_rows"]["not_verified_peeled_garlic_class"] += 1
            continue
        if product_id == "garlic":
            subclass = str(row.get("gds_sclsf_cd", ""))
            if query_subclass and subclass != query_subclass:
                meta["excluded_rows"]["payload_subclass_query_mismatch"] += 1
                continue
            if not query_subclass and (query_day_text, query_market_text, subclass) in preferred_garlic_scopes:
                meta["excluded_rows"]["overlap_preferred_specific_query"] += 1
                continue
        if str(row.get("trd_se", "")) != "경매":
            meta["excluded_rows"]["not_auction"] += 1
            continue
        if str(row.get("unit_nm", "")) != "kg":
            meta["excluded_rows"]["unit_not_kg"] += 1
            continue
        market = str(row.get("whsl_mrkt_cd", ""))
        query_market = str(query.get("cond[whsl_mrkt_cd::EQ]", "")) if query else ""
        if market not in MARKETS or market != query_market:
            meta["excluded_rows"]["market_mismatch"] += 1
            continue
        if row.get("whsl_mrkt_nm") not in (None, ""):
            market_names[market] = str(row["whsl_mrkt_nm"])

        query_day = _parse_date(query.get("cond[trd_clcln_ymd::EQ]") if query else None)
        source_day = _parse_date(row.get("trd_clcln_ymd"))
        if source_day is None:
            meta["excluded_rows"]["invalid_source_date"] += 1
            continue
        if query_day != source_day:
            product["diagnostics"]["sourceDateMismatchRows"] += 1
            meta["excluded_rows"]["source_date_request_mismatch"] += 1
            continue
        if source_day < context_start or source_day > context_end:
            meta["excluded_rows"]["source_date_out_of_context"] += 1
            continue

        amount = _as_decimal(row.get("totprc"))
        volume = _as_decimal(row.get("unit_tot_qty"))
        if amount is None or volume is None:
            meta["excluded_rows"]["invalid_amount_or_volume"] += 1
            continue
        product["diagnostics"]["includedRowCount"] += 1
        meta["contributing_request_ids"].add(int(record["request_id"]))
        if amount < 0:
            product["diagnostics"]["negativeAmountRowsRetained"] += 1
        if volume < 0:
            product["diagnostics"]["negativeVolumeRowsRetained"] += 1
        if volume == 0:
            product["diagnostics"]["zeroVolumeRowsRetained"] += 1
        unit_qty, qty = _as_decimal(row.get("unit_qty")), _as_decimal(row.get("qty"))
        if unit_qty is not None and qty is not None and unit_qty * qty != volume:
            product["diagnostics"]["unitQuantityMismatchRowsRetained"] += 1
        scsbd_price, trade_qty = _as_decimal(row.get("scsbd_prc")), _as_decimal(row.get("qty"))
        if scsbd_price is not None and trade_qty is not None and scsbd_price * trade_qty != amount:
            product["diagnostics"]["amountMismatchRowsRetained"] += 1
        bucket = row_buckets[product_id][source_day]
        bucket["amount"] += amount
        bucket["volume"] += volume
        bucket["row_count"] += 1
        bucket["markets"].add(market)

    all_dates: list[dt.date] = []
    for product_id, product in products.items():
        obs_map: dict[dt.date, dict[str, Any]] = {}
        market_states = market_day_states[product_id]
        status_counts: Counter = Counter()
        incomplete_days: set[str] = set()
        complete_days: set[str] = set()
        for offset in range((context_end - context_start).days + 1):
            expected_day = context_start + dt.timedelta(days=offset)
            day_text = expected_day.isoformat()
            invalid = 0
            for market_code in MARKETS:
                state = market_states.get((day_text, market_code), "unplanned")
                status_counts[state] += 1
                if state not in ("complete", "empty"):
                    invalid += 1
            if invalid:
                incomplete_days.add(day_text)
            else:
                complete_days.add(day_text)
        product["diagnostics"]["incompleteMarketDays"] = len(incomplete_days)
        product["diagnostics"]["missingOrIncompleteMarketQueries"] = sum(
            count for state, count in status_counts.items() if state not in ("complete", "empty")
        )
        product["diagnostics"]["marketQueryStatusCounts"] = dict(sorted(status_counts.items()))
        for day, bucket in sorted(row_buckets[product_id].items()):
            if day.isoformat() not in complete_days:
                continue
            if bucket["volume"] <= 0:
                product["diagnostics"]["nonpositiveNetVolumeDays"] += 1
                continue
            actual = bucket["amount"] / bucket["volume"]
            if actual <= 0:
                product["diagnostics"]["nonpositiveNetPriceDays"] += 1
                continue
            observation = {
                "date": day.isoformat(),
                "actual": _number(actual),
                "volumeKg": _number(bucket["volume"]),
                "rowCount": bucket["row_count"],
                "marketCodes": sorted(bucket["markets"]),
            }
            product["observations"].append(observation)
            obs_map[day] = observation
            all_dates.append(day)

        diagnostics = product["diagnostics"]
        item_meta = metadata[product_id]
        diagnostics["plannedQueryCount"] = len(item_meta["query_keys"])
        diagnostics["latestAttemptStates"] = dict(sorted(item_meta["attempt_states"].items()))
        diagnostics["selectedResponseStates"] = dict(sorted(item_meta["selected_states"].items()))
        diagnostics["validatedResponseQueryCount"] = sum(item_meta["selected_states"].get(s, 0) for s in ("complete", "empty"))
        diagnostics["emptyResponseQueryCount"] = item_meta["selected_states"].get("empty", 0)
        diagnostics["selectedRequestCount"] = len(item_meta["request_ids"])
        diagnostics["contributingRequestCount"] = len(item_meta["contributing_request_ids"])
        diagnostics["selectedJobIds"] = sorted(item_meta["job_ids"])
        diagnostics["selectedSourceUrls"] = sorted(item_meta["source_urls"])
        diagnostics["excludedEpochCandidates"] = len(item_meta["excluded_epoch_candidates"])
        diagnostics["excludedRows"] = dict(sorted(item_meta["excluded_rows"].items()))
        product["status"] = "sparse" if product_id == "pepper" and product["observations"] else ("available" if product["observations"] else "no_data")

        targets = [day for day in obs_map if evaluation_start <= day <= evaluation_end]
        diagnostics["evaluationTargetDays"] = len(targets)
        count_by_horizon: Counter = Counter()
        missing_by_horizon: Counter = Counter()
        stale_by_horizon: Counter = Counter()
        context_days = sorted(obs_map)
        for target in targets:
            for horizon in HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                prior_candidates = [day for day in context_days if day <= origin]
                if not prior_candidates:
                    missing_by_horizon[str(horizon)] += 1
                    continue
                source_day = prior_candidates[-1]
                age = (origin - source_day).days
                if age > MAX_SOURCE_AGE_DAYS:
                    stale_by_horizon[str(horizon)] += 1
                    continue
                source_actual = obs_map[source_day]["actual"]
                product["predictions"].append({
                    "origin": origin.isoformat(),
                    "target": target.isoformat(),
                    "horizon": horizon,
                    "predicted": source_actual,
                    "sourceDate": source_day.isoformat(),
                })
                count_by_horizon[str(horizon)] += 1
        diagnostics["predictionCountByHorizon"] = {str(h): count_by_horizon.get(str(h), 0) for h in HORIZONS}
        diagnostics["missingPriorObservationByHorizon"] = {str(h): missing_by_horizon.get(str(h), 0) for h in HORIZONS}
        diagnostics["stalePriorObservationByHorizon"] = {str(h): stale_by_horizon.get(str(h), 0) for h in HORIZONS}

    product_values = list(products.values())
    data_start = min(all_dates).isoformat() if all_dates else None
    data_end = max(all_dates).isoformat() if all_dates else None
    return {
        "schemaVersion": 1,
        "generatedAt": generated_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "meta": {
            "title": f"{evaluation_start.isoformat()}~{evaluation_end.isoformat()} 도매시장 경매가격 지속성 기준선",
            "modelName": "직전 관측가격 기준선",
            "modelKind": "baseline",
            "modelDescription": "예측 원점 이전 또는 당일의 마지막 유효 일별 가격을 그대로 사용합니다. 학습은 수행하지 않았습니다.",
            "contextStart": context_start.isoformat(),
            "contextEnd": context_end.isoformat(),
            "evaluationStart": evaluation_start.isoformat(),
            "evaluationEnd": evaluation_end.isoformat(),
            "dataStart": data_start,
            "dataEnd": data_end,
            "marketNames": [market_names.get(code, MARKETS[code]) for code in MARKETS],
            "sourceUrls": sorted(source_urls),
            "limitations": [
                "API 응답의 최초 공개 시점은 확인할 수 없으며, 수정본이 반영된 회고적 자료로 평가했습니다.",
                "관측일은 경매 정산자료의 거래일입니다. API의 mdfcn_dt는 공표시각으로 해석하지 않습니다.",
                "매매구분이 '경매'이고 단위명이 'kg'인 행만 사용하며, 원산지 국산·수입은 필터링하지 않습니다.",
                "가격은 시장별 합계금액/합계 kg로 계산한 뒤 두 시장의 금액과 중량을 합산해 일별 가중가격을 산출합니다.",
                "예측 기준일과 원자료 날짜의 간격은 최대 14일입니다. 더 오래된 값만 있으면 예측하지 않습니다.",
                f"{evaluation_start.isoformat()}~{evaluation_end.isoformat()} 실제 관측일에 대해 1~10일(달력일 기준) 평가만 포함하며, 미래 운영 예측은 없습니다.",
                "거래가 없는 날은 보간하거나 휴장·가격 0으로 해석하지 않습니다. 건고추는 희소 시계열입니다.",
                "각 품목·날짜별로 가락과 강서의 중분류 조회가 모두 정상 완료 또는 정상 0건이어야 관측치를 만들며, 미조회·실패 날짜는 제외합니다.",
            ],
            "aggregation": "각 시장별 부호 있는 총금액(totprc) 합계 ÷ 총중량(unit_tot_qty) 합계. 이후 두 시장의 금액·중량을 각각 합산해 일별 kg 가중가격을 계산합니다. 단위명은 'kg'만 포함합니다.",
            "provenance": {
                "selectionPolicy": "자료(dataset/source/task_key)별 최신 검증 완료 응답 epoch를 사용하며, 최신 시도가 미완료이면 이전 검증 완료 응답을 사용할 수 있습니다.",
                "publicationTimeKnown": False,
                "vintage": "수정본 반영 회고 버전",
                "garlicClassEvidence": {
                    "06": "docs/data-feasibility.md §6: 공식 표본 코드 목록의 명칭은 깐마늘입니다. 2018년 거래행에 소분류명이 null인 이력이 있어 코드 근거를 사용합니다.",
                    "16": "현재 DB codes_goods 원문 명칭: 깐마늘(수입)",
                    "41": "현재 DB codes_goods 원문 명칭: 깐마늘 한지",
                    "42": "현재 DB codes_goods 원문 명칭: 깐마늘 대서",
                    "43": "현재 DB codes_goods 원문 명칭: 깐마늘 남도",
                },
                "selectedJobIds": sorted({job for item in product_values for job in item["diagnostics"]["selectedJobIds"]}),
                "selectedRequestCount": sum(item["diagnostics"]["selectedRequestCount"] for item in product_values),
                "contributingRequestCount": sum(len(metadata[key]["contributing_request_ids"]) for key in PRODUCTS),
                "selectedQueryCount": sum(item["diagnostics"]["validatedResponseQueryCount"] for item in product_values),
                "queryPlanCount": sum(item["diagnostics"]["plannedQueryCount"] for item in product_values),
            },
        },
        "products": product_values,
    }


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, output)
    except (OSError, TypeError, ValueError) as exc:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise ValueError(f"could not atomically write dashboard JSON to {output}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export retrospective aT auction/kg persistence-baseline dashboard JSON.")
    parser.add_argument("--db", default="data/agri.sqlite", help="SQLite collector database (opened read-only)")
    parser.add_argument("--output", default="outputs/prices-dashboard.json", help="atomic JSON output path")
    parser.add_argument("--context-start", type=dt.date.fromisoformat, default=START_DATE)
    parser.add_argument("--context-end", type=dt.date.fromisoformat, default=CONTEXT_END)
    parser.add_argument("--evaluation-start", type=dt.date.fromisoformat, default=EVALUATION_START)
    parser.add_argument("--evaluation-end", type=dt.date.fromisoformat, default=EVALUATION_END)
    args = parser.parse_args(argv)
    db: sqlite3.Connection | None = None
    try:
        _validate_date_ranges(args.context_start, args.context_end,
                              args.evaluation_start, args.evaluation_end)
        changed_range = (args.context_start, args.context_end, args.evaluation_start, args.evaluation_end) != (
            START_DATE, CONTEXT_END, EVALUATION_START, EVALUATION_END)
        if changed_range and Path(args.output).resolve() == Path("outputs/prices-dashboard.json").resolve():
            raise ValueError("custom date ranges require a new --output path to preserve the 2015 export")
        if Path(args.db).resolve() == Path(args.output).resolve():
            raise ValueError("output path must differ from the read-only database path")
        db = coverage.open_readonly(args.db)
        report = build_dashboard(db, args.db, context_start=args.context_start, context_end=args.context_end,
                                 evaluation_start=args.evaluation_start, evaluation_end=args.evaluation_end)
        write_json_atomic(args.output, report)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"dashboard export failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if db is not None:
            db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
