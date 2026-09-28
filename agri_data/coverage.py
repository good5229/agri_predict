"""Read-only monthly coverage summaries for collected aT price queries."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import sqlite3
import sys
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any


PRICE_CODES = {
    ("10", "01"): "cabbage",
    ("11", "01"): "radish",
    ("12", "01"): "onion",
    ("12", "09"): "garlic",
    ("12", "07"): "pepper",
}
MISSING = "(missing)"


def _request_task_key(dataset: str, source_url: str, query_json: str) -> str | None:
    """Recreate collector.task_key after removing pagination-only parameters."""
    try:
        params = json.loads(query_json)
        if not isinstance(params, dict):
            return None
        for name in ("pageNo", "numOfRows", "returnType"):
            params.pop(name, None)
        canonical = json.dumps([dataset, source_url, sorted(params.items()), ""], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


SELECTION_CTE = """
WITH request_activity AS (
  SELECT job_id,dataset,source_url,epoch_id,MAX(retrieved_at) AS retrieved_at
  FROM source_requests WHERE dataset IN ('prices_origin','prices_sale')
  GROUP BY job_id,dataset,source_url,epoch_id
),
attempt_activity AS (
  SELECT job_id,epoch_id,MAX(started_at) AS started_at
  FROM request_attempts GROUP BY job_id,epoch_id
),
base AS (
  SELECT p.job_id,p.dataset,p.source_url,p.task_key,p.query_json,p.state,p.row_count,p.epoch_id,
         j.started_at,CASE WHEN aa.started_at IS NOT NULL THEN 1 ELSE 0 END AS has_attempt,
         CASE WHEN p.state IN ('complete','empty') THEN COALESCE(ra.retrieved_at,j.started_at)
              ELSE COALESCE(aa.started_at,ra.retrieved_at,j.started_at) END AS activity_at
  FROM planned_queries p JOIN jobs j USING(job_id)
  LEFT JOIN request_activity ra ON ra.job_id=p.job_id AND ra.dataset=p.dataset
    AND ra.source_url=p.source_url AND ra.epoch_id=p.epoch_id
  LEFT JOIN attempt_activity aa ON aa.job_id=p.job_id AND aa.epoch_id=p.epoch_id
  WHERE p.dataset IN ('prices_origin','prices_sale')
),
latest_ranked AS (
  SELECT base.*,row_number() OVER (
    PARTITION BY dataset,source_url,task_key ORDER BY activity_at DESC,job_id DESC
  ) AS latest_rank
  FROM base
),
latest AS (SELECT * FROM latest_ranked WHERE latest_rank=1),
request_pages AS (
  SELECT r.request_id,r.job_id,r.dataset,r.source_url,r.epoch_id,r.retrieved_at,r.state,
         r.expected_count,r.returned_count,r.api_status,r.page_no,r.published_at,r.version_published_at,
         coverage_task_key(r.dataset,r.source_url,r.query_json) AS task_key,
         CAST(json_extract(r.query_json,'$.pageNo') AS INTEGER) AS request_page,
         CAST(json_extract(r.query_json,'$.numOfRows') AS INTEGER) AS request_page_size,
         CASE WHEN r.state='complete' THEN
           (SELECT count(*) FROM raw_records rr WHERE rr.request_id=r.request_id) ELSE 0 END AS raw_n,
         CASE WHEN r.state='complete' THEN
           (SELECT count(*) FROM normalized_records nn WHERE nn.request_id=r.request_id) ELSE 0 END AS norm_n
  FROM source_requests r
  WHERE r.dataset IN ('prices_origin','prices_sale')
  GROUP BY r.request_id
),
candidate_pages AS (
  SELECT l.dataset,l.source_url,l.task_key,r.job_id,r.epoch_id,r.request_id,r.retrieved_at,
         r.state,r.expected_count,r.returned_count,r.request_page,r.request_page_size,r.raw_n,r.norm_n,
         origin_plan.state AS manifest_state,origin_plan.epoch_id AS manifest_epoch_id
  FROM latest l
  JOIN request_pages r ON r.dataset=l.dataset AND r.source_url=l.source_url AND r.task_key=l.task_key
  JOIN planned_queries origin_plan ON origin_plan.job_id=r.job_id AND origin_plan.dataset=r.dataset
    AND origin_plan.source_url=r.source_url AND origin_plan.task_key=r.task_key
),
epoch_measures AS (
  SELECT dataset,source_url,task_key,job_id,epoch_id,
         MAX(retrieved_at) AS response_at,MAX(request_id) AS last_request_id,
         COUNT(*) AS page_count,COUNT(DISTINCT request_page) AS distinct_page_count,
         MIN(request_page) AS first_page,MAX(request_page) AS last_page,
         MIN(expected_count) AS min_total,MAX(expected_count) AS max_total,
         MIN(request_page_size) AS min_page_size,MAX(request_page_size) AS max_page_size,
         SUM(returned_count) AS returned_sum,SUM(raw_n) AS raw_sum,SUM(norm_n) AS norm_sum,
         SUM(CASE WHEN manifest_epoch_id=epoch_id AND manifest_state NOT IN ('complete','empty')
                  THEN 1 ELSE 0 END) AS blocked_by_manifest,
         SUM(CASE WHEN state!='complete' THEN 1 ELSE 0 END) AS noncomplete_pages,
         SUM(CASE WHEN expected_count IS NULL OR returned_count IS NULL OR request_page IS NULL OR
                       request_page_size IS NULL OR request_page_size<=0 OR
                       returned_count!=MAX(0,MIN(request_page_size,
                                      expected_count-(request_page-1)*request_page_size))
                  THEN 1 ELSE 0 END) AS bad_page_sizes,
         SUM(CASE WHEN raw_n!=COALESCE(returned_count,0) OR norm_n!=COALESCE(returned_count,0)
                  THEN 1 ELSE 0 END) AS bad_record_links
  FROM candidate_pages
  GROUP BY dataset,source_url,task_key,job_id,epoch_id
),
ranked_epochs AS (
  SELECT e.*,
         CASE WHEN e.noncomplete_pages=0 AND e.blocked_by_manifest=0 AND e.min_total=e.max_total AND e.min_total IS NOT NULL
                   AND e.min_page_size=e.max_page_size AND e.min_page_size>0
                   AND e.distinct_page_count=e.page_count AND e.first_page=1 AND e.last_page=e.page_count
                   AND e.page_count=CASE WHEN e.max_total=0 THEN 1
                                         ELSE (e.max_total+e.min_page_size-1)/e.min_page_size END
                   AND e.returned_sum=e.max_total AND e.raw_sum=e.returned_sum AND e.norm_sum=e.returned_sum
                   AND e.bad_page_sizes=0 AND e.bad_record_links=0
              THEN 1 ELSE 0 END AS valid_epoch
  FROM epoch_measures e
),
successful AS (
  SELECT v.*,row_number() OVER (PARTITION BY v.dataset,v.source_url,v.task_key
                               ORDER BY v.response_at DESC,v.last_request_id DESC) AS success_rank
  FROM ranked_epochs v WHERE valid_epoch=1
),
invalid_epochs AS (
  SELECT dataset,source_url,task_key,COUNT(*) AS n FROM ranked_epochs WHERE valid_epoch=0
  GROUP BY dataset,source_url,task_key
),
picked AS (
  SELECT l.job_id AS latest_job_id,l.state AS latest_state,l.epoch_id AS latest_epoch_id,
         l.has_attempt AS latest_has_attempt,
         l.activity_at AS latest_activity_at,
         s.job_id AS selected_job_id,
         CASE WHEN s.job_id IS NULL THEN NULL WHEN s.max_total=0 THEN 'empty' ELSE 'complete' END AS selected_state,
         s.max_total AS selected_row_count,s.epoch_id AS selected_epoch_id,s.response_at AS selected_response_at,
         COALESCE(i.n,0) AS excluded_epoch_candidates,
         CASE WHEN s.job_id IS NOT NULL AND
              (s.job_id!=l.job_id OR s.epoch_id!=l.epoch_id OR l.state NOT IN ('complete','empty'))
              THEN 1 ELSE 0 END AS selected_previous,
         l.dataset,l.source_url,l.task_key,l.query_json
  FROM latest l LEFT JOIN successful s
    ON s.dataset=l.dataset AND s.source_url=l.source_url AND s.task_key=l.task_key AND s.success_rank=1
  LEFT JOIN invalid_epochs i ON i.dataset=l.dataset AND i.source_url=l.source_url AND i.task_key=l.task_key
)
"""


def open_readonly(path: str) -> sqlite3.Connection:
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"database does not exist: {path}")
    uri = "file:" + urllib.parse.quote(str(file.resolve()), safe="/") + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    db.create_function("coverage_task_key", 3, _request_task_key, deterministic=True)
    db.execute("PRAGMA query_only=ON")
    required = {"jobs", "planned_queries", "source_requests", "request_attempts", "raw_records", "normalized_records"}
    present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - present
    if missing:
        db.close()
        raise ValueError("database is missing collector tables: " + ", ".join(sorted(missing)))
    return db


def price_dimensions(query: dict[str, Any], dataset: str) -> tuple[str, str, str, str, str] | None:
    day = query.get("cond[trd_clcln_ymd::EQ]")
    market = query.get("cond[whsl_mrkt_cd::EQ]")
    large = str(query.get("cond[gds_lclsf_cd::EQ]", ""))
    medium = str(query.get("cond[gds_mclsf_cd::EQ]", ""))
    try:
        parsed = dt.date.fromisoformat(str(day))
    except (TypeError, ValueError):
        return None
    if market in (None, ""):
        return None
    crop = PRICE_CODES.get((large, medium), f"{large}:{medium}")
    sub = query.get("cond[gds_sclsf_cd::EQ]")
    scope = "all_subclasses" if sub in (None, "") else "subclass:" + str(sub)
    return parsed.strftime("%Y-%m"), str(market), crop, dataset.removeprefix("prices_"), scope


def canonical_day(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        if len(text) == 8 and text.isdigit():
            return dt.datetime.strptime(text, "%Y%m%d").date().isoformat()
        return dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _new_group(key: tuple[str, str, str, str, str]) -> dict[str, Any]:
    month, market, crop, source, scope = key
    year, month_no = (int(x) for x in month.split("-"))
    return {
        "month": month, "market": market, "crop": crop, "source": source, "scope": scope,
        "calendar_days": calendar.monthrange(year, month_no)[1],
        "planned_days": 0, "queried_days": 0, "observed_days": 0, "empty_days": 0,
        "pending_days": 0, "failed_days": 0, "incomplete_days": 0,
        "latest_attempt_state_days": Counter(), "selected_previous_completion_days": 0,
        "excluded_epoch_candidates": 0,
        "selected_job_ids": set(), "selected_epoch_ids": set(), "latest_attempt_job_ids": set(),
        "api_declared_rows": 0, "api_returned_rows": 0, "raw_rows": 0,
        "normalized_rows": 0, "row_count_mismatch_queries": 0, "page_integrity_errors": 0,
        "first_observed_date": None, "last_observed_date": None,
        "selected_response_first_retrieved_at": None, "selected_response_last_retrieved_at": None,
        "source_date_mismatch_rows": 0, "trade_type_counts": Counter(),
        "subclass_code_counts": Counter(), "unit_counts": Counter(),
        "unit_qty_counts": Counter(), "unit_tot_qty_counts": Counter(),
        "quality_flag_counts": Counter(), "mdfcn_dt_distribution": Counter(),
        "api_status_counts": Counter(), "publication_time_unknown_requests": 0,
    }


def _value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return MISSING


def _add_counter(counter: Counter, key: Any, amount: int = 1) -> None:
    counter[str(key) if key not in (None, "") else MISSING] += amount


def _close_query(group: dict[str, Any], query_stats: dict[str, Any] | None) -> None:
    if query_stats is None:
        return
    selected = query_stats["selected_state"]
    if selected is None:
        return
    returned = query_stats["returned"]
    actual_norm = query_stats["norm"]
    actual_raw = query_stats["raw"]
    expected = query_stats["manifest_rows"] or 0
    if returned != expected or actual_norm != expected or actual_raw != expected:
        group["row_count_mismatch_queries"] += 1
    page_numbers = query_stats["page_numbers"]
    page_totals = query_stats["page_totals"]
    if (not page_numbers or any(value is None for value in page_numbers)
            or len(set(page_numbers)) != len(page_numbers)
            or sorted(set(page_numbers)) != list(range(1, len(set(page_numbers)) + 1))
            or len(page_totals) != 1 or (page_totals and next(iter(page_totals)) != expected)):
        group["page_integrity_errors"] += 1


def _selection_sql() -> str:
    return SELECTION_CTE + """
SELECT p.*,
       r.request_id,r.page_no,r.expected_count,r.returned_count,r.api_status,r.retrieved_at,r.published_at,r.version_published_at,
       COALESCE(r.raw_n,0) AS raw_n,COALESCE(r.norm_n,0) AS norm_n,
       n.reference_date,n.payload_json,n.quality_flags
FROM picked p
LEFT JOIN request_pages r
  ON r.job_id=p.selected_job_id AND r.dataset=p.dataset AND r.source_url=p.source_url
 AND r.epoch_id=p.selected_epoch_id AND r.state='complete'
LEFT JOIN normalized_records n ON n.request_id=r.request_id
ORDER BY p.dataset,p.task_key,r.request_id,n.row_no
"""


def build_report(db: sqlite3.Connection, db_path: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    unclassified = 0
    current_task: str | None = None
    current_group: dict[str, Any] | None = None
    query_stats: dict[str, Any] | None = None
    seen_requests: set[int] = set()

    cursor = db.execute(_selection_sql())
    for row in cursor:
        task_key = f"{row['dataset']}:{row['task_key']}"
        if task_key != current_task:
            _close_query(current_group, query_stats)
            current_task = task_key
            seen_requests = set()
            try:
                query = json.loads(row["query_json"])
                dims = price_dimensions(query, row["dataset"])
            except (json.JSONDecodeError, TypeError):
                dims = None
            if dims is None:
                unclassified += 1
                current_group = None
                query_stats = None
                continue
            current_group = groups.setdefault(dims, _new_group(dims))
            query_stats = {
                "selected_state": row["selected_state"], "manifest_rows": row["selected_row_count"],
                "returned": 0, "raw": 0, "norm": 0, "page_numbers": [], "page_totals": set(),
            }
            group = current_group
            group["planned_days"] += 1
            latest_state = row["latest_state"] or "unknown"
            group["latest_attempt_state_days"][latest_state] += 1
            if row["latest_job_id"] is not None:
                group["latest_attempt_job_ids"].add(row["latest_job_id"])
            if row["selected_previous"]:
                group["selected_previous_completion_days"] += 1
            group["excluded_epoch_candidates"] += row["excluded_epoch_candidates"] or 0
            selected_state = row["selected_state"]
            has_selected = selected_state in ("complete", "empty")
            has_latest_attempt = bool(row["latest_has_attempt"])
            if has_selected or has_latest_attempt:
                group["queried_days"] += 1
            if has_selected:
                group["selected_job_ids"].add(row["selected_job_id"])
                group["selected_epoch_ids"].add(row["selected_epoch_id"])
                if selected_state == "empty":
                    group["empty_days"] += 1
                elif (row["selected_row_count"] or 0) > 0:
                    group["observed_days"] += 1
                group["api_declared_rows"] += row["selected_row_count"] or 0
            if latest_state == "pending":
                group["pending_days"] += 1
            elif latest_state == "failed":
                group["failed_days"] += 1
            elif latest_state == "incomplete":
                group["incomplete_days"] += 1
        if current_group is None or query_stats is None:
            continue
        request_id = row["request_id"]
        if request_id is not None and request_id not in seen_requests:
            seen_requests.add(request_id)
            query_stats["returned"] += row["returned_count"] or 0
            query_stats["raw"] += row["raw_n"] or 0
            query_stats["norm"] += row["norm_n"] or 0
            query_stats["page_numbers"].append(row["page_no"])
            if row["expected_count"] is not None:
                query_stats["page_totals"].add(row["expected_count"])
            current_group["api_returned_rows"] += row["returned_count"] or 0
            current_group["raw_rows"] += row["raw_n"] or 0
            current_group["normalized_rows"] += row["norm_n"] or 0
            _add_counter(current_group["api_status_counts"], row["api_status"])
            retrieved_at = row["retrieved_at"]
            if retrieved_at:
                if current_group["selected_response_first_retrieved_at"] is None or retrieved_at < current_group["selected_response_first_retrieved_at"]:
                    current_group["selected_response_first_retrieved_at"] = retrieved_at
                if current_group["selected_response_last_retrieved_at"] is None or retrieved_at > current_group["selected_response_last_retrieved_at"]:
                    current_group["selected_response_last_retrieved_at"] = retrieved_at
            if row["published_at"] is None and row["version_published_at"] is None:
                current_group["publication_time_unknown_requests"] += 1
        if row["payload_json"] is None:
            continue
        try:
            record = json.loads(row["payload_json"])
            flags = json.loads(row["quality_flags"] or "[]")
        except (json.JSONDecodeError, TypeError):
            record, flags = {}, ["coverage:invalid_normalized_json"]
        group = current_group
        ref_date = row["reference_date"]
        if ref_date:
            if group["first_observed_date"] is None or ref_date < group["first_observed_date"]:
                group["first_observed_date"] = ref_date
            if group["last_observed_date"] is None or ref_date > group["last_observed_date"]:
                group["last_observed_date"] = ref_date
        try:
            planned_day = json.loads(row["query_json"]).get("cond[trd_clcln_ymd::EQ]")
            actual_day = canonical_day(record.get("trd_clcln_ymd"))
            if actual_day != canonical_day(planned_day):
                group["source_date_mismatch_rows"] += 1
        except (json.JSONDecodeError, TypeError):
            group["source_date_mismatch_rows"] += 1
        _add_counter(group["trade_type_counts"], _value(record, "trd_se"))
        _add_counter(group["subclass_code_counts"], _value(record, "gds_sclsf_cd", "gds_scls_cd"))
        _add_counter(group["unit_counts"], _value(record, "unit", "unit_nm"))
        _add_counter(group["unit_qty_counts"], _value(record, "unit_qty"))
        _add_counter(group["unit_tot_qty_counts"], _value(record, "unit_tot_qty"))
        _add_counter(group["mdfcn_dt_distribution"], _value(record, "mdfcn_dt"))
        for flag in flags if isinstance(flags, list) else ["coverage:invalid_quality_flags"]:
            _add_counter(group["quality_flag_counts"], flag)
    _close_query(current_group, query_stats)

    output_groups = []
    for _, group in sorted(groups.items()):
        for field in ("latest_attempt_state_days", "trade_type_counts", "subclass_code_counts", "unit_counts",
                      "unit_qty_counts", "unit_tot_qty_counts", "quality_flag_counts", "mdfcn_dt_distribution", "api_status_counts"):
            group[field] = dict(sorted(group[field].items()))
        group["selected_job_ids"] = sorted(group["selected_job_ids"])
        group["selected_epoch_ids"] = sorted(group["selected_epoch_ids"])
        group["latest_attempt_job_ids"] = sorted(group["latest_attempt_job_ids"])
        group["unplanned_calendar_days"] = group["calendar_days"] - group["planned_days"]
        group["availability_basis"] = "manifest dates only; unplanned calendar days remain uncollected"
        group["empty_interpretation"] = "confirmed empty response; not a closure day and not a zero price"
        group["publication_time_interpretation"] = "mdfcn_dt is retained as a field distribution; publication/version time is unknown"
        output_groups.append(group)
    return {
        "database": db_path,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "selection_policy": "one latest validated completed response epoch per identical dataset/source/task_key; a newer incomplete/pending/failed attempt is reported while the prior validated completed epoch may supply rows",
        "scope_policy": "all-subclass medium-class queries and small-subclass-filtered queries are separate scopes; records are never deduplicated",
        "planned_dates_policy": "calendar coverage is compared with manifest-planned dates; a month is not considered fully queried from a partial date range",
        "day_count_policy": "queried/observed/empty describe the selected completed epoch or actual HTTP attempt; pending/failed/incomplete describe the latest manifest attempt and may overlap when an older completed epoch is selected",
        "publication_time_policy": "source publication and version publication times are unknown; mdfcn_dt is not treated as release time",
        "vintage_policy": "selected rows represent the chosen retrieved response vintage; no source release timestamp is inferred",
        "unclassified_price_queries": unclassified,
        "groups": output_groups,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agri-coverage", description="Read-only monthly price collection coverage report")
    parser.add_argument("--db", default="data/agri.sqlite", help="SQLite database path (read-only; default data/agri.sqlite)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    db = None
    try:
        db = open_readonly(args.db)
        report = build_report(db, args.db)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, sqlite3.Error) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
