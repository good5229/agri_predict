"""Post-hoc exploratory exact-stratum price comparison across three markets."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import coverage, dashboard, market_extension, regional_gap

START, END = dt.date(2017, 1, 1), dt.date(2020, 12, 31)
GWANGJU, GARAK, GANGSEO = regional_gap.GWANGJU, regional_gap.GARAK, regional_gap.GANGSEO
MARKETS = (GWANGJU, GARAK, GANGSEO)
PRODUCTS = tuple(dashboard.PRODUCTS)
GARLIC_CLASSES = regional_gap.GARLIC_CLASSES
CODE_FIELDS = ("gds_sclsf_cd", "grd_cd", "sz_cd", "pkg_cd")
NAME_FIELDS = ("gds_sclsf_nm", "grd_nm", "sz_nm", "pkg_nm", "corp_gds_item_nm", "corp_gds_vrty_nm")
MISSING = {None, "", ".", "-", "--", "?", "미상", "기타"}

PROTOCOL = {
    "id": "regional-standardize-exploratory-2026-10-05",
    "period": [START.isoformat(), END.isoformat()],
    "markets": {"gwangju": GWANGJU, "garak": GARAK, "gangseo": GANGSEO},
    "stratum": "date, product, subclass code+name, grade code+name, size code+name, package code+name, whitespace-normalized corporate item+variety names, exact Decimal unit_qty, unit code+name",
    "price": "signed sum(totprc)/signed sum(unit_tot_qty), auction/kg only",
    "pairing": "exact common stratum; positive net amount and weight on both sides; pooled Seoul prices; shared weight=min(Gwangju kg, Seoul kg)",
    "dailyWeighting": "normalize shared stratum kg within each date, then equal weight across paired dates",
    "coverageDenominator": "source-scoped finite amount/weight auction/kg rows in the verified product scope, grouped by recorded attributes including missing identifiers; sum only positive signed net kg per market/date/group; exclude nonpositive groups separately; matching-name/unit-qty eligibility is not required",
    "sizeDot": "sz_nm='.' is physical-size-unknown; report its paired kg/share; never infer physical-size equality",
    "inference": {"method": "regional_gap._infer", "HACCalendarDays": [28, 9], "bootstrapMovingBlockDays": 28,
        "bootstrapReplicates": 9999, "minimumDates": 90, "minimumYears": 2, "minimumSeasons": 3,
        "minimumDatesPerSeason": 10, "holmFamily": "5 products x 3 tests", "alpha": .05},
    "interpretation": "post-hoc exploratory observed market difference; not causal, not physical-size or quality equivalence",
}


def _json(value):
    try:
        return json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _norm_corp(value):
    return " ".join(str(value).split()) if value is not None else ""


def _dec(value):
    if value is None or value == "":
        return None
    try:
        out = Decimal(str(value))
        return out if out.is_finite() else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def _unit_code(payload):
    # The aT normalized payload uses `unit`; retain `unit_cd` if present in
    # newer source versions while preserving the field's recorded value.
    value = payload.get("unit_cd")
    return payload.get("unit") if value is None else value


def _raw_token(value):
    return ("null", "") if value is None else (type(value).__name__, str(value))


def _coverage_group_key(product, day, payload):
    """Group every valid source row before matching-field validation."""
    codes = tuple(_raw_token(payload.get(field)) for field in CODE_FIELDS)
    names = tuple(_raw_token(payload.get(field)) for field in NAME_FIELDS[:4])
    corp = tuple(("null", "") if payload.get(field) is None else ("str", _norm_corp(payload.get(field)))
                 for field in NAME_FIELDS[4:])
    unit_qty = _dec(payload.get("unit_qty"))
    qty_token = ("decimal", unit_qty) if unit_qty is not None else ("raw", _raw_token(payload.get("unit_qty")))
    return (product, day, *codes, *names, *corp, qty_token,
            _raw_token(_unit_code(payload)), _raw_token(payload.get("unit_nm")))


def _base_key(product, day, payload, subclass, unit_qty, unit_code, unit_name):
    codes = tuple(str(payload.get(field)).strip() if payload.get(field) is not None else "" for field in CODE_FIELDS)
    dimension_names = tuple(str(payload.get(field)) if payload.get(field) is not None else "" for field in NAME_FIELDS[1:4])
    # The subclass code/name pair and the remaining code/name pairs are all
    # exact stratum fields. Corporate item/variety alone normalize whitespace.
    subclass_name = str(payload.get("gds_sclsf_nm")) if payload.get("gds_sclsf_nm") is not None else ""
    corp = (_norm_corp(payload.get("corp_gds_item_nm")), _norm_corp(payload.get("corp_gds_vrty_nm")))
    return (product, day, subclass, subclass_name, *codes[1:], *dimension_names, *corp,
            unit_qty, str(unit_code).strip(), str(unit_name).strip())


def _empty_stratum():
    return {"markets": {}, "unitQtySourceValues": set()}


def _add_row(strata, eligible, exclusions, product, day, market, payload, amount, weight):
    def excluded(reason, row_weight=None):
        item = exclusions[product][reason]
        item["rows"] += 1
        if row_weight is not None:
            item["weightKg"] += row_weight
    if payload.get("trd_se") != "경매":
        excluded("not_auction", weight)
        return
    if payload.get("unit_nm") != "kg":
        excluded("not_kg", weight)
        return
    subclass = str(payload.get("gds_sclsf_cd", "")).strip()
    if product == "garlic" and subclass not in GARLIC_CLASSES:
        excluded("unverified_garlic_subclass", weight)
        return
    if weight is None:
        excluded("invalid_signed_amount_or_weight")
        return
    if amount is None:
        excluded("invalid_signed_amount_or_weight", weight)
        return
    # This source-scoped denominator includes rows with missing matching
    # identifiers. Signed weights are netted later within this raw-attribute
    # group, before summing positive eligible groups.
    eligible[(product, day, market, _coverage_group_key(product, day, payload))] += weight
    unit_qty = _dec(payload.get("unit_qty"))
    unit_code = _unit_code(payload)
    unit_name = payload.get("unit_nm")
    if unit_qty is None:
        excluded("missing_or_invalid_unit_qty", weight)
        return
    if unit_code is None or not str(unit_code).strip() or unit_name is None or not str(unit_name).strip():
        excluded("missing_unit_code_or_name", weight)
        return
    codes = [payload.get(field) for field in CODE_FIELDS]
    if any(x is None or not str(x).strip() or str(x).strip() in {".", "-", "--", "?", "미상", "기타"} for x in codes):
        excluded("missing_or_placeholder_dimension_code", weight)
        return
    key = _base_key(product, day, payload, subclass, unit_qty, unit_code, unit_name)
    rec = strata[key]
    cell = rec["markets"].setdefault(market, {"amount": Decimal(0), "weight": Decimal(0), "rows": 0,
        "names": {field: set() for field in NAME_FIELDS}})
    cell["amount"] += amount
    cell["weight"] += weight
    cell["rows"] += 1
    for field in NAME_FIELDS:
        raw = payload.get(field)
        label = str(raw) if raw is not None else ""
        if field in ("corp_gds_item_nm", "corp_gds_vrty_nm"):
            label = _norm_corp(raw)
        cell["names"][field].add(label)
    rec["unitQtySourceValues"].add(str(payload.get("unit_qty")))


def _source_strata(db: sqlite3.Connection):
    db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
    cte = market_extension._scoped_selection_cte(START, END, MARKETS)
    picked = db.execute(cte + """SELECT p.latest_job_id,p.selected_job_id,p.selected_epoch_id,p.source_url,p.task_key,
        p.query_json,p.selected_state,p.latest_state,p.selected_previous FROM picked p WHERE p.dataset='prices_origin'""").fetchall()
    specific, states, manifests = set(), defaultdict(int), []
    scope_states = defaultdict(list)
    for row in picked:
        q = _json(row[5]); day = coverage.canonical_day(q.get("cond[trd_clcln_ymd::EQ]"))
        market = str(q.get("cond[whsl_mrkt_cd::EQ]", "")); info = coverage.price_dimensions(q, "prices_origin")
        product = info[2] if info else ""; sub = str(q.get("cond[gds_sclsf_cd::EQ]", "")); state = row[6]
        normalized = ("source_incomplete:stale_fallback" if row[8] else state if state in ("complete", "empty")
                      else f"source_incomplete:{row[7] or state or 'unqueried'}")
        states[normalized] += 1
        if product in PRODUCTS and day and market in MARKETS:
            scope_states[f"{product}|{day}|{market}"].append(normalized)
        manifests.append({"latestJobId": row[0], "selectedJobId": row[1], "epoch": row[2], "task": row[4],
            "date": day, "market": market, "product": product, "subclass": sub or None, "state": normalized,
            "latestState": row[7], "selectedState": state, "selectedPrevious": bool(row[8])})
        scope = market_extension._selected_specific_scope(q, state)
        if scope:
            specific.add(scope)
    rows = db.execute(cte + """SELECT r.request_id,r.retrieved_at,s.response_sha256,p.selected_state,p.query_json,n.row_no,n.payload_json
        FROM picked p JOIN request_pages r ON r.job_id=p.selected_job_id AND r.dataset=p.dataset AND r.source_url=p.source_url
          AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id AND r.state IN ('complete','empty')
        JOIN source_requests s ON s.request_id=r.request_id LEFT JOIN normalized_records n ON n.request_id=r.request_id
        WHERE p.dataset='prices_origin' AND p.selected_state IN ('complete','empty')
        ORDER BY p.query_json,r.request_id,n.row_no""")
    strata = defaultdict(_empty_stratum); eligible = defaultdict(Decimal); exclusions = defaultdict(
        lambda: defaultdict(lambda: {"rows": 0, "weightKg": Decimal(0)}))
    requests = {}
    for rid, retrieved, response_hash, state, query_text, _, payload_text in rows:
        query, payload = _json(query_text), _json(payload_text)
        day = coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]")); market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
        prodinfo = coverage.price_dimensions(query, "prices_origin"); product = prodinfo[2] if prodinfo else ""
        if not day or market not in MARKETS or product not in PRODUCTS:
            continue
        requests[str(rid)] = {"responseSha256": response_hash, "retrievedAt": retrieved}
        if state != "complete" or not payload:
            continue
        valid, subclass = market_extension._valid_payload_scope(payload, query, product, market)
        query_sub = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
        reason = None if valid else subclass
        if not reason and market_extension._overlap_shadowed(product, day, market, query_sub, subclass, specific):
            reason = "overlap_shadowed"
        amount, weight = _dec(payload.get("totprc")), _dec(payload.get("unit_tot_qty"))
        if reason:
            exclusions[product][reason]["rows"] += 1
            reason_weight = _dec(payload.get("unit_tot_qty"))
            if reason_weight is not None:
                exclusions[product][reason]["weightKg"] += reason_weight
            continue
        _add_row(strata, eligible, exclusions, product, day, market, payload, amount, weight)
    manifests.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    return strata, {"counts": dict(states), "scopeStates": dict(scope_states)}, eligible, exclusions, {
        "period": [START.isoformat(), END.isoformat()], "markets": list(MARKETS), "selectedScopeCount": len(manifests),
        "selectedScopesSha256": _sha(manifests), "requestCount": len(requests), "requests": requests,
        "responseHashes": {k: v["responseSha256"] for k, v in requests.items()}}


def _name_decision(rec, field):
    vals = []
    for market in (GWANGJU, GARAK, GANGSEO):
        cell = rec["markets"].get(market)
        if not cell:
            continue
        names = cell["names"][field]
        if len(names) != 1:
            return "meaningful_name_conflict", None
        value = next(iter(names))
        if value in (None, ""):
            return "missing_meaningful_name", None
        if value in MISSING:
            if field == "sz_nm" and value == ".":
                pass
            else:
                return "missing_meaningful_name", None
        vals.append(value)
    if vals and len(set(vals)) > 1:
        return "meaningful_name_conflict", None
    return None, vals[0] if vals else None


def _scope_ready(states, product, day, market):
    if not isinstance(states, dict) or "scopeStates" not in states:
        return True
    found = states["scopeStates"].get(f"{product}|{day}|{market}", [])
    return bool(found) and all(x in ("complete", "empty") for x in found)


def analyze_strata(strata, states=None, eligible_weights=None, exclusions=None, provenance=None):
    eligible_weights = eligible_weights or {}
    exclusions = exclusions or {}
    reports, tests = [], []
    for product in PRODUCTS:
        daily = defaultdict(lambda: {"weighted": 0.0, "weight": 0.0, "strata": 0})
        excluded = defaultdict(lambda: {"strata": 0, "weightKg": 0.0})
        matched = 0; matched_kg = 0.0; size_unknown_strata = 0; size_unknown_kg = 0.0
        eligible_by_day = defaultdict(lambda: {m: Decimal(0) for m in MARKETS})
        eligible_days = set()
        invalid_name_keys = set()
        # Net every source-scoped valid-amount/weight auction/kg group before
        # summing positive eligible kg. Missing identifiers remain denominator
        # groups; negative groups are excluded separately and cannot cancel
        # positives from unrelated strata.
        for (p, day, market, _group), net in eligible_weights.items():
            if p != product:
                continue
            eligible_days.add(day)
            if net > 0:
                eligible_by_day[day][market] += net
            else:
                item = excluded["nonpositive_net_eligible_source_group"]
                item["strata"] += 1
                item["weightKg"] += float(abs(net))
                item["signedNetWeightKg"] = item.get("signedNetWeightKg", 0.0) + float(net)
        for key, rec in strata.items():
            if key[0] != product:
                continue
            reason = None
            total = sum((cell["weight"] for cell in rec["markets"].values()), Decimal(0))
            for market, cell in rec["markets"].items():
                for field in NAME_FIELDS:
                    values = cell["names"][field]
                    if len(values) != 1:
                        reason = "meaningful_name_conflict"; break
                    value = next(iter(values))
                    if value in (None, "") or (value in MISSING and not (field == "sz_nm" and value == ".")):
                        reason = "missing_or_placeholder_meaningful_name"; break
                if reason:
                    break
            if reason:
                excluded[reason]["strata"] += 1; excluded[reason]["weightKg"] += float(total)
                invalid_name_keys.add(key)
                continue
        for key, rec in strata.items():
            p, day = key[0], key[1]
            if p != product:
                continue
            if key in invalid_name_keys:
                continue
            present = [rec["markets"].get(m) for m in MARKETS]
            total_weight = sum((x["weight"] for x in present if x), Decimal(0))
            if states and any(not _scope_ready(states, product, day, m) for m in MARKETS):
                reason = "source_incomplete"
            elif not present[0] or not (present[1] or present[2]):
                reason = "missing_market_stratum"
            else:
                reason = None
            if reason:
                excluded[reason]["strata"] += 1; excluded[reason]["weightKg"] += float(total_weight); continue
            gw = present[0]
            seoul = {"amount": sum((rec["markets"].get(m, {}).get("amount", Decimal(0)) for m in (GARAK, GANGSEO)), Decimal(0)),
                "weight": sum((rec["markets"].get(m, {}).get("weight", Decimal(0)) for m in (GARAK, GANGSEO)), Decimal(0))}
            if gw["amount"] <= 0 or gw["weight"] <= 0 or seoul["amount"] <= 0 or seoul["weight"] <= 0:
                excluded["nonpositive_net_stratum"]["strata"] += 1; excluded["nonpositive_net_stratum"]["weightKg"] += float(max(gw["weight"], Decimal(0)) + max(seoul["weight"], Decimal(0))); continue
            for field in NAME_FIELDS:
                name_reason, _ = _name_decision(rec, field)
                if name_reason:
                    reason = name_reason; break
            if reason:
                excluded[reason]["strata"] += 1; excluded[reason]["weightKg"] += float(min(gw["weight"], seoul["weight"])); continue
            common = min(gw["weight"], seoul["weight"])
            gap = math.log(float(gw["amount"] / gw["weight"]) / float(seoul["amount"] / seoul["weight"]))
            day_rec = daily[day]; day_rec["weighted"] += float(common) * gap; day_rec["weight"] += float(common); day_rec["strata"] += 1
            matched += 1; matched_kg += float(common)
            if next(iter(rec["markets"][GWANGJU]["names"]["sz_nm"])) == ".":
                size_unknown_strata += 1; size_unknown_kg += float(common)
        values = {dt.date.fromisoformat(day): x["weighted"] / x["weight"] for day, x in daily.items() if x["weight"] > 0}
        daily_coverage = {}
        for day in sorted(set(daily) | eligible_days):
            d = daily.get(day, {"weight": 0, "strata": 0})
            gwkg = eligible_by_day[day][GWANGJU]
            seoulkg = eligible_by_day[day][GARAK] + eligible_by_day[day][GANGSEO]
            daily_coverage[day] = {"matchedStrata": d["strata"], "matchedKg": d["weight"],
                "gwangjuEligibleKg": float(gwkg), "seoulEligibleKg": float(seoulkg),
                "shareOfGwangjuEligibleKg": float(d["weight"]) / float(gwkg) if gwkg > 0 else None,
                "shareOfSeoulEligibleKg": float(d["weight"]) / float(seoulkg) if seoulkg > 0 else None}
        inference = regional_gap._infer(values)
        descriptive_mean = sum(values.values()) / len(values) if values else None
        report = {"productId": product, "matchedDates": len(values), "matchedStrata": matched, "matchedKg": matched_kg,
            "descriptiveMeanDailyLogGap": descriptive_mean,
            "descriptiveGeometricPriceRatio": math.exp(descriptive_mean) if descriptive_mean is not None else None,
            "sizeUnknownStrata": size_unknown_strata, "sizeUnknownPairedKg": size_unknown_kg,
            "sizeUnknownShareOfMatchedStrata": size_unknown_strata / matched if matched else None,
            "sizeUnknownShareOfMatchedKg": size_unknown_kg / matched_kg if matched_kg else None,
            "dailyCoverage": daily_coverage, "excluded": {k: {"strata": v["strata"], "weightKg": v["weightKg"],
                **({"signedNetWeightKg": v["signedNetWeightKg"]} if "signedNetWeightKg" in v else {})} for k, v in excluded.items()},
            "sourceExclusions": {k: {"rows": v["rows"], "weightKg": float(v["weightKg"])} for k, v in exclusions.get(product, {}).items()},
            "dailyLogGaps": {d.isoformat(): value for d, value in values.items()}, "inference": inference}
        hac = inference.get("meanDifferenceHAC28", {})
        report["descriptiveResult"] = {"pairedDates": len(values),
            "geometricPriceRatio": report["descriptiveGeometricPriceRatio"], "meanDailyLogGap": descriptive_mean,
            "priceRatioCI90": [math.exp(x) for x in hac["ci90"]] if hac.get("ci90") else None,
            "ciStatus": "available_HAC28" if hac.get("ci90") else "unavailable_inference_gate",
            "pValueStatus": "available" if inference.get("status") == "available" else "unavailable_inference_gate"}
        reports.append(report)
        for test_name, key in (("meanZeroTest", "meanZeroTest"), ("equivalenceTOST", "equivalenceTOST"), ("smallGapDays", "smallGapDays")):
            item = inference.get(key, {}) if inference.get("status") == "available" else {}
            tests.append({"productId": product, "test": test_name, "pValue": item.get("pValue"), "available": inference.get("status") == "available"})
    # Frozen family: retain unavailable tests with p=1.
    ordered = sorted(tests, key=lambda r: (r["pValue"] if r["pValue"] is not None else 1.0, r["productId"], r["test"]))
    running = 0.0
    for i, test in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * (test["pValue"] if test["pValue"] is not None else 1.0)))
        test["holmAdjustedPValue"] = running; test["holmSignificantAt005"] = running <= .05
    lookup = {(r["productId"], r["test"]): r for r in tests}
    for report in reports:
        report["holmTests"] = {name: lookup[(report["productId"], name)] for name in ("meanZeroTest", "equivalenceTOST", "smallGapDays")}
        inf = report["inference"]
        if inf.get("status") == "available":
            inf["conclusion"] = ("practically_small" if report["holmTests"]["equivalenceTOST"]["holmSignificantAt005"] and
                report["holmTests"]["smallGapDays"]["holmSignificantAt005"] and inf["smallGapDays"]["lower95"] >= .8 else "inconclusive")
    safe_provenance = provenance or {}
    protocol = {**PROTOCOL, "sha256": _sha(PROTOCOL)}
    return {"schemaVersion": 1, "status": "complete", "exploratoryPostHoc": True, "protocol": protocol,
        "provenance": safe_provenance, "inputSha256": _sha(safe_provenance), "sourceStates": states or {},
        "products": reports, "multipleTesting": tests}


def run_analysis(db_path: Path):
    db = sqlite3.connect("file:" + str(db_path.resolve()) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row; db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        strata, states, eligible, exclusions, provenance = _source_strata(db)
        report = analyze_strata(strata, states, eligible, exclusions, provenance)
        db.rollback()
        return report
    finally:
        db.close()


def _atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_report(report, json_path: Path, markdown_path: Path):
    _atomic(json_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 품종·단위물량 비례 표준화 탐색 비교", "",
        "사후 탐색 민감도 분석입니다. 정확한 코드·명칭·법인 품목/품종·Decimal 단위물량·단위 일치 층만 비교하며, 실제 크기나 품질이 같다는 뜻은 아닙니다.",
        "크기명 `.`은 물리 크기 미확인으로 별도 집계하며, 같은 물리 크기로 해석하지 않습니다.", "",
        "| 품목 | 짝지은 날짜 | 층 | 공통 중량 kg | 광주 적격 kg 비율 | 서울 적격 kg 비율 | 크기 미확인 kg 비율 | 평균 가격비 (90% CI) | ±10% 날짜 비율 (95% 하한) | 실무 판정 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    details = []
    for p in report["products"]:
        inf = p["inference"]; hac = inf.get("meanDifferenceHAC28", {}); ci = hac.get("ci90")
        ratio = p.get("descriptiveGeometricPriceRatio")
        ci_ratio = [math.exp(x) for x in ci] if ci else None
        avg = f"{ratio:.3f} ({ci_ratio[0]:.3f}–{ci_ratio[1]:.3f})" if ratio is not None and ci_ratio else f"{ratio:.3f} (90% CI 추론 불가)" if ratio is not None else "기술통계 추론 불가"
        small = inf.get("smallGapDays", {})
        share = f"{small['observed']:.1%} ({small['lower95']:.1%})" if small.get("status") == "available" else "추론 불가"
        covs = list(p["dailyCoverage"].values())
        gw = sum(c["matchedKg"] for c in covs); eg = sum(c["gwangjuEligibleKg"] for c in covs)
        es = sum(c["seoulEligibleKg"] for c in covs)
        unknown = f"{p['sizeUnknownPairedKg']:.1f} ({p['sizeUnknownShareOfMatchedKg']:.1%})" if p["sizeUnknownShareOfMatchedKg"] is not None else "0 (—)"
        verdict = "실무상 작은 차이" if inf.get("conclusion") == "practically_small" else "추론 불가" if inf.get("status") != "available" else "작은 차이 미확인"
        lines.append(f"| {p['productId']} | {p['matchedDates']} | {p['matchedStrata']} | {p['matchedKg']:.1f} | {gw/eg:.1%} | {gw/es:.1%} | {unknown} | {avg} | {share} | {verdict} |" if eg > 0 and es > 0 else f"| {p['productId']} | {p['matchedDates']} | {p['matchedStrata']} | {p['matchedKg']:.1f} | — | — | {unknown} | {avg} | {share} | {verdict} |")
        reasons = ", ".join(f"{key}: {item['strata']}층/{item['weightKg']:.1f}kg" for key, item in p["excluded"].items()) or "없음"
        details.append(f"- {p['productId']} 제외: {reasons}; source incomplete 상태는 날짜/시장 scope 기준으로 비교를 차단합니다.")
        tests = p["holmTests"]
        def p_text(name):
            item = tests[name]
            return f"{item['holmAdjustedPValue']:.3g}" if item["available"] else "추론불가"
        hac9 = inf.get("equivalenceTOSTHAC9", {}).get("pValue")
        details.append(f"- {p['productId']} Holm15 adjusted p (0차이 / ±5% TOST / ±10% 날짜비율): {p_text('meanZeroTest')} / {p_text('equivalenceTOST')} / {p_text('smallGapDays')}; HAC9 TOST p: {hac9:.3g}" if hac9 is not None else f"- {p['productId']} Holm15 adjusted p (0차이 / ±5% TOST / ±10% 날짜비율): {p_text('meanZeroTest')} / {p_text('equivalenceTOST')} / {p_text('smallGapDays')}; HAC9 TOST p: 추론불가")
    lines.extend(["", *details, ""])
    lines.extend(["", "추론은 regional_gap._infer의 90일·2년·3계절 게이트, HAC28 주 분석/HAC9 민감도, 28일 이동블록 부트스트랩을 사용하고 5품목×3검정 고정 Holm15를 적용합니다. 유의하지 않은 결과는 동등성 증거가 아닙니다.",
        "적격 중량 분모는 유효 숫자 금액·중량이 있는 원천 범위의 경매/kg 행을 기록 속성(누락값 포함)별로 먼저 부호 합산하고, 시장·날짜·그룹별 양수 순중량만 합합니다. 매칭에 필요한 명칭·코드·단위물량 누락도 분모에서 빼지 않으며 비양수 그룹은 따로 제외합니다. 층 가격은 부호 있는 총금액/순중량, 층 격차는 양쪽 순금액·순중량이 양수인 경우에만 계산합니다. 층 공통 중량을 날짜 안에서 정규화한 뒤 날짜를 동등 가중합니다. 빈 날짜는 보간하지 않습니다.",
        f"선택 scope {report['provenance'].get('selectedScopeCount', 0)}개; 요청 {report['provenance'].get('requestCount', 0)}개. 원천 응답 해시는 provenance에만 기록합니다.", ""])
    _atomic(markdown_path, "\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--db", default="data/agri.sqlite")
    ap.add_argument("--output", required=True); ap.add_argument("--report")
    args = ap.parse_args(argv); db = Path(args.db).resolve(); out = Path(args.output).resolve()
    md = Path(args.report).resolve() if args.report else out.with_suffix(".md")
    if out == db or md == db: raise ValueError("output paths must not overwrite the SQLite database")
    if out == md: raise ValueError("JSON and Markdown outputs must use different paths")
    write_report(run_analysis(db), out, md); return 0


if __name__ == "__main__":
    raise SystemExit(main())
