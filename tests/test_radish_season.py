import datetime as dt
import json
import math
import sqlite3
from decimal import Decimal

from agri_data import radish_season as rs


def _row(amount, weight, code, name):
    return {"amount": Decimal(str(amount)), "weight": Decimal(str(weight)), "rows": 1,
            "negativeRows": int(amount < 0 or weight < 0), "codes": {code}, "names": {name}}


def test_unknown_codes_reconcile_into_other_without_losing_audit():
    day = dt.date(2026, 10, 1)
    source = {"01": {day: _row(100, 10, "01", "봄무")},
              "99": {day: _row(-20, -2, "99", "미확인")}}
    cats = rs._source_series(source)
    assert cats["01"][day]["amount"] == Decimal("100")
    assert cats["other"][day]["amount"] == Decimal("-20")
    assert cats["other"][day]["weight"] == Decimal("-2")
    assert cats["other"][day]["codes"] == {"99"}
    assert cats["other"][day]["negativeRows"] == 1


def test_features_are_strictly_pre_origin_and_flag_nonpositive_denominator():
    origin = dt.date(2026, 10, 10)
    before, future = origin - dt.timedelta(days=1), origin
    markets = {"110001": {"01": {before: _row(200, 10, "01", "봄무"), future: _row(900, 9, "01", "봄무")}},
               "110008": {"02": {before: _row(100, -10, "02", "여름무")}}}
    values, audit = rs._feature_values(rs._combine_markets(markets), {}, origin)
    assert len(values) == 54
    assert audit["01/share7"]["denominatorAvailable"] is False
    assert audit["01/relativePrice"]["date"] is None
    assert audit["02/share7"]["value"] is None
    # No future category observation may leak into the recent-price feature.
    assert audit["01/relativePrice"]["available"] is False


def test_market_recompose_identity_uses_origin_target_horizon_key():
    origin, target = dt.date(2025, 1, 1), dt.date(2025, 1, 5)
    row = {"product": "radish", "horizon": 4, "origin": origin.isoformat(), "target": target.isoformat(), "candidatePrediction": 100.0}
    assert (row["product"], int(row["horizon"]), rs._day(row["target"])) == ("radish", 4, target)
    assert rs._day(row["origin"]) + dt.timedelta(days=row["horizon"]) == rs._day(row["target"])


def test_fixed_comparison_family_constants():
    assert len(rs.HORIZONS) * len(rs.COMPARATORS) * len(rs.LOSSES) == 60
    assert rs.BANDWIDTHS == (9, 28)
    assert rs.MIN_TRAIN == 90


def _synthetic_db(path, *, overlap=False):
    day = dt.date(2026, 1, 1)
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE jobs(job_id INTEGER,command TEXT,config_json TEXT,state TEXT,started_at TEXT);
      CREATE TABLE source_requests(request_id INTEGER,job_id INTEGER,dataset TEXT,source_url TEXT,epoch_id TEXT,
        state TEXT,retrieved_at TEXT,expected_count INTEGER,returned_count INTEGER,page_no INTEGER,
        api_status TEXT,published_at TEXT,version_published_at TEXT,query_json TEXT,response_sha256 TEXT);
      CREATE TABLE planned_queries(job_id INTEGER,dataset TEXT,source_url TEXT,task_key TEXT,query_json TEXT,
        state TEXT,epoch_id TEXT,row_count INTEGER);
      CREATE TABLE request_attempts(job_id INTEGER,epoch_id TEXT,started_at TEXT);
      CREATE TABLE raw_records(request_id INTEGER,row_no INTEGER,payload_json TEXT);
      CREATE TABLE normalized_records(request_id INTEGER,row_no INTEGER,dataset TEXT,reference_date TEXT,payload_json TEXT,quality_flags TEXT);
    """)
    request_id = 1
    def add_scope(mkt, amount, subclass=None, state="complete"):
        nonlocal request_id
        q = {"cond[trd_clcln_ymd::EQ]": day.isoformat(), "cond[whsl_mrkt_cd::EQ]": mkt,
             "cond[gds_lclsf_cd::EQ]": "11", "cond[gds_mclsf_cd::EQ]": "01", "pageNo": "1", "numOfRows": "1000", "returnType": "JSON"}
        if subclass is not None: q["cond[gds_sclsf_cd::EQ]"] = subclass
        query = json.dumps(q, separators=(",", ":")); url = "https://fixture.test/prices"
        task = rs.me.coverage._request_task_key("prices_origin", url, query)
        job = 1
        db.execute("INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?)", (job, "prices", "{}", "complete", "2026-01-01T00:00:00"))
        count = 0 if state == "empty" else int(amount is not None)
        db.execute("INSERT INTO planned_queries VALUES(?,?,?,?,?,?,?,?)", (job, "prices_origin", url, task, query, state, "", count))
        db.execute("INSERT INTO source_requests VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (request_id, job, "prices_origin", url, "", "complete", "2026-01-02T00:00:00", count, count, 1, "OK", None, None, query, f"sha-{request_id}"))
        if count:
            row = {"gds_lclsf_cd": "11", "gds_mclsf_cd": "01", "gds_sclsf_cd": "01", "gds_sclsf_nm": "봄무", "trd_clcln_ymd": day.isoformat(),
                   "whsl_mrkt_cd": mkt, "trd_se": "경매", "unit_nm": "kg", "totprc": str(amount), "unit_tot_qty": "10"}
            payload = json.dumps(row)
            db.execute("INSERT INTO normalized_records VALUES(?,?,?,?,?,?)", (request_id, 1, "prices_origin", day.isoformat(), payload, "[]"))
            db.execute("INSERT INTO raw_records VALUES(?,?,?)", (request_id, 1, payload))
        request_id += 1
    for mkt, market_amount in (("110001", 100), ("110008", 50)):
        add_scope(mkt, market_amount)
    if overlap: add_scope("110001", 100, "01")
    add_scope("110008", None, "77", "empty")
    db.commit(); db.close()
    return day


def test_synthetic_sqlite_extraction_uses_real_selection_and_ledger_for_no_overlap(tmp_path):
    day = _synthetic_db(tmp_path / "mini.sqlite")
    core = {"radish": {day: (7.5, 20.0)}}
    markets, audit = rs._extract_market_data(tmp_path / "mini.sqlite", {"meta": {"contextStart": day.isoformat(), "contextEnd": day.isoformat()}}, core)
    assert markets["110001"]["01"][day]["amount"] == Decimal(100)
    assert len(audit["selectedRequestHashes"]) == 3
    assert "empty" in audit["selectedQueryStates"]
    assert audit["codeNameAudit"][0]["knownName"] == "봄무"


def test_synthetic_sqlite_extraction_fails_closed_on_broad_specific_overlap(tmp_path):
    day = _synthetic_db(tmp_path / "overlap.sqlite", overlap=True)
    core = {"radish": {day: (7.5, 20.0)}}
    import pytest
    with pytest.raises(ValueError, match="ambiguous_radish_scope_overlap"):
        rs._extract_market_data(tmp_path / "overlap.sqlite", {"meta": {"contextStart": day.isoformat(), "contextEnd": day.isoformat()}}, core)


def test_hac9_gate_survives_hac28_sensitivity_unavailable(monkeypatch):
    import agri_data.direct_horizon as dh
    orig = dh._dm
    def fake(diff, bandwidth):
        if bandwidth == 28:
            return {"status": "zero_or_invalid_hac_variance", "n": len(diff), "meanLossDifference": 0.0, "pValue": 1.0}
        return orig(diff, bandwidth)
    monkeypatch.setattr(dh, "_dm", fake)
    d = {dt.date(2020, 1, 1) + dt.timedelta(days=i): float(i % 7) for i in range(40)}
    assert dh._dm(d, 9)["status"] == "tested"
    assert dh._dm(d, 28)["status"] == "zero_or_invalid_hac_variance"


def test_complete_build_fallback_common_pairs_and_serialization(monkeypatch, tmp_path):
    import agri_data.direct_horizon as dh
    import agri_data.modeling as modeling
    import agri_data.recent_window as rw
    start = dt.date(2022, 7, 1)
    series = {start + dt.timedelta(days=i): (100.0 + i % 9, 10.0) for i in range(230)}
    targets = [d for d in series if rs.EVAL_START <= d <= dt.date(2023, 2, 10)]
    ridge = {("radish", h, d): 95.0 for h in rs.HORIZONS for d in targets}
    actual = {("radish", h, d): series[d][0] for h in rs.HORIZONS for d in targets}
    persistence = {"radish": {(d, h): 97.0 for d in targets for h in rs.HORIZONS}}
    recompose = {"pairedRows": [{"product": "radish", "horizon": h, "origin": (d-dt.timedelta(days=h)).isoformat(), "target": d.isoformat(), "actual": series[d][0], "candidatePrediction": 96.0}
                                 for h in rs.HORIZONS for d in targets if d != targets[3]]}
    monkeypatch.setattr(dh, "_observations", lambda prices: {"radish": series})
    monkeypatch.setattr(dh, "_forecast_maps", lambda forecasts: ({}, {}, persistence))
    monkeypatch.setattr(rw, "_direct_ridge_reference", lambda report: (ridge, actual))
    def feature(obs, origin, target, horizon, cache):
        if target == targets[2]: return None
        day = max((d for d in obs if d <= origin), default=None)
        if day is None or (origin-day).days > modeling.MAX_SOURCE_AGE_DAYS: return None
        return day, (math.log(obs[day][0]), float(horizon))
    class NonfiniteModel:
        def fit(self, x, y): return self
        def predict(self, x): return [float("inf")] * len(x)
    monkeypatch.setattr(modeling, "_feature_row", feature)
    monkeypatch.setattr(modeling, "_new_estimator", lambda model_id: NonfiniteModel())
    monkeypatch.setattr(dh, "_dm", lambda diff, bandwidth: ({"status": "tested", "n": len(diff), "meanLossDifference": -1.0, "pValue": 0.001} if bandwidth == 9 else {"status": "zero_or_invalid_hac_variance", "n": len(diff), "meanLossDifference": -1.0, "pValue": 1.0}))
    result = rs.build({}, {}, {}, {}, recompose)
    rows = result["pairedRows"]
    assert rows
    assert all(r["candidate"] == r["directRidge"] and r["fallback"] for r in rows)
    assert all(math.isfinite(r["candidate"]) for r in rows)
    assert all(x["inferenceAvailable"] for x in result["jointConclusions"] if x["horizon"] == 1)
    assert any(x["dm"]["28"]["status"] != "tested" for x in result["comparisons"] if x["horizon"] == 1)
    # A target without pinned base features still receives the available frozen Ridge fallback.
    assert all(any(r["target"] == targets[2].isoformat() and r["fallback"] for r in rows if r["horizon"] == h) for h in (1, 2))
    # Missing market-recomposition rows stay outside the all-arm paired sample and coverage is independent.
    assert all(r["target"] != targets[3].isoformat() for r in rows)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    # Exercise the real SQLite extraction audit (whose source includes a None key),
    # the built report, and main's stable JSON writer together without live inputs.
    day = _synthetic_db(tmp_path / "pipeline.sqlite")
    extracted, provenance = rs._extract_market_data(
        tmp_path / "pipeline.sqlite",
        {"meta": {"contextStart": day.isoformat(), "contextEnd": day.isoformat()}},
        {"radish": {day: (7.5, 20.0)}},
    )
    assert provenance["broadScopeAudit"]["selectedStates"]["missing"] == 0
    result["inputs"] = provenance
    monkeypatch.setattr(rs, "run", lambda: result)
    output = tmp_path / "serialized.json"
    assert rs.main(["--output", str(output)]) == 0
    saved = output.read_bytes()
    assert json.loads(saved)["inputs"]["broadScopeAudit"]["selectedStates"]["missing"] == 0
    import pytest
    with pytest.raises(FileExistsError):
        rs.main(["--output", str(output)])
    assert output.read_bytes() == saved
