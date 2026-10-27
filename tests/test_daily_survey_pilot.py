import json
from urllib.parse import parse_qs, urlparse

from agri_data import daily_survey_pilot as pilot


class Response:
    status = 200

    def __init__(self, document):
        self.headers = {"Content-Type": "application/json", "X-Trace": "mock"}
        self.payload = json.dumps(document).encode()

    def read(self, size=-1):
        return self.payload[:size]

    def close(self):
        pass


class MockOpener:
    def __init__(self, callback):
        self.callback = callback
        self.urls = []

    def open(self, request, timeout):
        self.urls.append(request.full_url)
        return Response(self.callback(parse_qs(urlparse(request.full_url).query)))


def doc(page, total, items, size=1):
    return {"header": {"resultCode": "00", "resultMsg": "OK"}, "body": {
        "pageNo": page, "numOfRows": size, "totalCount": total, "items": {"item": items}}}


def row(item="211", day="2025-09-01"):
    return {"exmn_ymd": day, "se_cd": "1", "se_nm": "도매", "ctgry_cd": "200", "ctgry_nm": "채소류",
            "item_cd": item, "item_nm": "배추", "vrty_cd": None, "vrty_nm": None, "grd_cd": None,
            "grd_nm": None, "sgg_cd": None, "sgg_nm": None, "unit": "kg", "unit_sz": "1",
            "mrkt_cd": "M", "mrkt_nm": "시장", "exmn_dd_prc": "1000", "exmn_dd_cnvs_prc": "1000.00",
            "orgnl_reg_dt": "2025-09-01 09:00:00"}


def envfile(root):
    env = root / "pilot.env"
    env.write_text("OTHER=do-not-load\nDATA_GO_KR_SERVICE_KEY_DECODED=TEST_SECRET\n", encoding="utf-8")
    return env


def test_default_plan_is_bounded_and_has_no_live_io(tmp_path):
    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path)
    assert result["mode"] == "dry_run"
    assert result["live_requests"] == 0
    assert len(result["requests"]) == 6
    assert {r["item"] for r in result["requests"]} == {"211", "231", "241"}


def test_collects_all_pages_preserving_raw_hash_and_quality(tmp_path):
    calls = {}

    def response(q):
        item, page = q["cond[item_cd::EQ]"][0], int(q["pageNo"][0])
        calls[(item, page)] = calls.get((item, page), 0) + 1
        total = 2 if item == "211" and q["cond[exmn_ymd::GTE]"][0] == "2025-09-01" else 0
        return doc(page, total, [row(item)] if page <= total else [], int(q["numOfRows"][0]))

    opener = MockOpener(response)
    report = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, page_size=1,
                       opener=opener, sleep=lambda _: None, clock=lambda: 10.0)
    assert report["complete"]
    assert sum(v for (item, page), v in calls.items() if item == "211" and page == 1) == 2
    assert sum(v for (item, page), v in calls.items() if item == "211" and page == 2) == 1
    assert report["pages_saved"] == 7
    assert "TEST_SECRET" not in json.dumps(report)
    checkpoint = json.loads((tmp_path / "raw/daily-survey/checkpoint.json").read_text())
    first = next(v for k, v in checkpoint["pages"].items() if k.startswith("2025-09-01_2025-09-07_211_"))
    raw = (tmp_path / "raw/daily-survey" / first["raw_file"]).read_bytes()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == first["sha256"]
    assert first["quality"] == [{"index": 0, "issues": []}]
    assert first["retrieved_at"].endswith("Z") and first["endpoint"] == pilot.ENDPOINT
    attempts = [json.loads(line) for line in (tmp_path / "raw/daily-survey/attempts.jsonl").read_text().splitlines()]
    assert all("serviceKey" not in json.dumps(a) and "TEST_SECRET" not in json.dumps(a) for a in attempts)


def test_empty_response_is_distinct_and_resumes_without_redownload(tmp_path):
    opener = MockOpener(lambda q: doc(int(q["pageNo"][0]), 0, [], int(q["numOfRows"][0])))
    kwargs = dict(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, opener=opener,
                  sleep=lambda _: None, clock=lambda: 1.0)
    result = pilot.run(**kwargs)
    assert result["complete"] and result["pages_saved"] == 6
    assert all(x["state"] == "ok_empty" for x in result["outcomes"])
    calls = len(opener.urls)
    resumed = pilot.run(**kwargs)
    assert resumed["complete"]
    assert len(opener.urls) == calls


def test_auth_failure_is_preserved_without_retry(tmp_path):
    from urllib.error import HTTPError

    class Denied:
        def __init__(self):
            self.calls = 0

        def open(self, request, timeout):
            self.calls += 1
            raise HTTPError(request.full_url, 403, "denied", {}, None)

    opener = Denied()
    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True,
                       opener=opener, sleep=lambda _: None, clock=lambda: 1.0)
    assert not result["complete"]
    assert opener.calls == 6  # One attempt for each of the six frozen query scopes.
    assert result["outcomes"][0]["state"] == "auth_rejected"


def test_pagination_mismatch_does_not_complete_or_checkpoint_page(tmp_path):
    opener = MockOpener(lambda q: doc(int(q["pageNo"][0]), 2, [], int(q["numOfRows"][0])))
    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, page_size=1,
                       opener=opener, sleep=lambda _: None, clock=lambda: 1.0)
    assert not result["complete"]
    checkpoint = json.loads((tmp_path / "raw/daily-survey/checkpoint.json").read_text())
    assert not checkpoint["pages"]
    assert all(o["state"] == "pagination_count_mismatch" for o in result["outcomes"])


def test_malformed_success_is_not_interpreted_as_empty(tmp_path):
    opener = MockOpener(lambda q: {"header": {"resultCode": "00"}, "body": {
        "pageNo": 1, "numOfRows": 1000, "totalCount": 0}})
    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True,
                       opener=opener, sleep=lambda _: None, clock=lambda: 1.0)
    assert not result["complete"]
    assert all(o["state"] == "invalid_response" for o in result["outcomes"])


def test_http_attempts_are_saved_and_429_stops_other_scopes(tmp_path):
    from urllib.error import HTTPError

    class Limited:
        def __init__(self): self.calls = 0
        def open(self, request, timeout):
            self.calls += 1
            raise HTTPError(request.full_url, 429, "limited", {"Retry-After": "0"}, None)

    opener = Limited()
    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True,
                       opener=opener, sleep=lambda _: None, clock=lambda: 1.0)
    assert not result["complete"] and opener.calls == pilot.MAX_RETRIES + 1
    assert result["outcomes"][0]["state"] == "quota_exhausted"
    attempts = [json.loads(x) for x in (tmp_path / "raw/daily-survey/attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == opener.calls and all(a["http_status"] == 429 for a in attempts)
    assert json.loads((tmp_path / "raw/daily-survey/checkpoint.json").read_text())["failures"]


def test_redirect_location_echo_is_not_persisted_in_headers_or_report(tmp_path):
    from io import BytesIO
    from urllib.error import HTTPError

    class Redirect:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 302, "redirect", {"Location": "https://example.test/?serviceKey=TEST_SECRET"}, BytesIO(b"redirect"))

    report = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True,
                       opener=Redirect(), sleep=lambda _: None, clock=lambda: 1.0)
    attempt = json.loads((tmp_path / "raw/daily-survey/attempts.jsonl").read_text().splitlines()[0])
    assert "location" not in {k.lower() for k in attempt["headers"]}
    assert "TEST_SECRET" not in json.dumps(attempt) and "TEST_SECRET" not in json.dumps(report)
    assert (tmp_path / "raw/daily-survey" / attempt["raw_file"]).stat().st_mode & 0o777 == 0o600


def test_long_retry_after_stops_without_early_retry_or_other_scope(tmp_path):
    from urllib.error import HTTPError

    class Limited:
        def __init__(self): self.calls = 0
        def open(self, request, timeout):
            self.calls += 1
            raise HTTPError(request.full_url, 429, "limited", {"Retry-After": "3600"}, None)

    opener = Limited()
    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True,
                       opener=opener, sleep=lambda _: None, clock=lambda: 1.0)
    assert opener.calls == 1
    assert result["outcomes"][0]["state"] == "quota_wait_required"
    assert result["outcomes"][0]["retry_after_seconds"] == 3600


def test_registration_dates_compare_in_kst_at_midnight_and_ten_boundary():
    req = {"item": "211", "start": "2025-09-01", "end": "2025-09-07"}
    item = row()
    item["orgnl_reg_dt"] = "2025-09-01T15:00:00Z"  # 2025-09-02 00:00:00 KST
    issues = pilot._check_row(item, req)
    assert "registered_after_survey_day" in issues
    assert "registered_after_next_day_10_kst" not in issues
    item["orgnl_reg_dt"] = "2025-09-02T01:00:00.000001Z"
    assert "registered_after_next_day_10_kst" in pilot._check_row(item, req)


def test_saved_page_is_redownloaded_when_raw_missing_or_corrupt(tmp_path):
    calls = []
    def response(q):
        calls.append(q)
        item = q["cond[item_cd::EQ]"][0]
        total = 1 if item == "211" and q["cond[exmn_ymd::GTE]"][0] == "2025-09-01" else 0
        return doc(int(q["pageNo"][0]), total, [row(item)] if total else [], int(q["numOfRows"][0]))
    opener = MockOpener(response)
    kwargs = dict(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, opener=opener,
                  sleep=lambda _: None, clock=lambda: 1.0)
    assert pilot.run(**kwargs)["complete"]
    state_path = tmp_path / "raw/daily-survey/checkpoint.json"
    state = json.loads(state_path.read_text())
    first_key = next(k for k in state["pages"] if k.startswith("2025-09-01_2025-09-07_211_"))
    first_raw = tmp_path / "raw/daily-survey" / state["pages"][first_key]["raw_file"]
    first_raw.write_bytes(b"corrupt")
    before = len(opener.urls)
    assert pilot.run(**kwargs)["complete"]
    assert len(opener.urls) == before + 1
    state = json.loads(state_path.read_text())
    first_raw = tmp_path / "raw/daily-survey" / state["pages"][first_key]["raw_file"]
    first_raw.unlink()
    before = len(opener.urls)
    assert pilot.run(**kwargs)["complete"]
    assert len(opener.urls) == before + 1


def test_registration_diagnostics_expose_evaluable_denominators(tmp_path):
    def response(q):
        item = q["cond[item_cd::EQ]"][0]
        matching = item == "211" and q["cond[exmn_ymd::GTE]"][0] == "2025-09-01"
        if matching:
            missing, malformed = row(), row()
            missing["orgnl_reg_dt"] = None
            malformed["orgnl_reg_dt"] = "not-a-timestamp"
            return doc(1, 2, [missing, malformed], int(q["numOfRows"][0]))
        return doc(1, 0, [], int(q["numOfRows"][0]))

    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, page_size=2,
                       opener=MockOpener(response), sleep=lambda _: None, clock=lambda: 1.0)
    diagnostics = result["orgnl_reg_dt_diagnostics"]
    assert diagnostics["denominators"] == {"all_rows": 2, "evaluable_timestamps": 0}
    assert diagnostics["counts"]["missing_or_null_orgnl_reg_dt"] == 1
    assert diagnostics["counts"]["unparsed_orgnl_reg_dt"] == 1
    assert diagnostics["late_fractions_over_evaluable"]["after_survey_day"] is None


def test_registration_denominator_excludes_uncomparable_rows(tmp_path):
    def response(q):
        item = q["cond[item_cd::EQ]"][0]
        matching = item == "211" and q["cond[exmn_ymd::GTE]"][0] == "2025-09-01"
        if matching:
            invalid_date, valid_date = row(), row()
            invalid_date["exmn_ymd"] = "not-a-date"
            valid_date["orgnl_reg_dt"] = "2025-09-03 10:01:00"
            return doc(1, 2, [invalid_date, valid_date], int(q["numOfRows"][0]))
        return doc(1, 0, [], int(q["numOfRows"][0]))

    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, page_size=2,
                       opener=MockOpener(response), sleep=lambda _: None, clock=lambda: 1.0)
    diagnostics = result["orgnl_reg_dt_diagnostics"]
    assert diagnostics["rows"] == 2
    assert diagnostics["evaluable_rows"] == 1
    assert diagnostics["late_fractions_over_evaluable"]["after_next_day_10_kst"] == 1


def test_non_object_item_does_not_enter_registration_denominator(tmp_path):
    def response(q):
        item = q["cond[item_cd::EQ]"][0]
        matching = item == "211" and q["cond[exmn_ymd::GTE]"][0] == "2025-09-01"
        if matching:
            valid = row()
            valid["orgnl_reg_dt"] = "2025-09-03 10:01:00"
            return doc(1, 2, ["not-an-object", valid], int(q["numOfRows"][0]))
        return doc(1, 0, [], int(q["numOfRows"][0]))

    result = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True, page_size=2,
                       opener=MockOpener(response), sleep=lambda _: None, clock=lambda: 1.0)
    diagnostics = result["orgnl_reg_dt_diagnostics"]
    assert diagnostics["rows"] == 2
    assert diagnostics["evaluable_rows"] == 1


def test_transport_failure_attempt_has_metadata_and_raw_hash(tmp_path):
    from urllib.error import URLError

    class Offline:
        def open(self, request, timeout): raise URLError("offline")

    report = pilot.run(env_file=envfile(tmp_path), data_root=tmp_path, execute=True,
                       opener=Offline(), sleep=lambda _: None, clock=lambda: 1.0)
    assert not report["complete"]
    attempts = [json.loads(x) for x in (tmp_path / "raw/daily-survey/attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 6 * (pilot.MAX_RETRIES + 1)
    assert all(a["http_status"] is None and a["transport_error_type"] == "URLError" and a["bytes"] == 0
               for a in attempts)


def test_orgnl_registration_quality_flags_late_and_missing_values():
    late = row()
    late["orgnl_reg_dt"] = "2025-09-03 10:01:00"
    issues = pilot._check_row(late, {"item": "211", "start": "2025-09-01", "end": "2025-09-07"})
    assert "registered_after_next_day_10_kst" in issues
    missing = row()
    missing["orgnl_reg_dt"] = None
    issues = pilot._check_row(missing, {"item": "211", "start": "2025-09-01", "end": "2025-09-07"})
    assert "missing_or_null_orgnl_reg_dt" in issues
