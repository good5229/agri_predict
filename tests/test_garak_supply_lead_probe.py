import datetime as dt
import hashlib
import json

import pytest

from agri_data import garak_supply_lead_probe as probe


class Response:
    def __init__(self, body=b"", status=200):
        self.body, self.status_code, self.closed = body, status, False
        self.headers = {"Set-Cookie": "must-not-be-recorded"}

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 7):
            yield self.body[i:i + 7]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = iter(responses), [], {}

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def _at(hour=9, minute=0):
    return lambda: dt.datetime(2026, 10, 9, hour - 9, minute, tzinfo=dt.timezone.utc)


def test_matrix_and_dry_run_are_frozen(capsys):
    assert len(probe.request_matrix()) == 12
    assert [x["dueDate"] for x in probe.request_matrix()[:6]] == list(probe.DUE_DATES)
    assert {x["marketKind"] for x in probe.request_matrix()} == {1, 2}
    assert probe.main([]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["requestCount"] == 12 and summary["requestSent"] is False


def test_collect_preserves_raw_and_times_but_not_response_headers(tmp_path):
    data = [{"pumjongNm": name, "pumjongCd": str(i), "totqty": f"{i}.25", "alltotqty": "9"}
            for i, name in enumerate(("배추", "무", "양파", "깐마늘", "건고추"), 1)]
    raw = json.dumps({"data": data}, ensure_ascii=False).encode()
    session = Session([Response(raw) for _ in range(12)])
    manifests = probe.collect(tmp_path, session=session, now=_at(),
                              today_kst=lambda: probe.EXECUTION_DATE, sleep=lambda _: None)
    assert len(session.calls) == len(manifests) == 12
    assert all(call[0] == probe.ENDPOINT and call[1]["verify"] is True
               and call[1]["allow_redirects"] is False for call in session.calls)
    for m in manifests:
        attempt = m["attempts"][0]
        saved = json.loads((tmp_path / attempt["manifestFile"]).read_text())
        assert saved["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (tmp_path / saved["rawFile"]).read_bytes() == raw
        assert saved["retrievedAtKst"].startswith("2026-10-09T09:00")
        assert "responseHeaders" not in saved
        assert saved["result"]["cropPresence"] == {x: True for x in probe.TARGET_CROPS}
    concise = json.dumps(probe._concise(manifests), ensure_ascii=False)
    assert "totqty" not in concise and "pumjongNm" not in concise
    assert "건고추" in concise


@pytest.mark.parametrize("hour,minute", [(10, 0), (10, 1)])
def test_execute_time_window_is_enforced(tmp_path, hour, minute):
    session = Session([])
    now = _at(hour, minute)
    with pytest.raises(probe.CollectionError):
        probe.collect(tmp_path, session=session, now=now,
                      today_kst=lambda: probe.EXECUTION_DATE, sleep=lambda _: None)
    assert session.calls == []


def test_execution_is_disabled_on_other_kst_dates(tmp_path):
    session = Session([])
    with pytest.raises(probe.CollectionError):
        probe.collect(tmp_path, session=session, now=_at(), today_kst=lambda: dt.date(2026, 10, 10))
    assert not session.calls


def test_429_preserves_attempt_and_stops_without_spending_more_request_budget(tmp_path):
    raw = b'{"data":[]}'
    session = Session([Response(raw, status=429)] + [Response(b'{"data":[]}') for _ in range(11)])
    with pytest.raises(probe.CollectionError, match="429"):
        probe.collect(tmp_path, session=session, now=_at(),
                      today_kst=lambda: probe.EXECUTION_DATE, sleep=lambda _: None)
    assert len(session.calls) == 1
    manifests = list(tmp_path.glob("*.manifest.json"))
    raws = list(tmp_path.glob("*.raw"))
    assert len(manifests) == len(raws) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["httpStatus"] == 429
    assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
