import datetime as dt
import json

from agri_data import seoan_pepper_history as history


HEADERS=["일자","품목","총수량(근)","총수량(kg)","단가(근)","단가(kg)"]
EMPTY_TABLE=("<table><tr>"+"".join(f"<th>{x}</th>" for x in HEADERS)+"</tr></table>").encode()


class Response:
    def __init__(self,status,body):self.status_code=status;self.body=body;self.headers={"Content-Type":"text/html; charset=utf-8"};self.closed=False
    def iter_content(self,chunk_size):yield self.body
    def close(self):self.closed=True


class Session:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[];self.verify=False;self.headers={}
    def get(self,url,**kwargs):self.calls.append(("GET",url,kwargs));return next(self.responses)
    def post(self,url,**kwargs):self.calls.append(("POST",url,kwargs));return next(self.responses)


def test_month_plan_is_exactly_24_complete_calendar_months():
    plan=history.month_plan()
    assert len(plan)==24
    assert (plan[0]["start"],plan[0]["end"]) == ("2024-07-01","2024-07-31")
    assert (plan[-1]["start"],plan[-1]["end"]) == ("2026-06-01","2026-06-30")
    assert all(item["params"]["searchSpcs"]=="" for item in plan)
    assert history.dry_run()["requestSent"] is False


def test_nonretryable_429_stops_before_following_months(tmp_path):
    session=Session([Response(200,b"landing"),Response(429,b"quota")])
    result=history.run(tmp_path,session=session,sleep=lambda _:None,
                       now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len(session.calls)==2
    assert result["stopReason"]=="http_nonretryable_4xx"
    assert result["complete"] is False
    assert len(result["months"])==24
    assert result["months"][0]["state"]=="http_nonretryable_4xx"
    assert all(x["state"]=="not_attempted_after_stop" for x in result["months"][1:])
    again=history.run(tmp_path,session=Session([]),sleep=lambda _:None)
    assert again["stopReason"]=="http_nonretryable_4xx"
    assert again["landing"]["usable"] is True


def test_successful_month_captures_resume_without_network_calls(tmp_path):
    session=Session([Response(200,b"landing")]+[Response(200,EMPTY_TABLE) for _ in range(24)])
    first=history.run(tmp_path,session=session,sleep=lambda _:None,
                      now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len(session.calls)==25
    assert first["complete"] is False
    assert first["allMonthsUsable"] is True
    assert all(m["state"]=="normal_empty" and m["rowCount"]==0 for m in first["months"])
    again=history.run(tmp_path,session=Session([]),sleep=lambda _:None)
    assert again["allMonthsUsable"] is True
    assert len(again["months"])==24


def test_inflight_without_capture_is_unknown_and_never_reissued(tmp_path):
    checkpoint=history._checkpoint(tmp_path)
    checkpoint["inFlight"]="2024-07";checkpoint["inFlightAttempt"]=1
    history._atomic_json(tmp_path/history.CHECKPOINT,checkpoint)
    session=Session([])
    result=history.run(tmp_path,session=session,sleep=lambda _:None)
    assert session.calls==[]
    assert result["stopReason"]=="interrupted_outcome_unknown"
    assert result["months"][0]["state"]=="interrupted_outcome_unknown"


def _archive_landing_attempt(tmp_path,status,body):
    history._checkpoint(tmp_path)
    history._capture(tmp_path,Session([Response(status,body)]),"landing",1,
                     sleep=lambda _:None,now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    checkpoint["inFlight"]="landing";checkpoint["inFlightAttempt"]=1
    checkpoint["states"]["landing"]="in_flight"
    history._atomic_json(tmp_path/history.CHECKPOINT,checkpoint)


def test_archived_landing_429_is_terminal_after_stale_inflight_restart(tmp_path):
    _archive_landing_attempt(tmp_path,429,b"quota")
    session=Session([])
    result=history.run(tmp_path,session=session,sleep=lambda _:None)
    assert session.calls==[]
    assert result["stopReason"]=="http_nonretryable_4xx"
    assert result["landing"]["usable"] is False
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    assert checkpoint["inFlight"] is None and "inFlightAttempt" not in checkpoint


def test_corrupt_archived_landing_raw_is_terminal_after_stale_inflight_restart(tmp_path):
    _archive_landing_attempt(tmp_path,200,b"landing")
    raw=next(tmp_path.glob("seoan_history_landing_attempt*.raw"))
    raw.write_bytes(b"corrupted")
    session=Session([])
    result=history.run(tmp_path,session=session,sleep=lambda _:None)
    assert session.calls==[]
    assert result["stopReason"]=="raw_integrity_failure"
    assert result["landing"]["usable"] is False
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    assert checkpoint["inFlight"] is None and "inFlightAttempt" not in checkpoint


def test_manifest_missing_capture_fields_fails_closed_for_month(tmp_path):
    history._checkpoint(tmp_path)
    history._capture(tmp_path,Session([Response(200,b"landing")]),"landing",1,
                     sleep=lambda _:None,now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    history._capture(tmp_path,Session([Response(200,EMPTY_TABLE)]),"2024-07",1,
                     sleep=lambda _:None,now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    manifest=next(tmp_path.glob("seoan_history_2024-07_attempt*.manifest.json"))
    metadata=json.loads(manifest.read_text())
    for key in ("raw_file","response_bytes","sha256"):metadata.pop(key)
    manifest.write_text(json.dumps(metadata),encoding="utf-8")
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    checkpoint["inFlight"]="2024-07";checkpoint["inFlightAttempt"]=1
    checkpoint["states"]["2024-07"]="in_flight"
    history._atomic_json(tmp_path/history.CHECKPOINT,checkpoint)

    session=Session([])
    result=history.run(tmp_path,session=session,sleep=lambda _:None)
    assert session.calls==[]
    assert result["stopReason"]=="raw_integrity_failure"
    assert result["months"][0]["state"]=="raw_integrity_failure"
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    assert checkpoint["inFlight"] is None and "inFlightAttempt" not in checkpoint


def test_manifest_missing_capture_fields_fails_closed_for_landing(tmp_path):
    history._checkpoint(tmp_path)
    history._capture(tmp_path,Session([Response(200,b"landing")]),"landing",1,
                     sleep=lambda _:None,now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    manifest=next(tmp_path.glob("seoan_history_landing_attempt*.manifest.json"))
    metadata=json.loads(manifest.read_text())
    for key in ("raw_file","response_bytes","sha256"):metadata.pop(key)
    manifest.write_text(json.dumps(metadata),encoding="utf-8")
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    checkpoint["inFlight"]="landing";checkpoint["inFlightAttempt"]=1
    checkpoint["states"]["landing"]="in_flight"
    history._atomic_json(tmp_path/history.CHECKPOINT,checkpoint)

    session=Session([])
    result=history.run(tmp_path,session=session,sleep=lambda _:None)
    assert session.calls==[]
    assert result["stopReason"]=="raw_integrity_failure"
    checkpoint=json.loads((tmp_path/history.CHECKPOINT).read_text())
    assert checkpoint["inFlight"] is None and "inFlightAttempt" not in checkpoint
