import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile
import io

from agri_data import seoan_pepper_collection as base
from agri_data import seoan_pepper_completeness as check


class Response:
    def __init__(self,status,body,headers=None):
        self.status_code=status;self.body=body;self.headers=headers or {"Content-Type":"text/html; charset=utf-8"}
        self.closed=False
    def iter_content(self,chunk_size):
        for pos in range(0,len(self.body),13):yield self.body[pos:pos+13]
    def close(self):self.closed=True


class Session:
    def __init__(self,bodies):
        self.bodies=iter(bodies);self.calls=[];self.headers={};self.verify=False
    def get(self,url,**kwargs):
        self.calls.append(("GET",url,kwargs));return next(self.bodies)
    def post(self,url,**kwargs):
        self.calls.append(("POST",url,kwargs));return next(self.bodies)


HEADERS=["경락일","품종","중량단위","물량","가격단위","경락가"]
ROWS=[
    ["2026-08-01","화건","kg","12.50","원/kg","1,234.00"],
    ["2026-08-01","화건","kg","12.50","원/kg","1,234.00"],
    ["2026-08-08","화건(꼭무)","근","-3","원/근","900"],
    ["2026-08-15","화건(손꼭무)","kg","","원/kg","NULL"],
    ["2026-08-22","기타","kg","3","원/kg","1,000"],
]


def _html(rows):
    out=["<table><tr>"+"".join(f"<th>{v}</th>" for v in HEADERS)+"</tr>"]
    for row in rows:out.append("<tr>"+"".join(f"<td>{v}</td>" for v in row)+"</tr>")
    return ("".join(out)+"</table>").encode()


def _synthetic_bodies():
    return [Response(200,b"<html>form</html>"),Response(200,_html(ROWS)),
            Response(200,_html(ROWS[:2])),Response(200,_html(ROWS[2:3])),
            Response(200,_html(ROWS[3:4])),Response(200,_html(ROWS[4:])),
            Response(200,_html([]))]


def test_dry_run_has_fixed_seven_requests_and_sends_none(capsys):
    assert check.main([])==0
    result=json.loads(capsys.readouterr().out)
    assert result["requestSent"] is False
    assert result["landing"]["method"]=="GET"
    assert len(result["exports"])==6
    assert [(r["start"],r["end"]) for r in result["exports"]]==[check.MONTH,*check.WEEKS]
    assert all(r["params"]["searchSpcs"]=="" for r in result["exports"])
    assert result["complete"] is False


def test_header_only_is_normal_empty_but_bad_schema_is_incomplete():
    empty=check._validation(_html([]),*check.WEEKS[-1],"text/html")
    assert empty["state"]=="normal_empty" and empty["usable"] is True and empty["rows"]==[]
    bad=check._validation(b"<table><tr><th>hello</th></tr></table>",*check.WEEKS[-1])
    assert bad["state"]=="parse_incomplete" and bad["usable"] is False
    assert check._validation(b"",*check.WEEKS[-1])["usable"] is False


def test_month_and_disjoint_weeks_compare_raw_multiset_and_resume_without_requests(tmp_path):
    session=Session(_synthetic_bodies())
    sleeps=[]
    result=check.run(tmp_path,session=session,sleep=lambda delay:sleeps.append(delay),
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert session.verify is True
    assert len([c for c in session.calls if c[0]=="GET"])==1
    assert len([c for c in session.calls if c[0]=="POST"])==6
    assert all(c[2]["timeout"]==base.TIMEOUT_SECONDS and c[2]["stream"] is True for c in session.calls)
    assert sleeps==[check.MIN_REQUEST_INTERVAL_SECONDS]*7
    assert [c[2]["data"] for c in session.calls if c[0]=="POST"]==[r["params"] for r in check._request_plan()]
    assert result["requestSetUsable"] is True
    assert result["internalConsistency"] is True
    assert result["exactFullSourceValuesMultisetMatch"] is True
    assert result["monthRowCount"]==result["fiveWeekUnionRowCount"]==5
    names=result["monthCounts"]["frozenHwageonNameCounts"]
    assert names=={"화건":2,"화건(꼭무)":1,"화건(손꼭무)":1}
    week2=next(r for r in result["exports"] if r["requestId"]=="week_02")
    raw=(tmp_path/week2["rawFile"]).read_bytes()
    assert b"-3" in raw
    assert hashlib.sha256(raw).hexdigest()==week2["sha256"]
    assert (tmp_path/week2["manifestFile"]).is_file()
    week3=next(r for r in result["exports"] if r["requestId"]=="week_03")
    assert b"NULL" in (tmp_path/week3["rawFile"]).read_bytes()
    assert result["complete"] is False
    summary_before=(tmp_path/check.SUMMARY_NAME).read_bytes()
    calls=len(session.calls)
    resumed=check.run(tmp_path,session=session,sleep=lambda _:None,
                      now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len(session.calls)==calls
    assert resumed==result
    assert (tmp_path/check.SUMMARY_NAME).read_bytes()==summary_before


def test_retries_are_bounded_archived_and_failed_landing_stops_export(tmp_path):
    session=Session([Response(503,b"busy") for _ in range(check.MAX_ATTEMPTS)])
    result=check.run(tmp_path,session=session,sleep=lambda _:None,
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len(session.calls)==check.MAX_ATTEMPTS
    assert result["landing"]["state"]=="attempts_exhausted"
    assert result["exports"]==[]
    checkpoint=json.loads((tmp_path/check.CHECKPOINT_NAME).read_text())
    assert checkpoint["complete"] is False
    attempts=list(tmp_path.glob("seoan_recheck_landing_attempt*.manifest.json"))
    assert len(attempts)==check.MAX_ATTEMPTS
    assert all(r["state"]=="http_failure" for r in result["landing"]["attemptOutcomes"])


def test_interrupted_unknown_request_is_not_reissued_but_later_weeks_resume(tmp_path):
    checkpoint=check._checkpoint(tmp_path)
    checkpoint["inFlightAttempts"]["month"]=1
    checkpoint["states"]["month"]="in_flight"
    check._atomic_checkpoint(tmp_path/check.CHECKPOINT_NAME,checkpoint)
    session=Session([Response(200,b"<html>form</html>")]+[
        Response(200,_html(rows)) for rows in (ROWS[:2],ROWS[2:3],ROWS[3:4],ROWS[4:],[])])
    result=check.run(tmp_path,session=session,sleep=lambda _:None,
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len([c for c in session.calls if c[0]=="POST"])==5
    month=next(r for r in result["exports"] if r["requestId"]=="month")
    assert month["state"]=="interrupted_outcome_unknown"
    assert result["complete"] is False and result["internalConsistency"] is None


def test_unknown_units_or_out_of_range_rows_are_not_usable():
    body=("<table><tr><th>date</th><th>product</th><th>weight_unit</th><th>weight</th>"
          "<th>price_unit</th><th>price</th></tr><tr><td>2026-08-01</td><td>화건</td>"
          "<td></td><td>2</td><td></td><td>3</td></tr></table>").encode()
    assert check._validation(body,*check.WEEKS[0])["usable"] is False
    outside=_html([["2026-07-31","화건","kg","2","원/kg","3"]])
    assert check._validation(outside,*check.WEEKS[0])["usable"] is False
    for weight_unit,price_unit in (("100kg","원/kg"),("notkg","원/kg"),
                                   ("kg","천원/상자"),("kg","달러(원환산불가)")):
        weird=(f"<table><tr><th>date</th><th>product</th><th>weight_unit</th><th>weight</th>"
               f"<th>price_unit</th><th>price</th></tr><tr><td>2026-08-01</td><td>화건</td>"
               f"<td>{weight_unit}</td><td>2</td><td>{price_unit}</td><td>3</td></tr></table>").encode()
        result=check._validation(weird,*check.WEEKS[0])
        assert result["usable"] is False and "unit_value_unrecognized" in result["issues"]


def test_orphan_raw_capture_is_not_reissued_and_other_requests_continue(tmp_path):
    check._checkpoint(tmp_path)
    orphan=tmp_path/"seoan_recheck_month_attempt01_orphan.raw"
    orphan.write_bytes(b"response captured before manifest crash")
    session=Session([Response(200,b"<html>form</html>")]+[
        Response(200,_html(rows)) for rows in (ROWS[:2],ROWS[2:3],ROWS[3:4],ROWS[4:],[])])
    result=check.run(tmp_path,session=session,sleep=lambda _:None,
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len([c for c in session.calls if c[0]=="POST"])==5
    month=next(r for r in result["exports"] if r["requestId"]=="month")
    assert month["state"]=="orphaned_outcome_unknown"
    assert month["orphanedArtifacts"]==[orphan.name]


def test_malformed_final_attempt_manifest_is_unknown_and_not_reissued(tmp_path):
    check._checkpoint(tmp_path)
    orphan=tmp_path/"seoan_recheck_month_attempt01_corrupt.manifest.json"
    orphan.write_text("{",encoding="utf-8")
    session=Session([Response(200,b"<html>form</html>")]+[
        Response(200,_html(rows)) for rows in (ROWS[:2],ROWS[2:3],ROWS[3:4],ROWS[4:],[])])
    result=check.run(tmp_path,session=session,sleep=lambda _:None,
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len([c for c in session.calls if c[0]=="POST"])==5
    month=next(r for r in result["exports"] if r["requestId"]=="month")
    assert month["state"]=="orphaned_outcome_unknown"
    assert month["orphanedArtifacts"]==[orphan.name]


def test_quota_4xx_stops_remaining_exports_without_retry(tmp_path):
    session=Session([Response(200,b"<html>form</html>"),Response(429,b"rate limited")])
    result=check.run(tmp_path,session=session,sleep=lambda _:None,
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert len(session.calls)==2
    assert result["exports"][0]["state"]=="http_nonretryable_4xx"
    assert result["exports"][0]["attempts"]==1
    assert all(r["state"]=="not_attempted_after_nonretryable_4xx" for r in result["exports"][1:])


def test_all_normal_empty_responses_still_stop_enlargement(tmp_path):
    session=Session([Response(200,b"<html>form</html>")]+[Response(200,_html([])) for _ in range(6)])
    result=check.run(tmp_path,session=session,sleep=lambda _:None,
                     now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc))
    assert result["requestSetUsable"] is True
    assert result["internalConsistency"] is True
    assert result["noUsablePepperSignal"] is True
    assert result["stopEnlargement"] is True
    assert result["complete"] is False


def test_multiple_xlsx_worksheets_fail_closed():
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,"w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml","<worksheet/>")
        archive.writestr("xl/worksheets/sheet2.xml","<worksheet/>")
    result=check._validation(payload.getvalue(),*check.WEEKS[0])
    assert result["usable"] is False
    assert "multiple_worksheets_not_supported" in result["issues"]
