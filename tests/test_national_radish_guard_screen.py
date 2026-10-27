import csv
import datetime as dt
from decimal import Decimal
import io
import json
import zipfile

import pytest

from agri_data import national_radish_guard_screen as screen
from agri_data.national_transfer import EXPECTED_HEADERS, PERIODS


def _row(day, *, code="1101", name="무", qty="100", amount="1000", market="01"):
    return {"경매일자": day.strftime("%Y%m%d"), "도매시장코드": market,
            "도매시장명": "시장" + market, "품목코드": code, "품목명": name,
            "변경총물량": qty, "경매낙찰총금액": amount}


def _zip(rows_by_period):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, rows in enumerate(rows_by_period):
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=EXPECTED_HEADERS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(f"period-{index}.csv", stream.getvalue().encode("cp949"))
    return output.getvalue()


def _screen(rows):
    period_rows = [list(group) for group in rows]
    for i, (start, _) in enumerate(PERIODS):
        if not period_rows[i]:
            period_rows[i].append(_row(start, code="1001", name="배추"))
    return screen.build_report(_zip(period_rows), expected_sha256=None)


def test_flags_only_after_14_prior_eligible_observations_and_keeps_window_dates():
    start = PERIODS[0][0]
    rows = [_row(start + dt.timedelta(days=i)) for i in range(15)]
    # A bad-name day is retained as evidence but cannot become a prior observation.
    rows.append(_row(start + dt.timedelta(days=15), name="다른 무", qty="1", amount="40"))
    rows.append(_row(start + dt.timedelta(days=16), qty="0.9", amount="40"))
    report = _screen([rows, [], [], []])
    daily = {row["date"]: row for row in report["dailyObservations"]}
    flagged_date = (start + dt.timedelta(days=16)).isoformat()
    flagged = daily[flagged_date]
    assert report["flaggedDates"] == [flagged_date]
    assert flagged["priorEligibleObservationCount"] == 15
    assert flagged["priorEligibleDates"] == [(start + dt.timedelta(days=i)).isoformat() for i in range(15)]
    assert flagged["prior28MedianNetQuantity"] == "100"
    assert flagged["netQuantityToPriorMedianRatio"] == "0.009"
    assert flagged["priceToPriorMedianRatio"] == str(Decimal("40") / Decimal("0.9") / Decimal("10"))
    assert daily[(start + dt.timedelta(days=15)).isoformat()]["exclusionReason"] == "product_name_not_exact"
    assert report["counts"]["flaggedDates"] == 1
    json.dumps(report, allow_nan=False)


def test_calendar_gaps_do_not_create_observations_and_signed_anomalies_are_retained():
    start = PERIODS[0][0]
    rows = [_row(start + dt.timedelta(days=i * 2)) for i in range(13)]
    # Negative quantity leaves a signed aggregate that is ineligible and is not imputed over.
    rows.append(_row(start + dt.timedelta(days=26), qty="-5", amount="-50"))
    rows.append(_row(start + dt.timedelta(days=28), qty="0.5", amount="40"))
    report = _screen([rows, [], [], []])
    last = report["dailyObservations"][-1]
    assert last["priorEligibleObservationCount"] == 13
    assert last["prior28ObservationWindowAvailable"] is False
    assert last["flagged"] is False
    anomaly = next(row for row in report["dailyObservations"] if row["date"] == (start + dt.timedelta(days=26)).isoformat())
    assert anomaly["source"]["quantitySum"] == "-5"
    assert anomaly["source"]["negativeQuantityCount"] == 1
    assert anomaly["sourceAnomaly"] is True
    assert (start + dt.timedelta(days=26)).isoformat() in report["sourceAnomalyDates"]


def test_thresholds_are_strict_on_quantity_and_inclusive_on_price_ratio():
    start = PERIODS[0][0]
    rows = [_row(start + dt.timedelta(days=i)) for i in range(14)]
    rows.append(_row(start + dt.timedelta(days=14), qty="0.9", amount="40"))
    rows.append(_row(start + dt.timedelta(days=15), qty="0.9", amount="2.25"))
    report = _screen([rows, [], [], []])
    assert report["dailyObservations"][-2]["flagged"] is True
    assert report["dailyObservations"][-1]["flagged"] is True
    assert report["dailyObservations"][-1]["priceToPriorMedianRatio"] == "0.25"


def test_create_only_writer_refuses_overwrite(tmp_path):
    report = {"ok": True}
    path = tmp_path / "report.json"
    screen.write_create_only(path, report)
    assert json.loads(path.read_text()) == report
    with pytest.raises(FileExistsError):
        screen.write_create_only(path, {"ok": False})
    assert json.loads(path.read_text()) == report
