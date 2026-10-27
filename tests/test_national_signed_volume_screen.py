import datetime as dt
import io
import math
import zipfile

import numpy as np
import pytest

from agri_data import national_signed_volume_screen as screen
from agri_data import national_transfer


def _day(day, price=100.0, quantity=10.0, negative_rows=0):
    return {"_price": price, "_quantity": quantity, "negativeQuantityCount": negative_rows}


def test_pair_cutoff_is_o_minus_two_and_negative_signed_volume_is_retained():
    target = dt.date(2026, 6, 20)
    horizon = 5
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=2)
    days = {}
    cursor = origin - dt.timedelta(days=29)
    while cursor <= target:
        days[cursor] = _day(cursor, price=100 + (cursor - dt.date(2026, 1, 1)).days)
        cursor += dt.timedelta(days=1)
    days[cutoff] = _day(cutoff, price=110.0, quantity=10.0)
    days[cutoff + dt.timedelta(days=1)] = _day(cutoff + dt.timedelta(days=1), price=9999.0)
    negative_day = origin - dt.timedelta(days=4)
    days[negative_day] = _day(negative_day, price=105.0, quantity=-5.0, negative_rows=1)

    row = screen._make_pair(days, "radish", target, horizon)

    assert row["p0"] == {"date": cutoff.isoformat(), "price": 110.0}
    assert row["p7ReferenceDate"] == (cutoff - dt.timedelta(days=7)).isoformat()
    assert row["recentWindow"]["signedQuantityMean"] == pytest.approx(55 / 7)
    assert row["recentWindow"]["negativeSignedQuantityDays"] == 1
    assert row["negativeSignedQuantityDaysInFeatureWindows"] == 1
    assert row["state"] == "paired"
    assert math.isfinite(row["x"])


def test_calendar_month_fixed_effect_removes_between_month_rank_association():
    rows = []
    for month in range(1, 13):
        for i in range(20):
            date = dt.date(2025, month, 1) + dt.timedelta(days=i)
            rows.append({"target": date.isoformat(), "x": month * 100 + i,
                         "y": month * 100 + (i * 9) % 20, "z": 1.0})
    raw_rank_corr = float(np.corrcoef(screen._midranks([r["x"] for r in rows]),
                                      screen._midranks([r["y"] for r in rows]))[0, 1])

    result = screen._conditional_spearman(rows)

    assert raw_rank_corr > 0.8
    assert result["status"] == "computed"
    assert abs(result["coefficient"]) < 0.1
    assert "targetMonth=12" in result["controlColumns"]


@pytest.mark.parametrize("mode", ["x_equals_z", "constant_x", "month_only_x"])
def test_numerically_zero_x_residual_is_unavailable(mode):
    rows = []
    for i in range(120):
        month = i // 10 + 1
        date = dt.date(2025, month, 1) + dt.timedelta(days=i % 10)
        z = float(i % 37)
        if mode == "x_equals_z":
            x = z
        elif mode == "constant_x":
            x = 7.0
        else:
            x = float(month)
        rows.append({"target": date.isoformat(), "x": x,
                     "y": float((i * 17) % 121), "z": z if mode == "x_equals_z" else 1.0})

    result = screen._conditional_spearman(rows)

    assert result["status"] == "zero_residual_variance"
    assert result["coefficient"] is None
    assert result["residualStdX"] <= result["residualZeroTolerance"]


def test_frozen_threshold_requires_all_blocks_and_recent_negative_signal():
    coefficients = [-0.20, -0.12, 0.04, -0.10]
    blocks = [{"blockStart": start.isoformat(), "pairCount": 30, "coverage": 0.70,
               "conditionalSpearman": {"coefficient": coefficient}}
              for (start, _), coefficient in zip(screen.BLOCKS, coefficients)]
    passed, assessment = screen._block_pass(blocks)
    assert passed
    assert assessment["negativeBlockCount"] == 3
    assert assessment["latestBlockIncluded"] is True

    blocks[-1]["coverage"] = 0.69
    assert screen._block_pass(blocks)[0] is False
    blocks[-1]["coverage"] = 0.70
    blocks[-1]["conditionalSpearman"]["coefficient"] = 0.051
    assert screen._block_pass(blocks)[0] is False


def _synthetic_archive(*, missing_last_period=False):
    headers = national_transfer.EXPECTED_HEADERS
    periods = [
        ("p0.csv", "20240708"),
        ("p1.csv", "20250101"),
        ("p2.csv", "20250701"),
        ("p3.csv", "20260101"),
    ]
    if missing_last_period:
        periods[-1] = ("p3.csv", "20250702")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, day in periods:
            line = ",".join(headers) + "\n"
            line += f"{day},001,시장,1101,무,10,1000\n"
            archive.writestr(name, line.encode("cp949"))
    return buffer.getvalue()


def test_build_report_rejects_wrong_hash_and_incomplete_period_source():
    archive = _synthetic_archive()
    with pytest.raises(ValueError, match="SHA-256"):
        screen.build_report(archive)
    with pytest.raises(ValueError, match="duplicate archive period|missing frozen periods"):
        national_transfer.parse_archive(_synthetic_archive(missing_last_period=True), expected_sha256=None)
