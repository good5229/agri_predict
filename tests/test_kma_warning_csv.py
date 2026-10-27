import datetime as dt
import hashlib
from pathlib import Path

import pytest

from agri_data.kma_warning_csv import WarningCsvError, build_manifest, main, parse_csv


HEADER = "발표시간,지역,발효시각,해당지역,내용\r\n"


def _write(tmp_path: Path, body: str, *, encoding: str = "cp949") -> Path:
    path = tmp_path / "warnings.csv"
    path.write_bytes((HEADER + body).encode(encoding))
    return path


def test_cp949_preserves_source_strings_and_blank_content(tmp_path):
    path = _write(tmp_path, "2026-10-01 09:59:00,서울,2026-10-01 12:00:00,종로구,\r\n"
                           "2026-10-01 10:00:00,부산,2026-10-01 12:00:00,해운대구,강풍\r\n")
    rows, digest = parse_csv(path)
    assert rows[0].content == ""
    assert rows[0].issue_text == "2026-10-01 09:59:00"
    assert rows[0].issue_time == dt.datetime(2026, 10, 1, 9, 59)
    assert rows[1].content == "강풍"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("body", [
    "2026-10-01 09:00:00,서울,효력,지역\r\n",
    "2026-10-01 09:00:00,서울,효력,지역,내용,extra\r\n",
    "not-a-time,서울,효력,지역,내용\r\n",
])
def test_malformed_rows_are_rejected(tmp_path, body):
    with pytest.raises(WarningCsvError):
        parse_csv(_write(tmp_path, body))


def test_conflicting_duplicate_is_rejected_but_exact_duplicate_is_preserved(tmp_path):
    row = "2026-10-01 09:00:00,서울,2026-10-01 12:00:00,종로구,호우\r\n"
    parsed, _ = parse_csv(_write(tmp_path, row + row))
    assert len(parsed) == 2
    with pytest.raises(WarningCsvError, match="conflicting duplicate"):
        parse_csv(_write(tmp_path, row + row.replace("호우", "강풍")))


def test_10_boundary_prior_day_coverage_and_query_boundary(tmp_path):
    path = _write(tmp_path,
        "2026-10-01 09:59:00,서울,효력,종로구,이전\r\n"
        "2026-10-01 10:00:00,서울,효력,종로구,경계포함\r\n"
        "2026-10-02 09:59:00,부산,효력,해운대구,직전\r\n"
        "2026-10-02 10:00:00,부산,효력,해운대구,경계제외\r\n")
    manifest = build_manifest(path, dt.date(2026, 10, 1), dt.date(2026, 10, 2), ["서울", "부산"])
    # Oct 1 10:00 is omitted: its full prior day is outside the declared interval.
    # Oct 2 10:00 includes exactly [Oct 1 10:00, Oct 2 10:00).
    assert [x["origin_kst"] for x in manifest["ten_kst_origins"]] == ["2026-10-02T10:00"]
    assert manifest["ten_kst_origins"][0]["prior_24h_event_counts_by_region"] == {"부산": 1, "서울": 1}
    assert manifest["before_10_kst_issue_count"] == 2
    assert manifest["counts_by_issue_date"] == {"2026-10-01": 2, "2026-10-02": 2}
    assert "active-warning state" in manifest["coverage_note"]


def test_out_of_range_issue_date_fails_even_if_region_is_unexpected(tmp_path):
    path = _write(tmp_path, "2026-09-30 09:00:00,서울,효력,지역,내용\r\n")
    with pytest.raises(WarningCsvError, match="outside"):
        build_manifest(path, dt.date(2026, 10, 1), dt.date(2026, 10, 2), ["부산"])


def test_unexpected_source_regions_are_rejected_with_counts(tmp_path):
    path = _write(tmp_path,
        "2026-10-01 09:00:00,서울,효력,종로구,내용\r\n"
        "2026-10-01 09:01:00,인천,효력,연수구,내용\r\n"
        "2026-10-01 09:02:00,인천,효력,남동구,내용\r\n")
    with pytest.raises(WarningCsvError, match=r"인천 \(2\)"):
        build_manifest(path, dt.date(2026, 10, 1), dt.date(2026, 10, 1), ["서울", "부산"])


def test_cli_refuses_output_path_that_is_input_or_resolves_to_input(tmp_path):
    path = _write(tmp_path, "2026-10-01 09:00:00,서울,효력,종로구,내용\r\n")
    alias = tmp_path / "alias.csv"
    alias.symlink_to(path)
    for output in (path, alias):
        with pytest.raises(SystemExit) as exc:
            main(["--input", str(path), "--query-start", "2026-10-01",
                  "--query-end", "2026-10-01", "--output", str(output)])
        assert exc.value.code == 2
        assert path.read_bytes().startswith(HEADER.encode("cp949"))
