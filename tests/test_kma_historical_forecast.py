import hashlib
import io
import unittest
import zipfile

from agri_data.kma_historical_forecast import parse_kma_forecast_zip


NAME = "seoul-20250901-20250907-request2.zip"
HEADER = "format: day(UTC),hour(UTC),forecast,value,day(KST),hour(KST) location:60_127 Start : 20250901\r\n"


def archive(csv_bytes, members=None):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in (members or [("forecast.csv", csv_bytes)]):
            zf.writestr(name, payload)
    return out.getvalue()


def row(utc_day="1", utc_hour="0100", lead="+6", value="28.000000", kst_day="1", kst_hour="1000"):
    return f"{utc_day},{utc_hour},{lead},{value},{kst_day},{kst_hour}\r\n".encode()


class KmaHistoricalForecastTests(unittest.TestCase):
    def test_utc_kst_conversion_valid_time_and_strict_ten_am_cutoff(self):
        data = (HEADER.encode() + row(utc_hour="0059", kst_hour="0959")
                + row(utc_hour="0100", kst_hour="1000", value="29")
                + row(utc_hour="0101", kst_hour="1001", value="30"))
        result = parse_kma_forecast_zip(archive(data), NAME)
        first, second, third = result.records
        self.assertEqual(first.issue_utc.isoformat(), "2025-09-01T00:59:00+00:00")
        self.assertEqual(first.issue_kst.isoformat(), "2025-09-01T09:59:00+09:00")
        self.assertEqual(first.valid_kst.isoformat(), "2025-09-01T15:59:00+09:00")
        self.assertTrue(first.eligible_at_10_kst)
        self.assertEqual(second.issue_kst.hour, 10)
        self.assertFalse(second.eligible_at_10_kst)
        self.assertEqual(third.issue_kst.strftime("%H:%M"), "10:01")
        self.assertFalse(third.eligible_at_10_kst)

    def test_daily_markers_resolve_cross_month_days_and_are_range_checked(self):
        name = "seoul-20250830-20250902-request2.zip"
        first_header = HEADER.replace("20250901", "20250830").encode()
        next_header = b"Start : 20250831\r\n"
        # UTC Aug 31 23:00 is KST Sep 1 08:00, encoded as day 32 by KMA.
        result = parse_kma_forecast_zip(
            archive(first_header + row(utc_day="30", kst_day="30") + next_header + row(utc_day="31", utc_hour="2300", kst_day="32", kst_hour="0800")), name
        )
        record = result.records[1]
        self.assertEqual(record.issue_utc.date().isoformat(), "2025-08-31")
        self.assertEqual(record.issue_kst.date().isoformat(), "2025-09-01")
        outside_header = HEADER.encode() + b"Start : 20250908\r\n"
        with self.assertRaisesRegex(ValueError, "outside archive date range"):
            parse_kma_forecast_zip(archive(outside_header + row()), NAME)

    def test_kst_ordinal_day_rolls_across_month_and_year(self):
        name = "seoul-20251230-20260102-request2.zip"
        dec30 = HEADER.replace("20250901", "20251230").encode()
        dec31 = b"Start : 20251231\r\n"
        jan1 = b"Start : 20260101\r\n"
        payload = (
            dec30 + row(utc_day="30", kst_day="30")
            + dec31 + row(utc_day="31", utc_hour="1700", kst_day="32", kst_hour="0200")
            + jan1 + row(utc_day="1", kst_day="1")
        )
        result = parse_kma_forecast_zip(archive(payload), name)
        self.assertEqual(result.records[1].issue_kst.isoformat(), "2026-01-01T02:00:00+09:00")
        self.assertEqual(result.records[2].issue_kst.date().isoformat(), "2026-01-01")

    def test_legacy_four_field_rows_and_missing_dates_are_reported(self):
        name = "seoul-20241126-20241128-request2.zip"
        nov26 = HEADER.replace("20250901", "20241126").encode()
        nov28 = b"Start : 20241128\r\n"
        legacy = b"26,0200,+6,5.000000\r\n"
        current = b"28,0200,+6,6.000000,28,1100\r\n"
        result = parse_kma_forecast_zip(archive(nov26 + legacy + nov28 + current), name)
        first, second = result.records
        self.assertEqual(first.source_field_count, 4)
        self.assertEqual(first.kst_timestamp_source, "derived_from_utc_plus_9")
        self.assertEqual(first.issue_kst.isoformat(), "2024-11-26T11:00:00+09:00")
        self.assertEqual(second.source_field_count, 6)
        self.assertEqual(second.kst_timestamp_source, "source_columns")
        self.assertEqual(second.issue_kst.isoformat(), "2024-11-28T11:00:00+09:00")
        self.assertEqual(result.coverage_evidence["missing_marker_dates"], ["2024-11-27"])
        self.assertEqual(result.coverage_evidence["missing_issue_dates"], ["2024-11-27"])

    def test_row_width_must_match_kma_kst_column_switch_date(self):
        name = "seoul-20241128-20241128-request2.zip"
        header = HEADER.replace("20250901", "20241128").encode()
        with self.assertRaisesRegex(ValueError, "four-field row is on or after"):
            parse_kma_forecast_zip(archive(header + b"28,0200,+6,6.000000\n"), name)
        prior_name = "seoul-20241127-20241127-request2.zip"
        prior_header = HEADER.replace("20250901", "20241127").encode()
        with self.assertRaisesRegex(ValueError, "six-field row precedes"):
            parse_kma_forecast_zip(archive(prior_header + b"27,0200,+6,5.000000,27,1100\n"), prior_name)

    def test_observed_invalid_day_sentinel_is_quarantined_without_inference(self):
        name = "seoul-20240104-20240104-request2.zip"
        header = HEADER.replace("20250901", "20240104").encode()
        raw_row = b"-971917344,0200,+6,0.000000"
        result = parse_kma_forecast_zip(archive(header + raw_row), name)
        self.assertEqual(result.records, ())
        self.assertEqual(len(result.quarantined_rows), 1)
        quarantined = result.quarantined_rows[0]
        self.assertEqual(quarantined.line_number, 2)
        self.assertEqual(quarantined.marker_date.isoformat(), "2024-01-04")
        self.assertEqual(quarantined.raw_line, raw_row.decode())
        self.assertIn("issue date unresolved", quarantined.reason)
        self.assertEqual(result.raw_member, header + raw_row)
        self.assertEqual(result.coverage_evidence["missing_issue_dates"], ["2024-01-04"])
        self.assertEqual(result.coverage_evidence["quarantined_row_count"], 1)
        self.assertIsNone(result.coverage_evidence["first_issue_utc"])
        with self.assertRaisesRegex(ValueError, "invalid UTC calendar day"):
            parse_kma_forecast_zip(archive(header + b"-1,0200,+6,0.000000\n"), name)
        with self.assertRaisesRegex(ValueError, "unsupported field count"):
            parse_kma_forecast_zip(archive(header + b"-971917344,0200,+6,0.000000,4,1100\n"), name)

    def test_marker_day_disagreement_is_quarantined_not_reassigned(self):
        name = "seoul-20240602-20240602-request2.zip"
        header = HEADER.replace("20250901", "20240602").encode()
        raw_row = b"23,0200,+6,28.000000"
        result = parse_kma_forecast_zip(archive(header + raw_row), name)
        self.assertEqual(result.records, ())
        self.assertEqual(len(result.quarantined_rows), 1)
        quarantined = result.quarantined_rows[0]
        self.assertEqual(quarantined.line_number, 2)
        self.assertEqual(quarantined.marker_date.isoformat(), "2024-06-02")
        self.assertEqual(quarantined.raw_line, raw_row.decode())
        self.assertIn("UTC day 23 disagrees", quarantined.reason)
        self.assertEqual(result.coverage_evidence["missing_issue_dates"], ["2024-06-02"])
        self.assertEqual(result.coverage_evidence["quarantined_row_count"], 1)
        with self.assertRaisesRegex(ValueError, "invalid HHMM"):
            parse_kma_forecast_zip(archive(header + b"23,2500,+6,28.000000"), name)

    def test_repeated_full_cycles_deduplicate_exact_rows_and_reject_conflicts(self):
        name = "seoul-20250901-20250902-request1.zip"
        day1 = HEADER.encode()
        day2 = HEADER.replace("20250901", "20250902").encode()
        data = day1 + row() + day2 + row(utc_day="2", kst_day="2")
        result = parse_kma_forecast_zip(archive(data + day1 + row() + day2 + row(utc_day="2", kst_day="2")), name)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.exact_duplicate_count, 2)
        bad_cycle = data + day1 + row(value="29")
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            parse_kma_forecast_zip(archive(bad_cycle), name)
        out_of_order = day1 + row() + HEADER.replace("20250901", "20250903").encode() + row(utc_day="3", kst_day="3")
        with self.assertRaisesRegex(ValueError, "outside archive date range"):
            parse_kma_forecast_zip(archive(out_of_order), name)

    def test_combined_end_marker_and_repeated_header_starts_second_cycle(self):
        name = "seoul-20250901-20250903-request1.zip"
        day1_header = HEADER
        day2_marker = b"Start : 20250902\r\n"
        combined = (
            b"Start : 20250903 format: day(UTC),hour(UTC),forecast,value,day(KST),hour(KST) "
            b"location:60_127 Start : 20250901\r\n"
        )
        payload = (
            day1_header.encode() + row()
            + day2_marker + row(utc_day="2", kst_day="2")
            + combined + row()
            + day2_marker + row(utc_day="2", kst_day="2")
            + b"Start : 20250903\r\n" + row(utc_day="3", kst_day="3")
        )
        result = parse_kma_forecast_zip(archive(payload), name)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.exact_duplicate_count, 2)
        self.assertEqual(result.coverage_evidence["start_marker_cycles_observed"], 2)
        self.assertEqual(result.coverage_evidence["combined_marker_header_lines"], 1)
        self.assertEqual(result.coverage_evidence["missing_marker_dates"], [])

    def test_exact_duplicate_is_counted_but_conflicting_duplicate_rejected(self):
        one = row()
        result = parse_kma_forecast_zip(archive(HEADER.encode() + one + one), NAME)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.exact_duplicate_count, 1)
        self.assertEqual(result.raw_member, HEADER.encode() + one + one)
        self.assertEqual(result.archive_sha256, hashlib.sha256(archive(HEADER.encode() + one + one)).hexdigest())
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            parse_kma_forecast_zip(archive(HEADER.encode() + one + row(value="29")), NAME)

    def test_accepts_valid_unterminated_final_row_and_rejects_malformed_or_nonfinite_rows(self):
        valid_unterminated = (HEADER.encode() + row().rstrip(b"\r\n"))
        result = parse_kma_forecast_zip(archive(valid_unterminated), NAME)
        self.assertEqual(len(result.records), 1)
        padded = valid_unterminated + b"   \r\n\t  "
        padded_result = parse_kma_forecast_zip(archive(padded), NAME)
        self.assertEqual(padded_result.raw_member, padded)
        self.assertEqual(len(padded_result.records), 1)
        with self.assertRaisesRegex(ValueError, "blank line inside CSV"):
            parse_kma_forecast_zip(archive(HEADER.encode() + b"   \r\n" + row()), NAME)
        for bad_row in (b"1,0100,+6,28,1", b"1,0100,+6,nan,1,1000", b'1,0100,+6,"28,1,1000'):
            with self.subTest(bad_row=bad_row), self.assertRaises(ValueError):
                parse_kma_forecast_zip(archive(HEADER.encode() + bad_row), NAME)

    def test_checks_entry_count_filename_and_location(self):
        payload = HEADER.encode() + row()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_kma_forecast_zip(archive(payload, [("a.csv", payload), ("b.csv", payload)]), NAME)
        with self.assertRaisesRegex(ValueError, "location"):
            parse_kma_forecast_zip(archive(payload), NAME.replace("seoul", "busan"))
        with self.assertRaisesRegex(ValueError, "location code"):
            parse_kma_forecast_zip(archive(payload.replace(b"60_127", b"50_123")), NAME)

    def test_rejects_inconsistent_utc_kst_and_bad_zip(self):
        inconsistent = HEADER.encode() + row(kst_hour="1100")
        with self.assertRaisesRegex(ValueError, "timestamps disagree"):
            parse_kma_forecast_zip(archive(inconsistent), NAME)
        with self.assertRaisesRegex(ValueError, "invalid or truncated ZIP"):
            parse_kma_forecast_zip(b"not a zip", NAME)


if __name__ == "__main__":
    unittest.main()
