import datetime as dt
import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from agri_data import ecmwf_field_compare as compare


class FakeEccodes:
    def __init__(self, *, metadata=None, latitudes=None, longitudes=None, values=None, messages=1):
        self.metadata = metadata or {
            "shortName": "2t", "paramId": 167, "date": 20250831, "time": 1200,
            "step": 24, "validityDate": 20250901, "validityTime": 1200,
            "units": "K", "typeOfLevel": "heightAboveGround", "level": 2,
            "totalLength": len(b"synthetic-grib"),
        }
        self.latitudes = latitudes if latitudes is not None else [34.0, 34.625, 34.0]
        self.longitudes = longitudes if longitudes is not None else [127.0, 126.625, 126.0]
        self.values = values if values is not None else [280.0, 300.0, 320.0]
        self.messages = messages
        self.created = 0
        self.released = []

    def codes_grib_new_from_file(self, stream):
        self.created += 1
        if self.created > self.messages:
            return None
        return {"handle": self.created}

    def codes_release(self, handle):
        self.released.append(handle)

    def codes_get(self, handle, key):
        return self.metadata[key]

    def codes_get_array(self, handle, key):
        return self.latitudes if key == "latitudes" else self.longitudes

    def codes_get_values(self, handle):
        return self.values

    def codes_get_api_version(self):
        return "2.48.0"


def meteo_bytes(*, target_value=20.0, duplicate=False):
    start = dt.datetime(2025, 8, 31, 12, tzinfo=dt.timezone.utc)
    times = [(start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(240)]
    values = [18.0] * 240
    values[24] = target_value
    values[30] = None  # Unrelated source null remains represented in the source hash/count.
    if duplicate:
        times.append(compare.VALID_TIME)
        values.append(22.0)
    payload = {
        "latitude": 34.622143, "longitude": 126.59091, "elevation": 36,
        "timezone": "GMT", "utc_offset_seconds": 0,
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {"time": times, "temperature_2m": values},
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def invoke(grib=b"synthetic-grib", meteo=None, *, decoder=None):
    meteo = meteo if meteo is not None else meteo_bytes()
    if isinstance(decoder, FakeEccodes):
        decoder.metadata["totalLength"] = len(grib)
    return compare.compare_bytes(
        grib, meteo, eccodes=decoder or FakeEccodes(),
        expected_grib_sha256=hashlib.sha256(grib).hexdigest(),
        expected_meteo_sha256=hashlib.sha256(meteo).hexdigest())


def test_dry_run_does_not_read_inputs_or_load_eccodes(capsys):
    with mock.patch.object(compare, "compare_files", side_effect=AssertionError("inputs read")), \
            mock.patch.object(compare, "_load_eccodes", side_effect=AssertionError("ecCodes loaded")):
        assert compare.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["inputs_read"] is False
    assert result["valid_time_utc"] == "2025-09-01T12:00"


def test_one_point_comparison_validates_metadata_and_reports_signed_difference():
    decoder = FakeEccodes()
    result = invoke(decoder=decoder)
    assert result["status"] == "one_point_comparison_complete"
    assert result["sources"]["ecmwf"]["eccodes_version"] == "2.48.0"
    assert result["sources"]["ecmwf"]["metadata"]["paramId"] == 167
    assert result["sources"]["ecmwf"]["grid_point_index"] == 1
    assert result["sources"]["ecmwf"]["grid_latitude"] == 34.625
    assert result["sources"]["ecmwf"]["temperature_kelvin"] == 300.0
    assert result["sources"]["ecmwf"]["temperature_celsius"] == pytest.approx(26.85)
    assert result["sources"]["open_meteo"]["source_null_counts"] == {
        "time": 0, "temperature_2m": 1}
    assert result["temperature_difference_celsius"] == pytest.approx(6.85)
    assert result["protocol"]["difference_sign"] == "ECMWF minus Open-Meteo"
    assert "equivalence" in result["claim_scope"]
    assert decoder.released == [{"handle": 1}]


@pytest.mark.parametrize("key,value", [
    ("shortName", "t"), ("date", 20250901), ("time", 0), ("step", 12),
    ("validityDate", 20250902), ("validityTime", 0), ("units", "°C"),
    ("paramId", 130),
])
def test_conflicting_grib_metadata_rejected(key, value):
    metadata = FakeEccodes().metadata.copy()
    metadata[key] = value
    with pytest.raises(compare.ComparisonError):
        invoke(decoder=FakeEccodes(metadata=metadata))


def test_multiple_grib_messages_rejected():
    with pytest.raises(compare.ComparisonError, match="exactly one"):
        invoke(decoder=FakeEccodes(messages=2))


def test_trailing_bytes_rejected_even_if_first_message_parses():
    decoder = FakeEccodes()
    with pytest.raises(compare.ComparisonError, match="declared message length"):
        compare._one_grib_handle(b"synthetic-gribtrailing", decoder)


def test_real_eccodes_file_backed_single_message_smoke():
    eccodes = compare._load_eccodes()
    with tempfile.TemporaryFile(mode="w+b") as stream:
        message = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        eccodes.codes_write(message, stream)
        eccodes.codes_release(message)
        stream.flush()
        stream.seek(0)
        synthetic_grib = stream.read()
    handle = compare._one_grib_handle(synthetic_grib, eccodes)
    try:
        assert eccodes.codes_get(handle, "edition") == 2
    finally:
        eccodes.codes_release(handle)
    with pytest.raises(compare.ComparisonError, match="exactly one"):
        compare._one_grib_handle(synthetic_grib + synthetic_grib, eccodes)


def test_bad_nearest_grid_values_do_not_fall_back_or_coerce():
    decoder = FakeEccodes(values=[280.0, None, 320.0])
    with pytest.raises(compare.ComparisonError, match="nearest GRIB gridpoint"):
        invoke(decoder=decoder)


def test_source_hash_mismatch_fails_before_decoding():
    decoder = mock.Mock()
    with pytest.raises(compare.ComparisonError, match="GRIB SHA-256"):
        compare.compare_bytes(b"not-pinned", meteo_bytes(), eccodes=decoder)
    decoder.codes_grib_new_from_file.assert_not_called()


def test_explicit_execution_writes_immutable_report_from_mocked_comparison(tmp_path, capsys):
    target = tmp_path / "report.json"
    fake_report = {"status": "one_point_comparison_complete", "temperature_difference_celsius": 1.25}
    with mock.patch.object(compare, "compare_files", return_value=fake_report):
        assert compare.main(["--execute", "--output", str(target)]) == 0
    written = target.read_bytes()
    assert json.loads(written) == fake_report
    message = json.loads(capsys.readouterr().out)
    assert message["report"] == str(target)
    assert message["sha256"] == hashlib.sha256(written).hexdigest()
    with pytest.raises(FileExistsError):
        compare._write_immutable(target, b"replacement")
    assert Path(target).read_bytes() == written


@pytest.mark.parametrize("change,match", [
    (lambda p: (p["hourly"]["time"].append(compare.VALID_TIME),
                p["hourly"]["temperature_2m"].append(22.0)), "exactly one"),
    (lambda p: p["hourly"]["temperature_2m"].__setitem__(24, None), "null or non-finite"),
    (lambda p: p.update(timezone="Asia/Seoul", utc_offset_seconds=32400), "timezone"),
    (lambda p: p["hourly_units"].update(temperature_2m="K"), "unit is not Celsius"),
])
def test_open_meteo_timestamp_value_timezone_and_units_are_strict(change, match):
    payload = json.loads(meteo_bytes())
    change(payload)
    altered = json.dumps(payload).encode()
    with pytest.raises(compare.ComparisonError, match=match):
        invoke(meteo=altered)
