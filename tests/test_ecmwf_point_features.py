import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from agri_data import ecmwf_field_collection as collection
from agri_data import ecmwf_point_features as features


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FakeEccodes:
    def __init__(self, *, missing_station_ids=()):
        self.nearest_calls = 0
        self.missing_station_ids = set(missing_station_ids)

    def codes_grib_new_from_file(self, stream):
        data = stream.read()
        if not data:
            return None
        return {"data": data}

    def codes_release(self, handle):
        pass

    def codes_get(self, handle, key):
        data = handle["data"].decode()
        param, step = data.split(":")
        run_date = 20240315
        valid = dt.datetime(2024, 3, 15, 12, tzinfo=dt.timezone.utc) + dt.timedelta(hours=int(step))
        metadata = {"shortName": param, "paramId": 167 if param == "2t" else 228,
                    "date": run_date, "time": 1200, "step": int(step),
                    "validityDate": int(valid.strftime("%Y%m%d")),
                    "validityTime": int(valid.strftime("%H%M")),
                    "units": "K" if param == "2t" else "m",
                    "typeOfLevel": "heightAboveGround" if param == "2t" else "surface",
                    "level": 2 if param == "2t" else 0,
                    "totalLength": len(handle["data"]), "missingValue": 9999.0}
        return metadata[key]

    def codes_grib_find_nearest_multiple(self, handle, is_lsm, inlats, inlons):
        self.nearest_calls += 1
        param = handle["data"].decode().split(":")[0]
        results = []
        for position, (lat, lon) in enumerate(zip(inlats, inlons)):
            index = 1 if lat > 36 else 0
            value = ([0.005, 0.003] if param == "tp" else [280.0, 290.0])[index]
            if [100, 108, 261][position] in self.missing_station_ids:
                value = 9999.0
            results.append(SimpleNamespace(lat=[34.5, 37.5][index],
                                           lon=[126.5, 128.5][index],
                                           value=value, distance=10.0, index=index))
        return results

    def codes_get_api_version(self):
        return "fake-1"


def write_artifact(root, name, body, metadata):
    raw = root / name
    raw.write_bytes(body)
    metadata = {**metadata, "sha256": sha(body)}
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    meta_path = root / f"{name}.meta.json"
    meta_path.write_bytes(encoded)
    return {"metadata_file": meta_path.name, "metadata_sha256": sha(encoded),
            "raw_file": raw.name, "sha256": sha(body), "complete": True}


def make_checkpoint(root, *, late=False):
    run_date = "20240315"
    captures = {"indices": {}, "fields": {}}
    for step in features.STEPS:
        index_body = f"index:{step}".encode()
        index_attempt = write_artifact(root, f"index-{step}.index", index_body,
                                       {"kind": "index", "complete": True,
                                        "source_scope_hash": collection.SCOPE_HASH,
                                        "selected_records": {param: {"_offset": 0, "_length": len(f"{param}:{step}".encode())}
                                                             for param in features.PARAMS}})
        captures["indices"][f"{run_date}/{step}"] = {"complete": True, "latest": index_attempt}
        for param in features.PARAMS:
            body = f"{param}:{step}".encode()
            field_attempt = write_artifact(root, f"field-{step}-{param}.grib2", body,
                {"kind": "field", "complete": True, "source_scope_hash": collection.SCOPE_HASH,
                 "run_date": run_date, "step_hours": step, "parameter": param,
                 "index_sha256": sha(index_body), "last_modified_raw": "Fri, 15 Mar 2024 20:00:00 GMT",
                 "last_modified_utc": "2024-03-15T20:00:00+00:00" if not late else "2024-03-16T02:00:00+00:00",
                 "requested_offset": 0, "requested_length": len(body),
                 "captured_at_utc": "2026-10-07T00:00:00+00:00"})
            captures["fields"][f"{run_date}/{step}/{param}"] = {"complete": True, "latest": field_attempt}
    state = {"schema_version": 1, "source_scope": collection.SOURCE_SCOPE,
             "source_scope_hash": collection.SCOPE_HASH, "run_date": run_date,
             "captures": captures}
    payload = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode()
    name = f"day_{run_date}_checkpoint_0001_{sha(payload)[:12]}.json"
    (root / name).write_bytes(payload)


def test_dry_run_does_not_load_or_read_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(features, "_load_eccodes", lambda: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(features, "build_features", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert features.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["inputs_read"] is False
    assert result["network_requests"] == 0
    assert result["features_per_row"] == 18


def test_build_features_decodes_each_field_once_for_all_stations(tmp_path, monkeypatch):
    make_checkpoint(tmp_path)
    monkeypatch.setattr(features, "_run_dates", lambda: ["20240315"])
    decoder = FakeEccodes()
    result, manifest = features.build_features(tmp_path, eccodes=decoder)
    row = result["rows"][0]
    assert row["origin_date"] == "2024-03-16"
    assert row["candidate_valid"] is True
    assert row["available_feature_count"] == 18
    assert list(row["features"]) == list(features.FEATURE_ORDER)
    assert row["features"]["asos_100_2t_24h_c"] == pytest.approx(16.85)
    assert row["features"]["asos_100_tp_24h_mm"] == pytest.approx(3.0)
    assert row["feature_states"]["asos_100_tp_24h_mm"]["provenance"]["index_sha256"]
    assert manifest["ecmwf_decoding"].endswith("without deaccumulation")
    assert decoder.nearest_calls == len(features.STEPS) * len(features.PARAMS)


def test_selected_missing_value_sentinel_is_null_without_neighbor_fallback(tmp_path, monkeypatch):
    make_checkpoint(tmp_path)
    monkeypatch.setattr(features, "_run_dates", lambda: ["20240315"])
    result, _ = features.build_features(tmp_path, eccodes=FakeEccodes(missing_station_ids={100}))
    row = result["rows"][0]
    item = row["feature_states"]["asos_100_2t_24h_c"]
    assert item["value"] is None
    assert item["reason"] == "nearest_grid_value_missing_station_100"
    assert row["feature_states"]["asos_108_2t_24h_c"]["state"] == "valid"


def test_late_fields_stay_null_and_do_not_make_candidate_valid(tmp_path, monkeypatch):
    make_checkpoint(tmp_path, late=True)
    monkeypatch.setattr(features, "_run_dates", lambda: ["20240315"])
    result, _ = features.build_features(tmp_path, eccodes=FakeEccodes())
    row = result["rows"][0]
    assert row["candidate_valid"] is False
    assert set(row["features"].values()) == {None}
    item = row["feature_states"]["asos_100_2t_24h_c"]
    assert item["state"] == "late"
    assert item["reason"] == "last_modified_not_before_origin"


def test_corrupt_checkpoint_is_explicit_missing_without_decode(tmp_path, monkeypatch):
    (tmp_path / "day_20240315_checkpoint_0001_badbadbadbad.json").write_text("{}")
    monkeypatch.setattr(features, "_run_dates", lambda: ["20240315"])
    result, _ = features.build_features(tmp_path, eccodes=FakeEccodes())
    item = result["rows"][0]["feature_states"]["asos_100_2t_24h_c"]
    assert item["value"] is None
    assert item["reason"] == "checkpoint_hash_mismatch"


def test_immutable_output_rejects_overwrite(tmp_path):
    target = tmp_path / "out.json"
    features._write_immutable(target, b"first")
    try:
        features._write_immutable(target, b"different")
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable output unexpectedly replaced")
    assert target.read_bytes() == b"first"


def test_cli_writes_content_addressed_outputs_and_manifest_link(tmp_path, monkeypatch, capsys):
    rows = [{"origin_date": "2024-03-16", "features": {"x": 1.0}}]
    manifest = {"schema_version": 1, "row_count": 1}
    monkeypatch.setattr(features, "build_features", lambda *_: ({"rows": rows}, manifest))
    assert features.main(["--execute", "--raw-dir", str(tmp_path / "raw"),
                          "--output-dir", str(tmp_path / "out")]) == 0
    emitted = json.loads(capsys.readouterr().out)
    target = tmp_path / "out"
    feature_bytes = (target / emitted["feature_file"]).read_bytes()
    saved_manifest = json.loads((target / emitted["manifest_file"]).read_bytes())
    assert sha(feature_bytes) == saved_manifest["feature_sha256"]
    assert saved_manifest["feature_file"] == emitted["feature_file"]
    rows.append({"origin_date": "2024-03-17", "features": {"x": 2.0}})
    assert features.main(["--execute", "--raw-dir", str(tmp_path / "raw"),
                          "--output-dir", str(target)]) == 0
    capsys.readouterr()
    assert len(list(target.glob("features_*.json"))) == 2


def test_real_eccodes_sample_uses_native_multi_nearest_without_collected_data():
    eccodes = features._load_eccodes()
    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        for key, value in (("shortName", "2t"), ("date", 20240315), ("time", 1200),
                           ("step", 24), ("typeOfLevel", "heightAboveGround"), ("level", 2)):
            eccodes.codes_set(handle, key, value)
        with tempfile.TemporaryFile(mode="w+b") as stream:
            eccodes.codes_write(handle, stream)
            stream.flush()
            stream.seek(0)
            sample = stream.read()
    finally:
        eccodes.codes_release(handle)
    station = {"id": 999, "latitude": 34.0, "longitude": 10.0}
    decoded = features._decode_field(sample, eccodes, run_date="20240315", step=24,
                                     param="2t", stations=(station,))
    assert set(decoded) == {999}
    point = decoded[999]
    assert point["grib_metadata"]["shortName"] == "2t"
    assert 0 <= point["grid_point_index"] < 16 * 31
    assert point["grid_latitude"] >= 0
