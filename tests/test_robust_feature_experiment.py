import datetime as dt
import hashlib
import json

import pytest

from agri_data import robust_feature_experiment as exp
from agri_data import robust_price_features as robust


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _screen_inputs(monkeypatch):
    start = dt.date(2024, 9, 25)
    end = start
    monkeypatch.setattr(exp, "SCREEN_START", start)
    monkeypatch.setattr(exp, "SCREEN_END", end)
    monkeypatch.setattr(exp, "SCREEN_ORIGIN_START", start - dt.timedelta(days=10))
    monkeypatch.setattr(exp, "TRAIN_START", start - dt.timedelta(days=12))
    monkeypatch.setattr(exp, "MIN_TRAIN", 2)

    prices = {"schemaVersion": 1, "products": []}
    price_by_key = {}
    first_day = start - dt.timedelta(days=230)
    for crop_index, crop in enumerate(exp.PRODUCTS):
        rows = []
        day = first_day
        index = 0
        while day <= end:
            # Positive, smooth but nonconstant price and volume history.
            value = 100 + crop_index * 30 + 0.035 * index + 1.7 * ((index % 11) - 5) / 5
            rows.append({"date": day.isoformat(), "actual": f"{value:.8f}",
                         "volumeKg": str(500 + (index % 17) * 13), "rowCount": str(8 + index % 6),
                         "marketCodes": ["110001", "110008"]})
            price_by_key[(crop, day)] = value
            day += dt.timedelta(days=1)
            index += 1
        prices["products"].append({"id": crop, "observations": rows})

    screen_components = {"phase": "historical_components_only", "targetValuesAttached": False,
                        "forecasts": []}
    for crop in exp.PRODUCTS:
        for offset in range((end - start).days + 1):
            target = start + dt.timedelta(days=offset)
            actual = price_by_key[(crop, target)]
            for horizon in exp.HORIZONS:
                # The intentionally weak fixed50 benchmark makes the synthetic
                # scoring population useful for checking the selection gates.
                baseline = actual * 3.0
                screen_components["forecasts"].append({"product": crop, "target": target.isoformat(),
                    "horizon": horizon, "direct": baseline, "sarima": baseline})
    prices_raw = _json_bytes(prices)
    components_raw = _json_bytes(screen_components)
    monkeypatch.setattr(exp, "PRICES_SHA256", hashlib.sha256(prices_raw).hexdigest())
    monkeypatch.setattr(exp, "SCREEN_COMPONENTS_SHA256", hashlib.sha256(components_raw).hexdigest())
    return start, end, prices, prices_raw, screen_components, components_raw


def test_screen_build_write_replay_score_and_newline_bound_decision(monkeypatch, tmp_path):
    _, _, prices, prices_raw, components, components_raw = _screen_inputs(monkeypatch)
    manifest = exp.build(prices_raw, components_raw, "screen")
    path = tmp_path / "screen.json"
    exp._write_exclusive(path, manifest)
    manifest_raw = path.read_bytes()
    parsed = json.loads(manifest_raw)
    assert manifest_raw.endswith(b"\n")
    assert exp.validate_manifest_bytes(manifest_raw, prices_raw, components_raw, "screen") == parsed
    assert parsed["protocol"]["featureCountByModel"] == {"ridge106": 106, "ridge115": 115}
    assert len(robust.ORIGINAL_106_NAMES) == 106
    assert tuple(robust.FEATURE_NAMES[:106]) == tuple(robust.ORIGINAL_106_NAMES)
    assert len(robust.FEATURE_NAMES) == 115
    assert parsed["targetValuesAttached"] is False
    assert all("actual" not in row and "targetValue" not in row for row in parsed["forecasts"])
    assert all(audit["maxTrainingTarget"] < audit["fitOrigin"]
               for audit in parsed["fitAudit"] if audit["status"] == "fitted")
    assert len(parsed["fitAudit"]) == len(exp.PRODUCTS) * len(exp.HORIZONS) * len(exp.MODELS)
    assert all(audit["status"] == "fitted" and audit["distinctTargets"] >= 2
               for audit in parsed["fitAudit"])
    assert all(row["featureAudit"].get("featureCutoff") ==
               (dt.date.fromisoformat(row["origin_date"]) - dt.timedelta(days=1)).isoformat()
               for row in parsed["forecasts"] if row["eligible"])
    assert all(row["featureAudit"].get("featureCount") == 115
               for row in parsed["forecasts"] if row["eligible"])
    assert all(row[model]["status"] == "issued" and
               isinstance(row[model]["prediction"], (int, float)) and
               row[model]["prediction"] > 0
               for row in parsed["forecasts"] if row["eligible"]
               for model in exp.MODELS)

    report = exp.score_screen(parsed, prices, components, manifest_bytes=manifest_raw,
                              prices_bytes=prices_raw, components_bytes=components_raw)
    assert len(report["dm"]) == 600
    assert {band: sum(row["bandwidth"] == band for row in report["dm"])
            for band in exp.BANDWIDTHS} == {9: 300, 28: 300}
    for crop in exp.PRODUCTS:
        for horizon in exp.HORIZONS:
            cells = [row for row in report["metrics"] if row.get("crop") == crop
                     and row.get("horizon") == horizon]
            assert len({row["n"] for row in cells}) == 1
            assert cells[0]["n"] > 0

    report_raw = exp._pretty(report)
    decision = exp._decision(manifest_raw, report_raw, report["selection"])
    decision_raw = exp._pretty(decision)
    unsigned = {key: value for key, value in decision.items() if key != "decisionSha256"}
    expected_digest = hashlib.sha256((json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
    assert decision["decisionSha256"] == expected_digest
    screen_components_path = tmp_path / "screen-components.json"
    screen_components_path.write_bytes(components_raw)
    monkeypatch.setattr(exp, "SCREEN_COMPONENTS_PATH", screen_components_path)
    binding = exp.validate_screen_artifacts(manifest_raw, prices_raw, components_raw,
                                            report_raw, decision_raw)
    assert binding["selection"] == report["selection"]
    assert binding["decisionSha256"] == decision["decisionSha256"]
    with pytest.raises(ValueError, match="complete replay"):
        exp.validate_screen_artifacts(manifest_raw, prices_raw, components_raw,
                                      report_raw + b" ", decision_raw)
    tampered = dict(parsed)
    tampered["forecasts"] = [dict(row) for row in parsed["forecasts"]]
    tampered["forecasts"][0]["featureCount"] = 106
    with pytest.raises(ValueError, match="full deterministic replay"):
        exp.validate_manifest_bytes(exp._pretty(tampered), prices_raw, components_raw, "screen")


def test_eval_replay_rejects_tampered_screen_before_any_fit(monkeypatch, tmp_path):
    _, _, _, prices_raw, components, components_raw = _screen_inputs(monkeypatch)
    manifest = exp.build(prices_raw, components_raw, "screen")
    manifest_raw = exp._pretty(manifest)
    report = exp.score_screen(manifest, json.loads(prices_raw), components,
        manifest_bytes=manifest_raw, prices_bytes=prices_raw, components_bytes=components_raw)
    report_raw = exp._pretty(report)
    decision_raw = exp._pretty(exp._decision(manifest_raw, report_raw, report["selection"]))
    screen_components_path = tmp_path / "screen-components.json"
    screen_components_path.write_bytes(components_raw)
    monkeypatch.setattr(exp, "SCREEN_COMPONENTS_PATH", screen_components_path)
    phases = []
    original_payload = exp._source_payload
    def track_phase(raw, phase):
        phases.append(phase)
        assert phase == "screen", "evaluation reached feature/training preparation before replay rejection"
        return original_payload(raw, phase)
    monkeypatch.setattr(exp, "_source_payload", track_phase)
    with pytest.raises(ValueError, match="complete replay"):
        exp.build(prices_raw, b"{}", "evaluation", decision_raw=decision_raw,
                  screen_manifest_raw=manifest_raw, screen_report_raw=report_raw + b" ")
    assert phases and set(phases) == {"screen"}
    with pytest.raises(ValueError, match="no screen-selected model"):
        exp.build(prices_raw, b"{}", "evaluation", decision_raw=decision_raw,
                  screen_manifest_raw=manifest_raw, screen_report_raw=report_raw)
    assert set(phases) == {"screen"}


def test_full_constant_score_fixture_uses_one_three_way_population_and_300_test_families():
    start = dt.date(2025, 3, 1)
    rows, actuals = [], {}
    for crop_index, crop in enumerate(exp.PRODUCTS):
        for horizon in exp.HORIZONS:
            for offset in range(30):
                day = start + dt.timedelta(days=offset)
                actual = 100.0 + crop_index * 40 + offset * 1.3 + (offset % 4) * .6
                actuals[(crop, day)] = actual
                baseline = actual * (1.45 + .03 * (offset % 3))
                p106 = actual * (1.08 + .01 * (offset % 5))
                p115 = actual * (1.15 + .015 * ((offset + 2) % 4))
                rows.append({"crop": crop, "target_date": day.isoformat(), "origin_date": (day-dt.timedelta(days=horizon)).isoformat(),
                    "horizon": horizon, "eligible": True,
                    "fixed50": {"status": "issued", "prediction": baseline},
                    "blend106": {"status": "issued", "prediction": p106},
                    "blend115": {"status": "issued", "prediction": p115}})
    scored = exp._score_common(rows, actuals, start, start + dt.timedelta(days=29), require_threeway=True)
    for bandwidth in exp.BANDWIDTHS:
        family = [row for row in scored["dm"] if row["bandwidth"] == bandwidth]
        assert len(family) == len(exp.PRODUCTS) * len(exp.HORIZONS) * len(exp.COMPARISONS) * len(exp.LOSSES) == 300
        assert all(row["n"] == 30 for row in family)
    for crop in exp.PRODUCTS:
        for horizon in exp.HORIZONS:
            cell = [row for row in scored["metrics"] if row.get("crop") == crop and row.get("horizon") == horizon]
            assert len(cell) == len(exp.COMPARISONS)
            assert {row["n"] for row in cell} == {30}
    assert len(scored["coverageByCell"]) == len(exp.PRODUCTS) * len(exp.HORIZONS)
    gates = exp._gate_screen(rows, scored)
    assert len(gates) == len(exp.PRODUCTS) * 2


def test_three_way_screen_comparisons_exclude_any_missing_arm_from_common_population():
    start = dt.date(2025, 3, 1)
    rows, actuals = [], {}
    crop, horizon = exp.PRODUCTS[0], exp.HORIZONS[0]
    for offset in range(30):
        day = start + dt.timedelta(days=offset)
        actual = 200.0 + offset * 2.1
        actuals[(crop, day)] = actual
        row = {"crop": crop, "target_date": day.isoformat(), "origin_date": (day-dt.timedelta(days=horizon)).isoformat(),
               "horizon": horizon, "eligible": True,
               "fixed50": {"status": "issued", "prediction": actual * 1.5},
               "blend106": {"status": "issued", "prediction": actual * 1.1},
               "blend115": {"status": "issued", "prediction": actual * 1.2}}
        if offset == 0:
            row["fixed50"] = {"status": "not_issued", "prediction": None}
        elif offset == 1:
            row["blend115"] = {"status": "not_issued", "prediction": None}
        rows.append(row)
    scored = exp._score_common(rows, actuals, start, start + dt.timedelta(days=29), require_threeway=True)
    common_metrics = [row for row in scored["metrics"] if row.get("crop") == crop and row.get("horizon") == horizon]
    assert len(common_metrics) == 3
    assert {row["n"] for row in common_metrics} == {28}
    for bandwidth in exp.BANDWIDTHS:
        family = [row for row in scored["dm"] if row["bandwidth"] == bandwidth]
        assert len(family) == 300
        assert {row["n"] for row in family if row["crop"] == crop and row["horizon"] == horizon} == {28}


def test_evaluation_scoring_mixed_screen_selection_keeps_issued_candidate_after_gate_failure(monkeypatch):
    eval_day = dt.date(2026, 1, 5)
    monkeypatch.setattr(exp, "EVAL_START", eval_day)
    monkeypatch.setattr(exp, "EVAL_END", eval_day + dt.timedelta(days=1))
    prices = {"schemaVersion": 1, "products": []}
    for crop_index, crop in enumerate(exp.PRODUCTS):
        prices["products"].append({"id": crop, "observations": [
            {"date": (eval_day + dt.timedelta(days=offset)).isoformat(), "actual": str(100 + crop_index * 10 + offset),
             "volumeKg": "100", "rowCount": "10", "marketCodes": ["110001", "110008"]}
            for offset in range(2)]})
    prices_raw = _json_bytes(prices)
    selections = {crop: model for crop, model in zip(exp.PRODUCTS, ("ridge106", "ridge115", "retain_fixed50", "ridge106", "ridge115"))}
    rows = []
    for crop_index, crop in enumerate(exp.PRODUCTS):
        for offset in range(2):
            day = eval_day + dt.timedelta(days=offset)
            actual = float(100 + crop_index * 10 + offset)
            for horizon in exp.HORIZONS:
                rows.append({"crop": crop, "target_date": day.isoformat(),
                    "origin_date": (day - dt.timedelta(days=horizon)).isoformat(), "horizon": horizon,
                    "eligible": True, "fixed50KeyPresent": True,
                    "fixed50": {"status": "issued", "prediction": actual * 1.5},
                    "blend106": {"status": "issued", "prediction": actual * 1.01},
                    "blend115": {"status": "issued", "prediction": actual * 1.02}})
    manifest = {"forecasts": rows}
    binding = {"selection": selections, "manifestSha256": "screen-manifest",
               "screenReportSha256": "screen-report", "decisionSha256": "decision"}
    monkeypatch.setattr(exp, "validate_screen_artifacts", lambda *a, **k: binding)
    monkeypatch.setattr(exp, "validate_manifest_bytes", lambda *a, **k: manifest)
    report = exp.score_evaluation(manifest, prices, {}, manifest_bytes=b"eval-manifest",
        prices_bytes=prices_raw, components_bytes=b"eval-components", screen_manifest_bytes=b"screen-manifest",
        screen_report_bytes=b"screen-report", screen_decision_bytes=b"screen-decision",
        screen_components_bytes=b"screen-components")
    gates = {row["crop"]: row for row in report["evaluationGates"]}
    assert set(gates) == set(exp.PRODUCTS)
    assert all(gates[crop]["decision"] == "selected_candidate_fails_2026_gate_policy_retained"
               for crop in exp.PRODUCTS if selections[crop] in exp.MODELS)
    for row in report["policyRows"]:
        selected = selections[row["crop"]]
        assert row["screenSelection"] == selected
        if selected == "retain_fixed50":
            assert row["fallback"] is True and row["policy"] == row["fixed50"]
        else:
            assert row["fallback"] is False
            assert row["candidate"] == row["policy"]
            assert row["evaluationGateDecision"] == "selected_candidate_fails_2026_gate_policy_retained"


def test_policy_projection_preserves_8968_keys_and_candidate_fallback_semantics():
    selections = {crop: "ridge115" for crop in exp.PRODUCTS}
    rows, actuals = [], {}
    base_day = dt.date(2026, 1, 1)
    for index in range(8968):
        crop = exp.PRODUCTS[index % len(exp.PRODUCTS)]
        horizon = exp.HORIZONS[(index // len(exp.PRODUCTS)) % len(exp.HORIZONS)]
        day = base_day + dt.timedelta(days=(index // (len(exp.PRODUCTS) * len(exp.HORIZONS))) % 270)
        actual = 100.0 + index % 23
        eligible = index % 5 != 0
        missing_candidate = index % 4 == 0 or not eligible
        row = {"crop": crop, "target_date": day.isoformat(), "origin_date": (day - dt.timedelta(days=horizon)).isoformat(),
            "horizon": horizon, "eligible": eligible, "fixed50KeyPresent": True,
            "fixed50": {"status": "issued", "prediction": actual * 1.25},
            "blend115": {"status": "not_issued", "reason": "synthetic_gap", "prediction": None}
                if missing_candidate else {"status": "issued", "prediction": actual * .95}}
        rows.append(row)
        actuals[(crop, day)] = actual
    projected = exp._policy_rows(rows, actuals, selections)
    assert len(projected) == 8968
    for source_row, policy_row in zip(rows, projected):
        assert policy_row["screenSelection"] == "ridge115"
        assert policy_row["fixed50"] == source_row["fixed50"]["prediction"]
        if source_row["blend115"]["status"] == "issued":
            assert policy_row["fallback"] is False
            assert policy_row["candidate"] == policy_row["policy"] == source_row["blend115"]["prediction"]
        else:
            assert policy_row["fallback"] is True
            assert policy_row["candidate"] is None
            assert policy_row["policy"] == policy_row["fixed50"]
        if not source_row["eligible"]:
            assert policy_row["preperiodOriginFallback"] is True
            assert policy_row["fallback"] is True
    selected_issued = next(row for row in projected if not row["preperiodOriginFallback"] and not row["fallback"])
    # score_evaluation annotates the gate result separately; it must not rewrite
    # the screen-selected issued prediction when the 2026 test is inconclusive.
    selected_issued["evaluationGateDecision"] = "selected_candidate_fails_2026_gate_policy_retained"
    assert selected_issued["screenSelection"] == "ridge115"
    assert selected_issued["candidate"] == selected_issued["policy"]
    assert selected_issued["fallback"] is False
