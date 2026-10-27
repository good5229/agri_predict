from __future__ import annotations

import datetime as dt

import pytest

from agri_data import chronos2_cabbage_followup_eval as b


def _forecast_fixture():
    prepared = {"rows": [], "contexts": {}}
    results = {}
    for h in range(1, 11):
        origin = dt.date(2026, 7, 1) + dt.timedelta(days=h)
        prepared["contexts"][origin] = {"matrix": None, "support": {"cutoff": (origin-dt.timedelta(days=1)).isoformat(),
            "observedCounts": {"representative": 500, "110001": 450, "110008": 440}}}
        for cond, value in (("U", 80.0), ("M", 90.0)):
            results[(origin, cond)] = {"status": "complete", "representativePredictions": [value+i for i in range(11)],
                "taskId": f"cabbage-{origin}-{cond}"}
        prepared["rows"].append({"product": "cabbage", "origin": origin, "target": origin+dt.timedelta(days=h),
            "horizon": h, "B": 100.0, "baselineSourceStatus": "fallback",
            "baselineFallbackReason": "source-reason"})
        prepared["rows"].append({"product": "radish", "origin": origin, "target": origin+dt.timedelta(days=h),
            "horizon": h, "B": 200.0, "baselineSourceStatus": "issued", "baselineFallbackReason": None})
    return prepared, results


def test_answer_free_forecast_uses_selected_hm_only_for_cabbage_and_keeps_other_b():
    prepared, tasks = _forecast_fixture()
    protocol = {"protocolSha256": "a"*64, "input": {"inputSha256": "b"*64}}
    doc = b._forecast_doc(prepared, tasks, protocol, b.OUT)
    cabbage = next(r for r in doc["forecasts"] if r["key"]["product"] == "cabbage")
    radish = next(r for r in doc["forecasts"] if r["key"]["product"] == "radish")
    assert cabbage["conditions"]["HU"]["prediction"] == 90.5
    assert cabbage["conditions"]["HM"]["prediction"] == 95.5
    assert cabbage["conditions"]["S"]["prediction"] == 95.5
    assert cabbage["conditions"]["S"]["selectedCandidate"] == "HM"
    assert radish["conditions"]["S"]["prediction"] == 200.0
    assert radish["conditions"]["M"]["status"] == "not_applicable"
    assert doc["targetValuesAttached"] is False


def test_missing_m_never_falls_back_to_b():
    prepared, tasks = _forecast_fixture()
    origin = prepared["rows"][0]["origin"]
    tasks[(origin, "M")]["status"] = "failed"
    tasks[(origin, "M")]["failure"] = {"type": "RuntimeError", "message": "bad task"}
    doc = b._forecast_doc(prepared, tasks, {"protocolSha256": "a", "input": {"inputSha256": "b"}}, b.OUT)
    row = next(r for r in doc["forecasts"] if r["key"]["product"] == "cabbage" and r["key"]["originDate"] == origin.isoformat())
    assert row["conditions"]["HM"]["status"] == "not_issued"
    assert row["conditions"]["S"]["status"] == "not_issued"
    assert row["conditions"]["S"]["prediction"] is None


def test_stage_b_actual_filter_only_converts_test_window_rows():
    poison = {"date": "2026-09-28", "actual": "POISON", "volumeKg": "bad"}
    products = []
    from agri_data.dashboard import PRODUCTS as ALL_PRODUCTS
    for product in ALL_PRODUCTS:
        rows = ([{"date": "2026-07-11", "actual": "100", "volumeKg": "2"}, poison]
                if product == "cabbage" else [dict(poison)])
        products.append({"id": product, "observations": rows})
    actuals = b._stage_b_actuals({"schemaVersion": 1, "products": products})
    assert actuals == {("cabbage", dt.date(2026, 7, 11)): 100.0}


def test_score_report_has_fixed_40_dm_tests_and_gate_uses_equal_40_cells(monkeypatch):
    forecasts, actuals = [], {}
    for i, product in enumerate(b.PRODUCTS):
        for h in b.HORIZONS:
            target = dt.date(2026, 7, 11) + dt.timedelta(days=h-1)
            actual = 100.0
            actuals[(product, target)] = actual
            base = 110.0+i
            is_cabbage = product == "cabbage"
            s = base-2 if is_cabbage else base
            hu, hm = (base-4, base-2) if is_cabbage else (None, None)
            conditions = {k: {"status": "issued", "prediction": v} for k,v in
                          (("B",base),("S",s),("HU",hu),("HM",hm))}
            forecasts.append({"key": {"product":product,"originDate":(target-dt.timedelta(days=h)).isoformat(),
                "targetDate":target.isoformat(),"horizon":h}, "conditions":conditions})
    def fake_bootstrap(rows, **kwargs):
        return {"blockDays":kwargs["block"],"requested":kwargs["replicates"],"valid":kwargs["replicates"],
                "invalid":0,"interval95":[-1.0,0.0]}
    monkeypatch.setattr("agri_data.research_progress_diagnostic.circular_block_bootstrap", fake_bootstrap)
    report = b.score_report({"forecasts":forecasts}, actuals, {"manifestSha256":"c"*64,"protocolSha256":"d"*64})
    assert report["gate"]["passed"] is True
    assert len(report["primaryCells"]) == 40
    assert report["cabbageEqual10Cell"]["S_vs_B"]["supportedCells"] == 10
    assert report["cabbageEqual10Cell"]["HU_vs_B_diagnostic"]["supportedCells"] == 10
    assert len(report["monthlyEqualCellSummaries"]) == 3
    assert report["monthlyEqualCellSummaries"][0]["primary_S_vs_B"]["supportedCells"] == 40
    assert report["monthlyEqualCellSummaries"][0]["secondary_HM_vs_HU"]["supportedCells"] == 10
    assert "primaryDetail" not in report
    assert len(report["fixedDMFamily"]["tests"]) == 80
    assert all(sum(test["bandwidth"] == bw for test in report["fixedDMFamily"]["tests"]) == 40
               for bw in b.HAC_BANDWIDTHS)
    assert len(report["bootstrap"]) == 2
    assert set(report["bootstrap"][0]["comparisons"]["S_vs_B"]) == {"ae", "ape"}


def test_replay_missing_checkpoint_fails_without_creating_it(tmp_path):
    origin = dt.date(2026, 7, 1)
    prepared = {"contexts": {origin: {"matrix": None}}}
    with pytest.raises(FileNotFoundError):
        b._replay_task_outputs(tmp_path, prepared, {}, {"taskSha256": {}, "newTaskCount": 0}, {})
    assert not (tmp_path / "task-checkpoints").exists()


def test_terminal_recovery_is_idempotent_and_bound_to_claim(tmp_path):
    protocol = {"protocolSha256": "a"*64, "input": {"inputSha256": "b"*64}}
    claim = {"protocolSha256": protocol["protocolSha256"], "inputSha256": protocol["input"]["inputSha256"]}
    claim["claimSha256"] = b._sha(b._canonical(claim))
    (tmp_path / "test-run-claim.json").write_bytes(b._canonical(claim))
    manifest = {"manifestSha256": "c"*64}
    first = b._ensure_terminal(tmp_path, protocol, manifest)
    second = b._ensure_terminal(tmp_path, protocol, manifest)
    assert first == second
    assert b._read(tmp_path / "test-run-terminal.json")["manifestSha256"] == manifest["manifestSha256"]


def test_stage_a_protocol_file_hash_is_distinct_from_internal_protocol_hash(tmp_path, monkeypatch):
    import json
    selection = {"selected": "HM", "status": "selected", "protocolSha256": "internal-protocol-id"}
    protocol = {"protocolSha256": "internal-protocol-id", "selectionCodeSha256": b._file_sha(__import__(
        "pathlib").Path(b.stage_a.__file__))}
    selection_path = tmp_path / "selection.json"
    protocol_path = tmp_path / "protocol.json"
    selection_path.write_bytes(b._canonical(selection))
    protocol_path.write_bytes(b._canonical(protocol))
    monkeypatch.setattr(b.stage_a, "OUT", tmp_path)
    monkeypatch.setattr(b, "SELECTION_SHA256", b._file_sha(selection_path))
    monkeypatch.setattr(b, "SELECTION_PROTOCOL_FILE_SHA256", b._file_sha(protocol_path))
    monkeypatch.setattr(b, "SELECTION_PROTOCOL_SHA256", "internal-protocol-id")
    got_selection, got_protocol = b._verify_selection()
    assert got_selection["selected"] == "HM"
    assert got_protocol["protocolSha256"] == "internal-protocol-id"


def test_fixed_dm_keeps_identical_comparison_status():
    rows = [{"product":"cabbage", "horizon":1, "target":dt.date(2026,7,11),
             "actual":100.0, "B":90.0, "S":90.0}]
    family = b._dm_family(rows, "S_vs_B", "S", "B", ("cabbage",))
    test = next(x for x in family if x["loss"] == "ae" and x["bandwidth"] == 9)
    assert test["status"] == "identical_predictions"
    assert test["pValue"] == 1.0
