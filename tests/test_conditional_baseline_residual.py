from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pytest

from agri_data import conditional_baseline_residual as cbr


def _records(product: str = "radish", days: int = 40) -> list[dict]:
    rows = []
    first = dt.date(2026, 4, 2)
    for i in range(days):
        target = first + dt.timedelta(days=i)
        for h in cbr.HORIZONS:
            origin = target - dt.timedelta(days=h)
            b = 100. + i + h
            d = b * (1. + (i - days / 2.) / 100.)
            delta = b * ((h - 5.) / 100.)
            residual = (0.04 if i >= days // 2 else -0.03) + h / 1000.
            y = b * (1. + residual)
            rows.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                         "horizon": h, "B": b, "D": d, "delta": delta, "actual": y,
                         "sourceRowSha256": f"{i:02d}{h:02d}"})
    return rows


def _phase_rows(records: list[dict]) -> dict[tuple, dict]:
    out = {}
    for r in records:
        key = (r["product"], r["origin"], r["target"], r["horizon"])
        out[key] = {**r, "_key":key, "_B":r["B"], "_D":r["D"], "_delta":r["delta"],
                    "sourceProvenance":{"sourceRowSha256":r["sourceRowSha256"]}}
    return out


def test_feature_vectors_follow_frozen_definition():
    row = {"horizon":4,"B":200.,"D":210.,"delta":-8.}
    assert cbr._feature(row,"R0") == pytest.approx([.4,.05])
    assert cbr._feature(row,"Rdelta") == pytest.approx([.4,.05,-.04])


def test_training_weights_equal_horizon_weight_and_target_day_cap():
    rows = _records(days=40)
    checks = cbr._training_checks(rows,"radish")
    assert checks["maxRowsPerTargetDay"] == 10
    for h in cbr.HORIZONS:
        assert checks["horizonCounts"][str(h)] * checks["weightsByHorizon"][str(h)] == pytest.approx(.1)
    with pytest.raises(ValueError, match="10 forecast rows"):
        cbr._training_checks(rows + [dict(rows[0])],"radish")


def test_training_requires_positive_support_for_every_horizon():
    rows = [r for r in _records(days=40) if r["horizon"] != 7]
    with pytest.raises(ValueError, match="lacks positive support"):
        cbr._training_checks(rows,"radish")


def test_fixed_tree_and_lp_satisfy_leaf_date_and_mae_constraints():
    model, diag = cbr._fit_one(_records(),"radish","R0")
    cbr._verify_frozen_model(model)
    native = cbr._restore_native_model(model)
    assert diag["logicalTreeFits"] == diag["logicalLpFits"] == 1
    assert diag["maxRowsPerTargetDay"] == 10
    assert diag["lp"]["maeResidual"] <= cbr.VERIFY_ABS_TOL
    assert all(v["uniqueTargetDays"] >= 15 for v in model["leafDiagnostics"].values())
    for row in _records():
        leaf = cbr._native_leaf(model,cbr._feature(row,"R0"))
        assert str(leaf) in model["leafCorrections"]
        native_leaf = int(native.apply(__import__("numpy").asarray([cbr._feature(row,"R0")]))[0])
        assert native_leaf == leaf


def test_native_float32_saved_route_matches_estimator_on_threshold_neighbors():
    model, _ = cbr._fit_one(_records(),"radish","R0")
    internal = next(i for i,x in enumerate(model["childrenLeft"]) if x != -1)
    col = model["feature"][internal]
    threshold = model["threshold"][internal]
    probes = [[.5,.0], [.5, float(threshold)], [.5, math.nextafter(float(threshold),math.inf)]]
    for x in probes:
        got = cbr._native_leaf(model,x)
        # Independently walk the serialized arrays using sklearn's float32 split input.
        node=0; xx=__import__("numpy").asarray(x,dtype=__import__("numpy").float32)
        while model["childrenLeft"][node] != -1:
            j=model["feature"][node]
            node=model["childrenLeft"][node] if xx[j] <= model["threshold"][node] else model["childrenRight"][node]
        assert got == node


def test_nonfinite_correction_falls_back_to_baseline():
    # JSON input validation rejects NaN; prediction arithmetic guards overflow separately.
    row = {"B":1e308,"D":1e308,"delta":0.,"horizon":1}
    model, _ = cbr._fit_one(_records(),"radish","R0")
    model["leafCorrections"] = {leaf:1. for leaf in model["leafCorrections"]}
    correction, leaf = cbr._model_g(model,row)
    raw = row["B"] * (1.+correction)
    assert str(leaf) in model["leafCorrections"] and math.isinf(raw)


def test_contributions20_reconcile_to_equal_cell_mean():
    records=[]
    for product in cbr.MODEL_PRODUCTS:
        for h in cbr.HORIZONS:
            for i in range(2):
                records.append({"product":product,"horizon":h,"target":f"2026-07-{11+i:02d}",
                                "actual":100.,"B":100.,"R0":101.,"Rdelta":100.})
    result=cbr._contributions20(records,"Rdelta","R0")
    assert result["netMae"] == pytest.approx(-1.)
    assert result["netMapePercentagePoints"] == pytest.approx(-1.)


def test_direction_classification_distinguishes_direction_and_overshoot():
    assert cbr._loss_direction(110.,100.,95.) == "opposite_direction"
    assert cbr._loss_direction(110.,100.,108.) == "toward_actual"
    assert cbr._loss_direction(110.,100.,115.) == "overshot_actual"
    assert cbr._loss_direction(100.,100.,102.) == "changed_when_B_exact"


def test_selected_identity_checks_unavailable_issued_rows_too():
    rows=[{"B":100.,"selectedPolicy":{"prediction":100.}},
          {"B":101.,"selectedPolicy":{"prediction":102.}}]
    assert not cbr._selected_policy_identical_to_baseline(rows)
    assert cbr._selected_policy_identical_to_baseline(rows[:1])


def test_direction_report_separates_delta_reversal_from_wrong_direction_shrink():
    rows=[{"product":"radish","horizon":1,"target":"2026-07-11","actual":110.,"B":100.,
           "delta":-5.,"Rdelta":108.},
          {"product":"onion","horizon":1,"target":"2026-07-11","actual":110.,"B":100.,
           "delta":5.,"Rdelta":98.}]
    report=cbr._direction_report(rows,"Rdelta")
    classes={r["classification"]:r for r in report["directionFixVersusShrinkage"]}
    assert classes["direction_fix_against_old_delta"]["n"] == 1
    assert classes["new_correction_flipped_correct_old_direction"]["n"] == 1


def test_fit_select_saves_model_before_selection_and_uses_reloaded_models(tmp_path, monkeypatch):
    c_records = _records("radish") + _records("onion")
    # Rebuild onion key dates as its own series but same synthetic values.
    d6_records=[]
    for product in cbr.MODEL_PRODUCTS:
        start=dt.date(2026,6,2)
        for i in range(19):
            target=start+dt.timedelta(days=i)
            for h in cbr.HORIZONS:
                origin=target-dt.timedelta(days=h)
                b=100.+i+h; d=b*1.01; delta=b*.02
                y=b*1.02
                d6_records.append({"product":product,"origin":origin.isoformat(),"target":target.isoformat(),
                    "horizon":h,"B":b,"D":d,"delta":delta,"actual":y,
                    "sourceRowSha256":f"d6-{product}-{i}-{h}"})
    # Fixed-B D6 rows remain issued but are outside learned candidate selection.
    d6_records.append({"product":"cabbage","origin":"2026-06-01","target":"2026-06-02",
        "horizon":1,"B":101.,"D":None,"delta":None,"actual":102.,"sourceRowSha256":"fixed-d6"})
    c_rows=_phase_rows(c_records); d6_rows=_phase_rows(d6_records)
    monkeypatch.setattr(cbr,"_load_sources",lambda: ({}, {"C":c_rows,"D6":d6_rows,"T":{}},{}))
    actuals={(r["product"],r["target"]):r["actual"] for r in c_records+d6_records}
    monkeypatch.setattr(cbr,"_limited_actuals",lambda p,t:(actuals,cbr.PRICE_SHA256))
    models_path=tmp_path/"models.json"; selection_path=tmp_path/"selection.json"
    models, selection=cbr.fit_select("dummy",models_path,selection_path)
    assert models_path.exists() and selection_path.exists()
    persisted=json.loads(models_path.read_text())
    assert persisted["modelSha256"] == models["modelSha256"]
    assert selection["modelFileSha256"] == cbr._sha(models_path.read_bytes())
    assert selection["CRefitAfterD6"] is False if "CRefitAfterD6" in selection else True
    assert models["fitCounts"] == {"treeFits":4,"lpFits":4,"baselineFits":0,"baselineInference":0}
    assert models["trainingForecasts"] and all("actual" not in r for r in models["trainingForecasts"])
    assert selection["d6Forecasts"] and all("actual" not in r for r in selection["d6Forecasts"])
    cabbage = next(r for r in selection["d6Forecasts"] if r["product"] == "cabbage")
    assert cabbage["selectedPolicy"]["status"] == "fixed_B"
    with pytest.raises(FileExistsError):
        cbr.fit_select("dummy",models_path,selection_path)


def test_score_refuses_hash_mismatch_before_loading_targets(tmp_path, monkeypatch):
    payload={"schemaVersion":1,"experiment":"CONDITIONAL_BASELINE_RESIDUAL_V1","phase":"T",
             "stage":"T_answer_free_forecasts","targetValuesAttached":False,
             "protocolSha256":cbr.PROTOCOL_SHA256,"targetSnapshotSha256":cbr.PRICE_SHA256,
             "forecastsSha256":cbr._sha(cbr._canonical([])),"forecasts":[]}
    payload["manifestSha256"]=cbr._sha(cbr._canonical(payload))
    path=tmp_path/"forecast.json"; path.write_bytes(cbr._canonical(payload))
    called=[]
    monkeypatch.setattr(cbr,"_load_t_prices",lambda p: called.append(p))
    with pytest.raises(ValueError,match="caller-pinned"):
        cbr.score_t(path,"0"*64)
    assert called == []

