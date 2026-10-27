import datetime as dt

import pytest

from agri_data import preserved_improvement_composition as comp


def _model(prediction):
    return {"status": "issued", "prediction": float(prediction), "reason": None}


def _source_row(product, day, horizon, *, preserved_garlic=88.0):
    e_model = _model(100.0 if product != "garlic" else 110.0)
    controls = {"causalDirect": _model(101.0), "causalSarima": _model(102.0), "causal50": _model(110.0)}
    e_models = {**controls, "online71Blend": e_model, "causal50": controls["causal50"]}
    p_models = {**controls, "preserved106Blend": _model(preserved_garlic)}
    origin = day - dt.timedelta(days=horizon)
    return ({"product": product, "target": day.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
             "models": e_models},
            {"product": product, "target": day.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
             "models": p_models})


def test_composition_copies_e_except_raw_stored_garlic_and_checks_controls():
    day = dt.date(2026, 4, 11)
    e_index, p_index = {}, {}
    for product in comp.PRODUCTS:
        for horizon in comp.HORIZONS:
            e, p = _source_row(product, day, horizon)
            key = (product, day, horizon)
            e_index[key] = e
            p_index[key] = p
    selection = {"groups": [{"product": "cabbage", "regime": "aligned", "scale": 1.25, "supported": True}]}
    trend_index = {}
    for product in comp.PRODUCTS:
        for horizon in comp.HORIZONS:
            key = (product, day, horizon)
            e_row, _ = _source_row(product, day, horizon)
            base = e_row["models"]["causal50"]["prediction"]
            p0 = 80.0
            scale = 1.25 if product == "cabbage" and horizon >= 4 else 1.0
            saved = comp.trend._scale_movement(base, p0, scale)
            trend_index[key] = {"product": product, "target": day.isoformat(),
                "origin": (day-dt.timedelta(days=horizon)).isoformat(), "horizon": horizon,
                "baselinePrediction": base, "scale": scale,
                "correctionApplied": product == "cabbage" and horizon >= 4,
                "features": {"p0": p0, "regime": "aligned"},
                "model": _model(saved)}
    rows = comp._compose_rows(e_index, p_index, trend_index, selection, day, day, dt.date(2026, 4, 1))
    by_key = {(r["product"], r["horizon"]): r for r in rows}
    assert by_key[("garlic", 1)]["candidate"]["prediction"] == 88.0
    assert by_key[("garlic", 1)]["E"]["prediction"] == 110.0
    assert by_key[("cabbage", 4)]["candidate"] == by_key[("cabbage", 4)]["E"]
    assert by_key[("cabbage", 4)]["E"]["prediction"] != by_key[("cabbage", 4)]["causal50"]["prediction"]
    assert by_key[("radish", 4)]["E"] == by_key[("radish", 4)]["causal50"]
    assert by_key[("onion", 4)]["E"] == by_key[("onion", 4)]["causal50"]
    assert by_key[("radish", 4)]["candidate"]["prediction"] != 100.0  # online71Blend is never E
    assert by_key[("garlic", 1)]["causal50"]["prediction"] == 110.0
    assert by_key[("garlic", 1)]["candidateSource"] == "preserved106Blend"

    changed = dict(p_index)
    bad = dict(changed[("garlic", day, 1)])
    bad["models"] = dict(bad["models"])
    bad["models"]["causalDirect"] = _model(999.0)
    changed[("garlic", day, 1)] = bad
    with pytest.raises(ValueError, match="causalDirect differs"):
        comp._compose_rows(e_index, changed, trend_index, selection, day, day, dt.date(2026, 4, 1))
    wrong_e = dict(trend_index)
    wrong_row = dict(wrong_e[("onion", day, 4)])
    wrong_row["model"] = _model(999.0)
    wrong_e[("onion", day, 4)] = wrong_row
    with pytest.raises(ValueError, match="saved E causal50 differs"):
        comp._compose_rows(e_index, p_index, wrong_e, selection, day, day, dt.date(2026, 4, 1))


def test_trend_phase_manifest_rejects_wrong_phase_even_with_valid_digest():
    selection_sha = "selection"
    doc = {"phase": "correction_forecast_only", "targetValuesAttached": False,
           "baselineManifestSha256": comp.EXPECTED["eManifest"], "selectionSha256": selection_sha,
           "inputSha256": {"baselineManifest": comp.EXPECTED["eManifest"], "prices": comp.EXPECTED["prices"]},
           "implementationSha256": {"trendResponseCorrection": comp.EXPECTED["trendCorrectionBuilder"]}}
    doc["manifestSha256"] = comp.sha(comp.canonical(doc))
    comp._validate_trend_phase_manifest(doc, "correction_forecast_only", selection_sha)
    doc["phase"] = "confirmation_forecast_only"
    unsigned = dict(doc)
    unsigned.pop("manifestSha256")
    doc["manifestSha256"] = comp.sha(comp.canonical(unsigned))
    with pytest.raises(ValueError, match="phase or baseline"):
        comp._validate_trend_phase_manifest(doc, "correction_forecast_only", selection_sha)


def _scoring_rows(n=30):
    start = dt.date(2026, 4, 11)
    rows, actuals = [], {}
    for i in range(n):
        for product in comp.PRODUCTS:
            for horizon in comp.HORIZONS:
                day = start + dt.timedelta(days=i)
                actual = 100.0 + i
                actuals[(product, day)] = actual
                baseline = 110.0 + i if product == "garlic" else 105.0 + i
                candidate = 100.0 + i if product == "garlic" else baseline
                rows.append({"product": product, "target": day.isoformat(),
                    "origin": (day - dt.timedelta(days=horizon)).isoformat(), "horizon": horizon,
                    "eligible": True, "candidate": _model(candidate), "E": _model(baseline),
                    "causal50": _model(110.0 + i), "candidateSource": "preserved106Blend" if product == "garlic" else "E",
                    "eSource": "causal50" if product == "garlic" else "online71Blend"})
    prices = {"schemaVersion": 1, "products": [
        {"id": product, "observations": [{"date": day.isoformat(), "actual": value}
             for (crop, day), value in actuals.items() if crop == product]}
        for product in comp.price_helper.PRODUCTS
    ]}
    return rows, prices


def test_confirmation_gate_uses_strict_both_metric_rules_and_twenty_dm_family(monkeypatch):
    rows, prices = _scoring_rows()
    monkeypatch.setattr(comp, "verify_phase_manifest", lambda *_args, **_kwargs: True)
    report = comp.score_phase({"manifestSha256": "a" * 64, "forecasts": rows}, {}, {}, prices, "confirmation")
    gate = report["confirmationGate"]
    assert gate["passed"] is True
    assert gate["allTenHorizonsAtLeast30Pairs"] is True
    assert gate["allEIssuedKeysCoveragePreserved"] is True
    assert len([x for x in report["dm"] if x["product"] == "garlic" and x["bandwidth"] == 9]) == 20
    assert all(x["pValue"] == 1.0 for x in report["dm"] if x["product"] != "garlic")
    assert report["dmProtocol"]["familySizePerBandwidth"] == 20
    assert report["fourCrop40CellEqualWeightMacro"]["candidate"]["cellsIncluded"] == 40
    assert report["uniqueEligibleTargetDayCount"] == 30
    ape = next(x for x in report["dm"] if x["product"] == "garlic" and x["horizon"] == 1
               and x["loss"] == "ape" and x["bandwidth"] == 9)
    assert ape["meanLossDifference"] == pytest.approx(sum(-100 * 10 / (100 + i) for i in range(30)) / 30)


def test_confirmation_gate_fails_on_lost_e_issue_coverage_and_no_final_is_scored(monkeypatch):
    rows, prices = _scoring_rows()
    target = next(r for r in rows if r["product"] == "garlic" and r["horizon"] == 2)
    target["candidate"] = {"status": "not_issued", "prediction": None}
    monkeypatch.setattr(comp, "verify_phase_manifest", lambda *_args, **_kwargs: True)
    report = comp.score_phase({"manifestSha256": "b" * 64, "forecasts": rows}, {}, {}, prices, "confirmation")
    assert report["confirmationGate"]["passed"] is False
    assert report["confirmationGate"]["allEIssuedKeysCoveragePreserved"] is False
    assert report["confirmationGate"]["pairedCountsByHorizon"]["2"] == 29


def test_failed_confirmation_writes_forecast_only_manifest_before_scoring_and_never_builds_final(tmp_path, monkeypatch):
    e_doc, p_doc, prices = {}, {}, {}
    source_hashes = {"fixture": "fixed"}
    monkeypatch.setattr(comp, "_read_sources", lambda: (e_doc, p_doc, prices, {}, {}, {}, source_hashes))
    calls = []

    def build(_e, _p, phase, *_args, **_kwargs):
        calls.append(("build", phase))
        return {"phaseName": phase, "targetValuesAttached": False, "forecasts": [], "manifestSha256": phase}

    def score(manifest, *_args, **_kwargs):
        assert (tmp_path / "confirmation-manifest.json").exists()
        assert manifest["targetValuesAttached"] is False
        calls.append(("score", manifest["phaseName"]))
        return {"confirmationGate": {"passed": False}, "reportSha256": "report"}

    monkeypatch.setattr(comp, "build_phase_manifest", build)
    monkeypatch.setattr(comp, "score_phase", score)
    result = comp.run(tmp_path)
    assert result["confirmationGatePassed"] is False
    assert result["finalScored"] is False
    assert calls == [("build", "confirmation"), ("score", "confirmation")]
    assert not (tmp_path / "final-manifest.json").exists()


def test_final_report_preserves_result_and_has_per_horizon_and_month_diagnostics(monkeypatch):
    rows, prices = _scoring_rows()
    monkeypatch.setattr(comp, "verify_phase_manifest", lambda *_args, **_kwargs: True)
    report = comp.score_phase({"manifestSha256": "c" * 64, "forecasts": rows}, {}, {}, prices, "final")
    assert report["retentionAssessment"]["interpretation"].startswith("retrospective_candidate_only")
    assert len(report["monthlyAndDistanceDiagnostics"]) > 0
    assert len([x for x in report["dm"] if x["product"] == "garlic" and x["bandwidth"] == 28]) == 20
