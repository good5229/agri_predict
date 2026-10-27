from __future__ import annotations

import datetime as dt
import json

import pytest

from agri_data import final_ensemble_onion_score as score


@pytest.fixture(autouse=True)
def _small_synthetic_bootstrap(monkeypatch):
    monkeypatch.setattr(score, "REPLICATES", 5)


def _phase_rows(products=score.PRODUCTS, *, phase="V26", b=110.0, h_d=115.0, h_f=108.0):
    start, end, first_origin = score.PHASES[phase]
    rows = []
    actuals = {}
    for offset in range((end - start).days + 1):
        target = start + dt.timedelta(days=offset)
        actual_by_product = {product: 100.0 for product in products}
        for product in products:
            for h in score.HORIZONS:
                origin = target - dt.timedelta(days=h)
                if origin < first_origin:
                    continue
                actuals[(product, target)] = actual_by_product[product]
                if product == score.ONION:
                    row = {
                        "product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                        "horizon": h, "B": b, "ND": 2 * h_d - b, "NF": 2 * h_f - b,
                        "HD": h_d, "HF": h_f,
                        "BStatus": "issued", "NDStatus": "issued", "NFStatus": "issued",
                        "HDStatus": "issued", "HFStatus": "issued",
                    }
                else:
                    row = {"product": product, "origin": origin.isoformat(),
                           "target": target.isoformat(), "horizon": h,
                           "B": actual_by_product[product], "BStatus": "issued"}
                rows.append(row)
    return rows, actuals


def test_full_40_cell_policy_keeps_other_crops_identical_and_inference_fixed():
    rows, actuals = _phase_rows()
    result = score.score_phase("V26", rows, actuals)
    hf_b = result["metrics"]["HF_vs_B"]
    assert hf_b["cellCount"] == 40
    assert hf_b["aEqualCellMae"] - hf_b["bEqualCellMae"] == pytest.approx(-0.5)
    assert hf_b["cells"]["radish:h1"]["aMae"] == hf_b["cells"]["radish:h1"]["bMae"]
    assert result["inference"]["familyTestCountPerBandwidth"] == 40
    assert len([row for row in result["inference"]["tests"] if row["bandwidthDays"] == 9]) == 40
    assert result["outputMovement"]["HF"]["issuedDistinctFromB"] > 0
    assert result["representationFloor"]["rowsBelowHalfBaseline"] == 0
    assert result["priceLossContributions"]["HF_vs_B:ae"]["meanDifference"] == pytest.approx(-0.5)


def test_fallback_and_success_equal_to_baseline_remain_distinct_states():
    rows, actuals = _phase_rows(products=(score.ONION,))
    row = next(r for r in rows if r["horizon"] == 1)
    row["NF"] = {"status": "fallback", "prediction": row["B"],
                 "fallbackTo": "B", "reason": "nonfinite_raw_output", "rawStatus": "not_issued"}
    row["HF"] = {"status": "fallback", "prediction": row["B"],
                 "fallbackTo": "B", "reason": "nonfinite_raw_output", "rawStatus": "not_issued"}
    row["ND"] = {"status": "issued", "prediction": row["B"], "rawStatus": "issued"}
    row["HD"] = {"status": "issued", "prediction": row["B"], "rawStatus": "issued"}
    result = score.score_phase("V26", rows, actuals, products=(score.ONION,))
    assert result["outputMovement"]["HF"]["technicalFallbackToB"] == 1
    assert result["outputMovement"]["HD"]["issuedSameAsB"] == 1


def test_rejects_wrong_hybrid_arithmetic_and_missing_key():
    rows, actuals = _phase_rows(products=(score.ONION,))
    rows[0]["HF"] = 109.0
    with pytest.raises(ValueError, match="50:50 blend"):
        score.score_phase("V26", rows, actuals, products=(score.ONION,))
    rows, actuals = _phase_rows(products=(score.ONION,))
    with pytest.raises(ValueError, match="grid mismatch"):
        score.score_phase("V26", rows[:-1], actuals, products=(score.ONION,))


def test_representation_floor_uses_lower_bound_as_absolute_error():
    rows, actuals = _phase_rows(products=(score.ONION,), phase="E25", b=240.0, h_d=220.0, h_f=210.0)
    result = score.score_phase("E25", rows, actuals, products=(score.ONION,))
    assert result["representationFloor"]["rowsBelowHalfBaseline"] == result["positiveBIssuedPairs"]
    assert result["representationFloor"]["macroMaeLowerBound"] == pytest.approx(20.0)


def test_missing_target_actual_is_preserved_as_coverage_gap():
    rows, actuals = _phase_rows(products=(score.ONION,), phase="E25")
    omitted_key = (score.ONION, dt.date.fromisoformat(rows[0]["target"]))
    actuals.pop(omitted_key)
    result = score.score_phase("E25", rows, actuals, products=(score.ONION,))
    missing = result["targetCoverage"]["missingRows"]
    assert len(missing) == sum(r["target"] == omitted_key[1].isoformat() for r in rows)
    assert result["coverage"]["missingOrInvalidActualOnBIssuedRows"] == len(missing)
    assert all(r["reason"] == "missing_or_invalid_actual" for r in missing)


def test_lambda_fallback_rule_uses_only_worst_relative_ratio_and_ties_to_base():
    scored = {
        1: {"metrics": {"HF_vs_B": {"aEqualCellMape": 11.2, "bEqualCellMape": 10.0,
                                     "aEqualCellMae": 11.1, "bEqualCellMae": 10.0}}},
        3: {"metrics": {"HF_vs_B": {"aEqualCellMape": 11.0, "bEqualCellMape": 10.0,
                                     "aEqualCellMae": 11.2, "bEqualCellMae": 10.0}}},
    }
    assert score._choose_lambda(scored) == 1


def test_eligible_lambda_uses_exact_minimum_mape():
    scored = {
        1: {"metrics": {"HF_vs_B": {"aEqualCellMape": 9.000000005, "bEqualCellMape": 10.0,
                                     "aEqualCellMae": 9.0, "bEqualCellMae": 10.0}}},
        3: {"metrics": {"HF_vs_B": {"aEqualCellMape": 9.0, "bEqualCellMape": 10.0,
                                     "aEqualCellMae": 8.0, "bEqualCellMae": 10.0}}},
    }
    assert score._choose_lambda(scored) == 3


def test_lambda3_can_be_required_but_lose_the_frozen_d25_decision():
    l1_rows, actuals = _phase_rows(products=(score.ONION,), phase="D25", b=100.0, h_d=120.0, h_f=120.0)
    l3_rows, _ = _phase_rows(products=(score.ONION,), phase="D25", b=100.0, h_d=130.0, h_f=130.0)
    e_rows, _ = _phase_rows(products=(score.ONION,), phase="E25")
    v_rows, _ = _phase_rows(products=score.PRODUCTS, phase="V26")
    t_rows, _ = _phase_rows(products=score.PRODUCTS, phase="T26")
    result = score.score_documents(
        forecasts_by_phase={"D25": l1_rows, "E25": e_rows, "V26": v_rows, "T26": t_rows},
        targets={**actuals,
                 **_phase_rows(products=(score.ONION,), phase="E25")[1],
                 **_phase_rows(products=score.PRODUCTS, phase="V26")[1],
                 **_phase_rows(products=score.PRODUCTS, phase="T26")[1]},
        d25_by_lambda={1: l1_rows, 3: l3_rows})
    assert result["selection"]["scoresByLambda"]["1"]["metrics"]["HF_vs_B"]["aEqualCellMae"] > 0
    assert result["selectedLambda"] == 1


def test_full_policy_is_rebuilt_from_exact_baseline_copies():
    rows, actuals = _phase_rows()
    baseline = {}
    for row in rows:
        key = score._key(row)
        baseline[key] = {"B": row["B"], "BSourceStatus": "issued"}
    onion_rows = [r for r in rows if r["product"] == score.ONION]
    rebuilt = score._full_policy_rows("V26", onion_rows, baseline)
    result = score.score_phase("V26", rebuilt, actuals)
    assert result["metrics"]["HF_vs_B"]["cellCount"] == 40
    assert result["metrics"]["HF_vs_B"]["onion10Cell"]["cellCount"] == 10


def test_producer_shaped_d25_artifacts_validate_and_recompute_l3_loses(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "ROOT", tmp_path)
    weight = tmp_path / "synthetic-checkpoint.bin"
    weight.write_bytes(b"synthetic checkpoint")
    digest = score.file_sha256(weight)
    target_map = {}
    row_sets = {}
    baseline = {}
    for lam, neural in ((1, 120.0), (3, 140.0)):
        raw_rows, phase_actuals = _phase_rows(products=(score.ONION,), phase="D25",
                                               b=100.0, h_d=neural, h_f=neural)
        target_map.update(phase_actuals)
        rows = []
        for source in raw_rows:
            row = {**source, "lambda": lam,
                   "cutoff": (dt.date.fromisoformat(source["origin"]) - dt.timedelta(days=1)).isoformat(),
                   "BSourceStatus": "issued", "replayAgreement": True,
                   "weightPathD": weight.name, "weightPathF": weight.name,
                   "weightSha256D": digest, "weightSha256F": digest}
            baseline[score._key(row)] = {"B": 100.0, "BSourceStatus": "issued"}
            rows.append(row)
        doc = {"phase": "D25", "targetValuesAttached": False, "configSha256": "cfg",
               "baselineCacheManifestSha256": "cache", "rows": rows}
        row_sets[lam] = score._verify_phase_doc(doc, "D25", lam, baseline, "cache", "cfg")
    e_rows, e_actuals = _phase_rows(products=(score.ONION,), phase="E25")
    v_rows, v_actuals = _phase_rows(products=score.PRODUCTS, phase="V26")
    t_rows, t_actuals = _phase_rows(products=score.PRODUCTS, phase="T26")
    target_map.update(e_actuals); target_map.update(v_actuals); target_map.update(t_actuals)
    scored = score.score_documents(
        forecasts_by_phase={"D25": row_sets[1], "E25": e_rows, "V26": v_rows, "T26": t_rows},
        targets=target_map, d25_by_lambda=row_sets)
    assert scored["selectedLambda"] == 1
    assert scored["selection"]["scoresByLambda"]["1"]["metrics"]["HF_vs_B"]["aEqualCellMae"] > 0


def test_corrupt_hybrid_fallback_value_is_rejected_before_normalization():
    rows, actuals = _phase_rows(products=(score.ONION,))
    row = next(r for r in rows if r["horizon"] == 1)
    row["NF"] = {"status": "technical_fallback", "prediction": None,
                 "reason": "nonfinite_neural_prediction", "rawStatus": "not_issued"}
    row["HF"] = {"status": "technical_fallback", "prediction": row["B"] + 1,
                 "reason": "nonfinite_neural_prediction"}
    with pytest.raises(ValueError, match="technical fallback does not equal B"):
        score.score_phase("V26", rows, actuals, products=(score.ONION,))


def test_score_saved_run_end_to_end_with_synthetic_targetfree_artifacts(tmp_path, monkeypatch):
    import pandas as pd

    products = (score.ONION, "radish")
    monkeypatch.setattr(score, "PRODUCTS", products)
    monkeypatch.setattr(score, "REPLICATES", 3)
    phase_dates = {
        "D25": dt.date(2025, 4, 1), "E25": dt.date(2025, 7, 11),
        "V26": dt.date(2026, 4, 1), "T26": dt.date(2026, 7, 11),
    }
    phases = {name: (start, start + dt.timedelta(days=20), start)
              for name, start in phase_dates.items()}
    monkeypatch.setattr(score, "PHASES", phases)
    root = tmp_path
    out = root / "outputs" / "synthetic-run"
    out.mkdir(parents=True)
    monkeypatch.setattr(score, "ROOT", root)
    monkeypatch.setattr(score, "OUT", out)
    (root / "agri_data").mkdir()
    (root / "agri_data" / "final_ensemble_onion_score.py").write_bytes(
        __import__("pathlib").Path(score.__file__).read_bytes())

    target_days = sorted({start + dt.timedelta(days=i)
                          for start in phase_dates.values() for i in range(21)})
    price_path = out / "prices.json"
    known_products = tuple(score.price_helper.PRODUCTS)
    price_doc = {"schemaVersion": 1, "products": [
        {"id": product, "observations": [{"date": day.isoformat(), "actual": 100.0}
                                           for day in target_days]}
        for product in known_products]}
    price_path.write_bytes(score.canonical(price_doc))
    monkeypatch.setattr(score.price_helper, "PRICES_PATH", price_path)
    protocol = out / "protocol.txt"; protocol.write_text("frozen synthetic protocol")
    prior = out / "prior.json"; prior.write_text("{}")
    source_cache = out / "source-cache.json"; source_cache.write_text("{}")
    weight = out / "synthetic-weight.bin"; weight.write_bytes(b"synthetic weight")
    weight_hash = score.file_sha256(weight)

    phase_rows = {}
    target_map = {}
    baseline = {}
    for phase in ("D25", "E25", "V26", "T26"):
        phase_products = products if phase in ("V26", "T26") else (score.ONION,)
        b, neural = (100.0, 120.0) if phase == "D25" else (100.0, 100.0)
        rows, actuals = _phase_rows(products=phase_products, phase=phase,
                                    b=b, h_d=neural, h_f=neural)
        target_map.update(actuals)
        phase_rows[phase] = [row for row in rows if row["product"] == score.ONION]
        for row in rows:
            baseline.setdefault(score._key(row), {"product": row["product"],
                "origin": row["origin"], "target": row["target"], "horizon": row["horizon"],
                "B": row["B"], "BSourceStatus": "issued"})
    # Add D25 lambda 3: same exact keys, with worse target-free neural values.
    l3_rows, _ = _phase_rows(products=(score.ONION,), phase="D25",
                              b=100.0, h_d=140.0, h_f=140.0)
    phase_rows[("D25", 3)] = l3_rows

    cache_doc = {"phase": "answer_free", "targetValuesAttached": False,
                 "rows": sorted(baseline.values(), key=score._key)}
    cache_path = out / "baseline-cache.json"
    cache_path.write_bytes(score.canonical(cache_doc))
    cache_unsigned = {"phase": "answer_free", "targetValuesAttached": False, "complete": True,
        "cacheFileSha256": score.file_sha256(cache_path),
        "cacheRowsSha256": score.sha256_bytes(score.canonical(cache_doc["rows"]))}
    cache_manifest = {**cache_unsigned,
        "manifestSha256": score.sha256_bytes(score.canonical(cache_unsigned))}
    cache_manifest_path = out / "baseline-cache-manifest.json"
    cache_manifest_path.write_bytes(score.canonical(cache_manifest))
    baseline_manifest_sha = score.file_sha256(cache_manifest_path)

    source_rel = "agri_data/final_ensemble_onion_score.py"
    config = {"experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
        "producerCodeSha256": {source_rel: score.file_sha256(root / source_rel)},
        "protocolPath": str(protocol.relative_to(root)), "protocolSha256": score.file_sha256(protocol),
        "priceSnapshotPath": str(price_path.relative_to(root)),
        "priceSnapshotSha256": score.file_sha256(price_path),
        "priorResolvedConfigPath": str(prior.relative_to(root)),
        "priorResolvedConfigSha256": score.file_sha256(prior),
        "sourceCachePath": str(source_cache.relative_to(root)),
        "sourceCacheFileSha256": score.file_sha256(source_cache),
        "baselineCacheManifestSha256": baseline_manifest_sha}
    config["configSha256"] = score.sha256_bytes(score.canonical(config))
    (out / "resolved-config.json").write_bytes(score.canonical(config))

    def producer_rows(rows, lam):
        out_rows = []
        for row in rows:
            out_rows.append({**row, "phase": "D25", "lambda": lam,
                "statusD": row.get("statusD", row.get("NDStatus", "issued")),
                "statusF": row.get("statusF", row.get("NFStatus", "issued")),
                "statusHD": row.get("statusHD", row.get("HDStatus", "issued")),
                "statusHF": row.get("statusHF", row.get("HFStatus", "issued")),
                "BSourceStatus": "issued",
                "cutoff": (dt.date.fromisoformat(row["origin"]) - dt.timedelta(days=1)).isoformat(),
                "replayAgreement": True, "weightPathD": str(weight.relative_to(root)),
                "weightPathF": str(weight.relative_to(root)),
                "weightSha256D": weight_hash, "weightSha256F": weight_hash})
        return out_rows

    docs = {}
    forecast_entries = []
    d25_score_refs = {}
    d25_rows_issued = {}
    for phase, lam in (("D25", 1), ("D25", 3), ("E25", 1), ("V26", 1), ("T26", 1)):
        source_rows = (l3_rows if phase == "D25" and lam == 3 else phase_rows[phase])
        rows = producer_rows(source_rows, lam) if phase == "D25" else [
            {**row, "lambda": lam,
             "statusD": row.get("statusD", row.get("NDStatus", "issued")),
             "statusF": row.get("statusF", row.get("NFStatus", "issued")),
             "statusHD": row.get("statusHD", row.get("HDStatus", "issued")),
             "statusHF": row.get("statusHF", row.get("HFStatus", "issued")),
             "BSourceStatus": "issued",
             "cutoff": (dt.date.fromisoformat(row["origin"]) - dt.timedelta(days=1)).isoformat(),
             "replayAgreement": True, "weightPathD": str(weight.relative_to(root)),
             "weightPathF": str(weight.relative_to(root)),
             "weightSha256D": weight_hash, "weightSha256F": weight_hash}
            for row in source_rows]
        if phase == "D25":
            d25_rows_issued[lam] = rows
        doc = {"phase": phase, "targetValuesAttached": False,
               "configSha256": config["configSha256"],
               "baselineCacheManifestSha256": baseline_manifest_sha, "rows": rows}
        docs[(phase, lam)] = doc
        path = out / f"forecast-{phase}-L{lam}.json"
        path.write_bytes(score.canonical(doc))
        keys = [score._key(row) for row in rows]
        forecast_entries.append({"phase": phase, "lambda": lam,
            "path": str(path.relative_to(root)), "fileSha256": score.file_sha256(path),
            "rows": len(rows), "keyDigest": score.sha256_bytes(score.canonical(keys)),
            "targetValuesAttached": False})
        if phase == "D25":
            computed = score.score_phase("D25", rows, target_map, products=(score.ONION,))
            horizons = []
            for h in score.HORIZONS:
                cell = computed["metrics"]["HF_vs_B"]["cells"][f"{score.ONION}:h{h}"]
                hd = computed["metrics"]["HF_vs_HD"]["cells"][f"{score.ONION}:h{h}"]
                horizons.append({"horizon": h, "n": cell["n"],
                    "H_F_MAE": cell["aMae"], "B_MAE": cell["bMae"], "H_D_MAE": hd["aMae"],
                    "H_F_MAPE": cell["aMape"] / 100, "B_MAPE": cell["bMape"] / 100,
                    "H_D_MAPE": hd["aMape"] / 100})
            score_doc = {"configSha256": config["configSha256"], "phase": "D25", "lambda": lam,
                "priceSnapshotSha256": score.file_sha256(price_path),
                "forecastFileSha256": score.file_sha256(path), "pairedRows": computed["positiveBIssuedPairs"],
                "uniqueForecastKeys": computed["positiveBIssuedPairs"], "pairedPriceCoverage": True,
                "horizons": horizons,
                "metrics": {"H_F_MAE": computed["metrics"]["HF_vs_B"]["aEqualCellMae"],
                    "B_MAE": computed["metrics"]["HF_vs_B"]["bEqualCellMae"],
                    "H_F_MAPE": computed["metrics"]["HF_vs_B"]["aEqualCellMape"] / 100,
                    "B_MAPE": computed["metrics"]["HF_vs_B"]["bEqualCellMape"] / 100}}
            score_path = out / f"score-D25-L{lam}.json"
            score_path.write_bytes(score.canonical(score_doc))
            d25_score_refs[f"L{lam}"] = score.file_sha256(score_path)

    selection = {"configSha256": config["configSha256"], "phase": "D25",
        "selectedLambda": 1, "downstreamLambda": 1, "lambda3Required": True,
        "inputScoreSha256": d25_score_refs}
    selection_path = out / "D25-selection.json"
    selection_path.write_bytes(score.canonical(selection))
    for phase in ("E25", "V26", "T26"):
        key = (phase, 1)
        docs[key]["d25SelectionSha256"] = score.file_sha256(selection_path)
        path = root / forecast_entries[[ (e["phase"], int(e["lambda"])) for e in forecast_entries].index(key)]["path"]
        path.write_bytes(score.canonical(docs[key]))
        entry = next(e for e in forecast_entries if (e["phase"], int(e["lambda"])) == key)
        entry["fileSha256"] = score.file_sha256(path)
        entry["keyDigest"] = score.sha256_bytes(score.canonical([score._key(r) for r in docs[key]["rows"]]))
    (out / "forecast-manifest.json").write_bytes(score.canonical({
        "configSha256": config["configSha256"], "forecasts": forecast_entries}))

    parquet_rows = []
    for phase in ("D25", "E25", "V26", "T26"):
        src_rows = docs[(phase, 1)]["rows"]
        parquet_rows.extend([{**row, "phase": phase, "phaseLambda": 1,
                              "finalH": row["HF"], "finalStatus": row["statusF"]}
                             for row in src_rows])
        if phase in ("V26", "T26"):
            target_start, target_end, first_origin = phases[phase]
            for key, base in baseline.items():
                product, origin_s, target_s, horizon = key
                origin, target = score._date(origin_s), score._date(target_s)
                if (product == score.ONION or origin < first_origin or
                        not target_start <= target <= target_end):
                    continue
                b = base["B"]
                parquet_rows.append({"phase": phase, "product": product, "origin": origin_s,
                    "target": target_s, "horizon": horizon, "B": b,
                    "BSourceStatus": "issued", "ND": None, "NF": None, "HD": b, "HF": b,
                    "statusD": "baseline_copy", "statusF": "baseline_copy",
                    "statusHD": "baseline_copy", "statusHF": "baseline_copy",
                    "finalH": b, "finalStatus": "baseline_copy", "phaseLambda": 1})
    parquet_rows.sort(key=lambda r: (r["phase"], r["product"], r["origin"], r["target"], r["horizon"]))
    parquet_path = out / "forecasts.parquet"
    pd.DataFrame(parquet_rows).to_parquet(parquet_path, index=False)
    parquet_keys = sorted((str(r["phase"]), str(r["product"]), str(r["origin"]),
                           str(r["target"]), int(r["horizon"])) for r in parquet_rows)
    parquet_manifest = {"configSha256": config["configSha256"],
        "baselineCacheManifestSha256": baseline_manifest_sha,
        "D25SelectionSha256": score.file_sha256(selection_path),
        "forecastSourceFiles": {phase: {"sha256": next(e["fileSha256"] for e in forecast_entries
            if (e["phase"], int(e["lambda"])) == (phase, 1))}
            for phase in ("D25", "E25", "V26", "T26")},
        "path": str(parquet_path.relative_to(root)), "fileSha256": score.file_sha256(parquet_path),
        "rows": len(parquet_rows), "keyDigest": score.sha256_bytes(score.canonical(parquet_keys)),
        "targetValuesAttached": False}
    (out / "forecasts-parquet-manifest.json").write_bytes(score.canonical(parquet_manifest))
    result = score.score_saved_run(out)
    assert result["selectedLambda"] == 1
    assert result["selection"]["scoresByLambda"]["3"]["metrics"]["HF_vs_B"]["aEqualCellMae"] > 0
    assert score.score_saved_run(out) == result


def test_module_cli_prints_saved_report_summary_without_recalculating(tmp_path, monkeypatch, capsys):
    report = {"status": "completed_small_gain", "selectedLambda": 1,
              "finalGoal": {"T26HfMapeReductionPercentagePoints": 0.2},
              "phases": {"V26": {"metrics": {"HF_vs_B": {}, "HF_vs_HD": {}},
                                  "coverage": {}}}}
    monkeypatch.setattr(score, "OUT", tmp_path)
    monkeypatch.setattr(score, "score_saved_run", lambda out_dir: report)
    score.main()
    captured = json.loads(capsys.readouterr().out)
    assert captured["scoreReportPath"] == str(tmp_path / "score-report.json")
    assert captured["status"] == "completed_small_gain"

