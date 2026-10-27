import copy
import datetime as dt

import pytest

from agri_data import e106_mlp16 as exp


def _manifest(phase):
    rows = []
    for product, target, horizon in exp.keys(phase):
        eligible = target - dt.timedelta(days=horizon) >= exp.ORIGIN_START[phase]
        rows.append({"product": product, "target": target.isoformat(),
                     "origin": (target-dt.timedelta(days=horizon)).isoformat(),
                     "horizon": horizon, "eligible": eligible,
                     "status": "issued", "prediction": 100.0, "reason": None,
                     "usedMLP": False, "fallbackReason": None})
    doc = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": phase,
           "targetValuesAttached": False, "plannedCount": len(rows), "forecasts": rows}
    doc["manifestSha256"] = exp.digest(doc)
    return doc


def test_july_phase_uses_july_first_origin_even_with_july_11_target_start():
    assert exp.PHASES["final"][0] == dt.date(2026, 7, 11)
    assert exp.ORIGIN_START["final"] == dt.date(2026, 7, 1)
    audits = exp._r106_index()
    for h in (1, 10):
        key = ("garlic", dt.date(2026, 7, 11), h)
        fit, origin = exp._frozen_fit_audit(audits,h,[key])
        assert origin == dt.date(2026, 7, 1)
        assert fit["fitOrigin"] == "2026-07-01"


@pytest.mark.parametrize("phase", ["confirmation", "final"])
def test_answer_free_grid_validation_and_digest(phase):
    doc = _manifest(phase)
    exp.validate(doc, phase)
    altered = copy.deepcopy(doc)
    altered["forecasts"].pop()
    with pytest.raises(ValueError, match="digest"):
        exp.validate(altered, phase)


def test_model_definition_is_single_fixed_mlp16():
    assert exp.MODEL_PARAMS == {
        "hidden_layer_sizes": [16], "activation": "relu", "solver": "adam",
        "alpha": 10.0, "batch_size": 256, "learning_rate_init": 0.001,
        "max_iter": 40, "tol": 0.0, "n_iter_no_change": 41,
        "early_stopping": False, "shuffle": True, "random_state": 0,
    }


def test_failed_fit_falls_back_to_exact_e106_value():
    base={"status":"issued","prediction":123.5,"reason":None,"provenance":{"saved":True}}
    row=exp._fallback(base,"model_fit_failed:ValueError")
    assert row["prediction"]==base["prediction"]
    assert row["status"]==base["status"]
    assert row["fallbackReason"]=="model_fit_failed:ValueError"
    assert not row["usedMLP"]


def test_final_phase_requires_passing_confirmation(monkeypatch,tmp_path):
    monkeypatch.setattr(exp,"OUT",tmp_path)
    monkeypatch.setattr(exp,"verify_sources",lambda manifest,phase: None)
    expected={"primaryDecision":{"passes":True},"primaryGarlic10h":{"deltaMae":0.0,"deltaMape":-0.01,
              "candidateMae":10.0,"E106Mae":10.0,"candidateMape":5.0,"E106Mape":5.01}}
    monkeypatch.setattr(exp,"_primary_only",lambda manifest,phase: expected)
    manifest=tmp_path/"confirmation-manifest.json"; manifest.write_text('{"forecasts":[]}')
    report=tmp_path/"report.json"
    import json
    report.write_text(json.dumps({"phase":"confirmation","manifestFileSha256":exp.file_sha(manifest),
                                 "primaryDecision":{"passes":False}}))
    with pytest.raises(ValueError,match="gated"):
        exp._confirmation_gate(report)
    report.write_text(json.dumps({"phase":"confirmation","manifestFileSha256":exp.file_sha(manifest),
                                 "primaryDecision":{"passes":True},"primaryGarlic10h":expected["primaryGarlic10h"]}))
    assert exp._confirmation_gate(report)["primaryDecision"]["passes"]


def test_source_key_grid_is_exact_four_crop_eligible_e106():
    for phase in ('confirmation','final'):
        keys=exp.keys(phase)
        assert len(keys)==(3020 if phase=='confirmation' else 3160)
        assert {p for p,_,_ in keys}=={'cabbage','radish','onion','garlic'}
        assert all(day-dt.timedelta(days=h)>=exp.ORIGIN_START[phase] for _,day,h in keys)


def test_primary_gate_allows_equal_mae_but_requires_negative_mape_and_coverage():
    assert exp._primary_pass({"deltaMape":-0.01,"deltaMae":0.0},True,30)
    assert not exp._primary_pass({"deltaMape":0.0,"deltaMae":-1.0},True,30)
    assert not exp._primary_pass({"deltaMape":-0.01,"deltaMae":0.0},False,30)
    assert not exp._primary_pass({"deltaMape":-0.01,"deltaMae":0.0},True,29)


def test_saved_inference_state_replays_mlp_and_identical_50_50_blend():
    fit={"scalerMean":[0.0]*106,"scalerScale":[1.0]*106,"coefShapes":[[106,16],[16,1]],
         "coefs":[[0.0]*(106*16),[0.0]*16],"intercepts":[[0.0]*16,[0.0]]}
    fit["intercepts"][0][0]=1.0
    fit["coefs"][1][0]=2.0
    standalone,blend=exp._replay_prediction(fit,[0.0]*106,100.0,80.0)
    assert standalone==pytest.approx(100.0*__import__('math').exp(2.0))
    assert blend==pytest.approx(.5*standalone+.5*80.0)


def test_dm_family_is_fixed_at_twenty_tests():
    family=exp._dm_family({h:[] for h in exp.HORIZONS})
    assert len(family)==40
    assert sum(x['bandwidth']==9 for x in family)==20
    assert sum(x['bandwidth']==28 for x in family)==20
    assert {x['bandwidth'] for x in family}=={9,28}
    assert {x['loss'] for x in family}=={'ae','ape'}
    assert all(x['status']=='insufficient_pairs' for x in family)
