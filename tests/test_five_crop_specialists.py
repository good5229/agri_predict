import datetime as dt
import math

import pytest

from agri_data import five_crop_specialists as spec


class NoReadActual(dict):
    def get(self, key, default=None):
        if key == "actual":
            raise AssertionError("B actual was read during build")
        return super().get(key, default)


def _inputs(*, b_actual_trap=False):
    docs = {name: {} for name in spec.PATHS}
    docs["market"]["originEvidence"] = []
    docs["market"]["pairedRows"] = []
    docs["radish"]["pairedRows"] = []
    docs["garlic"]["pairedRows"] = []
    docs["pepper"]["selectionRows"] = []
    docs["pepper"]["pairedRows"] = []
    docs["historical"]["forecasts"] = []
    docs["test_components"]["forecasts"] = []
    docs["direct"]["pairedRows"] = []
    counts = {"A": spec.EXPECTED_A, "B": spec.EXPECTED_B}
    starts = {"A": spec.A[0], "B": spec.B[0]}
    for window in ("A", "B"):
        for crop in spec.CROPS:
            for row_index in range(counts[window][crop]):
                i, horizon = divmod(row_index, 10)
                horizon += 1
                target = starts[window]+dt.timedelta(days=i)
                origin = target-dt.timedelta(days=horizon)
                key = {"product": crop, "horizon": horizon, "origin": origin.isoformat(), "target": target.isoformat()}
                direct = 100.+i*.01+horizon*.03
                sarima = 101.+i*.005+horizon*.02
                fixed = (direct+sarima)/2 if window == "A" else 99.+i*.008+horizon*.025
                extra = 98.+i*.012+horizon*.04
                actual = fixed*math.exp(.03*math.log(extra/fixed)+.002*math.sin(i+horizon))
                if window == "A":
                    component = {**key,"status":"available","direct":direct,"sarima":sarima}
                else:
                    component = {**key,"predictions":{"direct":direct,"sarima":sarima,"fixed50":fixed}}
                (docs["historical"] if window=="A" else docs["test_components"])["forecasts"].append(component)
                source_key, signal_name = spec.SIGNAL_SOURCE[crop]
                source_row = {**key, signal_name:extra,"actual":actual}
                if crop in ("cabbage","onion"):
                    source_row.update({"fallback":False,"fallbackReasons":[]})
                    docs["market"]["originEvidence"].append({**source_row,"candidatePrediction":extra,"baselinePrediction":direct})
                    docs["market"]["pairedRows"].append({**key,"directRidge":direct})
                elif crop=="radish":
                    docs["radish"]["pairedRows"].append({k:v for k,v in source_row.items() if k!="product"} | {"candidate":extra,"directRidge":direct})
                elif crop=="garlic":
                    docs["garlic"]["pairedRows"].append({**source_row,"garlicMix":extra,"directRidge":direct})
                else:
                    field="selectionRows" if window=="A" else "pairedRows"
                    docs["pepper"][field].append({**source_row,"mix50":extra,"directRidge":direct,"fallbackReason":None})
                if window=="B" and b_actual_trap:
                    # Wrap the final row passed into the source after all expansions.
                    target_rows = (docs[source_key]["originEvidence"] if crop in ("cabbage","onion") else
                                   docs[source_key]["pairedRows"] if crop in ("radish","garlic") or window=="B" else
                                   docs[source_key]["selectionRows"])
                    # Market has separate evidence and direct rows. Trap the evidence
                    # row; all other crops store the signal on the paired/selection row.
                    target_index = len(target_rows)-1
                    target_rows[target_index] = NoReadActual(target_rows[target_index])
                direct_row={"product":crop,"target":target.isoformat(),"horizon":horizon,"direct":direct,"actual":actual}
                if window=="B" and b_actual_trap:
                    direct_row=NoReadActual(direct_row)
                docs["direct"]["pairedRows"].append(direct_row)
    return docs


def test_b_build_never_reads_b_actual_and_emits_answer_free_deterministic_manifest():
    docs=_inputs(b_actual_trap=True)
    forecasts,audit=spec._compose(docs)
    assert len(forecasts)==sum(spec.EXPECTED_B.values())==12_508
    assert all(spec._window(spec._date(r["target"]))=="B" for r in forecasts)
    assert all("actual" not in r and "targetValue" not in r for r in forecasts)
    pepper=next(r for r in forecasts if r["product"]=="pepper")
    assert set(pepper["predictions"])==set(spec.METHODS)
    assert audit["training"]["cabbage"]["status"]=="fitted"
    assert len(audit["training"]["cabbage"]["candidateScalerMean"])==7
    assert len(audit["training"]["cabbage"]["trainingRowProvenance"])>60
    garlic=next(r for r in forecasts if r["product"]=="garlic")
    pepper=next(r for r in forecasts if r["product"]=="pepper")
    assert garlic["sourceFallback"] is None and garlic["sourceFallbackStatus"]=="unknown"
    assert pepper["sourceFallback"] is False and pepper["sourceFallbackStatus"]=="known"


def test_key_direct_and_a_actual_mismatches_abort_without_intersection():
    docs=_inputs()
    docs["direct"]["pairedRows"][0]["direct"] += 1
    with pytest.raises(ValueError,match="direct prediction parity"):
        spec._compose(docs)
    docs=_inputs()
    docs["market"]["originEvidence"][0]["actual"] += 1
    with pytest.raises(ValueError,match="training actual parity"):
        spec._compose(docs)
    docs=_inputs()
    docs["pepper"]["pairedRows"].pop()
    with pytest.raises(ValueError,match="exact component, signal and direct key sets"):
        spec._compose(docs)
    docs=_inputs()
    docs["garlic"]["pairedRows"][0]["directRidge"]+=1
    with pytest.raises(ValueError,match="source direct prediction parity"):
        spec._compose(docs)


def test_invalid_b_signal_falls_back_both_models_to_fixed50():
    docs=_inputs()
    docs["pepper"]["pairedRows"][0]["mix50"]=None
    forecasts,audit=spec._compose(docs)
    row=next(r for r in forecasts if r["product"]=="pepper" and r["horizon"]==1 and r["target"]==docs["test_components"]["forecasts"][0]["target"])
    assert row["fallback"] is True
    assert row["predictions"][spec.CANDIDATE]==row["predictions"][spec.ABLATION]==row["predictions"][spec.FIXED]
    assert audit["testFallbacks"]["pepper"]["candidate"]>=1


def test_missing_extra_signal_reason_takes_precedence_when_model_unavailable():
    docs=_inputs()
    for row in docs["market"]["originEvidence"]:
        if spec._window(spec._date(row["target"]))=="A":
            row["candidatePrediction"]=None
    row=next(r for r in docs["market"]["originEvidence"] if spec._window(spec._date(r["target"]))=="B")
    row["candidatePrediction"]=None
    forecasts,_=spec._compose(docs)
    built=next(r for r in forecasts if r["product"]=="cabbage" and r["target"]==row["target"] and r["horizon"]==row["horizon"])
    assert built["fallbackReason"]=="extra_signal_not_positive_finite"
    manifest={"protocol":spec.PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,
              "inputSha256":spec.INPUT_HASHES,"plannedForecastCount":sum(spec.EXPECTED_B.values()),
              "forecasts":forecasts}
    manifest["manifestSha256"]=spec._hash(manifest)
    spec.verify_manifest(manifest)


def test_missing_b_sarima_uses_fixed50_fallback_and_preserves_comparator_missingness():
    docs=_inputs()
    component=docs["test_components"]["forecasts"][0]
    component["predictions"]["sarima"]=None
    forecasts,_=spec._compose(docs)
    row=next(r for r in forecasts if r["product"]==component["product"] and r["target"]==component["target"] and r["horizon"]==component["horizon"])
    assert row["predictions"][spec.SARIMA] is None
    assert row["predictions"][spec.CANDIDATE]==row["predictions"][spec.ABLATION]==row["predictions"][spec.FIXED]
    assert row["fallbackReason"]=="sarima_component_not_positive_finite"
    manifest={"protocol":spec.PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,
              "inputSha256":spec.INPUT_HASHES,"plannedForecastCount":sum(spec.EXPECTED_B.values()),
              "forecasts":forecasts}
    manifest["manifestSha256"]=spec._hash(manifest)
    spec.verify_manifest(manifest)


def test_manifest_verifier_rejects_answer_and_wrong_count():
    goodrow={"product":"cabbage","horizon":1,"origin":"2025-09-24","target":"2025-09-25",
             "predictions":{m:1. for m in spec.METHODS}}
    doc={"protocol":spec.PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,"inputSha256":spec.INPUT_HASHES,
         "plannedForecastCount":sum(spec.EXPECTED_B.values()),"forecasts":[goodrow]}
    doc["manifestSha256"]=spec._hash(doc)
    with pytest.raises(ValueError,match="12,508"):
        spec.verify_manifest(doc)
    doc["forecasts"]=[{**goodrow,"actual":1.}]
    doc["plannedForecastCount"]=sum(spec.EXPECTED_B.values())
    doc["manifestSha256"]=spec._hash({k:v for k,v in doc.items() if k!="manifestSha256"})
    with pytest.raises(ValueError,match="12,508"):
        spec.verify_manifest(doc)


def test_scoring_outputs_full_holm_family_and_distinguishes_joint_statuses():
    rows=[]
    start=spec.B[0]
    for crop in spec.CROPS:
        for h in spec.HORIZONS:
            for i in range(30):
                target=start+dt.timedelta(days=i); origin=target-dt.timedelta(days=h)
                rows.append({"product":crop,"horizon":h,"origin":origin.isoformat(),"target":target.isoformat(),
                             "predictions":{m:10.+i*.01 for m in spec.METHODS},"fallback":False})
    doc={"forecasts":rows}
    actuals={spec._key(r):10.+i*.015 for crop in spec.CROPS for h in spec.HORIZONS for i,r in enumerate([x for x in rows if x["product"]==crop and x["horizon"]==h])}
    report=spec._score(doc,actuals,{})
    assert len(report["dmComparisons"])==500
    assert report["holmFamilySizePerBandwidth"]==500
    assert len(report["jointConclusions"])==250
    assert report["coverageByCell"]["cabbage/1"]["planned"]==30
    assert report["gate"]["overallFiveCropMacro"]["candidateMapeAtMost10Pct"] is not None
    assert spec._joint(["improvement"]*4)=="joint_improvement"
    assert spec._joint(["inference_unavailable","inconclusive","inconclusive","inconclusive"])=="inference_unavailable"


def test_scored_coverage_separates_valid_targets_from_common_predictions():
    rows=[]; actuals={}
    for h in (1,2):
        for i in range(30):
            target=spec.B[0]+dt.timedelta(days=i); origin=target-dt.timedelta(days=h)
            predictions={m:10. for m in spec.METHODS}
            if h==1 and i==0: predictions[spec.EXTRA]=None
            row={"product":"cabbage","horizon":h,"origin":origin.isoformat(),"target":target.isoformat(),
                 "predictions":predictions,"fallback":False}
            rows.append(row); actuals[spec._key(row)]=11.
    report=spec._score({"forecasts":rows},actuals,{})
    cov=report["coverageByCell"]["cabbage/1"]
    assert cov["planned"]==30
    assert cov["validPositiveTargets"]==30
    assert cov["issued"][spec.EXTRA]==29
    assert cov["commonScored"]==29
