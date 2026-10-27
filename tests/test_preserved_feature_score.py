import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agri_data import preserved_feature_score as score


def prediction(value=None, status="issued"):
    return {"status":status,"reason":None if status=="issued" else "synthetic_missing","prediction":value if status=="issued" else None,"training":{},"provenance":{}}


def row(day, *, fixed=True, direct=True, eligible=True):
    models={"preserved106":prediction(100.5 if direct else None,"issued" if direct else "not_issued"),"preserved106Blend":prediction(100.5),"causal50":prediction(101.),"causalDirect":prediction(102.),"causalSarima":prediction(101.5),"fixed50":prediction(100.) if fixed else prediction(None,"not_issued")}
    return {"product":"cabbage","target":day.isoformat(),"origin":(day-dt.timedelta(days=1)).isoformat(),"horizon":1,"eligible":eligible,"selectedCrop":True,"models":models}


class PreservedFeatureScoreTests(unittest.TestCase):
    def test_main_contrasts_share_exact_three_way_common_population(self):
        start=dt.date(2026,9,22)
        rows=[row(start+dt.timedelta(days=i),fixed=i!=1,direct=i!=2) for i in range(3)]
        baseline={}
        for i in range(3):
            d=start+dt.timedelta(days=i)
            models={"causal50":prediction(101.),"causalDirect":prediction(102.),"causalSarima":prediction(101.5),"fixed50":prediction(100.) if i!=1 else prediction(None,"not_issued")}
            baseline[("cabbage",d,1)]={"product":"cabbage","target":d.isoformat(),"horizon":1,"models":models}
        actuals={("cabbage",start+dt.timedelta(days=i)):100. for i in range(3)}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,)):
            sets,_=score._common_sets(rows,baseline,actuals,("cabbage",),"confirmation")
        self.assertEqual(len(sets[score.MAIN_COMPARISONS[0][0]]),2)
        self.assertEqual(len(sets[score.MAIN_COMPARISONS[1][0]]),2)
        self.assertEqual(len(sets[score.COMPARISONS[2][0]]),1)
        self.assertEqual({r["target"] for r in sets[score.MAIN_COMPARISONS[0][0]]},{r["target"] for r in sets[score.MAIN_COMPARISONS[1][0]]})

    def test_300_family_shape_and_calendar_hac_reasoning_on_synthetic_period(self):
        start=dt.date(2026,1,1);rows=[row(start+dt.timedelta(days=i)) for i in range(35)]
        baseline={("cabbage",start+dt.timedelta(days=i),1):{"models":{"causal50":prediction(101.),"causalDirect":prediction(102.),"causalSarima":prediction(101.5),"fixed50":prediction(100.)}} for i in range(35)}
        actuals={("cabbage",start+dt.timedelta(days=i)):100. for i in range(35)}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,)):
            report=score._evaluate(rows,baseline,actuals,("cabbage",),"confirmation",start,start+dt.timedelta(days=34))
        self.assertEqual(len(report["dm"]),3*2*2)
        self.assertTrue(all(r["status"]=="zero_spread" for r in report["dm"]))
        self.assertEqual(report["dmSummaryByBandwidth"][0]["familySize"],6)

    def test_confirmation_gate_fails_when_horizon_support_is_below_30(self):
        start=dt.date(2025,10,1);rows=[row(start+dt.timedelta(days=i)) for i in range(29)]
        baseline={("cabbage",start+dt.timedelta(days=i),1):{"models":{"causal50":prediction(101.),"causalDirect":prediction(102.),"causalSarima":prediction(101.5),"fixed50":prediction(100.)}} for i in range(29)}
        actuals={("cabbage",start+dt.timedelta(days=i)):100. for i in range(29)}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,)):
            report=score._evaluate(rows,baseline,actuals,("cabbage",),"confirmation",start,start+dt.timedelta(days=28))
            gates=score._crop_gate(report)
        self.assertFalse(gates[0]["eligible"])
        self.assertFalse(gates[0]["all10HorizonsAtLeast30Pairs"])

    def test_unselected_evaluation_has_no_predictions_or_successful_dm(self):
        start=dt.date(2026,1,1);rows=[row(start+dt.timedelta(days=i)) for i in range(2)]
        baseline={("cabbage",start+dt.timedelta(days=i),1):{"models":{"causal50":prediction(101.),"causalDirect":prediction(102.),"fixed50":prediction(100.)}} for i in range(2)}
        actuals={("cabbage",start+dt.timedelta(days=i)):100. for i in range(2)}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,)):
            report=score._evaluate(rows,baseline,actuals,(),"evaluation",start,start+dt.timedelta(days=1))
        self.assertTrue(all(r["status"]=="not_selected" and r["pValue"]==1. for r in report["dm"]))
        self.assertEqual(report["equalHorizonMetrics"][0]["horizonsIncluded"],0)

    def test_decision_digest_detects_edit(self):
        d={"schemaVersion":1,"kind":"preserved_feature_confirmation_gate","selectedCrops":["cabbage"]}
        d["decisionSha256"]=score._unsigned_digest(d,"decisionSha256")
        self.assertEqual(d["decisionSha256"],score._unsigned_digest(d,"decisionSha256"))
        d["selectedCrops"]=[]
        self.assertNotEqual(d["decisionSha256"],score._unsigned_digest(d,"decisionSha256"))

    def test_confirmation_gate_replay_api_rejects_report_decision_drift(self):
        with TemporaryDirectory() as folder:
            forecast=Path(folder)/"confirmation.json";report_path=Path(folder)/"report.json";decision_path=Path(folder)/"decision.json"
            manifest={"manifestSha256":"a"*64};forecast.write_bytes(score._canonical(manifest))
            report={"cropDecisions":[{"product":"cabbage","eligible":True}],"reportSha256":"b"*64}
            report_path.write_bytes(score._canonical(report));decision=score.create_decision(forecast,manifest,report);decision_path.write_bytes(score._canonical(decision))
            with mock.patch.object(score,"score_confirmation_paths",return_value=report),mock.patch.object(score,"PRODUCTS",("cabbage",)):
                selected,_=score.validate_confirmation_gate_paths(forecast,report_path,decision_path,Path("baseline"),Path("prices"))
                self.assertEqual(selected,["cabbage"])
                decision["selectedCrops"]=[];decision["decisionSha256"]=score._unsigned_digest(decision,"decisionSha256");decision_path.write_bytes(score._canonical(decision))
                with self.assertRaisesRegex(ValueError,"differs"):
                    score.validate_confirmation_gate_paths(forecast,report_path,decision_path,Path("baseline"),Path("prices"))

    def test_unselected_policy_uses_fixed50_and_preserves_exact_availability(self):
        start=dt.date(2026,9,24);rows=[row(start+dt.timedelta(days=i),fixed=False) for i in range(3)]
        baseline={}
        for i in range(3):
            day=start+dt.timedelta(days=i)
            baseline[("cabbage",day,1)]={"models":{"fixed50":prediction(105.) if i<2 else prediction(None,"not_issued")}}
        actuals={("cabbage",start+dt.timedelta(days=i)):100. for i in range(3)}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,)):
            report=score._policy_portfolio(rows,baseline,actuals,(),)
        cell=report["cellMetrics"][0]
        self.assertFalse(cell["selectedCropUses106Blend"])
        self.assertEqual(cell["fixed50AvailableKeys"],2)
        self.assertEqual(cell["positiveActualCommonPairs"],2)
        self.assertEqual(cell["policyMae"],cell["fixed50Mae"])
        self.assertEqual(cell["policyMape"],cell["fixed50Mape"])

    def test_selected_policy_uses_fixed50_only_when_candidate_is_missing(self):
        day=dt.date(2026,1,1);r=row(day);r["models"]["preserved106Blend"]=prediction(None,"not_issued")
        base={("cabbage",day,1):{"models":{"fixed50":prediction(105.)}}}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,)):
            report=score._policy_portfolio([r],base,{("cabbage",day):100.},("cabbage",))
        cell=report["cellMetrics"][0]
        self.assertEqual(cell["selectedCandidateFixed50FallbackTargets"],[day.isoformat()])
        self.assertEqual(cell["policyMae"],cell["fixed50Mae"])

    def test_phase_manifest_validation_binds_phase_and_selection(self):
        day=dt.date(2026,1,1)
        models={m:prediction(100.) for m in ("causalDirect","causalSarima","causal50","fixed50","preserved106","preserved106Blend")}
        r={"product":"cabbage","target":day.isoformat(),"origin":(day-dt.timedelta(days=1)).isoformat(),"horizon":1,"eligible":False,"selectedCrop":True,"models":models}
        def doc(phase):
            d={"schemaVersion":1,"phase":"forecast_only","phaseName":phase,"targetValuesAttached":False,"plannedCount":1,"protocol":{"phase":phase,"evaluationSelection":["cabbage"] if phase=="evaluation" else None},"selectedCrops":["cabbage"] if phase=="evaluation" else None,"forecasts":[r]}
            d["manifestSha256"]=score._unsigned_digest(d,"manifestSha256")
            return d
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,),CONF_START=day,CONF_END=day,EVAL_START=day,EVAL_END=day):
            score._validate_phase_manifest(doc("evaluation"),"evaluation")
            bad=doc("evaluation");bad["forecasts"][0]["models"]["preserved106Blend"]=prediction(None,"not_selected");bad["manifestSha256"]=score._unsigned_digest(bad,"manifestSha256")
            with self.assertRaises(ValueError):score._validate_phase_manifest(bad,"evaluation")

    def test_confirmation_gate_uses_equal_horizon_means_not_each_cell(self):
        metrics=[];macros=[];coverage={}
        for name,_,base in score.MAIN_COMPARISONS:
            vals=[]
            for h,am,bm in ((1,1.,10.),(2,10.5,10.)):
                metrics.append({"product":"cabbage","horizon":h,"comparison":name,"n":30,"maeA":am,"maeB":bm,"mapeA":am,"mapeB":bm})
                coverage[f"cabbage/{h}"]={f"{score.CANDIDATE}Vs{base}CoverageLossTargets":[]}
                vals.append((am,bm))
            macros.append({"product":"cabbage","comparison":name,"horizonsIncluded":2,"aEqualHorizonMae":sum(x[0] for x in vals)/2,"bEqualHorizonMae":sum(x[1] for x in vals)/2,"aEqualHorizonMape":sum(x[0] for x in vals)/2,"bEqualHorizonMape":sum(x[1] for x in vals)/2})
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,2)):
            decision=score._crop_gate({"metrics":metrics,"equalHorizonMetrics":macros,"coverage":coverage})[0]
        self.assertTrue(decision["eligible"])

    def test_evaluation_joint_significance_needs_one_same_horizon_for_both_baselines(self):
        metrics=[];macros=[];coverage={};dm=[]
        for name,_,base in score.MAIN_COMPARISONS:
            macros.append({"product":"cabbage","comparison":name,"horizonsIncluded":2,"aEqualHorizonMape":1.,"bEqualHorizonMape":2.,"aEqualHorizonMae":1.,"bEqualHorizonMae":2.})
            for h in (1,2):
                metrics.append({"product":"cabbage","horizon":h,"comparison":name,"n":40,"mapeA":1.,"mapeB":2.})
                coverage[f"cabbage/{h}"]={f"{score.CANDIDATE}Vs{base}CoverageLossTargets":[]}
        for h in (1,2):
            for _,_,base in score.MAIN_COMPARISONS:
                comp=f"{score.CANDIDATE}_vs_{base}"
                for loss in ("ae","ape"):
                    for bw in score.BANDWIDTHS:
                        good=(h==1)
                        dm.append({"product":"cabbage","horizon":h,"comparison":comp,"loss":loss,"bandwidth":bw,"status":"tested","holmSignificant":good,"meanLossDifference":-1. if good else 0.1})
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,2)):
            result=score._eval_gates({"equalHorizonMetrics":macros,"metrics":metrics,"coverage":coverage,"dm":dm},("cabbage",))[0]
        self.assertTrue(result["sameHorizonJointAeApeHolmImprovementAgainstBothAtHac9And28"])
        self.assertEqual(result["decision"],"continue_candidate")
