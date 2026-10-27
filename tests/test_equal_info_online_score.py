import datetime as dt
import hashlib
import unittest
from unittest import mock

from agri_data import equal_info_online_score as score


def manifest(start, end, products=("cabbage",), horizons=(1,)):
    docs=[]
    for day in (start + dt.timedelta(days=i) for i in range((end-start).days+1)):
        for product in products:
            for h in horizons:
                origin=day-dt.timedelta(days=h)
                models={m:{"status":"issued","reason":None,"prediction":100.0,"training":None,"provenance":{}} for m in score.METHODS}
                docs.append({"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":origin>=start,"models":models})
    doc={"phase":"forecast_only","targetValuesAttached":False,"confirmationForecasts":docs,"evaluationForecasts":docs}
    doc["manifestSha256"]=hashlib.sha256(score._canonical(doc)).hexdigest()
    return doc


class ScoreTests(unittest.TestCase):
    def test_dm_calendar_gap_and_low_sample_reason(self):
        d={dt.date(2026,1,1)+dt.timedelta(days=i*2):-1.0 for i in range(30)}
        out=score._dm(d,9)
        self.assertEqual(out["status"],"zero_spread")
        low=score._dm({dt.date(2026,1,1):0.1},9)
        self.assertEqual(low["status"],"insufficient_pairs")
        self.assertEqual(low["pValue"],1.)

    def test_score_requires_verified_replay_and_keeps_answer_free_grid(self):
        start=dt.date(2026,1,1);end=dt.date(2026,1,3)
        doc=manifest(start,end)
        raw={"schemaVersion":1,"products":[{"id":"cabbage","observations":[{"date":(start+dt.timedelta(days=i)).isoformat(),"actual":100.0} for i in range(3)]}]}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,),CONF_START=start,CONF_END=end,EVAL_START=start,EVAL_END=end), mock.patch.object(score.price_helper,"PRODUCTS",("cabbage",)):
            with self.assertRaisesRegex(ValueError,"replay"):
                score.score(doc,raw)
            result=score.score(doc,raw,replay_verified=True)
            self.assertEqual(result["phase"],"scored")
            self.assertEqual(len(result["evaluation2026"]["dm"]),16)
            self.assertTrue(all(r["status"]=="insufficient_pairs" for r in result["evaluation2026"]["dm"]))
            self.assertEqual(result["evaluation2026"]["dmSummaryByBandwidth"][0]["familySize"],8)

    def test_gate_requires_all_confirmation_horizons(self):
        candidate=score.COMPARISONS[0][0]
        conf={"equalHorizonMetrics":[{"product":"cabbage","comparison":candidate,"horizonsIncluded":1,"aEqualHorizonMape":1.,"bEqualHorizonMape":2.,"aEqualHorizonMae":1.,"bEqualHorizonMae":2.}],"metrics":[],"coverage":{},"dm":[]}
        evaluation={"equalHorizonMetrics":[{"product":"cabbage","comparison":candidate,"horizonsIncluded":2,"aEqualHorizonMape":1.,"bEqualHorizonMape":2.,"aEqualHorizonMae":1.,"bEqualHorizonMae":2.}],"metrics":[{"product":"cabbage","horizon":h,"comparison":candidate,"n":30,"aMape":1.,"bMape":2.} for h in (1,2)],"coverage":{"cabbage/1":{},"cabbage/2":{}},"dm":[]}
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,2)):
            gate=score._gates(evaluation,conf)[0]
        self.assertFalse(gate["confirmationImprovedBoth"])
        self.assertFalse(gate["confirmationAllHorizonsSupported"])
        self.assertEqual(gate["confirmationSupportReason"],"confirmation_macro_does_not_include_all_horizons")

    def test_manifest_rejects_mutation_and_nonissued_prediction(self):
        start=dt.date(2026,1,1);doc=manifest(start,start)
        with mock.patch.multiple(score,PRODUCTS=("cabbage",),HORIZONS=(1,),CONF_START=start,CONF_END=start,EVAL_START=start,EVAL_END=start):
            score._validate_manifest(doc)
            doc["evaluationForecasts"][0]["models"]["online71"]["status"]="not_issued"
            with self.assertRaises(ValueError):score._validate_manifest(doc)
