import datetime as dt
import unittest
from decimal import Decimal
from unittest.mock import patch

from agri_data import regional_near_date as near, regional_standardize as standard


def record(gw=None, garak=None, gangseo=None, size_name="M"):
    markets={}
    for market, pair in ((near.GWANGJU,gw),(near.GARAK,garak),(near.GANGSEO,gangseo)):
        if pair is not None:
            price, weight = pair
            names={f:{"M"} for f in standard.NAME_FIELDS}
            names["sz_nm"]={size_name}
            markets[market]={"amount":Decimal(str(price*weight)),"weight":Decimal(str(weight)),"names":names}
    return {"markets":markets,"unitQtySourceValues":{"1"}}


def strata_row(day, rec, identity="same"):
    return ("cabbage",day,identity),rec


class NearDateTests(unittest.TestCase):
    def test_exact_first_and_one_to_one_and_earlier_seoul_tie(self):
        s=dict([strata_row("2020-01-01",record(None,(5,2))),
                strata_row("2020-01-02",record((20,2),None)),
                strata_row("2020-01-03",record(None,(5,2)))])
        r=near._match_tolerance(s,{}, {},2)["cabbage"]
        self.assertEqual(r["signedLagCounts"],{"-1":1})
        self.assertEqual(r["matchedCells"],1)

    def test_missing_and_dot_names_are_counted_unknown(self):
        x=record((10,2),(5,2),size_name=".")
        r=near._match_tolerance({strata_row("2020-01-01",x)[0]:x},{},{},0)["cabbage"]
        self.assertEqual(r["matchedCells"],1)
        self.assertEqual(r["sizeUnknownCells"],1)
        self.assertEqual(r["sizeUnknownShareOfMatchedWeightKg"],1.0)

    def test_conflicting_known_label_rejected_and_signed_lag(self):
        x=record((10,2),(5,2))
        x["markets"][near.GARAK]["names"]["grd_nm"]={"M","conflict"}
        key=("cabbage","2020-01-01","id")
        self.assertEqual(near._match_tolerance({key:x},{},{},0)["cabbage"]["matchedCells"],0)
        good_gw=record((10,2),None); good_seoul=record(None,(5,2))
        rows={("cabbage","2020-01-02","id"):good_gw,("cabbage","2020-01-01","id"):good_seoul}
        r=near._match_tolerance(rows,{},{},1)["cabbage"]
        self.assertEqual(r["signedLagCounts"],{"-1":1})

    def test_matched_kg_and_denominator_coverage(self):
        x=record((10,4),(5,2))
        eligible={("cabbage","2020-01-01",near.GWANGJU,"a"):Decimal(8),
                  ("cabbage","2020-01-01",near.GARAK,"a"):Decimal(4)}
        r=near._match_tolerance({("cabbage","2020-01-01","id"):x},{},eligible,0)["cabbage"]
        self.assertEqual(r["matchedWeightKg"],2)
        self.assertEqual(r["shareOfGwangjuEligibleKg"],.25)
        self.assertEqual(r["shareOfSeoulEligibleKg"],.5)

    def test_scope_incomplete_blocks_match(self):
        x=record((10,2),(5,2)); states={"scopeStates":{"cabbage|2020-01-01|240004":["source_incomplete:pending"]}}
        r=near._match_tolerance({("cabbage","2020-01-01","id"):x},states,{},0)["cabbage"]
        self.assertEqual(r["matchedCells"],0)

    def test_exclusions_report_rejected_candidate_edges_and_potential_kg(self):
        key=("cabbage","2020-01-01","id")
        incomplete=record((10,3),(5,2))
        states={"scopeStates":{"cabbage|2020-01-01|240004":["source_incomplete:pending"]}}
        r=near._match_tolerance({key:incomplete},states,{},0)["cabbage"]["excluded"]["source_incomplete"]
        self.assertEqual(r,{"candidatePairs":1,"potentialCommonKg":2.0})

        conflict=record((10,3),(5,2))
        conflict["markets"][near.GARAK]["names"]["grd_nm"]={"M","conflict"}
        r=near._match_tolerance({key:conflict},{},{},0)["cabbage"]["excluded"]["meaningful_name_conflict"]
        self.assertEqual(r,{"candidatePairs":1,"potentialCommonKg":2.0})

        nonpositive=record((10,-3),(5,2))
        r=near._match_tolerance({key:nonpositive},{},{},0)["cabbage"]["excluded"]["nonpositive_net_pair"]
        self.assertEqual(r,{"candidatePairs":1,"potentialCommonKg":0.0})

        zero_seoul=record((10,2),(5,0))
        r=near._match_tolerance({key:zero_seoul},{},{},0)["cabbage"]["excluded"]["nonpositive_net_pair"]
        self.assertEqual(r,{"candidatePairs":1,"potentialCommonKg":0.0})

    def test_zero_day_baseline_and_inference_only_two_day(self):
        d=dt.date(2020,1,1)
        s={ ("cabbage",d.isoformat(),"same"):record((10,2),(5,2)) }
        # Other registered products may be empty. Use mocked inference to assert scenario gating.
        fake={"status":"insufficient_sample","meanZeroTest":{},"equivalenceTOST":{},"smallGapDays":{}}
        with patch.object(near.regional_gap,"_infer",return_value=fake) as infer:
            result=near.analyze_strata(s)
        self.assertEqual(result["scenarios"]["0"]["cabbage"]["matchedDates"],1)
        self.assertNotIn("inference",result["scenarios"]["0"]["cabbage"])
        self.assertNotIn("inference",result["scenarios"]["1"]["cabbage"])
        self.assertEqual(infer.call_count,2*len(standard.PRODUCTS))
        self.assertIn("inference",result["scenarios"]["2"]["cabbage"])


if __name__=="__main__": unittest.main()
