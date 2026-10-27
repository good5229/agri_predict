import datetime as dt
import unittest
from decimal import Decimal
from unittest.mock import patch

from agri_data import regional_gap, regional_selective_near_date as selective


def cell(price, kg, label="M", codes=("41", "01", "100", "101")):
    names=[{"소분류"},{"등급"},{label},{"포장"}]
    return {"amount":Decimal(str(price*kg)),"weight":Decimal(str(kg)),"rows":1,"names":names}


def put(cells, day, market, price, kg, label="M", codes=("41", "01", "100", "101")):
    cells[("garlic",day,market,codes[0],tuple(codes))]=cell(price,kg,label,codes)


class SelectiveNearDateTests(unittest.TestCase):
    def test_tie_chooses_earlier_seoul_and_no_reuse(self):
        cells={}
        put(cells,"2020-01-02",selective.GWANGJU,10,2)
        put(cells,"2020-01-01",selective.GARAK,5,2)
        put(cells,"2020-01-03",selective.GANGSEO,5,2)
        r=selective._match_scenario(cells,{}, {},2)["garlic"]
        self.assertEqual(r["signedLagCounts"],{"-1":1})
        self.assertEqual([(x["gwangjuDate"],x["seoulDate"]) for x in r["matchedPairs"]],[("2020-01-02","2020-01-01")])

    def test_exact_date_first_over_nearer_lag_and_distinct_strata(self):
        cells={}
        # Two independent exact code strata; the first has an exact and a nearby Seoul candidate.
        put(cells,"2020-01-02",selective.GWANGJU,10,2)
        put(cells,"2020-01-01",selective.GARAK,5,2)
        put(cells,"2020-01-02",selective.GANGSEO,4,2)
        other=("41","02","200","101")
        put(cells,"2020-01-02",selective.GWANGJU,20,1,codes=other)
        put(cells,"2020-01-03",selective.GARAK,10,1,codes=other)
        r=selective._match_scenario(cells,{}, {},1)["garlic"]
        self.assertEqual(r["signedLagCounts"],{"0":1,"1":1})
        self.assertEqual(r["matchedCells"],2)

    def test_zero_net_is_counted_and_not_paired(self):
        cells={}
        put(cells,"2020-01-02",selective.GWANGJU,10,2)
        put(cells,"2020-01-01",selective.GARAK,5,0)
        r=selective._match_scenario(cells,{}, {},1)["garlic"]
        x=r["excluded"]["nonpositive_net_pair"]
        self.assertEqual(x["candidatePairs"],1)
        self.assertEqual(x["potentialCommonKg"],0)
        self.assertEqual(r["matchedCells"],0)

    def test_name_policy_unknown_placeholder_and_conflict(self):
        for name in ("", "."):
            cells={};put(cells,"2020-01-01",selective.GWANGJU,10,2,label=name);put(cells,"2020-01-01",selective.GARAK,5,2,label=name)
            r=selective._match_scenario(cells,{}, {},0)["garlic"]
            self.assertEqual(r["unknownLabelMatchedCells"],1)
        cells={};put(cells,"2020-01-01",selective.GWANGJU,10,2,label="-");put(cells,"2020-01-01",selective.GARAK,5,2)
        r=selective._match_scenario(cells,{}, {},0)["garlic"]
        self.assertIn("placeholder_dimension_name",r["excluded"])
        cells={};put(cells,"2020-01-01",selective.GWANGJU,10,2);put(cells,"2020-01-01",selective.GARAK,5,2)
        cells[("garlic","2020-01-01",selective.GARAK,"41",("41","01","100","101"))]["names"][2].add("other")
        r=selective._match_scenario(cells,{}, {},0)["garlic"]
        self.assertIn("dimension_name_conflict",r["excluded"])

    def test_scope_incomplete_blocks_both_anchor_and_seoul_date(self):
        cells={};put(cells,"2020-01-02",selective.GWANGJU,10,2);put(cells,"2020-01-01",selective.GARAK,5,2)
        states={"scopeStates":{"garlic|2020-01-02|240004":["source_incomplete:pending"]}}
        r=selective._match_scenario(cells,states,{},1)["garlic"]
        self.assertIn("source_incomplete",r["excluded"])
        self.assertEqual(r["matchedCells"],0)

    def test_coverage_and_signed_lag(self):
        cells={};put(cells,"2020-01-02",selective.GWANGJU,10,4);put(cells,"2020-01-01",selective.GARAK,5,2)
        eligible={("garlic","2020-01-02",selective.GWANGJU,("41","01","100","101")): {"weight":Decimal(8)},
                  ("garlic","2020-01-01",selective.GARAK,("41","01","100","101")): {"weight":Decimal(4)}}
        r=selective._match_scenario(cells,{},eligible,1)["garlic"]
        self.assertEqual(r["signedLagCounts"],{"-1":1})
        self.assertEqual(r["matchedWeightKg"],2)
        self.assertEqual(r["shareOfGwangjuEligibleKg"],.25)
        self.assertEqual(r["shareOfSeoulEligibleKg"],.5)

    def test_matched_pair_records_exactly_four_codes(self):
        cells={}; put(cells,"2020-01-01",selective.GWANGJU,10,2); put(cells,"2020-01-01",selective.GARAK,5,2)
        pair=selective._match_scenario(cells,{}, {},0)["garlic"]["matchedPairs"][0]
        self.assertEqual(pair["codes"],["41","01","100","101"])

    def test_output_paths_cannot_overwrite_frozen_baselines(self):
        baseline=selective.RECORDED_CODE_REPORT
        with self.assertRaises(ValueError):
            selective._validate_output_paths("/tmp/source.sqlite",baseline,"/tmp/report.md")
        with self.assertRaises(ValueError):
            selective._validate_output_paths("/tmp/source.sqlite","/tmp/out.json",selective.STANDARDIZED_REPORT)
        self.assertEqual(selective._validate_output_paths("/tmp/source.sqlite","/tmp/out.json","/tmp/out.md")[1:],
                         (selective.Path("/tmp/out.json").resolve(),selective.Path("/tmp/out.md").resolve()))

    def test_zero_day_baseline_checks_dates_cells_weight_and_daily_values(self):
        cells={};put(cells,"2020-01-01",selective.GWANGJU,10,2);put(cells,"2020-01-01",selective.GARAK,5,2)
        got=selective._match_scenario(cells,{}, {},0)
        p=got["garlic"]
        baseline={"products":[{"productId":"garlic","matchedDates":p["matchedDates"],"matchedCells":p["matchedCells"],
            "matchedWeightKg":p["matchedWeightKg"],"pairedDailyLogGaps":p["dailyLogGaps"]},
            {"productId":"pepper","matchedDates":0,"matchedCells":0,"matchedWeightKg":0,"pairedDailyLogGaps":{}}]}
        expected=dict(selective.BASELINE_EXPECTED)
        expected["garlic"]={"matchedDates":p["matchedDates"],"matchedCells":p["matchedCells"],"matchedWeightKg":p["matchedWeightKg"]}
        expected["pepper"]={"matchedDates":0,"matchedCells":0,"matchedWeightKg":0}
        with patch.dict(selective.BASELINE_EXPECTED,expected,clear=True):
            selective._assert_baseline(got,baseline)
            baseline["products"][0]["pairedDailyLogGaps"]={"2020-01-01":99}
            with self.assertRaises(ValueError): selective._assert_baseline(got,baseline)

    def test_two_day_inference_and_holm_six_only(self):
        # Verify analysis wiring without invoking expensive inference or loading any database.
        empty_standard={"products":[{"productId":p} for p in regional_gap.PRODUCTS]}
        fake_products=[]
        for p in selective.SELECTED_PRODUCTS:
            fake_products.append({"productId":p,"matchedDates":0,"matchedCells":0,"matchedWeightKg":0,
                                  "pairedDailyLogGaps":{},"inference":{}})
        base={"products":fake_products}
        expected={p:{"matchedDates":0,"matchedCells":0,"matchedWeightKg":0} for p in selective.SELECTED_PRODUCTS}
        fake_infer={"status":"inference_unavailable","meanZeroTest":{},"equivalenceTOST":{},"smallGapDays":{}}
        with patch.dict(selective.BASELINE_EXPECTED,expected,clear=True), patch.object(selective.regional_gap,"_infer",return_value=fake_infer) as infer:
            result=selective.analyze_cells({},recorded_code_baseline=base,standardized_baseline=empty_standard)
        self.assertNotIn("inference",result["scenarios"]["0"]["garlic"])
        self.assertNotIn("inference",result["scenarios"]["1"]["garlic"])
        self.assertIn("inference",result["scenarios"]["2"]["garlic"])
        self.assertEqual(len(result["multipleTestingTolerance2"]),6)
        self.assertEqual(infer.call_count,2)


if __name__=="__main__":unittest.main()
