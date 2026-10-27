import unittest

from agri_data.full_history_hyperparam_direct_diagnostic import _by_horizon


class SavedForecastDiagnosticTests(unittest.TestCase):
    def test_equal_horizon_metrics_use_same_rows_and_weight_horizons_equally(self):
        rows=[
            {"horizon":1,"actual":100.,"ridgeTuned":110.,"directRidge":105.,"fixed50":120.},
            {"horizon":1,"actual":100.,"ridgeTuned":110.,"directRidge":105.,"fixed50":120.},
            {"horizon":2,"actual":1000.,"ridgeTuned":1000.,"directRidge":1100.,"fixed50":900.},
        ]
        result=_by_horizon(rows,("ridgeTuned","directRidge","fixed50"))
        macro=result["equalHorizon"]
        self.assertEqual(macro["ridgeTuned"]["horizonsIncluded"],2)
        self.assertAlmostEqual(macro["ridgeTuned"]["equalHorizonMape"],5.)
        self.assertAlmostEqual(macro["directRidge"]["equalHorizonMape"],7.5)
        self.assertAlmostEqual(macro["fixed50"]["equalHorizonMape"],15.)
        self.assertEqual(result["horizons"]["1"]["ridgeTuned"]["n"],2)
        self.assertEqual(result["horizons"]["2"]["ridgeTuned"]["n"],1)


if __name__=="__main__":unittest.main()
