import datetime as dt
import unittest
from unittest.mock import patch

from agri_data import dashboard
from agri_data import state_space_pilot as pilot


class StateSpacePilotTests(unittest.TestCase):
    def test_frozen_dimensions(self):
        origins = list(pilot._dates())
        self.assertEqual((len(origins), origins[0], origins[-1]),
                         (61, dt.date(2025, 6, 1), dt.date(2025, 7, 31)))
        self.assertEqual(pilot.PLANNED, 915)
        self.assertEqual(len(dashboard.PRODUCTS), 5)
        self.assertEqual(pilot.FAMILY, 60)

    def test_state_space_refuses_insufficient_observations(self):
        series = {dt.date(2025, 1, 1) + dt.timedelta(days=i): (100.0, 0.0) for i in range(89)}
        value, audit = pilot._state_space(series, dt.date(2025, 6, 1), 1)
        self.assertIsNone(value)
        self.assertEqual(audit["reason"], "fewer_than_90_observations")

    def test_converged_state_space_uses_calendar_nan_window_and_h_plus_one(self):
        origin = dt.date(2025, 6, 1)
        series = {origin - dt.timedelta(days=i): (100.0 + i, 0.0) for i in range(1, 121)}
        captured = {}

        class Fit:
            mle_retvals = {"converged": True}

            def forecast(self, steps):
                captured["steps"] = steps
                return [0.0] * (steps - 1) + [__import__("math").log(123.0)]

        class Model:
            def __init__(self, y, **kwargs):
                captured["y"] = y.copy()
                captured["kwargs"] = kwargs

            def fit(self, **kwargs):
                captured["fit_kwargs"] = kwargs
                return Fit()

        with patch("statsmodels.tsa.statespace.structural.UnobservedComponents", Model):
            value, audit = pilot._state_space(series, origin, 1)
        self.assertAlmostEqual(value, 123.0)
        self.assertEqual(audit["statsmodelsVersion"], "0.14.0")
        self.assertEqual(audit["forecastSteps"], 2)
        self.assertEqual(captured["y"].shape, (365,))
        self.assertEqual(int(captured["y"].size - __import__("numpy").isfinite(captured["y"]).sum()), 245)
        self.assertEqual(captured["kwargs"], {"level": "local level", "seasonal": 7,
                                                "stochastic_seasonal": False})
        self.assertEqual(captured["fit_kwargs"], {"maxiter": 50, "disp": False})
        self.assertEqual(captured["steps"], 2)

    def test_dm_direction_and_holm_fixed_family(self):
        start = dt.date(2025, 1, 1)
        # Negative candidate-minus-comparator loss means improvement.
        result = pilot._dm({start + dt.timedelta(days=i): -1.0 + (i % 3) * .01 for i in range(61)}, 9)
        self.assertEqual(result["status"], "tested")
        self.assertLess(result["meanLossDifference"], 0)
        family = [dict(result) for _ in range(pilot.FAMILY)]
        pilot._holm(family)
        self.assertEqual(len(family), 60)
        self.assertTrue(all(x["holmSignificant"] for x in family))

    def test_dm_marks_small_samples_unavailable(self):
        result = pilot._dm({dt.date(2025, 1, 1): 1.0}, 28)
        self.assertEqual(result["status"], "insufficient_pairs")
        self.assertIsNone(result["meanLossDifference"])


if __name__ == "__main__":
    unittest.main()
