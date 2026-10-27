import datetime as dt
import math
import unittest
from unittest.mock import patch

import numpy as np

from agri_data import dashboard
from agri_data import harmonic_ar_pilot as pilot


class HarmonicArPilotTests(unittest.TestCase):
    def test_frozen_dimensions_and_fourier_epoch(self):
        origins = list(pilot._dates())
        self.assertEqual((len(origins), origins[0], origins[-1]),
                         (55, dt.date(2025, 8, 1), dt.date(2025, 9, 24)))
        self.assertEqual(pilot.PLANNED, 825)
        self.assertEqual(len(dashboard.PRODUCTS), 5)
        self.assertEqual(pilot.FAMILY, 60)
        self.assertEqual(pilot._fourier_row(pilot.EPOCH), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def test_refuses_fewer_than_90_observations_without_fitting(self):
        origin = dt.date(2025, 8, 1)
        series = {origin - dt.timedelta(days=i): (100.0 + i, 0.0) for i in range(1, 90)}
        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as sarimax:
            values, audit = pilot._harmonic_forecasts(series, origin)
        self.assertEqual(values, {1: None, 5: None, 10: None})
        self.assertEqual(audit["reason"], "fewer_than_90_observations")
        sarimax.assert_not_called()

    def test_mocked_sarimax_keeps_calendar_gaps_and_forecasts_h_plus_one(self):
        origin = dt.date(2025, 8, 1)
        begin = origin - dt.timedelta(days=pilot.WINDOW_DAYS)
        series = {begin + dt.timedelta(days=i): (100.0 + i / 10, 0.0)
                  for i in range(pilot.WINDOW_DAYS) if i % 3 != 0}
        captured = {}

        class Fit:
            mle_retvals = {"converged": True}

            def get_forecast(self, *, steps, exog):
                captured["forecast_steps"] = steps
                captured["future_exog"] = exog.copy()
                return type("Prediction", (), {"predicted_mean": np.log(np.arange(200, 200 + steps, dtype=float))})()

        class Model:
            def __init__(self, y, **kwargs):
                captured["y"] = y.copy()
                captured["exog"] = kwargs.pop("exog").copy()
                captured["kwargs"] = kwargs

            def fit(self, **kwargs):
                captured["fit_kwargs"] = kwargs
                return Fit()

        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX", Model):
            values, audit = pilot._harmonic_forecasts(series, origin)

        self.assertEqual(audit["fitStatus"], "converged")
        self.assertEqual(audit["observations"], sum(i % 3 != 0 for i in range(730)))
        self.assertEqual(captured["y"].shape, (730,))
        self.assertEqual(int(np.isnan(captured["y"]).sum()), sum(i % 3 == 0 for i in range(730)))
        self.assertEqual(captured["exog"].shape, (730, 6))
        self.assertEqual(captured["kwargs"], {"order": (1, 0, 0), "trend": "c",
                         "enforce_stationarity": True, "enforce_invertibility": True})
        self.assertEqual(captured["fit_kwargs"], {"maxiter": 50, "disp": False})
        self.assertEqual(captured["forecast_steps"], 11)
        self.assertEqual(captured["future_exog"].shape, (11, 6))
        np.testing.assert_allclose(captured["future_exog"], np.asarray([
            pilot._fourier_row(origin + dt.timedelta(days=i)) for i in range(11)]))
        for horizon, expected in {1: 201.0, 5: 205.0, 10: 210.0}.items():
            self.assertAlmostEqual(values[horizon], expected)
        self.assertEqual(audit["forecastStepsByHorizon"], {"1": 2, "5": 6, "10": 11})
        self.assertEqual(audit["statsmodelsVersion"], "0.14.0")

    def test_nonconverged_fit_does_not_fallback(self):
        origin = dt.date(2025, 8, 1)
        begin = origin - dt.timedelta(days=pilot.WINDOW_DAYS)
        series = {begin + dt.timedelta(days=i): (100.0 + i, 0.0) for i in range(100)}

        class Model:
            def __init__(self, *_args, **_kwargs):
                pass

            def fit(self, **_kwargs):
                return type("Fit", (), {"mle_retvals": {"converged": False}})()

        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX", Model):
            values, audit = pilot._harmonic_forecasts(series, origin)
        self.assertEqual(values, {1: None, 5: None, 10: None})
        self.assertEqual(audit["reason"], "fit_not_converged")


if __name__ == "__main__":
    unittest.main()
