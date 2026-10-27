import datetime as dt
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agri_data import dashboard
from agri_data import harmonic_residual_pilot as pilot


class HarmonicResidualPilotTests(unittest.TestCase):
    def test_frozen_grid_and_fourier_epoch(self):
        dates = list(pilot._dates())
        self.assertEqual((len(dates), dates[0], dates[-1]), (55, dt.date(2025, 8, 1), dt.date(2025, 9, 24)))
        self.assertEqual(pilot.PLANNED, 825)
        self.assertEqual(pilot.FAMILY, 90)
        self.assertEqual(pilot._fourier_row(pilot.EPOCH), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        row = pilot._fourier_row(pilot.EPOCH + dt.timedelta(days=1))
        expected_first = math.sin(2 * math.pi / 365.25)
        self.assertAlmostEqual(row[0], expected_first)

    def test_phi_selects_calendar_gap_dynamics_and_ties_to_smaller(self):
        residuals = [(dt.date(2025, 1, 1) + dt.timedelta(days=i), 2 * (0.8 ** i)) for i in range(40)]
        phi, n, _ = pilot._choose_phi(residuals)
        self.assertEqual((phi, n), (0.8, 39))
        same, n, _ = pilot._choose_phi([(d, 1.0) for d, _ in residuals])
        self.assertEqual((same, n), (0.99, 39))

    def test_harmonic_fit_uses_only_prior_observations_and_calendar_target(self):
        origin = dt.date(2025, 8, 1)
        series = {origin - dt.timedelta(days=i): (100 + i, 0) for i in range(1, 731)}
        series[origin] = (999999, 0)
        series[origin + dt.timedelta(days=1)] = (888888, 0)
        captured = {}

        class Ridge:
            def __init__(self, **kwargs): captured["kwargs"] = kwargs
            def fit(self, X, y): captured["X"], captured["y"] = X.copy(), y.copy(); return self
            def predict(self, X):
                captured.setdefault("predX", []).append(X.copy())
                return [float(sum(row) / 100) for row in X]

        with patch("sklearn.linear_model.Ridge", Ridge), patch.object(pilot, "_choose_phi", return_value=(0.5, 40, .1)):
            values, audit = pilot._harmonic_forecasts(series, origin)
        self.assertEqual(audit["observations"], 730)
        self.assertEqual(captured["kwargs"], {"alpha": 30.0, "fit_intercept": True})
        self.assertNotIn(math.log(999999), captured["y"])
        self.assertNotIn(math.log(888888), captured["y"])
        self.assertEqual(captured["X"].shape, (730, 6))
        self.assertEqual(len(captured["predX"]), 4)  # training predictions plus three target dates
        self.assertEqual(values["candidate"].keys(), {1, 5, 10})

    def test_underflow_makes_candidate_and_ablation_jointly_unavailable(self):
        origin = dt.date(2025, 8, 1)
        series = {origin - dt.timedelta(days=i): (100 + i, 0) for i in range(1, 731)}
        class Ridge:
            def __init__(self, **_kwargs): pass
            def fit(self, _X, y): self.y = y.copy(); return self
            def predict(self, X): return self.y + 1 if len(X) > 1 else [-745.0]
        with patch("sklearn.linear_model.Ridge", Ridge), patch.object(pilot, "_choose_phi", return_value=(.8, 40, .1)):
            values, _audit = pilot._harmonic_forecasts(series, origin)
        self.assertIsNone(values["candidate"][1])
        self.assertIsNone(values["ablation"][1])

    def test_post_fit_forecast_exception_keeps_shared_phi_and_marks_horizons(self):
        origin = dt.date(2025, 8, 1)
        series = {origin - dt.timedelta(days=i): (100 + i, 0) for i in range(1, 731)}
        class Ridge:
            def __init__(self, **_kwargs): pass
            def fit(self, _X, y): self.y = y.copy(); return self
            def predict(self, X):
                if len(X) > 1: return self.y
                raise RuntimeError("forecast failed")
        with patch("sklearn.linear_model.Ridge", Ridge), patch.object(pilot, "_choose_phi", return_value=(.8, 40, .1)):
            values, audit = pilot._harmonic_forecasts(series, origin)
        self.assertEqual(audit["fitStatus"], "fitted")
        self.assertEqual(audit["phi"], .8)
        self.assertEqual(set(audit["horizonReasons"].values()), {"invalid_harmonic_forecast"})
        self.assertTrue(all(values[m][h] is None for m in values for h in pilot.HORIZONS))

    def test_forecast_manifest_then_score_metrics_and_protocol_validation(self):
        observations = []
        start = dt.date(2023, 8, 1)
        for i in range(1300):
            day = start + dt.timedelta(days=i)
            observations.append({"date": day.isoformat(), "actual": str(100 + i % 17)})
        doc = {"schemaVersion": 1, "products": [{"id": p, "observations": observations} for p in dashboard.PRODUCTS]}
        raw = json.dumps(doc).encode()

        class Model:
            def predict(self, _x): return [0.0]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td); price = root / "prices.json"; price.write_bytes(raw)
            with patch.object(pilot, "PRICE_SHA256", hashlib.sha256(raw).hexdigest()), \
                 patch.object(pilot, "_harmonic_forecasts", return_value=({"candidate": {1: None, 5: None, 10: 5e-324}, "ablation": {1: None, 5: None, 10: 5e-324}}, {"phi": .8, "observations": 730, "residualPairs": 40, "fitStatus": "fitted", "reason": None, "horizonReasons": {"1": "invalid_harmonic_forecast", "5": "invalid_harmonic_forecast", "10": None}})), \
                 patch("agri_data.harmonic_residual_pilot._fit_direct", return_value=(Model(), 100)), \
                 patch("agri_data.harmonic_residual_pilot.live_forecast._strict_feature_row", return_value=(dt.date(2025, 7, 31), [0.])):
                manifest = pilot.forecast(price, root / "forecast")
                self.assertFalse(manifest["targetValuesAttached"])
                self.assertEqual(len(manifest["rows"]), 825)
                self.assertIsNone(manifest["rows"][0]["candidate"])
                self.assertIsNone(manifest["rows"][0]["ablation"])
                self.assertEqual(manifest["rows"][2]["candidate"], 5e-324)
                report = pilot.score(price, root / "forecast/forecast-manifest.json", root / "report.json")
            self.assertEqual(report["plannedRows"], 825)
            self.assertEqual(len(report["dmTestsByHacCalendarDays"]["9"]), 90)
            self.assertIn("ablation", report["pairedMetrics"])
            self.assertIn("candidateGainAtLeast0_5ppVsAblation", report["frozenExpansionGate"])
            tampered = json.loads((root / "forecast/forecast-manifest.json").read_text())
            tampered["rows"][0]["harmonicAudit"].update(phi=999, observations=0, residualPairs=0, fitStatus="error")
            tampered["rows"][0]["ablation"] = None
            bad_manifest = root / "tampered.json"
            bad_manifest.write_text(json.dumps(tampered))
            with patch.object(pilot, "PRICE_SHA256", hashlib.sha256(raw).hexdigest()):
                with self.assertRaisesRegex(ValueError, "invalid_harmonic_phi"):
                    pilot.score(price, bad_manifest, root / "must-not-exist.json")
                tampered = json.loads((root / "forecast/forecast-manifest.json").read_text())
                tampered["rows"][0]["harmonicAudit"]["horizonReasons"]["1"] = None
                bad_manifest.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError, "invalid_horizon_reason"):
                    pilot.score(price, bad_manifest, root / "must-not-exist.json")


if __name__ == "__main__": unittest.main()
