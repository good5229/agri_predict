import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agri_data import dashboard, harmonic_residual_pilot as harmonic
from agri_data import pepper_harmonic_confirm as confirm


class PepperHarmonicConfirmTests(unittest.TestCase):
    def _source(self):
        rows = []
        start = dt.date(2022, 1, 1)
        for i in range((confirm.END - start).days + 11):
            day = start + dt.timedelta(days=i)
            rows.append({"date": day.isoformat(), "actual": str(100 + i % 19)})
        return json.dumps({"schemaVersion": 1, "products": [
            {"id": p, "observations": rows if p == confirm.PEPPER else []} for p in dashboard.PRODUCTS
        ]}, sort_keys=True).encode()

    def test_frozen_other_period_grid_and_dependency(self):
        self.assertEqual(confirm.ORIGINS, 85)
        self.assertEqual(confirm.PLANNED, 255)
        self.assertEqual((confirm.START, confirm.END), (dt.date(2024, 8, 8), dt.date(2024, 10, 31)))
        self.assertEqual(confirm.HARMONIC_SHA256, hashlib.sha256(Path(harmonic.__file__).read_bytes()).hexdigest())

    def test_forecast_then_score_complete_synthetic_grid(self):
        raw = self._source()

        class Model:
            def predict(self, _x): return [0.0]

        audit = {"observations": 730, "residualPairs": 300, "phi": .8, "fitStatus": "fitted", "reason": None,
            "horizonReasons": {"1": None, "5": None, "10": None}}
        hvals = ({"candidate": {1: 101., 5: 105., 10: 110.}, "ablation": {1: 102., 5: 106., 10: 111.}}, audit)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "prices.json"; source.write_bytes(raw)
            with patch.object(confirm, "PRICE_SHA256", hashlib.sha256(raw).hexdigest()), \
                 patch.object(confirm.harmonic, "_harmonic_forecasts", return_value=hvals), \
                 patch.object(confirm, "_fit_direct", return_value=(Model(), 400)), \
                 patch.object(confirm.live_forecast, "_strict_feature_row", return_value=(dt.date(2024, 1, 1), [0.])):
                manifest = confirm.forecast(source, root / "forecast")
                self.assertEqual(len(manifest["rows"]), 255)
                self.assertFalse(manifest["targetValuesAttached"])
                self.assertEqual(manifest["dependencySha256"]["harmonic_residual_pilot.py"], confirm.HARMONIC_SHA256)
                report = confirm.score(source, root / "forecast/forecast-manifest.json", root / "report.json")
                frozen = json.loads((root / "forecast/forecast-manifest.json").read_text())
                tamper_cases = [
                    ("invalid_harmonic_phi", lambda a: a.update(phi=123.0)),
                    ("invalid_harmonic_audit_counts", lambda a: a.update(observations=0)),
                    ("available_harmonic_forecast_has_invalid_audit", lambda a: a.update(residualPairs=0)),
                    ("available_harmonic_forecast_has_invalid_audit", lambda a: a.update(fitStatus="error")),
                    ("available_harmonic_forecast_has_invalid_horizon_reason", lambda a: a["horizonReasons"].update({"1": "invalid_harmonic_forecast"})),
                ]
                tampered_path = root / "tampered-manifest.json"
                for expected_error, mutate in tamper_cases:
                    tampered = json.loads(json.dumps(frozen))
                    mutate(tampered["rows"][0]["harmonicAudit"])
                    tampered_path.write_text(json.dumps(tampered))
                    with self.assertRaisesRegex(ValueError, expected_error):
                        confirm.score(source, tampered_path, root / "should-not-exist.json")
            self.assertEqual(report["plannedRows"], 255)
            self.assertEqual([g["planned"] for g in report["coverage"]["groups"]], [85, 85, 85])
            self.assertEqual(len(report["dmTestsByHacCalendarDays"]["9"]), 18)
            self.assertEqual(len(report["dmTestsByHacCalendarDays"]["28"]), 18)
            self.assertIn("mape", report["equal3HorizonMetrics"]["candidate"])

    def test_manifest_and_source_hash_tampering_are_rejected(self):
        raw = self._source()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "prices.json"; source.write_bytes(raw)
            with patch.object(confirm, "PRICE_SHA256", hashlib.sha256(raw).hexdigest()):
                # A synthetically assembled but otherwise protocol-correct manifest tests
                # the frozen code/dependency/source checks before any target join.
                manifest = {"protocol": confirm.PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
                    "sourceSha256": hashlib.sha256(raw).hexdigest(), "dependencySha256": {"harmonic_residual_pilot.py": confirm.HARMONIC_SHA256},
                    "codeSha256": hashlib.sha256(Path(confirm.__file__).read_bytes()).hexdigest(), "period": [confirm.START.isoformat(), confirm.END.isoformat()],
                    "origins": confirm.ORIGINS, "products": [confirm.PEPPER], "horizons": list(confirm.HORIZONS),
                    "plannedRows": confirm.PLANNED, "candidateProtocol": confirm._protocol(), "rows": []}
                mf = root / "manifest.json"; mf.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "forecast_grid_incomplete"):
                    confirm.score(source, mf, root / "no-report.json")
                # Once rows are present, altered executable identity must be detected.
                manifest["rows"] = [{"productId": confirm.PEPPER, "origin": o.isoformat(),
                    "target": (o + dt.timedelta(days=h)).isoformat(), "horizon": h,
                    "candidate": None, "ablation": None, "direct": None, "persistence": None,
                    "harmonicAudit": {"horizonReasons": {str(x): "unavailable" for x in confirm.HORIZONS}}}
                    for o in confirm._dates() for h in confirm.HORIZONS]
                manifest["codeSha256"] = "0" * 64
                mf.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "code_hash_mismatch"):
                    confirm.score(source, mf, root / "no-report.json")
                manifest["codeSha256"] = hashlib.sha256(Path(confirm.__file__).read_bytes()).hexdigest()
                mf.write_text(json.dumps(manifest))
                changed = root / "changed.json"; changed.write_bytes(raw + b" ")
                with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
                    confirm.score(changed, mf, root / "no-report.json")


if __name__ == "__main__": unittest.main()
