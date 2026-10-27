from __future__ import annotations

import datetime as dt
import numpy as np

from agri_data import chronos2_market_context as c2


def test_calendar_context_keeps_trailing_missing_days_and_cutoff():
    origin = dt.date(2026, 4, 5)
    cutoff = origin - dt.timedelta(days=1)
    observed = cutoff - dt.timedelta(days=3)
    series = {observed: (42.0, 100.0)}
    markets = {market: {observed: {"price": 40.0 + i, "weight": 10.0}}
               for i, market in enumerate(c2.MARKETS)}

    matrix, support = c2._build_context(series, markets, origin)

    assert matrix.shape == (3, 730)
    assert support["cutoff"] == cutoff.isoformat()
    assert support["firstDate"] == (cutoff - dt.timedelta(days=729)).isoformat()
    assert support["lastDate"] == cutoff.isoformat()
    assert matrix[0, -4] == np.float32(42.0)
    assert np.isnan(matrix[:, -3:]).all()
    assert support["observedCounts"]["representative"] == 1
    assert support["observedCounts"][c2.MARKETS[0]] == 1


def test_u_and_m_share_identical_representative_channel():
    origin = dt.date(2026, 4, 5)
    day = origin - dt.timedelta(days=1)
    matrix, _ = c2._build_context({day: (50.0, 100.0)},
                                  {m: {day: {"price": 45.0, "weight": 20.0}} for m in c2.MARKETS}, origin)
    u, m = matrix[:1], matrix
    assert u.shape == (1, 730)
    assert m.shape == (3, 730)
    assert np.array_equal(u[0], m[0], equal_nan=True)


def test_phase_rows_freezes_c_issued_keys_and_filters_origin_window():
    rows = []
    for origin, target, status in [
        ("2026-03-31", "2026-04-01", "issued"),
        ("2026-04-01", "2026-04-02", "issued"),
        ("2026-04-01", "2026-04-03", "not_issued"),
        ("2026-04-01", "2026-04-04", "fallback"),
    ]:
        rows.append({"product": "cabbage", "origin": origin, "target": target,
                     "horizon": (dt.date.fromisoformat(target) - dt.date.fromisoformat(origin)).days,
                     "conditions": {"C": {"status": status, "blendPrediction": None if status == "not_issued" else 12.5,
                                           "fallbackReason": "preserved-control"}}})
    got = c2._phase_rows("V", {"phase": "V", "forecasts": rows})
    assert len(got) == 2
    fallback = got[1]
    assert fallback["origin"] == dt.date(2026, 4, 1)
    assert fallback["B"] == 12.5
    assert fallback["baselineSourceStatus"] == "fallback"
    assert fallback["baselineFallbackReason"] == "preserved-control"


def test_forecast_document_uses_horizon_index_and_no_h_fallback(tmp_path):
    protocol = c2._protocol()
    protocol["protocolSha256"] = c2._sha(c2._canonical(protocol))
    (tmp_path / "config.yaml").write_bytes(c2._canonical(protocol))
    origin = dt.date(2026, 4, 1)
    rows = [{"product": "cabbage", "origin": origin, "target": origin + dt.timedelta(days=h),
             "horizon": h, "B": 10.0, "baselineFallbackReason": None} for h in (1, 2)]
    contexts = {("cabbage", origin): {"cutoff": "2026-03-31", "observedCounts": {"representative": 5}}}
    results = {}
    for condition, shift in (("U", 0.0), ("M", 1.0)):
        results[("cabbage", origin, condition)] = {"representativePredictions": [100 + shift + i for i in range(11)],
                                                   "status": "complete", "failure": None}
    results[("cabbage", origin, "M")]["representativePredictions"][2] = None
    doc = c2._forecast_document("V", rows, results, contexts, {"inputSha256": "a" * 64}, tmp_path)
    assert doc["forecasts"][0]["conditions"]["U"]["prediction"] == 101.0
    assert doc["forecasts"][0]["conditions"]["M"]["prediction"] == 102.0
    assert doc["forecasts"][0]["conditions"]["H"]["prediction"] == 56.0
    assert doc["forecasts"][1]["conditions"]["M"]["status"] == "invalid_prediction"
    assert doc["forecasts"][1]["conditions"]["H"]["status"] == "not_issued"
    assert doc["forecasts"][1]["conditions"]["H"]["prediction"] is None

    results[("cabbage", origin, "U")]["status"] = "failed"
    results[("cabbage", origin, "U")]["failure"] = {"type": "RuntimeError", "message": "U failed"}
    failed_doc = c2._forecast_document("V", rows[:1], results, contexts,
                                       {"inputSha256": "a" * 64}, tmp_path)
    assert failed_doc["forecasts"][0]["conditions"]["U"]["status"] == "inference_error"
    assert failed_doc["forecasts"][0]["conditions"]["U"]["reason"] == "RuntimeError: U failed"
    assert failed_doc["forecasts"][0]["conditions"]["H"]["status"] == "issued"


def test_protocol_binds_fixed_checkpoint_and_revisions():
    protocol = c2._protocol()
    assert protocol["model"]["revision"] == c2.MODEL_REVISION
    assert protocol["model"]["weightSha256"] == c2.MODEL_WEIGHT_SHA256
    assert protocol["input"]["contextCalendarDays"] == 730
    assert protocol["prediction"]["evaluatedIndices"]["10"] == 10
    assert protocol["prediction"]["calls"] == "one model task per product/origin/condition"
    assert protocol["eligibility"] == "eligibility_unverified"
    assert "deterministicAlgorithmsEnabled" in protocol["runtime"]


def test_provenance_normalization_preserves_heterogeneous_key_types_and_order():
    first = {None: {"state": "missing-key"}, "None": {"state": "literal-key"},
             ("product", None): [1, "x"]}
    second = {("product", None): [1, "x"], "None": {"state": "literal-key"},
              None: {"state": "missing-key"}}

    normalized = c2._typed_provenance(first)
    assert c2._canonical(normalized) == c2._canonical(c2._typed_provenance(second))
    records = normalized["__mappingEntries__"]
    keys = [entry["key"] for entry in records]
    assert {item["type"] for item in keys} == {"null", "str", "tuple"}
    assert c2._sha(c2._canonical(normalized)) != c2._sha(c2._canonical(c2._typed_provenance(
        {"None": {"state": "missing-key"}, None: {"state": "literal-key"},
         ("product", None): [1, "x"]})))


def test_primary_gate_uses_scorer_validation_gate_shape():
    assert c2._primary_gate({"validationGate": {"passed": True}}) is True
    assert c2._primary_gate({"validationGate": {"passed": False}}) is False
    assert c2._primary_gate({"primaryDecision": True}) is False


def test_inference_failure_is_checkpointed_per_condition_without_fallback(tmp_path):
    class BrokenPipeline:
        def predict_quantiles(self, *_args, **_kwargs):
            raise RuntimeError("synthetic task failure")

    origin = dt.date(2026, 4, 1)
    matrix = np.ones((3, c2.CONTEXT_DAYS), dtype=np.float32)
    u = c2._task_checkpoint(tmp_path / "u.json", "cabbage", origin, "U", matrix[:1], BrokenPipeline())
    m = c2._task_checkpoint(tmp_path / "m.json", "cabbage", origin, "M", matrix, BrokenPipeline())

    assert u["status"] == m["status"] == "failed"
    assert u["failure"]["message"] == "synthetic task failure"
    assert u["representativePredictions"] == [None] * c2.PREDICTION_LENGTH
    assert m["representativePredictions"] == [None] * c2.PREDICTION_LENGTH
    assert u["processMaxRssBytesAfterTask"] > 0
    assert c2._task_checkpoint(tmp_path / "u.json", "cabbage", origin, "U", matrix[:1], None)["status"] == "failed"
