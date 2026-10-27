import datetime as dt
import hashlib
import json

import pytest

from agri_data import e106_mean_intervals as intervals
from agri_data import preserved_improvement_composition as composition


def test_exact_phase_grid_sizes_and_cutoffs():
    assert len(intervals._expected_grid(intervals.PHASES["confirmation"])) == 3020
    assert len(intervals._expected_grid(intervals.PHASES["final"])) == 3160
    confirmation = intervals.PHASES["confirmation"]
    assert (confirmation["start"] - dt.timedelta(days=1), confirmation["start"], 1) not in intervals._expected_grid(confirmation)


def test_manifest_grid_validation_requires_every_key_and_same_origin():
    phase = {"phaseName": "confirmation", "start": dt.date(2026, 4, 1), "end": dt.date(2026, 4, 1),
             "originFloor": dt.date(2026, 4, 1)}
    # A one-day exact grid has 10 keys per product after the origin cutoff.
    phase["start"] = dt.date(2026, 4, 11)
    phase["end"] = dt.date(2026, 4, 11)
    rows = []
    target = dt.date(2026, 4, 11)
    for product in intervals.PRODUCTS:
        for h in range(1, 11):
            rows.append({"product": product, "target": target.isoformat(), "origin": (target-dt.timedelta(days=h)).isoformat(), "horizon": h,
                         "eligible": True, **{m: {"status": "issued", "prediction": 10.0} for m in ("candidate", "E", "causal50")}})
    manifest = {"phase": "preserved_improvement_confirmation_forecast_only", "targetValuesAttached": False,
                "forecasts": rows}
    indexed = intervals._index_forecasts(manifest, "confirmation", phase)
    assert len(indexed) == 40
    with pytest.raises(ValueError, match="exact phase grid"):
        intervals._index_forecasts({**manifest, "forecasts": rows[:-1]}, "confirmation", phase)
    bad = [dict(r) for r in rows]
    bad[0] = {**bad[0], "origin": "2026-04-02"}
    with pytest.raises(ValueError, match="eligibility/origin"):
        intervals._index_forecasts({**manifest, "forecasts": bad}, "confirmation", phase)


def test_bootstrap_reuses_joint_circular_calendar_helper(monkeypatch):
    calls = []

    def stub(rows, **kwargs):
        calls.append((rows, kwargs))
        return {"valid": 3, "invalid": kwargs["replicates"] - 3, "interval95": [-2, 1]}

    monkeypatch.setattr(intervals.bootstrap_protocol, "circular_block_bootstrap", stub)
    rows = [{"product": "garlic", "horizon": 1, "target": "2026-04-01", "actual": 10, "a": 9, "b": 11}]
    result = intervals.bootstrap_estimate(rows, metric="ae", block=14,
        cells=frozenset({("garlic", 1)}), replicates=7,
        calendar_start=dt.date(2026, 4, 1), calendar_end=dt.date(2026, 6, 20))
    assert result["valid"] == 3 and result["invalid"] == 4
    assert calls[0][1]["seed"] == intervals.SEED
    assert calls[0][1]["block"] == 14
    assert calls[0][1]["calendar_start"] == dt.date(2026, 4, 1)


def test_bootstrap_refuses_incomplete_requested_cell_family():
    row = {"product": "garlic", "horizon": 1, "target": "2026-04-01", "actual": 10, "a": 9, "b": 11}
    with pytest.raises(ValueError, match="cover the requested"):
        intervals.bootstrap_estimate([row], metric="ae", block=14,
            cells=frozenset({("garlic", 1), ("garlic", 2)}), replicates=5,
            calendar_start=dt.date(2026, 4, 1), calendar_end=dt.date(2026, 4, 3))


def test_real_bootstrap_counts_invalid_replicates_and_is_deterministic():
    rows = [
        {"product": "garlic", "horizon": 1, "target": "2026-04-01", "actual": 10, "a": 8, "b": 9},
        {"product": "garlic", "horizon": 2, "target": "2026-04-03", "actual": 10, "a": 11, "b": 10},
    ]
    args = dict(metric="ae", block=1, cells=frozenset({("garlic", 1), ("garlic", 2)}), replicates=100,
                calendar_start=dt.date(2026, 4, 1), calendar_end=dt.date(2026, 4, 3))
    first = intervals.bootstrap_estimate(rows, **args)
    second = intervals.bootstrap_estimate(rows, **args)
    assert first == second
    assert first["valid"] + first["invalid"] == 100
    assert first["invalid"] > 0
    assert first["interval95"] is not None


@pytest.mark.parametrize("metric", ["ae", "ape"])
def test_same_sample_e106_minus_e_four_crop_point_is_quarter_of_garlic(metric):
    rows = []
    for product, horizon in sorted(intervals.ALL_CELLS):
        for day, actual, candidate, baseline in (("2026-04-01", 10, 8 if product == "garlic" else 10, 10),
                                                  ("2026-04-02", 20, 18 if product == "garlic" else 20, 20)):
            rows.append({"product": product, "horizon": horizon, "target": day, "actual": actual,
                         "a": candidate, "b": baseline})
    garlic = [r for r in rows if r["product"] == "garlic"]
    g = intervals.bootstrap_protocol._cell_means(garlic, "a", "b", metric, expected_cells=intervals.GARLIC_CELLS)
    full = intervals.bootstrap_protocol._cell_means(rows, "a", "b", metric, expected_cells=intervals.ALL_CELLS)
    assert full["difference"] == pytest.approx(g["difference"] / 4)


def test_canonical_digest_validation_detects_changes():
    doc = {"phase": "x", "payload": [1, 2]}
    unsigned_digest = hashlib.sha256(intervals.canonical(doc).rstrip(b"\n")).hexdigest()
    doc["manifestSha256"] = unsigned_digest
    intervals._verify_digest(doc, "manifestSha256", unsigned_digest, "test")
    doc["payload"] = [1, 3]
    with pytest.raises(ValueError, match="internal"):
        intervals._verify_digest(doc, "manifestSha256", unsigned_digest, "test")


def test_interval_block_protocol_is_frozen():
    assert intervals.BLOCKS == (14, 28)
    assert intervals.REPLICATES == 2000
    assert intervals.SEED == 20261011
    assert set(intervals.COMPARISONS) == {"E106_minus_E", "E106_minus_causal50"}


def _synthetic_report_fixture():
    target = dt.date(2026, 4, 11)
    phase = {"phaseName": "confirmation", "start": target, "end": target,
             "originFloor": dt.date(2026, 4, 1)}
    rows, actuals = [], {}
    for product in intervals.PRODUCTS:
        for horizon in intervals.HORIZONS:
            day = target.isoformat()
            actual = 100.0 + horizon
            causal = actual + 7.0
            e = causal if product == "garlic" else actual + 5.0
            candidate = actual + (3.0 if product == "garlic" else 5.0)
            row = {"product": product, "target": day, "origin": (target-dt.timedelta(days=horizon)).isoformat(),
                   "horizon": horizon, "eligible": True,
                   "candidate": {"status": "issued", "prediction": candidate},
                   "E": {"status": "issued", "prediction": e},
                   "causal50": {"status": "issued", "prediction": causal}}
            rows.append(row)
            actuals[(product, target)] = actual
    indexed = {(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]): r for r in rows}
    scored = composition._score_rows(rows, actuals, "confirmation")
    report = {"phase": "confirmation", "manifestSha256": "manifest-digest",
              "targetRange": [target.isoformat(), target.isoformat()], **scored}
    manifest = {"manifestSha256": "manifest-digest"}
    return manifest, report, indexed, actuals, phase


def test_saved_report_verifier_matches_original_scorer_end_to_end():
    manifest, report, indexed, actuals, phase = _synthetic_report_fixture()
    common, summary = intervals._verify_scored_report(manifest, report, indexed, actuals, "confirmation", phase)
    assert len(common) == 40
    assert len(summary["commonByCell"]) == 40


def test_saved_report_verifier_rejects_changed_paired_cell_delta():
    manifest, report, indexed, actuals, phase = _synthetic_report_fixture()
    changed = dict(report)
    changed["metricsByProductHorizonMethod"] = [dict(x) for x in report["metricsByProductHorizonMethod"]]
    row = next(x for x in changed["metricsByProductHorizonMethod"] if x.get("comparison") == "candidateMinusE")
    row["maeDelta"] += 1
    with pytest.raises(ValueError, match="saved metric mismatch"):
        intervals._verify_scored_report(manifest, changed, indexed, actuals, "confirmation", phase)
