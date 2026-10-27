import datetime as dt

import pytest

from agri_data import period_supply_scenario as m


def _fact(**kwargs):
    row = {
        "fact_id": "f1", "crop": "radish", "source_org": "MAFRA", "region": "National",
        "quantity_kind": "planned_release", "unit": "tonnes", "variety_or_crop_type": "",
        "asof_eligibility": "retrospective_publication_scenario", "evidence_status": "agent_visual_verified",
        "origin_review_status": "government_own_action_explicit", "series_admissibility": None,
        "target_mapping_status": "mapped", "publish_eligible_at_scenario": "2023-07-10T09:00:00+09:00",
        "effective_start": None, "effective_end": None, "value": "1200", "value_lower": None,
        "value_upper": None, "update_group_id": "upd1", "source_url": "https://example.test/source",
    }
    row.update(kwargs)
    return row


def _forecast(**kwargs):
    row = {"crop": "radish", "horizon": 2, "targetStart": "2023-07-11", "targetEnd": "2023-07-20",
           "origin": "2023-07-11", "inputCutoff": "2023-07-09", "base": 100.0, "p_ref": 95.0}
    row.update(kwargs)
    return row


def test_document_boundary_uses_origin_10_not_price_input_cutoff():
    forecast = _forecast(origin="2023-07-11", inputCutoff="2023-07-09")
    before = _fact(publish_eligible_at_scenario="2023-07-11T09:59:00+09:00")
    after = _fact(publish_eligible_at_scenario="2023-07-11T10:01:00+09:00")
    selected = m.select_facts([before, after], forecast)
    assert [m.fact_id(s["fact"]) for s in selected] == ["f1"]
    assert selected[0]["ageDays"] == pytest.approx(1 / 1440)


def test_date_only_publication_is_available_next_day_at_midnight():
    forecast = _forecast(origin="2023-07-11")
    fact = _fact(publish_eligible_at_scenario="2023-07-10")
    state = m.fact_match_state(fact, forecast)
    assert state is not None
    assert state["eligibleAt"] == "2023-07-11T00:00:00+09:00"
    assert state["ageDays"] == pytest.approx(10 / 24)


def test_unknown_period_expires_at_30_days_but_explicit_plan_lives_to_end():
    forecast = _forecast(origin="2023-08-10", targetStart="2023-08-11", targetEnd="2023-08-20")
    state = _fact(quantity_kind="harvest_progress", publish_eligible_at_scenario="2023-07-10T09:00:00+09:00")
    dated_plan = _fact(fact_id="f2", publish_eligible_at_scenario="2023-07-01T09:00:00+09:00",
                       effective_start="2023-07-01", effective_end="2023-08-10")
    assert m.fact_match_state(state, forecast) is None
    plan_state = m.fact_match_state(dated_plan, forecast)
    assert plan_state is not None and plan_state["validity"] == "dated_plan"
    assert plan_state["overlap"] == "known_nonoverlap"
    expired = _fact(fact_id="f3", publish_eligible_at_scenario="2023-07-01T09:00:00+09:00",
                    effective_start="2023-07-01", effective_end="2023-08-09")
    assert m.fact_match_state(expired, forecast) is None


@pytest.mark.parametrize("patch", [
    {"asof_eligibility": "ineligible"},
    {"asof_eligibility": "pending_source_review"},
    {"source_org": "KREI"},
    {"origin_review_status": "explicit_KREI_citation"},
    {"series_admissibility": "hold_denominator_change"},
    {"origin_review_status": "original_source_hold"},
    {"horizon_relevance": "historical_comparator_not_current_supply"},
])
def test_noneligible_and_held_facts_never_match(patch):
    assert m.fact_match_state(_fact(**patch), _forecast()) is None


def test_source_name_alias_does_not_merge_quantity_semantics_or_units():
    ministry = _fact(source_org="농림축산식품부")
    maff = _fact(source_org="MAFRA")
    other_kind = _fact(quantity_kind="planned_release_subset")
    other_unit = _fact(unit="tonnes/day")
    assert m.fact_group_key(ministry) == m.fact_group_key(maff)
    assert m.fact_group_key(ministry) != m.fact_group_key(other_kind)
    assert m.fact_group_key(ministry) != m.fact_group_key(other_unit)
    assert not m.QUANTITY_KIND_ALIASES


def test_negated_krei_provenance_note_does_not_exclude_jeju_fact():
    fact = _fact(source_org="제주특별자치도농업기술원", region="Jeju", provenance_origin="no KREI attribution on reviewed crop section")
    assert m.fact_is_eligible(fact)
    assert m.fact_match_state(fact, _forecast()) is not None


def test_v5_gb_status_and_unavailable_flag_preserve_only_valid_baselines():
    ok = {"predictions": {"G-B": {"status": "ok", "predicted": 123.4}}, "policyUnavailable": None}
    unavailable = {"predictions": {"G-B": {"status": None, "predicted": None}}, "policyUnavailable": True}
    assert m._v5_gb_prediction(ok) == pytest.approx(123.4)
    assert m._v5_gb_prediction(unavailable) is None


def test_same_time_conflicting_values_are_flagged_not_resolved():
    first = _fact(fact_id="f1", value="1200")
    second = _fact(fact_id="f2", value="1500")
    selected = m.select_facts([first, second], _forecast())
    assert len(selected) == 2
    assert all(row["tieConflict"] for row in selected)
    masks, numeric = m.feature_schema([first, second])
    d0, d1 = m.row_features(_forecast(), selected, masks, numeric)
    group = m._feature_key(m.fact_group_key(first))
    assert d0[f"supply::{group}::present"] == 1.0
    assert d0[f"supply::{group}::tie_conflict"] == 1.0
    assert all(d1[f"value::{group}::{bound}"] is None for bound in ("value", "lower", "upper"))


def test_typed_values_keep_bounds_and_provenance_separate():
    fact = _fact(value=None, value_lower="1000", value_upper="1400", unit="tonnes")
    selected = m.select_facts([fact], _forecast())
    masks, numeric = m.feature_schema([fact])
    d0, d1 = m.row_features(_forecast(), selected, masks, numeric)
    group = m._feature_key(m.fact_group_key(fact))
    assert d1[f"value::{group}::value"] is None
    assert d1[f"value::{group}::lower"] == 1000
    assert d1[f"value::{group}::upper"] == 1400
    assert -1.0 <= d0["calendar::dekad_sin"] <= 1.0
    row = {**_forecast(base=100.0, p_ref=95.0), "selectedFacts": selected}
    feature0 = m._features_from_row(row, masks, numeric, False)
    feature1 = m._features_from_row(row, masks, numeric, True)
    assert feature0 == {key: feature1[key] for key in feature0}
    assert feature1[f"value::{group}::lower"] == 1000


def _model_row(origin, target_end, row_key, update_group, value, actual, base=100.0):
    typed = {"crop": "radish", "source_org": "MAFRA", "region": "National", "quantity_kind": "planned_release",
             "unit": "tonnes", "variety_or_crop_type": ""}
    fact = _fact(**typed, update_group_id=update_group, value=None if value is None else str(value))
    group = m._feature_key(m.fact_group_key(fact))
    state = {"fact": fact, "numeric": {"value": value, "lower": None, "upper": None}, "tieConflict": False}
    common = ["price::x", f"missing::value::{group}::value"]
    d0 = {"price::x": float(value or 0.0), f"missing::value::{group}::value": float(value is None)}
    d1 = {**d0, f"value::{group}::value": value}
    return {"crop": "radish", "horizon": 2, "origin": origin, "targetStart": origin,
            "targetEnd": target_end, "rowKey": row_key, "base": base, "actual": actual,
            "selectedFacts": [state], "updateGroups": [update_group],
            "featureNames": {"d0": common, "d1": [*common, f"value::{group}::value"]},
            "featureRows": {"d0": d0, "d1": d1}}


def test_prequential_fit_maturity_pairing_constant_unseen_zero_effect_and_clip():
    current_origin = dt.date(2023, 7, 11)
    training = []
    for i in range(12):
        origin = dt.date(2023, 1, 1) + dt.timedelta(days=9 * i)
        training.append(_model_row(origin.isoformat(), origin.isoformat(), f"old-{i}", f"group-{i % 3}", 10.0, 1000.0))
    # Exactly mature at current origin: target end + 2 days equals fit origin.
    training.append(_model_row("2023-07-01", "2023-07-09", "boundary-mature", "group-1", 10.0, 1000.0))
    # One day beyond maturity is excluded, despite an earlier origin.
    training.append(_model_row("2023-07-02", "2023-07-10", "boundary-immature", "group-2", 10.0, 1000.0))
    row = _model_row("2023-07-11", "2023-07-20", "current", "group-1", 999.0, None)
    group = m._feature_key(m.fact_group_key(row["selectedFacts"][0]["fact"]))
    counters = {"fits": 0}
    out = m._forecast_one(row, [*training, row], [], [f"value::{group}::value"], counters)
    assert "boundary-mature" in out["trainingRowKeys"]
    assert "boundary-immature" not in out["trainingRowKeys"]
    assert out["d0TrainingRowKeys"] == out["d1TrainingRowKeys"]
    assert out["trainingOrigins"] == 13
    assert out["trainingUpdateGroups"] == 3
    assert out["trainingNonconstantNumericGroups"] == 0
    assert out["d0"] == pytest.approx(out["d1"])
    assert out["d1NumericNeutral"] == pytest.approx(out["d0"])
    assert out["d1NumericValueContributionLog"] == pytest.approx(0.0)
    assert out["d1NuisanceRefitDeltaLog"] == pytest.approx(0.0)
    assert out["d0CorrectionLog"] == pytest.approx(m.CORRECTION_CAP)
    assert out["d1CorrectionLog"] == pytest.approx(m.CORRECTION_CAP)
    assert out["d0"] == pytest.approx(125.0)
    assert counters["fits"] == 2


def test_no_numeric_current_fact_and_missing_baseline_keep_exact_base():
    row = _model_row("2023-07-11", "2023-07-20", "current", "group-1", None, None)
    out = m._forecast_one(row, [row], [], ["unused"], {"fits": 0})
    assert out["d0"] == out["d1"] == out["base"]
    assert out["fallbackReason"] == "matched_facts_without_numeric_values"
    row["base"] = None
    out = m._forecast_one(row, [row], [], ["unused"], {"fits": 0})
    assert out["d0"] is None and out["d1"] is None
    assert out["fallbackReason"] == "baseline_unavailable"


def test_numeric_support_is_column_specific_for_value_and_bounds():
    training = []
    for i in range(12):
        row = _model_row((dt.date(2023, 1, 1) + dt.timedelta(days=9 * i)).isoformat(),
                         (dt.date(2023, 1, 1) + dt.timedelta(days=9 * i)).isoformat(),
                         f"old-{i}", f"group-{i % 3}", 10.0, 100.0)
        group = m._feature_key(m.fact_group_key(row["selectedFacts"][0]["fact"]))
        lower = float(i + 1)
        row["selectedFacts"][0]["numeric"]["lower"] = lower
        row["featureRows"]["d1"][f"value::{group}::lower"] = lower
        row["featureNames"]["d1"].append(f"value::{group}::lower")
        training.append(row)
    current = _model_row("2023-07-11", "2023-07-20", "current", "group-1", 999.0, None)
    group = m._feature_key(m.fact_group_key(current["selectedFacts"][0]["fact"]))
    current["selectedFacts"][0]["numeric"]["lower"] = 25.0
    current["featureRows"]["d1"][f"value::{group}::lower"] = 25.0
    current["featureNames"]["d1"].append(f"value::{group}::lower")
    out = m._forecast_one(current, [*training, current], [], [f"value::{group}::value", f"value::{group}::lower", f"value::{group}::upper"], {"fits": 0})
    assert out["supportedCurrentNumericColumns"] == [f"value::{group}::lower"]
    assert out["unsupportedCurrentNumericColumns"] == [f"value::{group}::value"]
    assert out["trainingNonconstantNumericColumns"] == [f"value::{group}::lower"]
