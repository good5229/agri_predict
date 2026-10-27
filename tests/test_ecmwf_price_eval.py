import datetime as dt
import hashlib
import json

import numpy as np
import pytest

from agri_data import ecmwf_field_collection as collection
from agri_data import ecmwf_price_eval as evaluation


def feature_fixture():
    origin_days = evaluation._origin_dates()
    feature_doc = {"schema_version": 1, "rows": []}
    for day in origin_days:
        feature_doc["rows"].append({"origin_date": day, "candidate_valid": True,
            "features": {name: float(index) for index, name in enumerate(evaluation.FEATURE_ORDER)}})
    run_dates = {(dt.date.fromisoformat(day) - dt.timedelta(days=1)).strftime("%Y%m%d")
                 for day in origin_days}
    manifest = {"source_scope_hash": collection.SCOPE_HASH,
                "feature_order": list(evaluation.FEATURE_ORDER), "feature_count": 18,
                "row_count": 922, "source_checkpoint_hashes": {
                    day: hashlib.sha256(day.encode()).hexdigest() for day in run_dates}}
    return feature_doc, manifest


def test_full_feature_snapshot_requires_all_checkpoints_and_frozen_order():
    feature_doc, manifest = feature_fixture()
    vectors = evaluation.validate_feature_inputs(feature_doc, manifest)
    assert len(vectors) == 922
    assert vectors["2024-03-16"] == [float(i) for i in range(18)]
    missing = dict(manifest, source_checkpoint_hashes=dict(list(manifest["source_checkpoint_hashes"].items())[:-1]))
    with pytest.raises(evaluation.EvaluationError, match="922 source checkpoints"):
        evaluation.validate_feature_inputs(feature_doc, missing)
    wrong_order = dict(manifest, feature_order=list(reversed(evaluation.FEATURE_ORDER)))
    with pytest.raises(evaluation.EvaluationError, match="feature order"):
        evaluation.validate_feature_inputs(feature_doc, wrong_order)


def test_feature_row_missing_or_nonfinite_becomes_unavailable_not_imputed():
    feature_doc, manifest = feature_fixture()
    row = feature_doc["rows"][0]
    row["candidate_valid"] = False
    row["features"][evaluation.FEATURE_ORDER[4]] = None
    vector = evaluation.validate_feature_inputs(feature_doc, manifest)["2024-03-16"]
    assert vector is None


def test_json_sorted_feature_keys_are_reordered_to_frozen_manifest_order():
    feature_doc, manifest = feature_fixture()
    # The point-feature writer serializes JSON with sort_keys=True, so object
    # iteration order is lexical even though its manifest has semantic order.
    round_tripped = json.loads(json.dumps(feature_doc, sort_keys=True))
    vectors = evaluation.validate_feature_inputs(round_tripped, manifest)
    assert vectors["2024-03-16"] == [float(i) for i in range(18)]


def test_training_window_excludes_eval_labels_and_model_fits_training_only():
    direct = {}
    vectors = {}
    start = evaluation.TRAIN_START
    for offset in range(100):
        target = start + dt.timedelta(days=offset)
        origin = target - dt.timedelta(days=1)
        row = {"product": evaluation.PRODUCTS[0], "horizon": 1, "target": target,
               "origin": origin, "actual": 110.0 + offset, "direct": 100.0}
        direct[(row["product"], 1, target)] = row
        vectors[origin.isoformat()] = [float(offset)] + [0.0] * 17
    future_target = evaluation.TRAIN_END + dt.timedelta(days=1)
    future = {"product": evaluation.PRODUCTS[0], "horizon": 1, "target": future_target,
              "origin": future_target - dt.timedelta(days=1), "actual": 1e99, "direct": 1.0}
    direct[(future["product"], 1, future_target)] = future
    train = evaluation._training_rows(direct, evaluation.PRODUCTS[0], 1)
    assert len(train) == 100
    assert all(row["target"] <= evaluation.TRAIN_END for row in train)
    audit, fitted = evaluation._fit_model(train, vectors)
    assert audit["status"] == "fitted"
    assert audit["distinctTrainingTargetDays"] == 100
    assert future_target.isoformat() not in audit["trainingTargetDates"]
    assert len(audit["pairedTrainingInputSha256"]) == 64
    assert len(audit["eligibleFitSampleSha256"]) == 64
    assert fitted[0].mean_.shape == (18,)


def test_ridge_uses_intercept_train_only_scale_and_zero_variance_scale_one():
    train = []
    vectors = {}
    for offset in range(95):
        target = evaluation.TRAIN_START + dt.timedelta(days=offset)
        origin = target - dt.timedelta(days=1)
        train.append({"target": target, "origin": origin, "actual": 120.0, "direct": 100.0})
        vectors[origin.isoformat()] = [1.0] * 18
    audit, fitted = evaluation._fit_model(train, vectors)
    assert fitted is not None
    assert audit["scalerScale"] == [1.0] * 18
    assert len(audit["ridgeCoefficients"]) == 18
    assert audit["ridgeIntercept"] == pytest.approx(np.log(1.2))


class IdentityScaler:
    def transform(self, values):
        return values


class ConstantCorrection:
    def __init__(self, value):
        self.value = value

    def predict(self, _):
        return np.asarray([self.value])


def test_fallback_for_missing_model_feature_and_maturity_gap_and_clip():
    fitted = (IdentityScaler(), ConstantCorrection(3.0))
    origin = evaluation.TRAIN_END + dt.timedelta(days=24)
    pred, fallback, correction = evaluation._predict(100.0, origin, fitted, [0.0] * 18)
    assert fallback is None
    assert correction == 3.0
    assert pred == pytest.approx(100 * np.exp(0.20))
    pred, fallback, _ = evaluation._predict(100.0, origin, fitted, None)
    assert pred == 100.0 and fallback == "feature_vector_missing_or_nonfinite"
    pred, fallback, _ = evaluation._predict(100.0, origin, None, [0.0] * 18)
    assert pred == 100.0 and fallback == "model_unavailable"
    pred, fallback, _ = evaluation._predict(100.0, origin - dt.timedelta(days=1), fitted, [0.0] * 18)
    assert pred == 100.0 and fallback == "training_label_maturity_gap"


def test_nonfinite_model_output_falls_back_with_json_safe_null_correction():
    origin = evaluation.TRAIN_END + dt.timedelta(days=24)
    fitted = (IdentityScaler(), ConstantCorrection(float("nan")))
    prediction, reason, correction = evaluation._predict(100.0, origin, fitted, [0.0] * 18)
    assert prediction == 100.0
    assert reason == "nonfinite_model_prediction"
    assert correction is None
    json.dumps({"candidate": prediction, "candidateCorrectionLog": correction}, allow_nan=False)


def test_dm_uses_calendar_gaps_and_unavailable_tests_remain_in_fixed_holm_family():
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    contiguous = {day: (-1.0 + i * 0.03) for i, day in enumerate(dates)}
    sparse = {day + dt.timedelta(days=i): value for i, (day, value) in enumerate(contiguous.items())}
    assert evaluation._dm_calendar(contiguous, 9)["statistic"] != evaluation._dm_calendar(sparse, 9)["statistic"]
    unavailable = evaluation._dm_calendar({day: 0.0 for day in dates[:29]}, 9)
    assert unavailable["status"] == "insufficient_pairs"
    assert unavailable["pValue"] == 1.0
    family = [dict(unavailable) for _ in range(evaluation.FAMILY_SIZE)]
    evaluation._holm(family)
    assert len(family) == 200
    assert all(row["holmPValue"] == 1.0 and not row["holmSignificant"] for row in family)


def test_dry_run_does_not_read_or_fit_inputs(capsys, monkeypatch):
    monkeypatch.setattr(evaluation, "build", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    assert evaluation.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["inputs_read"] is False
    assert result["model_fit"] is False
    assert result["expected_eval_pairs"] == 12_508
