from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from agri_data import direct_price_path as model_code
from agri_data import direct_price_path_data as data


def test_training_labels_are_strictly_before_the_actual_fit_boundary(monkeypatch):
    boundary = dt.date(2026, 3, 1)

    class GuardedPrices(dict):
        def get(self, key, default=None):
            if isinstance(key, dt.date) and key >= boundary:
                raise AssertionError(f"fit tried to inspect a label at/after {boundary}")
            return super().get(key, default)

    representative = GuardedPrices({dt.date(2026, 1, 1) + dt.timedelta(days=i): 100.0 + i
                                     for i in range((boundary - dt.date(2026, 1, 1)).days + 1)})
    monkeypatch.setattr(data, "build_input_window",
                        lambda product, origin, context, input_type, representative, channels, details:
                        {"p0": representative[max(d for d in representative if d < origin)],
                         "representativeSigma": 0.1, "activeAuxCount": 0})

    records = model_code._records_for_period("radish", representative, {}, 84, "U", [],
                                              dt.date(2026, 2, 1), boundary,
                                              label_before=boundary)
    observed_targets = [origin + dt.timedelta(days=h)
                        for row in records for origin in [row["origin"]]
                        for h, valid in enumerate(row["targetMask"], start=1) if valid]
    assert records and observed_targets
    assert max(observed_targets) < boundary
    assert dt.date(2026, 3, 1) not in observed_targets


def test_prices_on_or_after_origin_cannot_mutate_the_input_window():
    origin = dt.date(2026, 5, 20)
    cutoff = origin - dt.timedelta(days=1)
    first = cutoff - dt.timedelta(days=83)
    representative = {first + dt.timedelta(days=i): 150.0 + i / 10 for i in range(84)}
    representative[origin] = 99999.0
    detail_id = ("110001", "radish", "A")
    channels = {detail_id: {first + dt.timedelta(days=i): {"price": 70.0 + i / 10,
                                                           "status": "observed"}
                            for i in range(84)}}
    channels[detail_id][origin] = {"price": 88888.0, "status": "observed"}
    selected = [{"channelId": list(detail_id)}]
    before = data.build_input_window("radish", origin, 84, "M", representative, channels, selected)
    representative[origin + dt.timedelta(days=1)] = 77777.0
    channels[detail_id][origin + dt.timedelta(days=1)] = {"price": 66666.0, "status": "observed"}
    after = data.build_input_window("radish", origin, 84, "M", representative, channels, selected)
    assert before == after
    assert before["cutoff"] == cutoff.isoformat()


def test_f23_f24_selection_q_and_training_records_ignore_2026_future_mutations():
    first = dt.date(2022, 1, 1)
    boundary = dt.date(2024, 1, 1)
    representative = {first + dt.timedelta(days=i): 100.0 + i / 10 for i in range(1460)}
    detail_id = ("110001", "radish", "A")
    detail = {first + dt.timedelta(days=i): {"price": 80.0 + i / 10, "status": "observed"}
              for i in range(1460)}
    channels = {detail_id: detail}
    # Future payload contains large values, but neither cutoff is near this data.
    rep_mutated = dict(representative)
    channels_mutated = {detail_id: dict(detail)}
    for i in range(60):
        day = dt.date(2026, 1, 1) + dt.timedelta(days=i)
        rep_mutated[day] = 1_000_000.0 + i
        channels_mutated[detail_id][day] = {"price": 2_000_000.0 + i, "status": "observed"}

    for fit_boundary in (dt.date(2023, 1, 1), dt.date(2024, 1, 1)):
        original_selected = model_code._selected_channels({"radish": channels}, "radish", fit_boundary)
        future_selected = model_code._selected_channels({"radish": channels_mutated}, "radish", fit_boundary)
        assert original_selected == future_selected
        assert model_code._q_scale(representative, fit_boundary) == model_code._q_scale(
            rep_mutated, fit_boundary)
        start = fit_boundary - dt.timedelta(days=100)
        original_records = model_code._records_for_period(
            "radish", representative, channels, 84, "M", original_selected,
            start, fit_boundary, label_before=fit_boundary)
        future_records = model_code._records_for_period(
            "radish", rep_mutated, channels_mutated, 84, "M", future_selected,
            start, fit_boundary, label_before=fit_boundary)
        assert original_records == future_records


def test_replay_serialization_restores_date_origin_for_baseline_join(tmp_path, monkeypatch):
    monkeypatch.setattr(model_code, "OUT", tmp_path)
    origin = dt.date(2026, 4, 1)
    record = {"product": "radish", "origin": origin,
              "window": {"p0": 100.0, "representativeSigma": 0.1, "columns": []},
              "calendar": [[0.0, 1.0, 0.0, 1.0]] * 10,
              "targets": [0.0] * 10, "targetMask": [False] * 10,
              "targetDays": [(origin + dt.timedelta(days=h)).isoformat()
                             for h in range(1, 11)]}
    schema = {"contextDays": 84, "inputType": "U", "repair": "base",
              "selectedInputChannels": []}
    _, payload = model_code._persist_replay_payload("V", "V-202604-radish-U",
                                                     "a" * 64, schema, [record])
    replay_records = model_code._records_from_replay_payload(payload)
    by_origin = {row["origin"]: [1.0] * 10 for row in replay_records}
    assert payload["records"][0]["origin"] == origin.isoformat()
    assert replay_records[0]["origin"] == origin
    assert by_origin.get(origin) == [1.0] * 10


def test_validation_metrics_pair_p0_by_key_and_weight_horizons_equally():
    records = []
    predictions = []
    for i in range(10):
        targets = [0.0] * 10
        mask = [False] * 10
        persistence = 100.0
        if i == 0:
            targets[0] = 100.0; mask[0] = True
            predictions.append([200.0] + [float("nan")] * 9)
        else:
            targets[1] = 100.0; mask[1] = True
            persistence = 200.0
            predictions.append([float("nan"), 100.0] + [float("nan")] * 8)
        records.append({"targets": targets, "targetMask": mask,
                        "window": {"p0": persistence}})
    metrics = model_code._metric_summary(records, predictions)
    assert metrics["model"]["MAPE"] == 50.0  # mean(100% at h1, 0% at h2)
    assert metrics["model"]["MAE"] == 50.0
    assert metrics["p0"]["MAPE"] == 50.0
    assert metrics["model"]["coverageComplete"] is True
    predictions[1][1] = float("nan")
    incomplete = model_code._metric_summary(records, predictions)
    assert incomplete["model"]["coverageComplete"] is False
    assert model_code._candidate_j(incomplete) == float("inf")


def test_finite_large_gradient_norm_alone_does_not_trigger_learning_rate_repair():
    base = {"inner": [], "outer": []}
    for fold in ("F23", "F24"):
        for input_type in ("U", "M"):
            base["inner"].append({"fold": fold, "inputType": input_type,
                                  "product": "radish", "contextDays": 84,
                                  "abnormalGradientEvents": 50,
                                  "history": [{"validationLoss": 1.0,
                                               "trainLoss": 1.0,
                                               "gradientNormMaxBeforeClip": 80.0}]})
            metrics = {"model": {"MAPE": 10.0, "MAE": 10.0},
                       "p0": {"MAPE": 10.0, "MAE": 10.0}}
            base["outer"].append({"fold": fold, "inputType": input_type,
                                  "product": "radish", "contextDays": 84,
                                  "metrics": metrics, "activeAuxCoverage": 0.0})
    assert model_code._trigger_repair(base, "radish", 84) is None


def test_all_infinite_2025_input_score_tie_selects_u():
    assert model_code._select_2025_input({"U": float("inf"), "M": float("inf")}) == "U"


def test_replay_value_tolerance_cannot_hide_issue_class_mismatch():
    result = model_code._compare_saved_model_replay([[0.0]], [[1e-8]])
    assert result["allRowsAgree"] is False
    assert result["issueClassAgreement"] is False


def test_linear_calendar_readout_keeps_direct_horizon_alignment():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    model = model_code.build_model(84, torch=torch, nn=nn, F=F).eval()
    with torch.no_grad():
        model.calendar_head.weight.zero_()
        model.calendar_head.bias.zero_()
        model.calendar_head.weight[0, 0] = 1.0
    z = torch.zeros(1, 11, 84)
    observed = torch.zeros_like(z)
    active = torch.zeros(1, 11, dtype=torch.bool)
    observed[:, 0, :] = 1
    active[:, 0] = True
    roles = torch.zeros(1, 11, dtype=torch.long)
    metadata = torch.zeros(1, 11, 4)
    calendar = torch.zeros(1, 10, 4)
    calendar[0, :, 0] = torch.arange(10, dtype=torch.float32)
    output = model(z, observed, active, roles, metadata, calendar)
    assert torch.equal(output[0], torch.arange(10, dtype=torch.float32))


def _native_record(index, *, targets=None, mask=None):
    length = 84
    columns = []
    for channel in range(11):
        columns.append({"normalized": [0.01 * (index + 1)] * length if channel == 0 else [0.0] * length,
                       "observedMask": [1] * length if channel == 0 else [0] * length,
                       "active": channel == 0, "role": 0 if channel == 0 else 1,
                       "metadata": [0.0] * 4})
    return {"origin": dt.date(2023, 1, 1) + dt.timedelta(days=index),
            "window": {"p0": 100.0 + index, "representativeSigma": 0.1,
                       "columns": columns},
            "calendar": [[0.0, 1.0, 0.0, 1.0]] * 10,
            "targets": list(targets or [0.0] * 10),
            "targetMask": list(mask or [False] * 10)}


def test_global_horizon_weights_add_across_last_partial_effective_batch():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    records = []
    for i in range(5):
        targets = [0.0] * 10
        mask = [False] * 10
        if i in (0, 1):
            targets[0] = 110.0 + i; mask[0] = True
        if i in (2, 3, 4):
            targets[1] = 90.0 + i; mask[1] = True
        if i in (1, 4):
            targets[2] = 120.0 + i; mask[2] = True
        records.append(_native_record(i, targets=targets, mask=mask))
    counts = np.asarray([r["targetMask"] for r in records], dtype=bool).sum(axis=0).tolist()
    model = model_code.build_model(84, torch=torch, nn=nn, F=F).eval()
    device = model_code._device(torch)
    model.to(device=device)
    full, _, _ = model_code._batch_loss(model, records[:3], device, torch,
                                        q_scale=100.0, mae_weight=1.0,
                                        horizon_counts=counts, population=5,
                                        effective_batch=3)
    accumulated = 0.0
    for start in (0, 2):
        micro = records[start:min(start + 2, 3)]
        loss, _, _ = model_code._batch_loss(model, micro, device, torch,
                                            q_scale=100.0, mae_weight=1.0,
                                            horizon_counts=counts, population=5,
                                            effective_batch=3)
        accumulated += float(loss.detach().cpu())
    expected = 0.0
    for h, count in enumerate(counts):
        if not count:
            continue
        for record in records[:3]:
            if record["targetMask"][h]:
                actual = record["targets"][h]
                persistence = record["window"]["p0"]
                error = abs(persistence - actual)
                expected += error / actual / count + error / 100.0 / count
    expected *= 5.0 / 3.0 / 10.0
    assert float(full.detach().cpu()) == pytest.approx(expected, rel=1e-6, abs=1e-8)
    assert accumulated == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_fixed_epoch_fit_ignores_patience_and_resume_keeps_exact_update_count(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    monkeypatch.setattr(model_code, "_batch_loss",
                        lambda model, batch, device, torch, **kwargs:
                        (next(model.parameters()).sum() * 0.0, None, None))
    monkeypatch.setattr(model_code, "_full_loss", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(model_code, "OUT", tmp_path)
    monkeypatch.setattr(model_code, "PROGRESS_PATH", tmp_path / "training-progress.json")
    records = [_native_record(0, targets=[100.0] * 10, mask=[True] * 10)]
    checkpoint = tmp_path / "fixed-fit.pt"
    fit = model_code.fit_model("radish", 84, "U", "base", records, records,
                               100.0, "fixed-31", checkpoint, fixed_epochs=31)
    assert len(fit["history"]) == 31
    assert fit["optimizerUpdates"] == 31
    assert fit["terminalReason"] == "fixed_epoch_complete"
    resumed = model_code.fit_model("radish", 84, "U", "base", records, records,
                                   100.0, "fixed-31", checkpoint, fixed_epochs=31)
    assert len(resumed["history"]) == 31
    assert resumed["optimizerUpdates"] == 31


def test_missing_auxiliary_permutation_does_not_change_prediction():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(22)
    model = model_code.build_model(84, torch=torch, nn=nn, F=F).eval()
    with torch.no_grad():
        model.attention_head.weight.normal_()
        model.attention_head.bias.normal_()
    z = torch.randn(1, 11, 84)
    observed = torch.zeros_like(z)
    observed[:, :2, :] = 1
    active = torch.zeros(1, 11, dtype=torch.bool)
    active[:, :2] = True
    roles = torch.arange(11, dtype=torch.long).remainder(4).unsqueeze(0)
    metadata = torch.randn(1, 11, 4)
    metadata[:, :2, :] = 0
    z[:, 2:, :] = torch.randn_like(z[:, 2:]) * 100
    observed[:, 2:, :] = 0
    active[:, 2:] = False
    calendar = torch.randn(1, 10, 4)
    first = model(z, observed, active, roles, metadata, calendar)
    order = torch.tensor([0, 1, 10, 8, 6, 4, 2, 9, 7, 5, 3])
    permuted = model(z[:, order], observed[:, order], active[:, order], roles[:, order],
                     metadata[:, order], calendar)
    assert torch.allclose(first, permuted, rtol=0, atol=1e-6)
