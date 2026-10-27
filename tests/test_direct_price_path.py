from __future__ import annotations

import datetime as dt

import numpy as np

from agri_data import direct_price_path as model_code


def test_calendar_features_use_actual_leap_year_length_and_target_weekday():
    features = model_code.calendar_features([dt.date(2024, 2, 29), dt.date(2023, 3, 1)])
    assert len(features) == 2 and all(len(row) == 4 for row in features)
    assert np.allclose(features[0][:2], features[1][:2]) is False
    leap_position = (60 - 1) / 366
    nonleap_position = (60 - 1) / 365
    assert np.isclose(features[0][2], np.sin(2 * np.pi * leap_position))
    assert np.isclose(features[1][2], np.sin(2 * np.pi * nonleap_position))


def test_decomposition_uses_replicated_mask_weighted_edges_and_momentum_support():
    z = np.array([[2.0, 0.0, 8.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]])
    mask = np.array([[1, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    trend, trend_support, seasonal, momentum, momentum_mask = model_code._decompose_numpy(z, mask)
    # At the left edge, the first value is replicated with its mask three times.
    assert np.isclose(trend[0, 0], 20.0 / 6.0)
    assert trend_support[0, 0]
    assert seasonal[0, 1] == 0
    assert momentum[0, 7] == 10.0
    assert momentum_mask[0, 7]
    assert not momentum_mask[0, 6]


def test_infinite_selection_ratio_is_explicit_and_json_safe():
    assert np.isinf(model_code._ratio(1, 0))
    assert model_code._ratio(0, 0) == 1
    assert model_code._safe_number(float("inf")) is None


def test_replay_hashes_bind_answer_free_inputs_and_joined_fit_rows(tmp_path, monkeypatch):
    record = {"product": "radish", "origin": "2026-04-01",
              "window": {"contextDays": 84, "selectedInputChannels": [],
                         "columns": [{"normalized": [0.0], "observedMask": [1]}]},
              "calendar": [[0.0, 1.0, 0.0, 1.0]]}
    first = model_code._inference_payload_digest([record])
    changed = {**record, "window": {**record["window"], "contextDays": 168}}
    assert first != model_code._inference_payload_digest([changed])
    preds = [[1.0, 2.0]]
    assert model_code._replay_prediction_digest([record], preds) == model_code._replay_prediction_digest([record], preds)
    fit = {"inputType": "U", "fitMonth": "2026-04", "product": "radish"}
    row = {"product": "radish", "origin": "2026-04-01", "target": "2026-04-02",
           "horizon": 1, "U": {"rawPrediction": 10.0, "rawStatus": "issued"}}
    digest = model_code._forecast_rows_for_fit_digest([row], fit)
    assert digest != model_code._forecast_rows_for_fit_digest(
        [{**row, "U": {"rawPrediction": 11.0, "rawStatus": "issued"}}], fit)
    monkeypatch.setattr(model_code, "OUT", tmp_path)
    schema = {"contextDays": 84, "inputType": "U", "repair": "base",
              "selectedInputChannels": []}
    path, saved = model_code._persist_replay_payload("V", "V-202604-radish-U",
                                                      "a" * 64, schema,
                                                      [{**record, "targetDays": ["2026-04-02"]}])
    assert path.is_file()
    assert saved["rowPayloadSha256"] == model_code._inference_payload_digest(saved["records"])
    assert saved["schemaSha256"] == model_code._sha(model_code._canonical(schema))


def test_saved_model_replay_comparison_reports_all_horizon_agreement():
    comparison = model_code._compare_saved_model_replay([[100.0, float("nan")]],
                                                         [[100.0 + 1e-7, float("nan")]])
    assert comparison["allRowsAgree"] is True
    assert comparison["inferenceRowCount"] == 1
    assert comparison["horizonCount"] == 2
    assert comparison["maxAbsoluteDifference"] > 0
    disagreement = model_code._compare_saved_model_replay([[100.0, 1.0]], [[101.0, 1.0]])
    assert disagreement["allRowsAgree"] is False


def test_model_shapes_zero_initial_output_and_u_auxiliary_invariance():
    import pytest
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    for length in (84, 168):
        model = model_code.build_model(length, torch=torch, nn=nn, F=F).eval()
        z = torch.zeros(2, 11, length)
        observed = torch.zeros_like(z)
        observed[:, 0, :] = 1
        active = torch.zeros(2, 11, dtype=torch.bool)
        active[:, 0] = True
        roles = torch.zeros(2, 11, dtype=torch.long)
        metadata = torch.zeros(2, 11, 4)
        calendar = torch.randn(2, 10, 4)
        first = model(z, observed, active, roles, metadata, calendar)
        assert first.shape == (2, 10)
        assert torch.equal(first, torch.zeros_like(first))
        changed = z.clone()
        changed[:, 1:, :] = 1000
        second = model(changed, observed, active, roles, metadata, calendar)
        assert torch.equal(first, second)


def test_channel_attention_gradients_flow_after_zero_readout_update():
    import pytest
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(7)
    model = model_code.build_model(84, torch=torch, nn=nn, F=F)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    z = torch.randn(4, 11, 84)
    observed = torch.ones_like(z)
    active = torch.ones(4, 11, dtype=torch.bool)
    roles = torch.zeros(4, 11, dtype=torch.long)
    metadata = torch.randn(4, 11, 4)
    calendar = torch.randn(4, 10, 4)
    # With zero final readouts, the first step updates those readouts. The next
    # backward pass must reach the attention projections through that learned head.
    optimizer.zero_grad(set_to_none=True)
    (model(z, observed, active, roles, metadata, calendar) - 1).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (model(z, observed, active, roles, metadata, calendar) - 1).square().sum().backward()
    grad = model.attention.in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0
