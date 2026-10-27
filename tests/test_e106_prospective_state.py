import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from agri_data import e106_prospective as issuer


def test_ridge_state_keeps_coefficients_scaler_feature_order_and_training_lineage():
    origin = dt.date(2026, 10, 1)
    target = dt.date(2026, 9, 27)
    input_day = dt.date(2026, 9, 26)
    model = SimpleNamespace(
        named_steps={
            "standardscaler": SimpleNamespace(with_mean=True, with_std=True,
                                               mean_=np.array([1.0, 2.0]),
                                               scale_=np.array([3.0, 4.0]),
                                               var_=np.array([9.0, 16.0])),
            "ridge": SimpleNamespace(alpha=10.0, fit_intercept=True,
                                     coef_=np.array([0.25, -0.5]), intercept_=1.5),
        }
    )
    result = issuer._ridge_state({"model": model, "modelId": "preserved106", "product": "garlic", "horizon": 2,
        "fitOrigin": origin, "trainingTargets": [target], "trainingInputDates": [input_day],
        "featureNames": ["a", "b"], "trainingAudit": {"count": 1}})

    assert result["status"] == "fitted"
    assert result["ridgeCoef"]["values"] == [0.25, -0.5]
    assert result["ridgeIntercept"] == 1.5
    assert result["scaler"]["mean"]["values"] == [1.0, 2.0]
    assert result["featureOrder"] == ["a", "b"]
    assert result["training"]["maxInputDate"] == input_day.isoformat()
    assert result["training"]["maxTruthDate"] == target.isoformat()
    assert len(result["training"]["trainingKeysSha256"]) == 64


def test_finite_array_represents_missing_values_as_json_safe_marked_nulls():
    record = issuer._finite_array(np.array([1.0, float("nan"), float("inf")]))

    assert record["values"] == [1.0, None, None]
    assert record["nonfinite"] == [
        {"flatIndex": 1, "kind": "nan"},
        {"flatIndex": 2, "kind": "positive_infinity"},
    ]


def test_sarima_as_of_selects_checkpoint_and_only_updates_through_cutoff():
    checkpoint = {"event": "pre_update_checkpoint", "calendarDate": "2026-09-30",
                  "filteredState": {"values": [1.0]}}
    updates = [
        {"event": "state_update", "calendarDate": "2026-10-01", "updateInput": {"state": "observed"},
         "filteredState": {"values": [2.0]}},
        {"event": "state_update", "calendarDate": "2026-10-02", "updateInput": {"state": "missing_calendar_observation"},
         "filteredState": {"values": [3.0]}},
        {"event": "state_update", "calendarDate": "2026-10-03", "filteredState": {"values": [99.0]}},
    ]
    state = {"status": "fitted", "preUpdateCheckpoint": checkpoint, "updates": updates}

    selected = issuer._sarima_state_as_of(state, dt.date(2026, 10, 2))

    assert selected["calendarDate"] == "2026-10-02"
    assert selected["filteredState"]["values"] == [3.0]
    assert selected["updateInput"]["state"] == "missing_calendar_observation"
    with pytest.raises(ValueError, match="predates the saved SARIMA checkpoint"):
        issuer._sarima_state_as_of(state, dt.date(2026, 9, 29))


def test_state_envelope_binds_all_fits_attempt_source_and_runtime():
    captured, sink = issuer._make_state_sink()
    for horizon in range(1, 11):
        for product in issuer.PRODUCTS:
            sink({"kind": "ridge", "product": product, "horizon": horizon,
                  "modelId": "direct40",
                  "fitOrigin": dt.date(2026, 10, 1), "status": "unavailable",
                  "reason": "synthetic fixture", "trainingAudit": {"count": 0}})
    for horizon in range(1, 11):
        sink({"kind": "ridge", "product": "garlic", "horizon": horizon,
              "modelId": "preserved106",
              "fitOrigin": dt.date(2026, 10, 1), "status": "unavailable",
              "reason": "synthetic fixture", "trainingAudit": {"count": 0}})
    for product in issuer.PRODUCTS:
        sink({"kind": "sarima", "event": "fit_failed", "product": product,
              "fitOrigin": dt.date(2026, 10, 1), "reason": "synthetic failure"})

    forecast = {"inputSha256": {"issuer": "a" * 64}, "runtime": {"python": "3.11"}}
    source = {"runId": "run", "snapshotSha256": "b" * 64, "cutoff": "2026-10-11",
              "pricesFileSha256": "c" * 64, "provenanceFileSha256": "d" * 64}
    doc = issuer._model_state_document(dt.date(2026, 10, 12), "attempt", "outputs/x/forecast.json",
                                       source, forecast, captured)

    assert len(doc["ridgeModels"]) == 50
    assert set(doc["sarimaModels"]) == set(issuer.PRODUCTS)
    assert doc["source"] == source
    assert doc["inputSha256"] == forecast["inputSha256"]
    assert doc["modelStateSha256"] == issuer._sha(issuer._canonical(
        {k: v for k, v in doc.items() if k != "modelStateSha256"}))


def test_issue_claim_and_terminal_are_linked_content_addressed_and_create_only(tmp_path, monkeypatch):
    issue_date = dt.date(2026, 10, 12)
    started = dt.datetime(2026, 10, 12, 18, 0, 1, tzinfo=issuer.KST)
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "protocol.json").write_text("{}", encoding="utf-8")
    protocol = {"protocolSha256": "a" * 64}
    monkeypatch.setattr(issuer.evaluator, "_load_protocol", lambda _path: protocol)

    claim = issuer._start_issue_claim(registry, issue_date, started, "outputs/prospective-e106/2026-10-12/forecast.json",
                                      {"runId": "run", "snapshotSha256": "b" * 64, "cutoff": "2026-10-11",
                                       "pricesFileSha256": "c" * 64, "provenanceFileSha256": "d" * 64})
    terminal = issuer._finish_issue_claim(registry, issue_date, claim, status="complete",
        finished_at=dt.datetime(2026, 10, 12, 19, 0, tzinfo=issuer.KST),
        artifact_path=claim["artifactTarget"], artifact_raw_sha256="e" * 64,
        artifact_canonical_sha256="f" * 64)

    assert claim["plannedForecastKeyCount"] == 40
    assert claim["schemaVersion"] == 1
    assert claim["protocolSha256"] == protocol["protocolSha256"]
    assert claim["plannedForecastGridSha256"] == issuer.evaluator.grid_sha256(issue_date)
    assert claim["claimSha256"] == issuer._sha(issuer._canonical({k: v for k, v in claim.items() if k != "claimSha256"}))
    assert terminal["claimSha256"] == claim["claimSha256"]
    assert terminal["schemaVersion"] == 1
    assert terminal["protocolSha256"] == claim["protocolSha256"]
    assert terminal["finishedAtKst"] == "2026-10-12T19:00:00+09:00"
    assert terminal["terminalSha256"] == issuer._sha(issuer._canonical({k: v for k, v in terminal.items() if k != "terminalSha256"}))
    with pytest.raises(FileExistsError, match="already claimed"):
        issuer._start_issue_claim(registry, issue_date, started, claim["artifactTarget"], claim["source"])


def test_future_claim_consumes_first_attempt_without_source_binding(tmp_path, monkeypatch):
    issue_date = dt.date(2026, 10, 12)
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "protocol.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(issuer.evaluator, "_load_protocol", lambda _path: {"protocolSha256": "a" * 64})
    started = dt.datetime(2026, 10, 12, 18, 0, tzinfo=issuer.KST)
    claim = issuer._start_issue_claim(registry, issue_date, started, "forecast.json", None,
        requested_source_paths={"prices": "prices.json"}, source_binding_reason="not available")
    assert claim["source"] is None
    assert claim["sourceBindingStatus"] == "unavailable"
    issuer._finish_issue_claim(registry, issue_date, claim, status="failed",
        finished_at=dt.datetime(2026, 10, 12, 18, 1, tzinfo=issuer.KST), failure_type="FileNotFoundError")
    with pytest.raises(FileExistsError, match="already claimed"):
        issuer._start_issue_claim(registry, issue_date, started, "forecast.json", None)


def test_claim_source_binding_must_match_before_fit():
    claim = {"source": {"runId": "run-a", "snapshotSha256": "b" * 64,
                        "cutoff": "2026-10-11", "pricesFileSha256": "c" * 64,
                        "provenanceFileSha256": "d" * 64}}
    changed = dict(claim["source"], runId="run-b")
    with pytest.raises(ValueError, match="differs from the immutable first-attempt claim"):
        issuer._require_claim_source(claim, changed)


def test_sarima_parameter_filter_append_state_restores_without_fit():
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    train = pd.Series(np.arange(12, dtype=float), index=dates)
    train.iloc[5] = np.nan
    kwargs = {"order": (1, 1, 1), "seasonal_order": (1, 0, 0, 7), "trend": "n",
              "enforce_stationarity": True, "enforce_invertibility": True}
    original = SARIMAX(train.iloc[:10], **kwargs).filter([0.1, 0.1, 0.1, 0.1])
    saved_checkpoint = issuer._sarima_snapshot(original, product="garlic",
        fit_origin=dt.date(2026, 1, 11), calendar_date=dt.date(2026, 1, 10),
        fit_audit={"trainStart": "2026-01-01", "trainEnd": "2026-01-10"}, event="pre_update_checkpoint")
    missing_update = pd.Series([np.nan], index=[dates[10]])
    saved = original.append(missing_update, refit=False)
    serialized = {"params": saved_checkpoint["model"]["params"]["values"],
                  "filteredState": saved_checkpoint["filteredState"],
                  "filteredStateCovariance": saved_checkpoint["filteredStateCovariance"],
                  "update": [None], "updateDate": dates[10].isoformat(),
                  "kwargs": {"order": tuple(saved_checkpoint["model"]["order"]),
                             "seasonal_order": tuple(saved_checkpoint["model"]["seasonalOrder"]),
                             "trend": saved_checkpoint["model"]["trend"],
                             "enforce_stationarity": saved_checkpoint["model"]["enforceStationarity"],
                             "enforce_invertibility": saved_checkpoint["model"]["enforceInvertibility"]}}
    assert saved_checkpoint["model"]["filterOptions"]
    assert saved_checkpoint["model"]["initialState"]["shape"]

    def shaped(item):
        return np.asarray(item["values"], dtype=float).reshape(item["shape"])

    restored_index = pd.date_range(start=serialized["updateDate"], periods=1, freq="D")
    restored_model = SARIMAX(pd.Series([np.nan], index=restored_index),
                             **serialized["kwargs"])
    restored_model.update(serialized["params"])
    system = restored_model.ssm
    transition = system.transition[:, :, 0]
    selection = system.selection[:, :, 0]
    state_covariance = system.state_cov[:, :, 0]
    checkpoint_state = shaped(serialized["filteredState"])
    checkpoint_covariance = shaped(serialized["filteredStateCovariance"])
    next_state = transition @ checkpoint_state
    next_covariance = (transition @ checkpoint_covariance @ transition.T
                       + selection @ state_covariance @ selection.T)
    restored_model.ssm.initialize_known(next_state, next_covariance)
    for option, value in saved_checkpoint["model"]["filterOptions"].items():
        setattr(restored_model.ssm, option, value)
    restored = restored_model.filter(serialized["params"])
    np.testing.assert_allclose(np.asarray(saved.forecast(1)), np.asarray(restored.forecast(1)))


def test_terminal_timestamp_cannot_precede_claim_start(tmp_path):
    claim = {"attemptId": "attempt", "startedAtKst": "2026-10-12T18:00:00+09:00",
             "protocolSha256": "a" * 64, "claimSha256": "b" * 64,
             "artifactTarget": "forecast.json"}
    with pytest.raises(ValueError, match="precedes claim start"):
        issuer._finish_issue_claim(tmp_path, dt.date(2026, 10, 12), claim, status="failed",
            finished_at=dt.datetime(2026, 10, 12, 17, 59, tzinfo=issuer.KST))


def test_claim_requires_future_issue_after_18_kst(tmp_path):
    with pytest.raises(ValueError, match="at or after 18:00"):
        issuer._start_issue_claim(tmp_path, dt.date(2026, 10, 12),
            dt.datetime(2026, 10, 12, 17, 59, tzinfo=issuer.KST), "forecast.json", None)
