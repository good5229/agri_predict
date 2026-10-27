"""Issue immutable onion D+1/D+2 forecasts from a verified local snapshot."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from . import dashboard, direct_horizon as dh, market_mix, modeling

ROOT = Path(__file__).resolve().parents[1]
MARKETS = tuple(dashboard.MARKETS)
HORIZONS = (1, 2)
MIN_TRAIN_DAYS = 90
MAX_AGE_DAYS = 14
SHARE_DAYS = 28
MIN_SHARE_DAYS = 5
KST = ZoneInfo("Asia/Seoul")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                        allow_nan=False) + "\n").encode("utf-8")


def _immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _positive(series):
    result = {}
    for day, row in series.items():
        try:
            price, weight = float(row["price"]), float(row["weight"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(price) and math.isfinite(weight) and price > 0 and weight > 0:
            result[day] = (price, weight)
    return result


def _share(markets, issue_date):
    start = issue_date - dt.timedelta(days=SHARE_DAYS - 1)
    g, j = (markets["onion"][m] for m in MARKETS)
    days = sorted(d for d in g.keys() & j.keys() if start <= d <= issue_date - dt.timedelta(days=1))
    last = days[-1] if days else None
    reason = None
    if len(days) < MIN_SHARE_DAYS:
        reason = "fewer_than_5_positive_cotrade_days_in_28_day_window"
    elif last is None or (issue_date - last).days > MAX_AGE_DAYS:
        reason = "latest_positive_cotrade_older_than_14_days"
    if reason:
        return None, days, last, reason
    values = sorted(g[d][1] / (g[d][1] + j[d][1]) for d in days)
    n = len(values)
    share = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    return share, days, last, None


def _fit_predict(series, issue_date, model_state=None, horizons=HORIZONS):
    """Fit a month's Ridge once, then use only the saved scaler and coefficients."""
    import numpy as np

    cutoff = issue_date - dt.timedelta(days=1)
    actuals = {d: v for d, v in series.items() if d <= cutoff}
    cache, output, states = {}, {}, {}
    horizons = tuple(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("horizons must be a nonempty tuple of unique positive integers")
    for h in horizons:
        samples = []
        for target, (actual, _) in sorted(actuals.items()):
            origin = target - dt.timedelta(days=h)
            feat = modeling._feature_row(actuals, origin, target, h, cache)
            if feat is not None:
                samples.append((target, origin, feat[1], math.log(actual / actuals[feat[0]][0])))
        train = [r for r in samples if r[0] < issue_date]
        n = len({r[0] for r in train})
        last_day = max(actuals, default=None)
        age = (issue_date - last_day).days if last_day else None
        row = {"horizon": h, "targetDate": (issue_date + dt.timedelta(days=h)).isoformat(),
               "trainingTargetDays": n, "trainingSampleCount": len(train),
               "trainingTargetEnd": max((r[0] for r in train), default=None).isoformat() if train else None,
               "sourceDate": last_day.isoformat() if last_day else None,
               "sourceAgeDays": age, "status": "unavailable", "reason": None}
        state = model_state.get("byHorizon", {}).get(str(h)) if model_state else None
        if model_state and not state:
            raise ValueError(f"monthly model state is missing horizon {h}")
        if model_state and not state.get("available"):
            row.update(trainingTargetDays=state["trainingTargetDays"],
                       trainingSampleCount=state["trainingSampleCount"],
                       trainingTargetEnd=state.get("trainingTargetEnd"))
            row["reason"] = state["reason"]
            output[h] = row
            states[str(h)] = state
            continue
        if model_state and state.get("fitOrigin", "")[:7] != issue_date.strftime("%Y-%m"):
            raise ValueError("monthly model state month mismatch")
        if not model_state and n >= MIN_TRAIN_DAYS:
            try:
                model = modeling._new_estimator("ridge")
                model.fit(np.asarray([r[2] for r in train]), np.asarray([r[3] for r in train]))
                scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
                state = {"available": True, "fitOrigin": issue_date.isoformat(),
                         "trainingTargetEnd": max(r[0] for r in train).isoformat(),
                         "trainingTargetDays": n, "trainingSampleCount": len(train),
                         "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                         "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}
                if not all(math.isfinite(float(x)) for key in ("mean", "scale", "coef") for x in state[key]) or not math.isfinite(state["intercept"]):
                    raise ValueError("non_finite_month_model_state")
                if any(float(x) <= 0 for x in state["scale"]):
                    raise ValueError("invalid_month_model_scale")
            except Exception as exc:
                state = {"available": False, "fitOrigin": issue_date.isoformat(),
                         "trainingTargetEnd": row["trainingTargetEnd"], "trainingTargetDays": n,
                         "trainingSampleCount": len(train), "reason": "ridge_fit_failed:" + type(exc).__name__}
        elif not model_state:
            state = {"available": False, "fitOrigin": issue_date.isoformat(),
                     "trainingTargetEnd": row["trainingTargetEnd"], "trainingTargetDays": n,
                     "trainingSampleCount": len(train), "reason": "fewer_than_90_distinct_training_target_days"}
        states[str(h)] = state
        row.update(trainingTargetDays=state["trainingTargetDays"],
                   trainingSampleCount=state["trainingSampleCount"],
                   trainingTargetEnd=state["trainingTargetEnd"])
        if not state["available"]:
            row["reason"] = state["reason"]
            output[h] = row
            continue
        if last_day is None or age is None or age > MAX_AGE_DAYS:
            row["reason"] = "latest_positive_source_older_than_14_days"
            output[h] = row
            continue
        fit = modeling._feature_row(actuals, issue_date, issue_date + dt.timedelta(days=h), h, cache)
        if fit is None:
            row["reason"] = "no_fresh_prediction_feature_source"
            output[h] = row
            continue
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                delta = float(((np.asarray(fit[1]) - np.asarray(state["mean"])) /
                               np.asarray(state["scale"])) @ np.asarray(state["coef"]) + state["intercept"])
            if not math.isfinite(delta):
                raise ValueError("non_finite_raw_log_delta")
            pred, _ = modeling._inverse_log_ratio(actuals[fit[0]][0], delta)
            if not math.isfinite(pred) or pred <= 0:
                raise ValueError("non_finite_or_nonpositive_prediction")
        except Exception as exc:
            row["reason"] = "prediction_failed:" + str(exc)[:100]
            output[h] = row
            continue
        row.update(status="available", directRidgePrediction=pred)
        output[h] = row
    return output, {"month": issue_date.strftime("%Y-%m"), "fitOrigin": (model_state or {}).get("fitOrigin", issue_date.isoformat()), "byHorizon": states}


def build(raw, markets, issue_date: dt.date, issued_at: dt.datetime, prices_sha256: str,
          selected_provenance, model_states=None, state_reference=None, horizons=HORIZONS):
    """Pure forecast construction; accepts injected data for offline unit tests."""
    if issued_at.tzinfo is None or issued_at.astimezone(KST).date() != issue_date:
        raise ValueError("issuedAtKst must fall on issueDateKst")
    horizons = tuple(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("horizons must be a nonempty tuple of unique positive integers")
    if not isinstance(prices_sha256, str) or len(prices_sha256) != 64:
        raise ValueError("prices.json SHA-256 is required")
    core = dh._observations(raw)
    if "onion" not in core or not isinstance(markets, dict) or "onion" not in markets:
        raise ValueError("onion data is required")
    market = {m: _positive(markets["onion"].get(m, {})) for m in MARKETS}
    model_states = model_states or {}
    if model_states and set(model_states) != {"direct", *MARKETS}:
        raise ValueError("monthly model state is incomplete")
    if model_states:
        for name, state in model_states.items():
            if (state.get("month") != issue_date.strftime("%Y-%m") or
                    dt.date.fromisoformat(state.get("fitOrigin", "")) > issue_date):
                raise ValueError(f"monthly model state is invalid: {name}")
    direct, direct_state = _fit_predict(core["onion"], issue_date, model_states.get("direct"), horizons)
    market_fits, market_states = {}, {}
    for m in MARKETS:
        market_fits[m], market_states[m] = _fit_predict(market[m], issue_date, model_states.get(m), horizons)
    share, share_days, last_share_day, share_reason = _share({"onion": market}, issue_date)
    records = []
    for h in horizons:
        fit = direct[h]
        reason = share_reason
        market_values = {}
        market_audits = {m: {k: market_fits[m][h].get(k) for k in (
            "status", "reason", "sourceDate", "sourceAgeDays", "trainingTargetDays",
            "trainingSampleCount", "trainingTargetEnd")} for m in MARKETS}
        if share is not None:
            for m in MARKETS:
                mfit = market_fits[m][h]
                market_values[m] = mfit.get("directRidgePrediction")
                try:
                    if market_values[m] is not None and (not math.isfinite(float(market_values[m])) or float(market_values[m]) <= 0):
                        market_values[m] = None
                except (TypeError, ValueError, OverflowError):
                    market_values[m] = None
                latest = mfit.get("sourceDate")
                if mfit["status"] != "available":
                    reason = f"market_{m}_unavailable:{mfit.get('reason')}"
                elif market_values[m] is None:
                    reason = f"market_{m}_non_finite"
                elif latest is None or (issue_date - dt.date.fromisoformat(latest)).days > MAX_AGE_DAYS:
                    reason = f"market_{m}_source_stale"
        direct_value = fit.get("directRidgePrediction")
        if (direct_value is not None and
                (not math.isfinite(float(direct_value)) or float(direct_value) <= 0)):
            direct_value = None
        if fit["status"] != "available" or direct_value is None:
            reason = "direct_ridge_unavailable:" + str(fit["reason"])
        if reason:
            chosen = direct_value
            method = "direct_ridge_fallback" if chosen is not None else "unissued"
        else:
            chosen = share * market_values[MARKETS[0]] + (1 - share) * market_values[MARKETS[1]]
            method = "market_ridge_28d_median_weight_share"
        persistence = None
        latest_actual = max((d for d in core["onion"] if d <= issue_date - dt.timedelta(days=1)), default=None)
        if latest_actual is not None and (issue_date - latest_actual).days <= MAX_AGE_DAYS:
            persistence = core["onion"][latest_actual][0]
        if chosen is not None and (not math.isfinite(float(chosen)) or float(chosen) <= 0):
            chosen = None
            method = "unissued"
            reason = "non_finite_or_nonpositive_final_prediction"
        records.append({"horizon": h, "targetDate": (issue_date + dt.timedelta(days=h)).isoformat(),
                        "status": "available" if chosen is not None else "unavailable",
                        "prediction": chosen, "method": method, "fallbackReason": reason,
                        "marketPredictions": market_values, "marketFitAudits": market_audits, "share": share,
                        "shareDayCount": len(share_days),
                        "lastShareObservationDate": last_share_day.isoformat() if last_share_day else None,
                        "sourceDate": fit.get("sourceDate"), "sourceAgeDays": fit.get("sourceAgeDays"),
                        "trainingTargetDays": fit["trainingTargetDays"],
                        "trainingSampleCount": fit["trainingSampleCount"],
                        "trainingTargetEnd": fit["trainingTargetEnd"],
                        "directRidgePrediction": direct_value,
                        "persistencePrediction": persistence})
    states = {"direct": direct_state, **market_states}
    state_digest = _sha(_json_bytes(states))
    return {"schemaVersion": 1, "issuedAtKst": issued_at.astimezone(KST).isoformat(timespec="seconds"),
            "issueDateKst": issue_date.isoformat(), "pricesSha256": prices_sha256,
            "selectedScopesSha256": selected_provenance["selectedScopesSha256"],
            "selectedScopeCount": selected_provenance["selectedScopeCount"],
            "selectedRequestHashes": selected_provenance["selectedRequestHashes"],
            "selectedQueryStates": selected_provenance.get("selectedQueryStates"),
            "extractionState": selected_provenance.get("extractionState"),
            "dataVintage": "latest_verified_response; first_publication_time_unknown",
            "featureRule": "all model features use observations on or before issueDateKst-1; training target < issueDateKst",
            "shareRule": "median positive Garak netweight share over positive co-trade dates in prior 28 calendar days; minimum 5 dates and latest <=14 days",
            "modelStateSha256": state_digest,
            "modelState": states if state_reference is None else None,
            "modelStateReference": state_reference,
            "products": [{"productId": "onion", "forecasts": records}]}


def _load_month_state(state_path: Path, issue_date: dt.date, issued_at: dt.datetime | None = None, horizons=HORIZONS):
    requested_horizons = tuple(horizons)
    state_bytes = state_path.read_bytes()
    previous = json.loads(state_bytes)
    previous_date = dt.date.fromisoformat(previous.get("issueDateKst", ""))
    if previous_date >= issue_date or previous_date.strftime("%Y-%m") != issue_date.strftime("%Y-%m"):
        raise ValueError("monthly model state month mismatch")
    states = previous.get("modelState")
    if not isinstance(states, dict):
        raise ValueError("monthly model state is incomplete")
    fit_origin = dt.date.fromisoformat(states.get("direct", {}).get("fitOrigin", ""))
    if fit_origin != previous_date or fit_origin > previous_date:
        raise ValueError("monthly model state fit origin mismatch")
    try:
        first_issued_at = dt.datetime.fromisoformat(previous["issuedAtKst"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("monthly model state issue timestamp is invalid") from exc
    if first_issued_at.tzinfo is None or first_issued_at.astimezone(KST).date() != previous_date:
        raise ValueError("monthly model state issue timestamp is invalid")
    if issued_at is not None and first_issued_at.astimezone(KST) >= issued_at.astimezone(KST):
        raise ValueError("monthly model state was not issued before this forecast")
    if _sha(_json_bytes(states)) != previous.get("modelStateSha256"):
        raise ValueError("monthly model state hash mismatch")
    if set(states) != {"direct", *MARKETS}:
        raise ValueError("monthly model state is incomplete")
    for name, model in states.items():
        if not isinstance(model, dict) or model.get("month") != issue_date.strftime("%Y-%m") or model.get("fitOrigin") != previous_date.isoformat():
            raise ValueError(f"monthly model state metadata is invalid: {name}")
        horizon_states = model.get("byHorizon")
        if not isinstance(horizon_states, dict) or set(horizon_states) != {str(h) for h in requested_horizons}:
            raise ValueError(f"monthly model state horizons are incomplete: {name}")
        for horizon, state in horizon_states.items():
            if not isinstance(state, dict) or state.get("fitOrigin") != previous_date.isoformat():
                raise ValueError(f"monthly model state fit audit is invalid: {name}/{horizon}")
            for key in ("trainingTargetDays", "trainingSampleCount"):
                if not isinstance(state.get(key), int) or state[key] < 0:
                    raise ValueError(f"monthly model state training audit is invalid: {name}/{horizon}")
            end = state.get("trainingTargetEnd")
            if end is not None and dt.date.fromisoformat(end) >= previous_date:
                raise ValueError(f"monthly model state training end is invalid: {name}/{horizon}")
            if state.get("available"):
                try:
                    mean, scale, coef = state["mean"], state["scale"], state["coef"]
                    if not (len(mean) == len(scale) == len(coef) == len(modeling.FEATURE_NAMES)):
                        raise ValueError
                    values = [*mean, *scale, *coef, state["intercept"]]
                    if not all(math.isfinite(float(x)) for x in values) or any(float(x) <= 0 for x in scale):
                        raise ValueError
                    if state["trainingTargetDays"] < MIN_TRAIN_DAYS or not state.get("trainingTargetEnd"):
                        raise ValueError
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"monthly model state parameters are invalid: {name}/{horizon}") from exc
            elif not state.get("reason"):
                raise ValueError(f"monthly unavailable model state lacks reason: {name}/{horizon}")
    return states, {"path": str(state_path), "sha256": _sha(state_bytes)}


def run(prices_path: Path, db_path: Path, output_path: Path, *, issue_date=None,
        now=None, allow_custom_date=False, state_path=None, first_month_issue=False, horizons=HORIZONS):
    now = (now or dt.datetime.now(KST)).astimezone(KST)
    issue_date = issue_date or now.date()
    if not allow_custom_date and issue_date != now.date():
        raise ValueError("real issuance is restricted to the current KST date")
    if now.date() != issue_date:
        raise ValueError("issuedAtKst date must match issueDateKst")
    if not allow_custom_date and bool(state_path) == bool(first_month_issue):
        raise ValueError("real issuance requires exactly one of --state or --first-issue")
    horizons = tuple(horizons)
    raw_bytes = prices_path.read_bytes()
    raw = json.loads(raw_bytes)
    core = dh._observations(raw)
    markets, provenance = market_mix.extract_market_data(db_path, raw, core)
    selected = {**provenance["selected"],
                "extractionState": {"reconciliation": provenance["reconciliation"],
                                    "broadScopesSha256": provenance["broad"].get("scopesSha256"),
                                    "broadScopeCount": provenance["broad"].get("scopeCount")}}
    previous_state, state_reference = None, None
    if state_path is not None:
        previous_state, state_reference = _load_month_state(state_path, issue_date, now, horizons)
    payload = build(raw, markets, issue_date, now, _sha(raw_bytes), selected,
                    model_states=previous_state, state_reference=state_reference, horizons=horizons)
    _immutable(output_path, _json_bytes(payload))
    return {"output": str(output_path), "sha256": _sha(_json_bytes(payload)),
            "issueDateKst": issue_date.isoformat(),
            "availableHorizons": sum(r["status"] == "available" for r in payload["products"][0]["forecasts"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "data/agri.sqlite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issue-date", type=dt.date.fromisoformat)
    parser.add_argument("--state", type=Path, help="first issue artifact for this month")
    parser.add_argument("--first-issue", action="store_true", help="explicitly establish this month's model state")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.prices, args.db, args.output, issue_date=args.issue_date,
                         state_path=args.state, first_month_issue=args.first_issue), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
