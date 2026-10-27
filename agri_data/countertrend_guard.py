"""Replay-gated fixed countertrend shrinkage of the frozen trend-correction candidate.

No model is fitted here. C only shrinks E toward the latest causal price when
the frozen 7- and 28-day trends agree and E points against both.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from . import equal_info_online_forecasts as forecasts
from . import equal_info_online_score as scoring
from . import direct_horizon
from . import trend_response_correction as prior

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = prior.MANIFEST
PRICES = prior.PRICES
PRIOR_DIR = ROOT / "outputs/experiments/trend-response-correction-20261010"
PRIOR_SELECTION = PRIOR_DIR / "selection.json"
PRIOR_CONFIRMATION = PRIOR_DIR / "confirmation-manifest.json"
PRIOR_CONFIRMATION_REPORT = PRIOR_DIR / "confirmation-report.json"
PRIOR_FINAL = PRIOR_DIR / "final-manifest.json"
PRIOR_FINAL_REPORT = PRIOR_DIR / "final-report.json"
OUT = ROOT / "outputs/experiments/countertrend-guard-20261010"
MANIFEST_SHA256 = "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
SELECTION_SHA256 = "59fa52c3f7c9664d52ad8ce23987eb4d6c70af0fca9907c25596a84eb77ba26b"
CONFIRMATION_MANIFEST_SHA256 = "bb929cadbdee4862b71ac93f97a0ecbed9ea7ecb3ff78ac5251e281bdea5d74b"
FINAL_MANIFEST_SHA256 = "7ec6453660e020298bf407b829fdda9c8807c739769b5e58abc4d04631662502"
CONFIRMATION_REPORT_SHA256 = "9f7d90fc2ef4b748577fe3637ee255da6e21ead859a735702e9665f46e1a2f55"
CONF_START, CONF_END = prior.CONF_START, prior.CONF_END
FINAL_START, FINAL_END = prior.FINAL_START, prior.FINAL_END
PRODUCTS = prior.ALL_PRODUCTS
GUARD_PRODUCTS = ("cabbage", "radish")
HORIZONS = prior.HORIZONS
BANDWIDTHS = (9, 28)
TEST_FAMILY_SIZE = 40


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def positive(value):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) and n > 0 else None


def apply_guard(product, horizon, p0, e, m7, m28):
    """Return (C, applied, reason); noneligible and invalid inputs preserve E."""
    base = positive(e)
    if base is None:
        return None, False, "baseline_not_issued"
    anchor = positive(p0)
    if product not in GUARD_PRODUCTS:
        return base, False, "product_out_of_scope"
    if horizon not in range(4, 11):
        return base, False, "horizon_out_of_scope"
    if anchor is None or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (m7, m28)):
        return base, False, "momentum_unavailable"
    if m7 == 0 or m28 == 0 or (m7 > 0) != (m28 > 0):
        return base, False, "momentum_not_strictly_aligned"
    forecast_move = math.log(base) - math.log(anchor)
    if forecast_move * m7 >= 0:
        return base, False, "forecast_not_countertrend"
    try:
        candidate = math.exp((math.log(anchor) + math.log(base)) / 2)
    except (OverflowError, ValueError):
        return base, False, "nonfinite_candidate_fallback_E"
    if not math.isfinite(candidate) or candidate <= 0:
        return base, False, "nonfinite_candidate_fallback_E"
    return candidate, True, None


def _validate_digest(doc, key, expected_phase=None):
    unsigned = dict(doc)
    digest = unsigned.pop(key, None)
    if not isinstance(digest, str) or digest != sha(canonical(unsigned)):
        raise ValueError(f"{key} digest mismatch")
    if expected_phase and doc.get("phase") != expected_phase:
        raise ValueError("unexpected phase")


def _verify_sources(manifest_raw, prices_raw, selection_raw, confirmation_raw, confirmation_report_raw, final_raw):
    expected = (MANIFEST_SHA256, PRICES_SHA256, SELECTION_SHA256, CONFIRMATION_MANIFEST_SHA256,
                CONFIRMATION_REPORT_SHA256, FINAL_MANIFEST_SHA256)
    actual = tuple(sha(x) for x in (manifest_raw, prices_raw, selection_raw, confirmation_raw,
                                    confirmation_report_raw, final_raw))
    if actual != expected:
        raise ValueError("a pinned source artifact SHA-256 mismatch")
    manifest = json.loads(manifest_raw)
    prices = json.loads(prices_raw)
    forecasts._validate(manifest)
    selection = json.loads(selection_raw)
    _validate_digest(selection, "selectionSha256", "selection_only")
    if selection.get("inputSha256") != {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256}:
        raise ValueError("selection source identity mismatch")
    conf = json.loads(confirmation_raw)
    _validate_digest(conf, "manifestSha256", "correction_forecast_only")
    final = json.loads(final_raw)
    _validate_digest(final, "manifestSha256", "final_forecast_only")
    report = json.loads(confirmation_report_raw)
    _validate_digest(report, "reportSha256", "confirmation_scored")
    if report.get("inputSha256") != {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256}:
        raise ValueError("prior confirmation report source identity mismatch")
    if (conf.get("selectionSha256") != selection["selectionSha256"]
            or final.get("selectionSha256") != selection["selectionSha256"]
            or final.get("confirmationReportSha256") != report.get("reportSha256")
            or final.get("frozenCrops") != ["cabbage"]):
        raise ValueError("prior E artifact cross-reference mismatch")
    return manifest, prices, selection, conf, report, final


def _implementation_hashes():
    paths = {"countertrendGuard": Path(__file__), "priorCorrection": Path(prior.__file__),
             "baselineForecastProtocol": Path(forecasts.__file__), "scoreProtocol": Path(scoring.__file__),
             "directHorizonPrices": Path(direct_horizon.__file__), "modelingFeatures": Path(forecasts.modeling.__file__)}
    return {name: sha(path.read_bytes()) for name, path in paths.items()}


def _period_rows(doc, period):
    key, start, end, origin_floor, has_eligibility = (("confirmationForecasts", CONF_START, CONF_END, CONF_START, True)
                                     if period == "confirmation" else
                                     ("finalForecasts", FINAL_START, FINAL_END, dt.date(2026, 7, 1), False))
    rows = doc.get(key)
    if not isinstance(rows, list):
        raise ValueError("missing prior forecast rows")
    expected = {(p, start + dt.timedelta(days=i), h) for p in PRODUCTS
                for i in range((end-start).days + 1) for h in HORIZONS
                if start + dt.timedelta(days=i) - dt.timedelta(days=h) >= origin_floor}
    seen = set()
    for row in rows:
        target, origin, h = dt.date.fromisoformat(row["target"]), dt.date.fromisoformat(row["origin"]), row["horizon"]
        ident = (row["product"], target, h)
        if ident not in expected or ident in seen or origin != target-dt.timedelta(days=h):
            raise ValueError("prior candidate grid invalid")
        if has_eligibility and row.get("eligible") is not True:
            raise ValueError("prior candidate includes ineligible row")
        if not has_eligibility:
            row["eligible"] = True
        seen.add(ident)
    if seen != expected:
        raise ValueError("prior candidate grid incomplete")
    return rows


def _get_e(row):
    return positive(row.get("model", {}).get("prediction"))


def _guard_forecast(row):
    e = _get_e(row)
    features = row.get("features") or {}
    p0 = positive(features.get("p0"))
    if features.get("reason"):
        c, applied, reason = e, False, "momentum_unavailable"
    else:
        c, applied, reason = apply_guard(row["product"], row["horizon"], p0, e,
                                         features.get("momentum7Log"), features.get("momentum28Log"))
    return {"product": row["product"], "target": row["target"], "origin": row["origin"],
            "horizon": row["horizon"], "eligible": row["eligible"],
            "model": {"status": "issued" if c is not None else "not_issued", "prediction": c,
                      "reason": None if c is not None else (row.get("model", {}).get("reason") or "prior_E_not_issued")},
            "baselinePrediction": e, "p0": p0, "momentum7Log": features.get("momentum7Log"),
            "momentum28Log": features.get("momentum28Log"), "guardApplied": applied,
            "fallbackReason": reason}


def build_train_diagnostic(manifest_raw, prices_raw, selection_raw, confirmation_raw, confirmation_report_raw, final_raw):
    manifest, prices, selection, conf, report, final = _verify_sources(manifest_raw, prices_raw, selection_raw,
                                                                         confirmation_raw, confirmation_report_raw, final_raw)
    series = direct_horizon._observations(prices)
    actuals = prior._actuals(prices, prior.TRAIN_START, prior.TRAIN_END)
    rows = []
    selection_map = prior._selection_map(selection)
    # Training targets are in the frozen original archive, not in the later
    # confirmation/final manifests. Reapply the already frozen E rule only;
    # never rerun the earlier selection or fit a baseline model.
    for row in prior._all_rows(manifest):
        target = dt.date.fromisoformat(row["target"])
        if (row["product"] not in GUARD_PRODUCTS or row["horizon"] not in range(4, 11)
                or row.get("eligible") is not True or not (prior.TRAIN_START <= target <= prior.TRAIN_END)):
            continue
        base = positive(row.get("models", {}).get("causal50", {}).get("prediction"))
        if base is None:
            continue
        features = prior._price_features(series, row["product"], dt.date.fromisoformat(row["origin"]))
        e = base
        if base is not None and not features.get("reason") and features.get("regime") != "neutral":
            chosen = selection_map.get((row["product"], features.get("regime")))
            if chosen and chosen.get("supported"):
                e = prior._scale_movement(base, features["p0"], chosen["scale"]) or base
        pseudo = {"product":row["product"],"horizon":row["horizon"],"target":row["target"],"origin":row["origin"],
                  "eligible":True,"model":{"prediction":e},"features":features}
        guarded = _guard_forecast(pseudo)
        actual = actuals.get((row["product"], target))
        if guarded["baselinePrediction"] is None or actual is None:
            continue
        conflict = guarded["guardApplied"]
        actual_move = math.log(actual/guarded["p0"]) if guarded["p0"] and positive(actual) else None
        e_move = math.log(guarded["baselinePrediction"]/guarded["p0"]) if guarded["p0"] else None
        rows.append({**guarded, "actual": actual, "actualLogChange": math.log(actual/guarded["p0"]) if guarded["p0"] else None,
                     "countertrendCollision": conflict,
                     "actualMovedWithMomentum7": (actual_move * guarded["momentum7Log"] > 0
                                                    if actual_move is not None and guarded["momentum7Log"] not in (None, 0) else None),
                     "eDirectionCorrect": (e_move * actual_move > 0
                                           if e_move is not None and actual_move is not None and e_move != 0 and actual_move != 0 else None)})
    # Diagnoses are descriptive: no parameter/selection is made from these targets.
    table = []
    for product in GUARD_PRODUCTS:
        for block, start, end in (("train_2025q4", dt.date(2025,10,1), dt.date(2025,12,31)),
                                  ("train_2026jan_mar20", dt.date(2026,1,1), prior.TRAIN_END)):
            subset = [r for r in rows if r["product"] == product and start <= dt.date.fromisoformat(r["target"]) <= end]
            for h in range(4, 11):
                for collision in (False, True):
                    group = [r for r in subset if r["horizon"] == h and r["countertrendCollision"] is collision]
                    e, c, y = ([r["baselinePrediction"] for r in group], [r["model"]["prediction"] for r in group], [r["actual"] for r in group])
                    table.append({"product":product,"block":block,"horizon":h,"countertrendCollision":collision,
                                  "n":len(group),"distinctOrigins":len({r["origin"] for r in group}),
                                  "distinctTargets":len({r["target"] for r in group}),
                                  "actualMovedWithMomentum7N":sum(r["actualMovedWithMomentum7"] is True for r in group),
                                  "eDirectionCorrectN":sum(r["eDirectionCorrect"] is True for r in group),
                                  "eMae":_mean([abs(a-b) for a,b in zip(e,y)]),"cMae":_mean([abs(a-b) for a,b in zip(c,y)]),
                                  "eMape":_mean([100*abs(a-b)/b for a,b in zip(e,y)]),"cMape":_mean([100*abs(a-b)/b for a,b in zip(c,y)]),
                                  "fallbackReasons":{reason:sum(r["fallbackReason"]==reason for r in group)
                                                     for reason in sorted({r["fallbackReason"] for r in group if r["fallbackReason"]})}})
    monthly=[]
    for p in GUARD_PRODUCTS:
        for month in sorted({r["target"][:7] for r in rows if r["product"]==p}):
            group=[r for r in rows if r["product"]==p and r["target"].startswith(month)]
            monthly.append({"product":p,"month":month,"n":len(group),"collisionN":sum(r["countertrendCollision"] for r in group),
                            "eMae":_mean([abs(r["baselinePrediction"]-r["actual"]) for r in group]),
                            "cMae":_mean([abs(r["model"]["prediction"]-r["actual"]) for r in group]),
                            "eMape":_mean([100*abs(r["baselinePrediction"]-r["actual"])/r["actual"] for r in group]),
                            "cMape":_mean([100*abs(r["model"]["prediction"]-r["actual"])/r["actual"] for r in group])})
    result = {"schemaVersion": 1, "phase": "descriptive_train_diagnostic", "targetValuesAttached": True,
              "trainingRange": [prior.TRAIN_START.isoformat(), prior.TRAIN_END.isoformat()],
              "inputsSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256,
                               "priorSelection": SELECTION_SHA256, "priorConfirmationManifest": CONFIRMATION_MANIFEST_SHA256,
                               "priorConfirmationReport": CONFIRMATION_REPORT_SHA256, "priorFinalManifest": FINAL_MANIFEST_SHA256},
              "implementationSha256": _implementation_hashes(), "summaryByCropBlockHorizonAndCollision": table,
              "monthlySummary":monthly,"rows": rows}
    result["diagnosticSha256"] = sha(canonical(result))
    return result


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def _build_period(prior_doc, period):
    rows = [_guard_forecast(row) for row in _period_rows(prior_doc, period)]
    key = "confirmationForecasts" if period == "confirmation" else "finalForecasts"
    doc = {"schemaVersion": 1, "phase": f"countertrend_{period}_forecast_only", "targetValuesAttached": False,
           "period": period, "priorManifestSha256": CONFIRMATION_MANIFEST_SHA256 if period == "confirmation" else FINAL_MANIFEST_SHA256,
           "inputSha256": {"baselineManifest": MANIFEST_SHA256, "prices": PRICES_SHA256,
                           "priorSelection": SELECTION_SHA256, "priorConfirmationManifest": CONFIRMATION_MANIFEST_SHA256,
                           "priorConfirmationReport": CONFIRMATION_REPORT_SHA256, "priorFinalManifest": FINAL_MANIFEST_SHA256},
           "implementationSha256": _implementation_hashes(), key: rows}
    doc["manifestSha256"] = sha(canonical(doc))
    return doc


def build_confirmation_manifest(manifest_raw, prices_raw, selection_raw, confirmation_raw, confirmation_report_raw, final_raw):
    _, _, _, conf, _, _ = _verify_sources(manifest_raw, prices_raw, selection_raw, confirmation_raw, confirmation_report_raw, final_raw)
    return _build_period(conf, "confirmation")


def _validate_guard_manifest(doc, period):
    key = "confirmationForecasts" if period == "confirmation" else "finalForecasts"
    _validate_digest(doc, "manifestSha256", f"countertrend_{period}_forecast_only")
    rows = doc.get(key)
    if not isinstance(rows, list): raise ValueError("missing guarded forecast rows")
    seen=set()
    for row in rows:
        ident=(row.get("product"),row.get("target"),row.get("horizon"))
        if ident in seen or positive(row.get("model",{}).get("prediction")) is None and row.get("baselinePrediction") is not None:
            raise ValueError("invalid or duplicate guarded row")
        seen.add(ident)
    return rows


def _score(rows, prices, period, selected):
    start,end=(CONF_START,CONF_END) if period=="confirmation" else (FINAL_START,FINAL_END)
    actuals=prior._actuals(prices,start,end)
    cell_metrics=[]; dm=[]; coverage={}; monthly={}
    for product in PRODUCTS:
        for h in HORIZONS:
            cell=[r for r in rows if r["product"]==product and r["horizon"]==h]
            pairs=[(r,actuals.get((product,dt.date.fromisoformat(r["target"])))) for r in cell]
            pairs=[(r,a) for r,a in pairs if positive(a) and positive(r.get("baselinePrediction")) and positive(r.get("model",{}).get("prediction"))]
            eae=[abs(r["baselinePrediction"]-a) for r,a in pairs]; cae=[abs(r["model"]["prediction"]-a) for r,a in pairs]
            eape=[100*x/a for x,(_r,a) in zip(eae,pairs)]; cape=[100*x/a for x,(_r,a) in zip(cae,pairs)]
            cell_metrics.append({"product":product,"horizon":h,"n":len(pairs),"eMae":_mean(eae),"cMae":_mean(cae),"eMape":_mean(eape),"cMape":_mean(cape)})
            coverage[f"{product}/{h}"]={"planned":len(cell),"eIssued":sum(positive(r.get("baselinePrediction")) is not None for r in cell),
                "cIssued":sum(positive(r.get("model",{}).get("prediction")) is not None for r in cell),"commonPairs":len(pairs),
                "coveragePreserved":all((positive(r.get("baselinePrediction")) is None)==(positive(r.get("model",{}).get("prediction")) is None) for r in cell),
                "positiveActualTargets":len({r["target"] for r in cell if positive(actuals.get((product,dt.date.fromisoformat(r["target"])))) is not None}),
                "distinctOrigins":len({r["origin"] for r in cell}),"distinctTargets":len({r["target"] for r in cell}),
                "guardActivations":sum(r.get("guardApplied",False) for r in cell)}
            for loss in ("ae","ape"):
                for bw in BANDWIDTHS:
                    if product not in GUARD_PRODUCTS or product not in selected:
                        test={"status":"not_tested_by_protocol","reason":"crop_not_selected_by_confirmation","n":len(pairs),"statistic":None,"meanLossDifference":None,"pValue":1.0}
                    else:
                        diffs={dt.date.fromisoformat(r["target"]):(abs(r["model"]["prediction"]-a)-abs(r["baselinePrediction"]-a) if loss=="ae" else (abs(r["model"]["prediction"]-a)-abs(r["baselinePrediction"]-a))/a) for r,a in pairs}
                        test=scoring._dm(diffs,bw)
                    dm.append({"product":product,"horizon":h,"loss":loss,"bandwidth":bw,**test})
            bymonth={}
            for r in cell:bymonth.setdefault(r["target"][:7],[]).append(r)
            for month,mrows in bymonth.items():
                mp=[(r,actuals.get((product,dt.date.fromisoformat(r["target"])))) for r in mrows]
                mp=[(r,a) for r,a in mp if positive(a) and positive(r.get("baselinePrediction")) and positive(r.get("model",{}).get("prediction"))]
                monthly[f"{product}/{h}/{month}"]={"planned":len(mrows),"commonPairs":len(mp),"eMae":_mean([abs(r["baselinePrediction"]-a) for r,a in mp]),
                    "cMae":_mean([abs(r["model"]["prediction"]-a) for r,a in mp]),"eMape":_mean([100*abs(r["baselinePrediction"]-a)/a for r,a in mp]),
                    "cMape":_mean([100*abs(r["model"]["prediction"]-a)/a for r,a in mp]),
                    "guardActivations":sum(r.get("guardApplied",False) for r in mrows),
                    "positiveActualTargets":len({r["target"] for r in mrows if positive(actuals.get((product,dt.date.fromisoformat(r["target"])))) is not None})}
    for bw in BANDWIDTHS:
        family=[r for r in dm if r["bandwidth"]==bw and r["product"] in GUARD_PRODUCTS]
        prior._holm(family)
    macro=[]
    for p in PRODUCTS:
        cs=[x for x in cell_metrics if x["product"]==p and x["n"]]
        macro.append({"product":p,"cells":len(cs),**{k:_mean([x[k] for x in cs]) for k in ("eMae","cMae","eMape","cMape")}})
    supported=[x for x in cell_metrics if x["n"]]
    overall={"cellsIncluded":len(supported),**{k:_mean([x[k] for x in supported]) for k in ("eMae","cMae","eMape","cMape")}}
    return {"metrics":cell_metrics,"macro":macro,"overall40CellMacro":overall,"dm":dm,"coverage":coverage,"monthly":monthly,
            "collisionCounts":{"all":sum(r.get("guardApplied",False) for r in rows),
                               "byProduct":{p:sum(r.get("guardApplied",False) for r in rows if r["product"]==p) for p in PRODUCTS}}}


def score_confirmation(manifest_raw, prices_raw, selection_raw, confirmation_raw, confirmation_report_raw, final_raw, candidate_manifest):
    expected=build_confirmation_manifest(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw)
    if canonical(expected)!=canonical(candidate_manifest): raise ValueError("confirmation forecast replay mismatch")
    rows=_validate_guard_manifest(candidate_manifest,"confirmation")
    _,prices,*_=_verify_sources(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw)
    scored=_score(rows,prices,"confirmation",GUARD_PRODUCTS)
    result={"schemaVersion":1,"phase":"countertrend_confirmation_scored","targetRange":[CONF_START.isoformat(),CONF_END.isoformat()],
            "inputsSha256":candidate_manifest["inputSha256"],"implementationSha256":_implementation_hashes(),**scored}
    result["decisions"]=_decisions(scored)
    result["reportSha256"]=sha(canonical(result))
    return result


def _decisions(scored):
    decisions=[]
    for p in GUARD_PRODUCTS:
        row=next(x for x in scored["macro"] if x["product"]==p)
        coverage=all(scored["coverage"][f"{p}/{h}"]["coveragePreserved"] for h in HORIZONS)
        passed=row["cells"]==len(HORIZONS) and coverage and row["cMae"] is not None and row["cMape"] is not None and row["cMae"]<row["eMae"] and row["cMape"]<row["eMape"]
        decisions.append({"product":p,"candidateMae":row["cMae"],"priorMae":row["eMae"],"candidateMape":row["cMape"],"priorMape":row["eMape"],"coveragePreserved":coverage,
                          "decision":"freeze_for_final_period" if passed else "retain_prior_E"})
    return decisions


def build_final_manifest(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw,confirmation_result):
    expected=score_confirmation(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw,
                                build_confirmation_manifest(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw))
    if canonical(expected)!=canonical(confirmation_result):raise ValueError("confirmation result replay mismatch")
    selected=[r["product"] for r in confirmation_result["decisions"] if r["decision"]=="freeze_for_final_period"]
    if not selected:raise ValueError("confirmation gate passed for no crop; stop before final-period forecasts")
    *_,prior_final=_verify_sources(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw)
    rows=[]
    for row in _period_rows(prior_final,"final"):
        guarded=_guard_forecast(row)
        if row["product"] not in selected:
            guarded["model"]["prediction"]=guarded["baselinePrediction"]
            guarded["guardApplied"]=False;guarded["fallbackReason"]="crop_not_selected_by_confirmation"
        rows.append(guarded)
    doc={"schemaVersion":1,"phase":"countertrend_final_forecast_only","targetValuesAttached":False,"frozenCrops":selected,
         "confirmationReportSha256":confirmation_result["reportSha256"],
         "inputSha256":confirmation_result["inputsSha256"],"implementationSha256":_implementation_hashes(),"finalForecasts":rows}
    doc["manifestSha256"]=sha(canonical(doc));return doc


def score_final(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw,confirmation_result,final_manifest):
    expected=build_final_manifest(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw,confirmation_result)
    if canonical(expected)!=canonical(final_manifest):raise ValueError("final forecast replay mismatch")
    rows=_validate_guard_manifest(final_manifest,"final")
    _,prices,*_=_verify_sources(manifest_raw,prices_raw,selection_raw,confirmation_raw,confirmation_report_raw,final_raw)
    scored=_score(rows,prices,"final",set(final_manifest["frozenCrops"]))
    # Retention is a decision about the next research candidate only. It is
    # deliberately separate from the already scored C-vs-E policy output.
    retention=[]
    for p in GUARD_PRODUCTS:
        row=next(x for x in scored["macro"] if x["product"]==p)
        keys=[scored["coverage"][f"{p}/{h}"] for h in HORIZONS]
        full_e_keys=all(k["coveragePreserved"] for k in keys)
        has_candidate=p in final_manifest["frozenCrops"]
        passed=(has_candidate and row["cells"]==len(HORIZONS) and full_e_keys and row["cMae"] is not None and row["cMape"] is not None
                and row["cMae"]<row["eMae"] and row["cMape"]<row["eMape"])
        retention.append({"product":p,"candidateEvaluated":has_candidate,"allPriorEKeysPreserved":full_e_keys,
                          "candidateMae":row["cMae"],"priorMae":row["eMae"],"candidateMape":row["cMape"],"priorMape":row["eMape"],
                          "retainedCandidate":"C" if passed else "E"})
    result={"schemaVersion":1,"phase":"countertrend_final_scored","targetRange":[FINAL_START.isoformat(),FINAL_END.isoformat()],"frozenCrops":final_manifest["frozenCrops"],**scored,
            "nextResearchCandidateRetention":retention,
            "policyForecastInterpretation":"These final scores compare the frozen confirmation-selected C policy with E. Retention decisions are separate and do not rewrite these forecasts or metrics.",
            "retrospectiveLimitation":"All periods are from an already inspected historical archive and this is a continuing sequence of retrospective candidate decisions, not independent prospective validation."}
    result["reportSha256"]=sha(canonical(result));return result


def write_exclusive(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(canonical(value))


def run(output_dir=OUT):
    """Run replay-gated stages; source review is required before execution."""
    raws=(MANIFEST.read_bytes(),PRICES.read_bytes(),PRIOR_SELECTION.read_bytes(),PRIOR_CONFIRMATION.read_bytes(),
          PRIOR_CONFIRMATION_REPORT.read_bytes(),PRIOR_FINAL.read_bytes())
    diagnostic=build_train_diagnostic(*raws)
    write_exclusive(Path(output_dir)/"train-diagnostic.json",diagnostic)
    candidate=build_confirmation_manifest(*raws)
    write_exclusive(Path(output_dir)/"confirmation-manifest.json",candidate)
    confirmation=score_confirmation(*raws,candidate)
    write_exclusive(Path(output_dir)/"confirmation-report.json",confirmation)
    selected=[x["product"] for x in confirmation["decisions"] if x["decision"]=="freeze_for_final_period"]
    result={"outputDir":str(Path(output_dir)),"confirmationDecisions":confirmation["decisions"],"finalScored":False}
    if selected:
        final=build_final_manifest(*raws,confirmation)
        write_exclusive(Path(output_dir)/"final-manifest.json",final)
        report=score_final(*raws,confirmation,final)
        write_exclusive(Path(output_dir)/"final-report.json",report)
        result.update({"finalScored":True,"finalRetention":report["nextResearchCandidateRetention"]})
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    command=sub.add_parser("run",help="run frozen diagnostic and replay-gated confirmation/final scoring")
    command.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args(argv)
    if args.command=="run":return run(args.output_dir)
    return 2


if __name__=="__main__":
    raise SystemExit(main())
