"""Frozen garlic-only R106-to-MLP16 replacement with answer-free phase manifests.

This module does not run on import. Preflight checks lineage and training
support without fitting. Build fits only the predeclared monthly models and
checkpoints each fit. Verify replays saved inference states without refitting;
score attaches the frozen price truths after answer-free manifests are saved.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

from . import preserved_feature_forecasts as pf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/e106-mlp16-20261011"
CONF_BASE = ROOT / "outputs/experiments/preserved-improvement-composition-20261011/confirmation-manifest.json"
FINAL_BASE = ROOT / "outputs/experiments/preserved-improvement-composition-20261011/final-manifest.json"
R106 = pf.EVALUATION_OUT
PRICES = pf.PRICES_PATH
PRODUCT = "garlic"
GARLIC_ID = "garlic"
PHASES = {"confirmation": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
          "final": (dt.date(2026, 7, 11), dt.date(2026, 9, 27))}
ORIGIN_START = {"confirmation": dt.date(2026, 4, 1), "final": dt.date(2026, 7, 1)}
HORIZONS = tuple(range(1, 11))
MODEL_PARAMS = {"hidden_layer_sizes": [16], "activation": "relu", "solver": "adam", "alpha": 10.0,
                "batch_size": 256, "learning_rate_init": 0.001, "max_iter": 40, "tol": 0.0,
                "n_iter_no_change": 41, "early_stopping": False, "shuffle": True, "random_state": 0}
SCHEMA = "e106-mlp16-v1"
PINNED_SHA256 = {
    "prices": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
    "r106": "35164a11143768f26e7567e80643ec323c7279083ac5e6ef166a7fe8d185aa20",
    "preservedFeatureHelper": "f7bee365ce894f7046904a64f575bb64c80099a483d12a2b0016417a8a84c49e",
    "confirmationE106": "1c32c11f0db65c1f06724cf8ae4ae02bd3455eedb1bafcdf827c9b0fef7e473b",
    "finalE106": "c45cc99de60cc9df752194e694c4de8984d424c4edc03de2e283c23ddea9bbe2",
}


def canonical(v: Any) -> bytes:
    return (json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(v: bytes) -> str: return hashlib.sha256(v).hexdigest()
def file_sha(p: Path) -> str: return sha(p.read_bytes())
def digest(doc: dict[str, Any]) -> str:
    x = dict(doc); x.pop("manifestSha256", None)
    return sha(canonical(x).rstrip(b"\n"))


def _check_pinned_inputs() -> None:
    actual = {"prices": file_sha(PRICES), "r106": file_sha(R106),
              "preservedFeatureHelper": file_sha(Path(pf.__file__)),
              "confirmationE106": file_sha(CONF_BASE), "finalE106": file_sha(FINAL_BASE)}
    for label, expected in PINNED_SHA256.items():
        if actual[label] != expected: raise ValueError(f"pinned {label} SHA-256 mismatch")
    # Check the transitive R106 feature stack (direct, modeling, rich features,
    # full-history library and online baseline) against its frozen hashes too.
    pf._check_inputs()


def keys(phase: str) -> list[tuple[str, dt.date, int]]:
    # The frozen E106 manifests contain only the four crops and eligible keys.
    # This grid is the authority: never synthesize or drop scored keys.
    doc = _base_doc(phase)
    rows = doc.get("forecasts")
    if not isinstance(rows, list): raise ValueError("E106 source lacks forecast grid")
    out = [(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]) for r in rows]
    if len(out) != len(set(out)) or any(p not in pf.PRODUCTS or h not in HORIZONS for p, _, h in out):
        raise ValueError("invalid or duplicate E106 source key")
    return out


def _base_doc(phase: str) -> dict[str, Any]:
    path = CONF_BASE if phase == "confirmation" else FINAL_BASE
    doc = json.loads(path.read_text())
    expected_phase = "preserved_improvement_confirmation_forecast_only" if phase == "confirmation" else "preserved_improvement_final_forecast_only"
    if doc.get("targetValuesAttached") is not False or doc.get("phase") != expected_phase or doc.get("phaseName") != phase:
        raise ValueError("E106 source must be an answer-free saved manifest")
    if doc.get("manifestSha256") != digest(doc): raise ValueError("E106 source digest mismatch")
    return doc


def _idx(rows: list[dict[str, Any]]) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    out = {}
    for r in rows:
        k = (r["product"], dt.date.fromisoformat(r["target"]), r["horizon"])
        if k in out: raise ValueError("duplicate key in saved manifest")
        out[k] = r
    return out


def _r106_index() -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    doc = json.loads(R106.read_text())
    pf._validate_manifest(doc, "evaluation")
    return {(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]): r for r in doc["forecasts"]}


def _identity_rows(phase: str, base: dict[tuple[str, dt.date, int], dict[str, Any]], r106: dict) -> list[dict[str, Any]]:
    expected = set(keys(phase))
    if set(base) != expected: raise ValueError(f"saved E106 {phase} grid mismatch")
    if not expected.issubset(r106): raise ValueError("R106 missing keys")
    out = []
    for p, day, h in keys(phase):
        e = base[(p, day, h)]
        origin = day - dt.timedelta(days=h)
        if e.get("origin") != origin.isoformat(): raise ValueError("E106 origin mismatch")
        eligible = True
        if e.get("eligible") is not True: raise ValueError("E106 source includes a noneligible scoring key")
        if p != PRODUCT:
            # Preserve the saved E106 policy output exactly, including null state/reason.
            out.append({"product": p, "target": day.isoformat(), "origin": origin.isoformat(), "horizon": h,
                        "eligible": eligible, "status": e["candidate"].get("status"),
                        "prediction": e["candidate"].get("prediction"), "reason": e["candidate"].get("reason"),
                        "usedMLP": False, "fallbackReason": "non_garlic_exact_E106"})
        else:
            out.append({"product": p, "target": day.isoformat(), "origin": origin.isoformat(), "horizon": h,
                        "eligible": eligible, "status": e["candidate"].get("status"),
                        "prediction": e["candidate"].get("prediction"), "reason": e["candidate"].get("reason"),
                        "usedMLP": False, "fallbackReason": None if eligible else "origin_before_phase_start"})
    return out


def _frozen_fit_audit(r106: dict, horizon: int, month_keys: list[tuple[str,dt.date,int]]) -> tuple[dict[str,Any],dt.date]:
    frozen = next((r106[k].get("models", {}).get("preserved106", {}).get("training") for k in month_keys
                   if isinstance(r106.get(k,{}).get("models", {}).get("preserved106", {}).get("training"), dict)), None)
    if frozen is None: raise ValueError(f"missing R106 audit garlic/h{horizon}")
    origin = dt.date.fromisoformat(frozen["fitOrigin"])
    if any(r106[k]["models"]["preserved106"]["training"].get("fitOrigin") != frozen["fitOrigin"] for k in month_keys):
        raise ValueError(f"inconsistent frozen R106 fitOrigin garlic/h{horizon}")
    return frozen, origin


def _fallback(base: dict[str,Any], reason: str) -> dict[str,Any]:
    return {"status":base.get("status"),"prediction":base.get("prediction"),"reason":base.get("reason"),
            "usedMLP":False,"fallbackReason":reason}


def _primary_pass(macro: dict[str,float], all_baseline_keys_retained: bool, minimum_pairs: int) -> bool:
    return macro["deltaMape"] < 0 and macro["deltaMae"] <= 0 and all_baseline_keys_retained and minimum_pairs >= 30


def _checkpoint_path(phase: str, horizon: int, month: tuple[int,int]) -> Path:
    return OUT / ".checkpoints" / phase / f"garlic-h{horizon}-{month[0]}-{month[1]:02d}.json"


def _load_fit_checkpoint(path: Path, expected: dict[str,Any]) -> dict[str,Any] | None:
    if not path.exists(): return None
    doc=json.loads(path.read_text())
    if doc.get("checkpointSha256") != digest({k:v for k,v in doc.items() if k!="checkpointSha256"}):
        raise ValueError(f"monthly fit checkpoint digest mismatch: {path}")
    if any(doc.get(k)!=v for k,v in expected.items()):
        raise ValueError(f"monthly fit checkpoint source/training mismatch: {path}")
    return doc.get("fitAudit")


def _write_fit_checkpoint(path: Path, fit_audit: dict[str,Any], expected: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    doc={**expected,"fitAudit":fit_audit};doc["checkpointSha256"]=digest(doc)
    with path.open("xb") as stream: stream.write(canonical(doc))


def _replay_prediction(fit: dict[str,Any], vector: list[float], p0: float, sarima: float) -> tuple[float,float]:
    if len(vector)!=106 or len(fit.get("scalerMean",[]))!=106 or len(fit.get("scalerScale",[]))!=106:
        raise ValueError("saved inference state has invalid feature width")
    if fit.get("coefShapes") != [[106,16],[16,1]]: raise ValueError("unexpected fitted MLP matrix shapes")
    z=[(float(v)-mu)/scale for v,mu,scale in zip(vector,fit["scalerMean"],fit["scalerScale"])]
    hidden=[max(0.0, math.fsum(z[i]*fit["coefs"][0][i*16+j] for i in range(106))+fit["intercepts"][0][j]) for j in range(16)]
    log_ratio=math.fsum(hidden[j]*fit["coefs"][1][j] for j in range(16))+fit["intercepts"][1][0]
    standalone=p0*math.exp(log_ratio)
    return standalone,.5*standalone+.5*sarima


def _dm_family(pairs_by_h: dict[int,list[dict[str,Any]]]) -> list[dict[str,Any]]:
    from . import equal_info_online_score as stats
    dm=[]
    for h in HORIZONS:
        xs=pairs_by_h[h]
        for loss in ("ae","ape"):
            diffs={dt.date.fromisoformat(x["target"]):(abs(x["candidate"]-x["actual"])-abs(x["E106"]-x["actual"])
                    if loss=="ae" else (abs(x["candidate"]-x["actual"])-abs(x["E106"]-x["actual"]))/x["actual"]) for x in xs}
            for bw in (9,28): dm.append({"horizon":h,"loss":loss,"bandwidth":bw,**stats._dm(diffs,bw)})
    if len(dm)!=40 or any(sum(x["bandwidth"]==bw for x in dm)!=20 for bw in (9,28)):
        raise ValueError("fixed garlic inference families must contain 20 tests per bandwidth")
    for bw in (9,28): stats._holm([x for x in dm if x["bandwidth"]==bw])
    return dm


def _make_features(phase: str, base_index: dict, r106: dict):
    # Use the exact archived prices and preserved-feature implementation.
    prices = json.loads(PRICES.read_text())
    end = PHASES[phase][1]
    trimmed, _ = pf._trim_price_snapshot(prices, end)
    direct = pf.direct_horizon._observations(trimmed)
    raw_maps, _ = pf.rich._raw_maps(trimmed)
    series = direct[PRODUCT]
    records = {PRODUCT: {d: dict(raw_maps[PRODUCT][d]) for d in series}}
    price_maps = {PRODUCT: {d: v[0] for d, v in series.items()}}
    ordered = {PRODUCT: sorted(series)}
    cache = {}
    all_phase = [k for k in keys(phase) if k[0] == PRODUCT]
    by_h = {h: [k for k in all_phase if k[2] == h] for h in HORIZONS}
    features = {h: {} for h in HORIZONS}
    targets = sorted(set(series) | {k[1] for k in all_phase})
    for h in HORIZONS:
        for t in targets:
            if t < pf.online.narrow.TRAIN_START and (PRODUCT, t, h) not in set(all_phase): continue
            row = pf._feature_pair(PRODUCT, t, h, {PRODUCT: series}, records, {PRODUCT: cache}, ordered, price_maps)
            features[h][t] = row
    return series, features, by_h


def _build(phase: str, progress=None, confirmation_report: Path | None = None) -> dict[str, Any]:
    if phase not in PHASES: raise ValueError("phase must be confirmation or final")
    import numpy as np
    import sklearn
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    if sklearn.__version__ != "1.9.1": raise RuntimeError(f"requires scikit-learn 1.9.1, found {sklearn.__version__}")
    _check_pinned_inputs()
    base_doc = _base_doc(phase)
    confirmation_lineage = None
    if phase == "final":
        report_path = confirmation_report or OUT / "confirmation-report.json"
        confirmation = _confirmation_gate(report_path)
        confirmation_lineage = {"reportFileSha256": file_sha(report_path),
                                "manifestFileSha256": confirmation["manifestFileSha256"],
                                "primaryDecision": confirmation["primaryDecision"]}
    bkey = "forecasts"
    base_index = _idx(base_doc[bkey]); r106 = _r106_index()
    rows = _identity_rows(phase, base_index, r106)
    series, features, by_h = _make_features(phase, base_index, r106)
    row_index = {(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]): r for r in rows}
    fit_audits = []
    for h in HORIZONS:
        phase_keys = by_h[h]
        monthly = {}
        for k in phase_keys:
            origin = k[1] - dt.timedelta(days=h)
            monthly.setdefault((origin.year, origin.month), []).append(k)
        if len(monthly) != 3: raise ValueError(f"expected exactly three monthly fits for garlic/h{h}; found {len(monthly)}")
        # Train all rows whose target precedes the first eligible origin in each month.
        for month in sorted(monthly):
            mk = monthly[month]
            frozen, fit_origin = _frozen_fit_audit(r106, h, mk)
            training = [t for t in series if t >= pf.online.narrow.TRAIN_START and t < fit_origin and features[h].get(t) is not None]
            audit = pf._fit_audit(training, fit_origin)
            pf._assert_training_support(audit, frozen, PRODUCT, h, month)
            # Explicit fingerprints bind target keys, matrix and target vector.
            x = np.asarray([features[h][t]["preserved106"] for t in training], dtype=np.float64)
            y = np.asarray([math.log(series[t][0] / features[h][t]["p0"]) for t in training], dtype=np.float64)
            keys_hash = sha(canonical([t.isoformat() for t in training]))
            matrix_hash = sha(x.tobytes(order="C")); target_hash = sha(y.tobytes(order="C"))
            fit_id = f"garlic-h{h}-{month[0]}-{month[1]:02d}"
            checkpoint_path=_checkpoint_path(phase,h,month)
            checkpoint_expected={"phase":phase,"fitId":fit_id,"experimentModuleSha256":file_sha(Path(__file__)),
                "r106Sha256":file_sha(R106),"e106Sha256":file_sha(CONF_BASE if phase=="confirmation" else FINAL_BASE),
                "pricesSha256":file_sha(PRICES),"trainingKeysSha256":keys_hash,"trainingMatrixSha256":matrix_hash,
                "trainingTargetSha256":target_hash,"fitOrigin":fit_origin.isoformat()}
            prov=_load_fit_checkpoint(checkpoint_path,checkpoint_expected)
            if prov is None:
                scaler = model = None; caught = []; fit_failure = None; scaler_hash = None; elapsed = 0.0
                if len(training) < pf.MIN_TRAIN:
                    fit_failure = "fewer_than_90_valid_training_targets"
                else:
                    fit_start = time.monotonic()
                    try:
                        scaler = StandardScaler()
                        xz = scaler.fit_transform(x)
                        scaler_hash = sha(scaler.mean_.tobytes() + scaler.scale_.tobytes() + scaler.var_.tobytes())
                        model = MLPRegressor(**MODEL_PARAMS)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(xz, y)
                        numeric_state=[*scaler.mean_,*scaler.scale_,*scaler.var_,*model.loss_curve_,model.loss_,
                                       *(v for a in model.coefs_ for v in a.ravel()),
                                       *(v for a in model.intercepts_ for v in a.ravel())]
                        if not all(math.isfinite(float(v)) for v in numeric_state):
                            raise FloatingPointError("nonfinite fitted state")
                    except Exception as exc:
                        model = None
                        fit_failure = f"model_fit_failed:{type(exc).__name__}"
                    elapsed = time.monotonic() - fit_start
                prov = {"training": audit, "trainingKeysSha256": keys_hash, "trainingMatrixSha256": matrix_hash,
                        "trainingTargetSha256": target_hash, "scalerFingerprintSha256": scaler_hash,
                        "lossCurve": [float(v) for v in model.loss_curve_] if model is not None else [],
                        "nIter": int(model.n_iter_) if model is not None else None,
                        "finalLoss": float(model.loss_) if model is not None else None,
                        "warnings": [f"{type(w.message).__name__}: {w.message}" for w in caught],
                        "fitFailure": fit_failure,
                        "trainingRows": len(training), "trainingUniqueDates": len(set(training)),
                        "fitSeconds": elapsed, "sklearnVersion": sklearn.__version__, "numpyVersion": np.__version__,
                        "model": MODEL_PARAMS, "target": "log(target/p0)", "featureCount": 106,
                        "fitId": fit_id}
                if model is not None:
                    prov["scalerMean"] = [float(v) for v in scaler.mean_]
                    prov["scalerScale"] = [float(v) for v in scaler.scale_]
                    prov["coefs"] = [[float(v) for v in layer.ravel()] for layer in model.coefs_]
                    prov["coefShapes"] = [list(layer.shape) for layer in model.coefs_]
                    prov["intercepts"] = [[float(v) for v in layer] for layer in model.intercepts_]
                    prov["inferenceStateSha256"] = sha(canonical({k:prov[k] for k in ("scalerMean","scalerScale","coefs","coefShapes","intercepts")}))
                _write_fit_checkpoint(checkpoint_path,prov,checkpoint_expected)
            fit_failure=prov.get("fitFailure")
            fit_audits.append({"horizon": h, "month": f"{month[0]}-{month[1]:02d}", **prov})
            if progress: progress(h, month, len(training))
            for k in mk:
                p, day, _ = k; fr = features[h].get(day); out = row_index[k]
                # A failed input/support/numeric condition reverts to the precise E106 output.
                base = base_index[k]["candidate"]
                reason = None
                pred = None
                if fit_failure: reason = fit_failure
                elif fr is None: reason = "feature_unavailable_or_source_too_old"
                elif r106[k]["models"]["causalSarima"].get("status") != "issued": reason = "sarima_unavailable"
                else:
                    try:
                        sarima = float(r106[k]["models"]["causalSarima"]["prediction"])
                        mlp_price, blend = _replay_prediction(prov,fr["preserved106"],fr["p0"],sarima)
                        if not all(math.isfinite(v) and v > 0 for v in (mlp_price, sarima, blend)): raise ValueError("nonfinite_or_nonpositive")
                        pred = blend
                    except Exception as exc: reason = f"numeric_or_prediction_failure:{type(exc).__name__}"
                if pred is None:
                    out.update({**_fallback(base, reason),
                                "training": audit, "fitProvenance": None})
                else:
                    out.update({"status": "issued", "prediction": pred, "reason": None, "usedMLP": True,
                                "fallbackReason": None, "training": audit, "fitId": fit_id,
                                "featureVector": [float(v) for v in fr["preserved106"]],
                                "mlpStandalonePrediction": mlp_price, "sarimaPrediction": sarima,
                                "p0": fr["p0"], "p0Date": fr["p0Date"].isoformat()})
    # Non-garlic and noneligible garlic remain exact E106; unfit eligible keys fallback to E106.
    doc = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": phase, "targetValuesAttached": False,
           "protocol": {"id": SCHEMA, "phase": phase, "products": list(pf.PRODUCTS), "horizons": list(HORIZONS),
                        "targetDates": [d.isoformat() for d in PHASES[phase]], "originStart": ORIGIN_START[phase].isoformat(),
                        "fitRule": "monthly refit at first eligible origin; training target strictly before that origin",
                        "features": "preserved R106 106-column feature rows; feature cutoff origin-1",
                        "model": MODEL_PARAMS, "scaler": "StandardScaler fit on this fit's training matrix only",
                        "blend": "0.5*MLP16 price + 0.5*identical saved R106 causalSarima; exact E106 fallback",
                        "nonGarlic": "exact saved E106 candidate", "trainingSupport": "asserted equal to saved R106 causalDirect audit",
                        "baseline": file_sha(CONF_BASE if phase == "confirmation" else FINAL_BASE),
                        "confirmationGate": confirmation_lineage,
                        "inputHashes": {"prices": file_sha(PRICES), "r106": file_sha(R106),
                                        "e106": file_sha(CONF_BASE if phase == "confirmation" else FINAL_BASE),
                                        "preservedFeatureHelper": file_sha(Path(pf.__file__)),
                                        "experimentModule": file_sha(Path(__file__))}},
           "plannedCount": len(rows), "forecasts": rows, "fitAudits": fit_audits}
    doc["manifestSha256"] = digest(doc)
    if len(fit_audits) > 30 or len(fit_audits) != 30: raise ValueError("fit budget must be exactly 30 per phase")
    return doc


def _confirmation_gate(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file(): raise ValueError("final phase requires a scored confirmation report")
    confirmation = json.loads(report_path.read_text())
    expected_manifest = OUT / "confirmation-manifest.json"
    if (confirmation.get("phase") != "confirmation" or not confirmation.get("primaryDecision", {}).get("passes")
            or not expected_manifest.is_file()
            or confirmation.get("manifestFileSha256") != file_sha(expected_manifest)):
        raise ValueError("final phase is gated on a passing, manifest-matched confirmation report")
    frozen=json.loads(expected_manifest.read_text())
    verify_sources(frozen,"confirmation")
    recomputed=_primary_only(frozen,"confirmation")
    if confirmation.get("primaryDecision") != recomputed["primaryDecision"]:
        raise ValueError("confirmation report primary decision does not replay from frozen predictions")
    for metric in ("deltaMae","deltaMape","candidateMae","E106Mae","candidateMape","E106Mape"):
        if not math.isclose(float(confirmation.get("primaryGarlic10h",{}).get(metric,float("nan"))),
                            float(recomputed["primaryGarlic10h"][metric]),rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("confirmation report primary metrics do not replay from frozen predictions")
    return {"reportFileSha256": file_sha(report_path),
            "manifestFileSha256": confirmation["manifestFileSha256"],
            "primaryDecision": confirmation["primaryDecision"]}


def _primary_only(manifest: dict[str,Any],phase: str) -> dict[str,Any]:
    """Recompute only the predeclared continuation gate; no model fit/bootstrap."""
    base=_idx(_base_doc(phase)["forecasts"]); prices=json.loads(PRICES.read_text())
    actuals=pf.direct_horizon._observations(prices)
    rows=[r for r in manifest["forecasts"] if r["product"]==PRODUCT and r["eligible"]]
    per_h=[]
    for h in HORIZONS:
        paired=[]
        for r in rows:
            if r["horizon"]!=h: continue
            d=dt.date.fromisoformat(r["target"]); y=actuals.get(PRODUCT,{}).get(d)
            b=base[(PRODUCT,d,h)]["candidate"].get("prediction")
            if y is not None and b is not None and r.get("status")=="issued" and r.get("prediction") is not None:
                paired.append((float(r["prediction"]),float(b),float(y[0])))
        per_h.append({"n":len(paired),"candidateMae":math.fsum(abs(a-y) for a,b,y in paired)/len(paired) if paired else None,
                      "E106Mae":math.fsum(abs(b-y) for a,b,y in paired)/len(paired) if paired else None,
                      "candidateMape":100*math.fsum(abs(a-y)/y for a,b,y in paired)/len(paired) if paired else None,
                      "E106Mape":100*math.fsum(abs(b-y)/y for a,b,y in paired)/len(paired) if paired else None})
    if any(x["candidateMae"] is None for x in per_h): raise ValueError("confirmation has unsupported primary horizon")
    macro={k:math.fsum(x[k] for x in per_h)/10 for k in ("candidateMae","E106Mae","candidateMape","E106Mape")}
    macro["deltaMae"]=macro["candidateMae"]-macro["E106Mae"];macro["deltaMape"]=macro["candidateMape"]-macro["E106Mape"]
    baseline_coverage=all(base[(PRODUCT,dt.date.fromisoformat(r["target"]),r["horizon"])]["candidate"].get("status")!="issued" or r.get("status")=="issued" for r in rows)
    min_pairs=min(x["n"] for x in per_h)
    return {"primaryGarlic10h":macro,"primaryDecision":{"passes":_primary_pass(macro,baseline_coverage,min_pairs),
        "deltaMapeNegative":macro["deltaMape"]<0,"deltaMaeNonpositive":macro["deltaMae"]<=0,
        "allEligibleE106IssuedKeysRetained":baseline_coverage,"atLeast30PairsAtEveryHorizon":min_pairs>=30}}


def _preflight(phase: str, confirmation_report: Path | None = None) -> dict[str,Any]:
    """Check source lineage, exact keys, feature support and fit budget without model fitting."""
    if phase not in PHASES: raise ValueError("phase must be confirmation or final")
    _check_pinned_inputs()
    gate=None
    if phase=="final": gate=_confirmation_gate(confirmation_report or OUT/"confirmation-report.json")
    base=_idx(_base_doc(phase)["forecasts"]);r106=_r106_index()
    _identity_rows(phase,base,r106)
    series,features,by_h=_make_features(phase,base,r106)
    fits=[]
    for h,phase_keys in by_h.items():
        monthly={}
        for k in phase_keys:
            origin=k[1]-dt.timedelta(days=h);monthly.setdefault((origin.year,origin.month),[]).append(k)
        if len(monthly)!=3: raise ValueError(f"expected exactly three monthly fits for garlic/h{h}; found {len(monthly)}")
        for month,mk in sorted(monthly.items()):
            frozen,fit_origin=_frozen_fit_audit(r106,h,mk)
            training=[t for t in series if t>=pf.online.narrow.TRAIN_START and t<fit_origin and features[h].get(t) is not None]
            audit=pf._fit_audit(training,fit_origin);pf._assert_training_support(audit,frozen,PRODUCT,h,month)
            fits.append({"horizon":h,"month":f"{month[0]}-{month[1]:02d}","fitOrigin":fit_origin.isoformat(),"trainingRows":len(training),
                         "firstTarget":audit["firstTarget"],"lastTarget":audit["lastTarget"]})
    if len(fits)!=30: raise ValueError("fit budget preflight expected exactly 30 monthly garlic fits")
    return {"phase":phase,"eligibleSourceKeys":len(base),"products":sorted({k[0] for k in base}),"fits":fits,
            "maximumFits":30,"confirmationGate":gate,"modelFitExecuted":False}


def validate(doc: dict[str, Any], phase: str) -> None:
    if doc.get("phase") != "forecast_only" or doc.get("phaseName") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("invalid answer-free manifest")
    if doc.get("manifestSha256") != digest(doc): raise ValueError("manifest digest mismatch")
    expected = set(keys(phase)); seen = set()
    for r in doc.get("forecasts", []):
        d = dt.date.fromisoformat(r["target"]); h = r["horizon"]; k = (r["product"], d, h)
        if k not in expected or k in seen or dt.date.fromisoformat(r["origin"]) != d-dt.timedelta(days=h): raise ValueError("bad forecast identity")
        if r["status"] == "issued":
            v = r.get("prediction")
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0: raise ValueError("bad prediction")
        elif r.get("prediction") is not None: raise ValueError("unissued forecast has prediction")
        seen.add(k)
    if seen != expected or doc.get("plannedCount") != len(expected): raise ValueError("forecast grid incomplete")


def verify_sources(doc: dict[str, Any], phase: str) -> None:
    """Verify immutable source lineage and deterministic policy invariants without refitting."""
    validate(doc, phase)
    _check_pinned_inputs()
    expected = {"prices": file_sha(PRICES), "r106": file_sha(R106),
                "e106": file_sha(CONF_BASE if phase == "confirmation" else FINAL_BASE),
                "preservedFeatureHelper": file_sha(Path(pf.__file__)),
                "experimentModule": file_sha(Path(__file__))}
    if doc.get("protocol", {}).get("inputHashes") != expected:
        raise ValueError("manifest input/source hash mismatch")
    base = _idx(_base_doc(phase)["forecasts"])
    r106 = _r106_index()
    _,features,_ = _make_features(phase,base,r106)
    fits = {x.get("fitId"):x for x in doc.get("fitAudits", []) if isinstance(x, dict)}
    expected_fit_ids={f"garlic-h{h}-{year}-{month:02d}" for h in HORIZONS
                      for year,month in {(dt.date.fromisoformat(r["origin"]).year,dt.date.fromisoformat(r["origin"]).month)
                                         for r in doc["forecasts"] if r["product"]==PRODUCT}}
    if len(fits) != len(doc.get("fitAudits", [])) or set(fits)!=expected_fit_ids or len(fits)!=30:
        raise ValueError("manifest must contain exactly the 30 expected monthly garlic fits")
    for fit_id, fit in fits.items():
        if fit.get("model") != MODEL_PARAMS or fit.get("featureCount") != 106:
            raise ValueError("fit audit model configuration mismatch")
        h = fit.get("horizon"); month = fit.get("month")
        day = next((k[1] for k in r106 if k[0] == PRODUCT and k[2] == h
                    and (k[1]-dt.timedelta(days=h)).strftime("%Y-%m") == month), None)
        ref = r106.get((PRODUCT, day, h), {}).get("models", {}).get("preserved106", {}).get("training") if day else None
        if not isinstance(ref, dict) or any(fit.get("training", {}).get(k) != ref.get(k) for k in ("count","firstTarget","lastTarget","fitOrigin")):
            raise ValueError("saved fit audit differs from frozen R106 training support")
        if not fit.get("fitFailure"):
            state = {k:fit[k] for k in ("scalerMean","scalerScale","coefs","coefShapes","intercepts")}
            if fit.get("inferenceStateSha256") != sha(canonical(state)):
                raise ValueError("saved inference state fingerprint mismatch")
    for r in doc["forecasts"]:
        k = (r["product"], dt.date.fromisoformat(r["target"]), r["horizon"])
        if r.get("eligible") is not True: raise ValueError(f"source E106 key is not eligible at {k}")
        b = base[k]["candidate"]
        if r["product"] != PRODUCT or not r["eligible"] or not r["usedMLP"]:
            if r.get("status") != b.get("status") or r.get("prediction") != b.get("prediction"):
                raise ValueError(f"E106 output changed outside valid MLP policy at {k}")
        if r.get("usedMLP"):
            fit = fits.get(r.get("fitId"))
            if (fit is None or fit.get("fitFailure") or fit.get("target") != "log(target/p0)"
                    or fit.get("training") != r.get("training")):
                raise ValueError(f"missing fixed MLP provenance at {k}")
            vector=r.get("featureVector")
            if not isinstance(vector,list) or len(vector)!=106 or any(not math.isfinite(float(v)) for v in vector):
                raise ValueError(f"missing finite replay feature vector at {k}")
            source_feature=features[r["horizon"]].get(k[1])
            if source_feature is None or vector != source_feature["preserved106"]:
                raise ValueError(f"saved prediction features do not replay from pinned source at {k}")
            if (r.get("p0Date") != source_feature["p0Date"].isoformat()
                    or not math.isclose(float(r.get("p0")),source_feature["p0"],rel_tol=0.0,abs_tol=1e-12)):
                raise ValueError(f"saved p0 does not replay from pinned source at {k}")
            sarima=r106[k]["models"]["causalSarima"]
            if sarima.get("status")!="issued" or not math.isclose(float(r.get("sarimaPrediction")),float(sarima["prediction"]),rel_tol=1e-12,abs_tol=1e-9):
                raise ValueError(f"candidate SARIMA is not identical to frozen R106 at {k}")
            p0=float(r.get("p0")); sarima=float(r.get("sarimaPrediction"))
            standalone,blend=_replay_prediction(fit,vector,p0,sarima)
            if not math.isclose(standalone,float(r["mlpStandalonePrediction"]),rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError(f"saved MLP inference replay mismatch at {k}")
            if not math.isclose(blend,float(r["prediction"]),rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError(f"saved 50:50 blend replay mismatch at {k}")


def _score(manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    """Score fixed paired populations, 20 garlic DM tests, and supporting diagnostics."""
    from . import equal_info_online_score as stats
    validate(manifest, phase)
    base = _idx(_base_doc(phase)["forecasts"])
    r106 = _r106_index()
    prices = json.loads(PRICES.read_text())
    actuals = pf.direct_horizon._observations(prices)
    rows = manifest["forecasts"]
    pairs_by_h = {h: [] for h in HORIZONS}; standalone_by_h = {h: [] for h in HORIZONS}; allcells = []
    for r in rows:
        if not r["eligible"]: continue
        p, d, h = r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]
        y = actuals.get(p, {}).get(d)
        b = base[(p, d, h)]["candidate"].get("prediction")
        if y is None or b is None or r.get("prediction") is None: continue
        item = {"target": d.isoformat(), "actual": y[0], "candidate": r["prediction"], "E106": b}
        if p == PRODUCT: pairs_by_h[h].append(item)
        if p == PRODUCT and r.get("usedMLP"):
            r106_direct = r106.get((p,d,h),{}).get("models",{}).get("preserved106",{}).get("prediction")
            if r106_direct is not None:
                standalone_by_h[h].append({"target":d.isoformat(),"actual":y[0],"MLP":r["mlpStandalonePrediction"],"R106":r106_direct})
        allcells.append((p, h, item))
    hmetrics=[]
    for h in HORIZONS:
        xs=pairs_by_h[h]
        maeA=math.fsum(abs(x['candidate']-x['actual']) for x in xs)/len(xs) if xs else None
        maeB=math.fsum(abs(x['E106']-x['actual']) for x in xs)/len(xs) if xs else None
        mapeA=100*math.fsum(abs(x['candidate']-x['actual'])/x['actual'] for x in xs)/len(xs) if xs else None
        mapeB=100*math.fsum(abs(x['E106']-x['actual'])/x['actual'] for x in xs)/len(xs) if xs else None
        hmetrics.append({"horizon":h,"n":len(xs),"uniqueTargets":len({x['target'] for x in xs}),"candidateMae":maeA,"E106Mae":maeB,"deltaMae":maeA-maeB if xs else None,"candidateMape":mapeA,"E106Mape":mapeB,"deltaMape":mapeA-mapeB if xs else None})
    dm=_dm_family(pairs_by_h)
    standalone_metrics=[]
    for h in HORIZONS:
        xs=standalone_by_h[h]
        standalone_metrics.append({"horizon":h,"n":len(xs),"mlpMae":math.fsum(abs(x['MLP']-x['actual']) for x in xs)/len(xs) if xs else None,
            "r106Mae":math.fsum(abs(x['R106']-x['actual']) for x in xs)/len(xs) if xs else None,
            "mlpMape":100*math.fsum(abs(x['MLP']-x['actual'])/x['actual'] for x in xs)/len(xs) if xs else None,
            "r106Mape":100*math.fsum(abs(x['R106']-x['actual'])/x['actual'] for x in xs)/len(xs) if xs else None})
    policy = []
    for h in HORIZONS:
        xs=[]
        for r in rows:
            if r['product']!=PRODUCT or r['horizon']!=h or not r['eligible']: continue
            d=dt.date.fromisoformat(r['target']); y=actuals.get(PRODUCT,{}).get(d)
            b=base[(PRODUCT,d,h)]['candidate'].get('prediction')
            if y is not None and b is not None:
                pred=r['prediction'] if r['status']=='issued' else b
                xs.append((pred,b,y[0]))
        policy.append({"horizon":h,"n":len(xs),"candidateMae":math.fsum(abs(a-c) for a,b,c in xs)/len(xs) if xs else None,
                       "E106Mae":math.fsum(abs(b-c) for a,b,c in xs)/len(xs) if xs else None,
                       "candidateMape":100*math.fsum(abs(a-c)/c for a,b,c in xs)/len(xs) if xs else None,
                       "E106Mape":100*math.fsum(abs(b-c)/c for a,b,c in xs)/len(xs) if xs else None})
    used = [r for r in rows if r['eligible'] and r['product']==PRODUCT]
    n_pairs = sum(x['n'] for x in hmetrics)
    macro = {k: math.fsum(x[k] for x in hmetrics if x[k] is not None)/10 for k in ('candidateMae','E106Mae','candidateMape','E106Mape')}
    macro['deltaMae']=macro['candidateMae']-macro['E106Mae']; macro['deltaMape']=macro['candidateMape']-macro['E106Mape']
    valid_mlp = {h:[x for x in pairs_by_h[h] if next(r for r in used if r['horizon']==h and r['target']==x['target'])['usedMLP']]
                 for h in HORIZONS}
    valid_metrics=[]
    for h in HORIZONS:
        xs=valid_mlp[h]
        valid_metrics.append({"horizon":h,"n":len(xs),"candidateMae":math.fsum(abs(x['candidate']-x['actual']) for x in xs)/len(xs) if xs else None,
                             "E106Mae":math.fsum(abs(x['E106']-x['actual']) for x in xs)/len(xs) if xs else None,
                             "candidateMape":100*math.fsum(abs(x['candidate']-x['actual'])/x['actual'] for x in xs)/len(xs) if xs else None,
                             "E106Mape":100*math.fsum(abs(x['E106']-x['actual'])/x['actual'] for x in xs)/len(xs) if xs else None})
    # Frozen diagnostics compare the full policy with E, causal50 and causal50 as the E reference.
    ref_metrics=[]
    for p in pf.PRODUCTS:
        for h in HORIZONS:
            xs=[]
            for r in rows:
                if r['product']!=p or r['horizon']!=h or not r['eligible']: continue
                d=dt.date.fromisoformat(r['target']); y=actuals.get(p,{}).get(d)
                br=base[(p,d,h)]
                e=br['E'].get('prediction'); e106=br['candidate'].get('prediction'); c=br['causal50'].get('prediction')
                if y is None or e is None or e106 is None or c is None: continue
                cand=r.get('prediction') if r['product']==PRODUCT else br['candidate'].get('prediction')
                if cand is None: cand=e
                xs.append((cand,e,e106,c,y[0]))
            if xs:
                ref_metrics.append({"product":p,"horizon":h,"n":len(xs),
                    "candidateMae":math.fsum(abs(a-d) for a,e,e106,c,d in xs)/len(xs),"EMae":math.fsum(abs(e-d) for a,e,e106,c,d in xs)/len(xs),"E106Mae":math.fsum(abs(e106-d) for a,e,e106,c,d in xs)/len(xs),"causal50Mae":math.fsum(abs(c-d) for a,e,e106,c,d in xs)/len(xs),
                    "candidateMape":100*math.fsum(abs(a-d)/d for a,e,e106,c,d in xs)/len(xs),"EMape":100*math.fsum(abs(e-d)/d for a,e,e106,c,d in xs)/len(xs),"E106Mape":100*math.fsum(abs(e106-d)/d for a,e,e106,c,d in xs)/len(xs),"causal50Mape":100*math.fsum(abs(c-d)/d for a,e,e106,c,d in xs)/len(xs)})
    # Monthly harm and equal-cell references are descriptive; they do not alter the gate.
    monthly_policy=[]
    for month in sorted({dt.date.fromisoformat(r['target']).strftime('%Y-%m') for r in rows}):
        for h in HORIZONS:
            xs=[]
            for r in rows:
                if r['product']!=PRODUCT or r['horizon']!=h or dt.date.fromisoformat(r['target']).strftime('%Y-%m')!=month: continue
                d=dt.date.fromisoformat(r['target']); y=actuals.get(PRODUCT,{}).get(d); e=base[(PRODUCT,d,h)]['candidate'].get('prediction')
                if y is None or e is None: continue
                candidate=r['prediction'] if r['prediction'] is not None else e
                xs.append((candidate,e,y[0]))
            monthly_policy.append({"month":month,"horizon":h,"n":len(xs),
                "candidateMae":math.fsum(abs(a-y) for a,e,y in xs)/len(xs) if xs else None,
                "E106Mae":math.fsum(abs(e-y) for a,e,y in xs)/len(xs) if xs else None,
                "candidateMape":100*math.fsum(abs(a-y)/y for a,e,y in xs)/len(xs) if xs else None,
                "E106Mape":100*math.fsum(abs(e-y)/y for a,e,y in xs)/len(xs) if xs else None})
    portfolio_cells=[]
    for p in pf.PRODUCTS:
        for h in HORIZONS:
            xs=[]
            for r in rows:
                if r['product']!=p or r['horizon']!=h: continue
                d=dt.date.fromisoformat(r['target']); y=actuals.get(p,{}).get(d); br=base[(p,d,h)]
                e=br['E'].get('prediction'); e106=br['candidate'].get('prediction'); c=br['causal50'].get('prediction')
                if y is None or e is None or e106 is None or c is None: continue
                a=r['prediction'] if p==PRODUCT and r['prediction'] is not None else e106
                xs.append((a,e,e106,c,y[0]))
            if xs:
                portfolio_cells.append({"product":p,"horizon":h,"n":len(xs),
                    "candidateMae":math.fsum(abs(a-y) for a,e,e106,c,y in xs)/len(xs),"EMae":math.fsum(abs(e-y) for a,e,e106,c,y in xs)/len(xs),"E106Mae":math.fsum(abs(e106-y) for a,e,e106,c,y in xs)/len(xs),"causal50Mae":math.fsum(abs(c-y) for a,e,e106,c,y in xs)/len(xs),
                    "candidateMape":100*math.fsum(abs(a-y)/y for a,e,e106,c,y in xs)/len(xs),"EMape":100*math.fsum(abs(e-y)/y for a,e,e106,c,y in xs)/len(xs),"E106Mape":100*math.fsum(abs(e106-y)/y for a,e,e106,c,y in xs)/len(xs),"causal50Mape":100*math.fsum(abs(c-y)/y for a,e,e106,c,y in xs)/len(xs)})
    reference_portfolio={}
    for field in ('candidate','E','E106','causal50'):
        mae=f'{field}Mae' if field!='candidate' else 'candidateMae'; mape=f'{field}Mape' if field!='candidate' else 'candidateMape'
        reference_portfolio[field]={"cellsIncluded":len(portfolio_cells),"equalCellMae":math.fsum(x[mae] for x in portfolio_cells)/len(portfolio_cells) if portfolio_cells else None,
                                    "equalCellMape":math.fsum(x[mape] for x in portfolio_cells)/len(portfolio_cells) if portfolio_cells else None}
    # E106's top ceil(10%) errors are selected once per phase/horizon before scoring candidates.
    large_error=[]
    for h in HORIZONS:
        eligible=[]
        for r in rows:
            if r['product']!=PRODUCT or r['horizon']!=h or not r['eligible']: continue
            d=dt.date.fromisoformat(r['target']); y=actuals.get(PRODUCT,{}).get(d); e=base[(PRODUCT,d,h)]['candidate'].get('prediction')
            if y is not None and e is not None: eligible.append((abs(e-y[0])/y[0],d,r,e,y[0]))
        eligible.sort(key=lambda z:(-z[0],z[1]))
        chosen=eligible[:math.ceil(.1*len(eligible))]
        cand_loss=math.fsum(abs((r['prediction'] if r['prediction'] is not None else e)-y)/y for _,_,r,e,y in chosen)
        base_loss=math.fsum(abs(e-y)/y for _,_,r,e,y in chosen)
        denom=math.fsum(abs(e-y)/y for _,_,r,e,y in eligible)
        large_error.append({"horizon":h,"populationN":len(eligible),"selectedN":len(chosen),"selectedTargetDates":[d.isoformat() for _,d,_,_,_ in chosen],
                           "E106ApeContribution":base_loss/denom if denom else None,"candidatePolicyApeContribution":cand_loss/denom if denom else None,
                           "deltaSelectedMeanApe":(cand_loss-base_loss)/len(chosen) if chosen else None})
    from .research_progress_diagnostic import circular_block_bootstrap
    boot_rows=[]
    for r in rows:
        if not r['eligible']: continue
        p,d,h=r['product'],dt.date.fromisoformat(r['target']),r['horizon']; y=actuals.get(p,{}).get(d)
        b=base[(p,d,h)]['candidate'].get('prediction')
        if y is None or b is None: continue
        a=r['prediction'] if p==PRODUCT and r['prediction'] is not None else b
        boot_rows.append({"product":p,"target":d.isoformat(),"horizon":h,"actual":y[0],"candidate":a,"E106":b})
    bootstrap={}
    for block in (14,28):
        args={"a":"candidate","b":"E106","block":block,"replicates":2000,"seed":20261011,
              "calendar_start":PHASES[phase][0],"calendar_end":PHASES[phase][1]}
        bootstrap[str(block)]={"fortyCellApe":circular_block_bootstrap(boot_rows,metric="ape",**args),
                               "fortyCellAe":circular_block_bootstrap(boot_rows,metric="ae",**args)}
        garlic_rows=[x for x in boot_rows if x['product']==PRODUCT]
        bootstrap[str(block)]["garlic10hApe"]=circular_block_bootstrap(garlic_rows,metric="ape",**args)
        bootstrap[str(block)]["garlic10hAe"]=circular_block_bootstrap(garlic_rows,metric="ae",**args)
    all_baseline_issued = all(base[(PRODUCT,dt.date.fromisoformat(r['target']),r['horizon'])]['candidate'].get('status')!='issued' or r.get('status')=='issued' for r in used)
    minimum_pairs=min((x['n'] for x in hmetrics),default=0)
    pass_primary = _primary_pass(macro,all_baseline_issued,minimum_pairs)
    dm_summary=[]
    for bw in (9,28):
        family=[x for x in dm if x['bandwidth']==bw]
        dm_summary.append({"bandwidth":bw,"familySize":len(family),
                           "significantImprovement":sum(x['status']=='tested' and x.get('holmSignificant') and x.get('meanLossDifference',0)<0 for x in family),
                           "significantWorsening":sum(x['status']=='tested' and x.get('holmSignificant') and x.get('meanLossDifference',0)>0 for x in family),
                           "inferenceUnavailable":sum(x['status']!='tested' for x in family),
                           "testedNotSignificantAfterHolm":sum(x['status']=='tested' and not x.get('holmSignificant') for x in family)})
    return {"phase":phase,"primaryGarlic10h":macro,"primaryDecision":{"passes":pass_primary,"deltaMapeNegative":macro['deltaMape']<0,"deltaMaeNonpositive":macro['deltaMae']<=0,
                "allEligibleE106IssuedKeysRetained":all_baseline_issued,"atLeast30PairsAtEveryHorizon":minimum_pairs>=30},
            "support": {"pairedRows":n_pairs,"minimumPairsPerHorizon":minimum_pairs,"allE106KeysRetained":all_baseline_issued},
            "garlicHorizonMetrics":hmetrics,"dm":dm,"dmSummaryByBandwidth":dm_summary,"policyMetrics":policy,
            "validMLPOnlyGarlicDiagnostic":valid_metrics,"validMLPStandaloneVsR106Direct":standalone_metrics,
            "fortyCellReferences":ref_metrics,"monthlyGarlicPolicyMetrics":monthly_policy,"fortyCellReferencePortfolio":reference_portfolio,
            "dmProtocol":{"testsPerBandwidth":20,"losses":{"ae":"absolute error in currency/kg","ape":"absolute percentage error as a fraction; multiply meanLossDifference by 100 for percentage points"},
                          "calendarHacDays":[9,28],"holmsAlpha":0.05,"minimumPairs":30},
            "calendarBlockBootstrap14_28":bootstrap,"largeErrorDiagnostic":large_error,
            "usedMLP":sum(r['usedMLP'] for r in used),"fallbackCount":sum(bool(r['fallbackReason']) for r in used),
            "fallbackReasons":{reason:sum(r['fallbackReason']==reason for r in used) for reason in sorted({r['fallbackReason'] for r in used if r['fallbackReason']})}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='cmd',required=True)
    b=s.add_parser('build');b.add_argument('--phase',choices=PHASES,required=True);b.add_argument('--output',type=Path);b.add_argument('--confirmation-report',type=Path)
    pfly=s.add_parser('preflight');pfly.add_argument('--phase',choices=PHASES,required=True);pfly.add_argument('--confirmation-report',type=Path)
    v=s.add_parser('verify');v.add_argument('--phase',choices=PHASES,required=True);v.add_argument('--manifest',type=Path,required=True)
    sc=s.add_parser('score');sc.add_argument('--phase',choices=PHASES,required=True);sc.add_argument('--manifest',type=Path,required=True);sc.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.cmd=='preflight':
        result=_preflight(a.phase,a.confirmation_report);print(json.dumps(result,ensure_ascii=False,sort_keys=True));return 0
    if a.cmd=='build':
        dest=a.output or OUT/(f"{a.phase}-manifest.json")
        if dest.exists(): raise FileExistsError(f"immutable output exists: {dest}")
        doc=_build(a.phase,lambda h,m,n: print(f"{a.phase} garlic/h{h} {m[0]}-{m[1]:02d}: {n} rows",flush=True), a.confirmation_report)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(canonical(doc))
        print(f"wrote {dest} {file_sha(dest)}")
    elif a.cmd=='verify':
        recorded=json.loads(a.manifest.read_text());verify_sources(recorded,a.phase)
        print("manifest digest, source hashes, E106 preservation and fixed MLP provenance verified (no refit)")
    else:
        m=json.loads(a.manifest.read_text());verify_sources(m,a.phase); report=_score(m,a.phase); report['manifestFileSha256']=file_sha(a.manifest)
        if a.output.exists(): raise FileExistsError(f"immutable output exists: {a.output}")
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('xb') as f:f.write(canonical(report))
        print(f"wrote {a.output} {file_sha(a.output)}")
    return 0

if __name__=='__main__': raise SystemExit(main())
