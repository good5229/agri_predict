"""Frozen offline 1–10 day Gwangju residual-correction evaluation.

The CLI reads the pinned price/report files and opens the validated collection
database read-only. Run it only after the source has received its required review.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, kma_residual_eval as common, market_extension

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = dh.INPUT_PRICES
REPORT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
DB_PATH = ROOT / "data/agri.sqlite"
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
TRAIN_START, TRAIN_END = dt.date(2023, 1, 1), dt.date(2025, 9, 1)
EVAL_START, EVAL_END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
MIN_TRAIN_DAYS, MAX_AGE = 90, 14
BANDWIDTHS = (9, 28)
FAMILY_SIZE = 5 * 10 * 2 * 2


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _feature(origin: dt.date, gw: dict[dt.date, dict[str, Any]], core: dict[dt.date, tuple[float, float]]):
    """Return the five frozen features and a source-selection audit."""
    gday = max((d for d, v in gw.items() if d <= origin and _positive_pair(v)), default=None)
    cday = max((d for d, v in core.items() if d <= origin and _positive_pair(v)), default=None)
    if gday is None or cday is None or (origin-gday).days > MAX_AGE or (origin-cday).days > MAX_AGE:
        return None, {"status": "source_missing_or_stale", "gwangjuDate": gday.isoformat() if gday else None,
                      "coreDate": cday.isoformat() if cday else None}
    gp, gwgt = float(gw[gday]["price"]), float(gw[gday]["weight"])
    cp = float(core[cday][0])
    lag_cut = origin - dt.timedelta(days=7)
    prev = max((d for d, v in gw.items() if d <= lag_cut and _positive_pair(v)), default=None)
    available = prev is not None and (lag_cut-prev).days <= MAX_AGE
    delta = math.log(gp / float(gw[prev]["price"])) if available else 0.0
    return [math.log(gp/cp), math.log(gwgt), float((origin-gday).days), delta, float(available)], {
        "status": "available", "gwangjuDate": gday.isoformat(), "coreDate": cday.isoformat(),
        "lagDate": prev.isoformat() if available else None,
    }


def _positive_pair(value):
    try:
        price, weight = (value["price"], value["weight"]) if isinstance(value, dict) else value
        return math.isfinite(float(price)) and float(price) > 0 and math.isfinite(float(weight)) and float(weight) > 0
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _test_direction(test: dict[str, Any]) -> str:
    """Direction after the fixed-family correction; positive loss delta favors comparator."""
    if test.get("status") != "tested":
        return "inference_unavailable"
    if not test.get("holmSignificant"):
        return "inconclusive"
    delta = test.get("meanLossDifference")
    return "worsening" if delta is not None and delta > 0 else "improvement" if delta is not None and delta < 0 else "inconclusive"


def _combine_directions(directions: list[str], *, require_all_improve: bool = False) -> str:
    adverse = "worsening" in directions
    favorable = "improvement" in directions
    if adverse and favorable:
        return "mixed_evidence"
    if adverse:
        return "worsening"
    if require_all_improve and directions and all(x == "improvement" for x in directions):
        return "improvement"
    if "inference_unavailable" in directions:
        return "inference_unavailable"
    return "inconclusive"


def _direct(prices, report):
    """Use the shared strict direct-report validator, retaining origin alignment."""
    return common._direct_rows(report, prices)


def build(prices: dict[str, Any], report: dict[str, Any], gwangju: dict[str, dict[dt.date, dict[str, Any]]], *, hashes=None, provenance=None):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    direct = _direct(prices, report)
    core_obs = dh._observations(prices)
    features = {}
    for row in direct.values():
        key = (row["product"], row["origin"])
        if key not in features:
            features[key] = _feature(row["origin"], gwangju.get(row["product"], {}), core_obs[row["product"]])

    fitted, audits = {}, {}
    for product in PRODUCTS:
        for h in HORIZONS:
            key = (product, h)
            train = [r for r in direct.values() if r["product"] == product and r["horizon"] == h
                     and TRAIN_START <= r["target"] <= TRAIN_END and features[(product, r["origin"])][0] is not None]
            n_days = len({r["target"] for r in train})
            audit = {"trainingRows": len(train), "distinctTrainingTargets": n_days, "status": "insufficient_training"}
            if n_days >= MIN_TRAIN_DAYS:
                x = np.asarray([features[(product, r["origin"])][0] for r in train], dtype=float)
                y = np.asarray([math.log(r["actual"]/r["direct"]) for r in train], dtype=float)
                scaler = StandardScaler().fit(x)
                scaler.scale_[scaler.scale_ == 0] = 1.0
                model = Ridge(alpha=100, fit_intercept=False).fit(scaler.transform(x), y)
                fitted[key] = (scaler, model)
                audit.update(status="fitted", trainTargetStart=min(r["target"] for r in train).isoformat(),
                             trainTargetEnd=max(r["target"] for r in train).isoformat())
            audits[f"{product}/{h}"] = audit

    eligible = [r for r in direct.values() if EVAL_START <= r["target"] <= EVAL_END]
    preds, grouped = [], defaultdict(list)
    for row in eligible:
        key = (row["product"], row["horizon"])
        vec, src = features[(row["product"], row["origin"])]
        candidate, reason = row["direct"], "model_unavailable"
        if vec is None:
            reason = src["status"]
        elif key in fitted:
            scaler, model = fitted[key]
            corr = float(model.predict(scaler.transform(np.asarray([vec], dtype=float)))[0])
            value = row["direct"] * math.exp(min(.2, max(-.2, corr))) if math.isfinite(corr) else float("nan")
            if math.isfinite(value) and value > 0:
                candidate, reason = value, None
            else:
                reason = "nonfinite_or_nonpositive_candidate"
        out = {**row, "candidate": candidate, "gwangjuFeatures": src, "fallbackReason": reason}
        preds.append(out); grouped[key].append(out)

    comparisons, strata = [], []
    for product in PRODUCTS:
        for h in HORIZONS:
            rows = grouped[(product, h)]
            for comparator in ("direct", "persistence"):
                for loss in ("ae", "ape"):
                    difference = {r["target"]: (abs(r["actual"]-r["candidate"]) if loss == "ae" else 100*abs(r["actual"]-r["candidate"])/r["actual"])
                                 - (abs(r["actual"]-r[comparator]) if loss == "ae" else 100*abs(r["actual"]-r[comparator])/r["actual"]) for r in rows}
                    metric = "mae" if loss == "ae" else "mape"
                    comparisons.append({"product": product, "horizon": h, "comparator": comparator, "loss": loss,
                        "n": len(rows), "candidateMetric": common._metrics(rows, "candidate")[metric],
                        "comparatorMetric": common._metrics(rows, comparator)[metric],
                        "dm": {str(b): common._dm(difference, b) for b in BANDWIDTHS}})
            for field, fn in (("year", lambda d: d.year), ("season", lambda d: (d.month-1)//3+1)):
                parts = defaultdict(list)
                for r in rows: parts[fn(r["target"])].append(r)
                for value, part in sorted(parts.items()):
                    strata.append({"product": product, "horizon": h, "stratum": field, "value": value, "n": len(part),
                                   **{m: common._metrics(part, m) for m in ("candidate", "direct", "persistence")}})
    for b in BANDWIDTHS:
        family = [x["dm"][str(b)] for x in comparisons]
        if len(family) != FAMILY_SIZE: raise AssertionError("fixed 200-test family changed")
        common._holm(family)
        for item in comparisons:
            item["dm"][str(b)]["direction"] = _test_direction(item["dm"][str(b)])
    joint = []
    for product in PRODUCTS:
        for h in HORIZONS:
            rows = grouped[(product, h)]
            tests = [x for x in comparisons if x["product"] == product and x["horizon"] == h]
            lower = all(x["candidateMetric"] is not None and x["candidateMetric"] < x["comparatorMetric"] for x in tests)
            primary = [x["dm"]["9"] for x in tests]
            sig = all(x["holmSignificant"] and x["meanLossDifference"] < 0 for x in primary)
            enough = all(x["status"] == "tested" for x in primary)
            comparator_conclusions = {}
            for comparator in ("direct", "persistence"):
                subset = [x for x in tests if x["comparator"] == comparator]
                directions = [x["dm"]["9"]["direction"] for x in subset]
                comparator_conclusions[comparator] = _combine_directions(directions, require_all_improve=True)
            adverse_directions = [x["dm"]["9"]["direction"] for x in tests]
            conclusion = "improvement" if lower and sig else _combine_directions(adverse_directions)
            # Keep the frozen joint gate: one adverse test can describe worsening, but only
            # all four corrected improvements plus both lower point estimates earns improvement.
            if conclusion == "improvement" and not (lower and sig):
                conclusion = "inconclusive"
            joint.append({"product": product, "horizon": h, "conclusion": conclusion,
                          "comparatorConclusions": comparator_conclusions,
                          "allMetricsLower": lower, "allHac9HolmTestsImprove": sig, "eligibleRows": len(rows),
                          "availableRows": sum(r["gwangjuFeatures"]["status"] == "available" and r["fallbackReason"] is None for r in rows),
                          "fallbackRows": sum(r["fallbackReason"] is not None for r in rows)})
    # Each product is first averaged equally over its ten horizons; then products equally.
    product_mape = {}
    for product in PRODUCTS:
        hm = [common._metrics(grouped[(product, h)], "candidate")["mape"] for h in HORIZONS]
        product_mape[product] = sum(hm)/10 if all(v is not None for v in hm) else None
    all_mape = [v for v in product_mape.values() if v is not None]
    return {"schemaVersion": 1, "status": "complete", "protocol": {"id": "gwangju-origin-residual-ridge-2026-10-07",
        "trainingTargets": [TRAIN_START.isoformat(), TRAIN_END.isoformat()], "evaluationTargets": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "maximumSourceAgeDays": MAX_AGE, "minimumDistinctTrainingTargets": MIN_TRAIN_DAYS,
        "features": ["log(gwangju/core price)", "log(gwangju net weight)", "origin minus last Gwangju date", "Gwangju log price change from origin-7 cutoff", "lag-change available"],
        "model": {"scaler": "training-only StandardScaler; zero scale set to 1", "estimator": "Ridge", "alpha": 100, "fitIntercept": False, "clipLogCorrection": [-.2, .2]},
        "dm": {"familySize": FAMILY_SIZE, "bandwidthsCalendarDays": list(BANDWIDTHS), "minimumPairs": 30, "holmAlpha": .05},
        "limitations": "Retrospective revised prices; source first publication times unknown; not a prospective or official national accuracy claim."},
        "hashes": hashes or {}, "gwangjuProvenance": provenance or {}, "fitAudits": audits,
        "coverage": {f"{p}/{h}": {"eligibleDirectRows": len(grouped[(p,h)]), "availableRows": sum(x["gwangjuFeatures"]["status"] == "available" for x in grouped[(p,h)]),
             "fallbackRows": sum(x["fallbackReason"] is not None for x in grouped[(p,h)]), "fallbackReasons": dict(Counter(x["fallbackReason"] for x in grouped[(p,h)] if x["fallbackReason"]))}
             for p in PRODUCTS for h in HORIZONS},
        "predictions": [{**{k:(v.isoformat() if isinstance(v, dt.date) else v) for k,v in r.items()}} for r in preds],
        "comparisons": comparisons, "jointConclusions": joint, "strata": strata,
        "goalGate": {"aggregation": "equal horizon MAPE within product, then equal product mean", "productEqualHorizonMape": product_mape,
                     "fiveProductEqualWeightMape": sum(all_mape)/len(all_mape) if len(all_mape)==5 else None,
                     "goalMet": len(all_mape)==5 and sum(all_mape)/5 <= 10}}


def write_report(report: dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output_dir/"report.json").write_text(payload, encoding="utf-8")
    goal = report["goalGate"]
    lines = ["# Gwangju origin-information residual correction", "", f"Status: {report['status']}",
             f"Five-product equal-weight MAPE: {goal['fiveProductEqualWeightMape']}",
             f"10% point-estimate goal met: {goal['goalMet']}", "", "| Product | Horizon | n | Candidate MAE | Candidate MAPE | Direct MAPE | Persistence MAPE | Conclusion |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for j in report["jointConclusions"]:
        p,h=j["product"],j["horizon"]; rows=[r for r in report["predictions"] if r["product"]==p and r["horizon"]==h]
        metrics={m:common._metrics(rows,m) for m in ("candidate","direct","persistence")}
        fmt=lambda v: "NA" if v is None else f"{v:.4f}"
        lines.append(f"| {p} | {h} | {len(rows)} | {fmt(metrics['candidate']['mae'])} | {fmt(metrics['candidate']['mape'])} | {fmt(metrics['direct']['mape'])} | {fmt(metrics['persistence']['mape'])} | {j['conclusion']} |")
    lines += ["", "This is a retrospective experiment using retrieved revisions. First publication times are unknown; results are not prospective or official national accuracy claims.", ""]
    (output_dir/"report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> str:
    """Persist selection provenance before model fitting; never replace an earlier manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis-manifest.json"
    data = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return sha256(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=PRICE_PATH); parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    pb, rb = args.prices.read_bytes(), args.report.read_bytes()
    hashes = {"pricesSha256": sha256(pb), "directReportSha256": sha256(rb)}
    if hashes["pricesSha256"] != PRICE_SHA256 or hashes["directReportSha256"] != REPORT_SHA256:
        parser.error("price or direct report does not match frozen SHA-256")
    db_uri = "file:" + str(args.db.resolve()) + "?mode=ro"
    db = sqlite3.connect(db_uri, uri=True); db.row_factory = sqlite3.Row; db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        gw, provenance = market_extension.extract_gwangju_period(db, dt.date(2022,12,1), EVAL_END)
        db.rollback()
    finally: db.close()
    manifest_hash = _write_manifest(args.output_dir, {"protocolId": "gwangju-origin-residual-ridge-2026-10-07",
        "inputHashes": hashes, "gwangjuSelection": provenance,
        "databasePath": str(args.db.resolve()), "databaseAccess": "read-only transaction"})
    result = build(json.loads(pb), json.loads(rb), gw, hashes=hashes, provenance=provenance)
    result["analysisManifest"] = {"path": str((args.output_dir / "analysis-manifest.json").resolve()), "sha256": manifest_hash,
                                  "persistedBeforeModelFit": True}
    write_report(result, args.output_dir)
    print(json.dumps({"outputDir": str(args.output_dir), "hashes": hashes, "goalGate": result["goalGate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
