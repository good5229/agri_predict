"""Frozen retrospective fixed-shrinkage experiment for daily forecast revisions."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import adaptive_selector as base
from . import experiments

INPUT_SHA256 = base.INPUT_SHA256
DEVELOP = (dt.date(2017, 1, 1), dt.date(2021, 12, 31))
VALIDATE = (dt.date(2022, 1, 1), dt.date(2023, 12, 31))
EVALUATE = (dt.date(2024, 1, 1), dt.date(2026, 9, 27))
WEIGHTS = (0.25, 0.50, 0.75, 1.00)
MIN_SPLIT = 180
LAGS = (9, 28)
COMPARATORS = ("baseline", "parent")
LOSSES = ("absolute_error", "absolute_percentage_error")


def _adjust(parent: float, model: float, weight: float) -> float:
    value = parent + weight * (model - parent)
    return value if math.isfinite(value) and value > 0 else math.nan


def _stats(rows):
    if not rows:
        return {"n": 0, "mae": None, "mape": None}
    n = len(rows)
    return {"n": n, "mae": sum(abs(r["pred"]-r["actual"]) for r in rows)/n,
            "mape": sum(abs(r["pred"]-r["actual"])/r["actual"]*100 for r in rows)/n}


def _holm(tests):
    order = sorted(tests, key=lambda t: (t["pValue"] if t["pValue"] is not None else 1., t["productId"], t["horizon"], t["comparator"], t["loss"]))
    run = 0.
    for i, t in enumerate(order):
        p = t["pValue"] if t["pValue"] is not None else 1.
        run = max(run, min(1., (len(order)-i)*p))
        t["holmAdjustedPValue"] = run
        t["holmSignificantAt005"] = t["status"] == "available" and run < .05


def _season(day):
    return ("winter" if day.month in (12, 1, 2) else "spring" if day.month in (3, 4, 5)
            else "summer" if day.month in (6, 7, 8) else "autumn")


def _difference(a, b):
    return {"mae": a["mae"]-b["mae"] if a["mae"] is not None and b["mae"] is not None else None,
            "mape": a["mape"]-b["mape"] if a["mape"] is not None and b["mape"] is not None else None}


def _conclusion(tests):
    if any(t["status"] != "available" for t in tests): return "inference_unavailable"
    directions = []
    for t in tests:
        diff = t["meanLossDifference"]
        directions.append("improved" if diff < 0 and t["holmSignificantAt005"] else
                          "worsened" if diff > 0 and t["holmSignificantAt005"] else "no_evidence")
    if all(x == "improved" for x in directions): return "improved"
    if all(x == "worsened" for x in directions): return "worsened"
    if "improved" in directions or "worsened" in directions: return "mixed_evidence"
    return "no_evidence"


def _group_metrics(rows):
    out = {}
    for comparator in COMPARATORS:
        paired = [(r, r["baseline"] if comparator == "baseline" else r["parent"]) for r in rows]
        selected = _stats([{"pred": r["pred"], "actual": r["actual"]} for r, _ in paired])
        other = _stats([{"pred": p, "actual": r["actual"]} for r, p in paired])
        out[comparator] = {"n": len(paired), "selected": selected, "comparator": other,
                           "meanLossDifference": _difference(selected, other),
                           "direction": "improved" if selected["mae"] is not None and selected["mae"] < other["mae"] and selected["mape"] < other["mape"] else
                                       "worsened" if selected["mae"] is not None and selected["mae"] > other["mae"] and selected["mape"] > other["mape"] else "mixed_or_equal"}
    return out


def analyze(source: dict[str, Any], source_sha256: str | None = None):
    products = base._validate(source)
    results, tests = [], {lag: [] for lag in LAGS}
    for pid in base.PRODUCTS:
        data = products[pid]
        for h in base.HORIZONS:
            maps = {m: {} for m in base.MODELS}
            for model, forecasts in data["forecasts"].items():
                for (origin, target, hh), value in forecasts.items():
                    if hh == h:
                        maps[model][(origin, target)] = value
            keys = set.intersection(*(set(maps[m]) for m in base.MODELS))
            common = []
            forecast_common = []
            for (origin, target) in keys:
                vals = [maps[m][(origin, target)] for m in ("baseline", "ridge", "hist_gb")]
                forecast_common.append({"origin": origin, "target": target, "values": vals})
                actual = data["actuals"].get(target)
                if actual and all(math.isfinite(v) and v > 0 for v in vals):
                    common.append({"origin": origin, "target": target, "actual": actual,
                                   "baseline": vals[0], "ridge": vals[1], "hist_gb": vals[2]})
            common.sort(key=lambda r: (r["target"], r["origin"]))
            dev = [r for r in common if DEVELOP[0] <= r["target"] <= DEVELOP[1]]
            val = [r for r in common if VALIDATE[0] <= r["target"] <= VALIDATE[1]]
            candidates = []
            for model in ("ridge", "hist_gb"):
                for weight in WEIGHTS:
                    candidates.append((model, weight, [_adjust(r["baseline"], r[model], weight) for r in dev]))
            chosen = ("baseline", 0.)
            dev_stats = {}
            baseline_dev = _stats([dict(pred=r["baseline"], actual=r["actual"]) for r in dev])
            viable = []
            for model, weight, preds in candidates:
                rows = [dict(pred=p, actual=r["actual"]) for r, p in zip(dev, preds) if math.isfinite(p)]
                stat = _stats(rows); dev_stats[f"{model}:{weight:.2f}"] = stat
                if stat["n"] >= MIN_SPLIT and stat["mae"] < baseline_dev["mae"] and stat["mape"] < baseline_dev["mape"]:
                    viable.append((stat["mape"], stat["mae"], weight, 0 if model == "ridge" else 1, model))
            if len(dev) >= MIN_SPLIT and viable:
                _, _, weight, _, model = min(viable)
                nominee = (model, weight)
            else:
                nominee = chosen
            val_base = _stats([dict(pred=r["baseline"], actual=r["actual"]) for r in val])
            val_nom = _stats([dict(pred=_adjust(r["baseline"], r[nominee[0]], nominee[1]), actual=r["actual"])
                              for r in val if nominee[0] != "baseline" and math.isfinite(_adjust(r["baseline"], r[nominee[0]], nominee[1]))]) if nominee[0] != "baseline" else val_base
            val_parent = _stats([dict(pred=r[nominee[0]], actual=r["actual"]) for r in val]) if nominee[0] != "baseline" else val_base
            valid = (nominee[0] != "baseline" and val_nom["n"] >= MIN_SPLIT and
                     val_nom["mae"] <= .99*val_base["mae"] and val_nom["mape"] <= .99*val_base["mape"] and
                     val_nom["mae"] <= val_parent["mae"] and val_nom["mape"] <= val_parent["mape"])
            selected = nominee if valid else chosen
            eval_rows = [r for r in common if EVALUATE[0] <= r["target"] <= EVALUATE[1] and r["origin"] >= EVALUATE[0]]
            adjusted = []
            for r in eval_rows:
                pred = r["baseline"] if selected[0] == "baseline" else _adjust(r["baseline"], r[selected[0]], selected[1])
                if math.isfinite(pred) and pred > 0:
                    adjusted.append({**r, "pred": pred})
            for r in adjusted: r["parent"] = r[selected[0]]
            per_year, per_season = defaultdict(list), defaultdict(list)
            for r in adjusted:
                per_year[str(r["target"].year)].append(r)
                per_season[_season(r["target"])].append(r)
            comps = {}
            for comp in COMPARATORS:
                paired = []
                for r in adjusted:
                    other = r["baseline"] if comp == "baseline" else r[selected[0]]
                    if math.isfinite(other) and other > 0:
                        paired.append((r, other))
                selected_stats = _stats([dict(pred=r["pred"], actual=r["actual"]) for r, _ in paired])
                comparator_stats = _stats([dict(pred=o, actual=r["actual"]) for r, o in paired])
                comps[comp] = {"n": len(paired), "adjusted": selected_stats,
                               "comparator": comparator_stats, "meanLossDifference": _difference(selected_stats, comparator_stats)}
                for loss in LOSSES:
                    for lag in LAGS:
                        rec = {"productId": pid, "horizon": h, "comparator": comp, "loss": loss}
                        metric = "mae" if loss == "absolute_error" else "mape"
                        diffs = []
                        for row, other in paired:
                            candidate_loss = abs(row["pred"]-row["actual"])
                            other_loss = abs(other-row["actual"])
                            if metric == "mape":
                                candidate_loss = candidate_loss/row["actual"]*100
                                other_loss = other_loss/row["actual"]*100
                            diffs.append(candidate_loss-other_loss)
                        rec["meanLossDifference"] = sum(diffs)/len(diffs) if diffs else None
                        if selected[0] == "baseline" or len(paired) < 30:
                            rec.update(status="unavailable_baseline_reselected" if selected[0] == "baseline" else "unavailable_insufficient_pairs", pValue=None)
                        else:
                            ca = [{"target": r["target"].isoformat(), "horizon": h, "predicted": r["pred"]} for r, _ in paired]
                            cb = [{"target": r["target"].isoformat(), "horizon": h, "predicted": o} for r, o in paired]
                            actuals = {(r["target"].isoformat(), h): r["actual"] for r, _ in paired}
                            test = experiments._dm(ca, cb, actuals, h, loss, max_lag=lag)
                            rec.update(test)
                            if test.get("status") != "available": rec["pValue"] = None
                        tests[lag].append(rec)
            eligible_days = {d for d in data["actuals"] if EVALUATE[0] <= d <= EVALUATE[1] and d-dt.timedelta(days=h) >= EVALUATE[0]}
            eval_forecast_keys = {(r["origin"], r["target"]) for r in forecast_common if EVALUATE[0] <= r["target"] <= EVALUATE[1] and r["origin"] >= EVALUATE[0]}
            eval_forecast_actual_keys = {(r["origin"], r["target"]) for r in common if EVALUATE[0] <= r["target"] <= EVALUATE[1] and r["origin"] >= EVALUATE[0]}
            results.append({"productId": pid, "horizon": h,
                "selection": {"development": {"start": DEVELOP[0].isoformat(), "end": DEVELOP[1].isoformat(), "n": len(dev), "baseline": baseline_dev, "candidateMetrics": dev_stats},
                    "validation": {"start": VALIDATE[0].isoformat(), "end": VALIDATE[1].isoformat(), "n": len(val), "baseline": val_base, "nominee": nominee, "candidate": val_nom, "parent": val_parent, "kept": valid},
                    "selectedModel": selected[0], "weight": selected[1]},
                "coverage": {"commonOrigins": len(common), "eligibleEvaluationActualTargets": len(eligible_days),
                    "evaluationCommonForecastOrigins": len(eval_forecast_keys), "evaluationCommonOriginsWithActual": len(eval_forecast_actual_keys),
                    "evaluationExcludedEligibleTargets": len(eligible_days)-len({t for _,t in eval_forecast_actual_keys}),
                    "evaluationN": len(adjusted), "evaluationOriginMinimum": EVALUATE[0].isoformat(),
                    "byYear": {k: _group_metrics(v) for k,v in per_year.items()}, "bySeason": {k:_group_metrics(v) for k,v in per_season.items()}},
                "comparisons": comps})
    for lag in LAGS: _holm(tests[lag])
    for result in results:
        pid, h = result["productId"], result["horizon"]
        combined = {}
        for comp in COMPARATORS:
            for lag in LAGS:
                family = [t for t in tests[lag] if t["productId"] == pid and t["horizon"] == h and t["comparator"] == comp]
                family.sort(key=lambda t: t["loss"])
                result["comparisons"][comp][f"conclusionHAC{lag}"] = _conclusion(family)
                combined.setdefault(lag, []).extend(family)
        result["evaluationConclusion"] = {f"HAC{lag}": _conclusion(combined[lag]) for lag in LAGS}
    return {"schemaVersion": 1, "status": "complete", "exploratoryPostHoc": True,
        "protocol": {"inputSha256": INPUT_SHA256, "developmentTargets": list(map(str, DEVELOP)), "validationTargets": list(map(str, VALIDATE)),
            "evaluationTargets": list(map(str, EVALUATE)), "evaluationRequiresOriginAtOrAfter": EVALUATE[0].isoformat(),
            "weights": list(WEIGHTS), "minimumSplitTargets": MIN_SPLIT, "familyPerLag": 200, "primaryLagDays": 9, "sensitivityLagDays": 28,
            "limitations": "Retrospective on revised actuals and previously inspected dates; exploratory only; not official competition or prospective performance."},
        "inputSha256": source_sha256, "results": results, "multipleTesting": {f"HAC{lag}": tests[lag] for lag in LAGS}}


def run(input_path):
    path = Path(input_path).resolve(); raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    if sha != INPUT_SHA256: raise ValueError(f"input SHA-256 mismatch: expected {INPUT_SHA256}, found {sha}")
    return analyze(json.loads(raw), sha)


def _atomic(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(content); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_report(report, json_path, markdown_path):
    _atomic(json_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    test_index = {(t["productId"], t["horizon"], t["comparator"], t["loss"]): t for t in report["multipleTesting"]["HAC9"]}
    sens_index = {(t["productId"], t["horizon"], t["comparator"], t["loss"]): t for t in report["multipleTesting"]["HAC28"]}
    lines = ["# 예측 수정폭 고정 보정 실험", "", "현재 수정본에 대한 후향적 탐색 결과입니다. 공식 타깃·점수 또는 전향 성능을 입증하지 않습니다.", "",
        "| 품목 | 일수 | 선택 | 가중치 | n | ΔMAE 지속/부모 | ΔMAPE 지속/부모 | HAC9 Holm p(지속 AE/MAPE) | HAC9 Holm p(부모 AE/MAPE) | HAC9 판정(4검정) | HAC28 판정(지속/부모) |", "|---|---:|---|---:|---:|---|---|---|---|---|---|"]
    for r in report["results"]:
        f = lambda x: "—" if x is None else f"{x:.3f}"
        base_cmp, parent_cmp = r["comparisons"]["baseline"], r["comparisons"]["parent"]
        fmtp = lambda t: "—" if t["holmAdjustedPValue"] is None else f"{t['holmAdjustedPValue']:.3g}"
        bpa = f"{fmtp(test_index[(r['productId'],r['horizon'],'baseline','absolute_error')])}/{fmtp(test_index[(r['productId'],r['horizon'],'baseline','absolute_percentage_error')])}"
        ppa = f"{fmtp(test_index[(r['productId'],r['horizon'],'parent','absolute_error')])}/{fmtp(test_index[(r['productId'],r['horizon'],'parent','absolute_percentage_error')])}"
        sens = f"{base_cmp['conclusionHAC28']}/{parent_cmp['conclusionHAC28']}"
        delta_mae = f"{f(base_cmp['meanLossDifference']['mae'])}/{f(parent_cmp['meanLossDifference']['mae'])}"
        delta_mape = f"{f(base_cmp['meanLossDifference']['mape'])}/{f(parent_cmp['meanLossDifference']['mape'])}"
        lines.append(f"| {r['productId']} | {r['horizon']} | {r['selection']['selectedModel']} | {r['selection']['weight']:.2f} | {r['coverage']['evaluationN']} | {delta_mae} | {delta_mape} | {bpa} | {ppa} | {r['evaluationConclusion']['HAC9']} | {sens} |")
    lines += ["", "Δ는 보정−대조의 쌍체 평균 손실차(원/kg 및 %)이며, HAC9 판정은 두 대조군×두 손실의 네 주 분석 검정을 모두 만족해야 개선으로 분류합니다. HAC28 열은 민감도 결론을 지속선/부모모델 순으로 표시합니다. JSON에는 각 비교의 Holm p값, 연도·계절별 세 예측의 paired 지표와 방향, 평가 가능 실제 목표일 분모 및 공통 예측 coverage가 들어 있습니다.", "", "선택은 개발기간의 동일 공통 예측·정답에서 수행하고, 확인기간의 사전 고정 관문을 통과할 때만 유지했습니다. 평가는 2024-01-01~2026-09-27 목표일, 원점 2024-01-01 이후입니다. 두 대조군(지속선 및 같은 원래 모델)×두 손실×50 품목·거리 조합의 Holm 200을 HAC9와 HAC28 각각 적용했습니다.", ""]
    _atomic(markdown_path, "\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True); ap.add_argument("--report")
    a = ap.parse_args(argv); source = Path(a.input).resolve(); out = Path(a.output).resolve(); md = Path(a.report).resolve() if a.report else out.with_suffix(".md")
    if out in (source, md) or md == source: raise ValueError("output paths must be distinct and must not overwrite input")
    write_report(run(source), out, md); return 0


if __name__ == "__main__": raise SystemExit(main())
