"""Read-only V9 movement-subgroup and C-clipping supplement; performs no fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/experiments/period-v9-run-now-correction"
CROPS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = (2, 3)
METHODS = ("BASE", "P", "B0", "C", "E")
MOVE_THRESHOLD = 0.20


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _score(rows: list[dict], method: str) -> dict:
    cells = {}
    for crop in CROPS:
        for horizon in HORIZONS:
            cell = [r for r in rows if r["crop"] == crop and int(r["horizon"]) == horizon
                    and _finite(r.get("actual")) is not None and _finite(r.get(method)) is not None]
            if not cell:
                cells[f"{crop}:h{horizon}"] = {"n": 0, "mae": None, "mape_pct": None}
                continue
            errors = [abs(float(r[method])-float(r["actual"])) for r in cell]
            cells[f"{crop}:h{horizon}"] = {"n": len(cell), "mae": float(np.mean(errors)),
                                            "mape_pct": float(100*np.mean([e/float(r["actual"]) for e, r in zip(errors, cell)]))}
    present = [v for v in cells.values() if v["n"]]
    return {"cells": cells,
            "equalCellMae": float(np.mean([v["mae"] for v in present])) if present else None,
            "equalCellMapePct": float(np.mean([v["mape_pct"] for v in present])) if present else None,
            "scoredRows": sum(v["n"] for v in cells.values())}


def _paired(rows: list[dict], candidate: str, baseline: str) -> dict:
    paired = [r for r in rows if _finite(r.get("actual")) is not None
              and _finite(r.get(candidate)) is not None and _finite(r.get(baseline)) is not None]
    c, b = _score(paired, candidate), _score(paired, baseline)
    return {"candidate": candidate, "baseline": baseline, "pairedRows": len(paired),
            "candidateScore": c, "baselineScore": b,
            "equalCellMaeDifference": c["equalCellMae"]-b["equalCellMae"] if c["equalCellMae"] is not None and b["equalCellMae"] is not None else None,
            "equalCellMapePctPointDifference": c["equalCellMapePct"]-b["equalCellMapePct"] if c["equalCellMapePct"] is not None and b["equalCellMapePct"] is not None else None}


def build(run: Path = RUN) -> dict:
    forecast_path = run / "forecasts.parquet"
    score_path = run / "score-report.json"
    manifest_path = run / "source-manifest.json"
    for path in (forecast_path, score_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forecast_sha, score_sha = _sha(forecast_path), _sha(score_path)
    if manifest.get("outputs", {}).get("forecasts.parquet", {}).get("sha256") != forecast_sha:
        raise ValueError("forecast SHA does not match the V9 source manifest")
    if manifest.get("outputs", {}).get("score-report.json", {}).get("sha256") != score_sha:
        raise ValueError("score-report SHA does not match the V9 source manifest")
    score_report = json.loads(score_path.read_text(encoding="utf-8"))
    if score_report.get("status") != "completed_no_gain" or score_report.get("fitCount") != 110:
        raise ValueError("the completed V9 run does not match the expected read-only audit target")
    rows = pd.read_parquet(forecast_path).to_dict(orient="records")
    if len(rows) != 792:
        raise ValueError(f"expected 792 forecast rows, found {len(rows)}")
    key_fields = ("crop", "horizon", "targetStart", "targetEnd", "origin")
    keys = [tuple(str(r[k]) for k in key_fields) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate forecast keys")
    usable = []
    for row in rows:
        actual, reference = _finite(row.get("actual")), _finite(row.get("p_ref_restored"))
        if actual is not None and reference is not None and actual > 0 and reference > 0:
            row["_move"] = actual/reference-1.
            usable.append(row)
    subgroups = {
        "nonrapid": [r for r in usable if abs(r["_move"]) < MOVE_THRESHOLD],
        "all_up": [r for r in usable if r["_move"] > 1e-12],
        "all_down": [r for r in usable if r["_move"] < -1e-12],
        "flat": [r for r in usable if abs(r["_move"]) <= 1e-12],
    }
    subgroup_output = {}
    for name, subset in subgroups.items():
        subgroup_output[name] = {
            "forecastRowsWithLabelAndReference": len(subset),
            "distinctTargetPeriods": len({(r["targetStart"], r["targetEnd"]) for r in subset}),
            "scoresByMethod": {method: _score(subset, method) for method in METHODS},
            "pairedCMinusBASE": _paired(subset, "C", "BASE"),
            "pairedEMinusC": _paired(subset, "E", "C"),
        }
    adjustment_pairs = [(a, b) for r in rows
                        if (a := _finite(r.get("C_raw_adjustment"))) is not None
                        and (b := _finite(r.get("C_used_adjustment"))) is not None]
    clipped = [(a, b) for a, b in adjustment_pairs if not math.isclose(a, b, rel_tol=0, abs_tol=1e-14)]
    return {
        "status": "completed_read_only_supplement",
        "sourceArtifacts": {"forecastsSha256": forecast_sha, "scoreReportSha256": score_sha,
                            "v9RunStatus": score_report["status"], "v9FitCount": score_report["fitCount"]},
        "movementDefinition": {"basis": "actual / p_ref_restored - 1",
                               "rapidThresholdAbsolute": MOVE_THRESHOLD,
                               "flatToleranceAbsolute": 1e-12,
                               "rowsMissingPositiveActualOrReference": len(rows)-len(usable)},
        "subgroups": subgroup_output,
        "CAdjustmentClipping": {"rowsWithStoredAdjustment": len(adjustment_pairs),
                                "clippedRows": len(clipped),
                                "unclippedRows": len(adjustment_pairs)-len(clipped),
                                "fallbackRows": sum(r.get("C_reason") != "ridge" for r in rows),
                                "logAdjustmentBounds": [math.log(.80), math.log(1.20)]},
        "interpretationLimit": "Descriptive post-fit subgroup scores only; these slices do not add independent validation or significance tests.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build(args.run)
    output = args.out or args.run / "supplementary-diagnostics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "subgroups": list(report["subgroups"]),
                      "CClippedRows": report["CAdjustmentClipping"]["clippedRows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
