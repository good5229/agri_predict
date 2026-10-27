"""Frozen, answer-free K generation and retrospective scoring for AID V1.

This utility never trains or invokes models. ``generate`` reads only saved
forecast artifacts; ``score`` is the sole command that opens the target-price
snapshot and it first verifies the caller-pinned forecast artifact hash.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from . import crop_specialist_assembly_score as assembly_score
from . import research_progress_diagnostic as diagnostics
from . import full_history_return_eval as price_helper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/anchored-increment-diagnostic-v1-20261013"
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
PERIODS = {
    "V": (dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1)),
    "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1)),
}
PROTOCOL_SHA256 = "b36b904883347795dff7acb23cf393aacade5852e6f7d2e5529408c7cf839c7a"
T_FORECAST_SHA256 = "2b7b20f1e58d179253da12385507e73ff3b73ecd5ecba6c43d2e533c1eedfdc2"
T_REPLAY_SHA256 = "3d5ff814ee6d53f1a23e13b1e1f7a0e7079fb84715cf130a8c3cf472b6d2d40b"
V_SELECTION_SHA256 = "73a64eee7213f29b3fb6ba04122cc5fbabe7d27746da986dc83a5b1e7a683e6f"
KEY_FIELDS = ("product", "origin", "target", "horizon")
TOL = 1e-12


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _doc_sha(doc: Mapping[str, Any], field: str) -> str:
    unsigned = dict(doc)
    expected = unsigned.pop(field, None)
    actual = _sha(_canonical(unsigned))
    if expected != actual:
        raise ValueError(f"{field} mismatch")
    return actual


def _path(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _write_once(dest: Path, content: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen output: {dest}") from None
    with os.fdopen(fd,"wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _positive(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean prediction is invalid")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prediction is not numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("prediction must be finite and positive")
    return result


def _pred(obj: Any, label: str) -> float:
    if not isinstance(obj, Mapping) or obj.get("status") != "issued":
        raise ValueError(f"required saved prediction is not issued: {label}")
    return _positive(obj.get("prediction"))


def _compose(baseline: float, delta: float) -> tuple[float, float | str, str | None]:
    """Return (issued K, raw K encoding, technical fallback reason)."""
    raw = baseline + delta
    reason = ("delta_overflow" if not math.isfinite(delta) else
              "raw_overflow" if not math.isfinite(raw) else
              "raw_nonpositive" if raw <= 0 else None)
    return (baseline if reason else raw,
            raw if math.isfinite(raw) else "nonfinite", reason)


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        p, o, t, h = row["product"], row["origin"], row["target"], row["horizon"]
        origin, target = dt.date.fromisoformat(o), dt.date.fromisoformat(t)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed forecast key") from exc
    if (p not in PRODUCTS or h not in HORIZONS or origin != target-dt.timedelta(days=h)):
        raise ValueError("invalid product/origin/target/horizon key")
    return p, o, t, h


def _rows(doc: Mapping[str, Any], field: str = "forecasts") -> dict[tuple, dict[str, Any]]:
    rows = doc.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"{field} is missing")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed forecast row")
        key = _key(row)
        if key in out:
            raise ValueError("duplicate forecast key")
        out[key] = row
    return out


def _verify_manifest(doc: dict[str, Any], file_sha: str, phase: str) -> None:
    if doc.get("phase") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("source is not the expected answer-free V manifest")
    _doc_sha(doc, "manifestSha256")
    if not isinstance(file_sha, str) or len(file_sha) != 64:
        raise ValueError("source manifest file hash is malformed")


def _load_phase_sources(phase: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if _sha((OUT / "protocol.md").read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("frozen protocol hash changed")
    if phase == "V":
        refs = {
            "A": "outputs/experiments/external-reference-v1-20261012/V/forecast-manifest.json",
            "B": "outputs/experiments/intra-crop-panel-cand-b-v1-20261013/V/forecast-manifest.json",
        }
        result, hashes = {}, {}
        for name, rel in refs.items():
            path = _path(rel); raw = path.read_bytes(); doc = _read(path)
            digest = _sha(raw)
            expected = ("1257f6c49c48b8dbc0f8467d91d025b8dbf9f5ea24db6127b3d2f7916a87138d"
                        if name == "A" else "b5c340eca95ed5bf9e31372552fb009bce21a28e366710e2a9fba7fee5e14077")
            if digest != expected: raise ValueError(f"pinned {name} V source file hash mismatch")
            _verify_manifest(doc, digest, "V")
            if doc.get("priceSnapshotSha256", doc.get("targetSnapshotSha256")) != "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8":
                raise ValueError("V source target snapshot differs")
            result[name], hashes[name] = doc, digest
        selection_path = OUT.parent / "crop-specialist-assembly-v1-20261013/selection.json"
        selection_raw = selection_path.read_bytes(); selection = _read(selection_path)
        if _sha(selection_raw) != V_SELECTION_SHA256:
            raise ValueError("pinned V selection file hash mismatch")
        _doc_sha(selection, "selectionSha256")
        if selection.get("experiment") != "CROP_SPECIALIST_ASSEMBLY_V1" or selection.get("stage") != "V":
            raise ValueError("selection source is not the frozen CROP specialist V artifact")
        result["selection"] = selection
        hashes["selection"] = _sha(selection_raw)
        return result, hashes
    path = OUT.parent / "crop-specialist-assembly-v1-20261013/T/forecasts.json"
    replay = path.parent / "replay-verification.json"
    doc = _read(path); digest = _sha(path.read_bytes())
    if digest != T_FORECAST_SHA256: raise ValueError("pinned T forecast file hash mismatch")
    if doc.get("experiment") != "CROP_SPECIALIST_ASSEMBLY_V1" or doc.get("phase") != "T":
        raise ValueError("T source is not CROP_SPECIALIST_ASSEMBLY_V1")
    _doc_sha(doc, "manifestSha256")
    replay_doc = _read(replay)
    replay_sha = _sha(replay.read_bytes())
    if replay_sha != T_REPLAY_SHA256: raise ValueError("pinned T replay audit file hash mismatch")
    if replay_doc.get("status") != "pass":
        raise ValueError("T forecast replay verification did not pass")
    if (replay_doc.get("experiment") != doc.get("experiment") or replay_doc.get("phase") != "T"
            or replay_doc.get("forecastsFileSha256") != digest
            or replay_doc.get("forecastRowsSha256") != doc.get("forecastRowsSha256")
            or replay_doc.get("targetSnapshotFileSha256") != doc.get("targetVersion")
            or replay_doc.get("sourceManifestHashes") != {k: v.get("fileSha256") for k, v in doc.get("sources", {}).items()}):
        raise ValueError("T replay audit is not bound to the exact saved forecasts/sources/targets")
    return {"T": doc, "T_replay": replay_doc}, {"T": digest, "T_replay": replay_sha}


def _source_maps(phase: str, docs: Mapping[str, dict[str, Any]]) -> tuple[dict[str, dict], set]:
    if phase == "V":
        a = _rows(docs["A"]); b = _rows(docs["B"]); selection = docs["selection"]
        if set(a) != set(b): raise ValueError("V source forecast grids differ")
        selected = {p: selection.get("products", {}).get(p, {}).get("selectedModelId") for p in PRODUCTS}
        if any(model not in ("B", "A", "L", "P") for model in selected.values()):
            raise ValueError("V selected G policy is incomplete")
        values = {}
        for k in b:
            ar, br = a[k], b[k]
            for legacy, original in (("CAND_A", ar.get("CAND_A")), ("CTRL_LGB", ar.get("CTRL_LGB"))):
                copied = br.get("A_old", {}).get(legacy)
                if (not isinstance(copied, Mapping) or not isinstance(original, Mapping)
                        or not math.isclose(_pred(copied, f"A_old/{legacy}"),
                                            _pred(original, legacy), rel_tol=TOL, abs_tol=TOL)):
                    raise ValueError(f"V copied legacy {legacy} differs from its original saved source")
            vals = {"B": _pred(br["B"]["E106_RHS"], "B/E106_RHS"),
                         "L": _pred(br["models"]["L"], "L"),
                         "P": _pred(br["models"]["P"], "P"),
                         "A": _pred(ar["CAND_A"], "CAND_A"),
                         "CTRL": _pred(ar["CTRL_LGB"], "CTRL_LGB")}
            chosen=selected[k[0]]
            vals["G"] = vals["B"] if chosen == "B" else vals[chosen]
            values[k] = vals
        return values, set(b)
    t = _rows(docs["T"])
    values = {}
    for k, row in t.items():
        models = row["models"]
        product = k[0]
        need = {"cabbage": ("L",), "radish": ("P", "L"),
                "onion": ("A", "CTRL"), "garlic": ()}[product]
        vals = {"B": _pred(row["B"], "T/B")}
        for name in need:
            vals[name] = _pred(models.get(name), f"T/{name}")
        vals["G"] = _pred(row["G"], "T/G")
        values[k] = vals
    return values, set(t)


def generate(phase: str, output: str | Path | None = None) -> dict[str, Any]:
    """Generate the single frozen K policy without opening target values."""
    if phase not in PERIODS: raise ValueError("phase must be V or T")
    docs, source_hashes = _load_phase_sources(phase)
    values, keys = _source_maps(phase, docs)
    start, end, origin_start = PERIODS[phase]
    expected = {(p, (t-dt.timedelta(days=h)).isoformat(), t.isoformat(), h)
                for p in PRODUCTS for h in HORIZONS
                for t in (start+dt.timedelta(days=i) for i in range((end-start).days+1))
                if (t-dt.timedelta(days=h)) >= origin_start}
    if keys != expected:
        raise ValueError(f"{phase} frozen key grid mismatch: missing={len(expected-keys)}, extra={len(keys-expected)}")
    out_rows, fallback_counts = [], defaultdict(int)
    for key in sorted(keys):
        product, origin, target, horizon = key
        v = values[key]; b = v["B"]
        if product == "radish":
            left, right, dname = "L", "P", "P-L"
            delta = v[right] - v[left]
        elif product == "onion":
            left, right, dname = "CTRL", "A", "A-CTRL"
            delta = v[right] - v[left]
        else:
            left = right = dname = None; delta = 0.0
        pred, raw, fallback = _compose(b, delta)
        if fallback: fallback_counts[f"{product}:{fallback}"] += 1
        g0 = v["L"] if product == "cabbage" else v["L"] if product == "radish" else v["CTRL"] if product == "onion" else b
        row = {"product": product, "origin": origin, "target": target, "horizon": horizon,
               "deltaDefinition": dname, "delta": delta if math.isfinite(delta) else "nonfinite",
               "baselineB": b, "kRaw": raw,
               "prediction": pred, "g0Prediction": g0, "gPrediction": v["G"],
               "fallbackReason": fallback,
               "sourcePredictions": v}
        if left:
            row.update({"directD": v[left], "directE": v[right],
                        "directDeltaReproduces": math.isclose(v[left]+delta, v[right], rel_tol=TOL, abs_tol=TOL)})
            if not row["directDeltaReproduces"]:
                raise ValueError(f"{product} direct contrast does not reproduce E=D+delta")
        out_rows.append(row)
    doc = {"schemaVersion": 1, "experiment": "ANCHORED_INCREMENT_DIAGNOSTIC_V1", "phase": phase,
           "protocolSha256": PROTOCOL_SHA256, "targetValuesAttached": False,
           "sourceArtifacts": source_hashes,
           "sourcePathRefs": {k: str(v) for k,v in
             ({"A": "outputs/experiments/external-reference-v1-20261012/V/forecast-manifest.json",
               "B": "outputs/experiments/intra-crop-panel-cand-b-v1-20261013/V/forecast-manifest.json",
               "selection": "outputs/experiments/crop-specialist-assembly-v1-20261013/selection.json"}.items()
              if phase == "V" else {"T": "outputs/experiments/crop-specialist-assembly-v1-20261013/T/forecasts.json",
                                    "T_replay": "outputs/experiments/crop-specialist-assembly-v1-20261013/T/replay-verification.json"}.items())},
           "targetSnapshotSha256": (docs["B"].get("targetSnapshotSha256") if phase=="V" else docs["T"].get("targetVersion")),
           "sourceManifestHashes": ({"A":docs["A"].get("manifestSha256"),"B":docs["B"].get("manifestSha256")}
                                    if phase=="V" else {k:v.get("manifestSha256") for k,v in docs["T"].get("sources",{}).items()}),
           "generatorSha256": _sha(Path(__file__).read_bytes()),
           "keyCount": len(out_rows), "fallbackCounts": dict(sorted(fallback_counts.items())),
           "forecasts": out_rows}
    doc["forecastRowsSha256"] = _sha(_canonical(out_rows))
    doc["manifestSha256"] = _sha(_canonical(doc))
    if output is not None:
        dest = _path(output)
        _write_once(dest,_canonical(doc))
    return doc


def _weighted_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    cells = []
    for p in PRODUCTS:
        for h in HORIZONS:
            cell = [r for r in pairs if r["product"] == p and r["horizon"] == h]
            n = len(cell)
            if n:
                mae = math.fsum(abs(r["prediction"]-r["actual"]) for r in cell)/n
                bm = math.fsum(abs(r["B"]-r["actual"]) for r in cell)/n
                mape = 100*math.fsum(abs(r["prediction"]-r["actual"])/r["actual"] for r in cell)/n
                bmap = 100*math.fsum(abs(r["B"]-r["actual"])/r["actual"] for r in cell)/n
            else: mae=bm=mape=bmap=None
            cells.append({"product":p,"horizon":h,"n":n,"mae":mae,"baselineMae":bm,"mape":mape,"baselineMape":bmap})
    complete = all(c["n"] for c in cells)
    return {"cells":cells,"pairedRows":sum(c["n"] for c in cells),"supportedCells":sum(bool(c["n"]) for c in cells),
            "complete40Cells":complete,
            "mae":math.fsum(c["mae"] for c in cells)/40 if complete else None,
            "baselineMae":math.fsum(c["baselineMae"] for c in cells)/40 if complete else None,
            "mape":math.fsum(c["mape"] for c in cells)/40 if complete else None,
            "baselineMape":math.fsum(c["baselineMape"] for c in cells)/40 if complete else None}


def _holm80(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 80: raise ValueError("main inference family must contain exactly 80 tests")
    stats = sorted(range(80), key=lambda i: rows[i]["pValue"])
    running = 0.0
    for rank, index in enumerate(stats):
        running = max(running, min(1.0, (80-rank)*rows[index]["pValue"]))
        rows[index]["holmPValue"] = running
        rows[index]["holmSignificant"] = running < .05


def _inference(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    raw_tests=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            cell=[r for r in pairs if r["product"]==p and r["horizon"]==h]
            for loss in ("ae","ape"):
                diffs={dt.date.fromisoformat(r["target"]):
                       ((abs(r["prediction"]-r["actual"])-abs(r["B"]-r["actual"])) if loss=="ae" else
                        100*(abs(r["prediction"]-r["actual"])-abs(r["B"]-r["actual"]))/r["actual"])
                       for r in cell}
                if not diffs:
                    items={str(bw):{"status":"no_pairs","pValue":1.0,"n":0,"meanLossDifference":None} for bw in (9,28)}
                elif all(r["prediction"] == r["B"] for r in cell):
                    items={str(bw):{"status":"identical_prediction","pValue":1.0,"n":len(diffs),"meanLossDifference":0.0} for bw in (9,28)}
                else:
                    items={str(bw):price_helper._dm(diffs,bw) for bw in (9,28)}
                raw_tests.append({"product":p,"horizon":h,"loss":loss,"bandwidthResults":items})
    by_band={}
    for bandwidth in (9,28):
        rows=[]
        for test in raw_tests:
            rows.append({"product":test["product"],"horizon":test["horizon"],"loss":test["loss"],
                         **test["bandwidthResults"][str(bandwidth)]})
        _holm80(rows)
        by_band[str(bandwidth)]=rows
    boot=[]
    for metric in ("ae","ape"):
        for block in (14,28):
            boot.append({"metric":metric,"blockDays":block,**diagnostics.circular_block_bootstrap(
                pairs,a="prediction",b="B",metric=metric,block=block,replicates=2000,seed=0,
                calendar_start=dt.date(2026,4,1) if min((dt.date.fromisoformat(r["target"]) for r in pairs),default=dt.date.min)<dt.date(2026,7,1) else dt.date(2026,7,11),
                calendar_end=dt.date(2026,6,20) if min((dt.date.fromisoformat(r["target"]) for r in pairs),default=dt.date.min)<dt.date(2026,7,1) else dt.date(2026,9,27))})
    return {"mainFamilyByBandwidth":by_band,"bootstrap":boot,
            "protocol":{"HACBandwidths":[9,28],"HolmAlpha":.05,"HolmFamilySizePerBandwidth":80,
                        "circularBlocks":[14,28],"replicates":2000,"seed":0}}


def _residual_kind(error: float, shift: float) -> str:
    tol=TOL*max(1.,abs(error),abs(shift))
    if math.isclose(shift,0.,rel_tol=TOL,abs_tol=tol): return "unchanged"
    if math.isclose(error,0.,rel_tol=TOL,abs_tol=tol): return "baseline_exact_changed"
    if error*shift < 0: return "opposite_direction"
    if math.isclose(abs(shift),2*abs(error),rel_tol=TOL,abs_tol=tol): return "equal_absolute_error_boundary"
    if error*shift>0 and abs(shift)<2*abs(error): return "correct_direction_reduces_error"
    if error*shift>0 and abs(shift)>2*abs(error): return "correct_direction_overshoot"
    return "numerical_boundary"


def _contribution(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    counts=defaultdict(int)
    for r in pairs: counts[(r["product"],r["horizon"])] += 1
    groups=defaultdict(lambda:{"n":0,"maeGain":0.,"maeLoss":0.,"maeNet":0.,"mapeGain":0.,"mapeLoss":0.,"mapeNet":0.})
    residual=defaultdict(lambda:{"n":0,"maeGain":0.,"maeLoss":0.,"mapeGain":0.,"mapeLoss":0.})
    direct_groups=defaultdict(lambda:{"n":0,"maeGain":0.,"maeLoss":0.,"maeNet":0.,"mapeGain":0.,"mapeLoss":0.,"mapeNet":0.})
    direct_counts=defaultdict(int)
    for r in pairs:
        if r.get("D") is not None and r.get("E") is not None:
            direct_counts[(r["product"],r["horizon"])] += 1
    for r in pairs:
        key=(r["product"],r["horizon"],r["target"][:7]); n=counts[(r["product"],r["horizon"])]; denom=40*n
        eb=r["actual"]-r["B"]; s=r["prediction"]-r["B"]
        ae=abs(eb-s)-abs(eb); ape=100*ae/r["actual"]
        g=groups[key]; g["n"]+=1
        for nm,val in (("mae",ae/denom),("mape",ape/denom)):
            g[nm+"Net"]+=val; g[nm+("Loss" if val>0 else "Gain")]+=abs(val)
        kind=_residual_kind(eb,s)
        z=residual[(r["product"],kind)]; z["n"]+=1
        for nm,val in (("mae",ae/denom),("mape",ape/denom)):
            z[nm+("Loss" if val>0 else "Gain")]+=abs(val)
        if r.get("D") is not None and r.get("E") is not None:
            direct_delta=float(r["E"])-float(r["D"])
            ed=r["actual"]-float(r["D"])
            dae=abs(ed-direct_delta)-abs(ed)
            dkind=_residual_kind(ed,direct_delta)
            dz=residual[(r["product"],"direct_"+dkind)]; dz["n"]+=1
            for nm,val in (("mae",dae/denom),("mape",100*dae/r["actual"]/denom)):
                dz[nm+("Loss" if val>0 else "Gain")]+=abs(val)
            dg=direct_groups[key]; dg["n"]+=1
            for nm,val in (("mae",dae/denom),("mape",100*dae/r["actual"]/denom)):
                dg[nm+"Net"]+=val; dg[nm+("Loss" if val>0 else "Gain")]+=abs(val)
    items=[{"product":k[0],"horizon":k[1],"month":k[2],**v} for k,v in sorted(groups.items())]
    m=_weighted_metrics(pairs)
    sums={n:math.fsum(x[n+"Net"] for x in items) for n in ("mae","mape")}
    exp={"mae":m["mae"]-m["baselineMae"] if m["complete40Cells"] else None,
         "mape":m["mape"]-m["baselineMape"] if m["complete40Cells"] else None}
    if exp["mae"] is not None and any(not math.isclose(sums[k],exp[k],rel_tol=1e-9,abs_tol=1e-9) for k in sums):
        raise ValueError("contribution table does not reconcile to the 40-cell score")
    return {"productHorizonMonth":items,"netSums":sums,"macroDifference":exp,
            "reconciled":exp["mae"] is None or all(math.isclose(sums[k],exp[k],rel_tol=1e-9,abs_tol=1e-9) for k in sums),
            "residualTypes":{f"{k[0]}:{k[1]}":v for k,v in sorted(residual.items())},
            "directEminusDProductHorizonMonth":[{"product":k[0],"horizon":k[1],"month":k[2],**v} for k,v in sorted(direct_groups.items())]}


def _existing_ed_evidence(phase: str) -> dict[str, Any]:
    """Select, without recomputing, the already-frozen E-D inference rows."""
    evidence = {}
    if phase == "V":
        specs = (("radish_P_minus_L", "outputs/experiments/intra-crop-panel-cand-b-v1-20261013/V/score-report.json",
                  "P_vs_L", "radish"),
                 ("onion_A_minus_CTRL", "outputs/experiments/external-reference-v1-20261012/V/score-report.json",
                  "CAND_vs_CTRL", "onion"))
        for name, rel, comparison, product in specs:
            path=_path(rel); doc=_read(path)
            expected_manifest=("5af84bd142eb5d8dddd02550787f66ec5b83d2dfa337a7b37ea93711909e1fc1"
                               if product=="radish" else
                               "fac1f0f8bf62fab5879453eb866e151c5b6b123cda3cabd1aaf352d757224a8a")
            observed=(doc.get("manifestSha256") if product=="radish" else doc.get("forecastManifestSha256"))
            if observed != expected_manifest:
                raise ValueError(f"existing V E-D score report is not bound to its original forecast source: {name}")
            rows=[r for r in doc.get("dm",[]) if r.get("comparison")==comparison and r.get("product")==product]
            if len(rows)!=40: raise ValueError(f"existing V E-D inference is incomplete: {name}")
            evidence[name]={"path":rel,"fileSha256":_sha(path.read_bytes()),"rows":rows,
                            "sourceForecastManifestSha256":observed}
    else:
        rel="outputs/experiments/crop-specialist-assembly-v1-20261013/T/score-report.json"
        path=_path(rel); doc=_read(path)
        provenance=doc.get("inputProvenance",{}).get("forecasts",{})
        if provenance.get("fileSha256")!=T_FORECAST_SHA256 or provenance.get("manifestSha256")!="9e1c6678e40b0a798d7196565c37a1c6e90b99e2cc9892e6cd102078f3fbd3c5":
            raise ValueError("existing T E-D score report is not bound to the pinned T forecast source")
        fixed=doc.get("secondaryDiagnostics",{}).get("fixedFamilies",{})
        rows=[r for r in fixed.get("rows",[]) if r.get("product") in ("radish","onion")]
        if len(rows)!=80: raise ValueError("existing T E-D inference is incomplete")
        evidence["P_L_radish__A_CTRL_onion"]={"path":rel,"fileSha256":_sha(path.read_bytes()),
                                                "family":fixed.get("family"),"rows":rows}
    return evidence


def _intervention_summary(rows: list[dict[str, Any]], paired_keys: set[tuple]) -> dict[str, Any]:
    full=defaultdict(int); paired=defaultdict(int); by_product=defaultdict(lambda:defaultdict(int))
    requested=realized=paired_requested=paired_realized=0
    for row in rows:
        product=row["product"]; reason=row.get("fallbackReason"); raw_delta=row.get("delta")
        shift=_positive(row["prediction"])-_positive(row["baselineB"])
        if product in ("cabbage","garlic"):
            group="policy_unchanged"
        elif reason:
            group="fallback:"+reason
        elif raw_delta == 0.0:
            group="exact_zero_delta"
        elif math.isclose(shift,0.,rel_tol=TOL,abs_tol=TOL):
            group="tolerance_unchanged_nonzero_delta"
        else:
            group="nonzero_applied_delta"
        full[group]+=1; by_product[product][group]+=1
        row_requested=(not reason and product not in ("cabbage","garlic") and
                       isinstance(raw_delta,(int,float)) and not isinstance(raw_delta,bool) and raw_delta!=0.0)
        row_realized=(product not in ("cabbage","garlic") and shift!=0.0)
        requested+=int(row_requested); realized+=int(row_realized)
        if _key(row) in paired_keys: paired[group]+=1
    n=len(rows); pn=len(paired_keys)
    for row in rows:
        if _key(row) not in paired_keys: continue
        reason=row.get("fallbackReason"); raw_delta=row.get("delta")
        shift=_positive(row["prediction"])-_positive(row["baselineB"])
        paired_requested+=int(not reason and row["product"] not in ("cabbage","garlic") and
                              isinstance(raw_delta,(int,float)) and not isinstance(raw_delta,bool) and raw_delta!=0.0)
        paired_realized+=int(row["product"] not in ("cabbage","garlic") and shift!=0.0)
    return {"fullForecastGrid":{"denominator":n,"counts":dict(sorted(full.items())),
                                "requestedNonzeroDeltaRows":requested,"realizedNonzeroChangeRows":realized,
                                "aboveToleranceChangeRows":full["nonzero_applied_delta"],
                                "interventionRate":realized/n if n else None},
            "positiveActualPairedGrid":{"denominator":pn,"counts":dict(sorted(paired.items())),
                                        "requestedNonzeroDeltaRows":paired_requested,"realizedNonzeroChangeRows":paired_realized,
                                        "aboveToleranceChangeRows":paired["nonzero_applied_delta"],
                                        "interventionRate":paired_realized/pn if pn else None},
            "byProductFullGrid":{p:dict(sorted(v.items())) for p,v in sorted(by_product.items())},
            "policyUnchangedProducts":["cabbage","garlic"],
            "tolerance":"1e-12 rel/abs on realized prediction change; raw loss values are never rounded or zeroed"}


def score(phase: str, forecast_path: str | Path, expected_forecast_sha256: str,
          prices_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    if phase not in PERIODS: raise ValueError("phase must be V or T")
    fp=_path(forecast_path); raw=fp.read_bytes(); actual_sha=_sha(raw)
    if actual_sha != expected_forecast_sha256: raise ValueError("caller-pinned forecast file hash mismatch")
    doc=_read(fp); _doc_sha(doc,"manifestSha256")
    if doc.get("experiment")!="ANCHORED_INCREMENT_DIAGNOSTIC_V1" or doc.get("phase")!=phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("expected frozen answer-free K manifest")
    if doc.get("protocolSha256") != PROTOCOL_SHA256: raise ValueError("protocol hash changed")
    rows=doc.get("forecasts")
    if _sha(_canonical(rows)) != doc.get("forecastRowsSha256"): raise ValueError("forecast rows digest mismatch")
    # Pin the exact saved output before touching actual prices.
    for name, source_ref in doc.get("sourcePathRefs",{}).items():
        source_path=_path(source_ref)
        expected_source=doc.get("sourceArtifacts",{}).get(name)
        if not source_path.is_file() or _sha(source_path.read_bytes()) != expected_source:
            raise ValueError(f"frozen source artifact hash mismatch: {name}")
    pp=_path(prices_path)
    actual_bytes=pp.read_bytes()
    target_sha=doc.get("targetSnapshotSha256")
    if _sha(actual_bytes) != target_sha: raise ValueError("pinned target price snapshot hash mismatch")
    actuals=price_helper._series(json.loads(actual_bytes))
    actual_map={(p,d.isoformat()):value for p,series in actuals.items() for d,value in series.items() if value>0}
    start,end,origin_start=PERIODS[phase]
    expected={(p,(t-dt.timedelta(days=h)).isoformat(),t.isoformat(),h) for p in PRODUCTS for h in HORIZONS
              for t in (start+dt.timedelta(days=i) for i in range((end-start).days+1))
              if t-dt.timedelta(days=h)>=origin_start}
    keyed={_key(r):r for r in rows}
    if set(keyed)!=expected: raise ValueError("K forecast grid differs from full frozen phase grid")
    joined=[]
    for k,r in keyed.items():
        if r.get("fallbackReason") not in (None,"delta_overflow","raw_overflow","raw_nonpositive"): raise ValueError("unknown fallback reason")
        actual=actual_map.get((k[0],k[2]))
        if actual is None: continue
        b=_positive(r.get("baselineB")); pred=_positive(r.get("prediction"))
        g0=_positive(r.get("g0Prediction"))
        joined.append({"product":k[0],"origin":k[1],"target":k[2],"horizon":k[3],"actual":actual,
                       "B":b,"prediction":pred,"G0":g0,"G":_positive(r.get("gPrediction")),
                       "D":r.get("directD"),"E":r.get("directE"),"delta":r.get("delta"),
                       "fallbackReason":r.get("fallbackReason")})
    if not joined: raise ValueError("no positive actual targets joined")
    expected_positive = 2571 if phase == "V" else 2470
    if len(joined) != expected_positive:
        raise ValueError(f"positive-actual coverage mismatch: expected {expected_positive}, found {len(joined)}")
    m=_weighted_metrics(joined); contributions=_contribution(joined)
    if not m["complete40Cells"]:
        raise ValueError("positive actual coverage does not support the frozen 40-cell macro")
    score_rel = ("outputs/experiments/intra-crop-panel-cand-b-v1-20261013/V/score-report.json"
                 if phase == "V" else
                 "outputs/experiments/crop-specialist-assembly-v1-20261013/T/score-report.json")
    saved_score_path=_path(score_rel); saved_score=_read(saved_score_path)
    saved_base=(saved_score["comparisons"]["P_vs_B"]["macro"] if phase=="V"
                else saved_score["primaryGvsB"]["metrics"])
    saved_mape=saved_base.get("bMape"); saved_mae=saved_base.get("bMae")
    if (not math.isclose(m["baselineMape"],saved_mape,rel_tol=1e-10,abs_tol=1e-10)
            or not math.isclose(m["baselineMae"],saved_mae,rel_tol=1e-10,abs_tol=1e-10)):
        raise ValueError("rejoined baseline losses differ from the already-frozen source score report")
    g0pairs=[{**r,"prediction":r["G0"]} for r in joined]
    g0m=_weighted_metrics(g0pairs)
    gpairs=[{**r,"prediction":r["G"]} for r in joined]
    gm=_weighted_metrics(gpairs)
    telescope={"g0MinusBMape":g0m["mape"]-m["baselineMape"],
               "gMinusG0Mape":gm["mape"]-g0m["mape"],
               "gMinusBMape":gm["mape"]-m["baselineMape"],
               "g0MinusBMae":g0m["mae"]-m["baselineMae"],
               "gMinusG0Mae":gm["mae"]-g0m["mae"],
               "gMinusBMae":gm["mae"]-m["baselineMae"],
               "identityVerified":True}
    for metric in ("Mape","Mae"):
        if not math.isclose(telescope[f"g0MinusB{metric}"]+telescope[f"gMinusG0{metric}"],
                            telescope[f"gMinusB{metric}"],rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError("original G/G0 loss telescope does not reconcile")
    direct_groups=contributions["directEminusDProductHorizonMonth"]
    direct_net={"mae":math.fsum(g["maeNet"] for g in direct_groups),
                "mape":math.fsum(g["mapeNet"] for g in direct_groups)}
    direct_matches_g_g0={"mae":math.isclose(direct_net["mae"],telescope["gMinusG0Mae"],rel_tol=1e-9,abs_tol=1e-9),
                         "mape":math.isclose(direct_net["mape"],telescope["gMinusG0Mape"],rel_tol=1e-9,abs_tol=1e-9)}
    if not all(direct_matches_g_g0.values()):
        raise ValueError("row-level E-D loss contribution does not reproduce saved G-G0")
    infer=_inference(joined)
    # Existing E-D evidence is linked as immutable references; not recalculated here.
    refs=_existing_ed_evidence(phase)
    target_gate=(m["mape"] < m["baselineMape"] and m["mae"] <= m["baselineMae"] and m["complete40Cells"]) if m["complete40Cells"] else False
    paired_keys={_key(r) for r in joined}
    result={"schemaVersion":1,"experiment":"ANCHORED_INCREMENT_DIAGNOSTIC_V1","phase":phase,
            "forecastManifestSha256":doc["manifestSha256"],"forecastFileSha256":actual_sha,
            "targetSnapshotSha256":_sha(actual_bytes),"targetSnapshotPath":str(pp.relative_to(ROOT)),
            "protocolSha256":PROTOCOL_SHA256,"keyCount":len(keyed),"positiveActualPairs":len(joined),
            "unavailableTargetKeys":len(keyed)-len(joined),"metrics40Cell":m,
            "baselineSourceScore":{"path":score_rel,"fileSha256":_sha(saved_score_path.read_bytes()),
                                    "savedBaselineMae":saved_mae,"savedBaselineMape":saved_mape,
                                    "rejoinedBaselineMatches":True},
            "originalGMetrics40Cell":gm,
            "g0Metrics40Cell":g0m,
            "originalGTelescope":telescope,
            "directDeltaVsBaseline":{"eMinusDWeightedContribution":direct_net,
                                      "matchesOriginalGMinusG0":direct_matches_g_g0},
            "kAppliedIntervention":_intervention_summary(rows,paired_keys),
            "decisionGate":"completed_exploratory_gain" if target_gate else "completed_no_incremental_gain",
            "contributions":contributions,"inference":infer,"existingEDEvidence":refs,
            "interpretation":"retrospective diagnostic on previously inspected V/T; not independent confirmation or causal evidence"}
    result["reportSha256"]=_sha(_canonical(result))
    if output is not None:
        dest=_path(output)
        _write_once(dest,_canonical(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate"); g.add_argument("phase",choices=PERIODS); g.add_argument("--output",required=True)
    s=sub.add_parser("score"); s.add_argument("phase",choices=PERIODS); s.add_argument("--forecasts",required=True)
    s.add_argument("--forecast-sha256",required=True); s.add_argument("--prices",required=True); s.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="generate": doc=generate(args.phase,args.output)
    else: doc=score(args.phase,args.forecasts,args.forecast_sha256,args.prices,args.output)
    print(json.dumps({"phase":doc["phase"],"output":args.output,"sha256":doc.get("manifestSha256",doc.get("reportSha256"))},ensure_ascii=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
