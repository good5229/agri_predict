"""Integrity-gated scoring for CROP_SPECIALIST_ASSEMBLY_V1.

Scoring consumes the frozen validation selector, a self-hashed T forecast
artifact, its replay audit, and the pinned price snapshot. It never fits a
model. Target availability and forecast issuance are tracked separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from . import equal_info_online_score as stats
from . import full_history_return_eval as price_helper
from . import research_progress_diagnostic as diagnostics

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
START, END = dt.date(2026, 7, 11), dt.date(2026, 9, 27)
ORIGIN_START = dt.date(2026, 7, 1)
BANDWIDTHS, BLOCKS, REPLICATES, SEED = (9, 28), (14, 28), 2000, 0


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document_sha(document: Mapping[str, Any], field: str) -> str:
    unsigned = dict(document)
    saved = unsigned.pop(field, None)
    actual = _sha(_canonical(unsigned))
    if saved != actual:
        raise ValueError(f"{field} mismatch")
    return actual


def _date(value: Any) -> dt.date:
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else (_ for _ in ()).throw(ValueError())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date {value!r}") from exc


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _forecast(obj: Any, name: str) -> tuple[str, float | None]:
    if not isinstance(obj, Mapping) or obj.get("status") not in ("issued", "not_issued"):
        raise ValueError(f"{name} has an invalid status")
    raw = obj.get("prediction")
    if obj["status"] == "issued":
        value = _positive(raw)
        if value is None:
            raise ValueError(f"{name} issued status has no finite positive prediction")
        return "issued", value
    if raw is not None:
        raise ValueError(f"{name} not_issued status contains a numerical prediction")
    return "not_issued", None


def _rooted_path(value: Any, *, base: Path | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact reference has no path")
    base = base or ROOT
    path = (base / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"artifact path escapes project root: {value}")
    return path


def _verify_ref(ref: Mapping[str, Any], *, hash_field: str | None = None,
                base: Path | None = None) -> Path:
    path = _rooted_path(ref.get("path"), base=base)
    digest = ref.get(hash_field) if hash_field else ref.get("fileSha256", ref.get("sha256"))
    if not path.is_file() or _sha(path.read_bytes()) != digest:
        raise ValueError(f"artifact file hash mismatch: {ref.get('path')}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_selection(selection: dict[str, Any], selection_file_sha: str | None = None,
                        *, verify_source_files: bool = False) -> dict[str, str]:
    _document_sha(selection, "selectionSha256")
    if selection.get("experiment") != "CROP_SPECIALIST_ASSEMBLY_V1" or selection.get("stage") != "V":
        raise ValueError("selection artifact is not CROP_SPECIALIST_ASSEMBLY_V1 V")
    target = selection.get("targetSnapshot")
    if not isinstance(target, Mapping) or target.get("sha256") != selection.get("targetVersion"):
        raise ValueError("selection target version is not bound to its snapshot")
    if verify_source_files:
        _verify_ref(target)
    if selection_file_sha is not None and not selection_file_sha:
        raise ValueError("selection file hash is empty")
    products = selection.get("products")
    if not isinstance(products, Mapping) or set(products) != set(PRODUCTS):
        raise ValueError("selection product map is incomplete")
    selected = {}
    for product in PRODUCTS:
        item = products[product]
        model = item.get("selectedModelId") if isinstance(item, Mapping) else None
        if model not in ("B", "A", "L", "P"):
            raise ValueError(f"invalid selected V policy for {product}")
        selected[product] = model
    sources = selection.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"A", "B", "L", "P"}:
        raise ValueError("selection source map is incomplete")
    source_docs: dict[str, dict[str, Any]] = {}
    for name, ref in sources.items():
        if verify_source_files:
            path = _verify_ref(ref)
            doc = _load_json(path)
            if doc.get("manifestSha256") != ref.get("manifestSha256"):
                raise ValueError(f"{name} V source manifest digest binding mismatch")
            copied_version = doc.get("priceSnapshotSha256") if name == "A" else doc.get("targetSnapshotSha256")
            if copied_version != selection.get("targetVersion"):
                raise ValueError(f"{name} V source target version differs from frozen target snapshot")
            if name in ("A", "B"):
                from .chronos2_market_context import BASELINE_SHA256
                if doc.get("baselineManifestSha256") != BASELINE_SHA256["V"]:
                    raise ValueError(f"{name} V source is not bound to pinned RHS manifest")
            if ref.get("forecastCount") != len(doc.get("forecasts", [])):
                raise ValueError(f"{name} V source row count differs from selection provenance")
            source_docs[name] = doc
    # The copied B forecast is checked against the independently pinned raw RHS.
    if verify_source_files:
        baseline = selection.get("pinnedRhsBaseline")
        if not isinstance(baseline, Mapping):
            raise ValueError("pinned raw V RHS baseline reference is missing")
        baseline_doc = _load_json(_verify_ref(baseline))
        from . import external_reference_v1 as source_a
        raw_index = source_a._baseline_index(baseline_doc, "V")
        copied = source_docs["B"].get("forecasts", [])
        indexed = {(str(row.get("product")), str(row.get("target")), int(row.get("horizon"))): row
                   for row in copied if isinstance(row, Mapping)}
        for key, raw in raw_index.items():
            row = indexed.get((key[0], key[1].isoformat(), key[2]))
            payload = row.get("B", row.get("baseline", {})) if isinstance(row, Mapping) else {}
            payload = payload.get("E106_RHS", {}) if isinstance(payload, Mapping) else {}
            expected = _positive(raw.get("E106_RHS"))
            got = _positive(payload.get("prediction")) if isinstance(payload, Mapping) else None
            if got != expected or (payload.get("status") == "issued") != (expected is not None):
                raise ValueError("selection B rows differ from pinned raw V RHS forecasts")
        if len(indexed) != len(raw_index):
            raise ValueError("selection B forecast grid differs from pinned raw V RHS")
        snapshot_path = _rooted_path(target.get("path"))
        if _sha(snapshot_path.read_bytes()) != selection.get("targetVersion"):
            raise ValueError("selection target snapshot bytes differ from targetVersion")
        recomputed = _recompute_v_map(source_docs, _prices(_load_json(snapshot_path)),
                                      str(selection["targetVersion"]))
        if recomputed != selected:
            raise ValueError(f"frozen V policy map is not reproducible from pinned sources: {recomputed}")
    return selected


def _prices(document: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    if document.get("schemaVersion") is None:
        raise ValueError("pinned price snapshot schema is missing")
    products = document.get("products")
    if not isinstance(products, list):
        raise ValueError("price snapshot products are missing")
    actuals: dict[tuple[str, str], float] = {}
    for p in products:
        if not isinstance(p, Mapping) or p.get("id") not in PRODUCTS:
            continue
        for observation in p.get("observations", []):
            if not isinstance(observation, Mapping):
                continue
            actual = _positive(observation.get("actual"))
            if actual is not None:
                key = (str(p["id"]), _date(observation.get("date")).isoformat())
                if key in actuals and actuals[key] != actual:
                    raise ValueError(f"conflicting target snapshot value: {key}")
                actuals[key] = actual
    return actuals


def _target_observations(document: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Retain every snapshot target state; null, absent, and nonpositive are distinct."""
    if document.get("schemaVersion") is None:
        raise ValueError("pinned price snapshot schema is missing")
    products = document.get("products")
    if not isinstance(products, list):
        raise ValueError("price snapshot products are missing")
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        if not isinstance(product, Mapping) or product.get("id") not in PRODUCTS:
            continue
        for observation in product.get("observations", []):
            if not isinstance(observation, Mapping):
                continue
            day = _date(observation.get("date")).isoformat()
            if "actual" not in observation:
                state, raw = "actual_field_missing", None
            else:
                raw = observation.get("actual")
                if raw is None:
                    state = "null_actual"
                elif isinstance(raw, bool):
                    state = "invalid_actual"
                else:
                    try:
                        numeric = float(raw)
                    except (TypeError, ValueError, OverflowError):
                        state, numeric = "invalid_actual", None
                    if numeric is not None:
                        state = ("nonfinite_actual" if not math.isfinite(numeric) else
                                 "positive_actual" if numeric > 0 else
                                 "zero_actual" if numeric == 0 else "negative_actual")
            result[(str(product["id"]), day)].append({"status": state, "actualRaw": raw})
    return dict(result)


def _selection_source_rows(document: Mapping[str, Any], name: str,
                           target_version: str) -> dict[tuple[str, str, str, int, str], tuple[str, float | None]]:
    if document.get("phase") != "V" or document.get("targetValuesAttached") is not False:
        raise ValueError(f"{name} is not an answer-free V source")
    unsigned = dict(document)
    digest = unsigned.pop("manifestSha256", None)
    if digest != _sha(_canonical(unsigned)):
        raise ValueError(f"{name} V source self-hash mismatch")
    result = {}
    for row in document.get("forecasts", []):
        product = row.get("product")
        origin, target, horizon = _date(row.get("origin")), _date(row.get("target")), row.get("horizon")
        if product not in PRODUCTS or origin > target or horizon not in HORIZONS or target-origin != dt.timedelta(days=horizon):
            raise ValueError(f"invalid {name} V source row")
        if origin < dt.date(2026, 4, 1) or target < dt.date(2026, 4, 2) or target > dt.date(2026, 6, 20):
            continue
        key = (product, origin.isoformat(), target.isoformat(), horizon, target_version)
        if key in result:
            raise ValueError(f"duplicate {name} V exact join key")
        if name == "B":
            outer = row.get("B", row.get("baseline", {}))
            obj = outer.get("E106_RHS", {}) if isinstance(outer, Mapping) else {}
        elif name == "A":
            obj = row.get("CAND_A", {})
        else:
            models = row.get("models", {})
            obj = models.get(name, {}) if isinstance(models, Mapping) else {}
            if name == "P" and isinstance(obj, Mapping):
                if obj.get("modelUsed") is False:
                    obj = {"status": "not_issued", "prediction": None}
                elif obj.get("rawModelStatus") is not None:
                    obj = {"status": obj.get("rawModelStatus"), "prediction": obj.get("rawModelPrediction")}
        if not isinstance(obj, Mapping) or obj.get("status") not in ("issued", "not_issued"):
            raise ValueError(f"{name} V forecast status is malformed")
        value = _positive(obj.get("prediction"))
        if (obj["status"] == "issued") != (value is not None):
            raise ValueError(f"{name} V status/prediction mismatch")
        result[key] = (str(obj["status"]), value)
    return result


def _recompute_v_map(source_docs: Mapping[str, dict[str, Any]], actuals: Mapping[tuple[str, str], float],
                     target_version: str) -> dict[str, str]:
    sources = {name: _selection_source_rows(source_docs[name], name, target_version)
               for name in ("B", "A", "L", "P")}
    key_set = set(sources["B"])
    if not key_set or any(set(sources[name]) != key_set for name in ("A", "L", "P")):
        raise ValueError("V source rows do not join on the exact common key set")
    chosen = {}
    for product in PRODUCTS:
        per_model = {}
        coverage = {}
        for name in ("B", "A", "L", "P"):
            by_h = defaultdict(list)
            all_keys = [key for key in key_set if key[0] == product]
            positive_actual_keys = [key for key in all_keys if (product, key[2]) in actuals]
            issued_all = {key for key in all_keys if sources[name][key][0] == "issued"}
            issued_positive = {key for key in positive_actual_keys if sources[name][key][0] == "issued"}
            coverage[name] = {"all": issued_all, "positive": len(issued_positive),
                              "positiveTargetCount": len(positive_actual_keys)}
            for key in issued_positive:
                y = actuals[(product, key[2])]
                pred = sources[name][key][1]
                by_h[key[3]].append((y, pred))
            mape, mae = [], []
            for h in HORIZONS:
                pairs = by_h.get(h, [])
                if not pairs:
                    raise ValueError(f"{product}/{name}/h{h}: no positive V forecast pairs")
                mape.append(100*math.fsum(abs(y-p)/y for y,p in pairs)/len(pairs))
                mae.append(math.fsum(abs(y-p) for y,p in pairs)/len(pairs))
            per_model[name] = (math.fsum(mape)/10, math.fsum(mae)/10)
        eligible = []
        for name in ("A", "L", "P"):
            cand, base = sources[name], sources["B"]
            common_by_h = defaultdict(lambda: [[], []])
            for key in key_set:
                if key[0] != product or (product, key[2]) not in actuals:
                    continue
                if cand[key][0] == base[key][0] == "issued":
                    y = actuals[(product, key[2])]
                    common_by_h[key[3]][0].append((y, cand[key][1]))
                    common_by_h[key[3]][1].append((y, base[key][1]))
            if any(not common_by_h[h][0] for h in HORIZONS):
                continue
            left_mape, right_mape, left_mae, right_mae = [], [], [], []
            for h in HORIZONS:
                left, right = common_by_h[h]
                left_mape.append(100*math.fsum(abs(y-p)/y for y,p in left)/len(left))
                right_mape.append(100*math.fsum(abs(y-p)/y for y,p in right)/len(right))
                left_mae.append(math.fsum(abs(y-p) for y,p in left)/len(left))
                right_mae.append(math.fsum(abs(y-p) for y,p in right)/len(right))
            no_all_loss = coverage[name]["all"].issuperset(coverage["B"]["all"])
            no_positive_loss = coverage[name]["positive"] >= coverage["B"]["positive"]
            if (no_all_loss and no_positive_loss
                    and math.fsum(left_mape)/10 < math.fsum(right_mape)/10
                    and math.fsum(left_mae)/10 <= math.fsum(right_mae)/10):
                eligible.append(name)
        chosen[product] = min(eligible, key=lambda name: (per_model[name][0], per_model[name][1],
                                                           {"B": 0, "L": 1, "A": 2, "P": 3}[name])) if eligible else "B"
    return chosen


def _validate_forecasts(doc: dict[str, Any], selected: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _document_sha(doc, "manifestSha256")
    if doc.get("experiment") != "CROP_SPECIALIST_ASSEMBLY_V1" or doc.get("stage") != "T":
        raise ValueError("expected CROP_SPECIALIST_ASSEMBLY_V1 T forecast artifact")
    if doc.get("targetValuesAttached") is not False:
        raise ValueError("T forecast artifact contains attached target values")
    rows = doc.get("forecasts")
    if not isinstance(rows, list):
        raise ValueError("T forecasts are missing")
    if doc.get("forecastRowsSha256") != _sha(_canonical(rows)):
        raise ValueError("T forecast row digest mismatch")
    seen = set()
    output = []
    counts = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed T forecast row")
        product, origin, target, horizon = row.get("product"), _date(row.get("origin")), _date(row.get("target")), row.get("horizon")
        key = (product, origin, target, horizon)
        if (product not in PRODUCTS or horizon not in HORIZONS or key in seen
                or target - origin != dt.timedelta(days=horizon)
                or origin < ORIGIN_START or target < START or target > END):
            raise ValueError(f"invalid T forecast identity: {key}")
        if (row.get("fit_month") != origin.strftime("%Y-%m")
                or row.get("cutoff") != (origin-dt.timedelta(days=1)).isoformat()):
            raise ValueError("T forecast fit month/cutoff differs from its origin")
        seen.add(key)
        if row.get("target_version") != doc.get("targetVersion"):
            raise ValueError("T forecast target version differs from manifest")
        if row.get("selected_model_id") != selected[product]:
            raise ValueError("T row uses a policy different from frozen V selection")
        if row.get("selected_policy_id") != f"G:{product}:{selected[product]}":
            raise ValueError("T forecast has an invalid selected policy identity")
        b_status, b_pred = _forecast(row.get("B"), "B")
        models = row.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("T direct model predictions are missing")
        planned = doc.get("plannedModelIdsByProduct", {}).get(product)
        if not isinstance(planned, list) or set(models) != set(planned):
            raise ValueError("T direct model map differs from its planned fit map")
        for model_name, model_obj in models.items():
            _forecast(model_obj, f"{model_name} direct model")
            if product == "radish" and model_name in ("P", "L") or product == "onion" and model_name in ("A", "CTRL"):
                if model_obj.get("status") == "issued":
                    ratio, p0, p0_date = model_obj.get("logRatio"), _positive(model_obj.get("p0")), _date(model_obj.get("p0Date"))
                    cutoff = _date(row.get("cutoff"))
                    if (not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not math.isfinite(ratio)
                            or p0 is None or p0_date is None or p0_date > cutoff
                            or not math.isclose(float(model_obj["prediction"]), p0 * math.exp(float(ratio)),
                                                rel_tol=1e-10, abs_tol=1e-10)):
                        raise ValueError(f"{product}/{model_name} lacks causal log-price evidence")
        if selected[product] == "B":
            raw_status, raw_pred = b_status, b_pred
        else:
            model_obj = models.get(selected[product])
            raw_status, raw_pred = _forecast(model_obj, f"{selected[product]} direct model")
            if model_obj.get("rawModelStatus") is not None and model_obj["rawModelStatus"] != raw_status:
                raise ValueError("selected direct-model status conflicts with its raw model status")
        policy = row.get("G")
        if not isinstance(policy, Mapping):
            raise ValueError("G policy forecast is missing")
        if policy.get("selectedModelId") != selected[product]:
            raise ValueError("G policy selected model differs from frozen V map")
        g_label = policy.get("status")
        if g_label not in ("issued", "fallback", "not_issued"):
            raise ValueError("G policy has an invalid status")
        policy_pred = _positive(policy.get("prediction"))
        if (g_label in ("issued", "fallback")) != (policy_pred is not None):
            raise ValueError("G policy status/prediction mismatch")
        policy_status = "issued" if g_label in ("issued", "fallback") else "not_issued"
        if selected[product] == "B":
            if (policy_status != b_status or policy_pred != b_pred or policy.get("modelUsed") is not False
                    or policy.get("rawModelStatus") != "not_applicable"
                    or policy.get("rawModelPrediction") is not None):
                raise ValueError("B-selected G policy differs from pinned B forecast")
        elif raw_status == "issued":
            if (g_label != "issued" or policy.get("fallbackTo") is not None
                    or policy.get("modelUsed") is not True
                    or policy.get("rawModelStatus") != "issued"
                    or not math.isclose(_positive(policy.get("rawModelPrediction")) or float("nan"), raw_pred,
                                        rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(policy_pred, raw_pred, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("G issued output is inconsistent with direct selected-model output")
        elif b_status == "issued":
            if (g_label != "fallback" or policy.get("fallbackTo") != "B"
                    or policy.get("modelUsed") is not False
                    or policy.get("rawModelStatus") != "not_issued"
                    or policy.get("rawModelPrediction") is not None
                    or not policy.get("fallbackReason") or not math.isclose(policy_pred, b_pred, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("G fallback does not reproduce B with a reason")
        elif (g_label != "not_issued" or policy.get("fallbackTo") is not None
              or policy.get("modelUsed") is not False
              or policy.get("rawModelStatus") != "not_issued"
              or policy.get("rawModelPrediction") is not None):
            raise ValueError("G must remain withheld when selected model and B are unavailable")
        if any(field in row for field in ("actual", "actualPrice", "targetValue")):
            raise ValueError("T forecast row contains a target value")
        for name, status in (("B", b_status), ("G", policy_status), ("direct", raw_status)):
            counts[f"{name}_{status}"] += 1
        direct_status, direct_pred = ((raw_status, raw_pred) if selected[product] != "B"
                                      else ("not_issued", None))
        output.append({**row, "_key": key, "_b": b_pred, "_b_status": b_status,
                       "_g": policy_pred, "_g_status": policy_status,
                       "_direct": direct_pred, "_direct_status": direct_status})
    expected = {(p, t - dt.timedelta(days=h), t, h)
                for p in PRODUCTS for h in HORIZONS
                for i in range((END - START).days + 1)
                for t in (START + dt.timedelta(days=i),)
                if t - dt.timedelta(days=h) >= ORIGIN_START}
    if seen != expected:
        raise ValueError(f"T forecast grid mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}")
    return output, dict(counts)


def _pairs(rows: list[dict[str, Any]], actuals: Mapping[tuple[str, str], float],
           left: str, right: str, *, products: tuple[str, ...] = PRODUCTS) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["product"] not in products:
            continue
        actual = actuals.get((row["product"], row["target"]))
        a, b = row.get(left), row.get(right)
        if actual is not None and a is not None and b is not None:
            result.append({"product": row["product"], "horizon": row["horizon"],
                           "origin": row["origin"], "target": row["target"],
                           "actual": actual, "a": a, "b": b})
    return result


def _metrics(pairs: list[dict[str, Any]], products: tuple[str, ...] = PRODUCTS) -> dict[str, Any]:
    cells = []
    for product in products:
        for h in HORIZONS:
            values = [r for r in pairs if r["product"] == product and r["horizon"] == h]
            if values:
                am = math.fsum(abs(r["a"]-r["actual"]) for r in values) / len(values)
                bm = math.fsum(abs(r["b"]-r["actual"]) for r in values) / len(values)
                ap = 100 * math.fsum(abs(r["a"]-r["actual"])/r["actual"] for r in values) / len(values)
                bp = 100 * math.fsum(abs(r["b"]-r["actual"])/r["actual"] for r in values) / len(values)
            else:
                am = bm = ap = bp = None
            cells.append({"product": product, "horizon": h, "n": len(values),
                          "gMae": am, "bMae": bm, "gMape": ap, "bMape": bp})
    complete = all(c["n"] > 0 for c in cells)
    return {"cells": cells, "supportedCells": sum(c["n"] > 0 for c in cells),
            "pairedRows": sum(c["n"] for c in cells), "complete": complete,
            "gMae": math.fsum(c["gMae"] for c in cells)/len(cells) if complete else None,
            "bMae": math.fsum(c["bMae"] for c in cells)/len(cells) if complete else None,
            "gMape": math.fsum(c["gMape"] for c in cells)/len(cells) if complete else None,
            "bMape": math.fsum(c["bMape"] for c in cells)/len(cells) if complete else None}


def _direct_metrics(pairs: list[dict[str, Any]], selected: Mapping[str, str]) -> dict[str, Any]:
    products = tuple(p for p in PRODUCTS if selected[p] != "B")
    if not products:
        return {"selectedProducts": [], "supportedProducts": [], "metrics": None,
                "plannedCells": 0, "interpretation": "no_direct_candidate_selected"}
    per_product = {p: _metrics(pairs, (p,)) for p in products}
    aggregate = _metrics(pairs, products)
    return {"selectedProducts": list(products),
            "supportedProducts": [p for p, m in per_product.items() if m["pairedRows"]],
            "plannedCells": len(products) * len(HORIZONS),
            "supportedCells": aggregate["supportedCells"],
            "pairedRows": aggregate["pairedRows"], "metrics": aggregate,
            "byProduct": per_product,
            "interpretation": ("direct selected-model-use subset vs B on common issued positive-actual pairs; "
                               "denominator is selectedProducts × 10 horizons, excludes B-selected products and "
                               "is not the 40-cell policy macro")}


def _dm(pairs: list[dict[str, Any]], products: tuple[str, ...], family: str) -> dict[str, Any]:
    rows = []
    for product in products:
        for h in HORIZONS:
            cell = [r for r in pairs if r["product"] == product and r["horizon"] == h]
            for loss in ("mae", "mape"):
                diffs = {}
                for r in cell:
                    day = _date(r["target"])
                    ae = abs(r["a"]-r["actual"]) - abs(r["b"]-r["actual"])
                    diffs[day] = ae if loss == "mae" else 100*ae/r["actual"]
                for bandwidth in BANDWIDTHS:
                    stat = stats._dm(diffs, bandwidth)
                    rows.append({"family": family, "product": product, "horizon": h,
                                 "loss": loss, "bandwidth": bandwidth, **stat})
    for bandwidth in BANDWIDTHS:
        subset = [r for r in rows if r["bandwidth"] == bandwidth]
        stats._holm(subset)
    return {"rows": rows, "testsPerBandwidth": len(products)*len(HORIZONS)*2,
            "bandwidths": list(BANDWIDTHS), "family": family}


def _contributions(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"maeGain": 0.0, "maeLoss": 0.0, "mapeGain": 0.0,
                 "mapeLoss": 0.0, "maeNet": 0.0, "mapeNet": 0.0, "n": 0})
    cell_counts = defaultdict(int)
    for r in pairs:
        cell_counts[(r["product"], r["horizon"])] += 1
    for r in pairs:
        group = grouped[(r["product"], r["horizon"], r["target"][:7])]
        denom = 40 * cell_counts[(r["product"], r["horizon"])]
        mae = (abs(r["a"]-r["actual"]) - abs(r["b"]-r["actual"])) / denom
        mape = mae / r["actual"] * 100
        for name, delta in (("mae", mae), ("mape", mape)):
            group[name+"Net"] += delta
            group[name+("Loss" if delta > 0 else "Gain")] += abs(delta)
        group["n"] += 1
    items = [{"product": k[0], "horizon": k[1], "month": k[2], **v}
             for k, v in sorted(grouped.items())]
    metrics = _metrics(pairs)
    sums = {key: math.fsum(row[key+"Net"] for row in items) for key in ("mae", "mape")}
    expected = {"mae": metrics["gMae"]-metrics["bMae"] if metrics["complete"] else None,
                "mape": metrics["gMape"]-metrics["bMape"] if metrics["complete"] else None}
    verified = {key: expected[key] is not None and math.isclose(sums[key], expected[key], rel_tol=1e-9, abs_tol=1e-9) for key in sums}
    if expected["mae"] is not None and not all(verified.values()):
        raise ValueError("month/product/horizon contributions fail macro reconciliation")
    return {"groups": items, "netSums": sums, "macroDifference": expected, "verified": verified}


def _eval_log_decomposition(rows: list[dict[str, Any]], product: str,
                            left_model: str, right_model: str) -> dict[str, Any]:
    sums = {"n": 0, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
            "squaredErrorDifferenceSum": 0.0}
    for row in rows:
        if row.get("product") != product or row.get("actual") is None:
            continue
        left, right = row.get("models", {}).get(left_model), row.get("models", {}).get(right_model)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        if left.get("status") != "issued" or right.get("status") != "issued":
            continue
        a, b = left.get("logRatio"), right.get("logRatio")
        p0_left, p0_right = _positive(left.get("p0")), _positive(right.get("p0"))
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in (a, b)):
            raise ValueError("issued secondary model lacks finite log-ratio evidence")
        if p0_left is None or p0_right is None or not math.isclose(p0_left, p0_right, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("evaluation log decomposition requires a common positive base price")
        actual_log = math.log(row["actual"] / p0_left)
        error_left, delta = actual_log - a, b - a
        squared_delta = delta * delta
        cross = -2.0 * error_left * delta
        difference = (actual_log - b) ** 2 - (actual_log - a) ** 2
        sums["n"] += 1
        sums["deltaSquaredSum"] += squared_delta
        sums["crossTermSum"] += cross
        sums["squaredErrorDifferenceSum"] += difference
    identity = sums["deltaSquaredSum"] + sums["crossTermSum"]
    sums["identityResidual"] = identity - sums["squaredErrorDifferenceSum"]
    sums["identityVerified"] = sums["n"] > 0 and math.isclose(identity, sums["squaredErrorDifferenceSum"], rel_tol=1e-9, abs_tol=1e-10)
    if sums["n"] and not sums["identityVerified"]:
        raise ValueError("evaluation log-loss decomposition identity failed")
    return {"product": product, "leftModel": left_model, "rightModel": right_model,
            "differenceDefinition": "right minus left squared log error", **sums}


def _training_decomposition_rows(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for fit in fits:
        for contrast, values in (fit.get("representativeTrainLogDecomposition") or {}).items():
            if not isinstance(values, Mapping):
                continue
            d2, cross, difference = (values.get("deltaSquaredSum"), values.get("crossTermSum"),
                                     values.get("squaredErrorDifferenceSum"))
            residual = d2 + cross - difference
            output.append({"product": fit.get("product"), "modelId": fit.get("modelId"),
                           "horizon": fit.get("horizon"), "fitMonth": fit.get("fitMonth"),
                           "contrast": contrast, **dict(values), "identityResidual": residual,
                           "identityVerified": math.isclose(residual, 0.0, rel_tol=1e-9, abs_tol=1e-8)})
    return output


def score_documents(selection: dict[str, Any], forecast_doc: dict[str, Any],
                    replay: dict[str, Any], prices: dict[str, Any], *,
                    selection_file_sha: str | None = None,
                    forecast_file_sha: str | None = None,
                    source_provenance: Mapping[str, Any] | None = None,
                    technical_integrity_ok: bool = True) -> dict[str, Any]:
    """Pure document scorer used by offline fixtures and the artifact CLI."""
    selected = _validate_selection(selection, selection_file_sha)
    if forecast_doc.get("targetVersion") != selection.get("targetVersion"):
        raise ValueError("T forecast target version differs from frozen V selection")
    _document_sha(replay, "replayVerificationSha256")
    forecasts, issuance = _validate_forecasts(forecast_doc, selected)
    if replay.get("forecastsFileSha256") != forecast_file_sha or (
            replay.get("forecastRowsSha256") != _sha(_canonical(forecast_doc.get("forecasts", [])))):
        raise ValueError("replay audit does not bind the supplied forecast artifact")
    if replay.get("selectionSha256") != selection.get("selectionSha256"):
        raise ValueError("replay audit does not bind the frozen selection")
    if replay.get("executionIdentitySha256") != forecast_doc.get("executionIdentitySha256"):
        raise ValueError("replay audit execution identity differs from forecast manifest")
    actuals = _prices(prices)
    target_observations = _target_observations(prices)
    rows = []
    coverage = {p: {"baselineIssued": 0, "policyIssued": 0,
                    "lostBaselineIssuedKeys": 0, "positiveActualBaselineIssued": 0,
                    "positiveActualPolicyIssued": 0, "directSelectedIssued": 0,
                    "fallback": 0, "withheld": 0, "missingActualKeys": 0,
                    "baselineSourceStatuses": {}, "selectedRawModelStatuses": {},
                    "policyStatuses": {}, "fallbackReasons": {}} for p in PRODUCTS}
    for row in forecasts:
        p = row["product"]
        c = coverage[p]
        if row["_b_status"] == "issued":
            c["baselineIssued"] += 1
        if row["_g_status"] == "issued":
            c["policyIssued"] += 1
        if row["_b_status"] == "issued" and row["_g_status"] != "issued":
            c["lostBaselineIssuedKeys"] += 1
        if row["_direct_status"] == "issued":
            c["directSelectedIssued"] += 1
        g = row["G"]
        c["fallback"] += int(g.get("status") == "fallback" and g.get("fallbackTo") == "B")
        c["withheld"] += int(row["_g_status"] == "not_issued")
        for key, val in (("baselineSourceStatuses", row.get("B", {}).get("sourceStatus")),
                         ("selectedRawModelStatuses", g.get("rawModelStatus")),
                         ("policyStatuses", g.get("status"))):
            bucket = c[key]
            name = str(val) if val is not None else "unspecified"
            bucket[name] = bucket.get(name, 0) + 1
        if g.get("fallbackTo") == "B":
            reasons = c["fallbackReasons"]
            name = str(g.get("fallbackReason") or "unspecified")
            reasons[name] = reasons.get(name, 0) + 1
        actual = actuals.get((p, row["target"]))
        c["missingActualKeys"] += int(actual is None)
        if actual is not None:
            c["positiveActualBaselineIssued"] += int(row["_b_status"] == "issued")
            c["positiveActualPolicyIssued"] += int(row["_g_status"] == "issued")
        rows.append({**row, "actual": actual, "g": row["_g"], "b": row["_b"],
                     "direct": row["_direct"]})
    primary = _pairs(rows, actuals, "g", "b")
    direct = _pairs(rows, actuals, "direct", "b")
    primary_metrics = _metrics(primary)
    dm_primary = _dm(primary, PRODUCTS, "G_vs_B")
    macro_complete = primary_metrics["complete"]
    coverage_ok = all(v["lostBaselineIssuedKeys"] == 0 for v in coverage.values())
    status_ok = technical_integrity_ok and replay.get("status") == "pass"
    promoted = bool(status_ok and coverage_ok and macro_complete
                    and primary_metrics["gMape"] < primary_metrics["bMape"]
                    and primary_metrics["gMae"] <= primary_metrics["bMae"])
    # Fixed secondary family: P-L on radish; A-CTRL on onion. These are diagnostics,
    # never a gate and never condition the primary pair set.
    secondary_rows = []
    for r in rows:
        p = r["product"]
        models = r.get("models", {})
        if p == "radish":
            left, right = models.get("P"), models.get("L")
        elif p == "onion":
            left, right = models.get("A"), models.get("CTRL")
        else:
            continue
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        _forecast(left, "secondary left")
        _forecast(right, "secondary right")
        y = actuals.get((p, r["target"]))
        a, b = _positive(left.get("prediction")), _positive(right.get("prediction"))
        if y is not None and a is not None and b is not None:
            secondary_rows.append({"product": p, "horizon": r["horizon"], "origin": r["origin"],
                                   "target": r["target"], "actual": y, "a": a, "b": b})
    secondary = _dm(secondary_rows, ("radish", "onion"), "P_L_radish__A_CTRL_onion")
    primary_pair_keys = {(r["product"], r["origin"], r["target"], r["horizon"]) for r in primary}
    direct_pair_keys = {(r["product"], r["origin"], r["target"], r["horizon"]) for r in direct}
    secondary_pair_keys = {(r["product"], r["origin"], r["target"], r["horizon"])
                           for r in secondary_rows}
    target_evaluation = []
    for row in rows:
        key = (row["product"], row["origin"], row["target"], row["horizon"])
        obs = target_observations.get((row["product"], row["target"]), [])
        states = [o["status"] for o in obs]
        target_status = states[0] if len(states) == 1 else "multiple_observations" if states else "no_observation"
        models = row.get("models", {})
        target_evaluation.append({
            "product": row["product"], "origin": row["origin"], "target": row["target"],
            "horizon": row["horizon"], "targetVersion": forecast_doc.get("targetVersion"),
            "targetObservationStatus": target_status,
            "targetObservationStates": states,
            "actualValue": row["actual"] if row["actual"] is not None else None,
            "actualRawValues": [o["actualRaw"] for o in obs],
            "baselineSourceStatus": row.get("B", {}).get("sourceStatus"),
            "baselinePredictionStatus": row["_b_status"],
            "selectedModelId": row.get("selected_model_id"),
            "selectedRawModelStatus": row.get("G", {}).get("rawModelStatus"),
            "policyStatus": row["G"].get("status"),
            "fallbackTo": row["G"].get("fallbackTo"),
            "fallbackReason": row["G"].get("fallbackReason"),
            "directSelectedModelStatus": ("not_applicable_B_selected" if selected[row["product"]] == "B"
                                           else row["_direct_status"]),
            "secondaryLeftStatus": (models.get("P", {}).get("status") if row["product"] == "radish"
                                    else models.get("A", {}).get("status") if row["product"] == "onion" else None),
            "secondaryRightStatus": (models.get("L", {}).get("status") if row["product"] == "radish"
                                     else models.get("CTRL", {}).get("status") if row["product"] == "onion" else None),
            "primaryPairEligible": key in primary_pair_keys,
            "directSelectedModelUsePairEligible": key in direct_pair_keys,
            "secondaryDiagnosticPairEligible": key in secondary_pair_keys,
        })
    eval_decompositions = [
        _eval_log_decomposition(rows, "radish", "L", "P"),
        _eval_log_decomposition(rows, "onion", "CTRL", "A"),
    ]
    bootstrap = {metric: {} for metric in ("MAE", "MAPE")}
    for block in BLOCKS:
        if primary_metrics["complete"]:
            metric_results = {
                "MAE": diagnostics.circular_block_bootstrap(primary, a="a", b="b", metric="ae",
                         block=block, replicates=REPLICATES, seed=SEED, calendar_start=START, calendar_end=END),
                "MAPE": diagnostics.circular_block_bootstrap(primary, a="a", b="b", metric="ape",
                         block=block, replicates=REPLICATES, seed=SEED, calendar_start=START, calendar_end=END),
            }
        else:
            metric_results = {metric: {"blockDays": block, "calendarDays": (END-START).days+1,
                              "requested": REPLICATES, "valid": 0, "invalid": REPLICATES,
                              "seed": SEED, "interval95": None,
                              "interpretation": "not_computable_incomplete_fixed_cells"}
                              for metric in bootstrap}
        for metric, result in metric_results.items():
            bootstrap[metric][str(block)] = result
    fits = forecast_doc.get("fitDiagnostics", [])
    training_support = []
    by_train_model = defaultdict(list)
    for fit in fits:
        if isinstance(fit, Mapping):
            by_train_model[(fit.get("product"), fit.get("modelId"))].append(fit)
    for (product, model_id), records in sorted(by_train_model.items()):
        training_support.append({"product": product, "modelId": model_id,
                                "foldCount": len(records),
                                "fitTargetCountMin": min(int(x["fitTargetCount"]) for x in records),
                                "fitTargetCountMax": max(int(x["fitTargetCount"]) for x in records),
                                "fitTargetCountTotal": sum(int(x["fitTargetCount"]) for x in records),
                                "fitOriginCount": len({x.get("fitOrigin") for x in records})})
    identical_cells = [{"product": product, "horizon": h,
                        "n": sum(1 for r in primary if r["product"] == product and r["horizon"] == h),
                        "allPredictionsIdentical": bool([r for r in primary if r["product"] == product and r["horizon"] == h])
                                                     and all(r["a"] == r["b"] for r in primary
                                                             if r["product"] == product and r["horizon"] == h)}
                       for product in PRODUCTS for h in HORIZONS
                       if any(r["product"] == product and r["horizon"] == h for r in primary)]
    product_metrics = {p: _metrics(primary, (p,)) for p in PRODUCTS}
    return {"experiment": "CROP_SPECIALIST_ASSEMBLY_V1", "phase": "T",
            "status": "completed" if status_ok else "blocked_integrity",
            "promotion": "retrospective_candidate_preserved" if promoted else "completed_not_promoted" if status_ok else "blocked_integrity",
            "noOperationalPromotion": True,
            "policy": {"selectedModelByProduct": dict(selected), "fallbackPolicy": "numerical selected-model failure falls back to B; pipeline/integrity failure blocks"},
            "inputProvenance": {"selection": {"path": source_provenance.get("selectionPath") if source_provenance else None,
                                                "fileSha256": selection_file_sha,
                                                "selectionSha256": selection.get("selectionSha256")},
                                "forecasts": {"path": source_provenance.get("forecastPath") if source_provenance else None,
                                               "fileSha256": forecast_file_sha,
                                               "manifestSha256": forecast_doc.get("manifestSha256"),
                                               "forecastRowsSha256": forecast_doc.get("forecastRowsSha256")},
                                "replay": {"path": source_provenance.get("replayPath") if source_provenance else None,
                                            "fileSha256": source_provenance.get("replayFileSha256") if source_provenance else None,
                                            "replayVerificationSha256": replay.get("replayVerificationSha256")},
                                "priceSnapshot": {"path": source_provenance.get("pricePath") if source_provenance else None,
                                                  "fileSha256": source_provenance.get("priceFileSha256") if source_provenance else None,
                                                  "targetVersion": forecast_doc.get("targetVersion")},
                                "sources": {name: {"path": ref.get("path"), "fileSha256": ref.get("fileSha256"),
                                                   "manifestSha256": ref.get("manifestSha256")}
                                            for name, ref in forecast_doc.get("sources", {}).items()}},
            "targetEvaluation": target_evaluation,
            "issuance": dict(issuance), "coverage": coverage,
            "primaryGvsB": {"metrics": primary_metrics, "byProduct10HorizonMacro": product_metrics,
                            "directSelectedVsB": _direct_metrics(direct, selected),
                            "coveragePreserved": coverage_ok, "priceGatePassed": promoted},
            "secondaryDiagnostics": {"comparisonDirections": {"radish": "P minus L", "onion": "A minus CTRL"},
                                     "metrics": _metrics(secondary_rows, ("radish", "onion")),
                                     "fixedFamilies": secondary,
                                     "evaluationLogLossDecompositions": eval_decompositions},
            "inference": {"primary": dm_primary, "secondary": secondary,
                          "primaryNotes": ["Garlic B-selected rows with identical G/B predictions remain in the fixed 80-test family; identical-prediction cells are listed explicitly."],
                          "holmFamilyScope": "separate fixed family within each bandwidth"},
            "jointCalendarBlockBootstrap": {"metrics": ["MAE", "MAPE"], "seed": SEED,
                                            "replicates": REPLICATES, "results": bootstrap},
            "primaryIdenticalPredictionCells": identical_cells,
            "priceContributions": _contributions(primary),
            "trainingSupport": training_support,
            "trainingLogLossDecompositions": _training_decomposition_rows(fits),
            "counts": {"uniqueOrigins": len({r["origin"] for r in rows}),
                       "uniqueTargets": len({r["target"] for r in rows}),
                       "forecastPairs": len(rows), "positiveActualPairs": sum(r["actual"] is not None for r in rows)},
            "inferenceLimitations": ["retrospective T was inspected repeatedly; p-values do not erase adaptive selection risk",
                                     "non-significance is inconclusive, not evidence of equivalence"]}


def score_files(selection_path: Path, forecast_path: Path, replay_path: Path,
                prices_path: Path | None = None) -> dict[str, Any]:
    selection_path, forecast_path, replay_path = map(lambda p: p.resolve(),
                                                      (selection_path, forecast_path, replay_path))
    prices_path = (prices_path or price_helper.PRICES_PATH).resolve()
    if prices_path != Path(price_helper.PRICES_PATH).resolve():
        raise ValueError("only the project-pinned price snapshot is accepted")
    if _sha(prices_path.read_bytes()) != price_helper.PRICES_SHA256:
        raise ValueError("pinned target snapshot file hash mismatch")
    selection_raw, forecast_raw = selection_path.read_bytes(), forecast_path.read_bytes()
    selection, forecast_doc, replay = map(_load_json, (selection_path, forecast_path, replay_path))
    experiment_root = forecast_path.parent.parent
    replay_file_sha = _sha(replay_path.read_bytes())
    if forecast_path.name != "forecasts.json" or selection_path.name != "selection.json":
        raise ValueError("unexpected selector/T artifact names")
    # Every source and runtime artifact is independently re-hashed from disk.
    from . import crop_specialist_assembly as producer
    from . import intra_crop_panel as panel_source
    pinned_sources = {"A": (producer.source_a.OUT / "V/forecast-manifest.json", producer.PINNED_A_V_MANIFEST_SHA256),
                      "B": (producer.source_b.OUT / "V/forecast-manifest.json", producer.PINNED_CAND_B_V_MANIFEST_SHA256),
                      "L": (producer.source_b.OUT / "V/forecast-manifest.json", producer.PINNED_CAND_B_V_MANIFEST_SHA256),
                      "P": (producer.source_b.OUT / "V/forecast-manifest.json", producer.PINNED_CAND_B_V_MANIFEST_SHA256)}
    if _sha(prices_path.read_bytes()) != price_helper.PRICES_SHA256:
        raise ValueError("pinned target snapshot file hash mismatch")
    if _sha(prices_path.read_bytes()) != selection.get("targetVersion"):
        raise ValueError("selection targetVersion differs from pinned target file bytes")
    for name, ref in selection.get("sources", {}).items():
        expected_path, expected_digest = pinned_sources[name]
        if _rooted_path(ref.get("path")) != expected_path.resolve() or ref.get("sha256") != expected_digest:
            raise ValueError(f"{name} V source differs from repository-pinned source identity")
    raw_rhs_ref = selection.get("pinnedRhsBaseline")
    if (_rooted_path(raw_rhs_ref.get("path")) != Path(producer.BASELINE_PATHS["V"]).resolve()
            or raw_rhs_ref.get("sha256") != producer.BASELINE_SHA256["V"]):
        raise ValueError("selection does not name the repository-pinned V RHS baseline")
    if selection.get("policyRule") != producer.POLICY_RULE:
        raise ValueError("selection policy rule differs from frozen implementation")
    _validate_selection(selection, _sha(selection_raw), verify_source_files=True)
    for name, ref in forecast_doc.get("sources", {}).items():
        if name not in pinned_sources:
            raise ValueError("unexpected T source reference")
        expected_path, expected_digest = pinned_sources[name]
        if _rooted_path(ref.get("path")) != expected_path.resolve() or ref.get("fileSha256") != expected_digest:
            raise ValueError(f"T source differs from repository-pinned source identity: {name}")
        _verify_ref(ref)
    for ref_name in ("targetSnapshot", "resolvedConfig", "sourceCache", "preflight", "pinnedRhsBaseline"):
        ref = forecast_doc.get(ref_name)
        if isinstance(ref, Mapping):
            _verify_ref(ref)
    raw_t_baseline = forecast_doc.get("pinnedRhsBaseline")
    from .chronos2_market_context import BASELINE_PATHS, BASELINE_SHA256
    if (not isinstance(raw_t_baseline, Mapping)
            or _rooted_path(raw_t_baseline.get("path")) != Path(BASELINE_PATHS["T"]).resolve()
            or raw_t_baseline.get("pinnedSha256") != BASELINE_SHA256["T"]
            or raw_t_baseline.get("fileSha256") != BASELINE_SHA256["T"]):
        raise ValueError("T artifact does not bind the repository-pinned RHS baseline")
    identity = forecast_doc.get("executionIdentity")
    execution = dict(identity) if isinstance(identity, Mapping) else {}
    embedded_execution_sha = execution.pop("executionIdentitySha256", None)
    if (not isinstance(identity, Mapping)
            or embedded_execution_sha != forecast_doc.get("executionIdentitySha256")
            or _sha(_canonical(execution)) != embedded_execution_sha
            or execution.get("sourceHashesSha256") != _sha(_canonical(producer._source_sha_map()))
            or execution.get("sourceHashes") != producer._source_sha_map()):
        raise ValueError("T execution identity self-digest mismatch")
    for ref_name, expected_path in (("resolvedConfig", panel_source.OUT / "resolved-config.json"),
                                    ("sourceCache", panel_source.SOURCE_CACHE),
                                    ("preflight", panel_source.OUT / "preflight.json")):
        ref = forecast_doc.get(ref_name)
        if not isinstance(ref, Mapping) or _rooted_path(ref.get("path")) != Path(expected_path).resolve():
            raise ValueError(f"T {ref_name} path differs from pinned CAND_B artifact")
    cache_doc = _load_json(Path(panel_source.SOURCE_CACHE))
    if (cache_doc.get("cacheSha256") != panel_source.SOURCE_CACHE_INTERNAL_SHA256
            or forecast_doc.get("sourceCache", {}).get("cacheSha256") != panel_source.SOURCE_CACHE_INTERNAL_SHA256):
        raise ValueError("T source-cache internal identity mismatch")
    config_doc = _load_json(Path(panel_source.OUT / "resolved-config.json"))
    config_hash = config_doc.get("configSha256")
    config_unsigned = dict(config_doc)
    config_unsigned.pop("configSha256", None)
    if (_sha(_canonical(config_unsigned)) != config_hash
            or forecast_doc.get("resolvedConfig", {}).get("configSha256") != config_hash):
        raise ValueError("T resolved-config internal identity mismatch")
    preflight_doc = _load_json(Path(panel_source.OUT / "preflight.json"))
    preflight_hash = preflight_doc.get("preflightSha256")
    preflight_unsigned = dict(preflight_doc)
    preflight_unsigned.pop("preflightSha256", None)
    if (_sha(_canonical(preflight_unsigned)) != preflight_hash
            or forecast_doc.get("preflight", {}).get("preflightSha256") != preflight_hash):
        raise ValueError("T preflight identity mismatch")
    if _rooted_path(forecast_doc.get("targetSnapshot", {}).get("path")) != Path(price_helper.PRICES_PATH).resolve():
        raise ValueError("T target snapshot path differs from pinned project snapshot")
    sel_ref = forecast_doc.get("selection")
    if not isinstance(sel_ref, Mapping) or _rooted_path(sel_ref.get("path")) != selection_path:
        raise ValueError("T artifact selection reference does not name supplied selection.json")
    if _sha(selection_raw) != sel_ref.get("fileSha256") or selection.get("selectionSha256") != sel_ref.get("selectionSha256"):
        raise ValueError("T artifact selection hash binding mismatch")
    if _sha(forecast_raw) != replay.get("forecastsFileSha256"):
        raise ValueError("replay audit forecast-file hash mismatch")
    if _rooted_path(forecast_doc.get("replayVerificationPath"), base=experiment_root) != replay_path:
        raise ValueError("forecast manifest names a different replay audit")
    replay_checks = {
        "manifestSha256": forecast_doc.get("manifestSha256"),
        "forecastRowsSha256": forecast_doc.get("forecastRowsSha256"),
        "selectionFileSha256": _sha(selection_raw),
        "selectionSha256": selection.get("selectionSha256"),
        "executionIdentitySha256": forecast_doc.get("executionIdentitySha256"),
        "sourceCacheFileSha256": forecast_doc.get("sourceCache", {}).get("fileSha256"),
        "resolvedConfigFileSha256": forecast_doc.get("resolvedConfig", {}).get("fileSha256"),
        "baselineFileSha256": forecast_doc.get("pinnedRhsBaseline", {}).get("fileSha256"),
        "targetSnapshotFileSha256": forecast_doc.get("targetSnapshot", {}).get("fileSha256"),
        "sourceManifestHashes": {name: ref.get("fileSha256") for name, ref in forecast_doc.get("sources", {}).items()},
    }
    if replay.get("status") != "pass" or any(replay.get(key) != val for key, val in replay_checks.items()):
        raise ValueError("replay audit does not reproduce forecast/source/config identity")
    if replay.get("selectionFileSha256") != _sha(selection_raw):
        raise ValueError("replay audit selection file hash mismatch")
    if replay.get("checkpointArtifacts") != [
            {"path": item.get("path"), "fileSha256": item.get("fileSha256"),
             "identityPath": item.get("identityPath"),
             "identityFileSha256": item.get("identityFileSha256"),
             "identitySha256": item.get("identitySha256")}
            for item in forecast_doc.get("modelArtifacts", [])]:
        raise ValueError("replay audit checkpoint list differs from T artifact")
    for artifact in forecast_doc.get("modelArtifacts", []):
        if not isinstance(artifact, Mapping):
            raise ValueError("malformed model artifact evidence")
        _verify_ref(artifact, base=experiment_root)
        identity = {"path": artifact.get("identityPath"),
                    "fileSha256": artifact.get("identityFileSha256")}
        if not identity.get("path"):
            raise ValueError("model identity sidecar missing")
        _verify_ref(identity, base=experiment_root)
        sidecar = _load_json(_rooted_path(identity["path"], base=experiment_root))
        identity_doc = sidecar.get("identity")
        checkpoint_hash_field = ("modelTextSha256" if Path(str(artifact.get("path", ""))).suffix == ".txt"
                                 else "checkpointFileSha256")
        if (_sha(_canonical(identity_doc)) != artifact.get("identitySha256")
                or sidecar.get("identitySha256") != artifact.get("identitySha256")
                or sidecar.get(checkpoint_hash_field) != artifact.get("fileSha256")
                or identity_doc != artifact.get("checkpointIdentity")):
            raise ValueError("checkpoint sidecar does not bind its model, identity, and digest")
    fits = forecast_doc.get("fitDiagnostics")
    if not isinstance(fits, list) or len(fits) != forecast_doc.get("plannedFitCount"):
        raise ValueError("T fit diagnostics do not account for every planned estimator")
    expected_fit_keys = set()
    models_by_product = forecast_doc.get("plannedModelIdsByProduct")
    if not isinstance(models_by_product, Mapping):
        raise ValueError("T planned model map is missing")
    fold_keys = {(r.get("product"), r.get("horizon"), r.get("fit_month"))
                 for r in forecast_doc.get("forecasts", [])}
    for product, horizon, month in fold_keys:
        expected_fit_keys.update((product, model_id, horizon, month)
                                 for model_id in models_by_product.get(product, []))
    observed_fit_keys = {(fit.get("product"), fit.get("modelId"), fit.get("horizon"), fit.get("fitMonth"))
                         for fit in fits if isinstance(fit, Mapping)}
    if observed_fit_keys != expected_fit_keys:
        raise ValueError("T fit diagnostic fold map differs from planned models and forecast months")
    newly_fit = sum(1 for fit in fits if isinstance(fit, Mapping) and fit.get("fitStatus") == "fit")
    valid_fits = sum(1 for fit in fits if isinstance(fit, Mapping) and fit.get("fitStatus") in ("fit", "reused"))
    if (forecast_doc.get("actualEstimatorFitCount") != valid_fits
            or forecast_doc.get("newEstimatorFitCount") != newly_fit):
        raise ValueError("fit/reuse counters disagree with per-fold records")
    artifacts = forecast_doc.get("modelArtifacts", [])
    artifact_index = {(a.get("product"), a.get("horizon"), a.get("fitMonth"), a.get("modelId")): a
                      for a in artifacts if isinstance(a, Mapping) and a.get("modelId")}
    pair_artifact_index = {(a.get("product"), a.get("horizon"), a.get("fitMonth"), mid): a
                           for a in artifacts if isinstance(a, Mapping)
                           for mid in a.get("modelIds", [])}
    for fit in fits:
        if not isinstance(fit, Mapping) or fit.get("fitStatus") not in ("fit", "reused"):
            raise ValueError("T fit diagnostics contain a failed or malformed fold")
        model_id = fit.get("modelId")
        product, horizon, month = fit.get("product"), fit.get("horizon"), fit.get("fitMonth")
        digest, identity_sha = fit.get("checkpointFileSha256"), fit.get("checkpointIdentitySha256")
        if (not isinstance(fit.get("fitTargetSha256"), str) or len(fit["fitTargetSha256"]) != 64
                or not isinstance(digest, str) or len(digest) != 64
                or not isinstance(identity_sha, str) or len(identity_sha) != 64
                or fit.get("fitTargetCount", 0) < 1
                or not isinstance(fit.get("representativeTrainDiagnostics"), Mapping)):
            raise ValueError("T fit evidence is incomplete")
        key = (product, horizon, month, model_id)
        artifact = artifact_index.get(key) or pair_artifact_index.get(key)
        if artifact is None:
            raise ValueError("fit diagnostics have no corresponding checkpoint artifact")
        if artifact.get("fileSha256") != digest or artifact.get("identitySha256") != identity_sha:
            raise ValueError("fit diagnostic checkpoint hash differs from artifact evidence")
        decomp = fit.get("representativeTrainLogDecomposition") or {}
        if model_id == "P" and product == "radish" and "P-L" not in decomp:
            raise ValueError("radish P fit lacks representative-train P-L decomposition")
        if model_id == "A" and product == "onion" and "A-CTRL" not in decomp:
            raise ValueError("onion A fit lacks representative-train A-CTRL decomposition")
        diagnostic = fit["representativeTrainDiagnostics"]
        for field in ("n", "logMse", "logMae", "priceMAE", "priceMAPE"):
            value = diagnostic.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError("representative training diagnostic contains invalid values")
        fit_month = fit.get("fitMonth")
        fit_origin = _date(fit.get("fitOrigin"))
        if fit_origin is None or fit_origin != dt.date.fromisoformat(fit_month + "-01"):
            raise ValueError("fit origin does not match frozen monthly fit month")
        if _date(fit.get("fitTargetMax")) >= fit_origin:
            raise ValueError("training target reaches or passes the fold fit origin")
        if fit.get("trainingTargetsBeforeFitOrigin") is not True or fit.get("allFeatureDatesAtOrBeforeCutoff") is not True:
            raise ValueError("training rows or features do not prove their causal cutoff")
        for label in (("P-L",) if model_id == "P" and product == "radish"
                      else ("A-CTRL",) if model_id == "A" and product == "onion" else ()):
            values = decomp[label]
            if not isinstance(values, Mapping):
                raise ValueError("training log decomposition is malformed")
            d2, cross, difference, n, error = (values.get("deltaSquaredSum"), values.get("crossTermSum"),
                                                values.get("squaredErrorDifferenceSum"), values.get("n"),
                                                values.get("identityMaxAbsError"))
            if (any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
                    for x in (d2, cross, difference, error))
                    or not isinstance(n, int) or isinstance(n, bool) or n < 1 or error < 0 or error > 1e-8
                    or not math.isclose(d2 + cross, difference, rel_tol=1e-9, abs_tol=1e-8)):
                raise ValueError("training log-loss decomposition identity failed")
    return score_documents(selection, forecast_doc, replay, _load_json(prices_path),
                           selection_file_sha=_sha(selection_raw), forecast_file_sha=_sha(forecast_raw),
                           source_provenance={"selectionPath": str(selection_path),
                                              "forecastPath": str(forecast_path),
                                              "replayPath": str(replay_path),
                                              "replayFileSha256": replay_file_sha,
                                              "pricePath": str(prices_path),
                                              "priceFileSha256": _sha(prices_path.read_bytes())},
    technical_integrity_ok=forecast_doc.get("actualEstimatorFitCount") == forecast_doc.get("plannedFitCount")
                           and forecast_doc.get("forecastKeyCount") == len(forecast_doc.get("forecasts", []))
                           and bool(replay_file_sha))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=price_helper.PRICES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = score_files(args.selection, args.forecasts, args.replay, args.prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(result))
    print(json.dumps({"status": result["status"], "promotion": result["promotion"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
