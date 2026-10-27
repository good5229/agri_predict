"""Stage A: select a cabbage HU/HM blend from the already-saved Chronos V run.

This module performs no model inference, fitting, API calls, or T-period access.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "outputs/experiments/chronos2-market-context-v1-20261012"
OUT = ROOT / "outputs/experiments/chronos2-cabbage-followup-v1-20261012"
FOLLOWUP_ID = "CHRONOS2_CABBAGE_FOLLOWUP_V1"
EXPECTED_C2_SOURCE = "d3a476a0d2825c6e54d756a40a7edeb771ed2aef3a755d6fe14dab622c87ebbc"
EXPECTED_CONFIG_SHA = "6cbba5ab130c729c214d74e0d6c87c3332986879dd85fb298ab9fc7cfed25017"
EXPECTED_FORECAST_SHA = "af0e0ecf3949889d1b9bc0d5c394d97e82e24f5f21f3063fc761ab939623e2ce"
EXPECTED_MANIFEST_SHA = "9d8ebeb0752bfbc438504b98c2336f9b94492908e345199a2a09b37a96a51bf5"
EXPECTED_REPORT_SHA = "d3c00603b8cb54ff22463d1f6e4ff30a15974fe4d8d504748d2a9068cff2e24f"
EXPECTED_CSV_SHA = "f503d186578c47e7414d422bddda61d0600a752e865903ccea26b2463e265786"
PLAN_PATH = ROOT / "docs/CHRONOS2_CABBAGE_FOLLOWUP_V1_2026-10-12.md"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    baseline_rel = config["baseline"]["V"]["path"]
    return {
        "protocol": SOURCE_RUN / "config.yaml",
        "manifest": SOURCE_RUN / "validation-manifest.json",
        "forecasts": SOURCE_RUN / "validation-forecasts.json",
        "csv": SOURCE_RUN / "validation-forecasts.csv",
        "report": SOURCE_RUN / "validation-report.json",
        "baseline": ROOT / baseline_rel,
        "prices": ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json",
        "plan": PLAN_PATH,
    }


def _verify_source_metadata(config: Mapping[str, Any], manifest: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    config_body = {k: v for k, v in config.items() if k != "protocolSha256"}
    if config.get("protocolSha256") != _sha(_canonical(config_body)):
        raise ValueError("source Chronos protocol self-hash mismatch")
    if config.get("sourceSha256", {}).get("agri_data/chronos2_market_context.py") != EXPECTED_C2_SOURCE:
        raise ValueError("source adapter pin differs from the frozen V run")
    if config.get("input", {}).get("priceUnit") != "KRW_per_kg":
        raise ValueError("source V price unit differs from frozen KRW/kg unit")
    if config.get("phases", {}).get("V") != {"targetStart": "2026-04-01", "targetEnd": "2026-06-20",
                                                "originStart": "2026-04-01"}:
        raise ValueError("source V target/origin window differs from frozen plan")
    if manifest.get("manifestSha256") != _sha(_canonical({k: v for k, v in manifest.items()
                                                            if k != "manifestSha256"})):
        raise ValueError("source V manifest self-hash mismatch")
    if manifest.get("phase") != "V" or manifest.get("protocolSha256") != config.get("protocolSha256"):
        raise ValueError("source V manifest is not bound to frozen config")
    if manifest.get("forecastFileSha256") != _file_sha(paths["forecasts"]):
        raise ValueError("source V forecast file hash differs from manifest")
    if manifest.get("csvFileSha256") != _file_sha(paths["csv"]):
        raise ValueError("source V CSV hash differs from manifest")
    if manifest.get("baselineManifestSha256") != config["baseline"]["V"]["sha256"]:
        raise ValueError("source V baseline hash differs from frozen config")
    if manifest.get("inputProvenance", {}).get("priceSnapshotSha256") != _file_sha(paths["prices"]):
        raise ValueError("source V price snapshot hash mismatch")


def freeze_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    """Freeze source and rule hashes without opening answer-bearing forecast/report files."""
    config = _read(SOURCE_RUN / "config.yaml")
    manifest = _read(SOURCE_RUN / "validation-manifest.json")
    paths = _source_paths(config)
    _verify_source_metadata(config, manifest, paths)
    source_hashes = {name: _file_sha(path) for name, path in paths.items()}
    if source_hashes["protocol"] != EXPECTED_CONFIG_SHA:
        raise ValueError("original V config hash differs from completed source run")
    for name, expected in (("manifest", EXPECTED_MANIFEST_SHA), ("forecasts", EXPECTED_FORECAST_SHA),
                           ("report", EXPECTED_REPORT_SHA), ("csv", EXPECTED_CSV_SHA)):
        if source_hashes[name] != expected:
            raise ValueError(f"pinned source {name} hash mismatch")
    for path, expected in config["sourceSha256"].items():
        if _file_sha(ROOT / path) != expected:
            raise ValueError(f"original run source hash mismatch: {path}")
    protocol = {
        "schemaVersion": 1, "experiment": FOLLOWUP_ID,
        "stage": "A_saved_V_selection_only",
        "selectionCodeSha256": _file_sha(Path(__file__)),
        "planSha256": source_hashes["plan"],
        "sourceRun": {"path": str(SOURCE_RUN.relative_to(ROOT)), "phase": "V",
                      "hashes": source_hashes, "manifestSha256": manifest["manifestSha256"],
                      "protocolSha256": config["protocolSha256"],
                      "inputSha256": manifest["inputSha256"],
                      "priceUnit": config["input"]["priceUnit"],
                      "targetWindow": config["phases"]["V"],
                      "baselineManifestSha256": config["baseline"]["V"]["sha256"],
                      "model": config["model"], "runtime": manifest["runtime"]},
        "selection": {"candidates": ["HU", "HM"], "definition": {"HU": "0.5*B+0.5*U",
                         "HM": "0.5*B+0.5*M"}, "metric": "10 horizon cells equally weighted",
                       "eligibleKeysRetained": "all original B-eligible cabbage V keys",
                       "requireNoCoverageLoss": True, "requireMapeStrictlyLowerThanB": True,
                       "requireMaeNoHigherThanB": True, "exactMapeTieWinner": "HU",
                       "expectedPositivePairs": 651, "reportReproductionAbsTolerance": 1e-6,
                       "blendIdentityAbsTolerance": 1e-9},
        "boundary": {"mayRead": ["saved V forecasts", "saved V manifest/report/config/CSV",
                                    "pinned V baseline", "shared pinned price JSON to filter cabbage V rows"],
                     "mustNotRead": ["T/test outputs", "out-of-window or non-cabbage numeric observations"],
                     "actualExtraction": "shared JSON is parsed/hashed; only cabbage observations dated 2026-04-01 through 2026-06-20 reach numeric conversion; no T/out-of-window value enters Stage A calculations",
                     "newInference": 0, "newFit": 0},
    }
    protocol["protocolSha256"] = _sha(_canonical(protocol))
    path = output_dir / "protocol.json"
    _write_once(path, _canonical(protocol))
    return protocol


def _verify_protocol(protocol: Mapping[str, Any], output_dir: Path) -> None:
    stored = dict(protocol)
    claimed = stored.pop("protocolSha256", None)
    if claimed != _sha(_canonical(stored)):
        raise ValueError("follow-up protocol self-hash mismatch")
    if protocol.get("selectionCodeSha256") != _file_sha(Path(__file__)):
        raise ValueError("selection code changed after protocol freeze")
    if protocol.get("planSha256") != _file_sha(PLAN_PATH):
        raise ValueError("follow-up plan changed after protocol freeze")
    if protocol.get("experiment") != FOLLOWUP_ID:
        raise ValueError("unexpected follow-up protocol")
    if (protocol.get("stage") != "A_saved_V_selection_only"
            or protocol.get("selection", {}).get("expectedPositivePairs") != 651
            or protocol.get("selection", {}).get("exactMapeTieWinner") != "HU"
            or protocol.get("boundary", {}).get("newInference") != 0
            or protocol.get("boundary", {}).get("newFit") != 0):
        raise ValueError("follow-up protocol does not match frozen Stage A rules")


def _parse_day(value: Any) -> dt.date:
    if not isinstance(value, str):
        raise ValueError("forecast dates must be ISO strings")
    return dt.date.fromisoformat(value)


def _v_key(key: tuple[str, dt.date, dt.date, int]) -> bool:
    return (key[1] >= dt.date(2026, 4, 1)
            and dt.date(2026, 4, 1) <= key[2] <= dt.date(2026, 6, 20))


def _key(row: Mapping[str, Any]) -> tuple[str, dt.date, dt.date, int]:
    key = row.get("key")
    if not isinstance(key, Mapping):
        key = row
    product = key.get("product")
    origin, target, horizon = _parse_day(key.get("originDate")), _parse_day(key.get("targetDate")), key.get("horizon")
    if product not in ("cabbage", "radish", "onion", "garlic") or type(horizon) is not int or not 1 <= horizon <= 10:
        raise ValueError("forecast key outside frozen domain")
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError("forecast horizon does not match dates")
    return product, origin, target, horizon


def _positive(item: Any) -> float | None:
    if not isinstance(item, Mapping) or item.get("status") != "issued":
        return None
    try:
        value = float(item["prediction"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _stage_a_actuals(raw_prices: Mapping[str, Any]) -> dict[tuple[str, dt.date], tuple[float, float]]:
    """Filter before numeric conversion so only cabbage V rows reach the converter."""
    products = raw_prices.get("products")
    if raw_prices.get("schemaVersion") != 1 or not isinstance(products, list):
        raise ValueError("pinned prices must use dashboard schemaVersion 1")
    filtered, seen = [], set()
    for product in products:
        if not isinstance(product, Mapping):
            raise ValueError("invalid price product record")
        pid = product.get("id")
        if pid not in ("cabbage", "radish", "onion", "garlic", "pepper") or pid in seen:
            raise ValueError("unknown or duplicate product in pinned prices")
        seen.add(pid)
        selected = []
        if pid == "cabbage":
            observations = product.get("observations", [])
            if not isinstance(observations, list):
                raise ValueError("cabbage observations must be a list")
            for row in observations:
                if not isinstance(row, Mapping):
                    raise ValueError("invalid cabbage observation record")
                day = row.get("date")
                if isinstance(day, str) and "2026-04-01" <= day <= "2026-06-20":
                    selected.append(dict(row))
        filtered.append({"id": pid, "observations": selected})
    from . import direct_horizon as dh
    converted = dh._observations({"schemaVersion": 1, "products": filtered})
    return {("cabbage", day): obs for day, obs in converted["cabbage"].items()}


def _candidate_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    per_h = []
    for h in range(1, 11):
        cell = [r for r in rows if r["key"][3] == h and r["actual"] is not None
                and r.get(name) is not None]
        if not cell:
            per_h.append({"horizon": h, "n": 0, "mae": None, "mape": None})
            continue
        mae = math.fsum(abs(r[name] - r["actual"]) for r in cell) / len(cell)
        mape = math.fsum(100 * abs(r[name] - r["actual"]) / r["actual"] for r in cell) / len(cell)
        per_h.append({"horizon": h, "n": len(cell), "mae": mae, "mape": mape})
    supported = [c for c in per_h if c["n"]]
    return {"issuedKeys": sum(r[name] is not None for r in rows),
            "positiveActualPairs": sum(r["actual"] is not None and r.get(name) is not None for r in rows),
            "equalHorizonMae": math.fsum(c["mae"] for c in supported) / 10 if len(supported) == 10 else None,
            "equalHorizonMape": math.fsum(c["mape"] for c in supported) / 10 if len(supported) == 10 else None,
            "byHorizon": per_h}


def calculate_selection(forecasts: list[Mapping[str, Any]], baseline_rows: list[Mapping[str, Any]],
                        actuals: Mapping[tuple[str, dt.date], float], report: Mapping[str, Any],
                        *, expected_pairs: int = 651, report_tolerance: float = 1e-6,
                        blend_tolerance: float = 1e-9) -> dict[str, Any]:
    """Pure V-only reconciliation, arithmetic blends, cell metrics, and frozen selection rule."""
    base: dict[tuple[str, dt.date, dt.date, int], dict[str, Any]] = {}
    for source in baseline_rows:
        k = _key(source)
        if not _v_key(k):
            continue
        item = source.get("conditions", {}).get("C")
        if k[0] != "cabbage" or item is None:
            continue
        if item.get("status") not in ("issued", "fallback"):
            continue
        prediction = item.get("blendPrediction")
        try:
            prediction = float(prediction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("eligible RHS C baseline lacks numeric price") from exc
        if not math.isfinite(prediction) or prediction <= 0:
            raise ValueError("eligible RHS C baseline price is invalid")
        if k in base:
            raise ValueError("duplicate cabbage RHS baseline key")
        base[k] = {"prediction": prediction, "status": item["status"], "fallbackReason": item.get("fallbackReason")}
    saved: dict[tuple[str, dt.date, dt.date, int], Mapping[str, Any]] = {}
    for row in forecasts:
        k = _key(row)
        if not _v_key(k):
            continue
        if k[0] != "cabbage":
            continue
        if k in saved:
            raise ValueError("duplicate saved V forecast key")
        saved[k] = row
    if not base or set(saved) != set(base):
        raise ValueError("saved cabbage forecast keys do not exactly match RHS C baseline keys")
    paired = []
    hm_repro_max = 0.0
    coverage = {n: {"issued": 0, "evaluation": 0} for n in ("B", "HU", "HM")}
    for k in sorted(base):
        row = saved[k]
        if row.get("context", {}).get("cutoff") != (k[1] - dt.timedelta(days=1)).isoformat():
            raise ValueError("saved forecast cutoff does not equal origin minus one day")
        cond = row.get("conditions", {})
        b = _positive(cond.get("B"))
        if b is None or not math.isclose(b, base[k]["prediction"], rel_tol=0, abs_tol=blend_tolerance):
            raise ValueError("saved B does not reproduce RHS C blendPrediction")
        if cond["B"].get("baselineSourceStatus", base[k]["status"]) != base[k]["status"]:
            raise ValueError("saved B issued/fallback source status differs from RHS")
        expected_reason = base[k]["fallbackReason"] or (
            "saved_baseline_status_fallback" if base[k]["status"] == "fallback" else None)
        if cond["B"].get("reason") != expected_reason:
            raise ValueError("saved B fallback reason differs from RHS C source")
        u, m, h_saved = (_positive(cond.get(c)) for c in ("U", "M", "H"))
        hu = 0.5 * b + 0.5 * u if u is not None else None
        hm = 0.5 * b + 0.5 * m if m is not None else None
        if hm is not None and h_saved is not None:
            hm_repro_max = max(hm_repro_max, abs(hm - h_saved))
        elif hm is not None or h_saved is not None:
            raise ValueError("saved H availability differs from recomputed HM")
        if hm_repro_max > blend_tolerance:
            raise ValueError("saved original H does not reproduce arithmetic HM")
        actual = actuals.get(("cabbage", k[2]))
        if actual is not None and (not math.isfinite(float(actual)) or float(actual) <= 0):
            actual = None
        for name, value in (("B", b), ("HU", hu), ("HM", hm)):
            if value is not None:
                coverage[name]["issued"] += 1
            if value is not None and actual is not None:
                coverage[name]["evaluation"] += 1
        paired.append({"key": k, "actual": float(actual) if actual is not None else None,
                       "B": b, "HU": hu, "HM": hm})
    baseline_positive = sum(r["actual"] is not None for r in paired)
    if baseline_positive != expected_pairs:
        raise ValueError(f"positive B/actual pairs {baseline_positive} != frozen count {expected_pairs}")
    metrics = {name: _candidate_metrics(paired, name) for name in ("B", "HU", "HM")}
    reference = _reported_cabbage_h_minus_b(report)
    for metric in ("equalHorizonMae", "equalHorizonMape"):
        if abs((metrics["HM"][metric] - metrics["B"][metric]) - reference[metric]) > report_tolerance:
            raise ValueError(f"recomputed HM-B {metric} does not reproduce original report")
    eligible = []
    for name in ("HU", "HM"):
        no_loss = (coverage[name]["issued"] == coverage["B"]["issued"]
                   and coverage[name]["evaluation"] == coverage["B"]["evaluation"])
        metrics[name]["noCoverageLoss"] = no_loss
        metrics[name]["coverageLoss"] = {"issued": coverage["B"]["issued"]-coverage[name]["issued"],
                                         "evaluation": coverage["B"]["evaluation"]-coverage[name]["evaluation"]}
        if (no_loss and metrics[name]["equalHorizonMape"] is not None
                and metrics[name]["equalHorizonMae"] is not None
                and metrics[name]["equalHorizonMape"] < metrics["B"]["equalHorizonMape"]
                and metrics[name]["equalHorizonMae"] <= metrics["B"]["equalHorizonMae"]):
            eligible.append(name)
    if not eligible:
        chosen, status = None, "NO_ELIGIBLE_CABBAGE_CANDIDATE"
    else:
        # HU is preferred only for an exact unrounded MAPE tie.
        chosen = min(eligible, key=lambda name: (metrics[name]["equalHorizonMape"], 0 if name == "HU" else 1))
        status = "selected"
    return {"status": status, "selected": chosen, "metrics": metrics, "coverage": coverage,
            "eligibleCandidates": eligible, "cabbageBEligibleKeys": len(base),
            "positiveActualPairs": baseline_positive, "originalHMReproductionMaxAbsError": hm_repro_max,
            "originalReportHMMinusB": reference,
            "selectionRule": "MAPE < B; MAE <= B; no issued/evaluation coverage loss; min exact 10-horizon equal-weight MAPE; exact tie favors HU"}


def _reported_cabbage_h_minus_b(report: Mapping[str, Any]) -> dict[str, float]:
    cells = [c for c in report.get("cells", []) if c.get("product") == "cabbage"]
    if len(cells) != 10:
        raise ValueError("original V report lacks ten cabbage horizon cells")
    if any(c.get("rightMae") is None or c.get("leftMae") is None or
           c.get("rightMape") is None or c.get("leftMape") is None for c in cells):
        raise ValueError("original V cabbage report has unsupported cells")
    return {"equalHorizonMae": math.fsum(c["leftMae"]-c["rightMae"] for c in cells)/10,
            "equalHorizonMape": math.fsum(c["leftMape"]-c["rightMape"] for c in cells)/10}


def run_selection(output_dir: Path = OUT) -> dict[str, Any]:
    protocol = _read(output_dir / "protocol.json")
    _verify_protocol(protocol, output_dir)
    paths = _source_paths(_read(SOURCE_RUN / "config.yaml"))
    hashes = protocol["sourceRun"]["hashes"]
    for name, path in paths.items():
        if _file_sha(path) != hashes[name]:
            raise ValueError(f"frozen Stage A source changed: {name}")
    config, manifest = _read(paths["protocol"]), _read(paths["manifest"])
    _verify_source_metadata(config, manifest, paths)
    for rel, digest in config["sourceSha256"].items():
        if _file_sha(ROOT / rel) != digest:
            raise ValueError(f"original run source hash changed: {rel}")
    forecast_doc, report, baseline = _read(paths["forecasts"]), _read(paths["report"]), _read(paths["baseline"])
    if forecast_doc.get("phase") != "V" or forecast_doc.get("protocolSha256") != config.get("protocolSha256"):
        raise ValueError("saved forecasts are not original V artifacts")
    if (report.get("phase") != "V" or report.get("manifestSha256") != manifest.get("manifestSha256")
            or report.get("forecastFileSha256") != manifest.get("forecastFileSha256")):
        raise ValueError("original V report is not bound to saved V manifest/forecasts")
    if baseline.get("phase") != "V" or _file_sha(paths["baseline"]) != config["baseline"]["V"]["sha256"]:
        raise ValueError("RHS V baseline manifest hash mismatch")
    price_path = paths["prices"]
    if _file_sha(price_path) != manifest["inputProvenance"]["priceSnapshotSha256"]:
        raise ValueError("original V actual-price snapshot changed")
    raw_prices = _read(price_path)
    actuals = {key: obs[0] for key, obs in _stage_a_actuals(raw_prices).items()}
    result = calculate_selection(forecast_doc["forecasts"], baseline["forecasts"], actuals, report)
    result.update({"protocolSha256": protocol["protocolSha256"],
                   "selectionCodeSha256": protocol["selectionCodeSha256"],
                   "sourceHashes": hashes,
                   "selectedAtKst": dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                   "interpretation": "Post-selection development result on saved V; not an independent validation."})
    path = output_dir / "selection.json"
    _write_once(path, _canonical(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "select"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = freeze_protocol(args.output_dir) if args.command == "freeze" else run_selection(args.output_dir)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
