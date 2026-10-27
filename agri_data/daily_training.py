"""Daily, immutable rolling-evaluation pipeline for already-collected public prices."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import collector, coverage, dashboard, experiments, modeling, significance

KST = ZoneInfo("Asia/Seoul")
CONTEXT_START = dt.date(2014, 1, 1)
EVALUATION_START = dt.date(2017, 1, 1)
DISK_FLOOR_BYTES = 5 * 1024**3
BASELINE_DASHBOARD = Path("outputs/dashboard-expanded-20260929.json")
EXPECTED_SCOPES = {(market, product) for market in dashboard.MARKETS
                   for product in dashboard.PRODUCTS}


class DailyTrainingError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pipeline_fingerprint(source: dict[str, Any], provenance: list[dict[str, Any]]) -> str:
    stable = {key: value for key, value in source.items() if key != "generatedAt"}
    modules = {"dashboard": dashboard, "modeling": modeling, "significance": significance,
               "experiments": experiments, "daily_training": sys.modules[__name__]}
    code_hashes = {name: _sha256(Path(module.__file__).read_bytes()) for name, module in modules.items()}
    scripts_dir = Path(dashboard.__file__).resolve().parents[1] / "scripts"
    for name in ("train-daily.sh", "pipeline-daily.sh"):
        code_hashes[name] = _sha256((scripts_dir / name).read_bytes())
    protocol = {"contextStart": CONTEXT_START.isoformat(), "evaluationStart": EVALUATION_START.isoformat(),
                "collectorCutoff": source.get("meta", {}).get("contextEnd"), "models": modeling.MODEL_IDS,
                "cadences": modeling.CADENCES, "horizons": dashboard.HORIZONS,
                "modelParameters": {model_id: modeling._model_parameters(model_id) for model_id in modeling.MODEL_IDS},
                "significanceFamilySize": significance.FAMILY_SIZE, "codeHashes": code_hashes,
                "pythonVersion": sys.version.split()[0],
                "libraryVersions": {name: _package_version(name) for name in
                                    ("scikit-learn", "numpy", "scipy", "joblib")}}
    payload = json.dumps({"snapshot": stable, "selectedProvenance": provenance, "protocol": protocol},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(payload.encode("utf-8"))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise DailyTrainingError(f"invalid JSON object: {path.name}")
    return value, raw


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _write_new_text(path: Path, text: str) -> None:
    """Create an immutable run artifact atomically without replacing prior data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace run artifact {path.name}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise DailyTrainingError("daily training status cannot be read safely") from exc
    if not isinstance(value, dict):
        raise DailyTrainingError("daily training status has an invalid shape")
    return value


def _coverage_query(db: sqlite3.Connection, start: dt.date, end: dt.date):
    sql = coverage.SELECTION_CTE + """
    SELECT dataset,source_url,query_json,selected_state,latest_state,selected_job_id,selected_epoch_id,
           selected_response_at,selected_row_count,selected_previous
    FROM picked
    WHERE dataset='prices_origin'
      AND json_extract(query_json,'$.\"cond[trd_clcln_ymd::EQ]\"') BETWEEN ? AND ?
    """
    return db.execute(sql, (start.isoformat(), end.isoformat()))


def _coverage_days(db: sqlite3.Connection, start: dt.date, cap: dt.date) -> tuple[dict[dt.date, set[tuple[str, str]]], list[dict[str, Any]]]:
    days: dict[dt.date, set[tuple[str, str]]] = {}
    provenance: list[dict[str, Any]] = []
    for row in _coverage_query(db, start, cap):
        try:
            query = json.loads(row["query_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise DailyTrainingError("malformed price query in coverage manifest") from exc
        dims = coverage.price_dimensions(query, row["dataset"])
        if dims is None:
            continue
        _, market, product, source, scope = dims
        if source != "origin" or scope != "all_subclasses" or (market, product) not in EXPECTED_SCOPES:
            continue
        day_text = coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
        if day_text is None:
            raise DailyTrainingError("invalid date in price query manifest")
        day = dt.date.fromisoformat(day_text)
        if row["selected_state"] not in ("complete", "empty"):
            continue
        key = (market, product)
        seen = days.setdefault(day, set())
        if key in seen:
            raise DailyTrainingError(f"duplicate selected broad query scope on {day}")
        seen.add(key)
        provenance.append({"date": day_text, "market": market, "product": product,
                           "jobId": row["selected_job_id"], "epochId": row["selected_epoch_id"],
                           "responseAt": row["selected_response_at"],
                           "rowCount": row["selected_row_count"],
                           "selectedPrevious": bool(row["selected_previous"])})
    return days, provenance


def contiguous_cutoff(days: dict[dt.date, set[tuple[str, str]]], start: dt.date,
                      cap: dt.date) -> tuple[dt.date | None, str | None]:
    """Return final day of the complete 10-query prefix, never skipping a gap."""
    expected = EXPECTED_SCOPES
    cursor = start
    last: dt.date | None = None
    while cursor <= cap:
        if days.get(cursor, set()) != expected:
            return last, cursor.isoformat()
        last = cursor
        cursor += dt.timedelta(days=1)
    return last, None


def _baseline_cutoff(path: Path) -> tuple[dt.date, str]:
    source, raw = _load_json(path)
    products, _, evaluation_start, evaluation_end = significance._validate_input(source)
    if evaluation_start != EVALUATION_START:
        raise DailyTrainingError("approved baseline dashboard has an unexpected evaluation start")
    meta = source.get("meta", {})
    if meta.get("contextStart") != CONTEXT_START.isoformat():
        raise DailyTrainingError("approved baseline dashboard has an unexpected context start")
    if set(products) != set(dashboard.PRODUCTS):
        raise DailyTrainingError("approved baseline dashboard is missing product scopes")
    return evaluation_end, _sha256(raw)


def _collection_status(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "status_unreadable", "error": type(exc).__name__}
    if not isinstance(result, dict):
        return {"status": "status_invalid"}
    return {key: result.get(key) for key in ("date", "status", "stopKind", "yearWorked", "httpAttemptsToday")}


def _incremental_source(source: dict[str, Any], start: dt.date, end: dt.date,
                        parent_sha256: str | None = None) -> dict[str, Any]:
    parent_meta = source.get("meta", {})
    lo, hi = start.isoformat(), end.isoformat()
    result_products = []
    for product in source.get("products", []):
        row = {"id": product["id"],
               "observations": product.get("observations", []),
               "predictions": [prediction for prediction in product.get("predictions", [])
                               if lo <= prediction.get("target", "") <= hi],
               "runs": []}
        for run in product.get("runs", []):
            row["runs"].append({"modelId": run["modelId"], "cadence": run["cadence"],
                                "predictions": [prediction for prediction in run.get("predictions", [])
                                                if lo <= prediction.get("target", "") <= hi]})
        result_products.append(row)
    return {"schemaVersion": 2,
            "meta": {"evaluationStart": lo, "evaluationEnd": hi,
                     "contextStart": parent_meta.get("contextStart", CONTEXT_START.isoformat()),
                     "contextEnd": parent_meta.get("contextEnd"),
                     "statisticsScope": "incremental_targets_only",
                     "parentEvaluationStart": parent_meta.get("evaluationStart"),
                     "parentEvaluationEnd": parent_meta.get("evaluationEnd"),
                     "parentPredictionSha256": parent_sha256},
            "products": result_products}


def _significance_summary(path: Path) -> dict[str, int]:
    report, _ = _load_json(path)
    counts: dict[str, int] = {}
    for row in report.get("conclusions", []):
        key = str(row.get("status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _daily_h1_metrics(path: Path) -> list[dict[str, Any]]:
    report, _ = _load_json(path)
    rows = {(row.get("productId"), row.get("modelId"), row.get("loss")): row
            for row in report.get("comparisons", [])
            if row.get("cadence") == "daily" and row.get("horizon") == 1}
    conclusions = {(row.get("productId"), row.get("modelId")): row.get("status", "unknown")
                   for row in report.get("conclusions", [])
                   if row.get("cadence") == "daily" and row.get("horizon") == 1}
    output = []
    for product in dashboard.PRODUCTS:
        for model in modeling.MODEL_IDS:
            ae = rows.get((product, model, "absolute_error"), {}).get("primary", {})
            ape = rows.get((product, model, "absolute_percentage_error"), {}).get("primary", {})
            ae_test, ape_test = ae.get("test", {}), ape.get("test", {})
            output.append({"productId": product, "modelId": model, "n": ae.get("n", 0),
                           "maeBaseline": ae.get("baselineMeanLoss"), "maeModel": ae.get("candidateMeanLoss"),
                           "mapeBaseline": ape.get("baselineMeanLoss"), "mapeModel": ape.get("candidateMeanLoss"),
                           "holmMae": ae_test.get("holmAdjustedPValue"), "holmMape": ape_test.get("holmAdjustedPValue"),
                           "statusMae": ae_test.get("status", "unavailable"),
                           "statusMape": ape_test.get("status", "unavailable"),
                           "conclusion": conclusions.get((product, model), "unknown")})
    return output


def _markdown(run_id: str, cutoff: dt.date, snapshot_hash: str, collection: dict[str, Any] | None,
              full_counts: dict[str, int] | None = None, incremental_counts: dict[str, int] | None = None,
              new_observations: dict[str, int] | None = None, *, status: str, detail: str = "",
              full_metrics: list[dict[str, Any]] | None = None,
              incremental_metrics: list[dict[str, Any]] | None = None,
              previous_run_directory: str | None = None) -> str:
    lines = [f"# 일일 가격모델 평가 — {cutoff.isoformat()}", "",
             f"실행 ID: `{run_id}`", f"상태: **{status}**", f"자료 cutoff: {cutoff.isoformat()}",
             f"가격 스냅샷 지문: `{snapshot_hash}`", "",
             f"초기 학습 자료는 {CONTEXT_START.isoformat()}~2016-12-31이며 이후는 시점별 rolling 재학습입니다. "
             "고정 설정 Ridge·HistGradientBoosting 및 일/주/월/계절 cadence를 사용한 회고적 rolling 평가입니다."
             " 매일 결과를 확인하는 반복 모니터링이므로 유의성은 탐색적이며, 기간별 800개 비교의 Holm 보정은 해당 보고서 내부 가족만 통제합니다.", ""]
    if collection:
        lines += [f"수집 상태: `{collection.get('status')}` (종료 사유: `{collection.get('stopKind') or '없음'}`; "
                  f"HTTP 시도: {collection.get('httpAttemptsToday')}; 수집기 exit: {collection.get('pipelineCollectionExitCode')}; "
                  f"첫 미완료일: {collection.get('firstMissingDay') or '없음'})", ""]
    if detail:
        lines += [detail, ""]
    if new_observations is not None:
        lines += ["## 이전 성공 cutoff 이후 새 관측일", "", "| 품목 | 새 관측일 수 |", "|---|---:|"]
        lines += [f"| {name} | {count} |" for name, count in new_observations.items()]
        lines.append("")
    lines += ["원자료는 최신 검증 vintage이며 최초 공개시각은 확인되지 않았습니다. 건고추 등 희소 품목은 표본 수를 그대로 표시합니다.", ""]
    for title, counts in (("누적 평가 유의성 결과", full_counts), ("신규 기간 유의성 결과", incremental_counts)):
        if counts is not None:
            lines += [f"## {title}", "", "| 결과 분류 | 건수 |", "|---|---:|"]
            lines += [f"| {key} | {value} |" for key, value in sorted(counts.items())]
            lines.append("")
    for title, metrics in (("누적 평가 — 일별 1일 예측", full_metrics),
                           ("신규 기간 — 일별 1일 예측", incremental_metrics)):
        if metrics is None:
            continue
        lines += [f"## {title}", "", "| 품목 | 모델 | n | 기준 MAE | 모델 MAE | 기준 MAPE% | 모델 MAPE% | Holm p(MAE) | Holm p(MAPE) | 판정 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in metrics:
            fmt = lambda value: "—" if value is None else f"{value:.4g}"
            mae_p = fmt(row["holmMae"]) if row["statusMae"] == "available" else "—"
            mape_p = fmt(row["holmMape"]) if row["statusMape"] == "available" else "—"
            lines.append(f"| {row['productId']} | {row['modelId']} | {row['n']} | {fmt(row['maeBaseline'])} | {fmt(row['maeModel'])} | {fmt(row['mapeBaseline'])} | {fmt(row['mapeModel'])} | {mae_p} | {mape_p} | {row['conclusion']} |")
        lines.append("")
    if status == "완료":
        refs = "누적 상세: `significance-full.md` 및 `significance-full.json`; 가격/예측: `prices.json`, `predictions.json`; 모델 artifact: `models/`."
        if incremental_counts is not None:
            refs += " 신규 기간 상세: `significance-incremental.md` 및 `significance-incremental.json`."
        lines += ["세부 cadence/horizon 전체 결과는 JSON과 Markdown에 있습니다. " + refs, ""]
    elif previous_run_directory:
        lines += [f"가격·예측·유의성 상세는 직전 성공 실행 `{previous_run_directory}`에 보존되어 있습니다.", ""]
    return "\n".join(lines)


def run_daily_training(*, root: str | Path = ".", db_path: str | Path = "data/agri.sqlite",
                       baseline_dashboard: str | Path = BASELINE_DASHBOARD,
                       today: dt.date | None = None, dry_run: bool = False,
                       run_id: str | None = None,
                       collection_exit_code: int | None = None,
                       exporter: Callable[..., dict[str, Any]] | None = None,
                       model_main: Callable[[list[str]], int] | None = None,
                       significance_main: Callable[[list[str]], int] | None = None) -> dict[str, Any]:
    project = Path(root).resolve()
    db_file = Path(db_path)
    if not db_file.is_absolute():
        db_file = project / db_file
    db_file = db_file.resolve()
    day = today or dt.datetime.now(KST).date()
    collection_cap = day - dt.timedelta(days=1)
    run_id = run_id or (dt.datetime.now(KST).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run_dir = project / "outputs" / "daily" / day.isoformat() / run_id
    state_path = project / "runs" / "daily-training-status.json"
    collection_status_path = project / "runs" / "daily-collection-status.json"
    collection_lock_path = project / "runs" / "daily-collection.lock"
    training_lock_path = project / "runs" / "daily-training.lock"
    outputs_root = project / "outputs" / "daily"
    if not db_file.is_file():
        raise DailyTrainingError("collector database does not exist")
    if dry_run:
        state = _read_status(state_path)
        latest_success = state.get("latestSuccess")
        train_lock = None
        collection_lock = None
    else:
        training_lock_path.parent.mkdir(parents=True, exist_ok=True)
        outputs_root.mkdir(parents=True, exist_ok=True)
        train_lock = training_lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(train_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            train_lock.close()
            raise DailyTrainingError("another daily training pipeline is running") from exc
        state = _read_status(state_path)
        latest_success = state.get("latestSuccess")
        try:
            collection_lock = collection_lock_path.open("a+", encoding="utf-8")
            fcntl.flock(collection_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            with contextlib.suppress(UnboundLocalError, OSError):
                collection_lock.close()
            fcntl.flock(train_lock.fileno(), fcntl.LOCK_UN)
            train_lock.close()
            raise DailyTrainingError("collection is active; training snapshot deferred") from exc
        except OSError:
            fcntl.flock(train_lock.fileno(), fcntl.LOCK_UN)
            train_lock.close()
            raise
    try:
        if latest_success is not None and not isinstance(latest_success, dict):
            raise DailyTrainingError("latestSuccess status record is invalid")
        if shutil.disk_usage(db_file.parent).free < DISK_FLOOR_BYTES and not dry_run:
            raise DailyTrainingError("disk space is below the 5 GiB safety floor")
        try:
            db = coverage.open_readonly(str(db_file))
            try:
                db.execute("BEGIN")
                selected_days, selection_provenance = _coverage_days(db, CONTEXT_START, collection_cap)
                cutoff, first_missing = contiguous_cutoff(selected_days, CONTEXT_START, collection_cap)
                if cutoff is None or cutoff < EVALUATION_START:
                    raise DailyTrainingError(f"no contiguous complete price prefix through evaluation start; first gap={first_missing}")
                snapshot = (exporter or dashboard.build_dashboard)(
                    db, str(db_file), context_start=CONTEXT_START, context_end=cutoff,
                    evaluation_start=EVALUATION_START, evaluation_end=cutoff)
            finally:
                db.rollback()
                db.close()
        finally:
            if collection_lock is not None:
                fcntl.flock(collection_lock.fileno(), fcntl.LOCK_UN)
                collection_lock.close()
        if train_lock is not None:
            # Training owns this lock after the collection snapshot is released.
            pass
        selection_provenance = [row for row in selection_provenance if row["date"] <= cutoff.isoformat()]
        selection_provenance.sort(key=lambda row: (row["date"], row["market"], row["product"]))
        snapshot_hash = _pipeline_fingerprint(snapshot, selection_provenance)
        collection = _collection_status(collection_status_path)
        if collection is None:
            collection = {"status": "status_missing"}
        collection["pipelineCollectionExitCode"] = collection_exit_code
        collection["firstMissingDay"] = first_missing
        if dry_run:
            return {"status": "dry_run", "cutoff": cutoff.isoformat(), "firstMissingDay": first_missing,
                    "snapshotSha256": snapshot_hash, "selectedQueryCount": len(selection_provenance),
                    "collectionStatus": collection}
        run_dir.mkdir(parents=True, exist_ok=False)
        provenance_path = run_dir / "selected-provenance.json"
        _write_new_text(provenance_path, json.dumps(selection_provenance, ensure_ascii=False, indent=2) + "\n")
        previous_cutoff = None
        previous_success_hash = None
        if latest_success:
            previous_cutoff = dt.date.fromisoformat(latest_success["cutoff"])
            previous_success_hash = latest_success.get("snapshotSha256")
        else:
            previous_cutoff, baseline_hash = _baseline_cutoff(project / baseline_dashboard)
        if previous_cutoff > cutoff:
            raise DailyTrainingError("latest successful cutoff is newer than available contiguous data")
        if latest_success and previous_cutoff == cutoff and previous_success_hash == snapshot_hash:
            note = _markdown(run_id, cutoff, snapshot_hash, collection, status="변경 없음",
                             detail="선택된 cutoff·가격·원천 provenance가 직전 성공 실행과 같아 학습과 통계 재계산을 건너뛰었습니다.",
                             previous_run_directory=latest_success.get("runDirectory"))
            _write_new_text(run_dir / "report.md", note)
            attempt = {"runId": run_id, "runDirectory": str(run_dir.relative_to(project)),
                       "status": "no_change", "cutoff": cutoff.isoformat(), "snapshotSha256": snapshot_hash,
                       "selectedProvenance": str(provenance_path.relative_to(project)),
                       "finishedAtKst": dt.datetime.now(KST).isoformat()}
            _atomic_json(state_path, {**state, "latestAttempt": attempt})
            _atomic_json(run_dir / "run-status.json", attempt)
            return attempt

        price_path = run_dir / "prices.json"
        if shutil.disk_usage(db_file.parent).free < DISK_FLOOR_BYTES:
            raise DailyTrainingError("disk space fell below the 5 GiB safety floor before training")
        _write_new_text(price_path, json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        full_model_path = run_dir / "predictions.json"
        artifact_dir = run_dir / "models"
        model_call = model_main or modeling.main
        model_args = ["--input", str(price_path), "--output", str(full_model_path),
                      "--artifacts-dir", str(artifact_dir), "--evaluation-start", EVALUATION_START.isoformat(),
                      "--evaluation-end", cutoff.isoformat()]
        if model_call(model_args) != 0:
            raise DailyTrainingError("fixed rolling model evaluation failed")
        full_sig_path = run_dir / "significance-full.json"
        full_sig_md = run_dir / "significance-full.md"
        sig_call = significance_main or significance.main
        if sig_call(["--input", str(full_model_path), "--output", str(full_sig_path), "--markdown", str(full_sig_md)]) != 0:
            raise DailyTrainingError("cumulative significance analysis failed")

        if previous_cutoff is None:
            raise DailyTrainingError("cannot determine previous evaluation cutoff")
        incremental_start = previous_cutoff + dt.timedelta(days=1)
        incremental_sig_path = None
        incremental_counts = None
        new_observations = {}
        if incremental_start <= cutoff:
            model_output, model_bytes = _load_json(full_model_path)
            incremental = _incremental_source(model_output, incremental_start, cutoff, _sha256(model_bytes))
            increment_path = run_dir / "predictions-incremental.json"
            _write_new_text(increment_path, json.dumps(incremental, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            incremental_sig_path = run_dir / "significance-incremental.json"
            if sig_call(["--input", str(increment_path), "--output", str(incremental_sig_path),
                         "--markdown", str(run_dir / "significance-incremental.md")]) != 0:
                raise DailyTrainingError("incremental significance analysis failed")
            incremental_counts = _significance_summary(incremental_sig_path)
            for product in snapshot["products"]:
                new_observations[product["id"]] = sum(
                    incremental_start.isoformat() <= row["date"] <= cutoff.isoformat()
                    for row in product.get("observations", []))
        else:
            for product in snapshot["products"]:
                new_observations[product["id"]] = 0
        full_counts = _significance_summary(full_sig_path)
        full_metrics = _daily_h1_metrics(full_sig_path)
        incremental_metrics = _daily_h1_metrics(incremental_sig_path) if incremental_sig_path else None
        detail = (f"평가 구간은 {EVALUATION_START.isoformat()}~{cutoff.isoformat()}이며, "
                  f"이전 기준 cutoff {previous_cutoff.isoformat()} 이후 구간은 "
                  f"{incremental_start.isoformat()}~{cutoff.isoformat()}입니다." if incremental_start <= cutoff
                  else f"자료 cutoff가 이전 성공 cutoff({previous_cutoff.isoformat()})와 같아 신규 평가일이 없습니다. vintage 변경에 따라 누적평가는 재계산했습니다.")
        report_md = _markdown(run_id, cutoff, snapshot_hash, collection, full_counts, incremental_counts,
                              new_observations, status="완료", detail=detail,
                              full_metrics=full_metrics, incremental_metrics=incremental_metrics)
        _write_new_text(run_dir / "report.md", report_md)
        attempt = {"runId": run_id, "runDirectory": str(run_dir.relative_to(project)),
                   "status": "complete", "cutoff": cutoff.isoformat(), "snapshotSha256": snapshot_hash,
                   "baselineReference": {"cutoff": previous_cutoff.isoformat(),
                       "source": str(baseline_dashboard) if latest_success is None else latest_success.get("runDirectory"),
                       "sha256": baseline_hash if latest_success is None else previous_success_hash},
                   "incrementalStart": incremental_start.isoformat() if incremental_start <= cutoff else None,
                   "newObservationsByProduct": new_observations,
                   "selectedProvenance": str(provenance_path.relative_to(project)),
                   "predictions": str(full_model_path.relative_to(project)),
                   "fullSignificance": str(full_sig_path.relative_to(project)),
                   "incrementalSignificance": str(incremental_sig_path.relative_to(project)) if incremental_sig_path else None,
                   "finishedAtKst": dt.datetime.now(KST).isoformat(),
                   "collectionStatus": collection}
        _atomic_json(run_dir / "run-status.json", attempt)
        success = {key: attempt[key] for key in ("runId", "runDirectory", "cutoff", "snapshotSha256",
                                                   "baselineReference", "predictions", "fullSignificance",
                                                   "incrementalSignificance", "selectedProvenance", "finishedAtKst")}
        _atomic_json(state_path, {"schemaVersion": 1, "latestSuccess": success, "latestAttempt": attempt})
        return attempt
    finally:
        if train_lock is not None:
            fcntl.flock(train_lock.fileno(), fcntl.LOCK_UN)
            train_lock.close()


def _failure_report(root: Path, run_id: str, day: dt.date, message: str,
                    previous_state: dict[str, Any]) -> dict[str, Any]:
    run_dir = root / "outputs" / "daily" / day.isoformat() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_detail = f"실행 실패: {message}"
    (run_dir / "report.md").write_text(f"# 일일 가격모델 평가 — {day.isoformat()}\n\n**상태: 실패**\n\n{safe_detail}\n",
                                        encoding="utf-8")
    attempt = {"runId": run_id, "runDirectory": str(run_dir.relative_to(root)), "status": "failed",
               "error": message, "finishedAtKst": dt.datetime.now(KST).isoformat()}
    _atomic_json(run_dir / "run-status.json", attempt)
    _atomic_json(root / "runs" / "daily-training-status.json",
                 {**previous_state, "latestAttempt": attempt})
    return attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--baseline-dashboard", default=str(BASELINE_DASHBOARD))
    parser.add_argument("--collection-exit-code", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    project = Path(args.root).resolve()
    run_id = dt.datetime.now(KST).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    try:
        result = run_daily_training(root=args.root, db_path=args.db, baseline_dashboard=args.baseline_dashboard,
                                    dry_run=args.dry_run, run_id=run_id,
                                    collection_exit_code=args.collection_exit_code)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (DailyTrainingError, OSError, ValueError, sqlite3.Error, json.JSONDecodeError, RuntimeError) as exc:
        if args.dry_run:
            print(f"daily training dry-run failed: {exc}", file=sys.stderr)
            return 2
        lock_path = project / "runs" / "daily-training.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_stream = lock_path.open("a+", encoding="utf-8")
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"daily training not recorded because another pipeline holds the lock: {type(exc).__name__}",
                  file=sys.stderr)
            return 2
        except OSError as lock_error:
            print(f"daily training failed without status write: {type(lock_error).__name__}", file=sys.stderr)
            return 2
        state_path = project / "runs" / "daily-training-status.json"
        try:
            previous = _read_status(state_path)
            failure = _failure_report(project, run_id, dt.datetime.now(KST).date(), str(exc), previous)
            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        except Exception as report_exc:
            print(f"daily training failed: {type(report_exc).__name__}", file=sys.stderr)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
