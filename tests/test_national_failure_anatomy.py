import datetime as dt
import json

import pytest

from agri_data import national_failure_anatomy as anatomy


def _row(target, horizon, actual, *, ratio=120.0, exact=115.0, fixed=105.0, ridge=108.0):
    day=dt.date.fromisoformat(target);origin=day-dt.timedelta(days=horizon);cutoff=origin-dt.timedelta(days=2)
    provenance={"fallback":False,"fallbackReason":None,"fitCutoff":"2026-03-01",
                "featureCutoff":cutoff.isoformat(),"lastObservedDate":cutoff.isoformat(),"lastObservedPrice":100.0}
    return {"product":"radish","origin":origin.isoformat(),"target":target,"horizon":horizon,"cutoff":cutoff.isoformat(),
        "predictions":{"ridge_alpha_100":ridge,"calendar_sarima":100.0,"fixed_blend_50_50":fixed,
                       "persistence":100.0,anatomy.ratio_model.METHOD:ratio,
                       anatomy.exact_model.weighted.METHOD:100.0,
                       anatomy.exact_model.METHOD:exact},
        "ratioLineage":dict(provenance),"exactLineage":dict(provenance),"actual":actual}


def _join(rows):
    indexed={(r["product"],r["origin"],r["target"],r["horizon"]):r for r in rows}
    actuals={p:{} for p in anatomy.PRODUCTS}
    for row in rows:actuals[row["product"]][dt.date.fromisoformat(row["target"])]=row["actual"]
    return anatomy._common_rows(indexed,actuals)


def test_common_join_requires_valid_actual_all_forecasts_and_matching_asof():
    valid=_row("2026-04-01",5,110)
    bad_prediction=_row("2026-04-02",5,110,exact=float("nan"))
    forecasts={(r["product"],r["origin"],r["target"],r["horizon"]):r for r in (valid,bad_prediction)}
    actuals={p:{} for p in anatomy.PRODUCTS};actuals["radish"][dt.date(2026,4,1)]=110
    actuals["radish"][dt.date(2026,4,2)]=110
    joined=anatomy._common_rows(forecasts,actuals)
    assert len(joined)==1
    assert joined[0]["actualLogChange"]==pytest.approx(__import__("math").log(1.1))
    bad_asof=_row("2026-04-03",5,110);bad_asof["exactLineage"]["lastObservedDate"]="2026-03-01"
    with pytest.raises(ValueError,match="as-of price provenance"):
        _join([bad_asof])


def test_cell_metrics_include_paired_mae_mape_and_signed_bias():
    rows=[_row("2026-04-01",5,100,ratio=110,exact=90,fixed=105,ridge=108),
          _row("2026-04-02",5,100,ratio=120,exact=100,fixed=100,ridge=100)]
    rows=_join(rows);cell=anatomy._cell(rows)
    assert cell["pairedN"]==2
    assert cell["metricsByMethod"][anatomy.ratio_model.METHOD]["mae"]==15
    assert cell["metricsByMethod"][anatomy.ratio_model.METHOD]["mape"]==pytest.approx(15)
    assert cell["metricsByMethod"][anatomy.ratio_model.METHOD]["signedBias"]==pytest.approx(15)
    assert cell["actualTargetToLastAsOfLogPriceChange"]["mean"]==pytest.approx(0.0)


def test_radish_top_dates_rank_ratio_minus_fixed_ape_and_keep_lineage_flags():
    rows=[_row(f"2026-07-{day:02d}",5,100,ratio=ratio,fixed=fixed) for day,ratio,fixed in
          ((1,140,105),(2,130,100),(3,115,110))]
    top=anatomy._radish_top(_join(rows),2)
    assert [r["target"] for r in top]==["2026-07-01","2026-07-02"]
    assert top[0]["ratioMinusFixed50ApePp"]==pytest.approx(35)
    assert top[0]["provenanceFlags"]["ratioFallback"] is False
    assert top[0]["asofPrice"]==100


def test_forecast_extrema_are_by_crop_horizon_method():
    r1=_row("2026-04-01",5,100,ratio=80)
    r2=_row("2026-09-30",5,100,ratio=140)
    forecasts={i:r for i,r in enumerate((r1,r2))}
    summary=anatomy._forecast_extrema(forecasts)
    entry=summary[f"radish/5/{anatomy.ratio_model.METHOD}"]
    assert entry["issued"]==2
    assert entry["minimum"]["value"]==80
    assert entry["maximum"]["value"]==140


def test_pinned_loader_fails_closed_on_byte_hash_mismatch(tmp_path):
    path=tmp_path/"manifest.json";path.write_text(json.dumps({"ok":True}),encoding="utf-8")
    with pytest.raises(ValueError,match="byte SHA-256 mismatch"):
        anatomy._load_pinned(path,"0"*64,lambda doc:None)
