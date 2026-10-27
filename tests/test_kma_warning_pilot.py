import datetime as dt
import math

from agri_data import dashboard
from agri_data.kma_warning_csv import WarningRow
from agri_data import kma_warning_pilot as pilot


def _row(stamp, region=pilot.REGIONS[0]):
    return WarningRow(stamp, region, "", "", "", dt.datetime.fromisoformat(stamp))


def test_warning_features_exclude_origin_and_count_half_open_windows():
    origin=dt.date(2024,8,8)
    rows=[_row("2024-08-07 09:59:00"),_row("2024-08-07 10:00:00"),_row("2024-08-08 09:59:00",pilot.REGIONS[1]),
          _row("2024-08-08 10:00:00",pilot.REGIONS[0])]
    got=pilot.warning_features(rows,origin,query_start=dt.date(2024,8,1),query_end=dt.date(2024,10,31))
    assert got==[math.log(2),math.log(2),math.log(3),math.log(2)]


def test_warning_features_refuse_incomplete_lookback():
    assert pilot.warning_features([],dt.date(2024,8,7),query_start=dt.date(2024,8,1),query_end=dt.date(2024,10,31)) is None


def test_direct_training_never_uses_targets_at_or_after_origin(monkeypatch):
    origin=dt.date(2024,1,10)
    series={origin-dt.timedelta(days=i):(float(100+i),1.) for i in range(1,140)}
    seen=[]
    def capture(_series,sample_origin,target,horizon,_days,_logs):
        seen.append(target)
        if target >= origin: raise AssertionError("leaked label")
        return target,(1.,2.)
    monkeypatch.setattr(pilot.live_forecast,"_strict_feature_row",capture)
    class Estimator:
        def fit(self,x,y): return self
    monkeypatch.setattr(pilot.modeling,"_new_estimator",lambda _name:Estimator())
    model,n=pilot._fit_direct(series,origin,1,tuple(sorted(series)),tuple(0. for _ in series))
    assert model is not None and n==len(series)
    assert max(seen)<origin


def test_missing_correction_does_not_fall_back_to_direct():
    assert pilot.corrected_prices(120.,100.,[0.,0.,0.,0.],None,0.)==(None,None)
    assert pilot.corrected_prices(120.,100.,None,object(),0.)==(None,None)


def test_metrics_use_only_common_evaluable_pairs_and_family_is_90():
    assert pilot.FAMILY_SIZE==90
    rows=[]
    for pid in dashboard.PRODUCTS:
        for horizon in pilot.HORIZONS:
            rows.append({"product":pid,"origin":"2024-08-08","target":f"2024-08-{8+horizon:02}","horizon":horizon,
                         "actual":100.,"candidate":101.,"ablation":102.,"direct":103.,"persistence":104.,
                         "statuses":{"candidate":"available","ablation":"available","direct":"available","persistence":"available"},"pairedEvaluable":True})
            if horizon==1:
                rows.append({"product":pid,"origin":"2024-08-09","target":"2024-08-10","horizon":horizon,
                             "actual":100.,"candidate":99.,"ablation":None,"direct":98.,"persistence":100.,
                             "statuses":{"candidate":"available","ablation":"unavailable","direct":"available","persistence":"available"},"pairedEvaluable":False})
    out=pilot._summarize(rows)
    assert out["coverage"]["commonPairs"]==len(dashboard.PRODUCTS)*len(pilot.HORIZONS)
    assert out["pairedMetrics"]["candidate"][f"{next(iter(dashboard.PRODUCTS))}/h1"]["n"]==1
    assert out["equalWeightMacroMape"]["candidate"]==1.0
    assert len(out["dmTestsByHacCalendarDays"]["9"])==pilot.FAMILY_SIZE


def test_dm_conclusion_requires_holm_significance():
    pid=next(iter(dashboard.PRODUCTS)); rows=[]
    for i in range(30):
        date=dt.date(2024,8,1)+dt.timedelta(days=i)
        direct=100.0 if i<16 else 100.2
        rows.append({"product":pid,"origin":date.isoformat(),"target":(date+dt.timedelta(days=1)).isoformat(),"horizon":1,
                     "actual":100.,"candidate":100.1,"ablation":direct,"direct":direct,"persistence":direct,
                     "statuses":{"candidate":"available","ablation":"available","direct":"available","persistence":"available"},"pairedEvaluable":True})
    out=pilot._summarize(rows)
    test=next(x for x in out["dmTestsByHacCalendarDays"]["9"] if x["product"]==pid and x["horizon"]==1 and x["comparator"]=="direct" and x["loss"]=="AE")
    assert test["status"]=="tested" and test["pValue"]>.05
    assert not test["holmSignificant"] and test["conclusion"]=="inconclusive"


def test_crop_mae_worsening_uses_ratio_of_equal_horizon_means():
    selected=next(iter(dashboard.PRODUCTS)); rows=[]
    for pid in dashboard.PRODUCTS:
        for horizon in pilot.HORIZONS:
            actual=2000.
            direct_error=1000. if pid==selected and horizon==10 else 100.
            candidate_error=1150. if pid==selected and horizon==10 else 90. if pid==selected else direct_error
            rows.append({"product":pid,"origin":"2024-08-08","target":f"2024-08-{8+horizon:02}","horizon":horizon,
                         "actual":actual,"candidate":actual+candidate_error,"ablation":actual+direct_error,"direct":actual+direct_error,"persistence":actual+direct_error,
                         "statuses":{"candidate":"available","ablation":"available","direct":"available","persistence":"available"},"pairedEvaluable":True})
    out=pilot._summarize(rows)
    assert math.isclose(out["cropWorseningVsDirect"][selected]["maeRelativeIncrease"],(443.3333333333333/400)-1)
    assert not out["frozenExpansionGate"]["cropWorseningWithinDirectLimits"]
