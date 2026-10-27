import datetime as dt
import math
import pytest

from agri_data import dashboard
from agri_data import national_volume_lead_screen as screen


def test_volume_pressure_uses_signed_observed_days_at_cutoff_only():
    cutoff=dt.date(2025,1,31)
    series={cutoff-dt.timedelta(days=6):10.0,cutoff-dt.timedelta(days=5):-2.0,
            cutoff-dt.timedelta(days=27):4.0,cutoff-dt.timedelta(days=20):8.0,
            cutoff+dt.timedelta(days=1):100000.0}
    pressure,detail=screen._volume_pressure(series,cutoff)
    assert pressure==pytest.approx(math.log(4.0/6.0))
    assert detail["sevenDayNegativeObserved"]==1
    assert detail["prior21DayObservedDays"]==2


def test_volume_pressure_does_not_fill_missing_days_or_hide_nonpositive_means():
    cutoff=dt.date(2025,2,1)
    series={cutoff:5.0,cutoff-dt.timedelta(days=8):2.0}
    value,detail=screen._volume_pressure(series,cutoff)
    assert value==pytest.approx(math.log(5/2))
    assert detail["sevenDayObservedDays"]==1
    assert detail["prior21DayObservedDays"]==1
    series[cutoff]=-5.0
    value,detail=screen._volume_pressure(series,cutoff)
    assert value is None
    assert detail["reason"]=="nonpositive_signed_volume_mean"


def test_volume_source_preserves_null_negative_and_zero_states():
    products=[]
    for product in dashboard.PRODUCTS:
        observations=[]
        if product in screen.PRODUCTS:
            observations=[{"date":"2025-01-01","volumeKg":None},
                          {"date":"2025-01-02","volumeKg":-2},
                          {"date":"2025-01-03","volumeKg":0}]
        products.append({"id":product,"observations":observations})
    values,coverage=screen._volume_source({"schemaVersion":1,"products":products})
    assert values["cabbage"][dt.date(2025,1,1)] is None
    assert values["cabbage"][dt.date(2025,1,2)]==-2
    assert coverage["cabbage"]["volumeValueStates"]=={"negative_signed":1,"null":1,"zero":1}


def test_target_date_blocks_use_all_calendar_origins_and_freeze_eligibility_rules():
    start=dt.date(2024,7,1);end=dt.date(2026,7,15)
    days=(end-start).days+1
    national={p:{} for p in screen.PRODUCTS};volumes={p:{} for p in screen.PRODUCTS}
    for i in range(days):
        day=start+dt.timedelta(days=i)
        for j,product in enumerate(screen.PRODUCTS):
            national[product][day]=100+j+i*.01
            volumes[product][day]=100+j+(i%19)
    report=screen.build_screen(national,volumes,{"synthetic":True})
    first_start,first_end=screen.BLOCKS[0]
    first=report["blocks"][f"{first_start}_{first_end}"]["cabbage/5"]
    assert first["plannedOrigins"]==(first_end-first_start).days+1
    assert first["targetObservedOrigins"]==first["plannedOrigins"]
    assert first["pairedFeatureTargetCount"]<=first["targetObservedOrigins"]
    assert len(report["followOnEligibility"])==len(screen.PRODUCTS)*len(screen.HORIZONS)
    criterion=report["followOnEligibility"]["cabbage/5"]
    assert criterion["thresholds"]["minimumPairCoverageEachBlock"]==.70
    assert criterion["heuristicOnly"] is True
    assert "significance" in report["interpretation"]
    assert report["reportSha256"]==screen._sha(__import__("json").dumps(
        {k:v for k,v in report.items() if k!="reportSha256"},sort_keys=True,ensure_ascii=False,
        separators=(",",":"),allow_nan=False).encode())


def test_archive_price_helper_keeps_signed_source_states_for_coverage():
    # A miniature parsed-source object verifies that only positive national
    # prices become targets while incomplete/nonpositive rows remain counted.
    parsed={"observations":{p:{"2025-01-01":{"price":"10","priceReason":None},
                                "2025-01-02":{"price":None,"priceReason":"nonpositive_signed_quantity"}}
                            for p in screen.PRODUCTS}}
    priced=screen.price_selection._prices(parsed)
    assert priced["cabbage"]=={dt.date(2025,1,1):10.0}
    assert parsed["observations"]["cabbage"]["2025-01-02"]["priceReason"]=="nonpositive_signed_quantity"
