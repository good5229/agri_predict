import datetime as dt
import math

import pytest

from agri_data import leaf_root_bias as lb


def hist(origin, dates):
    return [(t, 110.0, 100.0) for t in dates]


def test_correction_uses_only_closed_90_day_asof_window():
    origin = dt.date(2026, 5, 1)
    dates = [origin-dt.timedelta(days=d) for d in range(14, 104)]
    result, audit = lb._correct(hist(origin, dates), origin, 100.0)
    assert audit["distinctTargets"] == 90
    assert audit["historyStart"] == (origin-dt.timedelta(days=103)).isoformat()
    assert audit["historyEnd"] == (origin-dt.timedelta(days=14)).isoformat()
    assert result == pytest.approx(100*math.exp(.04))


def test_insufficient_history_falls_back_and_future_and_too_old_are_excluded():
    origin=dt.date(2026,5,1)
    dates=[origin-dt.timedelta(days=d) for d in range(14,28)]
    dates += [origin-dt.timedelta(days=13), origin-dt.timedelta(days=104)]
    pred,audit=lb._correct(hist(origin,dates),origin,123.0)
    assert pred==123.0
    assert audit["reason"]=="insufficient_history"
    assert audit["distinctTargets"]==14


def test_duplicate_and_malformed_report_rows_rejected():
    base={"product":"cabbage","horizon":1,"target":"2025-09-25","origin":"2025-09-24","actual":10,"direct":9,"persistence":8}
    report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[base,dict(base)]}
    with pytest.raises(ValueError,match="duplicate"):
        lb.rows(report)
    with pytest.raises(ValueError,match="positive and finite"):
        lb.rows({"protocol":report["protocol"],"pairedRows":[{**base,"direct":float("nan")}]})
    with pytest.raises(ValueError,match="mismatch"):
        lb.rows({"protocol":report["protocol"],"pairedRows":[{**base,"horizon":2}]})


def test_pinned_producer_shape_derives_origin_when_omitted():
    row={"product":"cabbage","horizon":3,"target":"2025-09-25","actual":10,"direct":9,"persistence":8}
    key=next(iter(lb.rows({"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[row]})))
    assert key[1]==dt.date(2025,9,22)


def test_dm_families_have_200_tests_each_and_holm_adjustment():
    rows=[]
    # The default bandwidth tests use p=1 for undersized samples; still retain
    # them in both complete frozen multiplicity families.
    for p in lb.dashboard.PRODUCTS:
        for h in lb.HORIZONS:
            for comp in lb.COMPARATORS:
                for loss in lb.LOSSES:
                    rows.append({"product":p,"horizon":h,"comparator":comp,"loss":loss,
                                 "dm":{"9":{"pValue":.01},"28":{"pValue":.02}}})
    assert len(rows)==lb.FAMILY_SIZE==200
    for bw in lb.BANDWIDTHS:
        family=[r["dm"][str(bw)] for r in rows]
        lb.dh._holm(family)
        assert all("holmPValue" in x for x in family)
        assert family[0]["holmPValue"]==1.0


def test_onion_garlic_direct_unchanged_and_eval_window_limits_pairs():
    rows=[]
    for p in lb.dashboard.PRODUCTS:
        for h in (1,):
            for day in (lb.START-dt.timedelta(days=1),lb.START,lb.END,lb.END+dt.timedelta(days=1)):
                origin=day-dt.timedelta(days=h)
                rows.append({"product":p,"horizon":h,"target":day.isoformat(),"origin":origin.isoformat(),
                             "actual":100,"direct":90,"persistence":95})
    report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":rows}
    with pytest.raises(ValueError,match="count"):
        lb.build(report)
    result=lb.build(report,expected_pairs=10)
    assert all(lb.START<=dt.date.fromisoformat(r["target"])<=lb.END for r in result["pairedRows"])
    assert len(result["pairedRows"])==len(lb.dashboard.PRODUCTS)*2
    assert all(r["candidate"]==r["directRidge"] for r in result["pairedRows"] if r["product"] in {"onion","garlic"})
    assert len(result["comparisons"])==200
    assert len(result["jointConclusions"])==50
    assert all(c["conclusion"]=="inference_unavailable" for c in result["jointConclusions"])
    assert result["limitations"] and result["protocol"]["expectedPairedRows"]==10
