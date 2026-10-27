import datetime as dt
import json

import pytest

from agri_data import dashboard, mape_quantile as mq


def _inputs(days=30):
    start=dt.date(2024,1,1); end=dt.date(2025,10,30)
    dates=[start+dt.timedelta(days=i) for i in range((end-start).days+1)]
    products=[]; paired=[]
    for pi, product in enumerate(dashboard.PRODUCTS):
        observations=[]
        prices={}
        for i,day in enumerate(dates):
            price=100+pi*7+i*.04+(i%13)*.2
            prices[day]=price
            observations.append({"date":day.isoformat(),"actual":price,"volumeKg":1000+i%17})
        products.append({"id":product,"observations":observations})
        for i in range(days):
            target=dt.date(2025,9,25)+dt.timedelta(days=i)
            actual=prices[target]
            for horizon in mq.HORIZONS:
                paired.append({"product":product,"origin":(target-dt.timedelta(days=horizon)).isoformat(),
                    "target":target.isoformat(),"horizon":horizon,"actual":actual,
                    "direct":actual*1.05,"persistence":actual*1.1})
    return {"schemaVersion":1,"products":products}, {"schemaVersion":1,
        "protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":paired}


class _FakeEstimator:
    def fit(self, x, y, sample_weight):
        assert len(x)==len(y)==len(sample_weight)
        assert all(v>0 for v in sample_weight)
        self.width=x.shape[1]
        return self
    def predict(self, x):
        assert x.shape[1]==self.width
        return [0.0]*len(x)


class _InfiniteEstimator(_FakeEstimator):
    def predict(self, x):
        return [float("inf")]*len(x)


def test_frozen_model_alignment_coverage_dm_family_and_serialization():
    prices, report=_inputs()
    result=mq.build(prices,report,estimator_factory=_FakeEstimator)
    assert result["protocol"]["parameters"]==mq.MODEL_PARAMS
    assert len(result["pairedRows"])==5*10*30
    assert len(result["comparisons"])==200
    assert all(row["n"]==30 for row in result["comparisons"])
    assert all("holmPValue" in row["dm"]["9"] and "holmPValue" in row["dm"]["28"] for row in result["comparisons"])
    assert all(row["candidateCoverageNondeclining"] for row in result["coverage"].values())
    for entries in result["fitAudit"].values():
        for fit in entries:
            assert dt.date.fromisoformat(fit["fitTrainTargetEnd"]) < dt.date.fromisoformat(fit["fitOrigin"])
            assert fit["fitDistinctTargets"]>=90
    assert result["pairedRows"][0]["sourceDate"] <= result["pairedRows"][0]["origin"]
    json.dumps(result,allow_nan=False)


def test_rejects_actual_disagreement_and_identity_errors():
    prices,report=_inputs(days=2)
    report["pairedRows"][0]["actual"]+=1
    with pytest.raises(ValueError,match="actual mismatch"):
        mq.build(prices,report,estimator_factory=_FakeEstimator)
    prices,report=_inputs(days=2)
    report["pairedRows"][0]["origin"]="2020-01-01"
    with pytest.raises(ValueError,match="origin/horizon mismatch"):
        mq.build(prices,report,estimator_factory=_FakeEstimator)


def test_insufficient_training_is_explicit_and_keeps_denominator():
    prices,report=_inputs(days=2)
    # Restrict every series to fewer than 90 distinct targets.
    for product in prices["products"]:
        product["observations"]=product["observations"][-70:]
    result=mq.build(prices,report,estimator_factory=_FakeEstimator)
    assert result["pairedRows"]==[]
    assert all(v["eligibleDirectReportRows"]==2 and v["candidatePairedCount"]==0 for v in result["coverage"].values())
    assert all("insufficient_training" in v["missingReasons"] for v in result["coverage"].values())


def test_nonfinite_predictions_are_missing_rows_not_serialization_errors():
    prices,report=_inputs(days=2)
    result=mq.build(prices,report,estimator_factory=_InfiniteEstimator)
    assert result["pairedRows"]==[]
    assert all("prediction_failed:nonfinite_prediction" in v["missingReasons"] for v in result["coverage"].values())
    assert all(entry["status"]=="fitted_with_prediction_gaps" for rows in result["fitAudit"].values() for entry in rows)
    json.dumps(result,allow_nan=False)


def test_immutable_writer_refuses_existing_path(tmp_path):
    path=tmp_path/"report.json"; path.write_bytes(b"original")
    with pytest.raises(FileExistsError): mq._write_immutable(path,b"replacement")
    assert path.read_bytes()==b"original"


def test_real_sklearn_quantile_estimator_accepts_frozen_weights():
    np=pytest.importorskip("numpy")
    model=mq._make_estimator()
    x=np.asarray([[float(i),float(i%3)] for i in range(24)])
    y=np.asarray([i*.01 for i in range(24)])
    weights=np.asarray([1.0+i/30 for i in range(24)])
    model.fit(x,y,sample_weight=weights)
    prediction=model.predict(x[:2])
    assert len(prediction)==2 and all(float(v)==float(v) for v in prediction)
