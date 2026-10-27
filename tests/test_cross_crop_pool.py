import datetime as dt
import json
import math

import numpy as np
import pytest

from agri_data import dashboard, cross_crop_pool as cp


def _inputs(days=30):
    hist_start,hist_end=dt.date(2021,9,16),dt.date(2025,8,31)
    historical=[]; day=hist_start
    while day<=hist_end:
        historical.append(day); day+=dt.timedelta(days=14)
    eval_start,eval_end=dt.date(2025,9,1),dt.date(2025,10,30)
    evaluation=[eval_start+dt.timedelta(days=i) for i in range((eval_end-eval_start).days+1)]
    dates=historical+evaluation
    products=[]; by_product={}; paired=[]
    for i,p in enumerate(cp.PRODUCTS):
        prices={d:100+10*i+.02*n+2*math.sin(n/45+i) for n,d in enumerate(dates)}
        by_product[p]=prices
        products.append({"id":p,"observations":[{"date":d.isoformat(),"actual":v,"volumeKg":500+n%19} for n,(d,v) in enumerate(prices.items())]})
        targets=[dt.date(2025,9,25)+dt.timedelta(days=j) for j in range(days)]
        for target in targets:
            for h in cp.HORIZONS:
                origin=target-dt.timedelta(days=h); actual=prices[target]
                paired.append({"product":p,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,
                    "actual":actual,"direct":actual*1.04,"persistence":prices[origin]})
    return {"schemaVersion":1,"products":products},{"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":paired},by_product


class _FakeEstimator:
    instances=[]
    def fit(self,x,y,sample_weight):
        self.features=x.shape[1]; self.weights=np.asarray(sample_weight); self.n=len(y)
        self.__class__.instances.append(self); return self
    def predict(self,x):
        assert x.shape[1]==self.features
        return np.zeros(len(x))


def test_global_pool_chronology_weights_features_and_holm_family():
    _FakeEstimator.instances=[]
    prices,report,_=_inputs()
    result=cp.build(prices,report,estimator_factory=_FakeEstimator)
    assert result["protocol"]["parameters"]==cp.MODEL_PARAMS
    assert len(result["pairedRows"])==5*10*30
    assert len(result["comparisons"])==200
    assert all(row["n"]==30 for row in result["comparisons"])
    assert all("holmPValue" in row["dm"]["9"] and "holmPValue" in row["dm"]["28"] for row in result["comparisons"])
    assert all(row["candidateCoverageNondeclining"] for row in result["coverage"].values())
    assert len(_FakeEstimator.instances)==2
    for fit in result["fitAudit"]:
        assert dt.date.fromisoformat(fit["trainTargetEnd"])<dt.date.fromisoformat(fit["fitOrigin"])
        assert dt.date.fromisoformat(fit["trainTargetStart"])>=dt.date.fromisoformat(fit["fitOrigin"])-dt.timedelta(days=1460)
        assert min(fit["trainingRowsByProduct"].values())>=90
        sums=list(fit["weightSumByProduct"].values())
        assert max(sums)-min(sums)<1e-8
    row=result["pairedRows"][0]
    assert list(row["crossCropFeatures"])==list(cp.PRODUCTS)
    assert row["crossCropFeatures"][row["product"]]["reason"]=="own_crop_intentionally_masked"
    assert all(row["crossCropFeatures"][p]["available"] for p in cp.PRODUCTS if p!=row["product"])
    assert len(_FakeEstimator.instances[0].weights)>0
    json.dumps(result,allow_nan=False)


def test_cross_crop_missingness_is_zero_plus_availability_flag():
    prices,report,_=_inputs(days=3)
    # Remove the other crop's observations for a period long enough to stale both reference prices.
    product=next(x for x in prices["products"] if x["id"]==cp.PRODUCTS[1])
    product["observations"]=[r for r in product["observations"] if not (dt.date(2025,9,1)<=dt.date.fromisoformat(r["date"])<=dt.date(2025,9,24))]
    series=cp.dh._observations(prices)
    caches={p:{} for p in cp.PRODUCTS}; days={p:tuple(series[p]) for p in cp.PRODUCTS}
    origin,target=dt.date(2025,9,24),dt.date(2025,9,25)
    feature=cp._features(cp.PRODUCTS[0],origin,target,1,series,caches,days)
    assert feature is not None
    assert feature[2][cp.PRODUCTS[1]]["available"] is False
    assert "reason" in feature[2][cp.PRODUCTS[1]]
    base=cp.modeling._feature_row(series[cp.PRODUCTS[0]],origin,target,1,{})
    cross_start=len(base[1])+len(cp.PRODUCTS)
    other_order=[p for p in cp.PRODUCTS if p!=cp.PRODUCTS[0]]
    missing_offset=other_order.index(cp.PRODUCTS[1])*2
    assert feature[1][cross_start+missing_offset]==0.0
    assert feature[1][cross_start+missing_offset+1]==0.0
    # Across target products, each cross-crop slot keeps the same global column meaning.
    schema_origin,schema_target=dt.date(2025,8,31),dt.date(2025,9,1)
    for target_product in cp.PRODUCTS:
        target_base=cp.modeling._feature_row(series[target_product],schema_origin,schema_target,1,{})
        current=cp._features(target_product,schema_origin,schema_target,1,series,{p:{} for p in cp.PRODUCTS},days)
        assert len(current[1])==len(target_base[1])+len(cp.PRODUCTS)+2*len(cp.PRODUCTS)
        assert current[1][len(target_base[1]):len(target_base[1])+len(cp.PRODUCTS)]==tuple(
            1.0 if target_product==label else 0.0 for label in cp.PRODUCTS)
        for slot_product in cp.PRODUCTS:
            slot=len(target_base[1])+len(cp.PRODUCTS)+2*cp.PRODUCTS.index(slot_product)
            if slot_product==target_product:
                assert current[1][slot:slot+2]==(0.0,0.0)
                assert current[2][slot_product]["reason"]=="own_crop_intentionally_masked"
            else:
                evidence=current[2][slot_product]
                if evidence["available"]:
                    assert current[1][slot:slot+2]==(evidence["logChange"],1.0)
                else:
                    assert current[1][slot:slot+2]==(0.0,0.0)


def test_insufficient_crop_rows_preserves_month_and_missing_evidence():
    prices,report,_=_inputs(days=2)
    # Keep test actuals but remove all earlier rows so none of the crops has enough pre-origin training samples.
    for product in prices["products"]:
        product["observations"]=[r for r in product["observations"] if dt.date.fromisoformat(r["date"])>=dt.date(2025,9,10)]
    result=cp.build(prices,report,estimator_factory=_FakeEstimator)
    assert result["pairedRows"]==[]
    assert all(fit["status"]=="insufficient_crop_training" for fit in result["fitAudit"])
    assert all(value["candidatePairedCount"]==0 for value in result["coverage"].values())
    assert all(value["missingOriginsTargets"] for value in result["coverage"].values())
    json.dumps(result,allow_nan=False)


def test_report_identity_validation_and_immutable_output(tmp_path):
    prices,report,_=_inputs(days=1)
    report["pairedRows"].append(dict(report["pairedRows"][0]))
    with pytest.raises(ValueError,match="duplicate direct report identity"):
        cp.build(prices,report,estimator_factory=_FakeEstimator)
    path=tmp_path/"audit.json"; path.write_bytes(b"existing")
    with pytest.raises(FileExistsError): cp._write_immutable(path,b"replacement")
    assert path.read_bytes()==b"existing"


def test_real_sklearn_model_accepts_frozen_configuration_and_weights():
    model=cp._make_estimator()
    x=np.asarray([[float(i),float(i%5)] for i in range(30)])
    y=np.asarray([math.sin(i/5) for i in range(30)])
    weights=np.ones(30)
    model.fit(x,y,sample_weight=weights)
    assert len(model.predict(x[:2]))==2
