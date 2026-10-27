import datetime as dt
import math

import pytest

from agri_data import five_crop_failure_diagnosis as diag
from agri_data import five_crop_specialists as spec


def _synthetic():
    forecasts=[]; actuals={}
    for ci,crop in enumerate(spec.CROPS):
        for i,target in enumerate((dt.date(2025,10,1),dt.date(2026,2,1),dt.date(2026,7,1))):
            h=i+1; origin=target-dt.timedelta(days=h)
            key={"product":crop,"horizon":h,"origin":origin.isoformat(),"target":target.isoformat()}
            fixed=100.
            extra=110.+ci
            candidate=105.+ci
            if crop=="radish" and i==0: actual=90.
            else: actual=120.+i
            preds={spec.CANDIDATE:candidate,spec.ABLATION:108.,spec.FIXED:fixed,
                   spec.EXTRA:extra,spec.DIRECT:99.,spec.SARIMA:101.}
            fb_status=("known" if crop=="pepper" and i<2 else "unknown")
            fb_value=(i==0) if fb_status=="known" else None
            row={**key,"predictions":preds,"fallback":False,"sourceFallback":fb_value,
                 "sourceFallbackStatus":fb_status,"sourceFallbackReason":None}
            forecasts.append(row); actuals[spec._key(row)]=actual
    training={c:{"trainingRowProvenance":[
        {"extraSignal":90.,"candidateFeatures":[0.,0.,0.,0.,-.1],"targetLogRatio":-.05},
        {"extraSignal":110.,"candidateFeatures":[0.,0.,0.,0.,.1],"targetLogRatio":.08}],
        "eligibleTrainingRows":2,"uniqueTrainingTargets":2,"status":"fitted"} for c in spec.CROPS}
    return forecasts,actuals,training


def test_diagnosis_reports_fixed_paired_groups_correction_and_associations():
    forecasts,actuals,training=_synthetic()
    result=diag.diagnose_rows(forecasts,actuals,training)
    assert result["baselineRule"]["cabbage"]==spec.DIRECT
    assert result["baselineRule"]["onion"]==spec.EXTRA
    assert result["byCrop"]["onion"]["pairedVsStrongest"]["n"]==3
    assert result["byCrop"]["onion"]["byQuarter"]["2025-Q4"]["n"]==1
    assert result["byCrop"]["onion"]["byHorizon"]["3"]["n"]==1
    assert result["byCrop"]["radish"]["correctionCalibration"]["wrongSignShareAmongTrueNonzero"]==pytest.approx(1/3)
    assert result["byCrop"]["cabbage"]["signalTargetAssociation"]["trainingA"]["n"]==2
    assert result["byCrop"]["garlic"]["sourceFallback"]["fallbackCounts"]["unknown"]==3
    pepper=result["byCrop"]["pepper"]
    assert pepper["pairedVsStrongestBySourceFallback"]["knownFallback"]["pairedVsStrongest"]["n"]==1
    assert pepper["pairedVsStrongestBySourceFallback"]["knownNonFallback"]["pairedVsStrongest"]["n"]==1
    assert pepper["pairedVsStrongestBySourceFallback"]["unknown"]["pairedVsStrongest"]["n"]==1
    ratios=pepper["meanSignalAndTargetLogRatios"]
    assert ratios["trainingA"]["n"]==2 and ratios["trainingA"]["meanSignalLogRatio"]==pytest.approx(0)
    assert ratios["testB"]["n"]==3 and ratios["testB"]["meanTargetLogRatio"] is not None
    assert "marketSignalDistance" in result["byCrop"]["onion"]
    assert "actual" not in str(result)


def test_diagnosis_excludes_missing_targets_and_reports_loss_delta():
    forecasts,actuals,training=_synthetic()
    missing=spec._key(forecasts[0]); actuals.pop(missing)
    result=diag.diagnose_rows(forecasts,actuals,training)
    cabbage=result["byCrop"]["cabbage"]
    assert cabbage["plannedRows"]==3
    assert cabbage["validActualRows"]==2
    assert cabbage["pairedVsStrongest"]["n"]==2
    assert cabbage["pairedVsStrongest"]["candidateMinusBaselineMapePp"] is not None


def test_quarter_and_correlation_helpers_are_deterministic_with_ties():
    assert diag._quarter("2025-12-31")=="2025-Q4"
    assert diag._quarter("2026-01-01")=="2026-Q1"
    assert diag._association([(1.,3.),(1.,4.),(2.,5.)])["spearman"]==pytest.approx(math.sqrt(.75))


def test_zero_correction_is_separate_from_wrong_sign_and_onion_ratio_skips_near_zero():
    correction=diag._correction([
        {"actual":110.,"predictions":{spec.FIXED:100.,spec.CANDIDATE:100.}},
        {"actual":90.,"predictions":{spec.FIXED:100.,spec.CANDIDATE:110.}},
    ])
    assert correction["wrongSignShareAmongTrueNonzero"]==pytest.approx(.5)
    assert correction["zeroPredictionShareAmongTrueNonzero"]==pytest.approx(.5)
    forecasts,actuals,training=_synthetic()
    row=next(r for r in forecasts if r["product"]=="onion")
    row["predictions"][spec.EXTRA]=100.5
    result=diag.diagnose_rows(forecasts,actuals,training)
    distance=result["byCrop"]["onion"]["marketSignalDistance"]
    assert distance["nearZeroDenominatorN"]==1
    assert distance["correctionRatioEligibleN"]==2
    assert distance["correctionRatioMedian"] is not None
    assert distance["correctionRatioP10"]<=distance["correctionRatioMedian"]<=distance["correctionRatioP90"]
