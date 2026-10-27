"""Deterministic post-forecast scoring for PERIOD_SEASONAL_RELATIVE_ANALOG_V4."""
from __future__ import annotations
import datetime as dt
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from . import period_proxy_experiment as proxy
from . import period_mape10_score as stats
from . import period_seasonal_analog as v4

COMPARISONS=(('Q1','P'),('Q1','B0'))
MECHANISM=(('Q1','Q0'),)

def metric(pairs):
    if not pairs:return {"n":0,"mae":None,"mape":None}
    return {"n":len(pairs),"mae":float(mean(abs(a-p) for a,p in pairs)),
        "mape":float(mean(abs(a-p)/a for a,p in pairs)*100)}

def _load_actuals():
    _,raw=proxy.validate_inputs();series=proxy.series_from_raw(raw)
    targets={c:proxy._target_periods(s) for c,s in series.items()}
    by_key={(c,proxy._period_index(tp),period):v for c in proxy.CROPS for tp,v in targets[c].items()
        for period in (v.get("actual"),) }
    return raw,series,targets,by_key

def _read_exact(meta):
    return v4.read_artifact(meta)

def _loss_pairs(rows,left,right):
    out=[]
    for r in rows:
        a=r.get("actual");l=r.get("forecasts",{}).get(left,{}).get("predicted");q=r.get("forecasts",{}).get(right,{}).get("predicted")
        if a is None or l is None or q is None:continue
        tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
        out.append((tp,float(a),float(l),float(q)))
    return out

def _hypotheses(rows):
    main=[];mech=[]
    for crop in proxy.CROPS:
      for h in proxy.HORIZONS:
        cell=[r for r in rows if r["crop"]==crop and int(r["horizon"])==h]
        for comps,dest in ((COMPARISONS,main),(MECHANISM,mech)):
          for left,right in comps:
            pairs=_loss_pairs(cell,left,right)
            for loss in ("ae","ape"):
              for lag in proxy.HAC_LAGS:
                dest.append({"crop":crop,"horizon":h,"comparison":f"{left}-{right}","loss":loss,
                  "lag":lag,**stats.hac(pairs,loss,lag)})
    if len(main)!=64 or len(mech)!=32:raise ValueError("frozen V4 hypothesis family size mismatch")
    for family in (main,mech):
      for lag in proxy.HAC_LAGS:
        indices=[i for i,x in enumerate(family) if x["lag"]==lag]
        for i,x in zip(indices,stats.holm([family[j] for j in indices])):family[i]=x
    return main,mech

def _combine_bandwidths(items):
    result={}
    for x in items:
        key=(x["crop"],int(x["horizon"]),x["comparison"],x["loss"])
        item=result.setdefault(key,{"crop":key[0],"horizon":key[1],"comparison":key[2],"loss":key[3]})
        item[f"hac{x['lag']}"]={k:v for k,v in x.items() if k not in ("crop","horizon","comparison","loss","lag")}
    return [result[k] for k in sorted(result)]

def _metrics_cells(rows,methods):
    result={}
    for c in proxy.CROPS:
      result[c]={}
      for h in proxy.HORIZONS:
        group=[r for r in rows if r["crop"]==c and int(r["horizon"])==h]
        result[c][str(h)]={}
        for m in methods:
            pairs=[(float(r["actual"]),float(r["forecasts"][m]["predicted"])) for r in group
                if r.get("actual") is not None and r.get("forecasts",{}).get(m,{}).get("predicted") is not None]
            result[c][str(h)][m]=metric(pairs)
    return result

def _macro8(cells,methods):
    result={}
    for m in methods:
      vals=[cells[c][str(h)][m] for c in proxy.CROPS for h in proxy.HORIZONS]
      maes=[v["mae"] for v in vals if v["mae"] is not None];mapes=[v["mape"] for v in vals if v["mape"] is not None]
      result[m]={"mae":float(np.mean(maes)) if maes else None,"mape":float(np.mean(mapes)) if mapes else None,
                 "supportedCells":len(maes),"meanCellN":float(np.mean([v["n"] for v in vals])) if vals else None}
    return result

def _paired_macro(rows,left,right):
    cells={};mae_delta=[];mape_delta=[]
    for c in proxy.CROPS:
      for h in proxy.HORIZONS:
        group=[r for r in rows if r["crop"]==c and int(r["horizon"])==h]
        pairs=_loss_pairs(group,left,right)
        if not pairs:cells[f"{c}:{h}"]={"n":0,"left":None,"right":None};continue
        lm=metric([(x[1],x[2]) for x in pairs]);rm=metric([(x[1],x[3]) for x in pairs])
        cells[f"{c}:{h}"]={"n":len(pairs),"left":lm,"right":rm,
             "deltaMAE":lm["mae"]-rm["mae"],"deltaMAPE":lm["mape"]-rm["mape"]}
        mae_delta.append(lm["mae"]-rm["mae"]);mape_delta.append(lm["mape"]-rm["mape"])
    return {"cells":cells,"supportedCells":len(mae_delta),
       "equalCellDeltaMAE":float(np.mean(mae_delta)) if mae_delta else None,
       "equalCellDeltaMAPE":float(np.mean(mape_delta)) if mape_delta else None}

def _yearly_metrics(rows,methods):
    out={}
    for year in sorted({r["targetStart"][:4] for r in rows}):
        subset=[r for r in rows if r["targetStart"][:4]==year]
        cells=_metrics_cells(subset,methods);out[year]={"byCell":cells,"MAPE8_MAE8":_macro8(cells,methods)}
    return out

def _diagnostic_bins(rows,features_by_key):
    bins={}
    for r in rows:
        if r.get("actual") is None:continue
        f=features_by_key[(r["crop"],int(r["horizon"]),v4.period_key((dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"])) ))]
        ref=f.get("referencePrice")
        if ref is None or ref<=0:continue
        actual=float(r["actual"])
        ratio=actual/ref
        groups={"actualVsReference":"rise" if ratio>1 else "fall" if ratio<1 else "flat",
          "jump20":"jump_ge20pct" if abs(ratio-1)>=.2 else "nonjump_lt20pct"}
        for feat in ("trend_1","volume_ratio"):
            value=f.get("features",{}).get(feat)
            groups[feat+"Sign"]="missing" if value is None else "negative" if value<0 else "positive" if value>0 else "zero"
        for dim,label in groups.items():
            cell=bins.setdefault(f"{r['crop']}:{r['horizon']}:{dim}:{label}",{})
            for method,forecast in r.get("forecasts",{}).items():
                pred=forecast.get("predicted")
                if pred is not None:cell.setdefault(method,[]).append((actual,float(pred)))
    return {k:{m:metric(pairs) for m,pairs in v.items()} for k,v in bins.items()}

def _oracle(rows,features_by_key):
    per_cell={};all_values=[]
    for crop in proxy.CROPS:
      for h in proxy.HORIZONS:
        vals=[];supported=0;unsupported=0
        for r in rows:
            if r["crop"]!=crop or int(r["horizon"])!=h or r.get("actual") is None:continue
            key=(crop,h,v4.period_key((dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))))
            f=features_by_key[key];lo=f.get("oraclePoolMinRatio");hi=f.get("oraclePoolMaxRatio")
            if not f.get("oracleEligible") or lo is None or hi is None or f.get("referencePrice") is None:
                unsupported+=1;continue
            truth=float(r["actual"])/float(f["referencePrice"])
            nearest=max(float(lo),min(float(hi),truth))
            vals.append(abs(nearest-truth)/truth*100);supported+=1
        avg=float(np.mean(vals)) if vals else None
        per_cell[f"{crop}:{h}"]={"meanOracleAPE":avg,"availableN":supported,"unavailableN":unsupported}
        if avg is not None:all_values.append(avg)
    return {"perCell":per_cell,"equalCellMeanSupported":float(np.mean(all_values)) if all_values else None,
       "supportedCells":len(all_values),"interpretation":"optimistic full-Q1-pool range bound on available supported rows only; cannot prove impossibility for excluded or fallback rows"}

def _coverage(rows,comparison_rows=None):
    stats_by={}
    for m in ("P","B0","Q0","Q1"):
        available=[r for r in rows if not r.get("policyUnavailable") and r.get("forecasts",{}).get(m,{}).get("predicted") is not None]
        stats_by[m]={"plannedRows":len(rows),"forecastRows":len(available),
           "scoredRows":sum(r.get("actual") is not None for r in available),
           "coverageOfPlanned":len(available)/len(rows) if rows else None,
           "comparisonUniverseRows":len(comparison_rows or []),
           "forecastRowsOnComparisonUniverse":sum(r in (comparison_rows or []) for r in available),
           "comparisonUniverseCoverage":(sum(r in (comparison_rows or []) for r in available)/len(comparison_rows) if comparison_rows else None)}
    q1={"directQ1":0,"fallbackQ0":0,"fallbackB0":0,"missing":0,"policyUnavailable":0,"reasons":{}}
    q0={"directQ0":0,"fallbackB0":0,"missing":0,"policyUnavailable":0,"reasons":{}}
    by_cell={}
    for r in rows:
        crop,h=r["crop"],int(r["horizon"]);cell=by_cell.setdefault(f"{crop}:{h}",{"Q1":{},"Q0":{}})
        for method,totals,field in (("Q1",q1,"q1"),("Q0",q0,"q0")):
            obj=r["forecasts"].get(method,{})
            if r.get("policyUnavailable"):
                totals["policyUnavailable"]+=1
                cell[method]["policyUnavailable"]=cell[method].get("policyUnavailable",0)+1
                continue
            eff=obj.get("effectiveMethod")
            cat="direct" if eff==method else "fallbackQ0" if method=="Q1" and eff=="Q0" else "fallbackB0" if eff=="B0" else "missing"
            if method=="Q0" and cat=="fallbackQ0":cat="missing"
            if obj.get("predicted") is None:cat="missing"
            total_key={"Q1":{"direct":"directQ1","fallbackQ0":"fallbackQ0","fallbackB0":"fallbackB0","missing":"missing"},
                       "Q0":{"direct":"directQ0","fallbackB0":"fallbackB0","missing":"missing"}}[method][cat]
            totals[total_key]+=1;cell[method][cat]=cell[method].get(cat,0)+1
            reason=obj.get("fallbackReason")
            if reason:
                totals["reasons"][reason]=totals["reasons"].get(reason,0)+1
                cell[method].setdefault("reasons",{});cell[method]["reasons"][reason]=cell[method]["reasons"].get(reason,0)+1
    return {"methods":stats_by,"Q1Path":q1,"Q0Path":q0,"byCropHorizon":by_cell}

def _stat_conclusions(paired,boot,main):
    result={}
    for comp in ("Q1-P","Q1-B0","Q1-Q0"):
        p=paired[comp];bs=boot[comp]
        citable=main if comp in ("Q1-P","Q1-B0") else []
        mechanism_significant=None
        # Filled from the separate 16-hypothesis Q1/Q0 family below.
        positive=[];intervals={}
        for n in ("6","12"):
            intervals[n]={}
            for loss in ("mae8","mape8"):
                ci=bs[n][loss].get("ci95")
                intervals[n][loss]=ci
                positive.append(bool(ci and ci[0]>0))
        sig=sum(x["comparison"]==comp and x["loss"]=="ape" and x["lag"]==3 and x.get("holmSignificant") for x in citable)
        delta=p.get("equalCellDeltaMAPE")
        result[comp]={"equalCellDeltaMAPE":delta,"equalCellDeltaMAE":p.get("equalCellDeltaMAE"),
          "bootstrapCI95":intervals,"holmSignificantMainAPECellsHAC3":sig,
          "holmSignificantMechanismAPECellsHAC3":mechanism_significant,
          "bothBlockLengthsCIsAboveZeroForBothLosses":all(positive),
          "interpretation":"observed loss difference only; nonsignificance is not equivalence"}
    return result

def render_korean(report):
    m=report["metrics8EqualCell"];goal=report["MAPE10Goal"]
    q1=m.get("Q1",{});p=m.get("P",{});b=m.get("B0",{});q0=m.get("Q0",{})
    lines=["# PERIOD_SEASONAL_RELATIVE_ANALOG_V4 결과", "",
      "이 결과는 수정 가능성이 있는 과거 가격 스냅숏의 후향 대리평가이며, 공식 전국 순가격·독립시험·전향 성능이 아닙니다.",
      "운영 교체 근거로 사용하지 않습니다.", "",
      f"Q1 8셀 동일가중 MAPE는 {q1.get('mape')}% (목표 10%: {'달성' if goal['met'] else '미달'}, 잔여 격차 {goal['gapPP']}%p), MAE는 {q1.get('mae')}원/kg입니다.",
      f"비교 기준은 P MAPE/MAE {p.get('mape')}% / {p.get('mae')}, B0 {b.get('mape')}% / {b.get('mae')}, Q0 {q0.get('mape')}% / {q0.get('mae')}입니다.",
      "통계적 유의성·목표 달성·coverage·MAE 비악화는 서로 다른 판단이며, 비유의를 동등성으로 해석하지 않습니다.", "",
      "## 품목·거리별 시험 지표", "", "| 품목 | 순 | Q1 n | Q1 MAE | Q1 MAPE | P MAPE | B0 MAPE | Q0 MAPE | 10% 초과분 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cells=report["metricsByCell"]
    for c in proxy.CROPS:
      for h in proxy.HORIZONS:
        cell=cells[c][str(h)];x=cell.get("Q1",{});lines.append(f"| {c} | {h} | {x.get('n')} | {x.get('mae')} | {x.get('mape')}% | {cell.get('P',{}).get('mape')}% | {cell.get('B0',{}).get('mape')}% | {cell.get('Q0',{}).get('mape')}% | {max(0,x['mape']-10) if x.get('mape') is not None else None}pp |")
    lines += ["",f"개별 셀 10% 이하(의도된 전체 표본 포함 시): {goal['cellsAtOrBelow10']}/8. 모든 셀 MAE 비악화(B0 대비): {report['maeNoWorseThanB0']}.",
      f"동일가중 8셀 평균 MAE 비악화(B0 대비): {report.get('macroMaeNoWorseThanB0')}.",
      f"정책 가용 전 12개 행은 유지했고, 계획 키 {report['support']['plannedRows']}개 / P·B0 공통 실제 키 {report['support']['pairedPAndB0']}개를 확인했습니다. 모든 방법은 이 공통 키 우주 안에서만 점수화했고, Q1 표본 누락은 coverage와 n에 드러납니다.", "",
      "## 대비·통계", "", "| 비교 | Q1−기준 MAPE8 | MAE8 | HAC3 Holm 유의 APE 셀 | 6/12순 블록 구간 |", "|---|---:|---:|---:|---|"]
    for comp in ("Q1-P","Q1-B0","Q1-Q0"):
        x=report["conclusions"][comp]; c=x["bootstrapCI95"]
        ci=lambda length,loss: "미산출" if c[length][loss] is None else f"[{c[length][loss][0]:.3g}, {c[length][loss][1]:.3g}]"
        sig=(x["holmSignificantMechanismAPECellsHAC3"] if comp=="Q1-Q0" else x["holmSignificantMainAPECellsHAC3"])
        sig_text=f"{sig}/8" if isinstance(sig,int) else str(sig)
        lines.append(f"| {comp} | {x['equalCellDeltaMAPE']}pp | {x['equalCellDeltaMAE']}원/kg | {sig_text} | MAE6 {ci('6','mae8')}; MAPE6 {ci('6','mape8')}; MAE12 {ci('12','mae8')}; MAPE12 {ci('12','mape8')} |")
    lines += ["", "DM HAC3/HAC6의 전체 셀별 p값·효과·Holm 조정값은 아래 표에 포함합니다. 유의성은 P/B0 주 비교 32가설 가족과 Q1/Q0 메커니즘 16가설 가족에서 별도 처리합니다.",
      "", "### 주 비교: P/B0 (Holm 가족별 32 가설)", "", "| 품목 | 순 | 비교 | 손실 | HAC3 n | 효과 | p | Holm p | HAC6 효과 | p | Holm p |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in report["main32"]:
      h3=x.get("hac3",{});h6=x.get("hac6",{})
      lines.append(f"| {x['crop']} | {x['horizon']} | {x['comparison']} | {x['loss']} | {h3.get('n')} | {h3.get('meanLossDifference')} | {h3.get('p')} | {h3.get('holmP')} | {h6.get('meanLossDifference')} | {h6.get('p')} | {h6.get('holmP')} |")
    lines += ["", "### 메커니즘 비교: Q1/Q0 (별도 Holm 가족 16 가설)", "", "| 품목 | 순 | 비교 | 손실 | HAC3 n | 효과 | p | Holm p | HAC6 효과 | p | Holm p |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in report["mechanism16"]:
      h3=x.get("hac3",{});h6=x.get("hac6",{})
      lines.append(f"| {x['crop']} | {x['horizon']} | {x['comparison']} | {x['loss']} | {h3.get('n')} | {h3.get('meanLossDifference')} | {h3.get('p')} | {h3.get('holmP')} | {h6.get('meanLossDifference')} | {h6.get('p')} | {h6.get('holmP')} |")
    lines += [
      "", "## 개발 구간 선택·점수", ""]
    dev=report.get("developmentMetrics")
    if dev:
      lines += ["P 선택은 V3 고정 정책을 승계했습니다. 개발 구간 B0는 저장된 선택 규칙을 재구성한 후향 선택이며, 시험 결과로 선택하지 않았습니다.",
       "| 품목 | 순 | Q1 MAE | Q1 MAPE | P MAE/MAPE | B0 MAE/MAPE | Q0 MAE/MAPE |", "|---|---:|---:|---:|---:|---:|---:|"]
      for c in proxy.CROPS:
       for h in proxy.HORIZONS:
        x=dev["metricsByCell"][c][str(h)]
        lines.append(f"| {c} | {h} | {x['Q1']['mae']} | {x['Q1']['mape']}% | {x['P']['mae']} / {x['P']['mape']}% | {x['B0']['mae']} / {x['B0']['mape']}% | {x['Q0']['mae']} / {x['Q0']['mape']}% |")
      lines.append(f"개발 구간 8셀 동일가중 Q1 MAPE/MAE: {dev['metrics8EqualCell']['Q1']['mape']}% / {dev['metrics8EqualCell']['Q1']['mae']}원/kg. B0 회고 재구성 가정: {dev['b0RetrospectiveSelectionAssumption']}.")
      lines += ["", "개발 후보 풀 및 fallback (품목·순)", "", "| 품목 | 순 | 평균 Q0 후보/원점 | 평균 Q1 후보/원점 | Q0 지원 원점 | Q1 특징 유효 원점 | Q1 직접 예측 | Q0 fallback | B0 fallback | 미출력 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
      for cell,x in dev["supportAndFallback"]["byCell"].items():
       crop,h=cell.split(":");lines.append(f"| {crop} | {h} | {x['q0CandidatesMean']} | {x['q1CandidatesMean']} | {x['q0CandidateOriginCoverage']} | {x['featureValidOrigins']} | {x['q1Direct']} | {x['q1FallbackQ0']} | {x['q1FallbackB0']} | {x['q1Missing']} |")
      lines += ["", "개발 연도별 동일가중 Q1/P/B0/Q0:", "", "| 연도 | 방법 | MAE8 | MAPE8 |", "|---:|---|---:|---:|"]
      for year,rec in dev["annualMetrics"].items():
       for method in ("Q1","P","B0","Q0"):
        x=rec["MAPE8_MAE8"][method];lines.append(f"| {year} | {method} | {x['mae']} | {x['mape']}% |")
    else: lines.append("개발 구간 점수 artifact가 없어 점수를 산출하지 않았습니다.")
    lines += ["", "## 연도별 시험 분해", "", "| 연도 | 방법 | n(평균 셀) | MAE8 | MAPE8 |", "|---:|---|---:|---:|---:|"]
    for year,record in report["annualMetrics"].items():
      for method in ("Q1","Q0","P","B0"):
        x=record["MAPE8_MAE8"][method]
        lines.append(f"| {year} | {method} | {x['meanCellN']} | {x['mae']} | {x['mape']}% |")
    lines += ["", "## 지원, 연도, 원인 진단", "",
      f"Q1 직접 사용 {report['coverage']['Q1Path']['directQ1']}, Q0 복귀 {report['coverage']['Q1Path']['fallbackQ0']}, B0 복귀 {report['coverage']['Q1Path']['fallbackB0']}, 가용 키 미출력 {report['coverage']['Q1Path']['missing']}, 정책 비가용 {report['coverage']['Q1Path']['policyUnavailable']}.",
      f"Q0 직접 사용 {report['coverage']['Q0Path']['directQ0']}, B0 복귀 {report['coverage']['Q0Path']['fallbackB0']}, 가용 키 미출력 {report['coverage']['Q0Path']['missing']}, 정책 비가용 {report['coverage']['Q0Path']['policyUnavailable']}.",
      "", "### 품목·순별 Q1 지원 경로", "", "| 품목 | 순 | 직접 Q1 | Q0 복귀 | B0 복귀 | 가용 키 미출력 | 정책 비가용 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cell,paths in report["coverage"]["byCropHorizon"].items():
      crop,h=cell.split(":")
      lines.append(f"| {crop} | {h} | {paths['Q1'].get('direct',0)} | {paths['Q1'].get('fallbackQ0',0)} | {paths['Q1'].get('fallbackB0',0)} | {paths['Q1'].get('missing',0)} | {paths['Q1'].get('policyUnavailable',0)} |")
    lines += ["", "### Oracle 지원 분모", "", "| 품목 | 순 | 지원 n | 미지원 n | 범위 하한 MAPE |", "|---|---:|---:|---:|---:|"]
    for cell,x in report["oracle"]["perCell"].items():
      crop,h=cell.split(":");lines.append(f"| {crop} | {h} | {x['availableN']} | {x['unavailableN']} | {x['meanOracleAPE']}% |")
    lines += ["", "전체 합법 Q1 후보 풀 범위 oracle은 낙관적 지표이며 예측 점수가 아닙니다. 미지원·복귀 키의 목표 불가능성을 뜻하지 않습니다.",
      "", "### 사후 진단 분해 (전체 분모와 방법별 점수)", "", "| 품목:순:진단 bin | 방법 | n | MAE | MAPE |", "|---|---|---:|---:|---:|"]
    for key,methods in report["diagnosticBins"].items():
      for method,x in methods.items():lines.append(f"| {key} | {method} | {x['n']} | {x['mae']} | {x['mape']}% |")
    lines += ["", "2024~2026은 이미 반복 열람된 후향 구간입니다. 후보창·K·가중치·품목 규칙은 결과를 보고 변경하지 않습니다.", "",
      f"코드 SHA-256: `{report['analysisCodeSha256']}`; 프로토콜 SHA-256: `{report['protocolSha256']}`.",
      f"입력 무결성 근거: 기존 결과보고서 `{report['preflightEvidence']['oldReportSha256']}`, V3 시험 예측 `{report['preflightEvidence']['v3TestForecastSha256']}`, 데이터 원천 `{report['preflightEvidence']['sourceSha256']}`. 사전검사에서 재집계 값과 고정 V3 점수를 대조했습니다."]
    return "\n".join(lines)+"\n"

def score_saved(out:Path)->dict:
    v4.load_checkpoint("preflight",out);fc=v4.load_checkpoint("forecast",out)
    rows=_read_exact(fc["forecastArtifact"])
    if any("actual" in r or "targetObserved" in r for r in rows):raise ValueError("persisted V4 forecast contains outcome leakage")
    features=_read_exact(fc["originFeaturesTest"])
    feature_map={(r["crop"],int(r["horizon"]),v4.period_key((dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"])))):r for r in features}
    raw,series,targets,actual_map=_load_actuals()
    scored=[]
    for r in rows:
        tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
        actual,observed=targets[r["crop"]].get(tp,{}).get("actual"),targets[r["crop"]].get(tp,{}).get("observedDays",0)
        scored.append({**r,"actual":actual,"observedDays":observed})
    if len(scored)!=792:raise ValueError("test score must retain all 792 planned keys")
    for crop in proxy.CROPS:
      for h,expected in ((2,98),(3,97)):
        cell=[r for r in scored if r["crop"]==crop and int(r["horizon"])==h]
        if len(cell)!=99 or sum(bool(r["policyUnavailable"]) for r in cell)!=(99-expected):raise ValueError("test calendar/support changed")
        pkeys={r["targetKey"] for r in cell if r["forecasts"].get("P",{}).get("predicted") is not None}
        bkeys={r["targetKey"] for r in cell if r["forecasts"].get("B0",{}).get("predicted") is not None}
        if pkeys!=bkeys or len(pkeys)!=expected:raise ValueError(f"P/B0 support mismatch at {crop}/{h}")
        if any(r.get("actual") is None for r in cell if r["targetKey"] in pkeys):raise ValueError("paired V3 key lost actual")
    methods=("Q1","Q0","P","B0")
    # Frozen comparison universe is exactly the shared P/B0 eligible support.
    # Keep all 792 planned rows in artifacts and coverage, but exclude the 12
    # policy-unavailable keys from every headline, annual, diagnostic and oracle.
    eligible=[r for r in scored if not r.get("policyUnavailable") and
       r.get("forecasts",{}).get("P",{}).get("predicted") is not None and
       r.get("forecasts",{}).get("B0",{}).get("predicted") is not None]
    cells=_metrics_cells(eligible,methods);macro=_macro8(cells,methods)
    paired={f"Q1-{m}":_paired_macro(eligible,"Q1",m) for m in ("P","B0","Q0")}
    main,mech=_hypotheses(eligible)
    # The pinned helper uses seed zero and identical calendar indices for every call.
    bootstrap={}
    # Preserve the full 99-dekad calendar for block-resampling while making the
    # 12 early policy-unavailable rows unavailable to every method in bootstrap.
    bootstrap_rows=[]
    for row in scored:
        copy={**row,"forecasts":{k:dict(v) for k,v in row["forecasts"].items()}}
        if row.get("policyUnavailable"):
            for method in methods:copy["forecasts"].setdefault(method,{})["predicted"]=None
        bootstrap_rows.append(copy)
    for a,b in (("Q1","P"),("Q1","B0"),("Q1","Q0")):
        bootstrap[f"{a}-{b}"]={str(n):stats.bootstrap(bootstrap_rows,(a,b),n,reps=2000) for n in (6,12)}
    coverage=_coverage(scored,eligible)
    features_by_key={(r["crop"],int(r["horizon"]),v4.period_key((dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"])) )):r for r in features}
    yearly=_yearly_metrics(eligible,methods)
    diagnostics=_diagnostic_bins(eligible,features_by_key)
    oracle=_oracle(eligible,features_by_key)
    goal={"thresholdMAPE8":10.0,"mape8":macro["Q1"]["mape"],"met":macro["Q1"]["mape"] is not None and macro["Q1"]["mape"]<=10,
      "gapPP":max(0.,macro["Q1"]["mape"]-10) if macro["Q1"]["mape"] is not None else None,
      "cellsAtOrBelow10":sum(cells[c][str(h)]["Q1"]["n"]==(98 if h==2 else 97) and cells[c][str(h)]["Q1"]["mape"] is not None and cells[c][str(h)]["Q1"]["mape"]<=10 for c in proxy.CROPS for h in proxy.HORIZONS),
      "allCellsFullIntendedCoverage":all(cells[c][str(h)]["Q1"]["n"]==(98 if h==2 else 97) for c in proxy.CROPS for h in proxy.HORIZONS),
      "cellCount":8,"cellExcessPP":{f"{c}:{h}":max(0.,cells[c][str(h)]["Q1"]["mape"]-10) if cells[c][str(h)]["Q1"]["mape"] is not None else None for c in proxy.CROPS for h in proxy.HORIZONS}}
    goal["met"]=goal["met"] and goal["allCellsFullIntendedCoverage"]
    support={"plannedRows":len(scored),"pairedPAndB0":sum(cells[c][str(h)]["P"]["n"] for c in proxy.CROPS for h in proxy.HORIZONS),
      "uniqueTargets":len({r["targetKey"] for r in scored}),"policyUnavailableRows":sum(bool(r["policyUnavailable"]) for r in scored),
      "sameCropCrossHorizonOverlap":{c:len({r["targetKey"] for r in scored if r["crop"]==c and int(r["horizon"])==2}&
          {r["targetKey"] for r in scored if r["crop"]==c and int(r["horizon"])==3}) for c in proxy.CROPS}}
    mae_no_worse=all(cells[c][str(h)]["Q1"]["mae"] is not None and cells[c][str(h)]["B0"]["mae"] is not None and
       cells[c][str(h)]["Q1"]["mae"]<=cells[c][str(h)]["B0"]["mae"] for c in proxy.CROPS for h in proxy.HORIZONS)
    macro_mae_no_worse=all(cells[c][str(h)]["Q1"]["mae"] is not None and cells[c][str(h)]["B0"]["mae"] is not None and
       cells[c][str(h)]["Q1"]["n"]==(98 if h==2 else 97) for c in proxy.CROPS for h in proxy.HORIZONS) and macro["Q1"]["mae"]<=macro["B0"]["mae"]
    conclusions=_stat_conclusions(paired,bootstrap,main)
    # Separate multiplicity family for Q1/Q0 mechanism; never borrow main-family p-values.
    conclusions["Q1-Q0"]["holmSignificantMechanismAPECellsHAC3"]=sum(x["comparison"]=="Q1-Q0" and x["loss"]=="ape" and x["lag"]==3 and x.get("holmSignificant") for x in mech)
    # Development score uses the same frozen metrics after the development artifact
    # has been persisted; the retrospective B0 selection assumption is explicit.
    devcp=v4.load_checkpoint("develop",out)
    devmetrics=None
    if "forecastArtifact" in devcp:
        devrows=_read_exact(devcp["forecastArtifact"])
        dev_scored=[]
        for r in devrows:
            tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
            dev_scored.append({**r,"actual":targets[r["crop"]].get(tp,{}).get("actual")})
        devcells=_metrics_cells(dev_scored,methods)
        devfeatures=_read_exact(devcp["originFeatures"])
        dev_feature_by_key={(r["crop"],int(r["horizon"]),v4.period_key((dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"])))):r for r in devfeatures}
        devcoverage=_coverage(dev_scored,dev_scored)
        devsupport={"byCell":{}}
        for crop in proxy.CROPS:
          for h in proxy.HORIZONS:
            subset=[r for r in devrows if r["crop"]==crop and int(r["horizon"])==h]
            paths=devcoverage["byCropHorizon"][f"{crop}:{h}"]
            cell_features=[f for f in devfeatures if f["crop"]==crop and int(f["horizon"])==h]
            devsupport["byCell"][f"{crop}:{h}"]={
              "q0CandidatesMean":float(np.mean([int(f.get("seasonPoolQ0",0)) for f in cell_features])) if cell_features else None,
              "q1CandidatesMean":float(np.mean([int(f.get("seasonPoolQ1",0)) for f in cell_features])) if cell_features else None,
              "q0CandidateOriginCoverage":sum(int(f.get("seasonPoolQ0",0))>=v4.K for f in cell_features),
              "featureValidOrigins":sum(all(f.get("features",{}).get(name) is not None and math.isfinite(float(f["features"][name])) for name in v4.FEATURES) for f in cell_features),
              "q1Direct":paths["Q1"].get("direct",0),"q1FallbackQ0":paths["Q1"].get("fallbackQ0",0),
              "q1FallbackB0":paths["Q1"].get("fallbackB0",0),"q1Missing":paths["Q1"].get("missing",0)}
        devmetrics={"metricsByCell":devcells,"metrics8EqualCell":_macro8(devcells,methods),
          "annualMetrics":_yearly_metrics(dev_scored,methods),"supportAndFallback":devsupport,
          "forecastRows":len(devrows),"b0RetrospectiveSelectionAssumption":devcp.get("developmentB0RetrospectiveSelectionAssumption")}
    report={"runId":v4.RUN_ID,"protocolSha256":v4.PROTOCOL_SHA,"analysisCodeSha256":v4.code_hash(),
      "dependencyIdentity":v4.run_identity()["dependencies"],"forecastArtifact":v4.artifact_identity(fc["forecastArtifact"]),
      "preflightEvidence":{"oldReportSha256":v4.OLD_REPORT_SHA,
        "v3TestForecastSha256":v4.V3_PINNED_FILES["forecasts_test.parquet"],"sourceSha256":v4.SOURCE_SHA,
        "preflightCheckpointSha256":v4.digest(out/"preflight.json"),
        "replayCheckpointSha256":v4.digest(out/"replay.json"),"developCheckpointSha256":v4.digest(out/"develop.json")},
      "metricsByCell":cells,"metrics8EqualCell":macro,"pairedLossDifferences":paired,
      "coverage":coverage,"support":{**support,"commonComparisonRows":len(eligible),"eligibleKeyPolicy":"the same 780 pinned V3 P/B0 eligible keys; 12 early policy-unavailable rows retained unscored"},
      "MAPE10Goal":goal,"maeNoWorseThanB0":mae_no_worse,"macroMaeNoWorseThanB0":macro_mae_no_worse,
      "developmentMetrics":devmetrics,
      "annualMetrics":yearly,"diagnosticBins":diagnostics,"oracle":oracle,
      "hypothesisCounts":{"mainPerLag":32,"mechanismPerLag":16},"main32":_combine_bandwidths(main),
      "mechanism16":_combine_bandwidths(mech),"jointTargetDekadBlockBootstrap":bootstrap,
      "conclusions":conclusions,"posthocOnlyDiagnostics":True,
      "evaluationCaveat":"retrospective, repeatedly inspected revised proxy; not prospective or official competition results"}
    (out/"score_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"result.md").write_text(render_korean(report),encoding="utf-8")
    return v4.save_checkpoint("score",{"runId":v4.RUN_ID,"stage":"score","complete":True,
      "reportSha256":v4.digest(out/"score_report.json"),"resultSha256":v4.digest(out/"result.md"),
      "outcomeStatus":"scored_retrospective_proxy_only"},out)
