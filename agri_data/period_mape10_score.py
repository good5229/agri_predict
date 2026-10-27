"""Frozen scoring for saved PERIOD_DIRECT_MAPE_ROLLING_V3 forecasts."""
from __future__ import annotations
import datetime as dt, json, math
from pathlib import Path
from statistics import mean
import numpy as np
from . import period_mape10 as v3, period_proxy_experiment as proxy

MAIN=(("P","F"),("P","B0"))
MECH=(("W","R"),("S","R"),("M","S"),("MW","M"))

def metrics(pairs):
    if not pairs: return {"n":0,"mae":None,"mape":None}
    return {"n":len(pairs),"mae":mean(abs(a-p) for a,p in pairs),"mape":mean(abs(a-p)/a for a,p in pairs)*100}

def read_rows(meta):
    """Read the exact artifact named and hashed by the test checkpoint."""
    fmt=meta.get("format")
    if fmt not in ("parquet","jsonl_fallback_no_parquet_engine"):
        raise ValueError(f"unsupported forecast artifact format: {fmt}")
    p=Path(meta["path"])
    if not p.is_file() or v3.digest(p)!=meta.get("sha256"):
        raise ValueError("test forecast artifact path/hash differs from verified checkpoint")
    if fmt=="parquet":
        if p.suffix!=".parquet": raise ValueError("checkpoint format/path mismatch")
        import pandas as pd
        return pd.read_parquet(p).to_dict("records"),p
    if p.suffix!=".jsonl": raise ValueError("checkpoint format/path mismatch")
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x],p

def paired(rows,left,right):
    pairs=[]
    for r in rows:
        f=r["forecasts"]
        if r.get("actual") is None or not f.get(left) or not f.get(right): continue
        if f[left].get("predicted") is None or f[right].get("predicted") is None: continue
        tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
        pairs.append((tp,float(r["actual"]),float(f[left]["predicted"]),float(f[right]["predicted"])))
    return pairs

def hac(pairs,loss,lag):
    n=len(pairs); unique=len({x[0] for x in pairs})
    if unique<30:return {"status":"insufficient_distinct_targets","n":n,"distinctTargets":unique}
    def delta(x):
        _,a,p,q=x
        return abs(p-a)-abs(q-a) if loss=="ae" else abs(p-a)/a-abs(q-a)/a
    vals={proxy._period_index(x[0]):delta(x) for x in pairs}; d=float(np.mean(list(vals.values())))
    z={i:v-d for i,v in vals.items()}; var=sum(v*v for v in z.values())/n
    for k in range(1,lag+1):
        cov=sum(z[i]*z.get(i-k,0.) for i in z if i-k in z)/n
        var+=2*(1-k/(lag+1))*cov
    if not math.isfinite(var) or var<=0:return {"status":"nonpositive_hac_variance","n":n,"meanLossDifference":d}
    stat=d/math.sqrt(var/n)
    return {"status":"tested","n":n,"distinctTargets":unique,"meanLossDifference":d,"hacVariance":var,
            "statistic":stat,"p":math.erfc(abs(stat)/math.sqrt(2))}

def holm(xs):
    valid=sorted((x["p"],i) for i,x in enumerate(xs) if x.get("status")=="tested")
    adj=[None]*len(xs); run=0.
    for rank,(p,i) in enumerate(valid):
        run=max(run,min(1.,(len(xs)-rank)*p)); adj[i]=run
    return [{**x,"holmP":adj[i],"holmSignificant":adj[i] is not None and adj[i]<.05} for i,x in enumerate(xs)]

def bootstrap(rows,comp,length,reps=2000):
    cells=[(c,h) for c in proxy.CROPS for h in proxy.HORIZONS]
    inds=[proxy._period_index((dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))) for r in rows]
    if not inds:return {"status":"no_calendar"}
    cal=np.arange(min(inds),max(inds)+1); rng=np.random.default_rng(0)
    deltas={}
    for cell in cells:
        rs=[r for r in rows if (r["crop"],int(r["horizon"]))==cell]
        deltas[cell]={proxy._period_index(x[0]):(abs(x[2]-x[1])-abs(x[3]-x[1]),
            100*(abs(x[2]-x[1])/x[1]-abs(x[3]-x[1])/x[1])) for x in paired(rs,*comp)}
    vals={"mae8":[],"mape8":[]}
    for _ in range(reps):
        nblocks=math.ceil(len(cal)/length); starts=rng.integers(0,len(cal),nblocks)
        sample=np.concatenate([(s+np.arange(length))%len(cal) for s in starts])[:len(cal)]
        means={"mae8":[],"mape8":[]}
        for cell in cells:
            found=[deltas[cell][cal[i]] for i in sample if cal[i] in deltas[cell]]
            if found:
                means["mae8"].append(float(np.mean([x[0] for x in found])))
                means["mape8"].append(float(np.mean([x[1] for x in found])))
        for k in vals: vals[k].append(float(np.mean(means[k])) if means[k] else float("nan"))
    out={"status":"tested","blockLengthDekads":length,"seed":0,"replicates":reps,
         "sameIndicesAcrossCells":True,"calendarGapsRetained":True}
    for k,a in vals.items():
        a=np.asarray(a);a=a[np.isfinite(a)]
        out[k]={"meanDelta":float(a.mean()) if len(a) else None,
                "ci95":[float(np.quantile(a,.025)),float(np.quantile(a,.975))] if len(a) else None,
                "validReplicates":len(a)}
    return out

def all_metrics(rows,methods):
    out={}
    for c in proxy.CROPS:
        out[c]={}
        for h in proxy.HORIZONS:
            cell=[r for r in rows if r["crop"]==c and int(r["horizon"])==h]
            out[c][str(h)]={m:metrics([(r["actual"],r["forecasts"][m]["predicted"]) for r in cell
                if r.get("actual") is not None and r["forecasts"].get(m) and r["forecasts"][m].get("predicted") is not None])
                for m in methods}
    return out

def frequency_metrics(rows,methods,monthly):
    out={}
    key=lambda r:r["targetStart"][:7] if monthly else r["targetStart"][:4]
    for k in sorted({key(r) for r in rows}):out[k]=all_metrics([r for r in rows if key(r)==k],methods)
    return out

def frequency_metrics8(freq,methods):
    out={}
    for period,crops in freq.items():
      out[period]={}
      for method in methods:
        maes=[crops[c][str(h)][method]['mae'] for c in proxy.CROPS for h in proxy.HORIZONS
              if crops[c][str(h)][method]['mae'] is not None]
        mapes=[crops[c][str(h)][method]['mape'] for c in proxy.CROPS for h in proxy.HORIZONS
              if crops[c][str(h)][method]['mape'] is not None]
        out[period][method]={'mae8':float(np.mean(maes)) if maes else None,'mape8':float(np.mean(mapes)) if mapes else None,
                             'supportedCells':len(maes)}
    return out

def crop_contribution(cells,methods):
    out={}
    for c in proxy.CROPS:
      out[c]={}
      for m in methods:
        mae=[cells[c][str(h)][m]['mae'] for h in proxy.HORIZONS if cells[c][str(h)][m]['mae'] is not None]
        mape=[cells[c][str(h)][m]['mape'] for h in proxy.HORIZONS if cells[c][str(h)][m]['mape'] is not None]
        out[c][m]={'maeContributionToMAE8':float(sum(mae)/8),'mapeContributionToMAPE8':float(sum(mape)/8),
                   'maeScaleShare':None}
    for m in methods:
      total=sum(out[c][m]['maeContributionToMAE8'] for c in proxy.CROPS)
      for c in proxy.CROPS:out[c][m]['maeScaleShare']=out[c][m]['maeContributionToMAE8']/total if total else None
    return out

def mechanism_decomposition(rows):
    result={}
    comps=(('R','F'),)+MECH
    for granularity in ('year','month'):
      keys=sorted({r['targetStart'][:4] if granularity=='year' else r['targetStart'][:7] for r in rows})
      result[granularity]={}
      for k in keys:
        by_comp={}
        subset=[r for r in rows if (r['targetStart'][:4] if granularity=='year' else r['targetStart'][:7])==k]
        for a,b in comps:
          cell_deltas=[];cell_pairs={}
          for c in proxy.CROPS:
            for h in proxy.HORIZONS:
              rs=[r for r in subset if r['crop']==c and int(r['horizon'])==h]
              ps=paired(rs,a,b)
              if not ps:continue
              lm=metrics([(x[1],x[2]) for x in ps]);rm=metrics([(x[1],x[3]) for x in ps])
              cell_pairs[f'{c}:{h}']={'n':len(ps),'left':lm,'right':rm,
                'deltaMAE':lm['mae']-rm['mae'],'deltaMAPE':lm['mape']-rm['mape']}
              cell_deltas.append((lm['mae']-rm['mae'],lm['mape']-rm['mape']))
          by_comp[f'{a}-{b}']={'cells':cell_pairs,
            'equalCellDeltaMAE':float(np.mean([x[0] for x in cell_deltas])) if cell_deltas else None,
            'equalCellDeltaMAPE':float(np.mean([x[1] for x in cell_deltas])) if cell_deltas else None}
        result[granularity][k]=by_comp
    return result

def support_summary(rows):
    out={}
    for c in proxy.CROPS:
      out[c]={}
      for h in proxy.HORIZONS:
        rs=[r for r in rows if r['crop']==c and int(r['horizon'])==h]
        out[c][str(h)]={'plannedForecastOrigins':len(rs),
          'uniqueTargetDekads':len({r['targetKey'] for r in rs}),
          'positiveTargetCount':sum(r.get('actual') is not None for r in rs),
          'policyUnavailableOrigins':sum(r.get('policyUnavailable',False) for r in rs),
          'seasonalSupport':{str(q):sum(dt.date.fromisoformat(r['targetStart']).month in range(3*q-2,3*q+1) for r in rs) for q in (1,2,3,4)}}
    overlap={}
    for c in proxy.CROPS:
      a={r['targetKey'] for r in rows if r['crop']==c and int(r['horizon'])==2 and r.get('actual') is not None}
      b={r['targetKey'] for r in rows if r['crop']==c and int(r['horizon'])==3 and r.get('actual') is not None}
      overlap[c]={'commonTargetDekads':len(a&b),'secondOnly':len(a-b),'thirdOnly':len(b-a)}
    return {'cells':out,'sameCropCrossHorizonTargetOverlap':overlap}

def combine_bandwidth_tests(items):
    combined={}
    for x in items:
        key=(x['crop'],int(x['horizon']),x['comparison'],x['loss'])
        row=combined.setdefault(key,{'crop':key[0],'horizon':key[1],'comparison':key[2],'loss':key[3]})
        row[f"hac{x['lag']}"]={k:v for k,v in x.items() if k not in ('crop','horizon','comparison','loss','lag')}
    return [combined[k] for k in sorted(combined)]

def diagnostics(rows,raw):
    series=proxy.series_from_raw(raw); targets={c:proxy._target_periods(s) for c,s in series.items()}
    med={}
    for c in proxy.CROPS:
        vals=[v["actual"] for tp,v in targets[c].items() if v.get("actual") is not None and 2014<=tp[0].year<=2021]
        med[c]=float(np.median(vals)) if vals else None
    bins={}
    for r in rows:
        c=r["crop"];tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
        actual=targets[c].get(tp,{}).get("actual"); f=r["forecasts"]
        if actual is None or not f.get("P") or "referencePrice" not in f["P"]:continue
        ref=f["P"]["referencePrice"]; days=sorted(d for d in series[c] if d<=dt.date.fromisoformat(r["inputCutoff"]))
        age=(dt.date.fromisoformat(r["inputCutoff"])-days[-1]).days if days else 999
        obs=targets[c].get(tp,{}).get("observedDays",0)
        groups={"priceLevel":("low" if actual<=med[c] else "high") if med[c] is not None else "historicalMedianUnavailable",
          "jump":"jump_ge20pct" if abs(actual/ref-1)>=.2 else "not_jump_ge20pct",
          "referenceFreshness":"le2" if age<=2 else "3to7" if age<=7 else "ge8",
          "observedDays":"3to5" if 3<=obs<=5 else "6to8" if 6<=obs<=8 else "ge9" if obs>=9 else "lt3"}
        for dim,val in groups.items():
            d=bins.setdefault(f"{c}:{dim}:{val}",{})
            for m,fc in f.items():
                if fc and "predicted" in fc:d.setdefault(m,[]).append((actual,fc["predicted"]))
    return {"historical2014_2021MedianByCrop":med,"bins":{k:{m:metrics(x) for m,x in v.items()} for k,v in bins.items()},
            "posthocActualOnly":True}

def score_saved(out:Path):
    v3._verified_inputs(out)
    test_checkpoint=v3._verify_stage_checkpoint("test",out)
    rows,path=read_rows(test_checkpoint["forecastArtifact"])
    if any("actual" in x for x in rows):raise ValueError("forecast artifact contains actual values")
    _,raw=proxy.validate_inputs(); series=proxy.series_from_raw(raw)
    targets={c:proxy._target_periods(s) for c,s in series.items()}
    scored=[]
    for r in rows:
        tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
        target=targets[r["crop"]].get(tp,{})
        scored.append({**r,"actual":target.get("actual"),"observedDays":target.get("observedDays",0)})
    methods=sorted({m for r in scored for m in r["forecasts"]});cells=all_metrics(scored,methods)
    # Refuse partial or unpaired test artifacts; counts are policy-specific forecast support.
    for c in proxy.CROPS:
      for h,expected in ((2,98),(3,97)):
        cell=[r for r in scored if r['crop']==c and int(r['horizon'])==h]
        fkeys={r['targetKey'] for r in cell if r['forecasts'].get('F')}
        pkeys={r['targetKey'] for r in cell if r['forecasts'].get('P')}
        if len(cell)!=99 or len(fkeys)!=expected or pkeys!=fkeys:
            raise ValueError(f"incomplete/unpaired test artifact for {c}/{h}: planned={len(cell)}, F={len(fkeys)}, P={len(pkeys)}")
    metrics8={m:{"mae":float(np.mean([cells[c][str(h)][m]["mae"] for c in proxy.CROPS for h in proxy.HORIZONS if cells[c][str(h)][m]["mae"] is not None])),
      "mape":float(np.mean([cells[c][str(h)][m]["mape"] for c in proxy.CROPS for h in proxy.HORIZONS if cells[c][str(h)][m]["mape"] is not None]))} for m in methods}
    main=[];mech=[]
    for c in proxy.CROPS:
      for h in proxy.HORIZONS:
        rs=[r for r in scored if r["crop"]==c and int(r["horizon"])==h]
        for comps,dest in ((MAIN,main),(MECH,mech)):
          for a,b in comps:
            ps=paired(rs,a,b)
            for loss in ("ae","ape"):
              for lag in proxy.HAC_LAGS:dest.append({"crop":c,"horizon":h,"comparison":f"{a}-{b}","loss":loss,"lag":lag,**hac(ps,loss,lag)})
    if len(main)!=64 or len(mech)!=128:raise ValueError("frozen statistical family cardinality mismatch")
    for family in (main,mech):
      for lag in proxy.HAC_LAGS:
        ii=[i for i,x in enumerate(family) if x["lag"]==lag]
        for i,x in zip(ii,holm([family[j] for j in ii])):family[i]=x
    boot={f"{a}-{b}":{str(n):bootstrap(scored,(a,b),n) for n in (6,12)} for a,b in MAIN}
    paired_metrics={}
    for a,b in MAIN:
      cells_pair={};lefts=[];rights=[]
      for c in proxy.CROPS:
       for h in proxy.HORIZONS:
        ps=paired([r for r in scored if r['crop']==c and int(r['horizon'])==h],a,b)
        if ps:
         lm=metrics([(x[1],x[2]) for x in ps]);rm=metrics([(x[1],x[3]) for x in ps])
         cells_pair[f'{c}:{h}']={'n':len(ps),'left':lm,'right':rm};lefts.append(lm);rights.append(rm)
      paired_metrics[f'{a}-{b}']={'cells':cells_pair,'n':sum(x['n'] for x in cells_pair.values()),
         'leftEqualCell':{'mae':float(np.mean([x['mae'] for x in lefts])),'mape':float(np.mean([x['mape'] for x in lefts]))} if lefts else None,
         'rightEqualCell':{'mae':float(np.mean([x['mae'] for x in rights])),'mape':float(np.mean([x['mape'] for x in rights]))} if rights else None}
    p=metrics8.get("P",{});f=metrics8.get("F",{});b0=metrics8.get("B0",{})
    goal={"MAPE8":p.get("mape"),"meetsAverage":p.get("mape") is not None and p["mape"]<=10,
      "cellsAtOrBelow10":sum(cells[c][str(h)].get("P",{}).get("mape") is not None and cells[c][str(h)]["P"]["mape"]<=10 for c in proxy.CROPS for h in proxy.HORIZONS),
      "cells":{f"{c}:{h}":{"mape":cells[c][str(h)].get("P",{}).get("mape"),
       "gapTo10pp":max(0,cells[c][str(h)]["P"]["mape"]-10) if cells[c][str(h)].get("P",{}).get("mape") is not None else None}
       for c in proxy.CROPS for h in proxy.HORIZONS}}
    coverage={m:{"planned":len(scored),"forecasted":sum(bool(r["forecasts"].get(m)) for r in scored),
      "scored":sum(r.get("actual") is not None and bool(r["forecasts"].get(m)) for r in scored)} for m in methods}
    clipping={m:{"forecastCount":sum(bool(r['forecasts'].get(m)) for r in scored),
        "clippedCount":sum(bool(r['forecasts'].get(m)) and bool(r['forecasts'][m].get('clippedAtFloor',False)) for r in scored),
        "clippedRate":(sum(bool(r['forecasts'].get(m)) and bool(r['forecasts'][m].get('clippedAtFloor',False)) for r in scored)/
            sum(bool(r['forecasts'].get(m)) for r in scored)) if any(r['forecasts'].get(m) for r in scored) else None}
        for m in methods}
    pf=paired_metrics.get('P-F',{}); pl=pf.get('leftEqualCell') or {};pr=pf.get('rightEqualCell') or {}
    pkeys={(r['crop'],int(r['horizon']),r['targetKey']) for r in scored if r.get('actual') is not None and r['forecasts'].get('P')}
    fkeys={(r['crop'],int(r['horizon']),r['targetKey']) for r in scored if r.get('actual') is not None and r['forecasts'].get('F')}
    coverage_maintained=fkeys.issubset(pkeys)
    gain=(pl.get("mape") is not None and pr.get("mape") is not None and pl["mape"]<pr["mape"] and
          pl.get("mae") is not None and pr.get("mae") is not None and pl["mae"]<=pr["mae"] and coverage_maintained)
    status=("completed_mape10_and_price_gain" if goal["meetsAverage"] and gain else
      "completed_price_gain_below_goal" if gain else
      "completed_mape10_with_mae_tradeoff" if goal["meetsAverage"] and pl.get('mae',math.inf)>pr.get('mae',-math.inf) else "completed_goal_not_met")
    annual=frequency_metrics(scored,methods,False);monthly=frequency_metrics(scored,methods,True)
    report={"runId":v3.RUN_ID,"protocolSha256":v3.PROTOCOL_SHA,"sourceSha256":proxy.SOURCE_SHA,
      "forecastArtifact":{"path":str(path),"sha256":v3.digest(path)},"evaluation":"retrospective revised proxy; not official results",
      "metricsByCell":cells,"metrics8EqualCell":metrics8,"coverage":coverage,"coveragePIncludesFKeys":coverage_maintained,"pairedMetrics":paired_metrics,
      "predictionFloorClipping":clipping,
      "cropContributions":crop_contribution(cells,methods),"annualMetrics8":frequency_metrics8(annual,methods),
      "monthlyMetrics8":frequency_metrics8(monthly,methods),
      "hypothesisCounts":{"mainPerHACBandwidth":32,"mechanismPerHACBandwidth":64},
      "MAPE10":goal,"observedOutcomeStatus":status,"main32":combine_bandwidth_tests(main),"mechanism64":combine_bandwidth_tests(mech),
      "jointTargetDekadBlockBootstrap":boot,"annual":annual,
      "monthly":monthly,"mechanismByMonthYear":mechanism_decomposition(scored),
      "supportSummary":support_summary(scored),"diagnostics":diagnostics(scored,raw)}
    (out/"score_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    md=[f"# {status}","","Observed retrospective proxy outcomes only; not prospective or official competition performance.",
      f"MAPE8 {p.get('mape')}%; MAE8 {p.get('mae')} 원/kg. F {f.get('mape')}% / {f.get('mae')} 원/kg. B0 {b0.get('mape')}% / {b0.get('mae')} 원/kg.",
      f"At or below 10%: {goal['cellsAtOrBelow10']}/8 cells.","","| Crop | Horizon | Method | n | MAE | MAPE |","|---|---:|---|---:|---:|---:|"]
    for c in proxy.CROPS:
      for h in proxy.HORIZONS:
       for m in ("P","F","B0","R","W","S","M","MW"):
        x=cells[c][str(h)].get(m,{})
        md.append(f"| {c} | {h} | {m} | {x.get('n')} | {x.get('mae')} | {x.get('mape')}% |")
    (out/"result.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    v3.save_checkpoint("score",{"runId":v3.RUN_ID,"stage":"score","complete":True,
      "reportSha256":v3.digest(out/"score_report.json"),"resultSha256":v3.digest(out/"result.md"),"status":status},out)
    return report
