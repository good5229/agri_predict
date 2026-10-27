"""Pure, fixed-protocol scorer for E106-FHS forecast manifests."""
from __future__ import annotations
import datetime as dt
import math
import random
from collections import defaultdict
from typing import Any, Mapping
from . import equal_info_online_score as inference

PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
CONDITIONS = ("B", "C", "F", "A")
COMPARISONS = (("F_vs_C", "F", "C"), ("F_vs_A", "F", "A"), ("F_vs_B", "F", "B"))
WINDOWS = {"V": (dt.date(2026,4,1),dt.date(2026,6,20)), "T": (dt.date(2026,7,11),dt.date(2026,9,27))}
ORIGIN_CUTOFF = {"V":dt.date(2026,4,1), "T":dt.date(2026,7,1)}
BANDWIDTHS=(9,28); BLOCKS=(14,28); REPS=2000; SEED=0

def _day(v):
    if isinstance(v,dt.datetime): return v.date()
    if isinstance(v,dt.date): return v
    if isinstance(v,str): return dt.date.fromisoformat(v)
    raise ValueError("invalid date")

def _num(v,positive=False):
    try: x=float(v)
    except (TypeError,ValueError,OverflowError): return None
    return x if math.isfinite(x) and (not positive or x>0) else None

def _close(x,y):
    a,b=_num(x),_num(y)
    return (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-6+1e-8*abs(b))

def _key(r):
    p=r.get("product"); o=_day(r["originDate"]); t=_day(r["targetDate"]); h=r.get("horizon")
    if p not in PRODUCTS or type(h) is not int or h not in HORIZONS or t-o!=dt.timedelta(days=h): raise ValueError("invalid forecast identity")
    return p,o,t,h

def _mapkey(k):
    if not isinstance(k,(tuple,list)) or len(k)!=4: raise ValueError("invalid baseline key")
    p,o,t,h=k
    if p not in PRODUCTS or type(h) is not int or h not in HORIZONS: raise ValueError("invalid baseline key")
    return p,_day(o),_day(t),h

def _months(phase):
    first,last=WINDOWS[phase]; d=first.replace(day=1); out=[]
    while d<=last:
        out.append(d.strftime("%Y-%m")); d=(d.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
    return out

def _metric(rows,c):
    pairs=[(r,r["conditions"][c]) for r in rows if c in r["issued"]]
    if not pairs: return {"n":0,"uniqueTargetDates":0,"uniqueOrigins":0,"mae":None,"mapePct":None,"directMae":None,"directMapePct":None,"directLogMse":None}
    n=len(pairs); ae=[abs(v["blendPrediction"]-r["actual"]) for r,v in pairs]
    dae=[abs(v["directPrediction"]-r["actual"]) for r,v in pairs]
    le=[v["directLogPrediction"]-r["actualLogReturn"] for r,v in pairs if r["actualLogReturn"] is not None]
    return {"n":n,"uniqueTargetDates":len({r["target"] for r,v in pairs}),"uniqueOrigins":len({r["origin"] for r,v in pairs}),
            "mae":math.fsum(ae)/n,"mapePct":100*math.fsum(a/r["actual"] for a,(r,v) in zip(ae,pairs))/n,
            "directMae":math.fsum(dae)/n,"directMapePct":100*math.fsum(a/r["actual"] for a,(r,v) in zip(dae,pairs))/n,
            "directLogMse":math.fsum(x*x for x in le)/len(le) if le else None}

def _diff(r,l,rr,loss):
    d=abs(r["conditions"][l]["blendPrediction"]-r["actual"])-abs(r["conditions"][rr]["blendPrediction"]-r["actual"])
    return d if loss=="ae" else 100*d/r["actual"]

def _infer(rows):
    out=[]
    for label,l,rr in COMPARISONS:
        for h in HORIZONS:
            cell=[r for r in rows if r["horizon"]==h and l in r["issued"] and rr in r["issued"]]
            for loss in ("ae","ape"):
                bd=defaultdict(list); same=bool(cell)
                for r in cell:
                    bd[r["target"]].append(_diff(r,l,rr,loss)); same &= r["conditions"][l]["blendPrediction"]==r["conditions"][rr]["blendPrediction"]
                diffs={d:math.fsum(v)/len(v) for d,v in bd.items()}
                support_status=("a_unavailable" if rr=="A" and not cell else
                                "forecast_unavailable" if not cell else
                                "supported" if len(diffs)>=30 else "insufficient_pairs")
                for bw in BANDWIDTHS:
                    if same and len(diffs)>=30:
                        z={"status":"identical_predictions","reason":"same blend prediction on every paired key","n":len(diffs),"statistic":None,"meanLossDifference":math.fsum(diffs.values())/len(diffs),"pValue":1.0}
                    else: z=inference._dm(diffs,bw)
                    out.append({"comparison":label,"horizon":h,"loss":loss,"unit":"price_per_kg" if loss=="ae" else "percentage_points","bandwidth":bw,"samePrediction":same,"supportStatus":support_status,**z})
    if len(out)!=120: raise ValueError("fixed 60-hypothesis family required per bandwidth")
    summaries=[]
    for bw in BANDWIDTHS:
        fam=[x for x in out if x["bandwidth"]==bw]
        inference._holm(fam)
        summaries.append({"bandwidth":bw,"familySize":60,"tested":sum(x["status"]=="tested" for x in fam),"identical":sum(x["status"]=="identical_predictions" for x in fam),"unavailable":sum(x["status"] not in ("tested","identical_predictions") for x in fam),"holmSignificantImprovement":sum(bool(x.get("holmSignificant")) and x.get("meanLossDifference",0)<0 for x in fam),"holmSignificantWorsening":sum(bool(x.get("holmSignificant")) and x.get("meanLossDifference",0)>0 for x in fam)})
    return out,summaries

def _bootstrap(rows,phase,block):
    first,last=WINDOWS[phase]; cal=[first+dt.timedelta(days=i) for i in range((last-first).days+1)]
    series={}
    for label,l,rr in COMPARISONS:
        for h in HORIZONS:
            cell=[r for r in rows if r["product"]=="radish" and r["horizon"]==h and l in r["issued"] and rr in r["issued"]]
            for loss in ("ae","ape"):
                d=defaultdict(list)
                for r in cell: d[r["target"]].append(_diff(r,l,rr,loss))
                series[label,h,loss]={x:math.fsum(v)/len(v) for x,v in d.items()}
    rng=random.Random(SEED); n=len(cal); draws=[]
    for _ in range(REPS):
        ix=[]
        while len(ix)<n:
            s=rng.randrange(n); ix.extend((s+j)%n for j in range(block))
        draws.append([cal[i] for i in ix[:n]])
    def est(draw,label,loss):
        hs=[]
        for h in HORIZONS:
            d=series[label,h,loss]; v=[d[x] for x in draw if x in d]
            if not v:return None
            hs.append(math.fsum(v)/len(v))
        return math.fsum(hs)/10
    result={}
    for label,_,_ in COMPARISONS:
        for loss in ("ae","ape"):
            point=est(cal,label,loss); vals=sorted(v for dr in draws if (v:=est(dr,label,loss)) is not None and math.isfinite(v))
            result[f"{label}/{loss}"]={"estimate":point,"replicatesRequested":REPS,"replicatesValid":len(vals),"ci95":[vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]] if vals else None}
    import hashlib
    dh=hashlib.sha256(repr([[x.isoformat() for x in d] for d in draws]).encode()).hexdigest()
    return {"blockDays":block,"seed":SEED,"replicatesRequested":REPS,"drawTableSha256":dh,"results":result,"scope":"shared circular calendar draws across comparisons, horizons, and missingness masks"}

def _training(fits):
    if not isinstance(fits,list): raise ValueError("fit_diagnostics must be a list")
    seen=set(); out=[]; grouped=defaultdict(dict)
    for f in fits:
        required={"fitId","fitOrigin","horizon","trainingCount","trainingRows"}
        if not isinstance(f,dict) or not required<=set(f): raise ValueError("invalid fit diagnostic")
        ident=(f["fitId"],f["horizon"])
        if ident in seen: raise ValueError("duplicate fit diagnostic")
        seen.add(ident); tr=f["trainingRows"]; origin=_day(f["fitOrigin"])
        if type(f["horizon"]) is not int or f["horizon"] not in HORIZONS: raise ValueError("invalid training horizon")
        if f["horizon"] in grouped[origin]: raise ValueError("duplicate fit origin/horizon")
        grouped[origin][f["horizon"]]=tr
        if len(tr)!=f["trainingCount"]: raise ValueError("training count mismatch")
        for r in tr:
            if not {"targetDate","actualLogReturn","B","C","F"}<=set(r): raise ValueError("training row missing paired output")
            if any(_num(r[x]) is None for x in ("actualLogReturn","B","C","F")): raise ValueError("nonfinite training output")
        item={k:v for k,v in f.items() if k!="trainingRows"}; monthly=defaultdict(list)
        item["recomputedTrainingRms"]={c:math.sqrt(math.fsum((r[c]-r["B"])**2 for r in tr)/len(tr)) for c in ("C","F","A") if tr and all(_num(r.get(c)) is not None for r in tr)}
        for r in tr: monthly[_day(r["targetDate"]).strftime("%Y-%m")].append(r)
        item["trainingRowCountByTargetMonth"]={m:len(rs) for m,rs in sorted(monthly.items())}
        out.append(item)
    intervention=[]; decomposition=[]
    for origin,by_h in sorted(grouped.items()):
        complete=len(by_h)==10 and set(by_h)==set(HORIZONS)
        entry={"fitOrigin":origin.isoformat(),"horizonsIncluded":len(by_h),"complete10HorizonSupport":complete}
        for label,left,right in (("F_B","F","B"),("C_B","C","B"),("A_B","A","B")):
            moment=[]
            for h in HORIZONS:
                tr=by_h.get(h,[])
                if not tr or any(_num(r.get(left)) is None or _num(r.get(right)) is None for r in tr): moment.append(None); continue
                ds=[r[left]-r[right] for r in tr]
                moment.append(math.fsum(x*x for x in ds)/len(ds))
            entry.setdefault("trainingOutputRmsEqualHorizon",{})[label]=math.sqrt(math.fsum(x for x in moment if x is not None)/10) if complete and all(x is not None for x in moment) else None
        # Joint cosine uses equal-horizon mean cross-products and norms, not
        # an average of horizon cosines.
        for label,left,other in (("F_B_vs_C_B","F","C"),("F_B_vs_A_B","F","A")):
            cross=[]; normx=[]; normy=[]
            for h in HORIZONS:
                tr=by_h.get(h,[])
                if not tr or any(_num(r.get(c)) is None for r in tr for c in (left,other,"B")):
                    cross.append(None); normx.append(None); normy.append(None); continue
                x=[r[left]-r["B"] for r in tr]; y=[r[other]-r["B"] for r in tr]; n=len(tr)
                cross.append(math.fsum(a*b for a,b in zip(x,y))/n); normx.append(math.fsum(a*a for a in x)/n); normy.append(math.fsum(b*b for b in y)/n)
            dot=math.fsum(x for x in cross if x is not None)/10 if complete and all(x is not None for x in cross) else None
            nx=math.fsum(x for x in normx if x is not None)/10 if complete and all(x is not None for x in normx) else None
            ny=math.fsum(x for x in normy if x is not None)/10 if complete and all(x is not None for x in normy) else None
            entry.setdefault("trainingVectorCosineEqualHorizon",{})[label]=dot/math.sqrt(nx*ny) if dot is not None and nx and ny else None
        intervention.append(entry)
        month_names=sorted({"all"} | {_day(r["targetDate"]).strftime("%Y-%m") for hrows in by_h.values() for r in hrows})
        for month in month_names:
            monthly_entry={"fitOrigin":origin.isoformat(),"targetMonth":month,"complete10HorizonSupport":complete}
            for label,left,right in (("F_vs_C","F","C"),("F_vs_A","F","A")):
                moments=[]; align=[]; delta2=[]
                for h in HORIZONS:
                    rs=[r for r in by_h.get(h,[]) if month=="all" or _day(r["targetDate"]).strftime("%Y-%m")==month]
                    if not rs or any(_num(r.get(left)) is None or _num(r.get(right)) is None for r in rs):
                        moments.append(None); align.append(None); delta2.append(None); continue
                    e=[r["actualLogReturn"]-r[right] for r in rs]; d=[r[left]-r[right] for r in rs]; n=len(rs)
                    moments.append(math.fsum((x-y)**2-x*x for x,y in zip(e,d))/n)
                    align.append(-2*math.fsum(x*y for x,y in zip(e,d))/n); delta2.append(math.fsum(x*x for x in d)/n)
                ok=complete and all(x is not None for x in moments)
                monthly_entry[label]={"n":sum(len([r for r in by_h.get(h,[]) if month=="all" or _day(r["targetDate"]).strftime("%Y-%m")==month]) for h in HORIZONS),"mseDifference":math.fsum(x for x in moments if x is not None)/10 if ok else None,"meanDeltaSquared":math.fsum(x for x in delta2 if x is not None)/10 if ok else None,"minus2ResidualDelta":math.fsum(x for x in align if x is not None)/10 if ok else None,"complete10HorizonSupport":ok}
            decomposition.append(monthly_entry)
    return {"fitCount":len(out),"fitDiagnostics":out,"interventionDiagnosticsByFitOrigin":intervention,"equalHorizonLogDecompositionByFitOriginMonth":decomposition,"scopeNote":"Monthly expanding fits and target-month summaries remain separated by fit origin; overlapping training rows are not pooled."}

def _movement_diagnostics(rows,by,phase):
    pairs={}
    all_issued={}
    for label,new,base in COMPARISONS:
        cell=[r for r in rows if new in r["issued"] and base in r["issued"]]
        pairs[label]=cell
        issued=[r for r in by.values() if r["product"]=="radish" and r["eligible"] and new in r["issued"] and base in r["issued"]]
        all_issued[label]={}
        for layer,pred in (("direct","directPrediction"),("blend","blendPrediction")):
            values=defaultdict(list)
            for r in issued:
                pnew=r["conditions"][new][pred]; pbase=r["conditions"][base][pred]
                values[r["horizon"]].append(pnew/pbase-1)
            byh={h:math.sqrt(math.fsum(x*x for x in values[h])/len(values[h])) if values[h] else None for h in HORIZONS}
            all_issued[label][layer]={"keyPooledRms":math.sqrt(math.fsum((x)**2 for vs in values.values() for x in vs)/sum(map(len,values.values()))) if sum(map(len,values.values())) else None,
                                      "fixed10HorizonRms":math.sqrt(math.fsum(v*v for v in byh.values() if v is not None)/10) if all(v is not None for v in byh.values()) else None,
                                      "n":sum(map(len,values.values())),"horizonCounts":{str(h):len(values[h]) for h in HORIZONS}}
    metric_fields=("baselineMae","conditionMae","baselineMapePct","conditionMapePct","movementMae","movementMapePct","improvementMae","improvementMapePp","lossReducingMae","lossIncreasingMae","lossReducingMapePp","lossIncreasingMapePp")
    records=[]
    for label,new,base in COMPARISONS:
        cell=pairs[label]
        hcounts={h:sum(r["horizon"]==h for r in cell) for h in HORIZONS}
        for r in cell:
            rdate=r["target"]; actual=r["actual"]
            for layer,pred in (("direct","directPrediction"),("blend","blendPrediction")):
                pn=r["conditions"][new][pred]; pb=r["conditions"][base][pred]
                movement=abs(pn-pb); improvement=abs(pb-actual)-abs(pn-actual)
                reducing=(movement+improvement)/2; increasing=(movement-improvement)/2
                if min(reducing,increasing)<-1e-9: raise ValueError("negative gain/harm movement component")
                if abs(improvement)>movement+1e-8: raise ValueError("movement/improvement identity is out of bounds")
                if not _close(reducing+increasing,movement) or not _close(reducing-increasing,improvement): raise ValueError("gain/harm components fail movement identities")
                records.append({"comparison":label,"layer":layer,"product":r["product"],"horizon":r["horizon"],"target":rdate,
                    "targetMonth":rdate.strftime("%Y-%m"),"actual":actual,"base":pb,"candidate":pn,
                    "baselineMae":abs(pb-actual),"conditionMae":abs(pn-actual),
                    "baselineMapePct":100*abs(pb-actual)/actual,"conditionMapePct":100*abs(pn-actual)/actual,
                    "movementMae":movement,"movementMapePct":100*movement/actual,
                    "improvementMae":improvement,"improvementMapePp":100*improvement/actual,
                    "lossReducingMae":reducing,"lossIncreasingMae":increasing,
                    "lossReducingMapePp":100*reducing/actual,"lossIncreasingMapePp":100*increasing/actual,
                    "relativePriceChange":pn/pb-1,"crossesActual":(pb-actual)*(pn-actual)<0,
                    "weight":1/(10*hcounts[r["horizon"]])})
    def aggregate(cell,weights):
        if not cell: return {"n":0,"changedPredictionCount":0,"crossesActualCount":0,**{f:None for f in metric_fields},"efficiencyMae":None,"efficiencyMape":None}
        out={"n":len(cell),"changedPredictionCount":sum(r["candidate"]!=r["base"] for r in cell),"crossesActualCount":sum(r["crossesActual"] for r in cell)}
        for f in metric_fields: out[f]=math.fsum(w*x[f] for w,x in zip(weights,cell))
        out["efficiencyMae"]=out["improvementMae"]/out["movementMae"] if out["movementMae"]>0 else None
        out["efficiencyMape"]=out["improvementMapePp"]/out["movementMapePct"] if out["movementMapePct"]>0 else None
        if any(v is not None and abs(v)>1+1e-8 for v in (out["efficiencyMae"],out["efficiencyMape"])): raise ValueError("movement efficiency outside [-1,1]")
        for f in ("lossReducingMae","lossIncreasingMae","lossReducingMapePp","lossIncreasingMapePp"):
            if out[f]<-1e-8: raise ValueError("negative aggregate gain/harm")
        return out
    intervals=[]; horizons=[]; months=[]; dates=[]
    labels=_months(phase)
    for label,_,_ in COMPARISONS:
        for layer in ("direct","blend"):
            cell=[r for r in records if r["comparison"]==label and r["layer"]==layer]
            cell_counts={h:sum(r["horizon"]==h for r in cell) for h in HORIZONS}
            per_h=[]
            for h in HORIZONS:
                hr=[r for r in cell if r["horizon"]==h]
                if hr: per_h.append(math.fsum(r["relativePriceChange"]**2 for r in hr)/len(hr))
            paired_rms=math.sqrt(math.fsum(per_h)/10) if len(per_h)==10 else None
            interval_global=aggregate(cell,[r["weight"] for r in cell])
            complete=all(cell_counts[h]>0 for h in HORIZONS)
            interval_entry={"comparison":label,"layer":layer,"n":len(cell),"complete10HorizonSupport":complete,
                            "availableFixedGlobalContribution":interval_global,
                            "relativePriceChangeRmsFixed10hMacro":paired_rms,"allIssued":all_issued[label][layer]}
            interval_entry.update({k:(interval_global[k] if complete else None) for k in (*metric_fields,"efficiencyMae","efficiencyMape")})
            intervals.append(interval_entry)
            for h in HORIZONS:
                part=[r for r in cell if r["horizon"]==h]
                horizons.append({"comparison":label,"layer":layer,"horizon":h,**aggregate(part,[1/len(part)]*len(part) if part else [])})
            for month in labels:
                part=[r for r in cell if r["targetMonth"]==month]
                glob=aggregate(part,[r["weight"] for r in part])
                counts={h:sum(r["horizon"]==h for r in part) for h in HORIZONS}
                if all(counts.values()):
                    local=aggregate(part,[1/(10*counts[r["horizon"]]) for r in part])
                else: local=aggregate([],[])
                months.append({"comparison":label,"layer":layer,"targetMonth":month,"horizonCounts":{str(h):counts[h] for h in HORIZONS},"hasAllTenHorizons":all(counts.values()),"fixedGlobalContribution":glob,"recomputedEqualHorizonMonthMean":local})
            for day in sorted({r["target"] for r in cell}):
                part=[r for r in cell if r["target"]==day]
                dates.append({"comparison":label,"layer":layer,"targetDate":day.isoformat(),"fixedGlobalContribution":aggregate(part,[r["weight"] for r in part])})
    return {"interval":intervals,"horizon":horizons,"targetMonth":months,"targetDate":dates}

def _log_error_report(rows,by,phase):
    result={}; month_labels=_months(phase)
    for label,l,rr in (("F_vs_C","F","C"),("F_vs_A","F","A")):
        cell=[r for r in rows if l in r["issued"] and rr in r["issued"] and r["actualLogReturn"] is not None]
        def one(part):
            if not part:return {"n":0,"mseDifference":None,"meanDeltaSquared":None,"minus2ResidualDelta":None}
            e=[r["actualLogReturn"]-r["conditions"][rr]["directLogPrediction"] for r in part]
            d=[r["conditions"][l]["directLogPrediction"]-r["conditions"][rr]["directLogPrediction"] for r in part]
            n=len(part)
            return {"n":n,"mseDifference":math.fsum((x-y)**2-x*x for x,y in zip(e,d))/n,"meanDeltaSquared":math.fsum(x*x for x in d)/n,"minus2ResidualDelta":-2*math.fsum(x*y for x,y in zip(e,d))/n}
        by_h={h:one([r for r in cell if r["horizon"]==h]) for h in HORIZONS}
        complete=all(by_h[h]["n"] for h in HORIZONS)
        overall={k:(math.fsum(by_h[h][k] for h in HORIZONS)/10 if complete else None) for k in ("mseDifference","meanDeltaSquared","minus2ResidualDelta")}
        overall.update({"n":sum(by_h[h]["n"] for h in HORIZONS),"complete10HorizonSupport":complete,"horizonsIncluded":sum(bool(by_h[h]["n"]) for h in HORIZONS)})
        monthly=[]
        for month in month_labels:
            parts={h:one([r for r in cell if r["horizon"]==h and r["target"].strftime("%Y-%m")==month]) for h in HORIZONS}
            ok=all(parts[h]["n"] for h in HORIZONS)
            monthly.append({"targetMonth":month,"comparison":label,"horizonsIncluded":sum(bool(parts[h]["n"]) for h in HORIZONS),"complete10HorizonSupport":ok,
                "n":sum(parts[h]["n"] for h in HORIZONS),**{k:(math.fsum(parts[h][k] for h in HORIZONS)/10 if ok else None) for k in ("mseDifference","meanDeltaSquared","minus2ResidualDelta")}})
        result[label]={"equalHorizonOverall":overall,"byHorizon":{str(h):by_h[h] for h in HORIZONS},"byTargetMonth":monthly}
    # Intervention summaries distinguish every issued forecast from paired
    # positive-actual keys; each has independent, explicit support.
    interventions={}; vectors={}
    for support_name,source in (("allIssued",[r for r in by.values() if r["product"]=="radish" and r["eligible"]]),("pairedActual",rows)):
        for label,l,rr in (("F_vs_B","F","B"),("C_vs_B","C","B"),("A_vs_B","A","B")):
            vals=[r for r in source if l in r["issued"] and rr in r["issued"]]
            by_h={}; moments={}
            for h in HORIZONS:
                part=[r for r in vals if r["horizon"]==h]
                d=[r["conditions"][l]["directLogPrediction"]-r["conditions"][rr]["directLogPrediction"] for r in part]
                by_h[str(h)]={"n":len(d),"rms":math.sqrt(math.fsum(x*x for x in d)/len(d)) if d else None}
                moments[h]=math.fsum(x*x for x in d)/len(d) if d else None
            complete=all(moments[h] is not None for h in HORIZONS)
            monthly_rms=[]
            for month in month_labels:
                mh=[]
                for h in HORIZONS:
                    hr=[r for r in vals if r["horizon"]==h and r["target"].strftime("%Y-%m")==month]
                    dx=[r["conditions"][l]["directLogPrediction"]-r["conditions"][rr]["directLogPrediction"] for r in hr]
                    mh.append(math.fsum(x*x for x in dx)/len(dx) if dx else None)
                ok=all(v is not None for v in mh)
                monthly_rms.append({"targetMonth":month,"complete10HorizonSupport":ok,"equalHorizonRms":math.sqrt(math.fsum(v for v in mh if v is not None)/10) if ok else None})
            interventions[f"{support_name}/{label}"]={"n":len(vals),"byHorizon":by_h,"byTargetMonth":monthly_rms,"complete10HorizonSupport":complete,"equalHorizonRms":math.sqrt(math.fsum(moments[h] for h in HORIZONS)/10) if complete else None}
        for name,first,second,base in (("F_B_vs_C_B","F","C","B"),("F_B_vs_A_B","F","A","B")):
            part=[r for r in source if all(c in r["issued"] for c in (first,second,base))]
            cross=[]; nx=[]; ny=[]; by_h={}
            for h in HORIZONS:
                hr=[r for r in part if r["horizon"]==h]
                x=[r["conditions"][first]["directLogPrediction"]-r["conditions"][base]["directLogPrediction"] for r in hr]
                y=[r["conditions"][second]["directLogPrediction"]-r["conditions"][base]["directLogPrediction"] for r in hr]
                n=len(hr); cross.append(math.fsum(a*b for a,b in zip(x,y))/n if n else None)
                nx.append(math.fsum(a*a for a in x)/n if n else None); ny.append(math.fsum(b*b for b in y)/n if n else None)
                den=math.sqrt(nx[-1]*ny[-1]) if n else 0
                by_h[str(h)]={"n":n,"cosine":cross[-1]/den if den else None}
            complete=all(x is not None for x in cross+nx+ny)
            dot=math.fsum(x for x in cross if x is not None)/10 if complete else None
            xx=math.fsum(x for x in nx if x is not None)/10 if complete else None
            yy=math.fsum(x for x in ny if x is not None)/10 if complete else None
            monthly_cos=[]
            for month in month_labels:
                cs=[]; xs=[]; ys=[]
                for h in HORIZONS:
                    hr=[r for r in part if r["horizon"]==h and r["target"].strftime("%Y-%m")==month]
                    x=[r["conditions"][first]["directLogPrediction"]-r["conditions"][base]["directLogPrediction"] for r in hr]
                    y=[r["conditions"][second]["directLogPrediction"]-r["conditions"][base]["directLogPrediction"] for r in hr]
                    n=len(hr); cs.append(math.fsum(a*b for a,b in zip(x,y))/n if n else None)
                    xs.append(math.fsum(a*a for a in x)/n if n else None); ys.append(math.fsum(b*b for b in y)/n if n else None)
                ok=all(v is not None for v in cs+xs+ys)
                d0=math.fsum(v for v in cs if v is not None)/10 if ok else None
                x0=math.fsum(v for v in xs if v is not None)/10 if ok else None
                y0=math.fsum(v for v in ys if v is not None)/10 if ok else None
                monthly_cos.append({"targetMonth":month,"complete10HorizonSupport":ok,"equalHorizonCosine":d0/math.sqrt(x0*y0) if ok and x0 and y0 else None})
            vectors[f"{support_name}/{name}"]={"n":len(part),"byHorizon":by_h,"byTargetMonth":monthly_cos,"complete10HorizonSupport":complete,"equalHorizonCosine":dot/math.sqrt(xx*yy) if complete and xx and yy else None}
    return {"comparisons":result,"evaluationInterventionLogMovement":interventions,"evaluationVectorSimilarity":vectors}

def score_manifest(doc:Mapping[str,Any],*,baseline_rows:Mapping[Any,Any],actuals:Mapping[Any,Any],fit_diagnostics:Any,phase:str)->dict[str,Any]:
    """Score a verified manifest with baseline rows, actual prices, and fit diagnostics."""
    if phase not in WINDOWS: raise ValueError("phase must be V or T")
    start,end=WINDOWS[phase]; cutoff=ORIGIN_CUTOFF[phase]
    forecasts=doc.get("forecasts") if isinstance(doc,Mapping) else None
    if not isinstance(forecasts,list): raise ValueError("manifest forecasts must be a list")
    expected={(p,t-dt.timedelta(days=h),t,h) for p in PRODUCTS for i in range((end-start).days+1) for t in [start+dt.timedelta(days=i)] for h in HORIZONS}
    bases={_mapkey(k):v for k,v in baseline_rows.items()}; by={}
    for row in forecasts:
        k=_key(row)
        if k in by: raise ValueError("duplicate forecast identity")
        p,o,t,h=k
        if not start<=t<=end: raise ValueError("target outside frozen phase window")
        if type(row.get("eligible")) is not bool or row["eligible"] is not (o>=cutoff): raise ValueError("eligibility mismatch")
        raw_conditions=row.get("conditions")
        if not isinstance(raw_conditions,dict) or set(raw_conditions)!=set(CONDITIONS): raise ValueError("conditions must be B/C/F/A")
        cs={name:dict(value) if isinstance(value,dict) else value for name,value in raw_conditions.items()}
        issued=set()
        for c,x in cs.items():
            if not isinstance(x,dict) or x.get("status") not in ("issued","not_issued","fallback"): raise ValueError("invalid condition status")
            if x["status"] in ("issued","fallback"):
                if _num(x.get("directLogPrediction")) is None or _num(x.get("directPrediction"),True) is None or _num(x.get("blendPrediction"),True) is None: raise ValueError("invalid issued forecast")
                p0=_num(row.get("p0"),True)
                if p0 and not _close(p0*math.exp(float(x["directLogPrediction"])),x["directPrediction"]): raise ValueError("direct log and price predictions disagree")
                sarima=_num(row.get("sarimaPrediction"),True)
                # The 50:50 direct/SARIMA rule is the radish candidate policy.
                # Other crops retain their pinned E106 composition, which can
                # use a distinct blend rule and is verified in forecast replay.
                if p=="radish" and c in ("F","A") and x["status"]=="issued":
                    if sarima is None or not _close((float(x["directPrediction"])+sarima)/2,x["blendPrediction"]): raise ValueError("issued radish candidate blend is not the frozen 50:50 direct/SARIMA combination")
                if sarima is not None: x.setdefault("sarimaPrediction",sarima)
                issued.add(c)
        if p=="radish":
            sarimas=[cs[c].get("sarimaPrediction") for c in issued]
            if sarimas and any(not _close(sarimas[0],v) for v in sarimas[1:]): raise ValueError("radish conditions do not share the same SARIMA prediction")
        # Preserve source-issued B/C rows on ineligible full-grid keys. They
        # remain visible in the manifest but never enter actual scoring.
        if p=="radish":
            if k not in bases: raise ValueError("missing saved radish E106 baseline")
            b=cs["B"]; ref=bases[k]
            if b["status"]!=ref.get("status") or not _close(b.get("blendPrediction"),ref.get("prediction")) or not _close(b.get("directPrediction"),ref.get("directPrediction")) or not _close(row.get("sarimaPrediction"),ref.get("sarimaPrediction")): raise ValueError("B does not replay saved E106 baseline")
        by[k]={**row,"product":p,"origin":o,"target":t,"horizon":h,"issued":issued,"conditions":cs}
    if set(by)!=expected: raise ValueError("manifest must contain the exact full four-crop grid")
    act={}
    for (p,d),raw in actuals.items():
        if p in PRODUCTS:
            v=raw[0] if isinstance(raw,(tuple,list)) and raw else raw
            if (v:=_num(v,True)) is not None: act[p,_day(d)]=v
    rows=[]
    for r in by.values():
        if not r["eligible"] or (r["product"],r["target"]) not in act: continue
        z=dict(r); z["actual"]=act[z["product"],z["target"]]; p0=_num(z.get("p0"),True)
        z["actualLogReturn"]=math.log(z["actual"]/p0) if p0 else None; rows.append(z)
    # All inferential and gate comparisons are on the experimental crop. The
    # three unchanged crop rows are retained only for the four-crop macro.
    radish_rows=[r for r in rows if r["product"]=="radish"]
    paired={label:[r for r in radish_rows if l in r["issued"] and rr in r["issued"]] for label,l,rr in COMPARISONS}
    comparisons={}
    for label,l,rr in COMPARISONS:
        cell=paired[label]; byh={str(h):{c:_metric([r for r in cell if r["horizon"]==h],c) for c in (l,rr)} for h in HORIZONS}
        complete=all(byh[str(h)][l]["n"] and byh[str(h)][rr]["n"] for h in HORIZONS)
        macro={c:{loss:(math.fsum(byh[str(h)][c][loss] for h in HORIZONS)/10 if complete else None) for loss in ("mae","mapePct")} for c in (l,rr)}
        comparisons[label]={"n":len(cell),"uniqueTargetDates":len({r["target"] for r in cell}),"uniqueOrigins":len({r["origin"] for r in cell}),"byHorizon":byh,"equalHorizon":macro,"complete10HorizonSupport":complete}
    # The primary F-C support does not require A; F-A absence remains in fixed family.
    dm,band_summaries=_infer(radish_rows)
    boots={str(b):_bootstrap(radish_rows,phase,b) for b in BLOCKS}
    train=_training(fit_diagnostics)
    coverage={}
    planned=[r for r in by.values() if r["eligible"]]
    for c in CONDITIONS:
        coverage[c]={"eligibleKeys":len(planned),"issued":sum(c in r["issued"] for r in planned),"fallback":sum(r["conditions"][c]["status"]=="fallback" for r in planned),"notIssued":sum(r["conditions"][c]["status"]=="not_issued" for r in planned)}
    transitions=defaultdict(int); lost=[]; coverage_loss=[]
    for r in planned:
        bs=r["conditions"]["C"]["status"]; fs=r["conditions"]["F"]["status"]
        transitions[f"{bs}->{fs}"]+=1
        if bs in ("issued","fallback") and fs=="not_issued":
            lost.append(f"{r['product']}/{r['origin'].isoformat()}/{r['target'].isoformat()}/{r['horizon']}")
        if bs=="issued" and fs!="issued":
            coverage_loss.append(f"{r['product']}/{r['origin'].isoformat()}/{r['target'].isoformat()}/{r['horizon']}:{fs}")
    coverage["C_to_F_transitions"]=dict(transitions)
    coverage["F_missing_when_C_available"]=lost
    coverage["F_coverageLossWhenCIssued"]=coverage_loss
    months=_months(phase); monthly=[]
    for m in months:
        for h in HORIZONS:
            cell=[r for r in rows if r["product"]=="radish" and r["horizon"]==h and r["target"].strftime("%Y-%m")==m]
            monthly.append({"targetMonth":m,"horizon":h,"metricsByCondition":{c:_metric(cell,c) for c in CONDITIONS}})
    # F/C is the experimental comparison and keeps its own support even when
    # A is unavailable; the all-four-condition table is separately ancillary.
    four_fc=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            cell=[r for r in rows if r["product"]==p and r["horizon"]==h and {"F","C"}<=r["issued"]]
            fm,cm=_metric(cell,"F"),_metric(cell,"C")
            four_fc.append({"product":p,"horizon":h,"n":len(cell),"F":fm,"C":cm,
                "F_minus_C":{"mae":fm["mae"]-cm["mae"] if fm["mae"] is not None and cm["mae"] is not None else None,
                             "mapePct":fm["mapePct"]-cm["mapePct"] if fm["mapePct"] is not None and cm["mapePct"] is not None else None}})
    four_aux=[{"product":p,"horizon":h,"condition":c,**_metric([r for r in rows if r["product"]==p and r["horizon"]==h and set(CONDITIONS)<=r["issued"]],c)} for p in PRODUCTS for h in HORIZONS for c in CONDITIONS]
    # Candidate−C point gate only; parent owns whether to continue to T.
    gate=comparisons["F_vs_C"]["equalHorizon"]
    gate={**gate,"differenceFMinusC":{"mae":(gate["F"]["mae"]-gate["C"]["mae"]) if gate["F"]["mae"] is not None and gate["C"]["mae"] is not None else None,
                                        "mapePct":(gate["F"]["mapePct"]-gate["C"]["mapePct"]) if gate["F"]["mapePct"] is not None and gate["C"]["mapePct"] is not None else None},
          "coverageLossCount":len(coverage_loss),"candidateMissingWhenCAvailable":coverage["F_missing_when_C_available"]}
    d=gate["differenceFMinusC"]
    reasons=[]
    if not comparisons["F_vs_C"]["complete10HorizonSupport"]: reasons.append("incomplete_common_10_horizon_support")
    if d["mapePct"] is None or not d["mapePct"]<0: reasons.append("F_mape_not_strictly_lower_than_C")
    if d["mae"] is None or d["mae"]>0: reasons.append("F_mae_higher_than_C")
    if coverage_loss: reasons.append("F_has_added_issue_or_coverage_loss")
    gate["passed"]=not reasons
    gate["reasons"]=reasons
    four_complete=all(r["F_minus_C"]["mae"] is not None and r["F_minus_C"]["mapePct"] is not None for r in four_fc)
    four_macro={"complete40Cells":four_complete,"cellsIncluded":sum(r["F_minus_C"]["mae"] is not None for r in four_fc),
                "F_minus_C":{"mae":math.fsum(r["F_minus_C"]["mae"] for r in four_fc if r["F_minus_C"]["mae"] is not None)/40 if four_complete else None,
                             "mapePct":math.fsum(r["F_minus_C"]["mapePct"] for r in four_fc if r["F_minus_C"]["mapePct"] is not None)/40 if four_complete else None}}
    return {"phase":phase,"scope":"retrospective; not independent future validation","comparisons":comparisons,"dmTests":dm,"dmBandSummaries":band_summaries,"calendarBootstrap":boots,"coverage":coverage,"monthlyRadishMetrics":monthly,"priceMovement":_movement_diagnostics(radish_rows,by,phase),"directLogErrorDecomposition":_log_error_report(radish_rows,by,phase),"trainingDiagnostics":train,"allFourFCCommonMetrics":four_fc,"fourCrop40CellMacro":four_macro,"allFourConditionCommonMetrics":four_aux,"vGateMetrics":gate,"vGateRule":"unrounded F MAPE < C, F MAE <= C, and no additional F issue/coverage loss; A does not gate"}
