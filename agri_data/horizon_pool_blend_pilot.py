"""Offline distance-specific/monthly-pooled Ridge blend pilot, with separate scoring."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT=Path(__file__).resolve().parents[1]
PRICE_PATH=dh.INPUT_PRICES
REPORT_PATH=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
PRICE_SHA="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS=tuple(dashboard.PRODUCTS); HORIZONS=(1,5,10)
START=dt.date(2024,8,8); END=dt.date(2024,10,31); BANDWIDTHS=(9,28)
COMPARATORS=("monthlyPooledRidge","direct","persistence"); LOSSES=("mae","mape"); FAMILY=5*3*3*2
PROTOCOL={"id":"horizon-pool-blend-pilot-2026-10-08","evaluationOrigins":[START.isoformat(),END.isoformat()],"products":list(PRODUCTS),"horizons":list(HORIZONS),"requiredGridRows":1275,"candidate":"0.5*direct + 0.5*monthlyPooledRidge; only if both are present","comparators":list(COMPARATORS),"aggregation":"equal three-horizon mean per product, then equal five-product mean","dm":{"twoSided":True,"calendarBartlettHacDays":list(BANDWIDTHS),"familySizePerBandwidth":FAMILY,"holmAlpha":.05,"minimumPairs":30,"unavailablePValue":1.0},"gates":{"commonPairsPerCell":30,"candidateCoverageAtLeastDirect":True,"candidateFiveProductMacroMapeAtMostPct":10,"candidateMapeImprovementVsDirectAndPooledPctPoints":.5,"perProductMapeWorseningAtMostPctPoints":1,"perProductGateBenchmark":"direct","perProductEqualThreeHorizonMaeWorseningAtMostPct":5},"limitation":"Pinned Ridge report inherits revised prices and origin-day information; this does not establish 10:00 issuance-time performance."}
class PilotError(ValueError):pass
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(x:Any)->bytes:return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def load(path):
 try:raw=Path(path).read_bytes(); obj=json.loads(raw)
 except (OSError,json.JSONDecodeError) as e:raise PilotError(f"cannot read JSON: {path}") from e
 if not isinstance(obj,dict):raise PilotError("JSON root must be object")
 return obj,raw
def _prices(raw):
 try:return dh._observations(raw)
 except (ValueError,TypeError,KeyError) as e:raise PilotError(f"invalid prices: {e}") from e
def _source_map(report):
 if report.get('schemaVersion',1)!=1 or report.get('protocol',{}).get('id')!='direct-horizon-ridge-2026-10-06':raise PilotError('unexpected direct report schema/protocol')
 out={}
 for row in report.get('pairedRows',[]):
  if not isinstance(row,dict):raise PilotError('direct report rows must be objects')
  p=row.get('product');h=row.get('horizon')
  if p not in PRODUCTS or isinstance(h,bool) or not isinstance(h,int) or h not in range(1,11):raise PilotError('invalid source product/horizon')
  try:t=dt.date.fromisoformat(row['target']);o=dt.date.fromisoformat(row.get('origin') or (t-dt.timedelta(days=h)).isoformat())
  except (KeyError,ValueError,TypeError) as e:raise PilotError('invalid source identity') from e
  if t-o!=dt.timedelta(days=h):raise PilotError('source origin/horizon mismatch')
  vals={}
  for field in ('direct','monthlyPooledRidge','persistence'):
   v=row.get(field)
   try:v=float(v) if v is not None else None
   except (ValueError,TypeError,OverflowError):v=None
   vals[field]=v if finite(v) and v>0 else None
  key=(p,o,h)
  if key in out:raise PilotError('duplicate source row identity')
  # `actual` is deliberately not read during forecast construction.
  out[key]={'product':p,'origin':o,'target':t,'horizon':h,**vals}
 return out

def _report_actual_map(report):
 """Read answers only in the score stage, indexed by exact forecast identity."""
 out={}
 for row in report.get('pairedRows',[]):
  try:
   p=row['product'];h=row['horizon'];t=dt.date.fromisoformat(row['target']);o=dt.date.fromisoformat(row.get('origin') or (t-dt.timedelta(days=h)).isoformat());y=float(row['actual'])
  except (KeyError,ValueError,TypeError,OverflowError) as e:raise PilotError('invalid report actual during scoring') from e
  if not finite(y) or y<=0:raise PilotError('report actual must be positive finite during scoring')
  key=(p,o,h)
  if key in out:raise PilotError('duplicate report actual identity during scoring')
  out[key]=(t,y)
 return out

def origins():return [START+dt.timedelta(days=i) for i in range((END-START).days+1)]
def build_forecast(report,hashes=None):
 source=_source_map(report); rows=[]
 for o in origins():
  for p in PRODUCTS:
   for h in HORIZONS:
    t=o+dt.timedelta(days=h); s=source.get((p,o,h)); vals={m:(s.get(m) if s else None) for m in ('direct','monthlyPooledRidge','persistence')}
    candidate=.5*vals['direct']+.5*vals['monthlyPooledRidge'] if vals['direct'] is not None and vals['monthlyPooledRidge'] is not None else None
    row={'product':p,'origin':o.isoformat(),'target':t.isoformat(),'horizon':h,**vals,'candidate':candidate,'sourceRowPresent':s is not None}
    if s is None:row['missingSourceReason']='no_direct_report_pair'
    else:
     for name in ('direct','monthlyPooledRidge','persistence'):
      if vals[name] is None:row[name+'MissingReason']='source_value_missing_or_invalid'
     if candidate is None:row['candidateMissingReason']='direct_or_pooled_source_missing'
    rows.append(row)
 if len(rows)!=1275:raise AssertionError('fixed 1275 plan grid mismatch')
 counts={m:{} for m in ('candidate','monthlyPooledRidge','direct','persistence')}
 for r in rows:
  for m in counts:
   if r.get(m) is None:
    why=(r.get('candidateMissingReason') or r.get('missingSourceReason')) if m=='candidate' else r.get(m+'MissingReason') or r.get('missingSourceReason')
    if why:counts[m][why]=counts[m].get(why,0)+1
 return {'schemaVersion':1,'status':'answer_free_forecast_manifest','protocol':PROTOCOL,'inputs':hashes or {},'missingCounts':counts,'forecasts':rows}

def expected_grid():return {(p,o.isoformat(),h,(o+dt.timedelta(days=h)).isoformat()) for o in origins() for p in PRODUCTS for h in HORIZONS}
def _validate_grid(rows):
 if not isinstance(rows,list) or len(rows)!=1275:raise PilotError('forecast grid must contain exactly 1275 rows')
 seen=set(); expected=expected_grid()
 for r in rows:
  if not isinstance(r,dict):raise PilotError('forecast row must be object')
  if 'actual' in r:raise PilotError('answer field forbidden in forecast manifest')
  try:p=r['product'];o=dt.date.fromisoformat(r['origin']);t=dt.date.fromisoformat(r['target']);h=r['horizon']
  except (KeyError,ValueError,TypeError):raise PilotError('malformed forecast identity')
  if not isinstance(p,str) or isinstance(h,bool) or not isinstance(h,int):raise PilotError('invalid forecast identity type')
  key=(p,o.isoformat(),h,t.isoformat())
  if key in seen:raise PilotError('duplicate forecast identity')
  seen.add(key)
  if key not in expected or t!=o+dt.timedelta(days=h):raise PilotError('forecast outside fixed plan grid')
  for method in ('candidate','direct','monthlyPooledRidge','persistence'):
   v=r.get(method); reason=(r.get('candidateMissingReason') or r.get('missingSourceReason')) if method=='candidate' else r.get(method+'MissingReason') or r.get('missingSourceReason')
   if v is not None and (not finite(v) or v<=0):raise PilotError(f'invalid {method} value')
   if v is None and not reason:raise PilotError(f'null {method} lacks reason')
   if v is not None and reason:raise PilotError(f'{method} value contradicts missing reason')
  expected_candidate=.5*r['direct']+.5*r['monthlyPooledRidge'] if r['direct'] is not None and r['monthlyPooledRidge'] is not None else None
  if r['candidate']!=expected_candidate:raise PilotError('candidate is not the frozen 50:50 blend')
 if seen!=expected:raise PilotError('forecast grid does not exactly cover fixed plan')
def _actuals(prices):return _prices(prices)
def _metric(rows,field):
 if not rows:return {'mae':None,'mape':None}
 return {'mae':sum(abs(r['actual']-r[field]) for r in rows)/len(rows),'mape':100*sum(abs(r['actual']-r[field])/r['actual'] for r in rows)/len(rows)}
def _dm(rows,comp,loss,bw):
 diff={dt.date.fromisoformat(r['target']):(abs(r['actual']-r['candidate'])-abs(r['actual']-r[comp]) if loss=='mae' else 100*(abs(r['actual']-r['candidate'])-abs(r[comp]-r['actual']))/r['actual']) for r in rows}
 return dh._dm(diff,bw)
def score(forecast,prices,report,forecast_bytes=None,prices_bytes=None):
 if forecast.get('status')!='answer_free_forecast_manifest' or forecast.get('protocol')!=PROTOCOL:raise PilotError('forecast status/protocol mismatch')
 if forecast_bytes is not None:
  x=dict(forecast); stated=x.pop('manifestSha256',None)
  if sha(canon(x))!=stated:raise PilotError('forecast manifest SHA mismatch')
 if prices_bytes is not None and forecast.get('inputs',{}).get('prices')!=sha(prices_bytes):raise PilotError('prices SHA mismatch')
 source=_source_map(report); report_actuals=_report_actual_map(report); price_map=_actuals(prices); rows=forecast.get('forecasts');_validate_grid(rows)
 source_actual_checks=0; joined=[]; drops={'missingActual':0,'missingCandidate':0,'missingPooled':0,'missingDirect':0,'missingPersistence':0}
 forecast_count={m:0 for m in ('candidate','monthlyPooledRidge','direct','persistence')}; scoreable_count=forecast_count.copy()
 for f in rows:
  key=(f['product'],dt.date.fromisoformat(f['origin']),f['horizon']); src=source.get(key)
  # Verify source predictions and report answer alignment only in the score stage.
  if src is None:
   if f['sourceRowPresent']:raise PilotError('manifest claims absent direct report source row')
  else:
   if not f['sourceRowPresent']:raise PilotError('manifest omits an existing direct report source row')
   for m in ('direct','monthlyPooledRidge','persistence'):
    if f[m]!=src[m]:raise PilotError(f'{m} source value differs from frozen report')
  for m in forecast_count:
   if finite(f.get(m)) and f[m]>0:forecast_count[m]+=1
  target=dt.date.fromisoformat(f['target']); actual=price_map[f['product']].get(target)
  if src is not None:
   report_identity=report_actuals.get(key)
   if report_identity is None or report_identity[0]!=target:raise PilotError('source report actual missing or invalid during scoring')
   report_actual=report_identity[1]
   if actual is None or not math.isclose(actual[0],report_actual,rel_tol=1e-12,abs_tol=1e-12):raise PilotError('source report actual does not agree with pinned prices')
   source_actual_checks+=1
  if actual is None or not finite(actual[0]) or actual[0]<=0:drops['missingActual']+=1;continue
  x={**f,'actual':actual[0]};joined.append(x)
  for m in scoreable_count:
   if finite(f.get(m)) and f[m]>0:scoreable_count[m]+=1
   else:drops[{'candidate':'missingCandidate','monthlyPooledRidge':'missingPooled','direct':'missingDirect','persistence':'missingPersistence'}[m]]+=1
 common=[r for r in joined if all(finite(r.get(m)) and r[m]>0 for m in forecast_count)]
 cells={};comparisons=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   group=[r for r in common if r['product']==p and r['horizon']==h];cells[f'{p}/{h}']={'n':len(group),'metrics':{m:_metric(group,m) for m in ('candidate','monthlyPooledRidge','direct','persistence')}}
   for comp in COMPARATORS:
    for loss in LOSSES:
     tests={str(b):_dm(group,comp,loss,b) for b in BANDWIDTHS}
     comparisons.append({'product':p,'horizon':h,'comparator':comp,'loss':loss,'n':len(group),'candidateMetric':cells[f'{p}/{h}']['metrics']['candidate'][loss],'comparatorMetric':cells[f'{p}/{h}']['metrics'][comp][loss],'dm':tests})
 for b in BANDWIDTHS:dh._holm([c['dm'][str(b)] for c in comparisons])
 product_metrics={};macro={}
 for method in ('candidate','monthlyPooledRidge','direct','persistence'):
  product_metrics[method]={}
  for p in PRODUCTS:
   groups=[cells[f'{p}/{h}']['metrics'][method] for h in HORIZONS]
   product_metrics[method][p]={k:(sum(g[k] for g in groups)/3 if all(g[k] is not None for g in groups) else None) for k in ('mae','mape')}
  macro[method]={k:(sum(product_metrics[method][p][k] for p in PRODUCTS)/5 if all(product_metrics[method][p][k] is not None for p in PRODUCTS) else None) for k in ('mae','mape')}
 deltas={p:{'mapeWorseningPctPointsVsDirect':(product_metrics['candidate'][p]['mape']-product_metrics['direct'][p]['mape'] if product_metrics['candidate'][p]['mape'] is not None and product_metrics['direct'][p]['mape'] is not None else None),'equalThreeHorizonMaeWorseningPctVsDirect':(100*(product_metrics['candidate'][p]['mae']-product_metrics['direct'][p]['mae'])/product_metrics['direct'][p]['mae'] if product_metrics['candidate'][p]['mae'] is not None and product_metrics['direct'][p]['mae'] not in (None,0) else None)} for p in PRODUCTS}
 gates={'all15CellsCommonPairsAtLeast30':all(cells[f'{p}/{h}']['n']>=30 for p in PRODUCTS for h in HORIZONS),'candidateCoverageAtLeastDirect':forecast_count['candidate']>=forecast_count['direct'],'candidateFiveProductMacroMapeAtMost10Pct':macro['candidate']['mape'] is not None and macro['candidate']['mape']<=10,'candidateMapeImprovesVsDirectByAtLeast0_5PctPoints':macro['candidate']['mape'] is not None and macro['direct']['mape'] is not None and macro['candidate']['mape']<=macro['direct']['mape']-.5,'candidateMapeImprovesVsPooledByAtLeast0_5PctPoints':macro['candidate']['mape'] is not None and macro['monthlyPooledRidge']['mape'] is not None and macro['candidate']['mape']<=macro['monthlyPooledRidge']['mape']-.5,'eachProductMapeWorseningAtMost1PctPoint':all(deltas[p]['mapeWorseningPctPointsVsDirect'] is not None and deltas[p]['mapeWorseningPctPointsVsDirect']<=1 for p in PRODUCTS),'eachProductEqualThreeHorizonMaeWorseningAtMost5Pct':all(deltas[p]['equalThreeHorizonMaeWorseningPctVsDirect'] is not None and deltas[p]['equalThreeHorizonMaeWorseningPctVsDirect']<=5 for p in PRODUCTS)};gates['expand']=all(gates.values())
 conclusions=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   family=[c for c in comparisons if c['product']==p and c['horizon']==h]; tests=[c['dm']['9'] for c in family];enough=all(t['status']=='tested' for t in tests)
   lower=all(c['candidateMetric'] is not None and c['comparatorMetric'] is not None and c['candidateMetric']<c['comparatorMetric'] for c in family)
   improve=all(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']<0 for t in tests);adverse=any(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']>0 for t in tests)
   conclusions.append({'product':p,'horizon':h,'conclusion':'worsening_or_mixed' if adverse else 'improvement' if enough and lower and improve else 'inference_unavailable' if not enough else 'inconclusive','allPairedLossesLower':lower,'allHac9HolmTestsImprove':improve,'inferenceAvailable':enough})
 return {'schemaVersion':1,'status':'scored','forecastManifestSha256':forecast.get('manifestSha256'),'fourWayCommonRows':len(common),'byProductHorizon':cells,'macroMetrics':macro,'byProductEqualThreeHorizonMetrics':product_metrics,'productGateDeltas':deltas,'gates':gates,'conclusions':conclusions,'coverage':{'plannedRows':1275,'forecastValidRows':forecast_count,'forecastFractionOfPlan':{m:n/1275 for m,n in forecast_count.items()},'rowsWithActual':len(joined),'scoreableValidRows':scoreable_count,'scoreableFractionOfRowsWithActual':{m:(n/len(joined) if joined else 0) for m,n in scoreable_count.items()},'fourWayCommonRows':len(common),'sourceRowsCheckedAgainstPrices':source_actual_checks,'dropReasons':drops},'dm':{'familySize':FAMILY,'comparisons':comparisons,'bandwidths':list(BANDWIDTHS)},'sampleLimits':{'cellsBelowThirty':[k for k,v in cells.items() if v['n']<30],'minimumPairs':30}}

def _write(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
def main(argv=None):
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
 f=sp.add_parser('forecast');f.add_argument('--prices',type=Path,default=PRICE_PATH);f.add_argument('--direct-report',type=Path,default=REPORT_PATH);f.add_argument('--output',type=Path,required=True)
 s=sp.add_parser('score');s.add_argument('--forecast-manifest',type=Path,required=True);s.add_argument('--direct-report',type=Path,default=REPORT_PATH);s.add_argument('--prices',type=Path,default=PRICE_PATH);s.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(argv)
 try:
  if a.cmd=='forecast':
   report,rb=load(a.direct_report);price_bytes=a.prices.read_bytes()
   if sha(rb)!=REPORT_SHA or sha(price_bytes)!=PRICE_SHA:raise PilotError('pinned input hash mismatch')
   hashes={'prices':sha(price_bytes),'directReport':sha(rb),'code':sha(Path(__file__).read_bytes())}
   result=build_forecast(report,hashes=hashes);result['manifestSha256']=sha(canon(result));_write(a.output,canon(result))
  else:
   forecast,fb=load(a.forecast_manifest);report,rb=load(a.direct_report);prices,pb=load(a.prices)
   if sha(rb)!=REPORT_SHA or forecast.get('inputs',{}).get('directReport')!=REPORT_SHA:raise PilotError('pinned direct report hash mismatch')
   if sha(pb)!=PRICE_SHA:raise PilotError('pinned price hash mismatch')
   if forecast.get('inputs',{}).get('code')!=sha(Path(__file__).read_bytes()):raise PilotError('forecast code hash mismatch')
   result=score(forecast,prices,report,fb,pb);result['inputHashes']={'directReport':sha(rb),'prices':sha(pb),'code':forecast['inputs']['code']};_write(a.output,canon(result))
 except (PilotError,ValueError,OSError,KeyError) as e:ap.error(str(e))
 return 0
if __name__=='__main__':raise SystemExit(main())
