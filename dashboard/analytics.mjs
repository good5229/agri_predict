const DAY = 86400000;
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const dayTime = date => Date.parse(`${date}T00:00:00Z`);
export function validDate(date) {
  return typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date) && Number.isFinite(dayTime(date)) && new Date(dayTime(date)).toISOString().slice(0, 10) === date;
}
export function validateData(data) {
  if (![1,2].includes(data?.schemaVersion) || !Array.isArray(data.products) || data.products.length !== 5 || !validDate(data.meta?.evaluationStart) || !validDate(data.meta?.evaluationEnd) || data.meta.evaluationStart > data.meta.evaluationEnd) throw new Error('평가 데이터 형식 또는 평가기간이 올바르지 않습니다. 데이터를 다시 생성해 주세요.');
  const ids = new Set(['cabbage', 'radish', 'onion', 'garlic', 'pepper']);
  for (const p of data.products) {
    if (!ids.delete(p.id) || typeof p.name !== 'string' || !Array.isArray(p.observations) || !Array.isArray(p.predictions)) throw new Error('품목별 가격과 예측 데이터 형식이 올바르지 않습니다.');
    const observations = new Map();
    for (const o of p.observations) {
      if (!validDate(o.date) || !finite(o.actual) || o.actual <= 0 || observations.has(o.date)) throw new Error(`${p.name}: 실제가격 날짜 또는 값이 올바르지 않습니다.`);
      observations.set(o.date, o.actual);
    }
    const runs=[{predictions:p.predictions},...(p.runs??[])];
    const runIds=new Set();
    for(const run of runs){
    if(run.modelId){const id=`${run.modelId}:${run.cadence}`;if(runIds.has(id)||!['ridge','hist_gb'].includes(run.modelId)||!['daily','weekly','monthly','seasonal'].includes(run.cadence)||!Array.isArray(run.predictions))throw new Error(`${p.name}: 모델 실행 구성이 올바르지 않습니다.`);runIds.add(id);}
    const predictions = new Set();
    for (const r of run.predictions) {
      const key = `${r.horizon}:${r.target}`;
      if (!validDate(r.origin) || !validDate(r.target) || !validDate(r.sourceDate) || !Number.isInteger(r.horizon) || r.horizon < 1 || r.horizon > 10 || dayTime(r.target) - dayTime(r.origin) !== r.horizon * DAY || r.sourceDate > r.origin || !finite(r.predicted) || r.predicted <= 0 || predictions.has(key)) throw new Error(`${p.name}: 예측 시점·가격 또는 중복 결과를 확인해 주세요.`);
      predictions.add(key);
      if(run.modelId&&(!validDate(r.fitCutoff)||r.fitCutoff>r.origin))throw new Error(`${p.name}: 학습 기준일이 예측 기준일보다 늦습니다.`);
    }
    }
  }
  return data;
}
export function selectRows(product, {start, end, horizon, model='baseline',cadence='daily',asOf=end}) {
  const selected=model==='baseline'?product.predictions:(product.runs??[]).find(r=>r.modelId===model&&r.cadence===cadence)?.predictions??[];
  const predictions = new Map(selected.filter(p => p.horizon === Number(horizon)).map(p => [p.target, p]));
  const baseline = new Map(product.predictions.filter(p=>p.horizon===Number(horizon)).map(p=>[p.target,p.predicted]));
  return product.observations.filter(o => o.date >= start && o.date <= end && o.date<=asOf).map(o => {
    const p = predictions.get(o.date);
    return {...o, predicted: p?.predicted ?? null, origin: p?.origin ?? null, sourceDate: p?.sourceDate ?? null,fitCutoff:p?.fitCutoff??null,baseline:baseline.get(o.date)??null};
  }).sort((a,b) => a.date.localeCompare(b.date));
}
export function metrics(rows) {
  const pairs = rows.filter(r => finite(r.actual) && finite(r.predicted));
  const n = pairs.length;
  const percentages = pairs.filter(r => r.actual !== 0);
  return {
    n, actualDays: rows.length,
    mae: n ? pairs.reduce((sum,r) => sum + Math.abs(r.predicted-r.actual),0)/n : null,
    rmse: n ? Math.sqrt(pairs.reduce((sum,r) => sum + (r.predicted-r.actual)**2,0)/n) : null,
    mape: percentages.length ? percentages.reduce((sum,r) => sum + Math.abs((r.predicted-r.actual)/r.actual)*100,0)/percentages.length : null,
    mapeN: percentages.length,
  };
}
export function periodKey(date, unit) {
  const d = new Date(dayTime(date));
  if (unit === 'week') d.setUTCDate(d.getUTCDate() - (d.getUTCDay()+6)%7);
  if (unit === 'month') d.setUTCDate(1);
  if (unit === 'quarter') { d.setUTCDate(1); d.setUTCMonth(Math.floor(d.getUTCMonth()/3)*3); }
  if (unit === 'season') {const month=d.getUTCMonth();d.setUTCDate(1);d.setUTCMonth(month<2?-1:month<5?2:month<8?5:month<11?8:11);}
  return d.toISOString().slice(0,10);
}
export function aggregate(rows, unit) {
  if (unit === 'day') return rows.map(r => ({...r, count:finite(r.predicted)?1:0, label:r.date}));
  const groups = new Map();
  for (const row of rows) {const key = periodKey(row.date,unit); if (!groups.has(key)) groups.set(key,[]); groups.get(key).push(row);}
  return [...groups].map(([key, group]) => {
    const pairs = group.filter(r => finite(r.predicted));
    const actualSet = pairs.length ? pairs : group;
    const label = unit === 'week' ? `${key} 주` : unit === 'quarter' ? `${key.slice(0,4)}년 ${Math.floor((Number(key.slice(5,7))-1)/3)+1}분기` : `${key.slice(0,7)}`;
    return {date:group[0].date, label, actual:actualSet.reduce((sum,r)=>sum+r.actual,0)/actualSet.length, predicted:pairs.length?pairs.reduce((sum,r)=>sum+r.predicted,0)/pairs.length:null, count:pairs.length, origin:null};
  });
}
export function scorePeriods(rows,unit) {
  const groups=new Map();
  for(const r of rows){const key=periodKey(r.date,unit);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);}
  return [...groups].map(([date,group])=>({date,start:group[0].date,end:group.at(-1).date,...metrics(group)}));
}
export function baselineComparison(rows) {
  const paired=rows.filter(r=>finite(r.predicted)&&finite(r.baseline));
  return {model:metrics(paired),baseline:metrics(paired.map(r=>({...r,predicted:r.baseline})))};
}
export function significanceSummary(meta, {product,model,cadence,horizon,start,end,asOf}) {
  if(model==='baseline')return '직전가격 기준모델은 통계 검정의 비교 기준입니다.';
  const report=meta.statisticalSignificance;
  if(!report)return '통계적 유의성 검토 미완료: 현재 예측 파일과 일치하는 검정 결과가 없습니다.';
  if(start!==meta.evaluationStart||end!==meta.evaluationEnd||asOf<meta.evaluationEnd)return '검정은 전체 평가기간을 대상으로 수행했습니다. 전체 기간을 선택하고 평가를 끝까지 진행하면 결과를 볼 수 있습니다.';
  const pair=report.comparisons.filter(r=>r.productId===product&&r.modelId===model&&r.cadence===cadence&&r.horizon===Number(horizon));
  const rows=['absolute_error','absolute_percentage_error'].map(loss=>pair.find(r=>r.loss===loss));
  if(rows.some(r=>!r?.primary?.test||!r?.sensitivity?.test))return '이 조합의 통계 검정 결과가 없습니다.';
  const classify=field=>{
    const tests=rows.map(r=>r[field].test);
    const signed=sign=>tests.map(t=>t.status==='available'&&finite(t.holmAdjustedPValue)&&t.holmAdjustedPValue<0.05&&sign*t.meanLossDifference>0);
    const positive=signed(1),negative=signed(-1);
    if(negative.every(Boolean))return 'improved';
    if(negative.some(Boolean)&&positive.some(Boolean))return 'mixed';
    if(positive.some(Boolean))return 'worsened';
    if(tests.some(t=>t.status!=='available'))return 'unavailable';
    return 'no_evidence';
  };
  const status=classify('primary'),sensitivityStatus=classify('sensitivity');
  if(status==='unavailable')return `통계 판단 보류: 같은 날짜의 평가쌍 ${rows[0].primary.n}개. 표본 부족 또는 검정 분산 추정 불가로 유의성을 판단하지 않았습니다.`;
  const p=test=>test.status!=='available'||!finite(test.holmAdjustedPValue)?'판단 불가':test.holmAdjustedPValue<0.0001?'<0.0001':test.holmAdjustedPValue.toFixed(4);
  const headline={improved:'두 오차 지표에서 기준모델보다 작은 오차의 통계적 근거가 있습니다.',worsened:'하나 이상의 오차 지표에서 기준모델보다 큰 오차의 통계적 근거가 있습니다.',mixed:'오차 지표에 따라 유의한 개선과 악화가 엇갈려 개선으로 판정하지 않습니다.',no_evidence:'두 오차 지표 모두에서 개선됐다는 통계적 근거는 부족합니다.'}[status];
  const sensitivity=status===sensitivityStatus?'28일 의존성 점검에서도 같은 판정입니다.':'28일 의존성 점검에서는 판정이 달라집니다.';
  return `${headline} 전체기간 ${meta.evaluationStart}~${meta.evaluationEnd}, ${rows[0].primary.n}쌍 · Holm 보정 p: MAE ${p(rows[0].primary.test)}, MAPE ${p(rows[1].primary.test)}. ${sensitivity} 유의하지 않음은 동등성의 증거가 아닙니다. 사후 탐색 DM 검정이며 미래 성능을 보장하지 않습니다.`;
}
export function toCsv(rows, horizon, productName,model='baseline',cadence='none') {
  const cell = x => `"${String(x??'').replaceAll('"','""')}"`;
  const records = [['품목','대상일','예측기준일','참조관측일','예측거리_일','실제가격_원kg','예측가격_원kg','예측마이너스실제_원kg','모델','재학습주기','학습종료일','기준모델가격_원kg'],...rows.map(r=>[productName,r.date,r.origin,r.sourceDate,horizon,r.actual,r.predicted,finite(r.predicted)?r.predicted-r.actual:null,model,cadence,r.fitCutoff,r.baseline])];
  return '\uFEFF'+records.map(row=>row.map(cell).join(',')).join('\r\n');
}
