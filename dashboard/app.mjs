import {validateData,selectRows,metrics,aggregate,toCsv,finite,dayTime,validDate,scorePeriods,baselineComparison} from './analytics.mjs';
const $ = id => document.getElementById(id);
const fmt = (value,digits=1) => finite(value)?new Intl.NumberFormat('ko-KR',{maximumFractionDigits:digits,minimumFractionDigits:digits}).format(value):'—';
const escape = text => String(text??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, rows=[], points=[];
const state = {product:'cabbage',horizon:1,unit:'day',start:'',end:'',asOf:'',model:'ridge',cadence:'daily',evaluationUnit:'month'};
const cadenceNames={daily:'매일',weekly:'매주',monthly:'매월',seasonal:'매계절'};
const modelNames={baseline:'직전가격 기준모델',ridge:'Ridge 회귀',hist_gb:'부스팅 트리'};
let plot = null, focusIndex = -1;
function metric(label,value,unit,help) {return `<div class="metric"><span class="metric-label">${label}</span><div class="metric-value">${value}<span class="metric-unit">${unit}</span></div><p class="metric-help">${help}</p></div>`;}
function renderProducts() {
  $('products').innerHTML = data.products.map((p,i)=>{
    const count=p.observations.filter(o=>o.date>=data.meta.evaluationStart&&o.date<=data.meta.evaluationEnd).length;
    return `<button class="product" type="button" data-product="${escape(p.id)}" aria-pressed="${p.id===state.product}"><span class="product-top">${escape(p.name)}<span class="product-index">0${i+1}</span></span><span class="product-bottom"><span>관측 ${count}일</span>${count<60?'<span class="sparse">자료 부족</span>':''}</span></button>`;
  }).join('');
}
function render() {
  if (!data) return;
  const invalid = !validDate(state.start)||!validDate(state.end)||state.start>state.end||state.start<data.meta.evaluationStart||state.end>data.meta.evaluationEnd;
  $('filter-error').hidden=!invalid;
  $('results').hidden=invalid;
  $('export').disabled=invalid;
  if (invalid) { $('filter-error').textContent=`조회기간은 ${data.meta.evaluationStart}~${data.meta.evaluationEnd} 안에서, 시작일이 종료일보다 늦지 않게 설정해 주세요.`;$('prediction-rows').innerHTML='<tr><td colspan="4">조회기간을 확인해 주세요.</td></tr>';$('evaluation-rows').innerHTML='<tr><td colspan="5">조회기간을 확인해 주세요.</td></tr>';$('baseline-comparison').textContent='조회기간을 확인해 주세요.';rows=[];return; }
  const product=data.products.find(p=>p.id===state.product);
  rows=selectRows(product,state);
  points=aggregate(rows,state.unit);
  $('chart-title').textContent=`${product.name} 가격 흐름`;
  $('model-badge').textContent=modelNames[state.model];
  $('cadence').disabled=state.model==='baseline';
  // Historical publication timestamps are unknown; the replay uses observed-date order.
  $('asof-label').textContent=`${state.asOf} 관측분까지 평가`;
  $('replay-next').disabled=state.asOf>=data.meta.evaluationEnd;
  document.querySelectorAll('[data-product]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.product===state.product)));
  const m=metrics(rows);
  $('metrics').innerHTML=metric('평균 가격 차이 · MAE',fmt(m.mae),'원/kg','예측이 실제가격에서 벗어난 평균 크기')+metric('평균 오차율 · MAPE',fmt(m.mape),'%','실제가격 대비 평균 오차 · 낮을수록 좋음')+metric('큰 오차까지 반영 · RMSE',fmt(m.rmse),'원/kg','큰 예측 오차에 더 민감한 지표')+metric('비교한 날짜',fmt(m.n,0),'일',`실제 관측 ${m.actualDays}일 중 예측·실제 쌍`);
  $('chart-caption').textContent=`${state.start} — ${state.end} · ${state.asOf}까지 평가 · ${state.horizon}일 전 기준 예측${state.unit==='day'?' · 빈 날짜는 연결하지 않습니다.':' · 같은 날짜쌍 평균 (지표는 일별 기준)'}`;
  $('coverage-note').classList.toggle('warning',m.n<60);
  const run=(product.runs??[]).find(r=>r.modelId===state.model&&r.cadence===state.cadence);
  const missingReason=state.model!=='baseline'&&(!run||run.status!=='available')?(run?.reason??'학습 가능한 표본이 부족해 이 모델의 예측 결과가 없습니다.')+' ':'';
  $('coverage-note').textContent=missingReason+(m.n<60?`평가 표본 ${m.n}일로 성능을 일반화할 수 없습니다. `:'')+`현재 진행일까지 실제 관측 ${m.actualDays}일, 비교 가능 ${m.n}일입니다. 빈 날짜는 임의로 채우지 않았습니다.`;
  const recent=rows.filter(r=>finite(r.predicted)).slice(-8).reverse();
  $('prediction-rows').innerHTML=recent.length?recent.map(r=>`<tr><td>${r.date}<small>${r.origin} 기준</small></td><td>${fmt(r.actual)}</td><td>${fmt(r.predicted)}</td><td>${r.predicted-r.actual>0?'+':''}${fmt(r.predicted-r.actual)}</td></tr>`).join(''):'<tr><td colspan="4">선택 기간에 비교 가능한 예측이 없습니다.</td></tr>';
  $('driver-title').textContent=state.model==='baseline'?'예측 기준일의 최근 관측가격':'과거 가격·거래량과 달력 정보';
  $('model-description').textContent=state.model==='baseline'?`대상일의 ${state.horizon}일 전까지 관측된 마지막 가격을 유지합니다. 참조가격이 기준일보다 14일 넘게 오래되면 예측하지 않습니다.`:`최근 가격과 7·28·90일 동안의 가격 흐름·변동폭·거래량, 요일과 연중 시기를 함께 봅니다. ${state.model==='ridge'?'Ridge 회귀는 이 정보와 이후 가격 변화 사이의 선형 관계를 학습합니다.':'부스팅 트리는 여러 조건의 조합에 따라 달라지는 가격 변화를 학습합니다.'}`;
  const lastPrediction=recent[0];
  const fit=run?.fits?.find(f=>f.cutoff===lastPrediction?.fitCutoff);
  $('training-caption').textContent=state.model==='baseline'?'학습 없는 비교 기준 · 직전 관측가격 유지':fit?`최근 표시 예측의 학습 범위: ${fit.trainStart} ~ ${fit.trainEnd} · ${cadenceNames[state.cadence]} 갱신`:`${data.meta.contextStart??data.meta.dataStart??'확보된 자료'}부터 학습 · 정답이 관측된 뒤 다음 학습에 반영`;
  const comparison=baselineComparison(rows);
  const c=$('baseline-comparison');
  const regressed=comparison.model.n>0&&(comparison.model.mae>comparison.baseline.mae||comparison.model.mape>comparison.baseline.mape);
  c.classList.toggle('warning',state.model!=='baseline'&&regressed);
  c.textContent=state.model==='baseline'?'직전가격 유지 예측은 학습모델 성능을 판단하기 위한 비교 기준입니다.':comparison.model.n?`같은 ${comparison.model.n}일에서 기준모델 MAE ${fmt(comparison.baseline.mae)}원/kg, MAPE ${fmt(comparison.baseline.mape)}%. 선택 모델의 MAE는 ${fmt(Math.abs(comparison.model.mae-comparison.baseline.mae))}원/kg ${comparison.model.mae>comparison.baseline.mae?'더 큽니다':comparison.model.mae<comparison.baseline.mae?'더 작습니다':'차이가 없습니다'}. ${regressed?'MAE·MAPE 중 악화된 지표가 있어 성능 향상으로 판정하지 않습니다.':'통계적 개선 여부는 추가 검증이 필요합니다.'}`:'비교 가능한 학습모델 예측이 없습니다. 기준모델을 선택하면 확보된 관측자료를 확인할 수 있습니다.';
  const limitations=[...(data.meta.limitations??[]),...(product.notes??[]),'깐마늘은 확인된 소분류, 나머지는 표준 중분류로 집계했습니다. 품종·등급·거래방식 등 공식 타깃과의 일치 여부는 추가 확인이 필요합니다.','주·월·분기에서는 실제·예측이 모두 있는 날짜의 가격을 단순 평균합니다. 비교 가능한 날짜가 없는 구간은 실제가격만 표시합니다. 지표와 다운로드는 항상 일별 평가값입니다.','2순·3순 중기 예측, 검증된 예상 가격 범위, 모델 간 DM 검정 결과는 아직 없습니다.'];
  $('limitations').innerHTML=limitations.map(x=>`<li>${escape(x)}</li>`).join('');
  renderEvaluation();
  renderChart();
}
function renderEvaluation(){
  const periods=scorePeriods(rows,state.evaluationUnit).reverse();
  $('evaluation-rows').innerHTML=periods.length?periods.map(p=>{
    let label=p.date;
    if(state.evaluationUnit==='week')label=`${p.date} 주`;
    if(state.evaluationUnit==='month')label=p.date.slice(0,7);
    if(state.evaluationUnit==='season')label=p.date.slice(5,7)==='12'?`${p.date.slice(0,4)}/${String(Number(p.date.slice(0,4))+1).slice(-2)} 겨울`:`${p.date.slice(0,4)}년 ${{'03':'봄','06':'여름','09':'가을'}[p.date.slice(5,7)]}`;
    return `<tr><td>${label}<small>${p.start} ~ ${p.end}</small></td><td>${p.n}</td><td>${fmt(p.mae)}</td><td>${fmt(p.mape)}</td><td>${fmt(p.rmse)}</td></tr>`;
  }).join(''):'<tr><td colspan="5">평가 진행일과 조회기간 안에 관측자료가 없습니다.</td></tr>';
  $('evaluation-footnote').textContent=`${modelNames[state.model]} · ${state.model==='baseline'?'학습 없음':cadenceNames[state.cadence]+' 재학습'} · ${state.horizon}일 뒤 예측 · ${state.asOf}까지 / 진행일 이후 정답은 집계하지 않습니다.`;
}
function renderChart() {
  if (!data || $('results').hidden) return;
  const svg=$('price-chart');
  const width=Math.max(300,$('chart-wrap').clientWidth),height=width<600?270:330;
  const left=width<600?51:62,right=17,top=25,bottom=38;
  const plotWidth=width-left-right,plotHeight=height-top-bottom;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  svg.setAttribute('tabindex','0');
  const name=data.products.find(p=>p.id===state.product).name;
  const title=`${name} 실제가격과 ${state.horizon}일 전 기준 예측가격 비교`;
  $('chart-empty').hidden=points.length>0;
  $('chart-empty').innerHTML='현재 평가 진행일과 조회기간 안에 관측자료가 없습니다.<br>기간을 바꾸거나 평가 진행일을 넘겨 주세요.';
  $('tooltip').hidden=true;focusIndex=-1;
  const values=points.flatMap(p=>[p.actual,p.predicted]).filter(finite);
  if (!values.length) {svg.innerHTML=`<title id="svg-title">${escape(title)}</title><desc id="svg-description">선택한 기간에 데이터가 없습니다.</desc>`;plot=null;return;}
  const lo=Math.min(...values),hi=Math.max(...values),spread=Math.max(hi-lo,hi*.12,1);
  const yMin=Math.max(0,lo-spread*.15),yMax=hi+spread*.15;
  const minTime=dayTime(state.start),maxTime=Math.max(dayTime(state.end),minTime+86400000);
  const x = date=>left+(dayTime(date)-minTime)/(maxTime-minTime)*plotWidth;
  const y = value=>top+(yMax-value)/(yMax-yMin)*plotHeight;
  let contents=`<title id="svg-title">${escape(title)}</title><desc id="svg-description">${state.start}부터 ${state.end}까지. 초록 실선은 실제가격, 황토색 점선은 예측가격입니다. 차트에 초점을 두고 좌우 방향키로 날짜별 가격을 확인할 수 있습니다. 아래 표와 내려받기로도 확인할 수 있습니다.</desc>`;
  for (let i=0;i<5;i++) {const value=yMin+(yMax-yMin)*i/4,pos=y(value);contents+=`<line class="grid-line" x1="${left}" x2="${width-right}" y1="${pos}" y2="${pos}"/><text class="axis-text" text-anchor="end" x="${left-10}" y="${pos+4}">${fmt(value,0)}</text>`;}
  contents+=`<text class="axis-text" x="${left}" y="12">원/kg</text>`;
  const tickCount=width<600?4:7;
  const spansYears=state.start.slice(0,4)!==state.end.slice(0,4);
  for (let i=0;i<tickCount;i++) {const time=minTime+(maxTime-minTime)*i/(tickCount-1);const date=new Date(time).toISOString().slice(0,10);const label=spansYears?date.slice(0,7):date.slice(5);contents+=`<text class="axis-text" x="${left+plotWidth*i/(tickCount-1)}" y="${height-10}" text-anchor="${i===0?'start':i===tickCount-1?'end':'middle'}">${label.replace('-','.')}</text>`;}
  for (const series of ['actual','predicted']) {
    let path='',previous=null;
    for (const p of points) {
      if (!finite(p[series])) {previous=null;continue;}
      const gap=previous&&(dayTime(p.date)-dayTime(previous.date))/86400000;
      const maxGap={day:1,week:13,month:61,quarter:184}[state.unit];
      path+=`${!previous||gap>maxGap?'M':'L'}${x(p.date).toFixed(2)},${y(p[series]).toFixed(2)} `;
      previous=p;
    }
    contents+=`<path class="plot-line ${series}" d="${path}"/>`;
    // Points preserve visibility for isolated trades; missing dates are never bridged in daily mode.
    for (const p of points) if (finite(p[series])) contents+=`<circle class="plot-dot ${series}" cx="${x(p.date)}" cy="${y(p[series])}" r="${points.length<50?2.8:1.2}"/>`;
  }
  contents+=`<line id="chart-crosshair" class="chart-focus" x1="${left}" x2="${left}" y1="${top}" y2="${height-bottom}" visibility="hidden"/>`;
  svg.innerHTML=contents;
  plot={x,y,width,left,right};
}
function showPoint(index) {
  if (!plot||!points.length) return;
  focusIndex=Math.max(0,Math.min(index,points.length-1));
  const p=points[focusIndex],tip=$('tooltip');
  tip.innerHTML=`<strong>${escape(p.label)}</strong><p><span>실제가격</span><b>${fmt(p.actual)}원</b></p><p><span>예측가격</span><b>${finite(p.predicted)?fmt(p.predicted)+'원':'없음'}</b></p>${p.origin?`<small>${p.origin} 예측 기준 · ${p.sourceDate} 관측 참조${p.fitCutoff?'<br>'+p.fitCutoff+'까지 학습':''}</small>`:`<small>비교 ${p.count}일 · 단위 원/kg</small>`}`;
  tip.hidden=false;
  tip.style.left=`${Math.max(0,Math.min(plot.x(p.date)+12,plot.width-tip.offsetWidth-6))}px`;
  const cross=$('chart-crosshair');cross.setAttribute('visibility','visible');cross.setAttribute('x1',plot.x(p.date));cross.setAttribute('x2',plot.x(p.date));
  $('price-chart').setAttribute('aria-label',`${p.label}, 실제가격 ${fmt(p.actual)}원, 예측가격 ${fmt(p.predicted)}원`);
}
$('price-chart').addEventListener('pointermove',e=>{if(!plot||!points.length)return;const rect=e.currentTarget.getBoundingClientRect();const pos=(e.clientX-rect.left)*plot.width/rect.width;let nearest=0;points.forEach((p,i)=>{if(Math.abs(plot.x(p.date)-pos)<Math.abs(plot.x(points[nearest].date)-pos))nearest=i;});showPoint(nearest);});
function hideTooltip(){ $('tooltip').hidden=true;$('chart-crosshair')?.setAttribute('visibility','hidden');$('price-chart').removeAttribute('aria-label'); }
$('price-chart').addEventListener('pointerleave',hideTooltip);
$('price-chart').addEventListener('blur',hideTooltip);
$('price-chart').addEventListener('keydown',e=>{if(['ArrowRight','ArrowLeft','Home','End','Escape'].includes(e.key)){e.preventDefault();if(e.key==='Escape'){hideTooltip();return;}showPoint(e.key==='Home'?0:e.key==='End'?points.length-1:focusIndex+(e.key==='ArrowRight'?1:-1));}});
$('products').addEventListener('click',e=>{const button=e.target.closest('[data-product]');if(button){state.product=button.dataset.product;render();}});
$('horizon').innerHTML=Array.from({length:10},(_,i)=>`<option value="${i+1}">${i+1}일 뒤 가격</option>`).join('');
$('horizon').addEventListener('change',e=>{state.horizon=Number(e.target.value);render();});
$('model').addEventListener('change',e=>{state.model=e.target.value;render();});
$('cadence').addEventListener('change',e=>{state.cadence=e.target.value;render();});
$('evaluation-unit').addEventListener('change',e=>{state.evaluationUnit=e.target.value;renderEvaluation();});
function updateReplay(offset){const days=Number($('asof').max);const step=Math.max(0,Math.min(offset,days));state.asOf=new Date(dayTime(data.meta.evaluationStart)+step*86400000).toISOString().slice(0,10);$('asof').value=step;render();}
$('asof').addEventListener('input',e=>updateReplay(Number(e.target.value)));
$('replay-start').addEventListener('click',()=>updateReplay(0));
$('replay-next').addEventListener('click',()=>updateReplay(Number($('asof').value)+1));
$('granularity').addEventListener('click',e=>{const b=e.target.closest('[data-unit]');if(!b)return;state.unit=b.dataset.unit;document.querySelectorAll('[data-unit]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render();});
$('filters').addEventListener('submit',e=>e.preventDefault());
for (const field of ['start','end']) $(field).addEventListener('change',e=>{state[field]=e.target.value;$('period').value='custom';render();});
$('period').addEventListener('change',e=>{if(e.target.value==='custom')return;state.end=data.meta.evaluationEnd;const d=new Date(dayTime(state.end));if(e.target.value==='all')state.start=data.meta.evaluationStart;else{d.setUTCDate(1);d.setUTCMonth(d.getUTCMonth()-(e.target.value==='half'?5:2));state.start=d.toISOString().slice(0,10);if(state.start<data.meta.evaluationStart)state.start=data.meta.evaluationStart;}$('start').value=state.start;$('end').value=state.end;render();});
$('export').addEventListener('click',()=>{const product=data.products.find(p=>p.id===state.product);const link=document.createElement('a');const url=URL.createObjectURL(new Blob([toCsv(rows,state.horizon,product.name,state.model,state.model==='baseline'?'none':state.cadence)],{type:'text/csv;charset=utf-8;'}));link.href=url;link.download=`${product.id}_${state.model}_${state.cadence}_h${state.horizon}_${state.start}_${state.asOf}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
async function load() {
  $('loading').hidden=false;$('workspace').hidden=true;$('error').hidden=true;$('export').disabled=true;
  try {
    const response=await fetch('/api/results',{signal:AbortSignal.timeout(30000)});
    if(!response.ok){const problem=await response.json();throw new Error(problem.message||'평가 데이터를 읽지 못했습니다.');}
    data=validateData(await response.json());
    if(data.schemaVersion!==2||data.meta.modelKind!=='comparison')throw new Error('학습모델 결과가 아직 없습니다. 가격자료 생성 후 .venv/bin/python -m agri_data.modeling을 실행해 주세요.');
    state.start=data.meta.evaluationStart;state.end=data.meta.evaluationEnd;
    state.asOf=state.end;$('asof').max=Math.round((dayTime(state.end)-dayTime(state.start))/86400000);$('asof').value=$('asof').max;
    for(const field of ['start','end']){$(field).value=state[field];$(field).min=data.meta.evaluationStart;$(field).max=data.meta.evaluationEnd;}
    $('period').value='all';
    $('evaluation-summary').textContent=`가락·강서 경매자료로 학습한 모델의 ${data.meta.evaluationStart} ~ ${data.meta.evaluationEnd} 순차 평가입니다. 현재 수정본을 사용한 실험이며, 전국 대표가격·공모전 공식 성능과는 구분됩니다.`;
    renderProducts();
    $('sources').innerHTML='';
    for(const [i,source] of (data.meta.sourceUrls??[]).entries()) {
      const value=typeof source==='string'?source:source.url;
      let parsed;try{parsed=new URL(value);}catch{continue;}
      if(!['https:','http:'].includes(parsed.protocol))continue;
      const a=document.createElement('a');a.href=parsed.href;a.target='_blank';a.rel='noreferrer noopener';a.textContent=`자료 출처 ${i+1} ↗`;$('sources').append(a);
    }
    $('updated').textContent=`데이터 생성 ${String(data.generatedAt??'').slice(0,10)} · 평가기간 ${data.meta.evaluationStart} ~ ${data.meta.evaluationEnd}`;
    $('workspace').hidden=false;render();
  } catch(error) {$('error-message').textContent=error.name==='TimeoutError'?'응답이 지연되었습니다. 로컬 서버 상태를 확인한 뒤 다시 불러오세요.':error.message;$('error').hidden=false;}
  finally{$('loading').hidden=true;}
}
$('retry').addEventListener('click',load);
new ResizeObserver(()=>{if(data&&!$('workspace').hidden)renderChart();}).observe($('chart-wrap'));
load();
