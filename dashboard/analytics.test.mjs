import test from 'node:test';
import assert from 'node:assert/strict';
import {selectRows,metrics,aggregate,periodKey,validateData,toCsv,scorePeriods,baselineComparison,significanceSummary} from './analytics.mjs';
test('scores use paired daily values, not displayed aggregated averages',()=>{
  const rows=[{date:'2015-01-01',actual:100,predicted:80},{date:'2015-01-02',actual:200,predicted:220},{date:'2015-01-03',actual:500,predicted:null}];
  assert.deepEqual(metrics(rows),{n:2,actualDays:3,mae:20,rmse:20,mape:15,mapeN:2});
  const monthly=aggregate(rows,'month');
  assert.equal(monthly[0].actual,150);assert.equal(monthly[0].predicted,150);assert.equal(monthly[0].count,2);
  assert.equal(aggregate([{date:'2015-01-01',actual:100,predicted:null}],'month')[0].predicted,null);
});
test('horizons and date windows are never pooled or forward-filled',()=>{
  const product={observations:[{date:'2015-01-01',actual:100},{date:'2015-01-03',actual:150}],predictions:[{target:'2015-01-03',horizon:1,predicted:110,origin:'2015-01-02',sourceDate:'2015-01-01'},{target:'2015-01-03',horizon:2,predicted:100,origin:'2015-01-01',sourceDate:'2015-01-01'}]};
  const rows=selectRows(product,{start:'2015-01-01',end:'2015-01-03',horizon:2});
  assert.equal(rows.length,2);assert.equal(rows[0].predicted,null);assert.equal(rows[1].predicted,100);
  assert.equal(metrics([]).mae,null);
});
test('week and quarter grouping use UTC calendar boundaries',()=>{
  assert.equal(periodKey('2015-01-01','week'),'2014-12-29');
  assert.equal(periodKey('2015-01-05','week'),'2015-01-05');
  assert.equal(periodKey('2015-03-31','quarter'),'2015-01-01');
  assert.equal(periodKey('2015-04-01','quarter'),'2015-04-01');
});
test('seasons cross the year boundary and scores are averages of daily losses',()=>{
  assert.equal(periodKey('2015-02-28','season'),'2014-12-01');
  assert.equal(periodKey('2015-03-01','season'),'2015-03-01');
  assert.equal(periodKey('2015-12-31','season'),'2015-12-01');
  const result=scorePeriods([{date:'2015-01-01',actual:100,predicted:80},{date:'2015-02-01',actual:100,predicted:120}],'season');
  assert.equal(result.length,1);assert.equal(result[0].mae,20);assert.equal(result[0].n,2);
});
test('daily replay filters unrevealed actuals and compares only common baseline dates',()=>{
  const product={observations:[{date:'2015-01-02',actual:100},{date:'2015-01-03',actual:110}],predictions:[{target:'2015-01-02',horizon:1,predicted:90}],runs:[{modelId:'ridge',cadence:'weekly',predictions:[{target:'2015-01-02',horizon:1,predicted:95,fitCutoff:'2015-01-01'},{target:'2015-01-03',horizon:1,predicted:200,fitCutoff:'2015-01-01'}]}]};
  const rows=selectRows(product,{start:'2015-01-01',end:'2015-12-31',asOf:'2015-01-02',horizon:1,model:'ridge',cadence:'weekly'});
  assert.equal(rows.length,1);assert.equal(rows[0].predicted,95);assert.equal(rows[0].fitCutoff,'2015-01-01');
  assert.equal(baselineComparison(rows).model.mae,5);assert.equal(baselineComparison(rows).baseline.mae,10);
});
function fixture(){return {schemaVersion:1,meta:{evaluationStart:'2015-01-01',evaluationEnd:'2015-12-31'},products:['cabbage','radish','onion','garlic','pepper'].map(id=>({id,name:id,observations:[{date:'2015-01-03',actual:100}],predictions:[{origin:'2015-01-01',target:'2015-01-03',sourceDate:'2014-12-31',horizon:2,predicted:99}]}))};}
test('invalid source dates, duplicate predictions and impossible dates fail closed',()=>{
  assert.doesNotThrow(()=>validateData(fixture()));
  const future=fixture();future.products[0].predictions[0].sourceDate='2015-01-02';assert.throws(()=>validateData(future));
  const duplicate=fixture();duplicate.products[0].predictions.push({...duplicate.products[0].predictions[0]});assert.throws(()=>validateData(duplicate));
  const bad=fixture();bad.products[0].observations[0].date='2015-02-30';assert.throws(()=>validateData(bad));
  const leaked=fixture();leaked.schemaVersion=2;leaked.products[0].runs=[{modelId:'ridge',cadence:'weekly',predictions:[{...leaked.products[0].predictions[0],fitCutoff:'2015-01-02'}]}];assert.throws(()=>validateData(leaked));
});
test('CSV retains unpaired observed dates and provenance',()=>{
  const csv=toCsv([{date:'2015-01-03',actual:100,predicted:null,origin:null,sourceDate:null}],1,'배추');
  assert.ok(csv.startsWith('\uFEFF'));assert.match(csv,/참조관측일/);assert.match(csv,/"배추","2015-01-03","","","1","100","",""/);
});
test('significance is tied to the selected comparison and never leaks full-period inference into replay',()=>{
  const meta={evaluationStart:'2017-01-01',evaluationEnd:'2018-12-20'};
  const state={product:'cabbage',model:'hist_gb',cadence:'daily',horizon:1,start:meta.evaluationStart,end:meta.evaluationEnd,asOf:meta.evaluationEnd};
  assert.match(significanceSummary(meta,state),/검토 미완료/);
  const available={status:'available',n:602,holmAdjustedPValue:.01,meanLossDifference:-2};
  meta.statisticalSignificance={comparisons:['absolute_error','absolute_percentage_error'].map(loss=>({productId:'cabbage',modelId:'hist_gb',cadence:'daily',horizon:1,loss,primary:{n:602,test:{...available}},sensitivity:{n:602,test:{...available}}}))};
  assert.match(significanceSummary(meta,state),/작은 오차의 통계적 근거/);
  assert.match(significanceSummary(meta,{...state,asOf:'2017-01-01'}),/끝까지 진행/);
  assert.doesNotMatch(significanceSummary(meta,{...state,start:'2018-01-01'}),/0\.0100/);
  assert.match(significanceSummary(meta,{...state,horizon:2}),/결과가 없습니다/);
  meta.statisticalSignificance.comparisons[1].primary.test.holmAdjustedPValue=.9;
  assert.match(significanceSummary(meta,state),/근거는 부족/);
  meta.statisticalSignificance.comparisons[0].primary.test.meanLossDifference=2;
  assert.match(significanceSummary(meta,state),/하나 이상의.*큰 오차/);
  meta.statisticalSignificance.comparisons[1].primary.test.holmAdjustedPValue=.01;
  assert.match(significanceSummary(meta,state),/개선과 악화가 엇갈려/);
  meta.statisticalSignificance.comparisons[0].primary={n:29,test:{status:'unavailable_insufficient_pairs',n:29}};
  assert.match(significanceSummary(meta,state),/판단 보류/);
});
