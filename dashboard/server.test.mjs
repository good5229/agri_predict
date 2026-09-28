import test from 'node:test';
import assert from 'node:assert/strict';
import {createDashboardServer} from './server.mjs';
test('server exposes only named dashboard assets and gives actionable missing-data response',async(t)=>{
  const server=createDashboardServer('/nonexistent/dashboard-fixture.json');
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
  t.after(()=>new Promise(resolve=>server.close(resolve)));
  const url=`http://127.0.0.1:${server.address().port}`;
  const home=await fetch(url);assert.equal(home.status,200);assert.match(await home.text(),/가격 예측, 실제와 나란히/);
  for(const p of ['/.env','/data/agri.sqlite','/../README.md','/%2e%2e/README.md','/server.mjs'])assert.equal((await fetch(url+p)).status,404);
  const missing=await fetch(url+'/api/results');assert.equal(missing.status,503);assert.match((await missing.json()).message,/agri_data.dashboard/);
  assert.equal((await fetch(url,{method:'POST'})).status,405);
  assert.match(home.headers.get('content-security-policy'),/frame-ancestors 'none'/);
});
