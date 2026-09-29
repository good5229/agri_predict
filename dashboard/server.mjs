import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
const here = path.dirname(fileURLToPath(import.meta.url));
const publicFiles = new Map([['/', ['index.html','text/html']],['/styles.css',['styles.css','text/css']],['/app.mjs',['app.mjs','text/javascript']],['/analytics.mjs',['analytics.mjs','text/javascript']]]);
export function createDashboardServer(dataPath = path.resolve(here,'../outputs/dashboard.json'), significancePath = path.join(path.dirname(dataPath),'significance.json')) {
  return http.createServer(async (req,res) => {
    const send = (status, type, body) => {res.writeHead(status, {'Content-Type':`${type}; charset=utf-8`, 'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"});res.end(req.method==='HEAD'?undefined:body);};
    if (!['GET','HEAD'].includes(req.method)) {send(405,'text/plain','Method not allowed');return;}
    let pathname;
    try {pathname = new URL(req.url,'http://localhost').pathname;} catch {send(400,'text/plain','Bad request');return;}
    if (pathname === '/api/results') {
      try {
        const payload = await readFile(dataPath);
        const result=JSON.parse(payload);
        // Never reuse a significance report after the prediction file changes.
        delete result.meta.statisticalSignificance;
        result.meta.statisticalSignificanceStatus='missing';
        try {
          const report=JSON.parse(await readFile(significancePath,'utf8'));
          const matches=report.schemaVersion===1&&report.protocol?.inputSha256===createHash('sha256').update(payload).digest('hex')
            &&report.protocol.evaluationStart===result.meta.evaluationStart&&report.protocol.evaluationEnd===result.meta.evaluationEnd&&Array.isArray(report.comparisons);
          result.meta.statisticalSignificanceStatus=matches?'verified':'mismatch';
          if(matches)result.meta.statisticalSignificance=report;
        } catch { /* A missing or unreadable report leaves the completed assessment pending. */ }
        send(200,'application/json',JSON.stringify(result));
      } catch {send(503,'application/json',JSON.stringify({error:'data_unavailable',message:'학습·평가 데이터가 없습니다. README의 순서대로 python3 -m agri_data.dashboard로 가격자료를 생성한 후 .venv/bin/python -m agri_data.modeling을 실행해 주세요.'}));}
      return;
    }
    const file = publicFiles.get(pathname);
    if (!file) {send(404,'text/plain','Not found');return;}
    try {send(200,file[1],await readFile(path.join(here,file[0])));} catch {send(500,'text/plain','대시보드 파일을 읽지 못했습니다. 서버 설치를 확인해 주세요.');}
  });
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4173);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be an integer from 1 to 65535');
  const server = createDashboardServer();
  server.on('error',error=>{console.error(`대시보드 서버를 시작하지 못했습니다: ${error.code || error.message}`);process.exitCode=1;});
  server.listen(port,'127.0.0.1',()=>console.log(`대시보드: http://127.0.0.1:${port}`));
}
