import assert from 'node:assert/strict';
import { once } from 'node:events';
import { get } from 'node:http';
import { mkdtemp, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test, { type TestContext } from 'node:test';
import { readEventLog, type EventLog } from '../src/event-log.ts';
import { analyzeRun } from '../src/trace-analysis.ts';
import { createObservatoryServer } from '../src/observatory-server.ts';
import type { ObservationEvent } from '../src/observation.ts';

function fixtures() {
  let seq = 0;
  const base = 9876543210987654321n; // Must subtract before conversion to Number.
  const e = (name: string, ms: number, requestId = 'A', data?: Record<string, unknown>, planId?: string, source = 'coordinator'): ObservationEvent => ({
    version: 1, runId: 'run-test', name, requestId, data, planId, source, pid: source === 'coordinator' ? 1 : 2,
    seq: ++seq, timeNs: (base+BigInt(Math.round(ms*1e6))).toString(), wallTimeMs: 1700000000000+ms,
  });
  const plan = (id: string, req: string, sent: number, start: number, end: number, received: number, phase = 'decode') => [
    e('plan.sent',sent,req,{ phase, worker: phase === 'prefill' ? 'prefill' : 'decode', tokens: 1 },id),
    e('plan.started',start,req,{},id,phase === 'prefill' ? 'prefill' : 'decode'),
    e('plan.completed',end,req,phase === 'transfer' ? { cudaEventMs: .1 } : {},id,phase === 'prefill' ? 'prefill' : 'decode'),
    e('plan.completion_received',received,req,{ phase },id),
  ];
  const log = (events: ObservationEvent[]): EventLog => ({ runId:'run-test',events:events.sort((a,b)=>BigInt(a.timeNs)<BigInt(b.timeNs)?-1:1),diagnostics:{
    cleanShutdown:false,rotatedFiles:0,retainedFiles:1,invalidLines:0,partialLines:0,unavailableFiles:0,duplicateEvents:0,sequenceGaps:0,
    truncated:false,evictedEvents:0,bytesRead:0,eventLimit:100000,byteLimit:67108864,losses:[],manifestUpdatedAt:'',capturedAt:'',
  } });
  return { e, plan, log };
}
async function directory(t: TestContext) {
  const root = await mkdtemp(join(tmpdir(),'sprout-viewer-'));
  t.after(()=>rm(root,{ recursive:true,force:true }));
  await writeFile(join(root,'manifest.json'),JSON.stringify({ version:1,runId:'run-test',cleanShutdown:false }));
  return root;
}

test('TTFT uses token index zero, distinguishes HTTP and handoff, and preserves nanosecond precision',()=>{
  const { e,plan,log }=fixtures();
  const result=analyzeRun(log([
    e('request.received',0),...plan('p','A',1,2,10,11,'prefill'),
    e('token.published',12,'A',{ index:0 }),e('http.token_written',14,'A',{ index:0 }),
    ...plan('x','A',12.5,13,14,15,'transfer'),...plan('d','A',16,17,23,24),
    e('token.published',25,'A',{ index:1 }),e('http.token_written',28,'A',{ index:1 }),
    e('generation.finished',29,'A',{ completionTokens:2 }),
  ]));
  const r=result.requests[0];
  assert.equal(r.ttftMs,12);assert.equal(r.httpTtftMs,14);assert.equal(r.firstToSecondMs,13);
  assert.equal(r.generationMs,29);assert.equal(r.itlP95,13);assert.equal(r.writtenItl[0].ms,14);
  assert.equal(r.ttftSegments.find(s=>s.kind==='transfer')!.ms,0);
  assert.equal(r.kvCudaMs,.1);
  const x=result.plans.find(p=>p.id==='x')!;
  assert.equal(x.dispatchMs,.5);assert.equal(x.executionMs,1);assert.equal(x.returnMs,1);
});

test('missing token zero does not masquerade as TTFT; missing tokens do not produce invented adjacent gaps',()=>{
  const { e,log }=fixtures();
  const r=analyzeRun(log([e('request.received',0),e('token.published',10,'A',{ index:1 }),e('token.published',30,'A',{ index:3 })])).requests[0];
  assert.equal(r.ttftMs,null);assert.equal(r.firstToSecondMs,null);assert.deepEqual(r.itl,[]);assert.equal(r.status,'incomplete');
});

test('overlapping admission and reason intervals are unioned; incomplete intervals are not counted as completed work',()=>{
  const { e,log }=fixtures();
  const a=analyzeRun(log([
    e('request.received',0),e('scheduler.queued',1),e('request.waiting',2,'A',{ reason:'max_sequences' }),
    e('request.wait_ended',8,'A',{ reason:'max_sequences' }),e('scheduler.admitted',10),
    e('plan.started',11,'A',{},'incomplete','prefill'),e('request.cancelled',12),
  ]));
  assert.equal(a.requests[0].breakdown.find(s=>s.kind==='wait')!.ms,9);
  assert.equal(a.requests[0].workerMs,0);assert.equal(a.requests[0].incompletePlans,1);
  assert.equal(a.requests[0].status,'cancelled');assert.ok(a.spans.some(s=>s.incomplete));
});

test('Worker gaps include other requests, not the apparent idle interval in one request',()=>{
  const {e,plan,log}=fixtures();
  const a=analyzeRun(log([e('request.received',0),...plan('a1','A',1,2,10,11),
    ...plan('b1','B',11,12,20,21),...plan('a2','A',22,23,30,31)]));
  assert.deepEqual(a.gaps.map(g=>g.ms),[3,2]);
  assert.equal(a.gaps[0].nextRequestId,'A');
});

test('cancelled in-flight completion and page release remain visible; reversed timestamps are diagnosed',()=>{
  const {e,plan,log}=fixtures();
  const events=plan('p','A',1,2,10,11);events[3].data={ phase:'decode',discarded:true };
  const a=analyzeRun(log([e('request.received',0),...events,e('request.cancelled',5),e('request.resources_released',12),
    ...plan('bad','B',20,19,18,21)]));
  assert.equal(a.requests.find(r=>r.id==='A')!.generationMs,5);
  assert.equal(a.requests.find(r=>r.id==='A')!.observedEnd,12);
  assert.equal(a.plans.find(p=>p.id==='p')!.discarded,true);
  assert.equal(a.plans.find(p=>p.id==='bad')!.executionMs,null);assert.ok(a.diagnostics.invalidPairs>=2);
});

test('reader validates timestamps, deduplicates, tolerates partial append, detects sequence loss and bounds reads',async t=>{
  const root=await directory(t),{e}=fixtures();
  const first=e('request.received',0),second=e('token.published',1,'A',{index:0});second.seq=4;
  const bad={...second,timeNs:'invalid'};
  const contents=[first,first,bad,second].map(e=>JSON.stringify(e)).join('\n')+'\n{"version":';
  await writeFile(join(root,'events-00000000.ndjson'),contents);
  const log=await readEventLog(root);
  assert.equal(log.events.length,2);assert.equal(log.diagnostics.invalidLines,1);assert.equal(log.diagnostics.duplicateEvents,1);
  assert.equal(log.diagnostics.partialLines,1);assert.equal(log.diagnostics.sequenceGaps,2);
  const capped=await readEventLog(root,1);assert.equal(capped.events.length,1);assert.equal(capped.diagnostics.truncated,true);
  assert.equal(capped.events[0].name,'token.published');assert.equal(capped.diagnostics.evictedEvents,1);
  const byteCapped=await readEventLog(root,100000,10);assert.equal(byteCapped.diagnostics.bytesRead,10);assert.equal(byteCapped.diagnostics.truncated,true);
});

test('reader skips linked event files and rejects malformed manifests',async t=>{
  const root=await directory(t);await symlink(join(root,'manifest.json'),join(root,'events-00000000.ndjson'));
  assert.equal((await readEventLog(root)).diagnostics.unavailableFiles,1);
  await writeFile(join(root,'manifest.json'),'{}');await assert.rejects(readEventLog(root),/manifest/);
});

test('read-only HTTP viewer serves the saved trace and rejects traversal, mutations and cross-origin access',async t=>{
  const root=await directory(t),{e}=fixtures();
  await writeFile(join(root,'events-00000000.ndjson'),JSON.stringify(e('request.received',0))+'\n');
  const server=createObservatoryServer([{directory:root}]);server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(()=>{server.closeAllConnections();server.close();});
  const address=server.address();assert.ok(address&&typeof address==='object');const url=`http://127.0.0.1:${address.port}`;
  const runs=await (await fetch(url+'/api/runs')).json();assert.equal(runs.runs.length,1);
  const result=await (await fetch(url+'/api/run?run='+runs.runs[0].key)).json();assert.equal(result.requests[0].id,'A');
  assert.equal((await fetch(url+'/api/run?run=../../etc')).status,404);
  assert.equal((await fetch(url+'/api/runs',{method:'POST'})).status,405);
  assert.equal((await fetch(url+'/api/runs',{headers:{Origin:'https://elsewhere.example'}})).status,403);
  const status=await new Promise<number | undefined>((resolve,reject)=>{
    get(url+'/api/runs',{headers:{Host:'elsewhere.example'}},res=>{res.resume();resolve(res.statusCode);}).on('error',reject);
  });
  assert.equal(status,403);
  const page=await fetch(url+'/');assert.equal(page.status,200);assert.match(page.headers.get('content-security-policy')!,/script-src 'self'/);
  assert.match(await page.text(),/执行时间线/);
});

test('real A100 snapshot has 145 complete plans and separate prefill, handoff, decode timing',async()=>{
  const directory=new URL('../snapshot/2026-10-20-request-events/raw-run/',import.meta.url);
  const {fileURLToPath}=await import('node:url');
  const a=analyzeRun(await readEventLog(fileURLToPath(directory))),r=a.requests[0];
  assert.equal(r.eventCount,1005);assert.equal(r.planCount,145);assert.equal(r.incompletePlans,0);
  assert.equal(r.observedTokens,136);assert.equal(r.tokens.length,136);assert.equal(r.itl.length,135);
  assert.ok(Math.abs(r.ttftMs!-2130.936242)<1e-6);assert.ok(Math.abs(r.httpTtftMs!-2131.313864)<1e-6);
  assert.equal(a.diagnostics.invalidPairs,0);assert.equal(a.diagnostics.sequenceGaps,0);
});
