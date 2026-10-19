import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { createRuntime } from '../src/runtime.ts';
import { createApiServer } from '../src/http.ts';

const directory=process.argv[2];if(!directory)throw new Error('Usage: node scripts/benchmark-pd.ts MODEL_DIR');
const loadStart=performance.now();
const runtime=await createRuntime({backend:'cuda-pd',modelDirectory:directory,maxContextTokens:256,
  cuda:{graph:true,reserveMiB:4096},pd:{prefillDevice:Number(process.env.PREFILL_DEVICE??0),decodeDevice:Number(process.env.DECODE_DEVICE??1),traceLimit:10000}});
assert.ok(runtime.scheduler);
const scheduler=runtime.scheduler,loadMs=performance.now()-loadStart,server=createApiServer(runtime);
server.listen(0,'127.0.0.1');await once(server,'listening');
const address=server.address();assert.ok(address&&typeof address==='object');const url=`http://127.0.0.1:${address.port}`;
const prompt='从1开始逐行数到100，每行只写一个数字。';
const health: number[]=[];let monitoring=true;
const monitor=(async()=>{while(monitoring){const start=performance.now();const r=await fetch(url+'/health');assert.equal(r.status,200);health.push(performance.now()-start);await delay(25);}})();
void monitor.catch(()=>{});
const summary=(values:number[])=>{const sorted=[...values].sort((a,b)=>a-b);return {count:sorted.length,meanMs:values.reduce((a,b)=>a+b,0)/values.length,p95Ms:sorted[Math.max(0,Math.ceil(sorted.length*.95)-1)],maxMs:sorted.at(-1)};};
async function request(onToken?:()=>void,messages=prompt,maxNewTokens=32,onQueued?:(id:string)=>void) {
  const start=performance.now();
  const response=await fetch(url+'/generate',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({messages:[{role:'user',content:messages}],maxNewTokens,stream:true})});
  assert.equal(response.status,200);const id=response.headers.get('x-request-id')!;
  const tokens:number[]=[],times:number[]=[];let terminal='',text='',buffer='';const decoder=new TextDecoder();
  for await(const chunk of response.body!) {
    buffer+=decoder.decode(chunk,{stream:true});let end:number;
    while((end=buffer.indexOf('\n\n'))>=0) {
      const frame=buffer.slice(0,end);buffer=buffer.slice(end+2);
      const line=frame.split('\n').find(l=>l.startsWith('data: '));if(!line)continue;
      const event=JSON.parse(line.slice(6));
      if(event.type==='queued')onQueued?.(id);
      else if(event.type==='token'){tokens.push(event.token);times.push(performance.now()-start);text+=event.textDelta;if(tokens.length===1)onToken?.();}
      else {terminal=event.type;if(event.type==='error')throw new Error(event.message);}
    }
  }
  return {id,output:tokens,text,terminal,firstTokenMs:times[0],totalMs:performance.now()-start,
    tokenTimesMs:times,interToken:times.length>1?summary(times.slice(1).map((t,i)=>t-times[i])):undefined};
}
async function waitFor(predicate:()=>boolean){const end=Date.now()+10000;while(!predicate()){assert.ok(Date.now()<end,'state wait timeout');await delay(5);}}
try {
  const cold=await request();assert.equal(cold.terminal,'finished');
  const steady=[];for(let i=0;i<3;i++){const r=await request();assert.deepEqual(r.output,cold.output);steady.push(r);}
  if(process.env.BASELINE_REPORT){const baseline=JSON.parse(await readFile(process.env.BASELINE_REPORT,'utf8'));assert.deepEqual(cold.output,baseline.runs[0].output);}
  let second:ReturnType<typeof request>|undefined;
  const first=await request(()=>{second=request();});const sibling=await second!;
  assert.deepEqual(first.output,cold.output);assert.deepEqual(sibling.output,cold.output);
  const p=scheduler.trace.filter(t=>t.id===sibling.id&&t.phase==='prefill');
  const d=scheduler.trace.filter(t=>t.id===first.id&&t.phase==='decode');
  const overlaps=d.filter(a=>p.some(b=>Math.min(a.result.endedAt,b.result.endedAt)>Math.max(a.result.startedAt,b.result.startedAt))).length;
  assert.ok(overlaps>0,'real P and D worker execution must overlap');
  let cancelled:ReturnType<typeof request>|undefined,cancelAction:Promise<void>|undefined;
  const survivor=await request(()=>{
    cancelled=request(undefined,'请详细介绍从一到一百的每一个数字，并解释它们的数学特点。'.repeat(3),8,id=>{
      cancelAction=(async()=>{
        await waitFor(()=>scheduler.snapshot().requests.some(r=>r.id===id&&r.inFlight));
        const r=await fetch(url+'/requests/'+id,{method:'DELETE'});assert.equal(r.status,200);
      })();
    });
  });
  await cancelAction;const cancellation=await cancelled!;
  assert.equal(cancellation.terminal,'cancelled');assert.deepEqual(survivor.output,cold.output);
  await waitFor(()=>!scheduler.pending);
  const state=scheduler.snapshot();assert.equal(state.prefill.freePages,state.prefill.totalPages);assert.equal(state.decode.freePages,state.decode.totalPages);
  monitoring=false;await monitor;
  const report={model:runtime.name,loadMs,cold,steady,concurrent:{first,sibling,overlappingDecodeSteps:overlaps},cancellation,survivor,
    health:summary(health),state,trace:scheduler.trace,
    limitations:['HTTP/SSE loopback timing, includes IPC and client overhead; not directly comparable to Engine-only benchmarks.',
      'No tensor batching, PD prefix cache or preemption; destination capacity reserved at admission.',
      'Worker time intervals use same-host wall-clock milliseconds; not GPU utilization.']};
  if(process.env.REPORT_PATH){const path=resolve(process.env.REPORT_PATH);await mkdir(dirname(path),{recursive:true});await writeFile(path,JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify({loadMs,steady:steady.map(r=>({firstTokenMs:r.firstTokenMs,interToken:r.interToken})),health:report.health,overlaps,state}));
} finally {monitoring=false;await monitor.catch(()=>{});server.close();server.closeAllConnections();await runtime.close();}
