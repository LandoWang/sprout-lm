import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import test from 'node:test';
import { PDWorkers } from '../src/pd-workers.ts';
import { PDScheduler } from '../src/pd-scheduler.ts';
import { GenerationService } from '../src/service.ts';
import { createApiServer } from '../src/http.ts';
import { argmax } from '../src/runner.ts';

const directory=fileURLToPath(new URL('./fixtures/qwen-tiny/',import.meta.url));
const reference=JSON.parse(await readFile(join(directory,'reference.json'),'utf8'));
const options={directory,tokens:32,blockSize:2,prefillDevice:Number(process.env.PREFILL_DEVICE??0),decodeDevice:Number(process.env.DECODE_DEVICE??1),cuda:{graph:true}};

test('cross-process CUDA IPC remaps noncontiguous KV pages and preserves all independent logits', async () => {
  const workers=await PDWorkers.start(options);
  let maxError=0;
  try {
    assert.equal(new Set([process.pid,workers.prefill.pid,workers.decode.pid]).size,3);
    for(const [caseIndex,sample] of reference.cases.entries()) {
      const p=[3,1,8,2,9],d=[7,0,6,11,4],n=Math.ceil(sample.tokens.length/2),request=`case-${caseIndex}`;
      let result=await workers.forward('prefill',{request,from:0,tokens:sample.tokens,blocks:p.slice(0,n),output:'logits'});
      const compare=async(step:number) => {
        const bytes=await readFile(join(directory,sample.logits[step]));
        assert.equal(result.logits!.length,bytes.length/4);
        for(let i=0;i<result.logits!.length;i++) {
          const error=Math.abs(result.logits![i]-bytes.readFloatLE(i*4));maxError=Math.max(maxError,error);
          assert.ok(error<1e-5,`case ${caseIndex}, step ${step}, logit ${i}: ${error}`);
        }
        assert.equal(argmax(Float32Array.from(result.logits!)),sample.output[step]);
      };
      await compare(0);
      await assert.rejects(workers.transfer({request,sourcePages:p.slice(0,n),targetPages:[...d.slice(0,n-1),99],tokens:sample.tokens.length}),/physical page/);
      const before=await workers.stats();
      const copied=await workers.transfer({request,sourcePages:p.slice(0,n),targetPages:d.slice(0,n),tokens:sample.tokens.length});
      assert.equal(copied.bytes,sample.tokens.length*2*workers.prefill.spec.layers*workers.prefill.layout.width*4);
      const after=await workers.stats();
      assert.equal((after.decode as Record<string,unknown>).deviceForwards,(before.decode as Record<string,unknown>).deviceForwards,'KV transfer does not recompute the prompt');
      for(let step=1;step<sample.output.length;step++) {
        const from=sample.tokens.length+step-1;
        result=await workers.forward('decode',{request,from,tokens:[sample.output[step-1]],blocks:d.slice(0,Math.ceil((from+1)/2)),output:'logits'});
        await compare(step);
      }
    }
    console.log(JSON.stringify({pdMaxError:maxError,stats:await workers.stats()}));
  } finally { await workers.close(); }
  for(const pid of [workers.prefill.pid,workers.decode.pid]) assert.throws(()=>process.kill(pid,0),{code:'ESRCH'});
});

test('three-process PD serves concurrent JSON/SSE and returns all page references', async () => {
  const workers=await PDWorkers.start(options),scheduler=new PDScheduler(workers,{prefillChunkSize:2,traceLimit:1000});
  const service=new GenerationService(scheduler),server=createApiServer({service});
  server.listen(0,'127.0.0.1'); await once(server,'listening');
  const address=server.address();assert.ok(address&&typeof address==='object');const url=`http://127.0.0.1:${address.port}`;
  try {
    const results=await Promise.all(reference.cases.map(async(sample: {tokens:number[];output:number[]},i:number)=> {
      const response=await fetch(url+'/generate',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({prompt:sample.tokens,maxNewTokens:sample.output.length,stream:i===1})});
      assert.equal(response.status,200);
      if(i===0)assert.deepEqual((await response.json() as {output:number[]}).output,sample.output);
      else {
        const events=(await response.text()).split('\n').filter(l=>l.startsWith('data: ')).map(l=>JSON.parse(l.slice(6)));
        assert.deepEqual(events.filter(e=>e.type==='token').map(e=>e.token),sample.output);
        assert.equal(events.at(-1).type,'finished');
      }
      return true;
    }));
    assert.ok(results.every(Boolean));
    assert.equal(scheduler.snapshot().prefill.freePages,16);assert.equal(scheduler.snapshot().decode.freePages,16);
    assert.equal(scheduler.snapshot().transfers.count,2);
    assert.equal((await fetch(url+'/health')).status,200);
    console.log(JSON.stringify({pdState:scheduler.snapshot(),trace:scheduler.trace}));
  } finally { service.close(); server.close();server.closeAllConnections();await service.drained(); }
});

test('worker exit closes the API service and tears down its peer', async () => {
  const workers=await PDWorkers.start(options),scheduler=new PDScheduler(workers),service=new GenerationService(scheduler);
  try {
    const stream=service.generate({prompt:[1,7,3,9],maxNewTokens:8});await stream.next();
    process.kill(workers.decode.pid,'SIGKILL');
    const events=[];for await(const event of stream)events.push(event);
    assert.equal(events.at(-1)?.type,'error');assert.equal(service.isClosed,true);
    await scheduler.shutdown();
    assert.equal(scheduler.snapshot().prefill.freePages,16);assert.equal(scheduler.snapshot().decode.freePages,16);
  } finally { service.close();await service.drained(); }
  for(const pid of [workers.prefill.pid,workers.decode.pid])assert.throws(()=>process.kill(pid,0),{code:'ESRCH'});
});
