import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate as nextTurn } from 'node:timers/promises';
import { PDScheduler } from '../src/pd-scheduler.ts';
import { GenerationService, type GenerationStream, type GenerationEvent } from '../src/service.ts';
import { TinyTransformer } from '../src/model.ts';
import type { ForwardCommand, PDExecutors, TransferCommand, WorkerInfo, WorkerResult, WorkerRole } from '../src/pd-protocol.ts';

class HeldWorkers implements PDExecutors {
  prefill: WorkerInfo; decode: WorkerInfo;
  calls: Array<{ role: WorkerRole; op: 'forward' | 'transfer'; value: ForwardCommand | TransferCommand;
    resolve: (r: WorkerResult) => void; reject: (e: Error) => void }> = [];
  private failure?: (error: Error) => void;
  private stop?: () => void;
  closing = false;
  constructor(pages = 8) {
    const model = new TinyTransformer();
    const common = { spec: { ...model.spec, kvDtype: 'float32' as const, maxContextLength: pages * 2 },
      layout: { numBlocks: pages, blockSize: 2, layers: model.spec.layers, width: model.spec.kvHeads * model.spec.headDim }, stats: {} };
    this.prefill = { ...common, role: 'prefill', device: 0, pid: 101 };
    this.decode = { ...common, role: 'decode', device: 1, pid: 102 };
  }
  forward(role: WorkerRole, value: ForwardCommand) {
    return new Promise<WorkerResult>((resolve,reject) => this.calls.push({ role, op:'forward', value, resolve, reject }));
  }
  transfer(value: TransferCommand) {
    return new Promise<WorkerResult>((resolve,reject) => this.calls.push({ role:'decode', op:'transfer', value, resolve, reject }));
  }
  complete(role: WorkerRole, token = 7): void {
    const i = this.calls.findIndex(c => c.role === role); assert.ok(i >= 0, `No ${role} work`);
    const call = this.calls.splice(i,1)[0];
    const bytes = call.op === 'transfer' ? (call.value as TransferCommand).tokens * this.prefill.spec.layers * 2 * this.prefill.layout.width * 4 : undefined;
    call.resolve({ request: call.value.request, token, bytes, eventMs: bytes === undefined ? undefined : 1, startedAt: 1, endedAt: 2 });
  }
  onFailure(callback: (error: Error) => void) { this.failure = callback; }
  fail() { this.failure!(new Error('worker lost')); }
  close(): Promise<void> { this.closing = true; return new Promise(resolve => { this.stop = resolve; }); }
  exited() { for (const call of this.calls.splice(0)) call.reject(new Error('worker exited')); this.stop!(); }
}
async function settle() { for (let i=0;i<3;i++) await nextTurn(); }
async function collect(stream: GenerationStream) { const result: GenerationEvent[]=[]; for await(const e of stream)result.push(e); return result; }

test('PD reserves destination capacity, overlaps workers, and holds both page sets until transfer acknowledgement', async () => {
  const workers = new HeldWorkers(4), scheduler = new PDScheduler(workers, { prefillChunkSize: 2 });
  scheduler.add({ id:'A',prompt:[1,2,3],maxNewTokens:3 });
  scheduler.add({ id:'B',prompt:[4,5,6],maxNewTokens:3 });
  let step = scheduler.step();
  assert.equal(scheduler.snapshot().decode.freePages,1); // A reserves three pages; B must wait.
  assert.equal(workers.calls.length,1);
  workers.complete('prefill'); await step;
  step=scheduler.step(); workers.complete('prefill',9);
  assert.equal((await step).events[0].type,'token');
  step=scheduler.step();
  assert.equal(workers.calls[0].op,'transfer');
  assert.equal(scheduler.snapshot().prefill.freePages,2);
  assert.equal(scheduler.snapshot().decode.freePages,1);
  workers.complete('decode'); await step;
  assert.equal(scheduler.snapshot().prefill.freePages,4);
  step=scheduler.step(); workers.complete('decode',10); await step;
  step=scheduler.step(); workers.complete('decode',11); assert.equal((await step).events.at(-1)?.type,'finished');
  assert.equal(scheduler.snapshot().decode.freePages,4);
  step=scheduler.step(); assert.equal(workers.calls[0].value.request,'B');
  scheduler.cancel('B'); await step;
  assert.equal(scheduler.snapshot().prefill.freePages,2,'cancel cannot recycle pages still written by GPU');
  workers.complete('prefill'); await settle();
  assert.equal(scheduler.snapshot().prefill.freePages,4);
  assert.equal(scheduler.snapshot().decode.freePages,4);
});

test('PD keeps decode running while another request has an unfinished prefill chunk', async () => {
  const workers=new HeldWorkers(), scheduler=new PDScheduler(workers,{prefillChunkSize:2});
  scheduler.add({id:'A',prompt:[1],maxNewTokens:4});
  let step=scheduler.step(); workers.complete('prefill'); await step;
  scheduler.add({id:'B',prompt:[1,2,3,4],maxNewTokens:2});
  step=scheduler.step();
  assert.deepEqual(workers.calls.map(c=>c.role),['decode','prefill']);
  workers.complete('decode'); await step; // B is deliberately still in flight.
  step=scheduler.step();
  assert.ok(workers.calls.some(c=>c.role==='decode' && c.op==='forward'));
  workers.complete('decode'); assert.equal((await step).events[0].type,'token');
  scheduler.cancel('A'); scheduler.cancel('B'); workers.complete('prefill'); await settle();
  assert.equal(scheduler.snapshot().prefill.freePages,8); assert.equal(scheduler.snapshot().decode.freePages,8);
});

test('cancellation during KV transfer suppresses late results and delays page reuse', async () => {
  const workers=new HeldWorkers(), scheduler=new PDScheduler(workers), service=new GenerationService(scheduler);
  const a=service.generate({prompt:[1,2],maxNewTokens:3}); await a.next(); await settle();
  workers.complete('prefill'); assert.equal((await a.next()).value?.type,'token'); await settle();
  assert.equal(workers.calls[0].op,'transfer');
  service.cancel(a.id); assert.equal((await collect(a)).at(-1)?.type,'cancelled');
  assert.equal(scheduler.snapshot().prefill.freePages,7);
  assert.equal(scheduler.snapshot().decode.freePages,6);
  workers.complete('decode'); await settle();
  assert.equal(scheduler.snapshot().prefill.freePages,8); assert.equal(scheduler.snapshot().decode.freePages,8);
  assert.equal(scheduler.snapshot().requests.length,0);
});

test('worker loss retires the service and retains page ownership until worker teardown finishes', async () => {
  const workers=new HeldWorkers(), scheduler=new PDScheduler(workers), service=new GenerationService(scheduler);
  const a=service.generate({prompt:[1,2],maxNewTokens:3}); await a.next(); await settle();
  workers.fail(); const events=await collect(a);
  assert.equal(events.at(-1)?.type,'error'); assert.equal(service.isClosed,true);
  assert.equal(workers.closing,true); assert.equal(scheduler.snapshot().prefill.freePages,7);
  assert.throws(()=>service.generate({prompt:[1],maxNewTokens:1}),{code:'closed'});
  workers.exited(); await service.drained();
  assert.equal(scheduler.snapshot().prefill.freePages,8); assert.equal(scheduler.snapshot().decode.freePages,8);
});

test('shutdown during prefill waits for teardown before releasing pages', async () => {
  const workers=new HeldWorkers(), scheduler=new PDScheduler(workers), service=new GenerationService(scheduler);
  const a=service.generate({prompt:[1,2],maxNewTokens:2}); await a.next(); await settle();
  service.close(); await settle();
  assert.equal((await collect(a)).at(-1)?.type,'cancelled');
  assert.equal(scheduler.snapshot().prefill.freePages,7);
  workers.exited(); await service.drained();
  assert.equal(scheduler.snapshot().prefill.freePages,8); assert.equal(scheduler.snapshot().decode.freePages,8);
});

test('a terminal first token releases both reservations without transferring KV', async () => {
  for (const eos of [false,true]) {
    const workers=new HeldWorkers();if(eos)workers.prefill.spec.eosTokenIds=[7];
    const scheduler=new PDScheduler(workers);
    scheduler.add({id:'terminal',prompt:[1,2],maxNewTokens:eos?4:1});
    const step=scheduler.step();workers.complete('prefill',7);
    const last=(await step).events.at(-1);assert.equal(last?.type,'finished');
    if(last?.type==='finished')assert.equal(last.finishReason,eos?'eos':'length');
    assert.equal(scheduler.snapshot().transfers.count,0);assert.equal(workers.calls.length,0);
    assert.equal(scheduler.snapshot().prefill.freePages,8);assert.equal(scheduler.snapshot().decode.freePages,8);
  }
});

test('a mismatched completion cannot publish tokens or free a still-live pool', async () => {
  const workers=new HeldWorkers(),scheduler=new PDScheduler(workers);
  scheduler.add({id:'current',prompt:[1],maxNewTokens:2});
  const step=scheduler.step(),call=workers.calls.shift()!;
  call.resolve({request:'obsolete',token:7,startedAt:1,endedAt:2});
  await assert.rejects(step,/identity mismatch/);
  assert.equal(scheduler.failed,true);assert.equal(scheduler.snapshot().prefill.freePages,7);
  workers.exited();await scheduler.shutdown();assert.equal(scheduler.snapshot().prefill.freePages,8);
});
