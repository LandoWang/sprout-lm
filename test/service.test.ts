import assert from 'node:assert/strict';
import { setImmediate as nextTurn } from 'node:timers/promises';
import test from 'node:test';
import { Engine } from '../src/engine.ts';
import { TinyTransformer, argmax, encode, type KVStore } from '../src/model.ts';
import { GenerationError, GenerationService, type GenerationEvent, type GenerationStream } from '../src/service.ts';
import { byteTokenizer } from '../src/tokenizer.ts';

class ObservedEngine extends Engine {
  reports: ReturnType<Engine['step']>[] = [];
  override step() {
    assert.ok(this.reports.length < 2000, 'scheduler must make progress');
    const report = super.step();
    this.reports.push(report);
    return report;
  }
}

async function collect(stream: GenerationStream) {
  const events: GenerationEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function output(events: GenerationEvent[]) {
  return events.filter(e => e.type === 'token').map(e => e.token);
}

function reference(model: TinyTransformer, prompt: number[], count: number) {
  const tokens: number[] = [];
  for (let i = 0; i < count; i++) tokens.push(argmax(model.forwardDense([...prompt, ...tokens])));
  return tokens;
}

test('shared pump streams concurrent and late requests through pressure without changing tokens', { timeout: 5000 }, async t => {
  const engine = new ObservedEngine({
    blockSize: 2, numBlocks: 5, maxBatchTokens: 6, prefillChunkSize: 2, recordTrace: false,
  });
  const service = new GenerationService(engine);
  t.after(() => service.close());
  const a = service.generate({ prompt: encode('aa'), maxNewTokens: 7 });
  const b = service.generate({ prompt: encode('bb'), maxNewTokens: 7 });
  const aEvents: GenerationEvent[] = [];
  aEvents.push((await a.next()).value!); // queued
  aEvents.push((await a.next()).value!); // first token; the pump must yield to this caller
  const c = service.generate({ prompt: encode('cc'), maxNewTokens: 7 });
  const results = await Promise.all([collect(a), collect(b), collect(c)]);
  results[0].unshift(...aEvents);
  for (const [i, stream] of [a, b, c].entries()) {
    assert.deepEqual(output(results[i]), reference(engine.model, encode(['aa', 'bb', 'cc'][i]), 7));
    assert.ok(results[i].every(e => e.id === stream.id));
    assert.deepEqual(results[i].filter(e => e.type === 'token').map(e => e.index), [0, 1, 2, 3, 4, 5, 6]);
    assert.deepEqual(results[i].filter(e => !['token', 'queued'].includes(e.type)), [{
      id: stream.id, type: 'finished', finishReason: 'length', usage: { promptTokens: 3, completionTokens: 7 },
    }]);
    assert.throws(() => engine.result(stream.id), /Unknown request/);
  }
  assert.ok(engine.reports.some(r => r.work.length > 1));
  assert.ok(engine.reports.some(r => r.work.some(w => w.phase === 'recompute')));
  assert.ok(engine.reports.every(r => r.work.reduce((sum, w) => sum + w.tokens, 0) <= 6));
  const events = engine.reports.flatMap(r => r.events);
  assert.ok(events.findIndex(e => e.id === c.id && e.type === 'admitted') < events.findIndex(e => e.id === b.id && e.type === 'finished'));
  assert.equal(engine.trace.length, 0, 'streaming does not need an unbounded debug trace');
  assert.equal(service.pendingCount, 0);
  assert.equal(engine.pending, false);
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
});

test('abort queued and active requests; releasing a shared prefix preserves the sibling', async t => {
  const engine = new Engine({ blockSize: 2, maxSequences: 2, recordTrace: false });
  const service = new GenerationService(engine);
  t.after(() => service.close());
  const abort = new AbortController();
  const prompt = encode('abc');
  const a = service.generate({ prompt, maxNewTokens: 8 }, { signal: abort.signal });
  await a.next();
  await a.next();
  const b = service.generate({ prompt, maxNewTokens: 5 });
  await b.next();
  const firstB = (await b.next()).value!;
  assert.ok(engine.cache.snapshot().pages.some(p => p.activeRefs === 2));
  const queuedAbort = new AbortController();
  const queued = service.generate({ prompt, maxNewTokens: 1 }, { signal: queuedAbort.signal });
  queuedAbort.abort();
  abort.abort();
  assert.equal((await collect(queued)).at(-1)?.type, 'cancelled');
  assert.equal((await collect(a)).at(-1)?.type, 'cancelled');
  assert.deepEqual(output([firstB, ...await collect(b)]), reference(engine.model, prompt, 5));
  assert.equal(service.cancel(a.id), false);
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
});

test('early iterator return, pre-aborted signal and service close release all requests', async () => {
  const engine = new Engine({ recordTrace: false });
  const service = new GenerationService(engine);
  const unopened = service.generate({ prompt: [1], maxNewTokens: 4 });
  await unopened.return?.();
  assert.equal(service.pendingCount, 0);
  const active = service.generate({ prompt: [1], maxNewTokens: 4 });
  for await (const event of active) if (event.type === 'token') break;
  assert.equal(service.pendingCount, 0);
  const aborted = service.generate({ prompt: [1], maxNewTokens: 1 }, { signal: AbortSignal.abort() });
  assert.deepEqual(await collect(aborted), [{ id: aborted.id, type: 'cancelled' }]);
  const pending = service.generate({ prompt: [1], maxNewTokens: 4 });
  service.close();
  assert.equal((await collect(pending)).at(-1)?.type, 'cancelled');
  assert.equal(engine.pending, false);
  assert.equal(engine.cache.snapshot().freePages, engine.options.numBlocks);
  assert.throws(() => service.generate({ prompt: [1], maxNewTokens: 1 }), { code: 'closed' });
});

test('slow consumer is bounded and cancelled while another stream completes', async t => {
  const engine = new Engine({ recordTrace: false });
  const service = new GenerationService(engine, { maxBufferedEvents: 2 });
  t.after(() => service.close());
  const slow = service.generate({ prompt: [1], maxNewTokens: 20 });
  const fast = service.generate({ prompt: [2], maxNewTokens: 8 });
  const fastEvents = await collect(fast);
  const slowEvents = await collect(slow);
  assert.equal(fastEvents.at(-1)?.type, 'finished');
  assert.deepEqual(output(fastEvents), reference(engine.model, [2], 8));
  assert.equal(slowEvents.length, 3, 'two buffered events plus one reserved terminal slot');
  assert.equal(slowEvents.at(-1)?.type, 'error');
  assert.equal((slowEvents.at(-1) as { code: string }).code, 'slow_consumer');
  assert.equal(service.pendingCount, 0);
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
});

test('invalid requests and admission limits fail without leaking queued work', async t => {
  const engine = new Engine({ recordTrace: false });
  const service = new GenerationService(engine, { maxPendingRequests: 1 });
  t.after(() => service.close());
  assert.throws(() => service.generate({ prompt: [-1], maxNewTokens: 1 }), { code: 'invalid_request' });
  const first = service.generate({ prompt: [1], maxNewTokens: 2 });
  assert.throws(() => service.generate({ prompt: [2], maxNewTokens: 1 }), { code: 'overloaded' });
  await collect(first);
  assert.equal(service.pendingCount, 0);
  assert.equal((await collect(service.generate({ prompt: [2], maxNewTokens: 1 }))).at(-1)?.type, 'finished');
  assert.throws(() => new GenerationService(new Engine(), { maxBufferedEvents: 0 }), /positive integer/);
});

test('failed model execution terminates all streams and retires partially written KV', async () => {
  class BrokenModel extends TinyTransformer {
    override forwardToken(token: number, position: number, kv: KVStore): never {
      super.forwardToken(token, position, kv);
      throw new Error('simulated forward failure');
    }
  }
  const engine = new Engine({ recordTrace: false }, new BrokenModel());
  const service = new GenerationService(engine);
  const a = service.generate({ prompt: [1], maxNewTokens: 3 });
  const b = service.generate({ prompt: [2], maxNewTokens: 3 });
  for (const events of await Promise.all([collect(a), collect(b)])) {
    assert.equal(events.at(-1)?.type, 'error');
    assert.equal((events.at(-1) as { code: string }).code, 'engine_error');
    assert.equal(events.filter(e => e.type === 'error').length, 1);
  }
  assert.equal(service.isClosed, true);
  assert.equal(engine.pending, false);
  assert.equal(engine.cache.snapshot().freePages, engine.options.numBlocks);
  assert.throws(() => service.generate({ prompt: [1], maxNewTokens: 1 }), GenerationError);
});

test('idle service does not keep scheduling empty steps; new arrivals restart it', async t => {
  const engine = new ObservedEngine({ recordTrace: false });
  const service = new GenerationService(engine);
  t.after(() => service.close());
  await collect(service.generate({ prompt: [1], maxNewTokens: 1 }));
  const count = engine.reports.length;
  for (let i = 0; i < 3; i++) await nextTurn();
  assert.equal(engine.reports.length, count);
  await collect(service.generate({ prompt: [2], maxNewTokens: 1 }));
  assert.equal(engine.reports.length, count + 1);
});

test('byte tokenizer preserves UTF-8 across token boundaries and ignores BOS', () => {
  const decoder = byteTokenizer.decoder();
  const parts = byteTokenizer.encode('你好🌱').map(token => decoder.push(token));
  assert.equal(parts.join('') + decoder.finish(), '你好🌱');
  assert.equal(parts[1], '', 'a leading UTF-8 byte is not a complete character');
  const incomplete = byteTokenizer.decoder();
  assert.equal(incomplete.push(0xe4), '');
  assert.equal(incomplete.finish(), '\ufffd');
});
