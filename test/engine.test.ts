import assert from 'node:assert/strict';
import test from 'node:test';
import { Engine } from '../src/engine.ts';
import { KVCache } from '../src/kv-cache.ts';
import { TinyTransformer, argmax, encode } from '../src/model.ts';

function drain(engine: Engine, limit = 2000) {
  const reports: ReturnType<Engine['step']>[] = [];
  while (engine.pending) {
    assert.ok(reports.length < limit, 'scheduler must make progress');
    const report = engine.step();
    assert.ok(report.work.reduce((sum, w) => sum + w.tokens, 0) <= engine.options.maxBatchTokens);
    assert.ok(report.memory.pages.every(p => p.refs >= 0 && p.activeRefs >= 0));
    reports.push(report);
  }
  return reports;
}

function reference(model: TinyTransformer, prompt: number[], count: number): number[] {
  const output: number[] = [];
  for (let i = 0; i < count; i++) output.push(argmax(model.forwardDense([...prompt, ...output])));
  return output;
}

test('paged attention matches independent dense recomputation at every position', () => {
  const model = new TinyTransformer();
  const cache = new KVCache({ numBlocks: 12, blockSize: 2, layers: model.layers, width: model.width });
  const seq = cache.create();
  const blocker = cache.create();
  assert.ok(blocker.reserveToken()); // Physical page 0 is unavailable to our sequence.
  const tokens = encode('paging!');
  for (let p = 0; p < tokens.length; p++) {
    assert.ok(seq.reserveToken());
    const actual = model.forwardToken(tokens[p], p, seq);
    seq.length++;
    const expected = model.forwardDense(tokens.slice(0, p + 1));
    for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i] - expected[i]) < 1e-12);
    if (p === 1) blocker.release(); // Next logical page can land at physical page 0.
  }
  assert.deepEqual(seq.blocks.slice(0, 2), [1, 0]);
  seq.release();
  assert.equal(cache.snapshot().freePages, 12);
});

test('continuous batches admit a late request and mix decode with chunked prefill', () => {
  const engine = new Engine({ maxBatchTokens: 6, prefillChunkSize: 3, maxSequences: 3 });
  const prompts = [encode('a'), encode('longer prompt'), encode('late')];
  engine.add({ id: 'A', prompt: prompts[0], maxNewTokens: 6 });
  engine.add({ id: 'B', prompt: prompts[1], maxNewTokens: 3 });
  engine.step();
  engine.add({ id: 'C', prompt: prompts[2], maxNewTokens: 2 });
  const reports = drain(engine);
  assert.ok(reports.some(r => r.work.some(w => w.phase === 'decode') && r.work.some(w => w.phase === 'prefill')));
  for (const [i, id] of ['A', 'B', 'C'].entries()) {
    assert.equal(engine.result(id).status, 'finished');
    assert.deepEqual(engine.result(id).output, reference(engine.model, prompts[i], [6, 3, 2][i]));
  }
  assert.ok(engine.trace.findIndex(e => e.id === 'C' && e.type === 'admitted') < engine.trace.findIndex(e => e.id === 'B' && e.type === 'finished'));
});

test('prefix sharing reuses physical pages and leaves the final prompt token for logits', () => {
  const engine = new Engine({ blockSize: 2, prefillChunkSize: 8 });
  const prompt = encode('abc'); // Exactly two full pages.
  engine.add({ id: 'A', prompt, maxNewTokens: 4 });
  engine.step();
  const aPages = engine.result('A').blockTable;
  engine.add({ id: 'B', prompt, maxNewTokens: 4 });
  const report = engine.step();
  assert.equal(engine.result('B').prefixHits, 2);
  assert.equal(engine.result('B').blockTable[0], aPages[0]);
  assert.equal(report.memory.pages[aPages[0]].activeRefs, 2);
  drain(engine);
  assert.deepEqual(engine.result('A').output, engine.result('B').output);
  assert.deepEqual(engine.result('B').output, reference(engine.model, prompt, 4));
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
  engine.cache.clearPrefixes();
  assert.equal(engine.cache.snapshot().freePages, engine.options.numBlocks);
});

test('same suffix under a different preceding context must not hit the prefix cache', () => {
  const engine = new Engine({ blockSize: 2, prefillChunkSize: 8 });
  engine.add({ id: 'A', prompt: [1, 2, 3, 4, 5], maxNewTokens: 1 });
  drain(engine);
  engine.add({ id: 'B', prompt: [9, 8, 3, 4, 5], maxNewTokens: 1 });
  drain(engine);
  assert.equal(engine.result('B').prefixHits, 0);
  assert.deepEqual(engine.result('B').output, reference(engine.model, [9, 8, 3, 4, 5], 1));
});

test('memory pressure preempts/recomputes requests without changing generation', () => {
  for (const prefixCaching of [false, true]) {
    const engine = new Engine({ blockSize: 2, numBlocks: 5, maxBatchTokens: 6, prefillChunkSize: 2, prefixCaching });
    const prompts = [encode('aa'), encode('bb'), encode('cc')];
    for (let i = 0; i < prompts.length; i++) engine.add({ id: String(i), prompt: prompts[i], maxNewTokens: 7 });
    const reports = drain(engine);
    assert.ok(engine.trace.some(e => e.type === 'preempted'));
    assert.ok(reports.some(r => r.work.some(w => w.phase === 'recompute')));
    for (let i = 0; i < prompts.length; i++) assert.deepEqual(engine.result(String(i)).output, reference(engine.model, prompts[i], 7));
    engine.cache.clearPrefixes();
    assert.equal(engine.cache.snapshot().freePages, 5);
  }
});

test('LRU eviction only removes cached pages with no active users', () => {
  const engine = new Engine({ blockSize: 2, numBlocks: 3, prefillChunkSize: 8 });
  engine.add({ id: 'A', prompt: [1, 2, 3, 4, 5], maxNewTokens: 1 });
  drain(engine);
  const reused = engine.cache.reusePrefix([1, 2, 9]);
  const protectedPage = reused.blocks[0];
  const other = engine.cache.create();
  for (let i = 0; i < 4; i++) {
    assert.ok(other.reserveToken());
    other.length++;
  }
  assert.ok(engine.cache.snapshot().evictions > 0);
  assert.equal(engine.cache.snapshot().pages[protectedPage].activeRefs, 1);
  assert.ok(!other.blocks.includes(protectedPage));
  assert.equal(other.reserveToken(), false);
  other.release();
  reused.release();
  engine.cache.clearPrefixes();
  assert.equal(engine.cache.snapshot().freePages, 3);
});

test('cancel running/queued requests and release references without damaging a sibling', () => {
  const engine = new Engine({ maxSequences: 2, blockSize: 2 });
  const prompt = encode('abc');
  engine.add({ id: 'A', prompt, maxNewTokens: 5 });
  engine.step();
  engine.add({ id: 'B', prompt, maxNewTokens: 5 });
  engine.step();
  engine.add({ id: 'queued', prompt, maxNewTokens: 1 });
  assert.equal(engine.cancel('A'), true);
  assert.equal(engine.cancel('queued'), true);
  assert.equal(engine.cancel('A'), false);
  drain(engine);
  assert.equal(engine.result('A').status, 'cancelled');
  assert.equal(engine.result('queued').status, 'cancelled');
  assert.deepEqual(engine.result('B').output, reference(engine.model, prompt, 5));
  engine.cache.clearPrefixes();
  assert.equal(engine.cache.snapshot().freePages, engine.options.numBlocks);
});

test('reject invalid inputs and requests that cannot fit even in isolation', () => {
  assert.throws(() => new Engine({ blockSize: 0 }), /positive integer/);
  assert.throws(() => new Engine({ maxBatchTokens: NaN }), /positive integer/);
  const engine = new Engine({ numBlocks: 1, blockSize: 2 });
  assert.throws(() => engine.add({ id: 'x', prompt: [], maxNewTokens: 1 }), /valid token/);
  assert.throws(() => engine.add({ id: 'x', prompt: [-1], maxNewTokens: 1 }), /valid token/);
  assert.throws(() => engine.add({ id: 'x', prompt: [1], maxNewTokens: 0 }), /positive integer/);
  assert.throws(() => engine.add({ id: 'x', prompt: [1, 2], maxNewTokens: 2 }), /cannot fit/);
  engine.add({ id: 'x', prompt: [1, 2], maxNewTokens: 1 });
  assert.throws(() => engine.add({ id: 'x', prompt: [1], maxNewTokens: 1 }), /unique/);
  drain(engine);
});
