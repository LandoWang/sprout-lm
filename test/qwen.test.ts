import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import test from 'node:test';
import { Engine } from '../src/engine.ts';
import { KVCache } from '../src/kv-cache.ts';
import { QwenRunner, parseQwenConfig } from '../src/qwen.ts';
import { argmax, type ModelRunner } from '../src/runner.ts';
import { GenerationService } from '../src/service.ts';

const directory = fileURLToPath(new URL('./fixtures/qwen-tiny/', import.meta.url));
const reference = JSON.parse(await readFile(join(directory, 'reference.json'), 'utf8'));

test('native Qwen GQA/RoPE/SwiGLU logits match independent Transformers full recomputation', async () => {
  const model = await QwenRunner.load(directory);
  for (const sample of reference.cases) {
    const cache = new KVCache({ blockSize: 2, numBlocks: 12, layers: model.spec.layers,
      width: model.spec.kvHeads * model.spec.headDim, dtype: 'float32' });
    const blocker = cache.create(); blocker.reserveToken();
    const sequence = cache.create();
    let actual: Float32Array | undefined;
    for (const token of sample.tokens) {
      assert.ok(sequence.reserveToken());
      actual = model.forwardToken(token, sequence.length, sequence);
      sequence.length++;
      if (sequence.length === 2) blocker.release();
    }
    assert.deepEqual(sequence.blocks.slice(0, 2), [1, 0]);
    assert.ok(cache.slot(0, 0, 0).key instanceof Float32Array);
    assert.equal(cache.width, 8, 'KV width differs from hidden size 16');
    for (let step = 0; step < sample.output.length; step++) {
      const bytes = await readFile(join(directory, sample.logits[step]));
      const expected = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
      for (let i = 0; i < expected.length; i++) assert.ok(Math.abs(actual![i] - expected[i]) < 2e-6, `logit ${i} at step ${step}`);
      assert.equal(argmax(actual!), sample.output[step]);
      if (step + 1 < sample.output.length) {
        assert.ok(sequence.reserveToken());
        actual = model.forwardToken(sample.output[step], sequence.length, sequence);
        sequence.length++;
      }
    }
    sequence.release();
    assert.equal(cache.snapshot().freePages, 12);
  }
});

test('Qwen mixed requests survive prefix sharing, chunking and pressure with reference outputs', async () => {
  const model = await QwenRunner.load(directory);
  for (const numBlocks of [4, 16]) {
    const engine = new Engine({ blockSize: 2, numBlocks, maxBatchTokens: 3, prefillChunkSize: 2 }, model);
    const a = reference.cases[0], b = reference.cases[1];
    engine.add({ id: 'A', prompt: a.tokens, maxNewTokens: a.output.length });
    engine.step();
    engine.add({ id: 'B', prompt: b.tokens, maxNewTokens: b.output.length });
    engine.add({ id: 'C', prompt: a.tokens, maxNewTokens: a.output.length });
    let count = 0;
    while (engine.pending) { assert.ok(++count < 2000); engine.step(); }
    assert.deepEqual(engine.result('A').output, a.output);
    assert.deepEqual(engine.result('B').output, b.output);
    assert.deepEqual(engine.result('C').output, a.output);
    assert.ok(engine.result('B').prefixHits > 0);
    if (numBlocks === 4) assert.ok(engine.trace.some(e => e.type === 'preempted'));
    assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
    engine.cache.clearPrefixes();
    assert.equal(engine.cache.snapshot().freePages, numBlocks);
  }
});

test('EOS is sampled once, counts toward usage, and releases shared pages before the token limit', async () => {
  const runner: ModelRunner = {
    spec: { id: 'eos-test', layers: 1, hiddenSize: 4, attentionHeads: 2, kvHeads: 1, headDim: 2,
      vocabSize: 4, maxContextLength: 16, kvDtype: 'float32', eosTokenIds: [3] },
    forwardToken(_token, position, kv) {
      kv.set(0, position, Float32Array.of(1, 2), Float32Array.of(3, 4));
      return Float32Array.of(0, 0, 0, 1);
    },
  };
  const engine = new Engine({ blockSize: 2, numBlocks: 4 }, runner);
  const service = new GenerationService(engine);
  try {
    const streams = [service.generate({ prompt: [1, 2], maxNewTokens: 5 }), service.generate({ prompt: [1, 2], maxNewTokens: 5 })];
    for (const stream of streams) {
      const events = [];
      for await (const event of stream) events.push(event);
      assert.deepEqual(events.filter(e => e.type === 'token').map(e => e.token), [3]);
      assert.deepEqual(events.at(-1), { id: stream.id, type: 'finished', finishReason: 'eos', usage: { promptTokens: 2, completionTokens: 1 } });
    }
    assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
    assert.throws(() => engine.add({ id: 'too-long', prompt: [1], maxNewTokens: 17 }), /context limit/);
  } finally { service.close(); }
});

test('unsupported Qwen configurations fail before loading weights', async () => {
  const config = JSON.parse(await readFile(join(directory, 'config.json'), 'utf8'));
  for (const extra of [{ model_type: 'llama' }, { rope_scaling: { type: 'linear' } }, { use_sliding_window: true },
    { num_key_value_heads: 3 }, { rms_norm_eps: 0 }, { eos_token_id: -1 }]) {
    assert.throws(() => parseQwenConfig({ ...config, ...extra }));
  }
});

test('cancelling a Qwen request preserves a sibling using its prefix pages', async () => {
  const model = await QwenRunner.load(directory);
  const engine = new Engine({ blockSize: 2, numBlocks: 12, prefillChunkSize: 2 }, model);
  const [a, b] = reference.cases;
  engine.add({ id: 'A', prompt: a.tokens, maxNewTokens: 4 });
  engine.step();
  engine.add({ id: 'B', prompt: b.tokens, maxNewTokens: 4 });
  engine.step();
  assert.ok(engine.cache.snapshot().pages.some(page => page.activeRefs === 2));
  assert.equal(engine.cancel('A'), true);
  let ticks = 0;
  while (engine.pending) { assert.ok(++ticks < 100); engine.step(); }
  assert.deepEqual(engine.result('B').output, b.output);
  engine.cache.clearPrefixes();
  assert.equal(engine.cache.snapshot().freePages, 12);
});
