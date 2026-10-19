import assert from 'node:assert/strict';
import { readFile, writeFile, copyFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { CudaDeviceBackend, cudaGraphOption } from '../src/cuda-device.ts';
import { CudaQwenRunner } from '../src/qwen-cuda.ts';
import { QwenRunner } from '../src/qwen.ts';
import { KVCache } from '../src/kv-cache.ts';
import { Engine } from '../src/engine.ts';
import { argmax } from '../src/runner.ts';

const directory = fileURLToPath(new URL('./fixtures/qwen-tiny/', import.meta.url));
const reference = JSON.parse(await readFile(join(directory, 'reference.json'), 'utf8'));
const options = { graph: cudaGraphOption(), device: Number(process.env.CUDA_DEVICE ?? 0), tileMiB: 1 };

test('all-device Qwen logits match Transformers with noncontiguous GPU KV pages', async () => {
  const backend = new CudaDeviceBackend(options);
  let maxError = 0;
  try {
    const model = await CudaQwenRunner.load(directory, backend);
    for (const sample of reference.cases) {
      const layout = { blockSize: 2, numBlocks: 12, layers: model.spec.layers, width: model.spec.kvHeads * model.spec.headDim };
      const cache = new KVCache({ ...layout, dtype: 'float32', deviceStorage: model.createKVStorage(layout) });
      assert.throws(() => cache.slot(0, 0, 0), /Device KV/);
      const blocker = cache.create(); assert.ok(blocker.reserveToken());
      const sequence = cache.create(); let logits: Float32Array | undefined;
      for (const token of sample.tokens) {
        assert.ok(sequence.reserveToken()); logits = model.forwardToken(token, sequence.length, sequence); sequence.length++;
        if (sequence.length === 2) blocker.release();
      }
      assert.deepEqual(sequence.blocks.slice(0, 2), [1, 0]);
      for (let step = 0; step < sample.output.length; step++) {
        const bytes = await readFile(join(directory, sample.logits[step]));
        for (let i = 0; i < model.spec.vocabSize; i++) {
          const error = Math.abs(logits![i] - bytes.readFloatLE(i * 4));
          maxError = Math.max(maxError, error); assert.ok(error < 1e-5, `step ${step} logit ${i}: ${error}`);
        }
        assert.equal(argmax(logits!), sample.output[step]);
        if (step + 1 < sample.output.length) {
          assert.ok(sequence.reserveToken()); logits = model.forwardToken(sample.output[step], sequence.length, sequence); sequence.length++;
        }
      }
      sequence.release(); assert.equal(cache.snapshot().freePages, layout.numBlocks);
    }
    console.log(JSON.stringify({ deviceQwenMaxError: maxError, stats: backend.stats() }));
  } finally { backend.close(); }
});

test('GPU sampling, shared prefixes, pressure/recompute and cancellation retain correct ownership', async () => {
  const backend = new CudaDeviceBackend(options);
  try {
    const model = await CudaQwenRunner.load(directory, backend);
    for (const numBlocks of [4, 16]) {
      const engine = new Engine({ blockSize: 2, numBlocks, maxBatchTokens: 3, prefillChunkSize: 2 }, model);
      const before = backend.stats();
      const [a, b] = reference.cases;
      engine.add({ id: 'A', prompt: a.tokens, maxNewTokens: a.output.length }); engine.step();
      engine.add({ id: 'B', prompt: b.tokens, maxNewTokens: b.output.length });
      engine.add({ id: 'C', prompt: a.tokens, maxNewTokens: a.output.length });
      let ticks = 0;
      while (engine.pending) { assert.ok(++ticks < 1000); engine.step(); }
      for (const [id, sample] of [['A', a], ['B', b], ['C', a]] as const) assert.deepEqual(engine.result(id).output, sample.output);
      assert.ok(engine.result('B').prefixHits > 0);
      if (numBlocks === 4) assert.ok(engine.trace.some(event => event.type === 'preempted'));
      engine.cache.clearPrefixes(); assert.equal(engine.cache.snapshot().freePages, numBlocks);
      const after = backend.stats();
      assert.equal(Number(after.deviceToHostBytes) - Number(before.deviceToHostBytes), 12 * 4, 'only sampled token IDs return to host');
      assert.equal(Number(after.hostToDeviceBytes) - Number(before.hostToDeviceBytes), Number(after.blockTableUploadBytes) - Number(before.blockTableUploadBytes) + Number(after.graphInputUploadBytes) - Number(before.graphInputUploadBytes), 'only block tables and graph token/position upload');
      assert.equal(after.matvecWeightUploadBytes, 0);
    }
    const engine = new Engine({ blockSize: 2, numBlocks: 12, prefillChunkSize: 2 }, model);
    const [a, b] = reference.cases;
    engine.add({ id: 'A', prompt: a.tokens, maxNewTokens: 4 }); engine.step();
    engine.add({ id: 'B', prompt: b.tokens, maxNewTokens: 4 }); engine.step();
    assert.ok(engine.cache.snapshot().pages.some(page => page.activeRefs === 2));
    assert.ok(engine.cancel('A'));
    let ticks = 0; while (engine.pending) { assert.ok(++ticks < 100); engine.step(); }
    assert.deepEqual(engine.result('B').output, b.output);
    engine.cache.clearPrefixes(); assert.equal(engine.cache.snapshot().freePages, 12);
  } finally { backend.close(); }
});

test('device handles, GPU argmax ties, nonfinite logits and closed resources are validated', () => {
  const backend = new CudaDeviceBackend(options), other = new CudaDeviceBackend(options);
  let x: object;
  try {
    x = backend.vector(4, Float32Array.of(1, 3, 3, -1));
    assert.equal(backend.argmax(x), 1);
    assert.deepEqual(backend.read(x), Float32Array.of(1, 3, 3, -1));
    assert.throws(() => other.read(x), /different or closed/);
    assert.throws(() => backend.add(x, backend.vector(3), x), /shape mismatch/);
    assert.throws(() => backend.argmax(backend.vector(1, Float32Array.of(NaN))), /Non-finite/);
    assert.throws(() => backend.rope(x, 3, 0, 10000), /RoPE shape/);
    backend.finish();
    backend.close(); backend.close();
    assert.throws(() => backend.read(x), /closed/);
  } finally { backend.close(); other.close(); }
});

test('device attention crosses 256 positions and reaches the configured KV boundary', async () => {
  const fixture = await mkdtemp(join(tmpdir(), 'sprout-device-context-'));
  const backend = new CudaDeviceBackend(options);
  try {
    const config = JSON.parse(await readFile(join(directory, 'config.json'), 'utf8'));
    config.max_position_embeddings = 384;
    await writeFile(join(fixture, 'config.json'), JSON.stringify(config));
    await copyFile(join(directory, 'model.safetensors'), join(fixture, 'model.safetensors'));
    const cpu = await QwenRunner.load(fixture), gpu = await CudaQwenRunner.load(fixture, backend, { maxContextTokens: 384 });
    const layout = { blockSize: 16, numBlocks: 24, layers: gpu.spec.layers, width: gpu.spec.kvHeads * gpu.spec.headDim };
    const hostCache = new KVCache({ ...layout, dtype: 'float32' });
    const deviceCache = new KVCache({ ...layout, dtype: 'float32', deviceStorage: gpu.createKVStorage(layout) });
    const blocker = deviceCache.create(); blocker.reserveToken();
    const host = hostCache.create(), device = deviceCache.create();
    const probes = new Set([0, 15, 16, 255, 256, 383]);
    for (let position = 0; position < 384; position++) {
      assert.ok(host.reserveToken()); assert.ok(device.reserveToken());
      const token = 1 + (position * 7) % 30, output = probes.has(position);
      const expected = cpu.forwardToken(token, position, host, output), actual = gpu.forwardToken(token, position, device, output);
      host.length++; device.length++;
      if (position === 15) blocker.release();
      if (output) for (let i = 0; i < gpu.spec.vocabSize; i++) assert.ok(Math.abs(actual![i] - expected![i]) < 1e-5, `position ${position}, logit ${i}`);
    }
    assert.deepEqual(device.blocks.slice(0, 2), [1, 0]);
    assert.equal(device.reserveToken(), false);
    assert.throws(() => gpu.forwardSample(1, 384, device, true), /position/);
    host.release(); device.release(); assert.equal(deviceCache.snapshot().freePages, 24);
  } finally { backend.close(); await rm(fixture, { recursive: true, force: true }); }
});

test('graph capture abort, handle ownership and explicit release are safe', { skip: !options.graph }, () => {
  const backend = new CudaDeviceBackend(options), other = new CudaDeviceBackend(options);
  try {
    const x = backend.vector(3, Float32Array.of(1, 2, 3));
    const y = backend.vector(3, Float32Array.of(2, 3, 4)), out = backend.vector(3);
    assert.throws(() => backend.capture(2, 2, out, false, () => {
      backend.add(x, y, out); throw new Error('capture failure');
    }), /capture failure/);
    assert.deepEqual(backend.read(out), Float32Array.of(0, 0, 0), 'aborted capture did not execute');
    assert.throws(() => backend.capture(2, 2, out, false, () => { backend.read(out); }), /not allowed during graph capture/);
    assert.throws(() => backend.capture(2, 2, out, true, () => backend.add(x,y,out)), /needs an argmax/);
    const nan = backend.vector(1, Float32Array.of(NaN));
    const invalidGraph = backend.capture(2, 2, nan, true, () => backend.enqueueArgmax(nan));
    assert.throws(() => backend.replay(invalidGraph, [0], 0, 0), /Non-finite/);
    backend.releaseGraph(invalidGraph);
    const graph = backend.capture(2, 2, out, false, () => backend.add(x, y, out));
    assert.deepEqual(backend.replay(graph, [1], 0, 0), Float32Array.of(3, 5, 7));
    assert.throws(() => other.replay(graph, [0], 0, 0), /different or closed/);
    assert.throws(() => backend.replay(graph, [2], 0, 0), /physical page/);
    assert.throws(() => backend.replay(graph, [0], 0, 2), /block table or position/);
    assert.equal(backend.stats().liveGraphs, 1);
    backend.releaseGraph(graph);
    assert.equal(backend.stats().liveGraphs, 0);
    assert.throws(() => backend.replay(graph, [0], 0, 0), /closed/);
    const graph2 = backend.capture(2, 2, out, true, () => { backend.add(x,y,out); backend.enqueueArgmax(out); });
    assert.equal(backend.replay(graph2, [0], 0, 0), 2);
    backend.close();
    assert.throws(() => backend.replay(graph2, [0], 0, 0), /closed/);
  } finally { backend.close(); other.close(); }
});
