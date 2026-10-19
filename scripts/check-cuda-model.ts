import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, dirname, resolve } from 'node:path';
import { CudaDeviceBackend, cudaGraphOption } from '../src/cuda-device.ts';
import { CudaQwenRunner } from '../src/qwen-cuda.ts';
import { KVCache } from '../src/kv-cache.ts';
import { argmax } from '../src/runner.ts';

const [directory, referenceDir] = process.argv.slice(2);
if (!directory || !referenceDir) throw new Error('Usage: node scripts/check-cuda-model.ts CHECKPOINT REFERENCE');
const reference = JSON.parse(await readFile(join(referenceDir, 'reference.json'), 'utf8'));
const backend = new CudaDeviceBackend({ graph: cudaGraphOption(), device: Number(process.env.CUDA_DEVICE ?? 0), reserveMiB: 4096 });
const started = performance.now();
try {
  const model = await CudaQwenRunner.load(directory, backend, { maxContextTokens: 256 });
  assert.equal(model.spec.id, reference.modelId);
  let maxError = 0, maxToleranceRatio = 0, positions = 0;
  const results = [];
  for (const sample of reference.cases) {
    const layout = { numBlocks: 16, blockSize: 16, layers: model.spec.layers, width: model.spec.kvHeads * model.spec.headDim };
    const cache = new KVCache({ ...layout, dtype: 'float32', deviceStorage: model.createKVStorage(layout) });
    const sequence = cache.create(); let logits: Float32Array | undefined;
    for (const [i, token] of (sample.tokens as number[]).entries()) {
      assert.ok(sequence.reserveToken()); logits = model.forwardToken(token, sequence.length, sequence, i === sample.tokens.length - 1); sequence.length++;
    }
    const output = [];
    for (let step = 0; step < sample.output.length; step++) {
      const expected = await readFile(join(referenceDir, sample.logits[step]));
      assert.equal(expected.length, model.spec.vocabSize * 4);
      for (let i = 0; i < model.spec.vocabSize; i++) {
        const value = expected.readFloatLE(i * 4), error = Math.abs(logits![i] - value), tolerance = 1e-3 + Math.abs(value) * 1e-4;
        maxError = Math.max(maxError, error); maxToleranceRatio = Math.max(maxToleranceRatio, error / tolerance);
        assert.ok(error <= tolerance, `case ${sample.prompt}, step ${step}, logit ${i}: ${logits![i]} vs ${value}`);
      }
      output.push(argmax(logits!)); assert.equal(output.at(-1), sample.output[step]); positions++;
      if (step + 1 < sample.output.length) {
        assert.ok(sequence.reserveToken()); logits = model.forwardToken(sample.output[step], sequence.length, sequence); sequence.length++;
      }
    }
    sequence.release(); assert.equal(cache.snapshot().freePages, layout.numBlocks);
    results.push({ prompt: sample.prompt, output });
  }
  const report = { model: model.spec.id, reference: { torch: reference.torch, transformers: reference.transformers,
    dtype: reference.dtype, device: reference.device, attention: reference.attention },
    positions, logitsPerPosition: model.spec.vocabSize, maxError, maxToleranceRatio,
    tolerance: '1e-3 + abs(reference) * 1e-4', results, durationMs: performance.now() - started, stats: backend.stats() };
  if (process.env.REPORT_PATH) {
    const path = resolve(process.env.REPORT_PATH); await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(report, null, 2)+'\n');
  }
  console.log(JSON.stringify(report));
} finally { backend.close(); }
