import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { CudaOffload } from '../src/cuda-offload.ts';
import { QwenRunner } from '../src/qwen.ts';
import { KVCache } from '../src/kv-cache.ts';
import { argmax } from '../src/runner.ts';

for (const weightMode of ['offload', 'resident'] as const) {
test(`CUDA ${weightMode} non-square GEMV handles F32/BF16/F16 and validates lifecycle`, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-cuda-'));
  const backend = new CudaOffload({ tileMiB: 1, device: Number(process.env.CUDA_DEVICE ?? 0), weightMode });
  let closedProbe: ReturnType<typeof backend.matrix> | undefined;
  try {
    const rows = 600, cols = 1024;
    for (const dtype of ['F32', 'BF16', 'F16'] as const) {
      const width = dtype === 'F32' ? 4 : 2, data = Buffer.alloc(rows * cols * width);
      const x = Float32Array.from({ length: cols }, (_, i) => (i % 7) - 3);
      const expected = new Float32Array(rows);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const value = (r + c) % 2 ? 1 : -2;
        if (dtype === 'F32') data.writeFloatLE(value, (r * cols + c) * width);
        else data.writeUInt16LE(dtype === 'BF16' ? (value === 1 ? 0x3f80 : 0xc000) : (value === 1 ? 0x3c00 : 0xc000), (r * cols + c) * width);
        expected[r] += value * x[c];
      }
      const path = join(directory, dtype); await writeFile(path, data);
      const matrix = backend.matrix({ path, offset: 0, bytes: data.length, shape: [rows,cols], dtype });
      closedProbe = matrix;
      assert.deepEqual(matrix.matvec(x), expected);
      const before = backend.stats();
      assert.deepEqual(matrix.matvec(x), expected);
      const after = backend.stats();
      assert.equal(Number(after.hostToDeviceBytes) - Number(before.hostToDeviceBytes),
        cols * 4 + (weightMode === 'resident' ? 0 : data.length));
      assert.equal(matrix.row(0)[0], -2); assert.equal(matrix.row(0)[1], 1);
      assert.throws(() => matrix.matvec(new Float32Array(3)), /shape mismatch/);
      assert.throws(() => matrix.row(rows), /Invalid matrix row/);
    }
    const stats = backend.stats();
    assert.equal(stats.matvecCalls, 6); assert.equal(stats.tiles, 18);
    assert.ok(Number(stats.peakScratchBytes) < 2 * 1024 * 1024);
    assert.equal(stats.residentWeightBytes, weightMode === 'resident' ? backend.mappedWeightBytes : 0);
    assert.equal(stats.weightUploadBytes, stats.residentWeightBytes);
    assert.equal(stats.matvecWeightUploadBytes, weightMode === 'resident' ? 0 : backend.mappedWeightBytes * 2);
    console.log(JSON.stringify({ cudaGEMV: stats }));
  } finally { backend.close(); backend.close(); await rm(directory, { recursive: true, force: true }); }
  assert.throws(() => backend.stats(), /closed/);
  assert.throws(() => closedProbe!.matvec(new Float32Array(1024)), /closed/);
  assert.throws(() => closedProbe!.row(0), /closed/);
});

test(`CUDA ${weightMode} preserves Qwen paged forward against independent Transformers logits`, async () => {
  const directory = fileURLToPath(new URL('./fixtures/qwen-tiny/', import.meta.url));
  const reference = JSON.parse(await readFile(join(directory, 'reference.json'), 'utf8'));
  const backend = new CudaOffload({ tileMiB: 1, device: Number(process.env.CUDA_DEVICE ?? 0), weightMode });
  let maxError = 0;
  try {
    const model = await QwenRunner.load(directory, { matrixLoader: backend.matrix });
    for (const sample of reference.cases) {
      const cache = new KVCache({ numBlocks: 12, blockSize: 2, layers: model.spec.layers,
        width: model.spec.kvHeads * model.spec.headDim, dtype: 'float32' });
      const blocker = cache.create(); blocker.reserveToken();
      const kv = cache.create(); let logits: Float32Array | undefined;
      for (const token of sample.tokens) {
        assert.ok(kv.reserveToken()); logits = model.forwardToken(token, kv.length, kv); kv.length++;
        if (kv.length === 2) blocker.release();
      }
      assert.deepEqual(kv.blocks.slice(0,2), [1,0]);
      for (let step = 0; step < sample.output.length; step++) {
        const bytes = await readFile(join(directory, sample.logits[step]));
        const expected = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
        for (let i = 0; i < expected.length; i++) {
          const error = Math.abs(logits![i] - expected[i]); maxError = Math.max(maxError, error);
          assert.ok(error < 1e-5, `step ${step}, logit ${i}: ${error}`);
        }
        assert.equal(argmax(logits!), sample.output[step]);
        if (step + 1 < sample.output.length) {
          assert.ok(kv.reserveToken()); logits = model.forwardToken(sample.output[step], kv.length, kv); kv.length++;
        }
      }
      kv.release(); assert.equal(cache.snapshot().freePages, 12);
    }
    console.log(JSON.stringify({ cudaQwenMaxError: maxError, stats: backend.stats() }));
  } finally { backend.close(); }
});
}

test('resident allocation failure preserves prior matrices and respects memory budgets', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-cuda-budget-'));
  const device = Number(process.env.CUDA_DEVICE ?? 0);
  const backend = new CudaOffload({ device, weightMode: 'resident', weightLimitMiB: 1 });
  const reserved = new CudaOffload({ device, weightMode: 'resident', reserveMiB: 1024 * 1024 });
  try {
    const path = join(directory, 'weights');
    const data = Buffer.alloc(2 * 1024 ** 2); data.writeFloatLE(2, 0); await writeFile(path, data);
    const descriptor = { path, offset: 0, bytes: 4, dtype: 'F32' as const, shape: [1, 1] };
    const small = backend.matrix(descriptor);
    assert.throws(() => backend.matrix({ ...descriptor, shape: [512, 1024], bytes: data.length }), /budget exceeded/);
    assert.equal(backend.stats().residentWeightBytes, 4);
    assert.equal(backend.mappedWeightBytes, 4);
    assert.deepEqual(small.matvec(new Float32Array([3])), new Float32Array([6]));
    assert.throws(() => reserved.matrix(descriptor), /Insufficient free CUDA memory/);
    assert.equal(reserved.stats().residentWeightBytes, 0);
    assert.equal(reserved.mappedWeightBytes, 0);
  } finally { backend.close(); reserved.close(); await rm(directory, { recursive: true, force: true }); }
});
