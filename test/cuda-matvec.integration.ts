import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { CudaDeviceBackend } from '../src/cuda-device.ts';
import { CudaOffload } from '../src/cuda-offload.ts';

test('direct 16-bit weights preserve FP32 activations, tails, reductions and graph replay', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-matvec-'));
  const device = Number(process.env.CUDA_DEVICE ?? 0);
  const direct = new CudaDeviceBackend({ device, matvec: 'direct', tileMiB: 1 });
  const baseline = new CudaDeviceBackend({ device, matvec: 'convert', tileMiB: 1 });
  const offload = new CudaOffload({ device, matvec: 'direct', tileMiB: 1 });
  let maxError = 0;
  try {
    for (const dtype of ['BF16', 'F16'] as const) {
      for (const [rows, cols] of [[1, 1], [3, 31], [5, 32], [7, 33], [9, 124], [11, 128],
        [13, 132], [17, 1025], [19, 5120], [21, 27648], [600, 1024]]) {
        const bytes = Buffer.alloc(rows * cols * 2 + 3);
        const x = Float32Array.from({ length: cols }, (_, i) => Math.sin(i * 0.37 + 0.1) * 1.234567);
        const expected = new Float64Array(rows);
        let seed = 123456;
        for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          const sign = seed >>> 31, exp = 1 + ((seed >>> 16) % 7), frac = seed & (dtype === 'BF16' ? 127 : 1023);
          const exponent = exp + (dtype === 'BF16' ? 120 : 8), mantissaBits = dtype === 'BF16' ? 7 : 10;
          // Include zeros, subnormals, mixed signs and cancellation, using exact decoded values.
          const bits = col % 29 === 0 ? (sign << 15) | frac : (sign << 15) | (exponent << mantissaBits) | frac;
          bytes.writeUInt16LE(bits, 3 + (row * cols + col) * 2);
          const e = (bits >>> mantissaBits) & (dtype === 'BF16' ? 255 : 31);
          const value = (sign ? -1 : 1) * (e ? 1 + frac / 2 ** mantissaBits : frac / 2 ** mantissaBits)
            * 2 ** ((e || 1) - (dtype === 'BF16' ? 127 : 15));
          expected[row] += value * x[col];
        }
        const path = join(directory, `${dtype}-${rows}-${cols}`); await writeFile(path, bytes);
        const descriptor = { path, offset: 3, bytes: rows * cols * 2, shape: [rows, cols], dtype };
        const results: Float32Array[] = [];
        for (const backend of [direct, baseline]) {
          const matrix = backend.deviceMatrix(descriptor), input = backend.vector(cols, x), output = backend.vector(rows);
          backend.linear(matrix, input, output); results.push(backend.read(output));
          // Capturing does not execute, replay may repeat, and direct mode needs no weight scratch.
          const graph = backend.capture(2, 16, output, false, () => backend.linear(matrix, input, output));
          for (let repeat = 0; repeat < 2; repeat++)
            results.push(backend.replay(graph, [1], 0, repeat) as Float32Array);
          backend.releaseGraph(graph);
        }
        results.push(offload.matrix(descriptor).matvec(x));
        for (const actual of results) for (let row = 0; row < rows; row++) {
          const error = Math.abs(actual[row] - expected[row]); maxError = Math.max(maxError, error);
          assert.ok(error <= 1e-4 + Math.abs(expected[row]) * 2e-5,
            `${dtype} ${rows}x${cols} row ${row}: ${actual[row]} vs ${expected[row]}`);
        }
      }
    }
    assert.equal(direct.stats().peakScratchBytes, 0);
    assert.equal(direct.stats().matvecWeightUploadBytes, 0);
    assert.equal(direct.stats().liveGraphs, 0);
    console.log(JSON.stringify({ directMatvecMaxError: maxError, stats: direct.stats() }));
  } finally {
    direct.close(); baseline.close(); offload.close(); await rm(directory, { recursive: true, force: true });
  }
});
