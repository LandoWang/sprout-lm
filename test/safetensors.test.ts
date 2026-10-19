import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { SafeTensors } from '../src/safetensors.ts';

function file(header: unknown, data: Buffer): Buffer {
  const json = Buffer.from(JSON.stringify(header));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(json.length));
  return Buffer.concat([length, json, data]);
}

test('safetensors converts BF16/F16/F32 including signed and subnormal values', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-weights-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const data = Buffer.alloc(24);
  [0x3f80, 0xc020, 0x0001].forEach((value, i) => data.writeUInt16LE(value, i * 2));
  [0x3c00, 0xc100, 0x0001].forEach((value, i) => data.writeUInt16LE(value, 6 + i * 2));
  [1, -2.5, 0.25].forEach((value, i) => data.writeFloatLE(value, 12 + i * 4));
  const path = join(directory, 'model.safetensors');
  await writeFile(path, file({
    bf: { dtype: 'BF16', shape: [3], data_offsets: [0, 6] },
    half: { dtype: 'F16', shape: [3], data_offsets: [6, 12] },
    full: { dtype: 'F32', shape: [3], data_offsets: [12, 24] },
  }, data));
  const tensors = await SafeTensors.open(path);
  try {
    assert.deepEqual([...await tensors.tensor('bf', [3])], [1, -2.5, 2 ** -133]);
    assert.deepEqual([...await tensors.tensor('half', [3])], [1, -2.5, 2 ** -24]);
    assert.deepEqual([...await tensors.tensor('full', [3])], [1, -2.5, 0.25]);
    await assert.rejects(tensors.tensor('bf', [1, 3]), /Shape mismatch/);
    await assert.rejects(tensors.tensor('missing', [1]), /Missing tensor/);
  } finally { await tensors.close(); }
});

test('safetensors rejects truncated, overlapping, out-of-range and unsupported tensors', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-bad-weights-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const path = join(directory, 'bad.safetensors');
  const valid = { dtype: 'F32', shape: [1], data_offsets: [0, 4] };
  for (const bytes of [Buffer.alloc(4), file({ a: valid }, Buffer.alloc(2)),
    file({ a: valid, b: valid }, Buffer.alloc(4)),
    file({ a: { ...valid, dtype: 'I64' } }, Buffer.alloc(4)),
    file({ a: { ...valid, shape: [2] } }, Buffer.alloc(4))]) {
    await writeFile(path, bytes);
    await assert.rejects(SafeTensors.open(path));
  }
});
