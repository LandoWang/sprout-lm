import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, copyFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { WeightFiles } from '../src/weight-files.ts';
import { QwenRunner } from '../src/qwen.ts';
import { Engine } from '../src/engine.ts';

const source = fileURLToPath(new URL('./fixtures/qwen-tiny/', import.meta.url));
test('indexed Qwen weights preserve all generated tokens and validate shard lookup', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'sprout-shards-'));
  try {
    await copyFile(join(source, 'config.json'), join(dir, 'config.json'));
    const bytes = await readFile(join(source, 'model.safetensors'));
    const start = 8 + Number(bytes.readBigUInt64LE());
    const entries = Object.entries(JSON.parse(bytes.subarray(8, start).toString())).filter(([name]) => name !== '__metadata__');
    const mapping: Record<string, string> = {};
    for (let shard = 0; shard < 2; shard++) {
      const name = `model-${shard}.safetensors`; let offset = 0;
      const header: Record<string, unknown> = {}; const data: Buffer[] = [];
      for (const [key, value] of entries.filter((_, index) => index % 2 === shard)) {
        const entry = value as { dtype: string; shape: number[]; data_offsets: number[] };
        const part = bytes.subarray(start + entry.data_offsets[0], start + entry.data_offsets[1]);
        header[key] = { dtype: entry.dtype, shape: entry.shape, data_offsets: [offset, offset + part.length] };
        data.push(part); offset += part.length; mapping[key] = name;
      }
      const json = Buffer.from(JSON.stringify(header)); const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(json.length));
      await writeFile(join(dir, name), Buffer.concat([prefix, json, ...data]));
    }
    await writeFile(join(dir, 'model.safetensors.index.json'), JSON.stringify({ weight_map: mapping }));
    const original = await QwenRunner.load(source), sharded = await QwenRunner.load(dir);
    const outputs = [original, sharded].map(model => {
      const engine = new Engine({ blockSize: 2, numBlocks: 12 }, model);
      engine.add({ id: 'a', prompt: [1,7,3,9], maxNewTokens: 4 });
      while (engine.pending) engine.step(); return engine.result('a').output;
    });
    assert.deepEqual(outputs[1], outputs[0]);
    const files = await WeightFiles.open(dir);
    try {
      const descriptor = await files.descriptor('model.embed_tokens.weight', [32,16]);
      assert.equal(descriptor.bytes, 32 * 16 * 4);
      await assert.rejects(files.tensor('absent', [1]), /Missing tensor in shard index/);
      await assert.rejects(files.descriptor('model.embed_tokens.weight', [16,32]), /Shape mismatch/);
    } finally { await files.close(); }
    await writeFile(join(dir, 'model.safetensors.index.json'), JSON.stringify({ weight_map: { bad: '../outside.safetensors' } }));
    await assert.rejects(WeightFiles.open(dir), /Invalid checkpoint shard filename/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
