import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { Engine } from '../src/engine.ts';
import { KVCache } from '../src/kv-cache.ts';
import { QwenRunner } from '../src/qwen.ts';
import { CudaOffload, type CudaWeightMode } from '../src/cuda-offload.ts';
import { CudaDeviceBackend, cudaGraphOption } from '../src/cuda-device.ts';
import { CudaQwenRunner } from '../src/qwen-cuda.ts';
import { argmax } from '../src/runner.ts';
import { GenerationService } from '../src/service.ts';
import { createApiServer } from '../src/http.ts';
import { defaultTextPlugins, validateTokenizer, type ChatMessage } from '../src/text-plugins.ts';

const directory = process.env.MODEL_DIR ?? 'models/qwen2.5-0.5b-instruct';
const referenceDir = process.env.REFERENCE_DIR ?? '.cache/qwen-reference';
interface Sample { tokens: number[]; output: number[]; logits: string[]; prompt?: string; messages?: ChatMessage[]; formatted?: string; text: string }

test('real Qwen checkpoint: reference logits, tokenizer, scheduling, HTTP and EOS', { timeout: 240000 }, async t => {
  const reference = JSON.parse(await readFile(join(referenceDir, 'reference.json'), 'utf8')) as { cases: Sample[] };
  const plugins = defaultTextPlugins();
  const tokenizer = await plugins.tokenizer('hf-bytelevel', directory);
  const template = await plugins.template('jinja', directory);
  const loadStart = performance.now();
  const backend = process.env.MATRIX_BACKEND === 'cuda-device' ? new CudaDeviceBackend({ graph: cudaGraphOption(), device: Number(process.env.CUDA_DEVICE ?? 0) })
    : process.env.MATRIX_BACKEND === 'cuda-offload' ? new CudaOffload({ device: Number(process.env.CUDA_DEVICE ?? 0),
    weightMode: (process.env.CUDA_WEIGHT_MODE ?? 'offload') as CudaWeightMode }) : undefined;
  if (backend) t.after(() => { t.diagnostic(`CUDA offload: ${JSON.stringify(backend.stats())}`); backend.close(); });
  const model = backend instanceof CudaDeviceBackend ? await CudaQwenRunner.load(directory, backend)
    : await QwenRunner.load(directory, { matrixLoader: backend?.matrix });
  t.diagnostic(`Load ${Math.round(performance.now() - loadStart)} ms; RSS ${Math.round(process.memoryUsage().rss / 1024 ** 2)} MiB`);
  validateTokenizer(tokenizer, model.spec);
  const probes = JSON.parse(await readFile(join(referenceDir, 'tokenizer.json'), 'utf8'));
  for (const probe of probes) {
    assert.deepEqual(tokenizer.encode(probe.text), probe.ids);
    const decoder = tokenizer.decoder();
    const result = probe.ids.map((id: number) => decoder.push(id)).join('') + decoder.finish();
    // Tokenizers can normalize input; compare to reference decoding, not the original text.
    assert.equal(result, probe.decoded);
  }
  let maxError = 0;
  for (const sample of reference.cases) {
    const prompt = sample.messages ? template.render(sample.messages) : sample.prompt!;
    if (sample.formatted) assert.equal(prompt, sample.formatted);
    assert.deepEqual(tokenizer.encode(prompt), sample.tokens);
    const layout = { blockSize: 4, numBlocks: 64, layers: model.spec.layers, width: model.spec.kvHeads * model.spec.headDim };
    const cache = new KVCache({ ...layout, dtype: 'float32',
      deviceStorage: model instanceof CudaQwenRunner ? model.createKVStorage(layout) : undefined });
    const sequence = cache.create();
    let logits: Float32Array | undefined;
    const start = performance.now();
    for (const [i, token] of sample.tokens.entries()) {
      assert.ok(sequence.reserveToken());
      logits = model.forwardToken(token, sequence.length, sequence, i === sample.tokens.length - 1);
      sequence.length++;
    }
    const firstMs = performance.now() - start;
    const decoder = tokenizer.decoder();
    let text = '';
    for (let step = 0; step < sample.output.length; step++) {
      const bytes = await readFile(join(referenceDir, sample.logits[step]));
      assert.equal(bytes.length, model.spec.vocabSize * 4);
      for (let i = 0; i < logits!.length; i++) {
        const expected = bytes.readFloatLE(i * 4);
        const error = Math.abs(logits![i] - expected);
        maxError = Math.max(maxError, error);
        assert.ok(error <= 1e-3 + Math.abs(expected) * 1e-4, `logit ${i}: actual=${logits![i]}, expected=${expected}`);
      }
      assert.equal(argmax(logits!), sample.output[step]);
      text += decoder.push(sample.output[step]);
      if (step + 1 < sample.output.length) {
        assert.ok(sequence.reserveToken());
        logits = model.forwardToken(sample.output[step], sequence.length, sequence);
        sequence.length++;
      }
    }
    text += decoder.finish();
    assert.equal(text, sample.text);
    const duration = performance.now() - start;
    t.diagnostic(`${sample.tokens.length} prompt tokens, ${sample.output.length} output tokens: first=${Math.round(firstMs)} ms, total=${Math.round(duration)} ms, text=${JSON.stringify(text)}`);
    sequence.release();
  }
  t.diagnostic(`Maximum absolute logit error versus Transformers FP32 eager: ${maxError}`);

  const shared = reference.cases.find(s => s.prompt === 'Hello, world')!;
  const pressure = new Engine({ blockSize: 2, numBlocks: 3, prefillChunkSize: 2, maxBatchTokens: 4 }, model);
  pressure.add({ id: 'A', prompt: shared.tokens, maxNewTokens: shared.output.length });
  pressure.step();
  pressure.add({ id: 'B', prompt: shared.tokens, maxNewTokens: shared.output.length });
  let iterations = 0;
  while (pressure.pending) { assert.ok(++iterations < 200); pressure.step(); }
  for (const id of ['A', 'B']) assert.deepEqual(pressure.result(id).output, shared.output);
  assert.ok(pressure.result('B').prefixHits > 0);
  assert.ok(pressure.trace.some(e => e.type === 'preempted'));
  pressure.cache.clearPrefixes();
  assert.equal(pressure.cache.snapshot().freePages, 3);

  const engine = new Engine({ blockSize: 4, numBlocks: 64, maxBatchTokens: 4, prefillChunkSize: 4 }, model);
  const service = new GenerationService(engine);
  const server = createApiServer({ service, tokenizer, template });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(async () => {
    service.close(); server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  });
  const address = server.address(); assert.ok(address && typeof address === 'object');
  const endpoint = `http://127.0.0.1:${address.port}/generate`;
  const chat = reference.cases.find(s => s.messages)!;
  assert.ok(model.spec.eosTokenIds.includes(chat.output.at(-1)!), 'reference chat must cover natural EOS');
  const request = (stream: boolean) => fetch(endpoint, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: chat.messages, maxNewTokens: 24, stream }) });
  const jsonResponse = await request(false);
  assert.equal(jsonResponse.status, 200);
  const body = await jsonResponse.json();
  assert.deepEqual(body.output, chat.output);
  assert.equal(body.text, chat.text);
  assert.equal(body.finishReason, 'eos');
  assert.deepEqual(body.usage, { promptTokens: chat.tokens.length, completionTokens: chat.output.length });
  const streamed = await request(true);
  assert.equal(streamed.status, 200);
  const frames = (await streamed.text()).trim().split('\n\n').map(frame => JSON.parse(frame.split('\n')[1].slice(6)));
  assert.deepEqual(frames.filter(e => e.type === 'token').map(e => e.token), chat.output);
  assert.equal(frames.map(e => e.textDelta).join(''), chat.text);
  assert.equal(frames.at(-1).finishReason, 'eos');
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
  t.diagnostic(`HTTP JSON/SSE natural EOS: ${JSON.stringify(chat.text)}`);
});
