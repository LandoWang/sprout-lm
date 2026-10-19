import assert from 'node:assert/strict';
import { once } from 'node:events';
import { setImmediate as nextTurn } from 'node:timers/promises';
import test, { type TestContext } from 'node:test';
import { Engine } from '../src/engine.ts';
import { createApiServer, type ApiOptions } from '../src/http.ts';
import { TinyTransformer, argmax, encode } from '../src/model.ts';
import { GenerationService, type GenerationEvent } from '../src/service.ts';

async function fixture(t: TestContext, engine = new Engine(), options: ApiOptions = {}) {
  const service = options.service ?? new GenerationService(engine);
  const server = createApiServer({ ...options, service });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => {
    service.close();
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  });
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const url = `http://127.0.0.1:${address.port}`;
  const generate = (body: unknown, signal?: AbortSignal) => fetch(`${url}/generate`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal,
  });
  return { url, engine, service, generate };
}

type WireEvent = GenerationEvent & { textDelta: string };
async function* readEvents(response: Response): AsyncGenerator<WireEvent> {
  assert.match(response.headers.get('content-type') ?? '', /text\/event-stream/);
  const decoder = new TextDecoder();
  let buffer = '';
  for await (const chunk of response.body!) {
    buffer += decoder.decode(chunk, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const [name, data] = frame.split('\n');
      const event = JSON.parse(data.slice('data: '.length)) as WireEvent;
      assert.equal(name, `event: ${event.type}`);
      yield event;
    }
  }
  assert.equal(buffer + decoder.decode(), '', 'stream ends at a complete SSE frame');
}

test('HTTP JSON and SSE expose identical tokens, incremental text, request IDs and usage', async t => {
  const { generate, engine, service } = await fixture(t);
  const prompt = '你好';
  const [plainResponse, streamResponse] = await Promise.all([
    generate({ prompt, maxNewTokens: 6 }), generate({ prompt, maxNewTokens: 6, stream: true }),
  ]);
  assert.equal(plainResponse.status, 200);
  const plain = await plainResponse.json();
  const events: WireEvent[] = [];
  for await (const event of readEvents(streamResponse)) events.push(event);
  assert.equal(events[0].type, 'queued');
  assert.equal(events.at(-1)?.type, 'finished');
  assert.equal(events.filter(e => e.type === 'finished').length, 1);
  assert.ok(events.every(e => e.id === streamResponse.headers.get('x-request-id')));
  assert.equal(plain.id, plainResponse.headers.get('x-request-id'));
  assert.notEqual(plain.id, events[0].id);
  const expected: number[] = [];
  for (let i = 0; i < 6; i++) expected.push(argmax(engine.model.forwardDense([...encode(prompt), ...expected])));
  assert.deepEqual(plain.output, expected);
  assert.deepEqual(events.filter(e => e.type === 'token').map(e => e.token), expected);
  assert.equal(events.map(e => e.textDelta).join(''), plain.text);
  assert.deepEqual(plain.usage, { promptTokens: encode(prompt).length, completionTokens: 6 });
  assert.equal(service.pendingCount, 0);
  const admitted = engine.trace.filter(e => e.type === 'admitted');
  const firstFinished = engine.trace.find(e => e.type === 'finished')!;
  assert.ok(admitted.filter(e => e.tick <= firstFinished.tick).length >= 2, 'HTTP connections share an engine before either finishes');
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
});

test('API rejects malformed and oversized input before streaming or admission', async t => {
  const { url, generate, service } = await fixture(t, new Engine({ recordTrace: false }), { maxBodyBytes: 128 });
  for (const body of [null, [], {}, { prompt: [null] }, { prompt: ['1'] }, { prompt: [-1] },
    { prompt: [257] }, { prompt: [] }, { prompt: 'x', stream: 'yes' },
    { prompt: 'x', maxNewTokens: null }, { prompt: 'x', maxNewTokens: 300 }, { prompt: 'x', model: 'unknown' },
    { prompt: 'x', messages: [] }, { messages: [{ role: 'user', content: 'hi' }] }]) {
    const response = await generate(body);
    assert.equal(response.status, 400, JSON.stringify(body));
    assert.equal((await response.json()).error.code, 'invalid_request');
  }
  const malformed = await fetch(`${url}/generate`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: '{',
  });
  assert.equal(malformed.status, 400);
  await malformed.body?.cancel();
  const wrongType = await fetch(`${url}/generate`, { method: 'POST', body: '{}' });
  assert.equal(wrongType.status, 415);
  await wrongType.body?.cancel();
  const large = await generate({ prompt: 'x'.repeat(200), stream: true });
  assert.equal(large.status, 413);
  assert.equal((await large.json()).error.code, 'body_too_large');
  assert.equal(service.pendingCount, 0);
});

test('client disconnect cancels running generation and releases its live KV', async t => {
  const engine = new Engine({ maxBatchTokens: 1, prefillChunkSize: 1 });
  const { generate, service, url } = await fixture(t, engine);
  const controller = new AbortController();
  const response = await generate({ prompt: [1], maxNewTokens: 200, stream: true }, controller.signal);
  const id = response.headers.get('x-request-id')!;
  const reader = response.body!.getReader();
  await reader.read();
  assert.ok(service.pendingCount > 0);
  controller.abort();
  for (let i = 0; i < 500 && service.pendingCount; i++) await nextTurn();
  assert.equal(service.pendingCount, 0);
  assert.ok(engine.trace.some(e => e.id === id && e.type === 'cancelled'));
  assert.ok(!engine.trace.some(e => e.id === id && e.type === 'finished'));
  assert.ok(engine.cache.snapshot().pages.every(p => p.activeRefs === 0));
  await reader.cancel().catch(() => {});
  const health = await fetch(`${url}/health`);
  assert.deepEqual(await health.json(), { status: 'ok', pendingRequests: 0 });
});

test('explicit cancellation ends the matching SSE stream and leaves a sibling running', async t => {
  const engine = new Engine({ maxBatchTokens: 2, prefillChunkSize: 1 });
  const { generate, url } = await fixture(t, engine);
  const response = await generate({ prompt: [1], maxNewTokens: 200, stream: true });
  const sibling = generate({ prompt: [2], maxNewTokens: 10 });
  const id = response.headers.get('x-request-id')!;
  const cancelled = await fetch(`${url}/requests/${id}`, { method: 'DELETE' });
  assert.equal(cancelled.status, 200);
  assert.deepEqual(await cancelled.json(), { id, cancelled: true });
  const events: WireEvent[] = [];
  for await (const event of readEvents(response)) events.push(event);
  assert.equal(events.at(-1)?.type, 'cancelled');
  assert.equal(events.filter(e => e.type === 'cancelled').length, 1);
  assert.equal((await (await sibling).json()).output.length, 10);
  const again = await fetch(`${url}/requests/${id}`, { method: 'DELETE' });
  assert.equal(again.status, 404);
  await again.body?.cancel();
});

test('HTTP model failures produce a terminal SSE error and a closed health status', async t => {
  class BrokenModel extends TinyTransformer {
    override forwardToken(): never { throw new Error('simulated backend failure'); }
  }
  const engine = new Engine({ recordTrace: false }, new BrokenModel());
  const { generate, url } = await fixture(t, engine);
  const response = await generate({ prompt: [1], maxNewTokens: 2, stream: true });
  const events: WireEvent[] = [];
  for await (const event of readEvents(response)) events.push(event);
  assert.deepEqual(events.map(e => e.type), ['queued', 'error']);
  assert.equal((events.at(-1) as { code: string }).code, 'engine_error');
  assert.equal(engine.cache.snapshot().freePages, engine.options.numBlocks);
  const health = await fetch(`${url}/health`);
  assert.equal(health.status, 503);
  await health.body?.cancel();
  const retry = await generate({ prompt: [1], maxNewTokens: 1 });
  assert.equal(retry.status, 503);
  assert.equal((await retry.json()).error.code, 'closed');
});
