import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, readdir, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test, { type TestContext } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { ObserverProcess, type ObserverOptions } from '../src/observer-process.ts';
import { EventClient } from '../src/observation.ts';
import { readRequestTrace } from '../src/request-trace.ts';
import { Engine } from '../src/engine.ts';
import { GenerationService } from '../src/service.ts';
import { createApiServer } from '../src/http.ts';

async function fixture(t: TestContext, options: Partial<ObserverOptions> = {}) {
  const root = await mkdtemp(join(tmpdir(), 'sprout-events-test-'));
  const observer = await ObserverProcess.start({ directory: root, ...options });
  t.after(async () => { await observer.close(); await rm(root, { recursive: true, force: true }); });
  return observer;
}
async function eventually(check: () => boolean) {
  const end = Date.now() + 3000;
  while (!check() && Date.now() < end) await delay(10);
  assert.ok(check(), 'Condition did not settle.');
}

test('independent collector correlates HTTP lifecycle under the server-generated request ID', async t => {
  const observer = await fixture(t);
  assert.notEqual(observer.child.pid, process.pid);
  const service = new GenerationService(new Engine({ recordTrace: false }), {}, observer.client.emit);
  const server = createApiServer({ service });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(async () => { service.close(); server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); });
  const address = server.address(); assert.ok(address && typeof address === 'object');
  const url = 'http://127.0.0.1:' + address.port;
  const request = async (stream: boolean) => {
    const response = await fetch(url + '/generate', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'hi', maxNewTokens: 3, stream }) });
    assert.equal(response.status, 200); await response.text();
    const id = response.headers.get('x-request-id')!;
    assert.ok(await observer.client.flush());
    const trace = await readRequestTrace(observer.directory, id);
    const names = trace.events.map(e => e.name);
    for (const name of ['request.received','http.body_read','text.encoded','service.queued','token.published','generation.finished','http.response_finished'])
      assert.ok(names.includes(name), name);
    assert.equal(names.filter(n => n === 'token.published').length, 3);
    assert.equal(names.filter(n => n === 'http.token_written').length, stream ? 3 : 0);
    assert.ok(names.indexOf('request.received') < names.indexOf('service.queued'));
    assert.ok(trace.events.every(e => e.requestId === id && e.source === 'coordinator'));
    assert.ok(trace.events.every(e => !('prompt' in (e.data ?? {})) && !('token' in (e.data ?? {}))), 'No text or token contents recorded');
    return id;
  };
  const first = await request(false), second = await request(true);
  assert.notEqual(first, second);
  // Trace collection is optional: a dead collector must not retire the generation engine.
  const exited = once(observer.child, 'exit'); observer.child.kill('SIGKILL'); await exited;
  const response = await fetch(url + '/generate', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt: 'hi', maxNewTokens: 3 }) });
  assert.equal(response.status, 200); assert.equal((await response.json()).output.length, 3);
  assert.equal(service.isClosed, false);
});

test('producer pressure stays bounded, reports losses, and retains cross-source plan identity', async t => {
  const observer = await fixture(t);
  const client = new EventClient(observer.connection, 'prefill', 2048);
  t.after(() => client.close());
  for (let i = 0; i < 1000; i++) client.emit('plan.started', { requestId: 'pressure', planId: 'P' + i });
  assert.ok(client.stats().dropped > 0);
  assert.ok(client.stats().bufferedBytes <= 2048);
  await eventually(() => client.stats().bufferedBytes === 0 && client.stats().unreportedDrops === 0);
  observer.client.emit('plan.sent', { requestId: 'kept', planId: 'prefill-7' });
  client.emit('plan.completed', { requestId: 'kept', planId: 'prefill-7' });
  assert.ok(await client.flush()); assert.ok(await observer.client.flush());
  const trace = await readRequestTrace(observer.directory, 'kept');
  assert.deepEqual(new Set(trace.events.map(e => e.source)), new Set(['coordinator','prefill']));
  assert.ok(trace.events.every(e => e.planId === 'prefill-7'));
  assert.ok(trace.diagnostics.losses.some(e => e.data?.reason === 'producer_buffer_or_transport'));
  await client.close();
});

test('collector rotation bounds disk usage and exposes expired retention', async t => {
  const observer = await fixture(t, { maxFileBytes: 8192, maxFiles: 2 });
  for (let batch = 0; batch < 80; batch++) {
    for (let i = 0; i < 4; i++) observer.client.emit('plan.sent', { requestId: 'rotating', data: { batch, i, detail: 'x'.repeat(100) } });
    assert.ok(await observer.client.flush());
  }
  observer.client.emit('generation.finished', { requestId: 'tail' });
  assert.ok(await observer.client.flush());
  await observer.close();
  const names = (await readdir(observer.directory)).filter(n => n.endsWith('.ndjson'));
  assert.equal(names.length, 2);
  for (const name of names) assert.ok((await stat(join(observer.directory, name))).size <= 8192);
  const trace = await readRequestTrace(observer.directory, 'tail');
  assert.equal(trace.events.length, 1);
  assert.equal(trace.diagnostics.cleanShutdown, true);
  assert.ok(trace.diagnostics.rotatedFiles > 0);
});

test('a stopped listener cannot grow producer memory or block close indefinitely', async t => {
  const observer = await fixture(t);
  const client = new EventClient(observer.connection, 'decode', 4096);
  await eventually(() => client.stats().connected);
  observer.child.kill('SIGSTOP');
  try {
    for (let i = 0; i < 20000; i++) client.emit('plan.completed', { requestId: 'blocked', data: { i } });
    assert.ok(client.stats().bufferedBytes <= 4096);
    assert.ok(client.stats().dropped > 0);
    const start = Date.now();
    await client.close(50);
    assert.ok(Date.now() - start < 1000);
  } finally { observer.child.kill('SIGCONT'); }
});

test('throwing instrumentation cannot change HTTP generation or reject malformed requests differently', async t => {
  const service = new GenerationService(new Engine({ recordTrace: false }), {}, () => { throw new Error('broken logger'); });
  const server = createApiServer({ service }); server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(() => { service.close(); server.closeAllConnections(); server.close(); });
  const address = server.address(); assert.ok(address && typeof address === 'object');
  const response = await fetch('http://127.0.0.1:' + address.port + '/generate', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: 'ok', maxNewTokens: 2 }),
  });
  assert.equal(response.status, 200); assert.equal((await response.json()).output.length, 2);
});
