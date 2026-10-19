import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const [baseUrl, smokePath] = process.argv.slice(2);
if (!baseUrl || !smokePath) throw new Error('Usage: node scripts/check-deployment.ts URL CHAT_SMOKE_JSON');
const expected = JSON.parse(await readFile(smokePath, 'utf8'));
assert.equal(expected.chat, true);
const health = await fetch(baseUrl + '/health', { signal: AbortSignal.timeout(10000) });
assert.equal(health.status, 200);
const post = (body: object) => fetch(baseUrl + '/generate', { method: 'POST',
  headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(120000) });
const request = { messages: [{ role: 'user', content: expected.prompt }], maxNewTokens: 64 };
const before = performance.now();
const jsonResponse = await post(request); assert.equal(jsonResponse.status, 200);
const json = await jsonResponse.json();
assert.deepEqual(json.output, expected.output); assert.equal(json.text, expected.text); assert.equal(json.finishReason, expected.finishReason);
const jsonMs = performance.now() - before;

async function* frames(response: Response): AsyncGenerator<Record<string, any>> {
  assert.ok(response.body); const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end: number;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
        const data = frame.split('\n').find(line => line.startsWith('data: '));
        if (data) yield JSON.parse(data.slice(6));
      }
    }
  } finally { await reader.cancel(); reader.releaseLock(); }
}
const streamStart = performance.now();
const streamResponse = await post({ ...request, stream: true }); assert.equal(streamResponse.status, 200);
const streamed = []; let firstTokenMs: number | undefined;
for await (const frame of frames(streamResponse)) { if (frame.type === 'token') firstTokenMs ??= performance.now() - streamStart; streamed.push(frame); }
assert.deepEqual(streamed.filter(e => e.type === 'token').map(e => e.token), expected.output);
assert.equal(streamed.map(e => e.textDelta ?? '').join(''), expected.text);
assert.equal(streamed.at(-1)?.finishReason, expected.finishReason);
const streamMs = performance.now() - streamStart;

const pending = await post({ messages: [{ role: 'user', content: '这是取消测试。'.repeat(12) }], maxNewTokens: 128, stream: true });
assert.equal(pending.status, 200);
const iterator = frames(pending), first = await iterator.next();
assert.equal(first.value?.type, 'queued'); const cancelledId = first.value!.id;
const siblingPromise = post(request);
const cancelled = await fetch(baseUrl + '/requests/' + encodeURIComponent(cancelledId), { method: 'DELETE', signal: AbortSignal.timeout(30000) });
assert.equal(cancelled.status, 200);
const terminal = []; for await (const frame of iterator) terminal.push(frame);
assert.equal(terminal.at(-1)?.type, 'cancelled');
const siblingResponse = await siblingPromise; assert.equal(siblingResponse.status, 200);
const sibling = await siblingResponse.json(); assert.deepEqual(sibling.output, expected.output);
const tooLarge = await post({ prompt: 'Hello', maxNewTokens: 3000 }); assert.equal(tooLarge.status, 400);
const finalHealth = await (await fetch(baseUrl + '/health')).json(); assert.equal(finalHealth.pendingRequests, 0);
const report = { model: expected.model.id, baseUrl, text: json.text, output: json.output, usage: json.usage,
  finishReason: json.finishReason, jsonMs, streaming: { firstTokenMs, totalMs: streamMs,
    note: 'After JSON request; shared prefix may be reused. Not a cold prefill measurement.' },
  cancellation: 'cancelled one stream while sibling completed with expected output',
  oversizedRequestStatus: tooLarge.status, health: finalHealth };
console.log(JSON.stringify(report));
if (process.env.REPORT_PATH) {
  const file = resolve(process.env.REPORT_PATH); await mkdir(dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(report, null, 2) + '\n');
}
