import { createServer, type Socket } from 'node:net';
import { open, unlink, writeFile, rename, type FileHandle } from 'node:fs/promises';
import { join } from 'node:path';
import { MAX_EVENT_BYTES, type ObservationEvent } from './observation.ts';

if (!process.send) throw new Error('Observer must be started by its coordinator.');
interface Options { directory: string; socketPath: string; runId: string; maxFileBytes?: number; maxFiles?: number; maxQueueBytes?: number }
let stop: (() => Promise<void>) | undefined;
let stopping = false;
process.on('disconnect', () => { if (stop) void stop(); else process.exit(1); });
process.on('message', message => {
  const m = message as { op: string; options: Options };
  if (m.op === 'start' && !stop) void start(m.options).catch(error => {
    console.error('[observation]', error); process.exit(1);
  });
  else if (m.op === 'stop') void stop?.();
});

async function start(options: Options): Promise<void> {
  const maxFileBytes = options.maxFileBytes ?? 8 * 1024 * 1024;
  const maxFiles = options.maxFiles ?? 4, maxQueueBytes = options.maxQueueBytes ?? 1024 * 1024;
  if (![maxFileBytes, maxFiles, maxQueueBytes].every(n => Number.isSafeInteger(n) && n > 0)
    || maxFileBytes < MAX_EVENT_BYTES || maxFiles < 2 || maxQueueBytes < MAX_EVENT_BYTES)
    throw new Error('Invalid observer retention/buffer limits.');
  let file: FileHandle | undefined, fileBytes = 0, index = 0, rotatedFiles = 0, collectorSeq = 0;
  let queuedBytes = 0, pumping: Promise<void> | undefined, fatal = false;
  let received = 0, written = 0, dropped = 0, unreportedDrops = 0, malformed = 0;
  const sockets = new Set<Socket>();
  const queue: Array<{ line?: string; bytes: number; ack?: () => void }> = [];
  const filename = (n: number) => join(options.directory, 'events-' + String(n).padStart(8, '0') + '.ndjson');
  const own = (name: string, data: Record<string, unknown>): ObservationEvent => ({
    version: 1, runId: options.runId, name, source: 'collector', pid: process.pid, seq: ++collectorSeq,
    timeNs: process.hrtime.bigint().toString(), wallTimeMs: Date.now(), data,
  });
  const manifest = async (clean: boolean) => {
    const path = join(options.directory, 'manifest.json');
    await writeFile(path + '.tmp', JSON.stringify({ version: 1, runId: options.runId, collectorPid: process.pid,
      maxFileBytes, maxFiles, maxQueueBytes, currentFile: index, rotatedFiles, received, written, dropped, malformed,
      cleanShutdown: clean, clock: 'same-host process.hrtime.bigint (ns); wallTimeMs for display', updatedAt: new Date().toISOString() }, null, 2));
    await rename(path + '.tmp', path);
  };
  const append = async (line: string) => {
    const bytes = Buffer.byteLength(line);
    if (!file || fileBytes + bytes > maxFileBytes) {
      if (file) { await file.close(); index++; }
      file = await open(filename(index), 'wx', 0o600); fileBytes = 0;
      if (index >= maxFiles) { await unlink(filename(index - maxFiles)); rotatedFiles++; }
      await manifest(false);
    }
    await file.appendFile(line); fileBytes += bytes; written++;
  };
  const pump = () => {
    if (pumping || fatal) return;
    pumping = (async () => {
      while (queue.length || unreportedDrops) {
        if (unreportedDrops) {
          const count = unreportedDrops; unreportedDrops = 0;
          await append(JSON.stringify(own('observer.events_dropped', { count, reason: 'collector_queue_full' })) + '\n');
        }
        const item = queue.shift();
        if (!item) continue;
        // Include the active write in the memory budget until it completes.
        if (item.line) await append(item.line);
        queuedBytes -= item.bytes;
        item.ack?.(); // All earlier accepted events are now written (not fsync'ed).
      }
    })().catch(error => {
      fatal = true; console.error('[observation] log writer failed:', error); process.exit(1);
    }).finally(() => { pumping = undefined; if (queue.length || unreportedDrops) pump(); });
  };
  const enqueue = (event: ObservationEvent) => {
    const line = JSON.stringify(event) + '\n', bytes = Buffer.byteLength(line);
    if (queuedBytes + bytes > maxQueueBytes) { dropped++; unreportedDrops++; pump(); return; }
    queue.push({ line, bytes }); queuedBytes += bytes; pump();
  };
  const server = createServer(socket => {
    if (stopping || sockets.size >= 16) { socket.destroy(); return; }
    sockets.add(socket); socket.on('error', () => {}); socket.setEncoding('utf8');
    let buffer = '', barrierPending = false, identity: { source: string; pid: number } | undefined;
    socket.on('data', chunk => {
      buffer += chunk;
      let end: number;
      while ((end = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        if (Buffer.byteLength(line) > MAX_EVENT_BYTES) { malformed++; socket.destroy(); return; }
        try {
          const event = JSON.parse(line);
          if (Number.isSafeInteger(event.flush) && event.flush > 0) {
            if (barrierPending || queuedBytes + 64 > maxQueueBytes) { socket.destroy(); return; }
            barrierPending = true;
            queue.push({ bytes: 64, ack: () => {
              barrierPending = false;
              if (!socket.destroyed && socket.writableLength < 1024) socket.write(JSON.stringify({ ack: event.flush }) + '\n');
            } }); queuedBytes += 64; pump(); continue;
          }
          if (event.version !== 1 || event.runId !== options.runId || !['coordinator','prefill','decode'].includes(event.source)
            || !Number.isSafeInteger(event.pid) || !Number.isSafeInteger(event.seq) || event.seq < 1
            || typeof event.name !== 'string' || typeof event.timeNs !== 'string' || !/^\d{1,24}$/.test(event.timeNs)
            || !Number.isFinite(event.wallTimeMs) || (event.requestId !== undefined && typeof event.requestId !== 'string')
            || (event.planId !== undefined && typeof event.planId !== 'string'))
            throw new Error('Invalid event.');
          if (identity && (identity.source !== event.source || identity.pid !== event.pid)) throw new Error('Producer identity changed.');
          identity ??= { source: event.source, pid: event.pid }; received++; enqueue(event);
        } catch { malformed++; socket.destroy(); return; }
      }
      if (Buffer.byteLength(buffer) > MAX_EVENT_BYTES) { malformed++; socket.destroy(); }
    });
    socket.on('close', () => {
      sockets.delete(socket);
      if (buffer) malformed++;
      enqueue(own('observer.producer_disconnected', { ...identity, partialLine: !!buffer }));
    });
  });
  server.on('error', error => { console.error('[observation]', error); process.exit(1); });
  stop = async () => {
    if (stopping) return; stopping = true;
    server.close();
    // Producers normally close after a persistence barrier. Bound abnormal shutdown.
    const deadline = Date.now() + 500;
    while (sockets.size && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
    for (const socket of sockets) socket.destroy();
    await new Promise(resolve => setImmediate(resolve));
    enqueue(own('observer.stopped', { received, dropped, malformed }));
    while (pumping || queue.length) { pump(); await pumping; }
    await file?.close(); await manifest(true);
    process.disconnect(); process.exit(0);
  };
  await manifest(false);
  server.listen(options.socketPath, () => {
    enqueue(own('observer.started', { maxFileBytes, maxFiles, maxQueueBytes }));
    process.send!({ ready: true });
  });
}
