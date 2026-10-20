import { createReadStream } from 'node:fs';
import { lstat, readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { MAX_EVENT_BYTES, type ObservationEvent } from './observation.ts';

export function isObservationEvent(value: unknown): value is ObservationEvent {
  if (!value || typeof value !== 'object') return false;
  const e = value as ObservationEvent;
  return e.version === 1 && typeof e.runId === 'string' && typeof e.source === 'string'
    && typeof e.name === 'string' && typeof e.timeNs === 'string' && /^\d{1,24}$/.test(e.timeNs)
    && Number.isSafeInteger(e.pid) && Number.isSafeInteger(e.seq) && e.seq > 0 && Number.isFinite(e.wallTimeMs)
    && (e.requestId === undefined || typeof e.requestId === 'string')
    && (e.planId === undefined || typeof e.planId === 'string')
    && (e.data === undefined || (!!e.data && typeof e.data === 'object' && !Array.isArray(e.data)));
}

/** A bounded point-in-time read; the writer can keep appending or rotating files. */
export async function readEventLog(directory: string, limit = 100000, byteLimit = 64 * 1024 * 1024) {
  if (!Number.isSafeInteger(limit) || limit < 1 || !Number.isSafeInteger(byteLimit) || byteLimit < 1) throw new Error('Invalid read limits.');
  const manifestPath = join(directory, 'manifest.json');
  const manifestStat = await lstat(manifestPath);
  if (!manifestStat.isFile() || manifestStat.size > 65536) throw new Error('Invalid event manifest.');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (manifest.version !== 1 || typeof manifest.runId !== 'string') throw new Error('Invalid event manifest.');
  const files = (await readdir(directory)).filter(n => /^events-\d{8}\.ndjson$/.test(n)).sort();
  const events: ObservationEvent[] = [], seen = new Set<string>();
  let invalidLines = 0, partialLines = 0, unavailableFiles = 0, duplicateEvents = 0, bytesRead = 0;
  let truncated = false, evictedEvents = 0, cursor = 0;
  const identity = (event: ObservationEvent) => JSON.stringify([event.source,event.pid,event.seq]);
  for (const name of files) {
    if (bytesRead >= byteLimit) { truncated = true; break; }
    const path = join(directory, name);
    try {
      const info = await lstat(path);
      if (!info.isFile()) { unavailableFiles++; continue; }
      if (!info.size) continue;
      const length = Math.min(info.size, byteLimit - bytesRead);
      if (length < info.size) truncated = true;
      const stream = createReadStream(path, { end: length - 1 });
      let pending = Buffer.alloc(0), oversized = false;
      try {
        for await (const chunk of stream) {
          bytesRead += chunk.length;
          const buffer = Buffer.concat([pending, chunk]);
          let from = 0, end: number;
          while ((end = buffer.indexOf(10, from)) >= 0) {
            const line = buffer.subarray(from, end); from = end + 1;
            if (oversized || line.length > MAX_EVENT_BYTES) { invalidLines++; oversized = false; continue; }
            try {
              const event: unknown = JSON.parse(line.toString('utf8'));
              if (!isObservationEvent(event) || event.runId !== manifest.runId) { invalidLines++; continue; }
              const key = identity(event);
              if (seen.has(key)) { duplicateEvents++; continue; }
              seen.add(key);
              if (events.length < limit) events.push(event);
              else {
                // Keep a bounded recent window, so polling doesn't freeze on the
                // first 25k events of a still-growing log file.
                seen.delete(identity(events[cursor])); events[cursor] = event;
                cursor = (cursor+1)%limit; evictedEvents++; truncated = true;
              }
            } catch { invalidLines++; }
          }
          pending = buffer.subarray(from);
          if (pending.length > MAX_EVENT_BYTES) { oversized = true; pending = Buffer.alloc(0); }
        }
        if (pending.length || oversized) partialLines++;
      } finally { stream.destroy(); }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') unavailableFiles++; else throw error;
    }
  }
  events.sort((a, b) => {
    const delta = BigInt(a.timeNs) - BigInt(b.timeNs);
    return delta < 0n ? -1 : delta > 0n ? 1 : a.seq - b.seq;
  });
  const sequences = new Map<string, Set<number>>();
  for (const event of events) {
    const key = JSON.stringify([event.source, event.pid]);
    if (!sequences.has(key)) sequences.set(key, new Set());
    sequences.get(key)!.add(event.seq);
  }
  let sequenceGaps = 0;
  for (const seqs of sequences.values()) {
    const sorted = [...seqs].sort((a,b) => a-b);
    for (let i = 1; i < sorted.length; i++) sequenceGaps += sorted[i] - sorted[i-1] - 1;
  }
  return { runId: manifest.runId as string, events, diagnostics: {
    cleanShutdown: manifest.cleanShutdown === true, rotatedFiles: Number(manifest.rotatedFiles) || 0,
    retainedFiles: files.length, invalidLines, partialLines, unavailableFiles, duplicateEvents, sequenceGaps,
    truncated, evictedEvents, bytesRead, eventLimit: limit, byteLimit,
    losses: events.filter(e => e.name === 'observer.events_dropped').slice(-100),
    // A live manifest is updated at rotation/shutdown, not on each event.
    manifestUpdatedAt: manifest.updatedAt, capturedAt: new Date().toISOString(),
  } };
}
export type EventLog = Awaited<ReturnType<typeof readEventLog>>;
