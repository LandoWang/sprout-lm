import { createReadStream } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { join } from 'node:path';
import type { ObservationEvent } from './observation.ts';

/** Read retained records; do not pretend an active, rotated or lossy trace is complete. */
export async function readRequestTrace(directory: string, requestId: string, limit = 20000) {
  const manifest = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8'));
  const files = (await readdir(directory)).filter(n => /^events-\d{8}\.ndjson$/.test(n)).sort();
  const events: ObservationEvent[] = [], losses: ObservationEvent[] = [];
  const lifecycle: ObservationEvent[] = [];
  let invalidLines = 0, omittedEvents = 0, unavailableFiles = 0;
  for (const name of files) {
    const stream = createReadStream(join(directory, name), { encoding: 'utf8' });
    try {
      for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
        try {
          const event = JSON.parse(line) as ObservationEvent;
          if (event.runId !== manifest.runId) { invalidLines++; continue; }
          if (event.name === 'observer.events_dropped' && losses.length < 100) losses.push(event);
          if (!event.requestId && /^(observer\.|worker\.|workers\.|scheduler\.|serving\.)/.test(event.name)) {
            lifecycle.push(event); if (lifecycle.length > 100) lifecycle.shift();
          }
          if (event.requestId === requestId) {
            if (events.length < limit) events.push(event); else omittedEvents++;
          }
        } catch { invalidLines++; }
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') unavailableFiles++; else throw error;
    } finally { stream.destroy(); }
  }
  events.sort((a,b) => BigInt(a.timeNs) < BigInt(b.timeNs) ? -1 : BigInt(a.timeNs) > BigInt(b.timeNs) ? 1 : a.seq - b.seq);
  return { requestId, runId: manifest.runId, events, diagnostics: {
    cleanShutdown: manifest.cleanShutdown === true, rotatedFiles: manifest.rotatedFiles,
    retainedFiles: files.length, invalidLines, unavailableFiles, omittedEvents, losses, lifecycle,
    completeness: 'best-effort; missing records, process crashes and expired retention may leave gaps',
  } };
}
