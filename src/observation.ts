import { createConnection, type Socket } from 'node:net';

export interface EventFields {
  requestId?: string; planId?: string; data?: Record<string, unknown>;
}
export type EventSink = (name: string, fields?: EventFields) => void;
export interface ObservationEvent extends EventFields {
  version: 1; runId: string; name: string; source: string; pid: number; seq: number;
  timeNs: string; wallTimeMs: number;
}
export interface EventConnection { socketPath: string; runId: string }
export const MAX_EVENT_BYTES = 8192;

/** Instrumentation is never allowed to change serving control flow. */
export function record(sink: EventSink | undefined, name: string, fields?: EventFields): void {
  try { sink?.(name, fields); } catch { /* Best effort, including caller-provided sinks. */ }
}

/** Independent, bounded Unix socket. No inference RPC or GPU synchronization is used. */
export class EventClient {
  private readonly socket: Socket;
  private seq = 0;
  private lost = 0;
  private firstLost = 0;
  private lastLost = 0;
  private dropped = 0;
  private dead = false;
  private closing?: Promise<void>;
  private barriers = new Map<number, (ok: boolean) => void>();
  private barrier = 0;
  private input = '';
  private warned = false;
  private pendingBytes = 0;
  readonly connection: EventConnection;
  readonly source: string;
  private readonly maxBufferedBytes: number;
  readonly emit: EventSink;
  constructor(connection: EventConnection, source: string, maxBufferedBytes = 65536) {
    if (!Number.isSafeInteger(maxBufferedBytes) || maxBufferedBytes < 1024) throw new Error('Invalid observation buffer size.');
    this.connection = connection; this.source = source; this.maxBufferedBytes = maxBufferedBytes;
    this.socket = createConnection({ path: connection.socketPath });
    this.emit = (name, fields) => {
      if (this.closing) return;
      this.reportLoss();
      const event = this.event(name, fields);
      if (!this.write(event)) {
        this.dropped++; this.lost++;
        if (!this.firstLost) this.firstLost = event.seq;
        this.lastLost = event.seq;
      }
    };
    this.socket.on('connect', () => this.reportLoss());
    this.socket.on('drain', () => this.reportLoss());
    this.socket.on('error', () => {
      if (!this.warned) { this.warned = true; console.error('[observation] collector unavailable; inference continues without event delivery.'); }
    });
    this.socket.on('close', () => {
      this.dead = true;
      for (const resolve of this.barriers.values()) resolve(false);
      this.barriers.clear();
    });
    this.socket.setEncoding('utf8');
    this.socket.on('data', chunk => {
      this.input += chunk;
      if (this.input.length > 16384) { this.socket.destroy(); return; }
      let end: number;
      while ((end = this.input.indexOf('\n')) >= 0) {
        const line = this.input.slice(0, end); this.input = this.input.slice(end + 1);
        try {
          const reply = JSON.parse(line);
          const resolve = this.barriers.get(reply.ack);
          if (resolve) { this.barriers.delete(reply.ack); resolve(true); }
        } catch { this.socket.destroy(); }
      }
    });
  }
  private event(name: string, fields?: EventFields): ObservationEvent {
    return { ...fields, version: 1, runId: this.connection.runId, name, source: this.source,
      pid: process.pid, seq: ++this.seq, timeNs: process.hrtime.bigint().toString(), wallTimeMs: Date.now() };
  }
  private write(event: ObservationEvent): boolean {
    try {
      const line = JSON.stringify(event) + '\n', bytes = Buffer.byteLength(line);
      if (this.dead || this.socket.destroyed || bytes > MAX_EVENT_BYTES
        || this.pendingBytes + bytes > this.maxBufferedBytes) return false;
      this.pendingBytes += bytes;
      this.socket.write(line, () => { this.pendingBytes -= bytes; this.reportLoss(); }); return true;
    } catch { return false; }
  }
  private reportLoss(): void {
    if (!this.lost || this.dead) return;
    const event = this.event('observer.events_dropped', {
      data: { count: this.lost, firstSeq: this.firstLost, lastSeq: this.lastLost, reason: 'producer_buffer_or_transport' },
    });
    if (this.write(event)) { this.lost = this.firstLost = this.lastLost = 0; }
    else this.seq--; // The loss marker itself is retried; it did not consume a source event.
  }
  stats() { return { dropped: this.dropped, unreportedDrops: this.lost, bufferedBytes: this.pendingBytes, connected: !this.dead && !this.socket.connecting }; }

  /** Bounded diagnostic/shutdown barrier, never called from the inference hot path. */
  flush(timeoutMs = 500): Promise<boolean> {
    if (this.dead || this.socket.destroyed || this.barriers.size) return Promise.resolve(false);
    this.reportLoss();
    if (this.lost || this.pendingBytes > this.maxBufferedBytes - 128) return Promise.resolve(false);
    return new Promise(resolve => {
      const id = ++this.barrier;
      const timer = setTimeout(() => { this.barriers.delete(id); resolve(false); }, timeoutMs);
      this.barriers.set(id, ok => { clearTimeout(timer); resolve(ok); });
      this.socket.write(JSON.stringify({ flush: id }) + '\n');
    });
  }
  close(timeoutMs = 500): Promise<void> {
    return this.closing ??= (async () => {
      const persisted = await this.flush(timeoutMs);
      if (!persisted && !this.dead) console.error('[observation] event flush incomplete at shutdown.');
      this.socket.destroy();
    })();
  }
}
