import { randomUUID } from 'node:crypto';
import { Engine, type Event } from './engine.ts';
import type { FinishReason } from './runner.ts';

/** A scheduler may execute locally or await completion from worker processes. */
export interface GenerationEngine {
  readonly pending: boolean;
  readonly failed?: boolean;
  readonly cache: { clearPrefixes(): void };
  add(input: { id: string; prompt: readonly number[]; maxNewTokens: number }): void;
  cancel(id: string): boolean;
  forget(id: string): boolean;
  step(): { events: Event[] } | Promise<{ events: Event[] }>;
  shutdown?(): Promise<void>;
}

export interface GenerationRequest {
  prompt: readonly number[];
  maxNewTokens: number;
}
export type ErrorCode = 'invalid_request' | 'overloaded' | 'closed' | 'engine_error' | 'slow_consumer';
export class GenerationError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = 'GenerationError';
    this.code = code;
  }
}
export type GenerationEvent = { id: string } & (
  | { type: 'queued' }
  | { type: 'token'; token: number; index: number }
  | { type: 'finished'; finishReason: FinishReason; usage: { promptTokens: number; completionTokens: number } }
  | { type: 'cancelled' }
  | { type: 'error'; code: ErrorCode; message: string }
);
type TerminalEvent = Exclude<GenerationEvent, { type: 'queued' | 'token' }>;
export interface GenerationStream extends AsyncIterableIterator<GenerationEvent> {
  readonly id: string;
}

/** One consumer per stream. A terminal event has its own slot, even if the buffer is full. */
class EventStream implements GenerationStream {
  readonly id: string;
  private readonly limit: number;
  private readonly onReturn: () => void;
  private readonly buffer: GenerationEvent[] = [];
  private readonly readers: Array<(value: IteratorResult<GenerationEvent>) => void> = [];
  private terminal?: TerminalEvent;
  private ended = false;

  constructor(id: string, limit: number, onReturn: () => void) {
    this.id = id;
    this.limit = limit;
    this.onReturn = onReturn;
  }

  [Symbol.asyncIterator](): GenerationStream { return this; }

  next(): Promise<IteratorResult<GenerationEvent>> {
    const event = this.buffer.shift();
    if (event) return Promise.resolve({ value: event, done: false });
    if (this.terminal) {
      const terminal = this.terminal;
      this.terminal = undefined;
      this.ended = true;
      return Promise.resolve({ value: terminal, done: false });
    }
    if (this.ended) return Promise.resolve({ value: undefined, done: true });
    return new Promise(resolve => this.readers.push(resolve));
  }

  push(event: GenerationEvent): boolean {
    if (this.ended || this.terminal) return false;
    const reader = this.readers.shift();
    if (reader) reader({ value: event, done: false });
    else {
      if (this.buffer.length >= this.limit) return false;
      this.buffer.push(event);
    }
    return true;
  }

  finish(event: TerminalEvent): void {
    if (this.ended || this.terminal) return;
    const reader = this.readers.shift();
    if (reader) {
      reader({ value: event, done: false });
      this.endReaders();
    } else this.terminal = event;
  }

  async return(): Promise<IteratorResult<GenerationEvent>> {
    this.onReturn(); // Also cancels streams that have never been iterated.
    this.buffer.length = 0;
    this.terminal = undefined;
    this.endReaders();
    return { value: undefined, done: true };
  }

  private endReaders(): void {
    this.ended = true;
    for (const resolve of this.readers.splice(0)) resolve({ value: undefined, done: true });
  }
}

interface Job {
  stream: EventStream;
  promptTokens: number;
  completionTokens: number;
  removeAbortListener: () => void;
}
export interface ServiceOptions {
  maxPendingRequests: number;
  maxBufferedEvents: number;
}

/** Owns its Engine exclusively: one step per event-loop turn, shared by every client. */
export class GenerationService {
  private readonly engine: GenerationEngine;
  private readonly options: ServiceOptions;
  private readonly jobs = new Map<string, Job>();
  private scheduled?: NodeJS.Immediate;
  private closed = false;
  private pumping = false;
  private stopping?: Promise<void>;

  constructor(engine: GenerationEngine = new Engine({ recordTrace: false }), options: Partial<ServiceOptions> = {}) {
    if (engine.pending) throw new Error('GenerationService needs an idle Engine and exclusive ownership.');
    this.engine = engine;
    this.options = { maxPendingRequests: 128, maxBufferedEvents: 64, ...options };
    for (const [name, value] of Object.entries(this.options)) {
      if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
    }
  }

  get pendingCount(): number { return this.jobs.size; }
  get isClosed(): boolean { return this.closed || this.engine.failed === true; }

  generate(input: GenerationRequest, { signal }: { signal?: AbortSignal } = {}): GenerationStream {
    if (this.isClosed) throw new GenerationError('closed', 'Generation service is closed.');
    const id = randomUUID();
    const stream = new EventStream(id, this.options.maxBufferedEvents, () => { this.cancel(id); });
    if (signal?.aborted) {
      stream.finish({ id, type: 'cancelled' });
      return stream;
    }
    if (this.jobs.size >= this.options.maxPendingRequests) {
      throw new GenerationError('overloaded', 'Too many pending requests.');
    }
    try {
      this.engine.add({ ...input, id });
    } catch (error) {
      throw new GenerationError('invalid_request', error instanceof Error ? error.message : 'Invalid request.');
    }
    const abort = () => { this.cancel(id); };
    const job: Job = {
      stream, promptTokens: input.prompt.length, completionTokens: 0,
      removeAbortListener: () => signal?.removeEventListener('abort', abort),
    };
    this.jobs.set(id, job);
    stream.push({ id, type: 'queued' });
    signal?.addEventListener('abort', abort, { once: true });
    this.schedule();
    return stream;
  }

  cancel(id: string): boolean {
    if (!this.jobs.has(id)) return false;
    this.finish(id, { id, type: 'cancelled' });
    return true;
  }

  close(): void {
    this.closed = true;
    for (const id of this.jobs.keys()) this.cancel(id);
    this.engine.cache.clearPrefixes();
    this.stopping ??= Promise.resolve().then(() => this.engine.shutdown?.());
    void this.stopping.catch(() => {}); // The runtime can await drained() to observe shutdown errors.
  }
  async drained(): Promise<void> { await this.stopping; }

  private schedule(): void {
    if (!this.scheduled && !this.pumping && this.jobs.size && !this.closed) {
      this.scheduled = setImmediate(() => { void this.pump(); });
    }
  }

  private async pump(): Promise<void> {
    this.scheduled = undefined;
    this.pumping = true;
    try {
      const report = await this.engine.step();
      for (const event of report.events) {
        const job = this.jobs.get(event.id);
        if (!job) continue;
        if (event.type === 'token') {
          job.completionTokens++;
          if (!job.stream.push({ id: event.id, type: 'token', token: event.token, index: event.index })) {
            this.finish(event.id, {
              id: event.id, type: 'error', code: 'slow_consumer',
              message: 'Output buffer is full; request cancelled. Consume events as they arrive.',
            });
          }
        } else if (event.type === 'finished') {
          this.finish(event.id, {
            id: event.id, type: 'finished', finishReason: event.finishReason,
            usage: { promptTokens: job.promptTokens, completionTokens: job.completionTokens },
          });
        }
      }
    } catch (error) {
      // A failed forward may have partially written KV. Retire the engine instead of reusing it.
      this.closed = true;
      for (const id of this.jobs.keys()) this.finish(id, {
        id, type: 'error', code: 'engine_error',
        message: error instanceof Error ? error.message : 'Model execution failed.',
      });
      this.engine.cache.clearPrefixes();
      this.close();
    }
    this.pumping = false;
    this.schedule();
  }

  private finish(id: string, event: TerminalEvent): void {
    const job = this.jobs.get(id);
    if (!job) return;
    this.engine.cancel(id); // Finished requests are already released; cancellation is idempotent.
    this.engine.forget(id);
    this.jobs.delete(id);
    job.removeAbortListener();
    job.stream.finish(event);
    if (!this.jobs.size && this.scheduled) {
      clearImmediate(this.scheduled);
      this.scheduled = undefined;
    }
  }
}
