import { KVCache, type PagedSequence } from './kv-cache.ts';
import type { Event } from './engine.ts';
import type { GenerationEngine } from './service.ts';
import type { PDExecutors, WorkerResult, WorkerRole } from './pd-protocol.ts';
import type { FinishReason, ModelSpec } from './runner.ts';

type Stage = 'waiting' | 'prefill' | 'transfer' | 'decode' | 'finished' | 'cancelled';
interface Request {
  id: string; order: number; prompt: number[]; maxNewTokens: number; output: number[]; stage: Stage;
  p?: PagedSequence; d?: PagedSequence; inFlight: boolean; forget: boolean;
}
export interface PDTrace {
  id: string; phase: 'prefill' | 'transfer' | 'decode'; worker: WorkerRole; from: number;
  tokens: number; sourcePages: number[]; targetPages: number[]; result: WorkerResult;
}
export interface PDSchedulerOptions { maxSequences?: number; prefillChunkSize?: number; traceLimit?: number }

/** The only owner of page references. Workers see immutable, numbered execution plans. */
export class PDScheduler implements GenerationEngine {
  readonly spec: ModelSpec;
  readonly prefillCache: KVCache;
  readonly decodeCache: KVCache;
  readonly cache = { clearPrefixes() {} }; // PD v1 deliberately has no prefix cache.
  readonly trace: PDTrace[] = [];
  private readonly workers: PDExecutors;
  private readonly maxSequences: number;
  private readonly chunk: number;
  private readonly traceLimit: number;
  private requests = new Map<string, Request>();
  private order = 0;
  private cursor = { prefill: -1, decode: -1 };
  private busy = { prefill: false, decode: false };
  private events: Event[] = [];
  private wake?: () => void;
  private stepping = false;
  private failure?: Error;
  private reportFailure!: (error: Error) => void;
  readonly failureSignal = new Promise<Error>(resolve => { this.reportFailure = resolve; });
  private stopping?: Promise<void>;
  private closed = false;
  private tick = 0;
  private transfers = { count: 0, bytes: 0, eventMs: 0 };
  constructor(workers: PDExecutors, options: PDSchedulerOptions = {}) {
    this.workers = workers; this.spec = workers.prefill.spec;
    this.maxSequences = options.maxSequences ?? 4; this.chunk = options.prefillChunkSize ?? 4; this.traceLimit = options.traceLimit ?? 0;
    if (![this.maxSequences,this.chunk].every(n => Number.isSafeInteger(n) && n > 0)
      || !Number.isSafeInteger(this.traceLimit) || this.traceLimit < 0) throw new Error('Invalid PD scheduler options.');
    const cache = (role: WorkerRole) => {
      const info = workers[role];
      // Opaque remote storage descriptors: the coordinator allocates no KV tensor.
      return new KVCache({ ...info.layout, dtype: 'float32', deviceStorage: { ...info.layout, owner: workers, handle: info } });
    };
    this.prefillCache = cache('prefill'); this.decodeCache = cache('decode');
    workers.onFailure(error => this.fail(error));
  }
  get failed(): boolean { return !!this.failure; }
  get pending(): boolean { return [...this.requests.values()].some(r => !this.terminal(r) || r.inFlight); }
  private terminal(r: Request) { return r.stage === 'finished' || r.stage === 'cancelled'; }
  add(input: { id: string; prompt: readonly number[]; maxNewTokens: number }): void {
    if (this.closed || this.failure) throw new Error('PD scheduler is closed.');
    if (!input.id || this.requests.has(input.id)) throw new Error('Request ID must be nonempty and unique.');
    if (!input.prompt.length || input.prompt.some(t => !Number.isInteger(t) || t < 0 || t >= this.spec.vocabSize)) throw new Error('Invalid prompt tokens.');
    if (!Number.isSafeInteger(input.maxNewTokens) || input.maxNewTokens < 1) throw new Error('Invalid maxNewTokens.');
    const slots = input.prompt.length + input.maxNewTokens - 1;
    if (slots > this.spec.maxContextLength || input.prompt.length > this.prefillCache.numBlocks * this.prefillCache.blockSize
      || slots > this.decodeCache.numBlocks * this.decodeCache.blockSize) throw new Error('Request exceeds model context or PD KV pool capacity.');
    this.requests.set(input.id, { ...input, prompt: [...input.prompt], output: [], order: this.order++, stage: 'waiting', inFlight: false, forget: false });
    this.wake?.();
  }
  cancel(id: string): boolean {
    const r = this.requests.get(id); if (!r || this.terminal(r)) return false;
    r.stage = 'cancelled';
    if (!r.inFlight && !this.failure) this.release(r);
    this.wake?.(); return true;
  }
  forget(id: string): boolean {
    const r = this.requests.get(id); if (!r || !this.terminal(r)) return false;
    r.forget = true;
    if (!r.inFlight && !r.p && !r.d) this.requests.delete(id);
    return true;
  }
  private release(r: Request): void {
    r.p?.release(); r.d?.release(); r.p = r.d = undefined;
    if (r.forget) this.requests.delete(r.id);
  }
  private reserve(pool: KVCache, tokens: number): PagedSequence | undefined {
    const n = Math.ceil(tokens / pool.blockSize);
    if (pool.snapshot().freePages < n) return;
    const sequence = pool.create();
    for (let i = 0; i < n; i++) sequence.blocks.push(pool.allocate()!);
    return sequence;
  }
  private admit(): void {
    let active = [...this.requests.values()].filter(r => r.p || r.d).length;
    for (const r of this.requests.values()) {
      if (active >= this.maxSequences) break;
      if (r.stage !== 'waiting') continue;
      const p = this.reserve(this.prefillCache, r.prompt.length);
      if (!p) break;
      const d = this.reserve(this.decodeCache, r.prompt.length + r.maxNewTokens - 1);
      if (!d) { p.release(); break; }
      r.p = p; r.d = d; r.stage = 'prefill'; active++;
    }
  }
  private choose(role: WorkerRole): Request | undefined {
    const eligible = [...this.requests.values()].filter(r => !r.inFlight && (role === 'prefill'
      ? r.stage === 'prefill' : r.stage === 'transfer' || r.stage === 'decode'));
    return eligible.find(r => r.order > this.cursor[role]) ?? eligible[0];
  }
  private schedule(): void {
    if (this.closed || this.failure) return;
    this.admit();
    for (const role of ['decode','prefill'] as const) {
      if (this.busy[role]) continue;
      const r = this.choose(role); if (!r) continue;
      this.cursor[role] = r.order; this.busy[role] = true; r.inFlight = true;
      const phase = r.stage as PDTrace['phase'];
      const from = phase === 'prefill' ? r.p!.length : phase === 'decode' ? r.d!.length : 0;
      const tokens = phase === 'prefill' ? Math.min(this.chunk, r.prompt.length - from) : phase === 'decode' ? 1 : r.prompt.length;
      const pageCount = Math.ceil((from + tokens) / this.prefillCache.blockSize);
      const sourcePages = phase === 'transfer' ? r.p!.blocks.slice(0,pageCount) : [];
      const targetPages = (phase === 'prefill' ? r.p! : r.d!).blocks.slice(0,pageCount);
      const command = phase === 'transfer' ? this.workers.transfer({ request: r.id, sourcePages, targetPages, tokens })
        : this.workers.forward(role, { request: r.id, from, blocks: targetPages,
          tokens: phase === 'prefill' ? r.prompt.slice(from,from+tokens) : [r.output.at(-1)!],
          output: phase === 'prefill' && from+tokens < r.prompt.length ? 'none' : 'sample' });
      void command.then(result => {
        if (result.request !== r.id) throw new Error('PD completion request identity mismatch.');
        if (this.traceLimit) {
          this.trace.push({ id: r.id, phase, worker: role, from, tokens, sourcePages, targetPages, result });
          if (this.trace.length > this.traceLimit) this.trace.shift();
        }
        if (this.failure || this.closed || this.terminal(r)) return;
        if (phase === 'transfer') {
          const expected = tokens * this.spec.layers * 2 * this.spec.kvHeads * this.spec.headDim * 4;
          if (result.bytes !== expected || !Number.isFinite(result.eventMs)) throw new Error('Invalid KV transfer acknowledgement.');
          this.transfers.count++; this.transfers.bytes += expected; this.transfers.eventMs += result.eventMs!;
          r.d!.length = r.prompt.length; r.p!.release(); r.p = undefined; r.stage = 'decode';
        } else {
          const kv = phase === 'prefill' ? r.p! : r.d!; kv.length += tokens;
          if (phase === 'prefill' && kv.length < r.prompt.length) return;
          const token = result.token;
          if (token === undefined || !Number.isInteger(token) || token < 0 || token >= this.spec.vocabSize) throw new Error('Invalid PD sampled token.');
          r.output.push(token);
          this.events.push({ type: 'token', tick: this.tick, id: r.id, detail: String(token), token, index: r.output.length-1 });
          const reason: FinishReason | undefined = this.spec.eosTokenIds.includes(token) ? 'eos' : r.output.length === r.maxNewTokens ? 'length' : undefined;
          if (reason) {
            r.stage = 'finished'; this.release(r);
            this.events.push({ type: 'finished', tick: this.tick, id: r.id, detail: reason, finishReason: reason });
          } else if (phase === 'prefill') r.stage = 'transfer';
        }
      }).catch(error => this.fail(error instanceof Error ? error : new Error(String(error))))
        .finally(() => {
          this.busy[role] = false; r.inFlight = false;
          if (this.terminal(r) && !this.failure && !this.closed) this.release(r);
          this.wake?.();
        });
    }
  }
  async step(): Promise<{ events: Event[] }> {
    if (this.stepping) throw new Error('Concurrent PD scheduler steps are forbidden.');
    this.stepping = true; this.tick++;
    try {
      if (this.failure) throw this.failure;
      this.schedule();
      if (!this.events.length && (this.busy.prefill || this.busy.decode)) await new Promise<void>(resolve => { this.wake = resolve; });
      if (this.failure) throw this.failure;
      return { events: this.events.splice(0) };
    } finally { this.wake = undefined; this.stepping = false; }
  }
  private fail(error: Error): void {
    if (this.failure || this.closed) return;
    this.failure = error; this.reportFailure(error); this.wake?.(); void this.shutdown();
  }
  shutdown(): Promise<void> {
    return this.stopping ??= (async () => {
      this.closed = true;
      for (const r of this.requests.values()) if (!this.terminal(r)) r.stage = 'cancelled';
      await this.workers.close(); // Exit of D before P proves both pools are no longer in use.
      for (const r of this.requests.values()) { r.inFlight = false; this.release(r); }
      this.busy.prefill = this.busy.decode = false; this.wake?.();
    })();
  }
  snapshot() {
    return { coordinatorPid: process.pid, workers: { prefill: { pid: this.workers.prefill.pid, device: this.workers.prefill.device },
      decode: { pid: this.workers.decode.pid, device: this.workers.decode.device } }, busy: { ...this.busy },
      requests: [...this.requests.values()].map(r => ({ id: r.id, stage: r.stage, inFlight: r.inFlight })),
      prefill: this.prefillCache.snapshot(), decode: this.decodeCache.snapshot(), transfers: { ...this.transfers } };
  }
}
