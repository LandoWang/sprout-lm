import { KVCache, PagedSequence } from './kv-cache.ts';
import { TinyTransformer, argmax } from './model.ts';

type Status = 'waiting' | 'running' | 'finished' | 'cancelled';
type Phase = 'prefill' | 'decode' | 'recompute';
interface Request {
  id: string;
  order: number;
  prompt: number[];
  maxNewTokens: number;
  output: number[];
  status: Status;
  kv?: PagedSequence;
  prefixHits: number;
}
export interface EngineOptions {
  blockSize: number;
  numBlocks: number;
  maxSequences: number;
  maxBatchTokens: number;
  prefillChunkSize: number;
  prefixCaching: boolean;
}
export interface Work {
  id: string;
  phase: Phase;
  from: number;
  tokens: number;
  blockTable: number[];
}
export interface Event {
  tick: number;
  type: 'queued' | 'admitted' | 'token' | 'finished' | 'cancelled' | 'preempted';
  id: string;
  detail: string;
}

export class Engine {
  readonly model: TinyTransformer;
  readonly cache: KVCache;
  readonly options: EngineOptions;
  readonly trace: Event[] = [];
  private readonly requests = new Map<string, Request>();
  private readonly active = new Map<string, Request>();
  private waiting: string[] = [];
  private tick = 0;

  constructor(options: Partial<EngineOptions> = {}, model = new TinyTransformer()) {
    this.options = {
      blockSize: 4, numBlocks: 64, maxSequences: 4,
      maxBatchTokens: 8, prefillChunkSize: 4, prefixCaching: true, ...options,
    };
    for (const [name, value] of Object.entries(this.options)) {
      if (name !== 'prefixCaching' && (!Number.isSafeInteger(value) || (value as number) <= 0)) {
        throw new Error(`${name} must be a positive integer.`);
      }
    }
    this.model = model;
    this.cache = new KVCache({
      blockSize: this.options.blockSize, numBlocks: this.options.numBlocks,
      layers: model.layers, width: model.width,
    });
  }

  add(input: { id: string; prompt: readonly number[]; maxNewTokens: number }): void {
    if (!input.id || this.requests.has(input.id)) throw new Error('Request ID must be nonempty and unique.');
    if (!input.prompt.length || input.prompt.some(t => !Number.isInteger(t) || t < 0 || t >= this.model.vocabSize)) {
      throw new Error('Prompt must contain valid token IDs.');
    }
    if (!Number.isSafeInteger(input.maxNewTokens) || input.maxNewTokens <= 0) throw new Error('maxNewTokens must be a positive integer.');
    // The final sampled token does not need a KV slot: it will never be fed back in.
    const slots = input.prompt.length + input.maxNewTokens - 1;
    if (slots > this.options.numBlocks * this.options.blockSize) {
      throw new Error('Request cannot fit alone in the KV pool. Increase numBlocks or shorten it.');
    }
    const request: Request = {
      id: input.id, order: this.requests.size, prompt: [...input.prompt], maxNewTokens: input.maxNewTokens,
      output: [], status: 'waiting', prefixHits: 0,
    };
    this.requests.set(request.id, request);
    this.waiting.push(request.id);
    this.emit('queued', request, `${request.prompt.length} prompt tokens`);
  }

  get pending(): boolean { return this.active.size > 0 || this.waiting.length > 0; }

  cancel(id: string): boolean {
    const request = this.requests.get(id);
    if (!request || request.status === 'finished' || request.status === 'cancelled') return false;
    request.kv?.release();
    request.kv = undefined;
    request.status = 'cancelled';
    this.active.delete(id);
    this.waiting = this.waiting.filter(candidate => candidate !== id);
    this.emit('cancelled', request, 'Released live KV references');
    return true;
  }

  result(id: string) {
    const request = this.requests.get(id);
    if (!request) throw new Error(`Unknown request: ${id}`);
    return {
      id, status: request.status, output: [...request.output],
      computedTokens: request.kv?.length ?? 0,
      blockTable: [...(request.kv?.blocks ?? [])], prefixHits: request.prefixHits,
    };
  }

  /** One scheduling iteration; callers can admit/cancel requests between iterations. */
  step() {
    this.tick++;
    const eventStart = this.trace.length;
    this.admit();
    let budget = this.options.maxBatchTokens;
    const work: Work[] = [];
    // Decode gets latency priority; chunked prefill uses the remaining token budget.
    const batch = [...this.active.values()].sort((a, b) =>
      Number(this.phase(b) === 'decode') - Number(this.phase(a) === 'decode') || a.order - b.order);
    for (const request of batch) {
      if (!budget) break;
      if (request.status !== 'running') continue; // Another request may have preempted it.
      const history = [...request.prompt, ...request.output];
      const phase = this.phase(request);
      const from = request.kv!.length;
      const remaining = history.length - from - Number(phase === 'recompute');
      const count = Math.min(budget, phase === 'decode' ? 1 : this.options.prefillChunkSize, remaining);
      let computed = 0;
      let blockTable: number[] = [];
      for (let i = 0; i < count; i++) {
        if (!this.reserve(request)) break;
        const kv = request.kv!;
        const logits = this.model.forwardToken(history[kv.length], kv.length, kv);
        kv.length++;
        computed++;
        budget--;
        blockTable = [...kv.blocks];
        if (this.options.prefixCaching && kv.length <= request.prompt.length && kv.length % this.options.blockSize === 0) {
          this.cache.publish(request.prompt.slice(0, kv.length), kv);
        }
        if (kv.length === history.length) {
          const token = argmax(logits);
          request.output.push(token);
          this.emit('token', request, String(token));
          if (request.output.length === request.maxNewTokens) {
            request.status = 'finished';
            kv.release();
            request.kv = undefined;
            this.active.delete(request.id);
            this.emit('finished', request, `${request.output.length} output tokens`);
          }
          break; // At most one output token per request per iteration.
        }
      }
      if (computed) work.push({ id: request.id, phase, from, tokens: computed, blockTable });
    }
    return { tick: this.tick, work, events: this.trace.slice(eventStart), memory: this.cache.snapshot() };
  }

  private phase(request: Request): Phase {
    if (!request.output.length) return 'prefill';
    return request.kv!.length < request.prompt.length + request.output.length - 1 ? 'recompute' : 'decode';
  }

  private admit(): void {
    while (this.active.size < this.options.maxSequences && this.waiting.length) {
      const request = this.requests.get(this.waiting.shift()!)!;
      const history = [...request.prompt, ...request.output];
      request.kv = this.options.prefixCaching ? this.cache.reusePrefix(history) : this.cache.create();
      request.prefixHits += request.kv.length;
      request.status = 'running';
      this.active.set(request.id, request);
      this.emit('admitted', request, `Reused ${request.kv.length} prefix tokens; pages [${request.kv.blocks.join(',')}]`);
    }
  }

  private reserve(request: Request): boolean {
    while (!request.kv!.reserveToken()) {
      // Protect older requests so a crowded pool still makes forward progress.
      const younger = [...this.active.values()]
        .filter(other => other.order > request.order)
        .sort((a, b) => b.order - a.order)[0];
      if (!younger) {
        this.preempt(request);
        return false;
      }
      this.preempt(younger);
    }
    return true;
  }

  private preempt(request: Request): void {
    request.kv!.release();
    request.kv = undefined;
    request.status = 'waiting';
    this.active.delete(request.id);
    this.waiting.push(request.id);
    this.emit('preempted', request, 'Keep output tokens; rebuild missing KV after readmission');
  }

  private emit(type: Event['type'], request: Request, detail: string): void {
    this.trace.push({ tick: this.tick, type, id: request.id, detail });
  }
}
