import type { KVStore, NumericVector as Vector, KVDtype, DeviceKVStorage } from './runner.ts';

interface PrefixBlock { page: number; lastUsed: number }

export class PagedSequence implements KVStore {
  readonly blocks: number[] = [];
  length = 0;
  released = false;
  private readonly pool: KVCache;

  constructor(pool: KVCache) { this.pool = pool; }

  reserveToken(): boolean {
    this.assertLive();
    if (this.length < this.blocks.length * this.pool.blockSize) return true;
    const page = this.pool.allocate();
    if (page === undefined) return false;
    this.blocks.push(page);
    return true;
  }

  set(layer: number, position: number, key: Vector, value: Vector): void {
    this.assertLive();
    // Only the new position is writable. Published full pages remain immutable.
    if (position !== this.length) throw new Error('KV writes must append to the sequence.');
    const slot = this.get(layer, position);
    if (key.length !== this.pool.width || value.length !== this.pool.width) throw new Error('KV width mismatch.');
    slot.key.set(key);
    slot.value.set(value);
  }

  get(layer: number, position: number): { key: Vector; value: Vector } {
    this.assertLive();
    if (!Number.isInteger(position) || position < 0 || position > this.length) throw new Error('Invalid KV position.');
    const page = this.blocks[Math.floor(position / this.pool.blockSize)];
    if (page === undefined) throw new Error('Reserve a physical page before writing a token.');
    return this.pool.slot(layer, page, position % this.pool.blockSize);
  }

  deviceView() {
    this.assertLive();
    if (!this.pool.deviceStorage) throw new Error('Sequence has no device KV storage.');
    return { storage: this.pool.deviceStorage, blocks: this.blocks, length: this.length };
  }

  release(): void {
    if (this.released) return;
    for (const page of this.blocks) this.pool.releasePage(page);
    this.blocks.length = 0;
    this.released = true;
  }

  private assertLive(): void {
    if (this.released) throw new Error('Sequence KV was released.');
  }
}

/** Physical K/V storage is fixed-size and shared across all requests. */
export class KVCache {
  readonly numBlocks: number;
  readonly blockSize: number;
  readonly layers: number;
  readonly width: number;
  readonly dtype: KVDtype;
  readonly deviceStorage?: DeviceKVStorage;
  private readonly keys: Vector[];
  private readonly values: Vector[];
  private readonly refs: number[];
  private readonly free: number[];
  private readonly prefixes = new Map<string, PrefixBlock>();
  private clock = 0;
  evictions = 0;

  constructor(options: { numBlocks: number; blockSize: number; layers: number; width: number; dtype?: KVDtype; deviceStorage?: DeviceKVStorage }) {
    for (const [name, value] of Object.entries(options)) {
      if (name === 'dtype' || name === 'deviceStorage') continue;
      if (!Number.isSafeInteger(value) || (value as number) <= 0) throw new Error(`${name} must be a positive integer.`);
    }
    this.numBlocks = options.numBlocks;
    this.blockSize = options.blockSize;
    this.layers = options.layers;
    this.width = options.width;
    this.dtype = options.dtype ?? 'float64';
    this.deviceStorage = options.deviceStorage;
    if (this.deviceStorage && (this.dtype !== 'float32' || ['numBlocks', 'blockSize', 'layers', 'width'].some(
      key => this.deviceStorage![key as keyof DeviceKVStorage] !== options[key as keyof typeof options]))) {
      throw new Error('Device KV layout mismatch.');
    }
    if (!['float32', 'float64'].includes(this.dtype)) throw new Error('Unsupported KV dtype.');
    const size = this.numBlocks * this.blockSize * this.width;
    const ArrayType = this.dtype === 'float32' ? Float32Array : Float64Array;
    this.keys = this.deviceStorage ? [] : Array.from({ length: this.layers }, () => new ArrayType(size));
    this.values = this.deviceStorage ? [] : Array.from({ length: this.layers }, () => new ArrayType(size));
    this.refs = Array(this.numBlocks).fill(0);
    this.free = Array.from({ length: this.numBlocks }, (_, i) => this.numBlocks - 1 - i);
  }

  create(): PagedSequence { return new PagedSequence(this); }

  reusePrefix(tokens: readonly number[]): PagedSequence {
    const sequence = this.create();
    // Always recompute the final input token to obtain logits. The cache stores K/V, not logits.
    for (let end = this.blockSize; end < tokens.length; end += this.blockSize) {
      const entry = this.prefixes.get(tokens.slice(0, end).join(','));
      if (!entry) break;
      this.refs[entry.page]++;
      entry.lastUsed = ++this.clock;
      sequence.blocks.push(entry.page);
      sequence.length = end;
    }
    return sequence;
  }

  publish(tokens: readonly number[], sequence: PagedSequence): void {
    if (sequence.released || sequence.length === 0 || sequence.length % this.blockSize !== 0) return;
    if (tokens.length !== sequence.length) throw new Error('Prefix tokens must match the computed KV length.');
    // Full preceding context is in the key: identical suffix blocks alone are NOT interchangeable.
    const key = tokens.join(',');
    if (this.prefixes.has(key)) return;
    const page = sequence.blocks.at(-1)!;
    this.refs[page]++; // One cache reference, in addition to live request references.
    this.prefixes.set(key, { page, lastUsed: ++this.clock });
  }

  allocate(): number | undefined {
    if (!this.free.length) {
      let candidate: [string, PrefixBlock] | undefined;
      for (const entry of this.prefixes) {
        if (this.refs[entry[1].page] === 1 && (!candidate || entry[1].lastUsed < candidate[1].lastUsed)) candidate = entry;
      }
      if (candidate) {
        this.prefixes.delete(candidate[0]);
        this.releasePage(candidate[1].page);
        this.evictions++;
      }
    }
    const page = this.free.pop();
    if (page !== undefined) this.refs[page] = 1;
    return page;
  }

  releasePage(page: number): void {
    if (!this.refs[page]) throw new Error('KV reference count underflow.');
    if (--this.refs[page] === 0) this.free.push(page);
  }

  clearPrefixes(): void {
    for (const entry of this.prefixes.values()) this.releasePage(entry.page);
    this.prefixes.clear();
  }

  slot(layer: number, page: number, offset: number): { key: Vector; value: Vector } {
    if (this.deviceStorage) throw new Error('Device KV cannot be read as a host array.');
    if (!Number.isInteger(layer) || layer < 0 || layer >= this.layers) throw new Error('Invalid layer.');
    const start = (page * this.blockSize + offset) * this.width;
    return {
      key: this.keys[layer].subarray(start, start + this.width),
      value: this.values[layer].subarray(start, start + this.width),
    };
  }

  snapshot() {
    const cached = new Set([...this.prefixes.values()].map(entry => entry.page));
    return {
      totalPages: this.numBlocks,
      freePages: this.free.length,
      cachedPages: cached.size,
      evictions: this.evictions,
      pages: this.refs.map((refs, id) => ({ id, refs, cached: cached.has(id), activeRefs: refs - Number(cached.has(id)) })),
    };
  }
}
