/** A real, very small causal Transformer. Seeded weights are NOT trained. */
export type Vector = Float64Array;
export interface KVStore {
  set(layer: number, position: number, key: Vector, value: Vector): void;
  get(layer: number, position: number): { key: Vector; value: Vector };
}

interface Layer {
  q: Vector; k: Vector; v: Vector; out: Vector;
  up: Vector; down: Vector;
}

export function argmax(values: Vector): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function linear(x: Vector, weights: Vector, rows: number): Vector {
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < x.length; j++) y[i] += weights[i * x.length + j] * x[j];
  }
  return y;
}

function norm(x: Vector): Vector {
  let sum = 0;
  for (const v of x) sum += v * v;
  const scale = 1 / Math.sqrt(sum / x.length + 1e-5);
  return x.map(v => v * scale);
}

function residual(x: Vector, delta: Vector): Vector {
  return x.map((v, i) => v + delta[i]);
}

function softmax(scores: Vector): Vector {
  let max = -Infinity;
  for (const score of scores) max = Math.max(max, score);
  const exps = scores.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

export class TinyTransformer {
  readonly width = 16;
  readonly heads = 2;
  readonly layers = 2;
  readonly hidden = 32;
  readonly vocabSize = 257; // UTF-8 bytes + BOS (256).
  private readonly embedding: Vector;
  private readonly output: Vector;
  private readonly weights: Layer[];

  constructor(seed = 42) {
    let state = seed >>> 0;
    const matrix = (rows: number, cols: number): Vector => {
      const scale = 1 / Math.sqrt(cols);
      return Float64Array.from({ length: rows * cols }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return (state / 2 ** 32 * 2 - 1) * scale;
      });
    };
    this.embedding = matrix(this.vocabSize, this.width);
    this.output = matrix(this.vocabSize, this.width);
    this.weights = Array.from({ length: this.layers }, () => ({
      q: matrix(this.width, this.width), k: matrix(this.width, this.width),
      v: matrix(this.width, this.width), out: matrix(this.width, this.width),
      up: matrix(this.hidden, this.width), down: matrix(this.width, this.hidden),
    }));
  }

  private embed(token: number, position: number): Vector {
    if (!Number.isInteger(token) || token < 0 || token >= this.vocabSize) {
      throw new Error(`Token must be an integer in [0, ${this.vocabSize}).`);
    }
    return Float64Array.from({ length: this.width }, (_, i) => {
      const angle = position / 10000 ** (2 * Math.floor(i / 2) / this.width);
      return this.embedding[token * this.width + i] + (i % 2 ? Math.cos(angle) : Math.sin(angle));
    });
  }

  private feedForward(x: Vector, layer: Layer): Vector {
    const hidden = linear(norm(x), layer.up, this.hidden).map(v => v / (1 + Math.exp(-v)));
    return residual(x, linear(hidden, layer.down, this.width));
  }

  /** Append one token. Attention reads every past K/V directly through the page table. */
  forwardToken(token: number, position: number, kv: KVStore): Vector {
    let x = this.embed(token, position);
    const headWidth = this.width / this.heads;
    for (let l = 0; l < this.layers; l++) {
      const layer = this.weights[l];
      const input = norm(x);
      const q = linear(input, layer.q, this.width);
      kv.set(l, position, linear(input, layer.k, this.width), linear(input, layer.v, this.width));
      const attended = new Float64Array(this.width);
      for (let h = 0; h < this.heads; h++) {
        const start = h * headWidth;
        const scores = new Float64Array(position + 1);
        for (let p = 0; p <= position; p++) {
          const { key } = kv.get(l, p);
          for (let d = start; d < start + headWidth; d++) scores[p] += q[d] * key[d];
          scores[p] /= Math.sqrt(headWidth);
        }
        const probabilities = softmax(scores);
        for (let p = 0; p <= position; p++) {
          const { value } = kv.get(l, p);
          for (let d = start; d < start + headWidth; d++) attended[d] += probabilities[p] * value[d];
        }
      }
      x = this.feedForward(residual(x, linear(attended, layer.out, this.width)), layer);
    }
    return linear(norm(x), this.output, this.vocabSize);
  }

  /** Reference: recompute the whole sequence, layer by layer, without any KV cache. */
  forwardDense(tokens: readonly number[]): Vector {
    if (!tokens.length) throw new Error('Dense input cannot be empty.');
    let states = tokens.map((token, p) => this.embed(token, p));
    const headWidth = this.width / this.heads;
    for (const layer of this.weights) {
      const inputs = states.map(norm);
      const queries = inputs.map(x => linear(x, layer.q, this.width));
      const keys = inputs.map(x => linear(x, layer.k, this.width));
      const values = inputs.map(x => linear(x, layer.v, this.width));
      states = states.map((x, position) => {
        const attended = new Float64Array(this.width);
        for (let h = 0; h < this.heads; h++) {
          const offset = h * headWidth;
          const scores = Float64Array.from({ length: position + 1 }, (_, p) => {
            let dot = 0;
            for (let d = 0; d < headWidth; d++) dot += queries[position][offset + d] * keys[p][offset + d];
            return dot / Math.sqrt(headWidth);
          });
          const probabilities = softmax(scores);
          for (let d = 0; d < headWidth; d++) {
            for (let p = 0; p <= position; p++) attended[offset + d] += probabilities[p] * values[p][offset + d];
          }
        }
        return this.feedForward(residual(x, linear(attended, layer.out, this.width)), layer);
      });
    }
    return linear(norm(states.at(-1)!), this.output, this.vocabSize);
  }
}

export function encode(text: string): number[] {
  return [256, ...new TextEncoder().encode(text)];
}
