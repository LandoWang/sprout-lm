import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { WeightFiles } from './weight-files.ts';
import type { TensorDescriptor } from './safetensors.ts';
import type { KVStore, ModelRunner, ModelSpec } from './runner.ts';

export interface QwenConfig {
  model_type: 'qwen2'; hidden_size: number; intermediate_size: number;
  num_hidden_layers: number; num_attention_heads: number; num_key_value_heads: number;
  vocab_size: number; max_position_embeddings: number; rms_norm_eps: number;
  rope_theta: number; tie_word_embeddings: boolean; eos_token_id: number | number[];
}

export function parseQwenConfig(raw: unknown): QwenConfig {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid Qwen config.');
  const c = raw as Record<string, unknown>;
  if (c.model_type !== 'qwen2' || c.hidden_act !== 'silu') throw new Error('Only Qwen2 with SiLU is supported.');
  if (c.use_sliding_window || c.rope_scaling || c.quantization_config || c.use_mrope) {
    throw new Error('Sliding window, scaled/multimodal RoPE and quantized checkpoints are not supported.');
  }
  for (const key of ['hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'max_position_embeddings']) {
    if (!Number.isSafeInteger(c[key]) || (c[key] as number) <= 0) throw new Error(`Invalid model config: ${key}.`);
  }
  for (const key of ['rms_norm_eps', 'rope_theta']) {
    if (typeof c[key] !== 'number' || !Number.isFinite(c[key]) || c[key] <= 0) throw new Error(`Invalid model config: ${key}.`);
  }
  if (typeof c.tie_word_embeddings !== 'boolean') throw new Error('tie_word_embeddings must be boolean.');
  const config = c as unknown as QwenConfig;
  const headDim = config.hidden_size / config.num_attention_heads;
  if (!Number.isInteger(headDim) || headDim % 2 || config.num_attention_heads % config.num_key_value_heads
    || (c.head_dim !== undefined && c.head_dim !== headDim)) throw new Error('Unsupported Qwen head dimensions.');
  eosIds(config.eos_token_id, config.vocab_size);
  return config;
}

function eosIds(value: unknown, vocabSize: number): number[] {
  const ids = Array.isArray(value) ? value : [value];
  if (!ids.length || ids.some(id => !Number.isInteger(id) || id < 0 || id >= vocabSize)) throw new Error('Invalid EOS token IDs.');
  return [...new Set(ids)];
}

/** A matrix may live outside JS memory. Model structure remains in this TypeScript runner. */
export interface MatrixOperator {
  row(index: number): Float32Array;
  matvec(input: Float32Array, bias?: Float32Array): Float32Array;
}
export type MatrixWeight = Float32Array | MatrixOperator;

interface LayerWeights {
  inputNorm: Float32Array; postNorm: Float32Array;
  q: MatrixWeight; k: MatrixWeight; v: MatrixWeight;
  qBias: Float32Array; kBias: Float32Array; vBias: Float32Array;
  out: MatrixWeight; gate: MatrixWeight; up: MatrixWeight; down: MatrixWeight;
}
export interface QwenWeights {
  embedding: MatrixWeight; norm: Float32Array; output: MatrixWeight; layers: LayerWeights[];
}

function linear(x: Float32Array, weights: MatrixWeight, rows: number, bias?: Float32Array): Float32Array {
  if (!(weights instanceof Float32Array)) return weights.matvec(x, bias);
  const y = new Float32Array(rows);
  const cols = x.length;
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    const offset = row * cols;
    for (let col = 0; col < cols; col++) sum += weights[offset + col] * x[col];
    y[row] = sum + (bias?.[row] ?? 0);
  }
  return y;
}

function norm(x: Float32Array, weight: Float32Array, epsilon: number): Float32Array {
  let variance = 0;
  for (const value of x) variance += value * value;
  const scale = 1 / Math.sqrt(variance / x.length + epsilon);
  return x.map((value, i) => Math.fround(value * scale) * weight[i]);
}

function add(x: Float32Array, delta: Float32Array): Float32Array { return x.map((value, i) => value + delta[i]); }

function rope(x: Float32Array, headDim: number, position: number, theta: number): void {
  const half = headDim / 2;
  for (let start = 0; start < x.length; start += headDim) {
    for (let d = 0; d < half; d++) {
      const angle = position / theta ** (2 * d / headDim);
      const cos = Math.cos(angle), sin = Math.sin(angle);
      const a = x[start + d], b = x[start + half + d];
      x[start + d] = a * cos - b * sin;
      x[start + half + d] = b * cos + a * sin;
    }
  }
}

/** Qwen2 structure in TypeScript, with optional matrix operators; attention and KV remain on CPU. */
export class QwenRunner implements ModelRunner {
  readonly spec: ModelSpec;
  readonly config: QwenConfig;
  /** Optional numerical debugging hook. Receives a copy of each layer's residual output. */
  inspectLayer?: (position: number, layer: number, hidden: Float32Array) => void;
  private readonly weights: QwenWeights;

  constructor(config: QwenConfig, weights: QwenWeights, id = 'qwen2', stopIds = eosIds(config.eos_token_id, config.vocab_size)) {
    this.config = config;
    this.weights = weights;
    this.spec = {
      id, layers: config.num_hidden_layers, hiddenSize: config.hidden_size,
      attentionHeads: config.num_attention_heads, kvHeads: config.num_key_value_heads,
      headDim: config.hidden_size / config.num_attention_heads, vocabSize: config.vocab_size,
      maxContextLength: config.max_position_embeddings, kvDtype: 'float32', eosTokenIds: stopIds,
    };
  }

  static async load(directory: string, options: {
    matrixLoader?: (descriptor: TensorDescriptor) => MatrixOperator;
    onLayerLoaded?: (layer: number, total: number) => void;
  } = {}): Promise<QwenRunner> {
    const config = parseQwenConfig(JSON.parse(await readFile(join(directory, 'config.json'), 'utf8')));
    let id = directory;
    try {
      const manifest = JSON.parse(await readFile(join(directory, 'sprout-model.json'), 'utf8'));
      if (typeof manifest.repo !== 'string' || typeof manifest.revision !== 'string') throw new Error('Invalid model provenance manifest.');
      id = `${manifest.repo}@${manifest.revision}`;
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    let stops = eosIds(config.eos_token_id, config.vocab_size);
    try {
      const generation = JSON.parse(await readFile(join(directory, 'generation_config.json'), 'utf8'));
      if (generation.eos_token_id !== undefined) stops = eosIds(generation.eos_token_id, config.vocab_size);
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    const file = await WeightFiles.open(directory);
    try {
      const matrix = async (name: string, shape: number[]): Promise<MatrixWeight> => options.matrixLoader
        ? options.matrixLoader(await file.descriptor(name, shape)) : file.tensor(name, shape);
      const w = config.hidden_size, f = config.intermediate_size;
      const kv = config.num_key_value_heads * w / config.num_attention_heads;
      const embedding = await matrix('model.embed_tokens.weight', [config.vocab_size, w]);
      const layers: LayerWeights[] = [];
      // Read sequentially to avoid keeping multiple full BF16 tensors in temporary buffers.
      for (let l = 0; l < config.num_hidden_layers; l++) {
        const prefix = `model.layers.${l}.`;
        const tensor = (name: string, shape: number[]) => file.tensor(prefix + name, shape);
        layers.push({
          inputNorm: await tensor('input_layernorm.weight', [w]), postNorm: await tensor('post_attention_layernorm.weight', [w]),
          q: await matrix(prefix + 'self_attn.q_proj.weight', [w, w]), qBias: await tensor('self_attn.q_proj.bias', [w]),
          k: await matrix(prefix + 'self_attn.k_proj.weight', [kv, w]), kBias: await tensor('self_attn.k_proj.bias', [kv]),
          v: await matrix(prefix + 'self_attn.v_proj.weight', [kv, w]), vBias: await tensor('self_attn.v_proj.bias', [kv]),
          out: await matrix(prefix + 'self_attn.o_proj.weight', [w, w]),
          gate: await matrix(prefix + 'mlp.gate_proj.weight', [f, w]), up: await matrix(prefix + 'mlp.up_proj.weight', [f, w]),
          down: await matrix(prefix + 'mlp.down_proj.weight', [w, f]),
        });
        options.onLayerLoaded?.(l + 1, config.num_hidden_layers);
      }
      const output = config.tie_word_embeddings ? embedding : await matrix('lm_head.weight', [config.vocab_size, w]);
      const normWeight = await file.tensor('model.norm.weight', [w]);
      return new QwenRunner(config, { embedding, output, norm: normWeight, layers }, id, stops);
    } finally { await file.close(); }
  }

  forwardToken(token: number, position: number, kv: KVStore, outputLogits = true): Float32Array | undefined {
    const s = this.spec, c = this.config;
    if (!Number.isInteger(token) || token < 0 || token >= s.vocabSize) throw new Error('Invalid token ID.');
    if (!Number.isInteger(position) || position < 0 || position >= s.maxContextLength) throw new Error('Invalid position.');
    let x: Float32Array = this.weights.embedding instanceof Float32Array
      ? this.weights.embedding.slice(token * s.hiddenSize, (token + 1) * s.hiddenSize) : this.weights.embedding.row(token);
    for (let l = 0; l < s.layers; l++) {
      const weights = this.weights.layers[l];
      const input = norm(x, weights.inputNorm, c.rms_norm_eps);
      const q = linear(input, weights.q, s.hiddenSize, weights.qBias);
      const k = linear(input, weights.k, s.kvHeads * s.headDim, weights.kBias);
      const v = linear(input, weights.v, s.kvHeads * s.headDim, weights.vBias);
      rope(q, s.headDim, position, c.rope_theta);
      rope(k, s.headDim, position, c.rope_theta);
      kv.set(l, position, k, v);
      const past = Array.from({ length: position + 1 }, (_, p) => kv.get(l, p));
      const attended = new Float32Array(s.hiddenSize);
      for (let h = 0; h < s.attentionHeads; h++) {
        const qStart = h * s.headDim;
        const kvStart = Math.floor(h / (s.attentionHeads / s.kvHeads)) * s.headDim;
        const scores = new Float32Array(position + 1);
        let max = -Infinity;
        for (let p = 0; p <= position; p++) {
          let dot = 0;
          for (let d = 0; d < s.headDim; d++) dot += q[qStart + d] * past[p].key[kvStart + d];
          scores[p] = dot / Math.sqrt(s.headDim);
          max = Math.max(max, scores[p]);
        }
        let total = 0;
        for (let p = 0; p <= position; p++) { scores[p] = Math.exp(scores[p] - max); total += scores[p]; }
        for (let p = 0; p <= position; p++) scores[p] /= total;
        for (let d = 0; d < s.headDim; d++) {
          let value = 0;
          for (let p = 0; p <= position; p++) value += scores[p] * past[p].value[kvStart + d];
          attended[qStart + d] = value;
        }
      }
      x = add(x, linear(attended, weights.out, s.hiddenSize));
      const ffnInput = norm(x, weights.postNorm, c.rms_norm_eps);
      const gate = linear(ffnInput, weights.gate, c.intermediate_size);
      const up = linear(ffnInput, weights.up, c.intermediate_size);
      for (let i = 0; i < gate.length; i++) gate[i] = Math.fround(gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
      x = add(x, linear(gate, weights.down, s.hiddenSize));
      this.inspectLayer?.(position, l, x.slice());
    }
    return outputLogits ? linear(norm(x, this.weights.norm, c.rms_norm_eps), this.weights.output, s.vocabSize) : undefined;
  }
}
