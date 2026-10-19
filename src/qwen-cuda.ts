import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseQwenConfig, type QwenConfig } from './qwen.ts';
import { WeightFiles } from './weight-files.ts';
import type { ModelRunner, ModelSpec, KVStore, KVLayout, DeviceKVStorage } from './runner.ts';
import { CudaDeviceBackend } from './cuda-device.ts';

interface Layer {
  inputNorm: object; postNorm: object; q: object; k: object; v: object;
  qBias: object; kBias: object; vBias: object; out: object; gate: object; up: object; down: object;
}
interface Weights { embedding: object; output: object; norm: object; layers: Layer[] }

/** TS owns the Qwen layer loop; all numerical forward work and paged KV live on CUDA. */
export class CudaQwenRunner implements ModelRunner {
  readonly spec: ModelSpec;
  readonly config: QwenConfig;
  readonly maxContextTokens: number;
  private readonly backend: CudaDeviceBackend;
  private graphPool?: object;
  private graphs = new Map<'logits' | 'sample', object>();
  private readonly weights: Weights;
  private readonly work: Record<'x' | 'normed' | 'q' | 'k' | 'v' | 'attended' | 'projected' | 'gate' | 'up' | 'logits' | 'scores', object>;
  private constructor(config: QwenConfig, spec: ModelSpec, backend: CudaDeviceBackend, weights: Weights, maxTokens: number) {
    this.config = config; this.spec = spec; this.backend = backend; this.weights = weights; this.maxContextTokens = maxTokens;
    const h = spec.hiddenSize, kv = spec.kvHeads * spec.headDim, f = config.intermediate_size;
    this.work = { x: backend.vector(h), normed: backend.vector(h), q: backend.vector(h), k: backend.vector(kv), v: backend.vector(kv),
      attended: backend.vector(h), projected: backend.vector(h), gate: backend.vector(f), up: backend.vector(f),
      logits: backend.vector(spec.vocabSize), scores: backend.vector(spec.attentionHeads * maxTokens) };
  }

  static async load(directory: string, backend: CudaDeviceBackend, options: {
    maxContextTokens?: number; onLayerLoaded?: (layer: number, total: number) => void;
  } = {}): Promise<CudaQwenRunner> {
    const c = parseQwenConfig(JSON.parse(await readFile(join(directory, 'config.json'), 'utf8')));
    const maxTokens = options.maxContextTokens ?? Math.min(2048, c.max_position_embeddings);
    if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || maxTokens > c.max_position_embeddings) throw new Error('Invalid GPU context capacity.');
    let id = directory;
    try {
      const m = JSON.parse(await readFile(join(directory, 'sprout-model.json'), 'utf8'));
      if (typeof m.repo !== 'string' || typeof m.revision !== 'string') throw new Error('Invalid model provenance manifest.');
      id = `${m.repo}@${m.revision}`;
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    let eos: unknown = c.eos_token_id;
    try { eos = JSON.parse(await readFile(join(directory, 'generation_config.json'), 'utf8')).eos_token_id ?? eos; }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    const stops = Array.isArray(eos) ? eos : [eos];
    if (!stops.length || stops.some(n => !Number.isInteger(n) || n < 0 || n >= c.vocab_size)) throw new Error('Invalid EOS token IDs.');
    const spec: ModelSpec = { id, layers: c.num_hidden_layers, hiddenSize: c.hidden_size, attentionHeads: c.num_attention_heads,
      kvHeads: c.num_key_value_heads, headDim: c.hidden_size / c.num_attention_heads, vocabSize: c.vocab_size,
      maxContextLength: Math.min(maxTokens, c.max_position_embeddings), kvDtype: 'float32', eosTokenIds: [...new Set(stops)] };
    const files = await WeightFiles.open(directory);
    try {
      const matrix = async (name: string, shape: number[]) => backend.deviceMatrix(await files.descriptor(name, shape));
      const vector = async (name: string, size: number) => backend.vector(size, await files.tensor(name, [size]));
      const h = c.hidden_size, kv = spec.kvHeads * spec.headDim, f = c.intermediate_size;
      const embedding = await matrix('model.embed_tokens.weight', [c.vocab_size, h]);
      const layers: Layer[] = [];
      for (let i = 0; i < c.num_hidden_layers; i++) {
        const p = `model.layers.${i}.`;
        layers.push({ inputNorm: await vector(p + 'input_layernorm.weight', h), postNorm: await vector(p + 'post_attention_layernorm.weight', h),
          q: await matrix(p + 'self_attn.q_proj.weight', [h, h]), k: await matrix(p + 'self_attn.k_proj.weight', [kv, h]),
          v: await matrix(p + 'self_attn.v_proj.weight', [kv, h]), qBias: await vector(p + 'self_attn.q_proj.bias', h),
          kBias: await vector(p + 'self_attn.k_proj.bias', kv), vBias: await vector(p + 'self_attn.v_proj.bias', kv),
          out: await matrix(p + 'self_attn.o_proj.weight', [h, h]), gate: await matrix(p + 'mlp.gate_proj.weight', [f, h]),
          up: await matrix(p + 'mlp.up_proj.weight', [f, h]), down: await matrix(p + 'mlp.down_proj.weight', [h, f]) });
        options.onLayerLoaded?.(i + 1, c.num_hidden_layers);
      }
      const output = c.tie_word_embeddings ? embedding : await matrix('lm_head.weight', [c.vocab_size, h]);
      const norm = await vector('model.norm.weight', h);
      return new CudaQwenRunner(c, spec, backend, { embedding, output, norm, layers }, maxTokens);
    } finally { await files.close(); }
  }

  createKVStorage(layout: KVLayout): DeviceKVStorage {
    if (Object.values(layout).some(n => !Number.isSafeInteger(n) || n < 1) || layout.layers !== this.spec.layers
      || layout.width !== this.spec.kvHeads * this.spec.headDim
      || layout.numBlocks * layout.blockSize > Math.ceil(this.maxContextTokens / layout.blockSize) * layout.blockSize) {
      throw new Error('GPU KV pool exceeds the configured layout or context capacity.');
    }
    const elements = 2 * layout.layers * layout.numBlocks * layout.blockSize * layout.width;
    if (!Number.isSafeInteger(elements)) throw new Error('GPU KV allocation overflow.');
    return { ...layout, owner: this.backend, handle: this.backend.vector(elements) };
  }

  forwardToken(token: number, position: number, kv: KVStore, outputLogits = true): Float32Array | undefined {
    return this.forward(token, position, kv, outputLogits ? 'logits' : 'none') as Float32Array | undefined;
  }
  forwardSample(token: number, position: number, kv: KVStore, outputToken: boolean): number | undefined {
    return this.forward(token, position, kv, outputToken ? 'sample' : 'none') as number | undefined;
  }
  private forward(token: number, position: number, kv: KVStore, output: 'logits' | 'sample' | 'none'): Float32Array | number | undefined {
    const s = this.spec, c = this.config, b = this.backend, w = this.work;
    if (!Number.isInteger(token) || token < 0 || token >= s.vocabSize || !Number.isInteger(position) || position < 0 || position >= s.maxContextLength) throw new Error('Invalid GPU token or position.');
    const view = kv.deviceView?.();
    if (!view || view.storage.owner !== b || view.length !== position || view.blocks.length !== Math.floor(position / view.storage.blockSize) + 1
      || view.blocks.some(page => !Number.isInteger(page) || page < 0 || page >= view.storage.numBlocks)) throw new Error('Invalid GPU sequence or block table.');
    const pool = view.storage;
    if (b.graphEnabled && output !== 'none') {
      // Bound the cache to two graphs for one KV pool. Request page tables remain dynamic.
      if (this.graphPool !== pool.handle) {
        for (const graph of this.graphs.values()) b.releaseGraph(graph);
        this.graphs.clear(); this.graphPool = pool.handle;
      }
      const graph = this.graphs.get(output);
      if (graph) return b.replay(graph, view.blocks, token, position);
    }
    const submit = () => {
      b.embedding(this.weights.embedding, token, w.x);
      for (const [layer, weights] of this.weights.layers.entries()) {
        b.norm(w.x, weights.inputNorm, w.normed, c.rms_norm_eps);
        b.linear(weights.q, w.normed, w.q, weights.qBias);
        b.linear(weights.k, w.normed, w.k, weights.kBias);
        b.linear(weights.v, w.normed, w.v, weights.vBias);
        b.rope(w.q, s.headDim, position, c.rope_theta);
        b.rope(w.k, s.headDim, position, c.rope_theta);
        b.attention(w.q, w.k, w.v, pool.handle, w.attended, w.scores, layer, position, pool.blockSize, pool.numBlocks, s.kvHeads, s.headDim);
        b.linear(weights.out, w.attended, w.projected); b.add(w.x, w.projected, w.x);
        b.norm(w.x, weights.postNorm, w.normed, c.rms_norm_eps);
        b.linear(weights.gate, w.normed, w.gate); b.linear(weights.up, w.normed, w.up);
        b.swiglu(w.gate, w.up, w.gate);
        b.linear(weights.down, w.gate, w.projected); b.add(w.x, w.projected, w.x);
      }
      if (output === 'none') return;
      b.norm(w.x, this.weights.norm, w.normed, c.rms_norm_eps);
      b.linear(this.weights.output, w.normed, w.logits);
    };
    // First execution warms cuBLAS and returns the eager result; capture is paid once.
    b.begin(view.blocks);
    let result: Float32Array | number | undefined;
    try {
      submit();
      if (output !== 'none') result = output === 'sample' ? b.argmax(w.logits) : b.read(w.logits);
    } finally { b.finish(); }
    if (b.graphEnabled && output !== 'none') {
      const graph = b.capture(pool.numBlocks, pool.blockSize, w.logits, output === 'sample', () => {
        submit();
        if (output === 'sample') b.enqueueArgmax(w.logits);
      });
      this.graphs.set(output, graph);
    }
    return result;
  }
}
