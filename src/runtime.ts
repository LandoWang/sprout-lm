import { Engine } from './engine.ts';
import { GenerationService } from './service.ts';
import { QwenRunner } from './qwen.ts';
import { CudaQwenRunner } from './qwen-cuda.ts';
import { CudaDeviceBackend, type CudaDeviceOptions } from './cuda-device.ts';
import { defaultTextPlugins, validateTokenizer, type TextPluginRegistry } from './text-plugins.ts';
import { byteTokenizer } from './tokenizer.ts';
import { PDWorkers } from './pd-workers.ts';
import { PDScheduler } from './pd-scheduler.ts';
import type { EventConnection, EventSink } from './observation.ts';

export interface RuntimeOptions {
  modelDirectory?: string;
  tokenizerPlugin?: string;
  tokenizerDirectory?: string;
  templatePlugin?: string;
  templateDirectory?: string;
  maxContextTokens?: number;
  plugins?: TextPluginRegistry;
  backend?: 'cpu' | 'cuda' | 'cuda-pd';
  cuda?: CudaDeviceOptions;
  observation?: { connection: EventConnection; emit: EventSink };
  pd?: { prefillDevice?: number; decodeDevice?: number; maxSequences?: number; prefillChunkSize?: number; traceLimit?: number };
}

/** Composition root: weights, text plugins and engine are assembled once before accepting requests. */
export async function createRuntime(options: RuntimeOptions = {}) {
  const observer = options.observation?.emit;
  if (options.backend && !['cpu', 'cuda', 'cuda-pd'].includes(options.backend)) throw new Error('Unknown execution backend.');
  if (!options.modelDirectory) {
    if (options.backend === 'cuda' || options.backend === 'cuda-pd') throw new Error('CUDA requires a model directory.');
    const service = new GenerationService(undefined, {}, observer);
    return { service, tokenizer: byteTokenizer, name: 'TinyTransformer (untrained)', close: () => service.close() };
  }
  const registry = options.plugins ?? defaultTextPlugins();
  const directory = options.modelDirectory;
  const tokenizer = await registry.tokenizer(options.tokenizerPlugin ?? 'hf-bytelevel', options.tokenizerDirectory ?? directory);
  const template = await registry.template(options.templatePlugin ?? 'jinja', options.templateDirectory ?? directory);
  const tokens = options.maxContextTokens ?? (options.backend?.startsWith('cuda') ? 2048 : 512);
  if (options.backend === 'cuda-pd') {
    const workers = await PDWorkers.start({ directory, tokens, ...options.pd, cuda: options.cuda,
      events: options.observation?.connection, observer });
    try {
      validateTokenizer(tokenizer, workers.prefill.spec);
      const scheduler = new PDScheduler(workers, { ...options.pd, observer });
      const service = new GenerationService(scheduler, {}, observer);
      return { service, tokenizer, template, name: `${workers.prefill.spec.id} (three-process CUDA PD)`,
        scheduler, failure: scheduler.failureSignal, stats: () => scheduler.snapshot(),
        close: async () => { service.close(); await service.drained(); } };
    } catch (error) { await workers.close(); throw error; }
  }
  const backend = options.backend === 'cuda' ? new CudaDeviceBackend(options.cuda) : undefined;
  try {
    const runner = backend ? await CudaQwenRunner.load(directory, backend, { maxContextTokens: tokens,
      onLayerLoaded: (layer, total) => { if (layer % 10 === 0 || layer === total) console.log(JSON.stringify({ phase: 'loaded', layer, total })); } })
      : await QwenRunner.load(directory);
    validateTokenizer(tokenizer, runner.spec);
    if (!Number.isSafeInteger(tokens) || tokens < 1 || tokens > runner.spec.maxContextLength) throw new Error('Invalid maxContextTokens.');
    const engine = new Engine({
      blockSize: 16, numBlocks: Math.ceil(tokens / 16), maxBatchTokens: 4, prefillChunkSize: 4, recordTrace: false,
    }, runner);
    const service = new GenerationService(engine, {}, observer);
    return { service, tokenizer, template, name: `${runner.spec.id} (${backend ? 'CUDA forward + device KV' : 'CPU FP32'})`,
      stats: backend ? () => backend.stats() : undefined,
      close: () => { service.close(); backend?.close(); } };
  } catch (error) { backend?.close(); throw error; }
}
