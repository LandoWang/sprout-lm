import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { CudaOffload, type CudaWeightMode } from '../src/cuda-offload.ts';
import { QwenRunner } from '../src/qwen.ts';
import { CudaDeviceBackend, cudaGraphOption } from '../src/cuda-device.ts';
import { CudaQwenRunner } from '../src/qwen-cuda.ts';
import { Engine } from '../src/engine.ts';
import { defaultTextPlugins, validateTokenizer } from '../src/text-plugins.ts';

const directory = process.argv[2];
if (!directory) throw new Error('Usage: npm run smoke:cuda -- /path/to/checkpoint');
const prompt = process.env.PROMPT ?? 'Hello';
const maxNewTokens = Number(process.env.MAX_NEW_TOKENS ?? 4);
if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 128) throw new Error('MAX_NEW_TOKENS must be 1..128.');
const started = performance.now();
const execution = process.env.CUDA_EXECUTION ?? 'hybrid';
if (!['hybrid', 'full'].includes(execution)) throw new Error('CUDA_EXECUTION must be hybrid or full.');
const options = { device: Number(process.env.CUDA_DEVICE ?? 0), tileMiB: Number(process.env.CUDA_TILE_MIB ?? 128),
  weightMode: (process.env.CUDA_WEIGHT_MODE ?? 'offload') as CudaWeightMode,
  weightLimitMiB: Number(process.env.CUDA_WEIGHT_LIMIT_MIB ?? 0), reserveMiB: Number(process.env.CUDA_RESERVE_MIB ?? 1024) };
const backend = execution === 'full' ? new CudaDeviceBackend({ ...options, graph: cudaGraphOption() }) : new CudaOffload(options);
const kind = execution === 'full' ? 'cuda-device' : `cuda-${backend.weightMode}-gemv`;
try {
  console.log(JSON.stringify({ phase: 'backend', kind, pid: process.pid, stats: backend.stats() }));
  const onLayerLoaded = (layer: number, total: number) => { if (layer % 10 === 0 || layer === total) console.log(JSON.stringify({ phase: 'loaded', layer, total })); };
  const model = backend instanceof CudaDeviceBackend
    ? await CudaQwenRunner.load(directory, backend, { maxContextTokens: 256, onLayerLoaded })
    : await QwenRunner.load(directory, { matrixLoader: backend.matrix, onLayerLoaded });
  const plugins = defaultTextPlugins();
  const tokenizer = await plugins.tokenizer('hf-bytelevel', directory);
  validateTokenizer(tokenizer, model.spec);
  const inputText = process.env.CHAT === '1' ? (await plugins.template('jinja', directory)).render([{ role: 'user', content: prompt }]) : prompt;
  const input = tokenizer.encode(inputText);
  if (!input.length || input.length + maxNewTokens > 256) throw new Error('This smoke CLI limits prompt + output to 256 tokens.');
  const loadMs = performance.now() - started;
  const engine = new Engine({ blockSize: 16, numBlocks: Math.ceil((input.length + maxNewTokens) / 16),
    maxSequences: 1, maxBatchTokens: 1, prefillChunkSize: 1, recordTrace: false }, model);
  const decode = tokenizer.decoder(); let text = '', firstTokenMs: number | undefined;
  const rounds: { round: number; ms: number; phase: string; tokens: number[] }[] = [];
  const loadStats = backend.stats();
  console.log(JSON.stringify({ phase: 'ready', model: model.spec, input, loadMs, stats: loadStats }));
  engine.add({ id: 'cuda-smoke', prompt: input, maxNewTokens });
  const generation = performance.now();
  while (engine.pending) {
    const before = performance.now();
    const result = engine.step();
    const tokens = result.events.filter(event => event.type === 'token').map(event => Number(event.detail));
    for (const token of tokens) { firstTokenMs ??= performance.now() - generation; text += decode.push(token); }
    const round = { round: rounds.length + 1, ms: performance.now() - before, phase: result.work[0]?.phase ?? 'none', tokens };
    rounds.push(round); console.log(JSON.stringify({ ...round, text }));
    await new Promise<void>(resolve => setImmediate(resolve));
  }
  text += decode.finish();
  const generationMs = performance.now() - generation;
  const result = engine.result('cuda-smoke');
  engine.cache.clearPrefixes();
  const stats = backend.stats();
  const generationTransfers = Object.fromEntries(['hostToDeviceBytes', 'deviceToHostBytes', 'uploadWallMs',
    'weightUploadBytes', 'matvecWeightUploadBytes'].map(key => [key, Number(stats[key]) - Number(loadStats[key])]));
  const decodeRounds = rounds.filter(round => round.phase === 'decode');
  const decodeMs = decodeRounds.reduce((sum, round) => sum + round.ms, 0);
  const report = { model: model.spec, backend: kind, chat: process.env.CHAT === '1', prompt,
    input, output: result.output, text, finishReason: result.finishReason, loadMs, firstTokenMs, generationMs,
    processMemory: process.memoryUsage(), processResourceUsage: process.resourceUsage(), loadStats, stats, generationTransfers,
    decodeSummary: { steps: decodeRounds.length, totalMs: decodeMs,
      meanMs: decodeRounds.length ? decodeMs / decodeRounds.length : null,
      stepsPerSecond: decodeMs ? decodeRounds.length * 1000 / decodeMs : null },
    kv: engine.cache.snapshot(), rounds,
    limitations: execution === 'full' ? ['All numerical forward operators, paged KV and greedy argmax on GPU; TypeScript orchestration.',
      backend.matvecMode === 'direct' ? 'BF16/F16 weights read directly with FP32 accumulation; F32 uses SGEMV. FP32 activations/KV; no Tensor Core GEMM.'
        : 'BF16/F16 weights convert in GPU tiles for FP32 SGEMV. FP32 activations/KV; no Tensor Core GEMM.',
      'Synchronous per-token completion; sequential prefill and request execution; no tensor batching or multi-GPU.']
      : ['Synchronous CUDA GEMV; TypeScript attention, norms and KV.',
      backend.weightMode === 'resident'
        ? 'Matrix weights stay on GPU in checkpoint dtype. Norms/biases and embedding row reads use CPU.'
        : 'File-backed host weights upload in bounded GPU tiles.',
      backend.matvecMode === 'direct' ? 'BF16/F16 matvec reads weights directly with FP32 accumulation; F32 uses SGEMV.'
        : 'BF16/F16 tiles convert to FP32 for SGEMV.',
      'No multi-GPU execution, tensor batching or asynchronous serving.'] };
  if (process.env.REPORT_PATH) {
    const path = resolve(process.env.REPORT_PATH); await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(report, null, 2) + '\n');
  }
  console.log(JSON.stringify({ phase: 'finished', report }));
} finally { backend.close(); }
