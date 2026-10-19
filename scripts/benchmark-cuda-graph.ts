import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { CudaDeviceBackend, cudaGraphOption } from '../src/cuda-device.ts';
import { CudaQwenRunner } from '../src/qwen-cuda.ts';
import { Engine } from '../src/engine.ts';
import { defaultTextPlugins } from '../src/text-plugins.ts';

const directory = process.argv[2];
if (!directory) throw new Error('Usage: node scripts/benchmark-cuda-graph.ts MODEL_DIR');
const repetitions = Number(process.env.REPETITIONS ?? 3);
const maxNewTokens = Number(process.env.MAX_NEW_TOKENS ?? 32);
if (!Number.isSafeInteger(repetitions) || repetitions < 1 || repetitions > 20) throw new Error('REPETITIONS must be 1..20.');
if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 2 || maxNewTokens > 128) throw new Error('MAX_NEW_TOKENS must be 2..128.');
const graph = cudaGraphOption();
const backend = new CudaDeviceBackend({ device: Number(process.env.CUDA_DEVICE ?? 1), graph,
  tileMiB: Number(process.env.CUDA_TILE_MIB ?? 128), reserveMiB: 4096 });
const summary = (values: number[]) => {
  if (!values.length) throw new Error('Benchmark needs at least one decode step.');
  const sorted = [...values].sort((a,b) => a-b);
  return { count: sorted.length, meanMs: values.reduce((a,b) => a+b, 0)/values.length,
    medianMs: sorted[Math.floor(sorted.length/2)], p95Ms: sorted[Math.max(0,Math.ceil(sorted.length*.95)-1)], minMs: sorted[0], maxMs: sorted.at(-1) };
};
try {
  const started = performance.now();
  const model = await CudaQwenRunner.load(directory, backend, { maxContextTokens: 256 });
  const plugins = defaultTextPlugins();
  const tokenizer = await plugins.tokenizer('hf-bytelevel', directory), template = await plugins.template('jinja', directory);
  const prompt = process.env.PROMPT ?? '从1开始逐行数到100，每行只写一个数字。';
  const input = tokenizer.encode(template.render([{ role: 'user', content: prompt }]));
  const engine = new Engine({ blockSize: 16, numBlocks: 16, maxSequences: 1, maxBatchTokens: 4,
    prefillChunkSize: 4, prefixCaching: false, recordTrace: false }, model);
  const loadMs = performance.now()-started;
  const runs: Array<{ run: number; cold: boolean; firstTokenMs: number | undefined; totalMs: number;
    output: number[]; finishReason: string | undefined; text: string; prefillMs: number[]; decodeMs: number[];
    decode: ReturnType<typeof summary>; counters: Record<string, number> }> = [];
  console.log(JSON.stringify({ phase:'ready', graph, loadMs, promptTokens:input.length, pid:process.pid, stats:backend.stats() }));
  for (let run = 0; run <= repetitions; run++) {
    const before = backend.stats();
    const id = 'bench-' + run, start = performance.now();
    engine.add({ id, prompt:input, maxNewTokens });
    let firstTokenMs: number | undefined;
    const decodeMs: number[] = [], prefillMs: number[] = [];
    while (engine.pending) {
      const tickStart = performance.now();
      const report = engine.step();
      const tickMs = performance.now()-tickStart;
      if (report.work[0]?.phase === 'decode') decodeMs.push(tickMs); else prefillMs.push(tickMs);
      if (report.events.some(event => event.type === 'token')) firstTokenMs ??= performance.now()-start;
      await new Promise<void>(resolve => setImmediate(resolve));
    }
    const totalMs = performance.now()-start, result = engine.result(id);
    assert.equal(engine.cache.snapshot().freePages, 16);
    const decoder = tokenizer.decoder();
    const text = result.output.map(token => decoder.push(token)).join('') + decoder.finish();
    const after = backend.stats();
    const counters = Object.fromEntries(['graphCaptures','graphLaunches','graphCaptureWallMs','deviceForwards','deviceForwardEventMs',
      'hostToDeviceBytes','deviceToHostBytes','graphInputUploadBytes','blockTableUploadBytes','matvecWeightUploadBytes']
      .map(key => [key,Number(after[key])-Number(before[key])]));
    const row = { run, cold:run===0, firstTokenMs, totalMs, output:result.output, finishReason:result.finishReason, text, prefillMs, decodeMs, decode:summary(decodeMs), counters };
    if (run) assert.deepEqual(row.output,runs[0].output,'Repeated deterministic output changed');
    runs.push(row); engine.forget(id);
    console.log(JSON.stringify({ phase:'run', ...row }));
  }
  const measured = runs.slice(1), decode = measured.flatMap(run => run.decodeMs);
  const report = { model:model.spec.id, device:backend.stats().deviceName, graph, matvec:backend.matvecMode, prompt, input, maxNewTokens, repetitions,
    scheduler:{ maxBatchTokens:4, prefillChunkSize:4, prefixCaching:false, maxSequences:1 }, loadMs, runs,
    steady:{ firstToken:summary(measured.map(run => run.firstTokenMs!)), decode:summary(decode), decodeTokensPerSecond:1000/summary(decode).meanMs },
    stats:backend.stats(), limitations:['Sequential single-request benchmark, no tensor batching.',
      'First output warms eager execution and captures a graph; cold run includes this cost.',
      'Non-output prefill stays eager; only logits/sample forwards replay.', 'CUDA event intervals include submission gaps and readback; not GPU utilization.'] };
  if (process.env.REPORT_PATH) {
    const path = resolve(process.env.REPORT_PATH); await mkdir(dirname(path),{recursive:true});
    await writeFile(path,JSON.stringify(report,null,2)+'\n');
  }
  console.log(JSON.stringify({phase:'finished',steady:report.steady,stats:report.stats}));
} finally { backend.close(); }
