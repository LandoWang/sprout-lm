import { createApiServer } from './http.ts';
import { cudaGraphOption } from './cuda-device.ts';
import { createRuntime } from './runtime.ts';
import { ObserverProcess } from './observer-process.ts';
import { record } from './observation.ts';

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? '127.0.0.1';
if (!Number.isSafeInteger(port) || port < 0 || port > 65535) throw new Error('PORT must be an integer from 0 to 65535.');
let observation: ObserverProcess | undefined;
if (process.env.OBSERVABILITY === '1') {
  try {
    observation = await ObserverProcess.start({ directory: process.env.EVENT_DIR ?? 'runs/events' });
    console.log(JSON.stringify({ phase: 'observer_ready', pid: observation.child.pid, directory: observation.directory }));
  } catch (error) { console.error('[observation] disabled:', error instanceof Error ? error.message : String(error)); }
}
const runtime = await createRuntime({
  observation: observation ? { connection: observation.connection, emit: observation.client.emit } : undefined,
  modelDirectory: process.env.MODEL_DIR,
  tokenizerPlugin: process.env.TOKENIZER_PLUGIN, tokenizerDirectory: process.env.TOKENIZER_DIR,
  templatePlugin: process.env.TEMPLATE_PLUGIN, templateDirectory: process.env.TEMPLATE_DIR,
  maxContextTokens: process.env.KV_TOKENS ? Number(process.env.KV_TOKENS) : undefined,
  backend: (process.env.BACKEND ?? 'cpu') as 'cpu' | 'cuda' | 'cuda-pd',
  pd: { prefillDevice: Number(process.env.PREFILL_DEVICE ?? 0), decodeDevice: Number(process.env.DECODE_DEVICE ?? 1),
    maxSequences: Number(process.env.MAX_SEQUENCES ?? 4), prefillChunkSize: Number(process.env.PREFILL_CHUNK_SIZE ?? 4) },
  cuda: { graph: cudaGraphOption(), device: Number(process.env.CUDA_DEVICE ?? 0), tileMiB: Number(process.env.CUDA_TILE_MIB ?? 128),
    weightLimitMiB: Number(process.env.CUDA_WEIGHT_LIMIT_MIB ?? 0), reserveMiB: Number(process.env.CUDA_RESERVE_MIB ?? 1024) },
}).catch(async error => { await observation?.close(); throw error; });
const server = createApiServer(runtime);
let stopping: Promise<void> | undefined;
const closeRuntime = () => stopping ??= (async () => {
  try { await runtime.close(); }
  finally { record(observation?.client.emit, 'serving.stopped'); await observation?.close(); }
})();
server.listen(port, host, () => {
  const address = server.address();
  console.log(`SproutLM API: http://${host}:${typeof address === 'object' ? address?.port : port}`);
  console.log(runtime.name);
  record(observation?.client.emit, 'serving.ready', { data: { backend: process.env.BACKEND ?? 'cpu' } });
  if (runtime.stats) console.log(JSON.stringify({ phase: 'ready', stats: runtime.stats() }));
});
server.on('error', error => {
  console.error(error.message);
  void closeRuntime().catch(error => console.error(error));
  process.exitCode = 1;
});
const shutdown = () => {
  void closeRuntime().catch(error => { console.error(error); process.exitCode = 1; });
  server.close();
  server.closeAllConnections();
};
process.once('SIGINT', shutdown);
process.once('SIGTERM', shutdown);
if (runtime.failure) void runtime.failure.then(error => {
  console.error('PD worker failure:', error.message);
  process.exitCode = 1;
  shutdown(); // A supervisor can restart all three processes as one unit.
});
server.once('close', () => { void closeRuntime().catch(error => console.error(error)); });
