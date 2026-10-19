import { createApiServer } from './http.ts';
import { cudaGraphOption } from './cuda-device.ts';
import { createRuntime } from './runtime.ts';

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? '127.0.0.1';
if (!Number.isSafeInteger(port) || port < 0 || port > 65535) throw new Error('PORT must be an integer from 0 to 65535.');
const runtime = await createRuntime({
  modelDirectory: process.env.MODEL_DIR,
  tokenizerPlugin: process.env.TOKENIZER_PLUGIN, tokenizerDirectory: process.env.TOKENIZER_DIR,
  templatePlugin: process.env.TEMPLATE_PLUGIN, templateDirectory: process.env.TEMPLATE_DIR,
  maxContextTokens: process.env.KV_TOKENS ? Number(process.env.KV_TOKENS) : undefined,
  backend: (process.env.BACKEND ?? 'cpu') as 'cpu' | 'cuda' | 'cuda-pd',
  pd: { prefillDevice: Number(process.env.PREFILL_DEVICE ?? 0), decodeDevice: Number(process.env.DECODE_DEVICE ?? 1),
    maxSequences: Number(process.env.MAX_SEQUENCES ?? 4), prefillChunkSize: Number(process.env.PREFILL_CHUNK_SIZE ?? 4) },
  cuda: { graph: cudaGraphOption(), device: Number(process.env.CUDA_DEVICE ?? 0), tileMiB: Number(process.env.CUDA_TILE_MIB ?? 128),
    weightLimitMiB: Number(process.env.CUDA_WEIGHT_LIMIT_MIB ?? 0), reserveMiB: Number(process.env.CUDA_RESERVE_MIB ?? 1024) },
});
const server = createApiServer(runtime);
server.listen(port, host, () => {
  const address = server.address();
  console.log(`SproutLM API: http://${host}:${typeof address === 'object' ? address?.port : port}`);
  console.log(runtime.name);
  if (runtime.stats) console.log(JSON.stringify({ phase: 'ready', stats: runtime.stats() }));
});
server.on('error', error => {
  console.error(error.message);
  void Promise.resolve(runtime.close()).catch(error => console.error(error));
  process.exitCode = 1;
});
const shutdown = () => {
  void Promise.resolve(runtime.close()).catch(error => { console.error(error); process.exitCode = 1; });
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
server.once('close', () => { void Promise.resolve(runtime.close()).catch(error => console.error(error)); });
