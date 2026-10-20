import { CudaDeviceBackend } from './cuda-device.ts';
import { CudaQwenRunner } from './qwen-cuda.ts';
import type { DeviceKVStorage, KVStore } from './runner.ts';
import type { Command, Envelope, WorkerInfo, WorkerRole } from './pd-protocol.ts';
import { EventClient, record } from './observation.ts';

if (!process.send) throw new Error('PD workers must be started by their coordinator.');
let backend: CudaDeviceBackend | undefined, model: CudaQwenRunner | undefined;
let pool: DeviceKVStorage, imported: object | undefined, role: WorkerRole;
let busy = false, lastId = 0, closing = false;
let events: EventClient | undefined;
async function exit(code: number): Promise<void> {
  backend?.close(); record(events?.emit, 'worker.stopped', { data: { code } });
  await events?.close(); process.exit(code);
}
process.on('disconnect', () => { void exit(1); });

async function execute(command: Command): Promise<unknown> {
  if (command.op === 'init') {
    if (backend) throw new Error('Worker already initialized.');
    const config = command.value; role = config.role;
    if (config.events) events = new EventClient(config.events, role);
    record(events?.emit, 'worker.starting', { data: { device: config.device } });
    if (!['prefill', 'decode'].includes(role) || !Number.isSafeInteger(config.blockSize) || config.blockSize < 1)
      throw new Error('Invalid worker configuration.');
    backend = new CudaDeviceBackend({ ...config.cuda, device: config.device });
    model = await CudaQwenRunner.load(config.directory, backend, { maxContextTokens: config.tokens });
    const layout = { blockSize: config.blockSize, numBlocks: Math.ceil(config.tokens / config.blockSize),
      layers: model.spec.layers, width: model.spec.kvHeads * model.spec.headDim };
    pool = model.createKVStorage(layout);
    const info: WorkerInfo = { role, pid: process.pid, device: config.device, spec: model.spec, layout,
      ipc: role === 'prefill' ? backend.exportIPC(pool.handle) : undefined, stats: backend.stats() };
    record(events?.emit, 'worker.ready', { data: { device: config.device, kvTokens: layout.numBlocks * layout.blockSize } });
    return info;
  }
  if (command.op === 'close') { closing = true; backend?.close(); return null; }
  if (!backend || !model) throw new Error('Worker is not initialized.');
  if (command.op === 'stats') return backend.stats();
  if (command.op === 'attach') {
    const source = command.value;
    if (role !== 'decode' || imported || source.role !== 'prefill' || !source.ipc
      || JSON.stringify(source.spec) !== JSON.stringify(model.spec)
      || JSON.stringify(source.layout) !== JSON.stringify({ blockSize: pool.blockSize, numBlocks: pool.numBlocks, layers: pool.layers, width: pool.width }))
      throw new Error('Incompatible prefill pool or model.');
    const l = source.layout;
    imported = backend.importIPC(source.ipc, source.device, 2 * l.layers * l.numBlocks * l.blockSize * l.width);
    return null;
  }
  const startedAt = Date.now(), request = command.value.request;
  if (typeof request !== 'string' || !request) throw new Error('Missing request identity.');
  if (command.op === 'transfer') {
    if (role !== 'decode' || !imported) throw new Error('Decode has no imported prefill pool.');
    const v = command.value;
    const result = backend.copyKV(imported, pool.handle, v.sourcePages, v.targetPages, v.tokens, pool.blockSize, pool.layers, pool.width);
    return { request, ...result, startedAt, endedAt: Date.now() };
  }
  const v = command.value;
  if (!Number.isSafeInteger(v.from) || v.from < 0 || !v.tokens.length || v.from + v.tokens.length > model.spec.maxContextLength
    || !['none', 'sample', 'logits'].includes(v.output) || v.blocks.length !== Math.ceil((v.from + v.tokens.length) / pool.blockSize)
    || new Set(v.blocks).size !== v.blocks.length || v.blocks.some(p => !Number.isInteger(p) || p < 0 || p >= pool.numBlocks)
    || (role === 'decode' && (v.tokens.length !== 1 || !imported))) throw new Error('Invalid worker forward plan.');
  let token: number | undefined, logits: number[] | undefined;
  for (let i = 0; i < v.tokens.length; i++) {
    const position = v.from + i, output = i === v.tokens.length - 1 ? v.output : 'none';
    const kv: KVStore = { set() { throw new Error('Device KV only.'); }, get() { throw new Error('Device KV only.'); },
      deviceView: () => ({ storage: pool, length: position, blocks: v.blocks.slice(0, Math.floor(position / pool.blockSize) + 1) }) };
    if (output === 'logits') logits = Array.from(model.forwardToken(v.tokens[i], position, kv, true)!);
    else token = model.forwardSample(v.tokens[i], position, kv, output === 'sample');
  }
  return { request, token, logits, startedAt, endedAt: Date.now() };
}

process.on('message', async (message: Envelope) => {
  if (busy || closing || !Number.isSafeInteger(message.id) || message.id <= lastId) {
    // Protocol loss invalidates pool ownership; do not accept another command.
    process.exit(1);
  }
  busy = true; lastId = message.id;
  const plan = message.command.op === 'forward' || message.command.op === 'transfer' ? message.command.value : undefined;
  const fields = plan ? { requestId: plan.request, planId: plan.planId ?? `${role}-rpc-${message.id}`,
    data: { operation: message.command.op, rpcId: message.id } } : undefined;
  if (fields) record(events?.emit, 'plan.started', fields);
  try {
    const value = await execute(message.command);
    if (fields) record(events?.emit, 'plan.completed', { ...fields,
      data: { ...fields.data, ...('eventMs' in (value as object) ? { cudaEventMs: (value as { eventMs: number }).eventMs } : {}) } });
    process.send!({ id: message.id, value }, error => {
      if (closing || error) void exit(error ? 1 : 0);
    });
  } catch (error) {
    if (fields) record(events?.emit, 'plan.execution_failed', fields);
    process.send!({ id: message.id, error: error instanceof Error ? error.message : String(error) });
  } finally { busy = false; }
});
