import { fork, type ChildProcess } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import type { CudaDeviceOptions } from './cuda-device.ts';
import { record, type EventConnection, type EventSink } from './observation.ts';
import type { Command, ForwardCommand, PDExecutors, Reply, TransferCommand, WorkerInfo, WorkerResult, WorkerRole } from './pd-protocol.ts';

class WorkerProcess {
  readonly child: ChildProcess;
  private nextId = 0;
  private pending?: { id: number; resolve: (value: unknown) => void; reject: (error: Error) => void; timer: NodeJS.Timeout };
  private stopping?: Promise<void>;
  private closing = false;
  private dead = false;
  private error?: Error;
  private exited: Promise<void>;
  constructor(onFailure: (error: Error) => void) {
    this.child = fork(fileURLToPath(new URL('./pd-worker.ts', import.meta.url)), [], {
      execArgv: [], serialization: 'advanced', stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
    });
    const fail = (error: Error) => {
      this.error ??= error;
      if (this.pending) { clearTimeout(this.pending.timer); this.pending.reject(error); this.pending = undefined; }
      if (!this.closing) onFailure(error);
    };
    this.exited = new Promise(resolve => {
      this.child.once('exit', (code, signal) => {
        this.dead = true;
        fail(new Error(`PD worker ${this.child.pid} exited (${code ?? signal}).`)); resolve();
      });
      this.child.once('error', error => { fail(error); if (!this.child.pid) { this.dead = true; resolve(); } });
    });
    this.child.on('message', (message: Reply) => {
      const pending = this.pending;
      if (!pending || !message || message.id !== pending.id) {
        fail(new Error('Unexpected PD completion identity.')); this.child.kill('SIGKILL'); return;
      }
      clearTimeout(pending.timer); this.pending = undefined;
      if (message.error) pending.reject(new Error(message.error)); else pending.resolve(message.value);
    });
  }
  call<T>(command: Command, timeoutMs = 120000): Promise<T> {
    if (this.error || this.dead) return Promise.reject(this.error ?? new Error('PD worker is closed.'));
    if (this.pending || (this.closing && command.op !== 'close')) return Promise.reject(new Error('PD worker already has in-flight work.'));
    return new Promise<T>((resolve, reject) => {
      const id = ++this.nextId;
      const timer = setTimeout(() => {
        this.child.kill('SIGKILL');
        // Rejection comes from exit: a timeout alone is not proof that GPU work stopped.
      }, timeoutMs);
      this.pending = { id, resolve: value => resolve(value as T), reject, timer };
      this.child.send({ id, command }, error => { if (error) this.child.kill('SIGKILL'); });
    });
  }
  stop(): Promise<void> {
    return this.stopping ??= (async () => {
      this.closing = true;
      if (this.dead) return;
      if (this.pending || this.error) this.child.kill('SIGKILL');
      else { try { await this.call({ op: 'close' }, 5000); } catch { this.child.kill('SIGKILL'); } }
      // Keep the exporter alive until the importer has actually exited.
      const timer = setTimeout(() => this.child.kill('SIGKILL'), 5000);
      await this.exited; clearTimeout(timer);
    })();
  }
}

export interface PDWorkerOptions {
  directory: string; tokens: number; blockSize?: number; prefillDevice?: number; decodeDevice?: number; cuda?: CudaDeviceOptions;
  events?: EventConnection; observer?: EventSink;
}
/** One native IPC channel per child (Unix socket pair on Linux); no KV bytes cross it. */
export class PDWorkers implements PDExecutors {
  readonly prefill: WorkerInfo;
  readonly decode: WorkerInfo;
  private workers: Record<WorkerRole, WorkerProcess>;
  private closing?: Promise<void>;
  private failures: Array<(error: Error) => void> = [];
  private failure?: Error;
  private constructor(workers: Record<WorkerRole, WorkerProcess>, prefill: WorkerInfo, decode: WorkerInfo) {
    this.workers = workers; this.prefill = prefill; this.decode = decode;
  }
  static async start(options: PDWorkerOptions): Promise<PDWorkers> {
    const p = options.prefillDevice ?? 0, d = options.decodeDevice ?? 1;
    if (![p,d].every(n => Number.isSafeInteger(n) && n >= 0) || p === d) throw new Error('PD needs two distinct CUDA devices.');
    let cluster: PDWorkers | undefined, earlyError: Error | undefined;
    const failure = (error: Error) => {
      record(options.observer, 'worker.failed', { data: { message: error.message } });
      if (cluster) cluster.fail(error); else earlyError ??= error;
    };
    const prefill = new WorkerProcess(failure), decode = new WorkerProcess(failure), workers = { prefill, decode };
    try {
      // Load separately to keep checkpoint I/O and startup failures easy to diagnose.
      const init = { directory: options.directory, tokens: options.tokens, blockSize: options.blockSize ?? 16, cuda: options.cuda ?? {}, events: options.events };
      const pi = await prefill.call<WorkerInfo>({ op: 'init', value: { ...init, role: 'prefill', device: p } });
      const di = await decode.call<WorkerInfo>({ op: 'init', value: { ...init, role: 'decode', device: d } });
      if (earlyError) throw earlyError;
      if (JSON.stringify(pi.spec) !== JSON.stringify(di.spec) || JSON.stringify(pi.layout) !== JSON.stringify(di.layout))
        throw new Error('PD model or KV layout mismatch.');
      await decode.call({ op: 'attach', value: pi });
      cluster = new PDWorkers(workers, pi, di);
      if (earlyError) throw earlyError;
      return cluster;
    } catch (error) { await decode.stop(); await prefill.stop(); throw error; }
  }
  private fail(error: Error): void {
    if (this.closing || this.failure) return;
    this.failure = error;
    for (const callback of this.failures) callback(error);
    void this.close();
  }
  onFailure(callback: (error: Error) => void): void {
    this.failures.push(callback); if (this.failure) callback(this.failure);
  }
  forward(role: WorkerRole, value: ForwardCommand): Promise<WorkerResult> {
    return this.workers[role].call({ op: 'forward', value });
  }
  transfer(value: TransferCommand): Promise<WorkerResult> { return this.workers.decode.call({ op: 'transfer', value }); }
  async stats() {
    return { prefill: await this.workers.prefill.call({ op: 'stats' }), decode: await this.workers.decode.call({ op: 'stats' }) };
  }
  /** Importer first, exporter second, including partial startup and failed workers. */
  close(): Promise<void> {
    return this.closing ??= (async () => { await this.workers.decode.stop(); await this.workers.prefill.stop(); })();
  }
}
