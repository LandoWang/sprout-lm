import { fork, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { EventClient, type EventConnection } from './observation.ts';

export interface ObserverOptions { directory: string; maxFileBytes?: number; maxFiles?: number; maxQueueBytes?: number }
/** One collector child per serving group. Its failure is deliberately not a PD worker failure. */
export class ObserverProcess {
  readonly client: EventClient;
  private stopping?: Promise<void>;
  readonly child: ChildProcess;
  readonly connection: EventConnection;
  readonly directory: string;
  private readonly socketDirectory: string;
  private readonly exited: Promise<void>;
  private constructor(child: ChildProcess, connection: EventConnection, directory: string,
    socketDirectory: string, exited: Promise<void>) {
    this.child = child; this.connection = connection; this.directory = directory;
    this.socketDirectory = socketDirectory; this.exited = exited;
    this.client = new EventClient(connection, 'coordinator');
  }
  static async start(options: ObserverOptions): Promise<ObserverProcess> {
    const runId = randomUUID(), directory = resolve(options.directory, 'run-' + runId);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    // Keep the Unix pathname short even when the log directory is deeply nested.
    const socketDirectory = await mkdtemp(join(tmpdir(), 'sprout-ev-'));
    const connection = { runId, socketPath: join(socketDirectory, 'events.sock') };
    const child = fork(fileURLToPath(new URL('./observer-worker.ts', import.meta.url)), [], {
      execArgv: [], stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
    });
    let dead = false;
    const exited = new Promise<void>(resolveExit => {
      child.once('exit', () => { dead = true; resolveExit(); });
      child.once('error', () => { if (!child.pid) { dead = true; resolveExit(); } });
    });
    try {
      await new Promise<void>((resolveReady, reject) => {
        const cleanup = () => { clearTimeout(timer); child.removeListener('message', message); child.removeListener('exit', exit); child.removeListener('error', error); };
        const message = (value: unknown) => {
          const m = value as { ready?: boolean; error?: string };
          cleanup(); m?.ready ? resolveReady() : reject(new Error(m?.error ?? 'Invalid observer startup reply.'));
        };
        const exit = () => { cleanup(); reject(new Error('Observer exited during startup.')); };
        const error = (e: Error) => { cleanup(); reject(e); };
        const timer = setTimeout(() => { cleanup(); reject(new Error('Observer startup timeout.')); }, 5000);
        child.once('message', message); child.once('exit', exit); child.once('error', error);
        child.send({ op: 'start', options: { ...options, directory, ...connection } }, e => { if (e) error(e); });
      });
      child.on('error', e => console.error('[observation]', e.message));
      child.on('exit', (code, signal) => {
        if (code !== 0 || signal) console.error('[observation] collector exited; traces may be incomplete.', code ?? signal);
        void rm(socketDirectory, { recursive: true, force: true }).catch(() => {});
      });
      return new ObserverProcess(child, connection, directory, socketDirectory, exited);
    } catch (error) {
      if (!dead) child.kill('SIGKILL');
      await exited; await rm(socketDirectory, { recursive: true, force: true }); throw error;
    }
  }
  close(): Promise<void> {
    return this.stopping ??= (async () => {
      await this.client.close();
      const timer = setTimeout(() => this.child.kill('SIGKILL'), 2000);
      try {
        if (this.child.connected) this.child.send({ op: 'stop' }, () => {});
        await this.exited;
      } finally { clearTimeout(timer); await rm(this.socketDirectory, { recursive: true, force: true }); }
    })();
  }
}
