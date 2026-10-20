import type { CudaDeviceOptions } from './cuda-device.ts';
import type { KVLayout, ModelSpec } from './runner.ts';
import type { EventConnection } from './observation.ts';

export type WorkerRole = 'prefill' | 'decode';
export interface WorkerInit {
  role: WorkerRole; directory: string; device: number; tokens: number; blockSize: number;
  cuda: CudaDeviceOptions;
  events?: EventConnection;
}
export interface WorkerInfo {
  role: WorkerRole; pid: number; device: number; spec: ModelSpec; layout: KVLayout;
  ipc?: string; stats: Record<string, number | string>;
}
export interface ForwardCommand {
  planId?: string;
  request: string; from: number; tokens: number[]; blocks: number[]; output: 'none' | 'sample' | 'logits';
}
export interface TransferCommand { planId?: string; request: string; sourcePages: number[]; targetPages: number[]; tokens: number }
export interface WorkerResult {
  request: string; token?: number; logits?: number[]; bytes?: number; eventMs?: number;
  startedAt: number; endedAt: number;
}
export type Command = { op: 'init'; value: WorkerInit } | { op: 'attach'; value: WorkerInfo }
  | { op: 'forward'; value: ForwardCommand } | { op: 'transfer'; value: TransferCommand }
  | { op: 'stats' } | { op: 'close' };
export interface Envelope { id: number; command: Command }
export interface Reply { id: number; value?: unknown; error?: string }

export interface PDExecutors {
  readonly prefill: WorkerInfo; readonly decode: WorkerInfo;
  forward(role: WorkerRole, command: ForwardCommand): Promise<WorkerResult>;
  transfer(command: TransferCommand): Promise<WorkerResult>;
  onFailure(callback: (error: Error) => void): void;
  close(): Promise<void>;
}
