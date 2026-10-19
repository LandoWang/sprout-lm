import { CudaOffload, type Native, type CudaMatvecMode } from './cuda-offload.ts';
import type { TensorDescriptor } from './safetensors.ts';

interface DeviceNative extends Native {
  ipcExport(context: object, tensor: object): Buffer;
  ipcImport(context: object, handle: Buffer, device: number, elements: number): object;
  ipcCopyKV(context: object, source: object, target: object, sourcePages: Int32Array, targetPages: Int32Array,
    tokens: number, blockSize: number, layers: number, width: number): { bytes: number; eventMs: number };
  graphBegin(context: object, pages: number, blockSize: number, output: object, sample: boolean): object;
  graphEnd(context: object, graph: object): void;
  graphAbort(context: object): void;
  graphClose(context: object, graph: object): void;
  graphLaunch(context: object, graph: object, blocks: Int32Array, token: number, position: number): number | Float32Array;
  deviceEnqueueArgmax(context: object, output: object): void;
  deviceVector(context: object, size: number, source: Float32Array | null, reserveBytes: number): object;
  deviceBegin(context: object, blocks: Int32Array): void;
  deviceFinish(context: object): void;
  deviceEmbedding(context: object, matrix: object, index: number, output: object): void;
  deviceMatvec(context: object, matrix: object, input: object, output: object): void;
  deviceAdd(context: object, a: object, b: object, output: object): void;
  deviceNorm(context: object, input: object, weight: object, output: object, epsilon: number): void;
  deviceRope(context: object, input: object, headDim: number, position: number, theta: number): void;
  deviceSwiglu(context: object, gate: object, up: object, output: object): void;
  deviceAttention(context: object, q: object, k: object, v: object, cache: object, output: object, scores: object,
    layer: number, position: number, blockSize: number, numBlocks: number, kvHeads: number, headDim: number): void;
  deviceRead(context: object, tensor: object): Float32Array;
  deviceArgmax(context: object, tensor: object): number;
}
export interface CudaDeviceOptions { graph?: boolean; matvec?: CudaMatvecMode; device?: number; tileMiB?: number; weightLimitMiB?: number; reserveMiB?: number }

export function cudaGraphOption(value = process.env.CUDA_GRAPH): boolean {
  if (value !== undefined && value !== '0' && value !== '1') throw new Error('CUDA_GRAPH must be 0 or 1.');
  return value === '1';
}

/** Opaque device tensors and primitive operators; this class has no model architecture. */
export class CudaDeviceBackend extends CudaOffload {
  private buffers: object[] = [];
  readonly graphEnabled: boolean;
  private allocationReserve: number;
  private get ops(): DeviceNative { return this.native as DeviceNative; }
  constructor(options: CudaDeviceOptions = {}) {
    super({ ...options, weightMode: 'resident' });
    this.graphEnabled = options.graph ?? false;
    this.allocationReserve = (options.reserveMiB ?? 1024) * 1024 ** 2;
  }
  deviceMatrix(descriptor: TensorDescriptor): object { return this.loadMatrix(descriptor); }
  vector(size: number, source?: Float32Array): object {
    const handle = this.ops.deviceVector(this.context, size, source ?? null, this.allocationReserve);
    this.buffers.push(handle); return handle;
  }
  begin(blocks: readonly number[]): void { this.ops.deviceBegin(this.context, Int32Array.from(blocks)); }
  finish(): void { this.ops.deviceFinish(this.context); }
  embedding(matrix: object, index: number, output: object): void { this.ops.deviceEmbedding(this.context, matrix, index, output); }
  linear(matrix: object, input: object, output: object, bias?: object): void {
    this.ops.deviceMatvec(this.context, matrix, input, output);
    if (bias) this.add(output, bias, output);
  }
  add(a: object, b: object, output: object): void { this.ops.deviceAdd(this.context, a, b, output); }
  norm(input: object, weight: object, output: object, epsilon: number): void { this.ops.deviceNorm(this.context, input, weight, output, epsilon); }
  rope(input: object, headDim: number, position: number, theta: number): void { this.ops.deviceRope(this.context, input, headDim, position, theta); }
  swiglu(gate: object, up: object, output: object): void { this.ops.deviceSwiglu(this.context, gate, up, output); }
  attention(q: object, k: object, v: object, cache: object, output: object, scores: object,
    layer: number, position: number, blockSize: number, numBlocks: number, kvHeads: number, headDim: number): void {
    this.ops.deviceAttention(this.context, q, k, v, cache, output, scores, layer, position, blockSize, numBlocks, kvHeads, headDim);
  }
  read(tensor: object): Float32Array { return this.ops.deviceRead(this.context, tensor); }
  argmax(tensor: object): number { return this.ops.deviceArgmax(this.context, tensor); }
  capture(pages: number, blockSize: number, output: object, sample: boolean, submit: () => void): object {
    const graph = this.ops.graphBegin(this.context, pages, blockSize, output, sample);
    try { submit(); this.ops.graphEnd(this.context, graph); return graph; }
    catch (error) {
      this.ops.graphAbort(this.context);
      // abortCapture already closes a plan when capture has not ended yet.
      try { this.ops.graphClose(this.context, graph); } catch { /* Preserve the original capture error. */ }
      throw error;
    }
  }
  enqueueArgmax(output: object): void { this.ops.deviceEnqueueArgmax(this.context, output); }
  replay(graph: object, blocks: readonly number[], token: number, position: number): number | Float32Array {
    return this.ops.graphLaunch(this.context, graph, Int32Array.from(blocks), token, position);
  }
  releaseGraph(graph: object): void { this.ops.graphClose(this.context, graph); }
  exportIPC(tensor: object): string { return this.ops.ipcExport(this.context, tensor).toString('hex'); }
  importIPC(handle: string, device: number, elements: number): object {
    if (!/^[0-9a-f]{128}$/.test(handle)) throw new Error('Invalid CUDA IPC handle encoding.');
    return this.ops.ipcImport(this.context, Buffer.from(handle, 'hex'), device, elements);
  }
  copyKV(source: object, target: object, sourcePages: readonly number[], targetPages: readonly number[],
    tokens: number, blockSize: number, layers: number, width: number) {
    for (const page of [...sourcePages, ...targetPages]) if (!Number.isInteger(page) || page < 0 || page > 0x7fffffff) throw new Error('Invalid IPC page.');
    return this.ops.ipcCopyKV(this.context, source, target, Int32Array.from(sourcePages), Int32Array.from(targetPages), tokens, blockSize, layers, width);
  }
  override close(): void { super.close(); this.buffers = []; }
}
