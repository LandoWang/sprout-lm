import { createRequire } from 'node:module';
import type { MatrixOperator } from './qwen.ts';
import type { TensorDescriptor } from './safetensors.ts';

export interface Native {
  create(device: number, tileMiB: number, directWeights: boolean): object;
  mapMatrix(path: string, offset: number, dtype: string, rows: number, cols: number): object;
  uploadMatrix(context: object, matrix: object, limitBytes: number, reserveBytes: number): void;
  row(matrix: object, index: number): Float32Array;
  matvec(context: object, matrix: object, input: Float32Array): Float32Array;
  stats(context: object): Record<string, number | string>;
  closeContext(context: object): void;
  closeMatrix(matrix: object): void;
}

export type CudaWeightMode = 'offload' | 'resident';
export type CudaMatvecMode = 'direct' | 'convert';

export function cudaMatvecOption(value = process.env.CUDA_MATVEC): CudaMatvecMode {
  if (value !== undefined && value !== 'direct' && value !== 'convert') throw new Error('CUDA_MATVEC must be direct or convert.');
  return value ?? 'direct';
}

/** Experimental GEMV with streamed or resident weights. Attention, norms, activations and KV still run on CPU.
 * Calls are synchronous; use the dedicated smoke CLI, not a latency-sensitive HTTP server.
 */
export class CudaOffload {
  protected native: Native;
  protected context: object;
  private matrices: object[] = [];
  private closed = false;
  readonly weightMode: CudaWeightMode;
  readonly matvecMode: CudaMatvecMode;
  private weightLimitBytes: number;
  private reserveBytes: number;
  mappedWeightBytes = 0;
  constructor(options: { device?: number; tileMiB?: number; weightMode?: CudaWeightMode;
    weightLimitMiB?: number; reserveMiB?: number; matvec?: CudaMatvecMode } = {}) {
    this.weightMode = options.weightMode ?? 'offload';
    this.matvecMode = cudaMatvecOption(options.matvec);
    if (this.weightMode !== 'offload' && this.weightMode !== 'resident') throw new Error('Invalid CUDA weight mode.');
    const toBytes = (value: number) => {
      const bytes = value * 1024 ** 2;
      if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error('Invalid CUDA memory budget.');
      return bytes;
    };
    this.weightLimitBytes = toBytes(options.weightLimitMiB ?? 0);
    this.reserveBytes = toBytes(options.reserveMiB ?? 1024);
    this.native = createRequire(import.meta.url)('../build/sprout_cuda.node') as Native;
    this.context = this.native.create(options.device ?? 0, options.tileMiB ?? 128, this.matvecMode === 'direct');
  }
  protected loadMatrix(descriptor: TensorDescriptor): object {
    if (this.closed) throw new Error('CUDA backend is closed.');
    if (descriptor.shape.length !== 2) throw new Error('CUDA matrix must have rank two.');
    const [rows, cols] = descriptor.shape;
    const handle = this.native.mapMatrix(descriptor.path, descriptor.offset, descriptor.dtype, rows, cols);
    try {
      if (this.weightMode === 'resident') this.native.uploadMatrix(this.context, handle, this.weightLimitBytes, this.reserveBytes);
    } catch (error) { this.native.closeMatrix(handle); throw error; }
    this.matrices.push(handle); this.mappedWeightBytes += descriptor.bytes;
    return handle;
  }
  matrix = (descriptor: TensorDescriptor): MatrixOperator => {
    const handle = this.loadMatrix(descriptor), [rows] = descriptor.shape;
    return {
      row: index => this.native.row(handle, index),
      matvec: (input, bias) => {
        if (bias && bias.length !== rows) throw new Error('CUDA bias shape mismatch.');
        const output = this.native.matvec(this.context, handle, input);
        if (bias) for (let i = 0; i < rows; i++) output[i] += bias[i];
        return output;
      },
    };
  };
  stats(): Record<string, number | string> {
    return { ...this.native.stats(this.context), mappedWeightBytes: this.mappedWeightBytes,
      weightMode: this.weightMode, matvecMode: this.matvecMode, weightLimitBytes: this.weightLimitBytes, reserveBytes: this.reserveBytes };
  }
  close() {
    if (this.closed) return;
    this.native.closeContext(this.context);
    for (const matrix of this.matrices) this.native.closeMatrix(matrix);
    this.matrices = []; this.closed = true;
  }
}
