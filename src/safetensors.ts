import { open, type FileHandle } from 'node:fs/promises';

interface TensorEntry { dtype: 'F32' | 'BF16' | 'F16'; shape: number[]; data_offsets: [number, number] }
export interface TensorDescriptor { path: string; offset: number; bytes: number; dtype: TensorEntry['dtype']; shape: number[] }

async function readExactly(file: FileHandle, buffer: Buffer, position: number): Promise<void> {
  let offset = 0;
  while (offset < buffer.length) {
    const { bytesRead } = await file.read(buffer, offset, buffer.length - offset, position + offset);
    if (!bytesRead) throw new Error('Truncated safetensors file.');
    offset += bytesRead;
  }
}

/** Reads only the requested tensors; conversion uses a bounded scratch buffer. */
export class SafeTensors {
  private readonly file: FileHandle;
  private readonly start: number;
  private readonly entries: Map<string, TensorEntry>;
  private readonly path: string;

  private constructor(file: FileHandle, start: number, entries: Map<string, TensorEntry>, path: string) {
    this.file = file; this.start = start; this.entries = entries; this.path = path;
  }

  static async open(path: string): Promise<SafeTensors> {
    const file = await open(path, 'r');
    try {
      const { size } = await file.stat();
      const prefix = Buffer.alloc(8);
      await readExactly(file, prefix, 0);
      const headerSize = Number(prefix.readBigUInt64LE());
      if (!Number.isSafeInteger(headerSize) || headerSize <= 0 || headerSize > 16 * 1024 * 1024 || headerSize + 8 > size) {
        throw new Error('Invalid safetensors header length.');
      }
      const header = Buffer.alloc(headerSize);
      await readExactly(file, header, 8);
      const parsed = JSON.parse(header.toString('utf8'));
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Invalid safetensors header.');
      const entries = new Map<string, TensorEntry>();
      for (const [name, raw] of Object.entries(parsed)) {
        if (name === '__metadata__') continue;
        const entry = raw as TensorEntry;
        if (!entry || !['F32', 'BF16', 'F16'].includes(entry.dtype) || !Array.isArray(entry.shape)
          || entry.shape.some(n => !Number.isSafeInteger(n) || n <= 0)
          || !Array.isArray(entry.data_offsets) || entry.data_offsets.length !== 2) {
          throw new Error(`Unsupported or malformed tensor: ${name}.`);
        }
        const elements = entry.shape.reduce((a, b) => a * b, 1);
        const [begin, end] = entry.data_offsets;
        const bytes = elements * (entry.dtype === 'F32' ? 4 : 2);
        if (!Number.isSafeInteger(bytes) || !Number.isSafeInteger(begin) || !Number.isSafeInteger(end)
          || begin < 0 || end - begin !== bytes || end > size - 8 - headerSize) {
          throw new Error(`Invalid tensor offsets: ${name}.`);
        }
        entries.set(name, entry);
      }
      let previousEnd = 0;
      for (const entry of [...entries.values()].sort((a, b) => a.data_offsets[0] - b.data_offsets[0])) {
        if (entry.data_offsets[0] !== previousEnd) throw new Error('Safetensors data contains gaps or overlapping tensors.');
        previousEnd = entry.data_offsets[1];
      }
      if (previousEnd !== size - headerSize - 8) throw new Error('Safetensors data size does not match its header.');
      return new SafeTensors(file, headerSize + 8, entries, path);
    } catch (error) { await file.close(); throw error; }
  }

  has(name: string): boolean { return this.entries.has(name); }
  close(): Promise<void> { return this.file.close(); }

  descriptor(name: string, shape: number[]): TensorDescriptor {
    const entry = this.entries.get(name);
    if (!entry) throw new Error(`Missing tensor: ${name}.`);
    if (JSON.stringify(shape) !== JSON.stringify(entry.shape)) throw new Error(`Shape mismatch for ${name}: expected [${shape}], got [${entry.shape}].`);
    return { path: this.path, offset: this.start + entry.data_offsets[0],
      bytes: entry.data_offsets[1] - entry.data_offsets[0], dtype: entry.dtype, shape: [...entry.shape] };
  }

  async tensor(name: string, shape: number[]): Promise<Float32Array> {
    const entry = this.entries.get(name);
    if (!entry) throw new Error(`Missing tensor: ${name}.`);
    if (JSON.stringify(shape) !== JSON.stringify(entry.shape)) throw new Error(`Shape mismatch for ${name}: expected [${shape}], got [${entry.shape}].`);
    const count = shape.reduce((a, b) => a * b, 1);
    const output = new Float32Array(count);
    const bits = new Uint32Array(output.buffer);
    const stride = entry.dtype === 'F32' ? 4 : 2;
    const buffer = Buffer.alloc(Math.min(8 * 1024 * 1024, count * stride));
    for (let offset = 0; offset < count; offset += buffer.length / stride) {
      const length = Math.min(buffer.length / stride, count - offset);
      await readExactly(this.file, buffer.subarray(0, length * stride), this.start + entry.data_offsets[0] + offset * stride);
      for (let i = 0; i < length; i++) {
        if (entry.dtype === 'F32') output[offset + i] = buffer.readFloatLE(i * 4);
        else if (entry.dtype === 'BF16') bits[offset + i] = buffer.readUInt16LE(i * 2) << 16;
        else {
          const value = buffer.readUInt16LE(i * 2);
          const sign = value & 0x8000 ? -1 : 1;
          const exponent = (value >>> 10) & 31;
          const fraction = value & 1023;
          output[offset + i] = sign * (exponent === 0 ? fraction * 2 ** -24
            : exponent === 31 ? (fraction ? NaN : Infinity) : (1 + fraction / 1024) * 2 ** (exponent - 15));
        }
      }
    }
    return output;
  }
}
