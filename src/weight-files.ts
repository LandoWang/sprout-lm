import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { SafeTensors, type TensorDescriptor } from './safetensors.ts';

/** Single-file and indexed checkpoints share the same tensor lookup. At most eight files stay open. */
export class WeightFiles {
  private directory: string;
  private mapping?: Record<string, string>;
  private files = new Map<string, SafeTensors>();
  private constructor(directory: string, mapping?: Record<string, string>) {
    this.directory = resolve(directory); this.mapping = mapping;
  }
  static async open(directory: string): Promise<WeightFiles> {
    try {
      const index = JSON.parse(await readFile(resolve(directory, 'model.safetensors.index.json'), 'utf8'));
      if (!index.weight_map || typeof index.weight_map !== 'object' || Array.isArray(index.weight_map)
        || !Object.keys(index.weight_map).length) throw new Error('Invalid safetensors weight map.');
      for (const name of Object.values(index.weight_map)) {
        if (typeof name !== 'string' || !/^[a-zA-Z0-9_-][a-zA-Z0-9_.-]*\.safetensors$/.test(name)) {
          throw new Error('Invalid checkpoint shard filename.');
        }
      }
      return new WeightFiles(directory, index.weight_map);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return new WeightFiles(directory);
      throw error;
    }
  }
  private async file(name: string): Promise<SafeTensors> {
    const shard = this.mapping ? (Object.hasOwn(this.mapping, name) ? this.mapping[name] : undefined) : 'model.safetensors';
    if (!shard) throw new Error(`Missing tensor in shard index: ${name}.`);
    let file = this.files.get(shard);
    if (file) { this.files.delete(shard); this.files.set(shard, file); return file; }
    if (this.files.size >= 8) {
      const [oldName, oldFile] = this.files.entries().next().value!;
      await oldFile.close(); this.files.delete(oldName);
    }
    file = await SafeTensors.open(resolve(this.directory, shard));
    this.files.set(shard, file); return file;
  }
  async tensor(name: string, shape: number[]): Promise<Float32Array> {
    return (await this.file(name)).tensor(name, shape);
  }
  async descriptor(name: string, shape: number[]): Promise<TensorDescriptor> {
    return (await this.file(name)).descriptor(name, shape);
  }
  async close(): Promise<void> {
    const results = await Promise.allSettled([...this.files.values()].map(file => file.close()));
    this.files.clear();
    for (const result of results) if (result.status === 'rejected') throw result.reason;
  }
}
