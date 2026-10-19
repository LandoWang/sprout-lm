export type NumericVector = Float32Array | Float64Array;
export type KVDtype = 'float32' | 'float64';
export type FinishReason = 'eos' | 'length';

export interface KVStore {
  set(layer: number, position: number, key: NumericVector, value: NumericVector): void;
  get(layer: number, position: number): { key: NumericVector; value: NumericVector };
  deviceView?(): { storage: DeviceKVStorage; blocks: readonly number[]; length: number };
}

export interface KVLayout { numBlocks: number; blockSize: number; layers: number; width: number }
/** Opaque storage; logical ownership and prefix references remain in KVCache. */
export interface DeviceKVStorage extends KVLayout { readonly owner: object; readonly handle: object }

/** Only execution metadata belongs here. Tokenizers and chat templates are separate plugins. */
export interface ModelSpec {
  id: string;
  layers: number;
  hiddenSize: number;
  attentionHeads: number;
  kvHeads: number;
  headDim: number;
  vocabSize: number;
  maxContextLength: number;
  kvDtype: KVDtype;
  eosTokenIds: readonly number[];
}

/** Synchronous completion: returned work is safe to publish and its pages safe to release. */
export interface ModelRunner {
  readonly spec: ModelSpec;
  forwardToken(token: number, position: number, kv: KVStore, outputLogits?: boolean): NumericVector | undefined;
  /** Device runners can sample on GPU and return only the token ID. */
  forwardSample?(token: number, position: number, kv: KVStore, outputToken: boolean): number | undefined;
  createKVStorage?(layout: KVLayout): DeviceKVStorage;
}

export function argmax(values: NumericVector): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
