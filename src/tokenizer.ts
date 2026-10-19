export function encode(text: string): number[] {
  return [256, ...new TextEncoder().encode(text)];
}

export interface Tokenizer {
  readonly metadata?: { name: string; fingerprint: string; maxTokenId: number; specialTokenIds: readonly number[] };
  encode(text: string): number[];
  decoder(): { push(token: number): string; finish(): string };
}

/** Model-specific text handling lives outside the scheduler and the serving protocol. */
export const byteTokenizer: Tokenizer = {
  metadata: { name: 'byte', fingerprint: 'utf8-byte-bos256-v1', maxTokenId: 256, specialTokenIds: [256] },
  encode,
  decoder() {
    const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
    return {
      // BOS is not text; UTF-8 characters can span several byte tokens.
      push: token => token === 256 ? '' : decoder.decode(Uint8Array.of(token), { stream: true }),
      finish: () => decoder.decode(),
    };
  },
};
