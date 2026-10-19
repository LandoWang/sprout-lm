import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { Tokenizer as HFTokenizer } from '@huggingface/tokenizers';
import { Template } from '@huggingface/jinja';
import { byteTokenizer, type Tokenizer } from './tokenizer.ts';
import type { ModelSpec } from './runner.ts';

export interface ChatMessage { role: 'system' | 'user' | 'assistant'; content: string }
export interface ChatTemplate { render(messages: readonly ChatMessage[]): string }
export type TokenizerFactory = (directory: string) => Promise<Tokenizer>;
export type TemplateFactory = (directory: string) => Promise<ChatTemplate>;

/** Config selects a factory. Neither the scheduler nor the runner knows the implementation. */
export class TextPluginRegistry {
  private readonly tokenizers = new Map<string, TokenizerFactory>();
  private readonly templates = new Map<string, TemplateFactory>();
  registerTokenizer(name: string, factory: TokenizerFactory): this {
    if (this.tokenizers.has(name)) throw new Error(`Duplicate tokenizer plugin: ${name}.`);
    this.tokenizers.set(name, factory); return this;
  }
  registerTemplate(name: string, factory: TemplateFactory): this {
    if (this.templates.has(name)) throw new Error(`Duplicate template plugin: ${name}.`);
    this.templates.set(name, factory); return this;
  }
  async tokenizer(name: string, directory: string): Promise<Tokenizer> {
    const factory = this.tokenizers.get(name);
    if (!factory) throw new Error(`Unknown tokenizer plugin: ${name}.`);
    return factory(directory);
  }
  async template(name: string, directory: string): Promise<ChatTemplate> {
    const factory = this.templates.get(name);
    if (!factory) throw new Error(`Unknown template plugin: ${name}.`);
    return factory(directory);
  }
}

/** GPT-2/Qwen ByteLevel alphabet, inverted to reconstruct bytes before UTF-8 decoding. */
function byteAlphabet(): Map<string, number> {
  const bytes = Array.from({ length: 256 }, (_, n) => n).filter(n => (n >= 33 && n <= 126) || (n >= 161 && n <= 172) || n >= 174);
  const map = new Map(bytes.map(n => [String.fromCodePoint(n), n]));
  let extra = 0;
  for (let n = 0; n < 256; n++) if (!bytes.includes(n)) map.set(String.fromCodePoint(256 + extra++), n);
  return map;
}

export async function loadByteLevelTokenizer(directory: string): Promise<Tokenizer> {
  const jsonText = await readFile(join(directory, 'tokenizer.json'), 'utf8');
  const configText = await readFile(join(directory, 'tokenizer_config.json'), 'utf8');
  const json = JSON.parse(jsonText), config = JSON.parse(configText);
  if (json.decoder?.type !== 'ByteLevel' || json.model?.type !== 'BPE' || config.add_bos_token === true) {
    throw new Error('hf-bytelevel plugin supports BPE / ByteLevel without automatic BOS only.');
  }
  const hf = new HFTokenizer(json, config);
  const added = hf.get_added_tokens_decoder();
  const special = new Set<number>();
  const literals = new Set<number>();
  for (const [id, token] of added) {
    if (token.special) special.add(id); else literals.add(id);
  }
  const alphabet = byteAlphabet();
  let maxTokenId = 0;
  for (const id of hf.get_vocab().values()) maxTokenId = Math.max(maxTokenId, id);
  return {
    metadata: {
      name: 'hf-bytelevel', fingerprint: createHash('sha256').update(jsonText).update(configText).digest('hex'),
      maxTokenId, specialTokenIds: [...special],
    },
    encode: text => hf.encode(text, { add_special_tokens: false }).ids,
    decoder() {
      const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
      const encoder = new TextEncoder();
      return {
        push(token: number) {
          if (special.has(token)) return '';
          const piece = hf.id_to_token(token);
          if (piece === undefined) return ''; // Padded embedding rows have no text vocabulary entry.
          const bytes = literals.has(token) ? encoder.encode(piece) : Uint8Array.from([...piece].map(char => {
            const byte = alphabet.get(char);
            if (byte === undefined) throw new Error(`Invalid ByteLevel vocabulary character: ${char}.`);
            return byte;
          }));
          return decoder.decode(bytes, { stream: true });
        },
        finish: () => decoder.decode(),
      };
    },
  };
}

export async function loadJinjaTemplate(directory: string): Promise<ChatTemplate> {
  const config = JSON.parse(await readFile(join(directory, 'tokenizer_config.json'), 'utf8'));
  if (typeof config.chat_template !== 'string') throw new Error('A single string chat_template is required.');
  const template = new Template(config.chat_template);
  return {
    render(messages) {
      return template.render({ messages, add_generation_prompt: true, tools: null,
        bos_token: config.bos_token ?? '', eos_token: config.eos_token ?? '', pad_token: config.pad_token ?? '' });
    },
  };
}

export function defaultTextPlugins(): TextPluginRegistry {
  return new TextPluginRegistry().registerTokenizer('byte', async () => byteTokenizer)
    .registerTokenizer('hf-bytelevel', loadByteLevelTokenizer).registerTemplate('jinja', loadJinjaTemplate);
}

export function validateTokenizer(tokenizer: Tokenizer, model: ModelSpec): void {
  const info = tokenizer.metadata;
  if (!info || info.maxTokenId >= model.vocabSize || model.eosTokenIds.some(id => !info.specialTokenIds.includes(id))) {
    throw new Error('Tokenizer vocabulary or special token IDs do not match the model.');
  }
  // This checks ranges and special IDs; semantic compatibility still requires fixed resources/reference samples.
}
