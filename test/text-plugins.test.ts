import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { defaultTextPlugins, validateTokenizer } from '../src/text-plugins.ts';
import { TinyTransformer } from '../src/model.ts';

test('plugin factories are independent, reject unknown names and verify model special IDs', async () => {
  const plugins = defaultTextPlugins();
  const tokenizer = await plugins.tokenizer('byte', '');
  validateTokenizer(tokenizer, new TinyTransformer().spec);
  plugins.registerTokenizer('custom', async () => tokenizer);
  assert.equal(await plugins.tokenizer('custom', 'separate-resources'), tokenizer);
  await assert.rejects(plugins.tokenizer('missing', ''), /Unknown tokenizer/);
  await assert.rejects(plugins.template('missing', ''), /Unknown template/);
  assert.throws(() => plugins.registerTokenizer('byte', async () => tokenizer), /Duplicate/);
  assert.throws(() => validateTokenizer(tokenizer, { ...new TinyTransformer().spec, vocabSize: 128 }), /do not match/);
  assert.throws(() => validateTokenizer(tokenizer, { ...new TinyTransformer().spec, eosTokenIds: [1] }), /do not match/);
});

test('ByteLevel plugin streams split UTF-8, literal tokens, BOM and independent decoders', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-tokenizer-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  // The symbols here are GPT-2's byte alphabet for UTF-8 bytes of 你, 🌱 and BOM.
  const tokens = ['ä', '½', 'ł', 'ð', 'Ł', 'Į', '±', 'ï', '»', '¿', 'a'];
  const vocab = Object.fromEntries(tokens.map((token, id) => [token, id]));
  const added = [{ id: 11, content: '<eos>', special: true }, { id: 12, content: '<literal>', special: false }];
  await writeFile(join(directory, 'tokenizer.json'), JSON.stringify({
    version: '1.0', added_tokens: added, normalizer: null,
    pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: true },
    post_processor: null, decoder: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: true },
    model: { type: 'BPE', vocab, merges: [], unk_token: null },
  }));
  await writeFile(join(directory, 'tokenizer_config.json'), JSON.stringify({ add_bos_token: false, clean_up_tokenization_spaces: false }));
  const tokenizer = await defaultTextPlugins().tokenizer('hf-bytelevel', directory);
  assert.deepEqual(tokenizer.encode('你🌱'), [0, 1, 2, 3, 4, 5, 6]);
  const a = tokenizer.decoder(), b = tokenizer.decoder();
  assert.equal(a.push(0), '');
  assert.equal(b.push(10), 'a');
  assert.equal(a.push(1), '');
  assert.equal(a.push(2), '你');
  assert.equal(a.push(11), '');
  assert.equal(a.push(12), '<literal>');
  assert.equal(a.finish(), '');
  assert.equal(b.finish(), '');
  const bom = tokenizer.decoder();
  assert.equal([7, 8, 9].map(id => bom.push(id)).join('') + bom.finish(), '\ufeff');
  const invalid = tokenizer.decoder();
  assert.equal(invalid.push(0), '');
  assert.equal(invalid.finish(), '\ufffd');
});

test('Jinja chat template loads independently from tokenizer implementation', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'sprout-template-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(join(directory, 'tokenizer_config.json'), JSON.stringify({
    chat_template: "{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
  }));
  const template = await defaultTextPlugins().template('jinja', directory);
  assert.equal(template.render([{ role: 'user', content: '你好' }]), 'user:你好\nassistant:');
});
