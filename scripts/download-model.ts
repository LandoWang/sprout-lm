import { createWriteStream, createReadStream } from 'node:fs';
import { mkdir, rename, writeFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';

// Positional directory remains compatible with the original 0.5B downloader.
const directory = process.argv[2] ?? 'models/qwen2.5-0.5b-instruct';
const repo = process.env.MODEL_REPO ?? 'Qwen/Qwen2.5-0.5B-Instruct';
const revision = process.env.MODEL_REVISION ?? (repo === 'Qwen/Qwen2.5-0.5B-Instruct'
  ? '7ae557604adf67be50417f59c2c2f167def9a775' : '');
if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error('Set MODEL_REVISION to a pinned 40-character commit hash.');
const concurrency = Number(process.env.DOWNLOAD_CONCURRENCY ?? 2);
if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 8) throw new Error('Invalid DOWNLOAD_CONCURRENCY.');
await mkdir(directory, { recursive: true });
interface FileInfo { rfilename: string; size: number; blobId: string; lfs?: { sha256: string } }
const metadata = await fetch(`https://huggingface.co/api/models/${repo}/revision/${revision}?blobs=true`);
if (!metadata.ok) throw new Error(`Cannot fetch checkpoint manifest: HTTP ${metadata.status}.`);
const manifest = await metadata.json() as { sha: string; siblings: FileInfo[] };
if (manifest.sha !== revision) throw new Error('Checkpoint revision mismatch.');
async function verify(path: string, info: FileInfo): Promise<boolean> {
  try { if ((await stat(path)).size !== info.size) return false; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error; }
  const hash = createHash(info.lfs ? 'sha256' : 'sha1');
  if (!info.lfs) hash.update(`blob ${info.size}\0`);
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex') === (info.lfs?.sha256 ?? info.blobId);
}
const files = ['config.json', 'tokenizer.json', 'tokenizer_config.json'];
for (const optional of ['generation_config.json', 'LICENSE', 'special_tokens_map.json']) {
  if (manifest.siblings.some(file => file.rfilename === optional)) files.push(optional);
}
const indexName = 'model.safetensors.index.json';
if (manifest.siblings.some(file => file.rfilename === indexName)) {
  const response = await fetch(`https://huggingface.co/${repo}/resolve/${revision}/${indexName}`);
  if (!response.ok) throw new Error(`Cannot fetch shard index: ${response.status}.`);
  const index = await response.json() as { weight_map: Record<string, string> };
  if (!index.weight_map || typeof index.weight_map !== 'object') throw new Error('Invalid shard index.');
  files.push(indexName, ...new Set(Object.values(index.weight_map)));
} else files.push('model.safetensors');
for (const name of files) {
  if (!/^[a-zA-Z0-9_.-]+$/.test(name) || name === '.' || name === '..') throw new Error(`Unsafe checkpoint filename: ${name}.`);
  if (!manifest.siblings.some(file => file.rfilename === name)) throw new Error(`Missing checkpoint entry: ${name}.`);
}
console.log(JSON.stringify({ repo, revision, directory, files: files.length,
  downloadBytes: files.reduce((n, name) => n + manifest.siblings.find(file => file.rfilename === name)!.size, 0) }));
async function download(name: string): Promise<void> {
  const destination = join(directory, name);
  const info = manifest.siblings.find(file => file.rfilename === name);
  if (!info) throw new Error(`Missing checkpoint entry: ${name}.`);
  if (await verify(destination, info)) { console.log(`Verified: ${name}`); return; }
  console.log(`Downloading ${name} (${repo}@${revision})`);
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch(`https://huggingface.co/${repo}/resolve/${revision}/${name}`);
      if (!response.ok || !response.body) throw new Error(`Download failed: ${name}: HTTP ${response.status}.`);
      await pipeline(response.body, createWriteStream(destination + '.part'));
      if (!await verify(destination + '.part', info)) throw new Error(`Checksum mismatch: ${name}.`);
      await rename(destination + '.part', destination);
      console.log(`Ready: ${name}`);
      return;
    } catch (error) { if (attempt >= 2) throw error; console.warn(`Retry ${attempt + 1}: ${name}: ${error}`); }
  }
}
let cursor = 0;
await Promise.all(Array.from({ length: Math.min(concurrency, files.length) }, async () => {
  while (cursor < files.length) await download(files[cursor++]);
}));
await writeFile(join(directory, 'sprout-model.json'), JSON.stringify({ repo, revision }, null, 2) + '\n');
console.log(`Ready checkpoint: ${directory}. Select the CPU service or CUDA smoke entry point described in README.md.`);
