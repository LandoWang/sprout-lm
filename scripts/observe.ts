import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createObservatoryServer, type ObservatoryRoot } from '../src/observatory-server.ts';

const directories = process.argv.slice(2);
if (directories.includes('--help')) {
  console.log('Usage: npm run observe -- [EVENT_ROOT_OR_RUN_DIRECTORY ...]\nDefault: runs/events plus the saved A100 32B experiment.\nOBSERVE_PORT=3001; bound to 127.0.0.1.');
  process.exit(0);
}
const roots: ObservatoryRoot[] = directories.length ? directories.map(directory => ({ directory: resolve(directory) })) : [
  { directory: resolve(process.env.EVENT_DIR ?? 'runs/events') },
  { directory: fileURLToPath(new URL('../snapshot/2026-10-20-request-events/raw-run',import.meta.url)), label: 'A100 · Qwen2.5-32B · 实验快照', snapshot: true },
];
const port = Number(process.env.OBSERVE_PORT ?? 3001);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid OBSERVE_PORT.');
const server = createObservatoryServer(roots);
server.on('error',error => { console.error(error.message); process.exitCode = 1; });
server.listen(port,'127.0.0.1',() => console.log(`SproutLM Observatory → http://127.0.0.1:${port}\nRead-only event viewer; no inference process is started.`));
for (const signal of ['SIGINT','SIGTERM']) process.on(signal,() => { server.closeAllConnections(); server.close(); });
