import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { lstat, readFile, readdir, realpath } from 'node:fs/promises';
import { join } from 'node:path';
import { readEventLog } from './event-log.ts';
import { analyzeRun, type RunAnalysis } from './trace-analysis.ts';

export interface ObservatoryRoot { directory: string; label?: string; snapshot?: boolean }
/** Read-only companion server. It has no connection to the inference processes. */
export function createObservatoryServer(roots: ObservatoryRoot[]) {
  const allowed = new Map<string, { directory: string; label: string; snapshot: boolean }>();
  const cache = new Map<string, { at: number; value: RunAnalysis }>();
  const pending = new Map<string, Promise<RunAnalysis>>();
  const assets = new Map([
    ['/', ['../observatory/index.html','text/html; charset=utf-8']],
    ['/app.js', ['../observatory/app.js','text/javascript; charset=utf-8']],
    ['/style.css', ['../observatory/style.css','text/css; charset=utf-8']],
    ['/sproutlm.svg', ['../docs/assets/sproutlm.svg','image/svg+xml']],
  ]);
  async function discover() {
    const runs: Array<{ key: string; runId: string; label: string; snapshot: boolean; modifiedAt: number }> = [];
    let skipped = 0;
    const add = async (directory: string, root: ObservatoryRoot, name: string) => {
      if (runs.length >= 100) { skipped++; return; }
      try {
        const stat = await lstat(join(directory,'manifest.json'));
        if (!stat.isFile() || stat.size > 65536) { skipped++; return; }
        const manifest = JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));
        if (manifest.version !== 1 || typeof manifest.runId !== 'string') { skipped++; return; }
        const key = createHash('sha256').update(directory).digest('hex').slice(0,24);
        const entry = { directory, label: root.label ?? name, snapshot: root.snapshot === true };
        allowed.set(key,entry);
        runs.push({ key, runId: manifest.runId, label: entry.label, snapshot: entry.snapshot, modifiedAt: stat.mtimeMs });
      } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') skipped++; }
    };
    allowed.clear();
    for (const root of roots) {
      try {
        const directory = await realpath(root.directory);
        const entries = await readdir(directory,{ withFileTypes: true });
        if (entries.some(e => e.name === 'manifest.json')) await add(directory,root,directory.split('/').at(-1)!);
        else for (const entry of entries.filter(e => e.isDirectory() && e.name.startsWith('run-')).sort((a,b) => b.name.localeCompare(a.name)).slice(0,100)) {
          await add(join(directory,entry.name),root,entry.name);
        }
      } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') skipped++; }
    }
    return { runs: runs.sort((a,b) => Number(a.snapshot)-Number(b.snapshot) || b.modifiedAt-a.modifiedAt), skipped };
  }
  let discovering: ReturnType<typeof discover> | undefined;
  const list = () => discovering ??= discover().finally(() => { discovering = undefined; });
  const server = createServer((req,res) => {
    void (async () => {
      res.setHeader('Cache-Control','no-store');
      res.setHeader('X-Content-Type-Options','nosniff');
      res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");
      const json = (status: number, value: unknown) => {
        res.writeHead(status,{ 'content-type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(value));
      };
      try {
        // Loopback binding plus Host/Origin checks prevent remote pages from reading local traces.
        const hosts = ['127.0.0.1','localhost','[::1]'].map(h => `${h}:${req.socket.localPort}`);
        if (!hosts.includes(req.headers.host ?? '') || (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`)) {
          json(403,{ error: 'Local same-origin access only.' }); return;
        }
        if (req.method !== 'GET') { req.resume(); json(405,{ error: 'Read-only: GET only.' }); return; }
        const url = new URL(req.url ?? '/','http://localhost');
        const asset = assets.get(url.pathname);
        if (asset) {
          const body = await readFile(new URL(asset[0],import.meta.url));
          res.writeHead(200,{ 'content-type': asset[1] }); res.end(body); return;
        }
        if (url.pathname === '/api/runs') { json(200,await list()); return; }
        if (url.pathname !== '/api/run') { json(404,{ error: 'Not found.' }); return; }
        const key = url.searchParams.get('run') ?? '';
        if (!allowed.has(key)) await list();
        const entry = allowed.get(key);
        if (!entry) { json(404,{ error: 'Unknown run. Refresh the run list.' }); return; }
        const hit = cache.get(key);
        if (hit && Date.now()-hit.at < 2000) { json(200,hit.value); return; }
        if (!pending.has(key)) {
          if (pending.size >= 2) { json(503,{ error: 'Reading other runs; retry shortly.' }); return; }
          const work = readEventLog(entry.directory,25000).then(log => {
            const value = analyzeRun(log);
            if (cache.size >= 2) cache.delete(cache.keys().next().value!);
            cache.set(key,{ at: Date.now(), value }); return value;
          }).finally(() => { pending.delete(key); });
          pending.set(key,work);
        }
        json(200,await pending.get(key));
      } catch (error) {
        console.error('[observatory]',error instanceof Error ? error.message : error);
        if (!res.headersSent) json(500,{ error: 'Could not read this run. Check the server log and refresh.' }); else res.end();
      }
    })();
  });
  return server;
}
