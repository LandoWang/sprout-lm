import { Engine } from './engine.ts';
import { encode } from './model.ts';

const json = process.argv.includes('--json');
const pressure = process.argv.includes('--pressure');
const engine = new Engine({
  blockSize: 4, numBlocks: pressure ? 5 : 18,
  maxSequences: 3, maxBatchTokens: 6, prefillChunkSize: 4,
});
const reports: ReturnType<Engine['step']>[] = [];
if (!json) {
  console.log('SproutLM · CPU / fixed UNTRAINED Transformer weights');
  console.log('Token IDs demonstrate scheduling and numerical correctness, not language quality.\n');
}

engine.add({ id: 'A', prompt: encode('shared: hi'), maxNewTokens: 7 });
engine.add({ id: 'B', prompt: encode('other'), maxNewTokens: 4 });
let iterations = 0;
while (engine.pending) {
  if (iterations === 2) {
    // Arrives while A/B are running. A has already published two shared prefix pages.
    engine.add({ id: 'C', prompt: encode('shared: bye'), maxNewTokens: 3 });
  }
  if (++iterations > 2000) throw new Error('Demo did not converge.');
  const report = engine.step();
  reports.push(report);
  if (!json) {
    console.log(`Tick ${report.tick} | free pages ${report.memory.freePages}/${report.memory.totalPages} | cached ${report.memory.cachedPages}`);
    console.table(report.work.map(w => ({ request: w.id, phase: w.phase, position: w.from, tokens: w.tokens, 'logical → physical pages': w.blockTable.map((page, logical) => `${logical}→${page}`).join(' ') })));
    for (const event of report.events) console.log(`  ${event.id} ${event.type}: ${event.detail}`);
    console.log();
  }
}
const results = ['A', 'B', 'C'].map(id => engine.result(id));
if (json) console.log(JSON.stringify({ mode: pressure ? 'pressure' : 'normal', reports, results }, null, 2));
else {
  console.log('Results:');
  console.table(results.map(r => ({ request: r.id, status: r.status, 'prefix tokens reused': r.prefixHits, output: r.output.join(', ') })));
  console.log('Cached pages may remain after requests finish; their activeRefs are zero.');
}
