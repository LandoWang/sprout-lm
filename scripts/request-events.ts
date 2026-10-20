import { readRequestTrace } from '../src/request-trace.ts';

const [directory, requestId] = process.argv.slice(2);
if (!directory || !requestId) throw new Error('Usage: npm run trace:request -- RUN_DIRECTORY REQUEST_ID');
console.log(JSON.stringify(await readRequestTrace(directory, requestId), null, 2));
