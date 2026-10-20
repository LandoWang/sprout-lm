import type { EventLog } from './event-log.ts';
import type { ObservationEvent } from './observation.ts';

export type TimedEvent = ObservationEvent & { at: number };
export interface Span {
  id: string; requestId: string; lane: string; kind: string; label: string;
  start: number; end: number; planId?: string; incomplete: boolean; data?: Record<string, unknown>;
}
export interface Plan {
  id: string; requestId: string; phase: string; worker: string;
  sent?: number; started?: number; completed?: number; received?: number;
  dispatchMs: number | null; executionMs: number | null; returnMs: number | null;
  cudaEventMs?: number; incomplete: boolean; failed: boolean; discarded: boolean;
  data: Record<string, unknown>;
}
const delta = (a: number | undefined, b: number | undefined): number | null =>
  a !== undefined && b !== undefined && b >= a ? b-a : null;
const numeric = (v: unknown): number | undefined => typeof v === 'number' && Number.isFinite(v) ? v : undefined;
function union(intervals: Array<[number, number]>): number {
  let end = -Infinity, total = 0;
  for (const [a,b] of intervals.sort((x,y) => x[0]-y[0])) {
    if (b > end) { total += b - Math.max(a,end); end = b; }
  }
  return total;
}
function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a,b) => a-b);
  return sorted[Math.max(0, Math.ceil(sorted.length * p)-1)];
}

export function analyzeRun(log: EventLog) {
  const origin = log.events.length ? BigInt(log.events[0].timeNs) : 0n;
  const events: TimedEvent[] = log.events.map(e => ({ ...e, at: Number(BigInt(e.timeNs)-origin)/1e6 }));
  const byRequest = new Map<string, TimedEvent[]>();
  for (const e of events) if (e.requestId) {
    if (!byRequest.has(e.requestId)) byRequest.set(e.requestId, []);
    byRequest.get(e.requestId)!.push(e);
  }
  const spans: Span[] = [], plans: Plan[] = [];
  let invalidPairs = 0;
  function span(requestId: string, lane: string, kind: string, label: string,
    start: number | undefined, end: number | undefined, extra: Partial<Span> = {}) {
    if (start === undefined || end === undefined) return;
    if (end < start) { invalidPairs++; return; }
    spans.push({ id: `span-${spans.length}`, requestId, lane, kind, label, start, end, incomplete: false, ...extra });
  }
  const requests = [...byRequest].map(([id, es]) => {
    const spanStart = spans.length;
    const first = (name: string) => es.find(e => e.name === name);
    const time = (name: string) => first(name)?.at;
    const observedStart = es[0].at, observedEnd = es.at(-1)!.at;
    const received = time('request.received');
    const terminal = es.find(e => ['generation.finished','request.failed','request.cancelled','http.request_failed'].includes(e.name));
    const httpFinish = first('http.response_finished'), httpClose = first('http.response_closed');
    const failed = es.some(e => ['request.failed','http.request_failed'].includes(e.name));
    const status = failed ? 'failed' : first('request.cancelled') ? 'cancelled'
      : first('generation.finished') ? 'finished' : httpClose ? 'closed' : 'incomplete';
    const requestPlans = new Map<string, TimedEvent[]>();
    for (const e of es) if (e.planId && e.name.startsWith('plan.')) {
      if (!requestPlans.has(e.planId)) requestPlans.set(e.planId, []);
      requestPlans.get(e.planId)!.push(e);
    }
    const ownPlans: Plan[] = [];
    for (const [planId, pe] of requestPlans) {
      const sent = pe.find(e => e.name === 'plan.sent');
      const started = pe.find(e => e.name === 'plan.started');
      const completed = pe.find(e => ['plan.completed','plan.execution_failed'].includes(e.name));
      const acknowledged = pe.find(e => ['plan.completion_received','plan.failed'].includes(e.name));
      const worker = String(sent?.data?.worker ?? started?.source ?? completed?.source ?? 'unknown');
      const phase = String(sent?.data?.phase ?? acknowledged?.data?.phase
        ?? (started?.data?.operation === 'transfer' ? 'transfer' : worker));
      const p: Plan = { id: planId, requestId: id, phase, worker, sent: sent?.at, started: started?.at,
        completed: completed?.at, received: acknowledged?.at,
        dispatchMs: delta(sent?.at, started?.at), executionMs: delta(started?.at, completed?.at),
        returnMs: delta(completed?.at, acknowledged?.at),
        cudaEventMs: numeric(completed?.data?.cudaEventMs ?? acknowledged?.data?.cudaEventMs),
        incomplete: !sent || !started || !completed || !acknowledged,
        failed: pe.some(e => e.name.endsWith('failed')), discarded: acknowledged?.data?.discarded === true,
        data: sent?.data ?? {} };
      const extra = { planId, data: { ...p.data, cudaEventMs: p.cudaEventMs, discarded: p.discarded, failed: p.failed } };
      span(id, 'dispatch', 'dispatch', `${planId} · 发出 → 开始`, p.sent, p.started, extra);
      span(id, worker, phase, `${planId} · ${phase}`, p.started, p.completed, extra);
      span(id, 'return', 'return', `${planId} · 完成 → 回执`, p.completed, p.received, extra);
      if (started && !completed) span(id, worker, phase, `${planId} · 未见完成`, started.at, observedEnd, { ...extra, incomplete: true });
      ownPlans.push(p); plans.push(p);
    }
    for (const [startName, endName, label] of [
      ['request.received','http.body_read','读取请求'],
      ['text.template_started','text.template_completed','对话模板'],
      ['text.encode_started','text.encoded','Tokenizer'],
      ['scheduler.queued','scheduler.admitted','准入队列'],
    ]) span(id, startName.startsWith('scheduler') ? 'admission' : 'api',
      startName.startsWith('scheduler') ? 'wait' : 'api', label, time(startName), time(endName));

    // Match repeated intervals by process + reason; never bridge unrelated waits.
    const open = new Map<string, TimedEvent>();
    for (const e of es) {
      const wait = e.name === 'request.waiting' || e.name === 'request.wait_ended';
      const backpressure = e.name === 'http.backpressure_started' || e.name === 'http.backpressure_ended';
      if (!wait && !backpressure) continue;
      const key = JSON.stringify([e.source,e.pid,wait ? 'wait' : 'backpressure', wait ? e.data?.reason : '']);
      const isStart = e.name === 'request.waiting' || e.name === 'http.backpressure_started';
      if (isStart) {
        const old = open.get(key);
        if (old) span(id, wait ? 'wait' : 'http', wait ? 'wait' : 'backpressure', String(old.data?.reason ?? 'HTTP 背压'), old.at, e.at, { incomplete: true });
        open.set(key,e);
      } else {
        const start = open.get(key);
        if (start) span(id, wait ? 'wait' : 'http', wait ? 'wait' : 'backpressure', String(start.data?.reason ?? 'HTTP 背压'), start.at, e.at);
        else invalidPairs++;
        open.delete(key);
      }
    }
    for (const e of open.values()) span(id, e.name === 'request.waiting' ? 'wait' : 'http',
      e.name === 'request.waiting' ? 'wait' : 'backpressure', String(e.data?.reason ?? 'HTTP 背压'), e.at, observedEnd, { incomplete: true });

    const tokens = new Map<number, { index: number; ready?: number; published?: number; written?: number }>();
    for (const e of es) {
      const field = e.name === 'token.ready' ? 'ready' : e.name === 'token.published' ? 'published' : e.name === 'http.token_written' ? 'written' : undefined;
      const index = numeric(e.data?.index);
      if (!field || index === undefined || !Number.isSafeInteger(index) || index < 0) continue;
      if (!tokens.has(index)) tokens.set(index,{ index });
      tokens.get(index)![field] ??= e.at;
    }
    const tokenList = [...tokens.values()].sort((a,b) => a.index-b.index);
    const token0 = tokens.get(0);
    const itl: Array<{ index: number; at: number; ms: number }> = [];
    const writtenItl: typeof itl = [];
    for (const t of tokenList) {
      const prev = tokens.get(t.index-1);
      const ms = delta(prev?.published,t.published), writtenMs = delta(prev?.written,t.written);
      if (ms !== null) itl.push({ index: t.index, at: t.published!, ms });
      if (writtenMs !== null) writtenItl.push({ index: t.index, at: t.written!, ms: writtenMs });
    }
    const ownSpans = spans.slice(spanStart);
    const sums = (kind: string) => union(ownSpans.filter(s => s.kind === kind && !s.incomplete).map(s => [s.start,s.end]));
    const completeWork = ownSpans.filter(s => ['prefill','decode'].includes(s.lane) && !s.incomplete);
    const breakdown = ['prefill','decode','transfer','dispatch','return','wait','backpressure','api'].map(kind => ({ kind, ms: sums(kind) }));
    const firstToken = token0?.published;
    const ttftSegments = received === undefined || firstToken === undefined ? [] : breakdown.map(row => ({
      kind: row.kind, ms: union(ownSpans.filter(s => s.kind === row.kind && !s.incomplete && s.start < firstToken && s.end > received)
        .map(s => [Math.max(s.start,received), Math.min(s.end,firstToken)])),
    }));
    return { id, status, observedStart, observedEnd, received, wallTimeMs: es[0].wallTimeMs,
      eventCount: es.length, planCount: ownPlans.length, incompletePlans: ownPlans.filter(p => p.incomplete).length,
      promptTokens: es.map(e => numeric(e.data?.promptTokens)).find(n => n !== undefined),
      completionTokens: numeric(terminal?.data?.completionTokens), observedTokens: tokenList.filter(t => t.published !== undefined).length,
      ttftMs: delta(received,token0?.published), httpTtftMs: delta(received,token0?.written),
      generationMs: delta(received,terminal?.at), responseMs: delta(received,httpFinish?.at),
      firstToSecondMs: delta(token0?.published,tokens.get(1)?.published),
      itlP50: percentile(itl.map(t => t.ms),0.5), itlP95: percentile(itl.map(t => t.ms),0.95),
      itlMax: itl.length ? Math.max(...itl.map(t => t.ms)) : null,
      workerMs: union(completeWork.map(s => [s.start,s.end])),
      kvCudaMs: ownPlans.filter(p => p.phase === 'transfer').reduce((s,p) => s+(p.cudaEventMs ?? 0),0),
      hasKvCuda: ownPlans.some(p => p.phase === 'transfer' && p.cudaEventMs !== undefined),
      maxBufferedEvents: Math.max(0,...es.map(e => numeric(e.data?.bufferedEvents) ?? 0)),
      maxWritableBytes: Math.max(0,...es.map(e => numeric(e.data?.writableBytes) ?? 0)),
      resourcesReleased: !!first('request.resources_released'), missingStart: received === undefined,
      terminalSeen: !!terminal, tokens: tokenList, itl, writtenItl, breakdown, ttftSegments,
    };
  });
  // Gaps are between recorded Worker executions across ALL requests. They are
  // neither GPU idle measurements nor attributable to a particular lock.
  const gaps: Array<{ worker: string; start: number; end: number; ms: number; nextRequestId: string; nextPlanId: string }> = [];
  for (const worker of ['prefill','decode']) {
    const ps = plans.filter(p => p.worker === worker && p.started !== undefined).sort((a,b) => a.started!-b.started!);
    let priorEnd: number | undefined;
    for (const p of ps) {
      if (priorEnd !== undefined && p.started! > priorEnd) gaps.push({ worker, start: priorEnd,
        end: p.started!, ms: p.started!-priorEnd, nextRequestId: p.requestId, nextPlanId: p.id });
      priorEnd = p.completed === undefined ? undefined : Math.max(priorEnd ?? p.completed,p.completed);
    }
  }
  return { runId: log.runId, originNs: origin.toString(), end: events.at(-1)?.at ?? 0,
    events, requests, spans, plans, gaps: gaps.sort((a,b) => b.ms-a.ms),
    diagnostics: { ...log.diagnostics, invalidPairs },
  };
}
export type RunAnalysis = ReturnType<typeof analyzeRun>;
