import { once } from 'node:events';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { GenerationError, GenerationService, type GenerationEvent } from './service.ts';
import { byteTokenizer, type Tokenizer } from './tokenizer.ts';
import type { ChatTemplate, ChatMessage } from './text-plugins.ts';

class HttpError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(body));
}

async function readBody(request: IncomingMessage, limit: number): Promise<unknown> {
  if (request.headers['content-type']?.split(';')[0].trim().toLowerCase() !== 'application/json') {
    request.resume();
    throw new HttpError(415, 'unsupported_media_type', 'Use Content-Type: application/json.');
  }
  const chunks: Buffer[] = [];
  let bytes = 0;
  // Keep the socket alive on a rejected body so the caller receives the HTTP error.
  try {
    for await (const chunk of request.iterator({ destroyOnReturn: false })) {
      bytes += chunk.length;
      if (bytes > limit) throw new HttpError(413, 'body_too_large', `Body exceeds ${limit} bytes.`);
      chunks.push(chunk);
    }
  } finally {
    if (!request.readableEnded) request.resume();
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw new HttpError(400, 'invalid_request', 'Body must be valid JSON.');
  }
}

function parseRequest(body: unknown, tokenizer: Tokenizer, template?: ChatTemplate) {
  const invalid = (message: string): never => { throw new HttpError(400, 'invalid_request', message); };
  if (!body || typeof body !== 'object' || Array.isArray(body)) return invalid('Body must be an object.');
  const input = body as Record<string, unknown>;
  for (const key of Object.keys(input)) {
    if (!['prompt', 'messages', 'maxNewTokens', 'stream'].includes(key)) return invalid(`Unsupported field: ${key}.`);
  }
  if ((input.prompt !== undefined) === (input.messages !== undefined)) return invalid('Provide exactly one of prompt or messages.');
  if (input.messages === undefined && typeof input.prompt !== 'string' && !Array.isArray(input.prompt)) {
    return invalid('prompt must be a string or an array of token IDs.');
  }
  if (input.stream !== undefined && typeof input.stream !== 'boolean') return invalid('stream must be a boolean.');
  const maxNewTokens = input.maxNewTokens === undefined ? 16 : input.maxNewTokens;
  if (typeof maxNewTokens !== 'number' || !Number.isSafeInteger(maxNewTokens) || maxNewTokens <= 0) {
    return invalid('maxNewTokens must be a positive integer.');
  }
  // Engine validates numeric token IDs against the model vocabulary and KV capacity.
  let prompt: number[];
  if (input.messages !== undefined) {
    if (!template) return invalid('This server has no chat template plugin. Use prompt instead.');
    if (!Array.isArray(input.messages) || !input.messages.length || input.messages.some(m =>
      !m || typeof m !== 'object' || Array.isArray(m) || !['system', 'user', 'assistant'].includes(m.role)
      || typeof m.content !== 'string' || Object.keys(m).some(k => !['role', 'content'].includes(k)))) {
      return invalid('messages must contain text-only system, user or assistant messages.');
    }
    try { prompt = tokenizer.encode(template.render(input.messages as ChatMessage[])); }
    catch (error) { return invalid(error instanceof Error ? error.message : 'Failed to format messages.'); }
  } else {
    try { prompt = typeof input.prompt === 'string' ? tokenizer.encode(input.prompt) : input.prompt as number[]; }
    catch (error) { return invalid(error instanceof Error ? error.message : 'Failed to encode prompt.'); }
  }
  return { prompt, maxNewTokens, stream: input.stream === true };
}

function errorResponse(error: unknown) {
  if (error instanceof HttpError) return { status: error.status, code: error.code, message: error.message };
  if (error instanceof GenerationError) {
    const status = error.code === 'invalid_request' ? 400
      : error.code === 'overloaded' ? 429 : error.code === 'closed' ? 503 : 500;
    return { status, code: error.code, message: error.message };
  }
  return { status: 500, code: 'internal_error', message: 'Request failed.' };
}

function sse(event: { type: string; [key: string]: unknown }): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}

async function writeEvent(response: ServerResponse, event: { type: string; [key: string]: unknown }, signal: AbortSignal): Promise<void> {
  signal.throwIfAborted();
  if (!response.write(sse(event))) await once(response, 'drain', { signal });
}

export interface ApiOptions {
  service?: GenerationService;
  tokenizer?: Tokenizer;
  template?: ChatTemplate;
  maxBodyBytes?: number;
}

/** Native SproutLM protocol. Protocol adapters call the service, never Engine.step(). */
export function createApiServer({ service = new GenerationService(), tokenizer = byteTokenizer, template, maxBodyBytes = 65536 }: ApiOptions = {}) {
  if (!Number.isSafeInteger(maxBodyBytes) || maxBodyBytes <= 0) throw new Error('maxBodyBytes must be a positive integer.');
  const server = createServer((request, response) => {
    void handle(request, response);
  });
  server.on('close', () => service.close());
  return server;

  async function handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    request.once('aborted', abort);
    response.once('close', abort);
    response.once('error', abort);
    let id: string | undefined;
    try {
      const path = new URL(request.url ?? '/', 'http://localhost').pathname;
      if (request.method === 'GET' && path === '/health') {
        json(response, service.isClosed ? 503 : 200, {
          status: service.isClosed ? 'closed' : 'ok', pendingRequests: service.pendingCount,
        });
        return;
      }
      if (request.method === 'DELETE' && path.startsWith('/requests/')) {
        const requestId = path.slice('/requests/'.length);
        if (!service.cancel(requestId)) throw new HttpError(404, 'not_found', 'No active request with that ID.');
        json(response, 200, { id: requestId, cancelled: true });
        return;
      }
      if (request.method !== 'POST' || path !== '/generate') {
        request.resume();
        throw new HttpError(404, 'not_found', 'Use POST /generate, DELETE /requests/:id, or GET /health.');
      }
      const input = parseRequest(await readBody(request, maxBodyBytes), tokenizer, template);
      controller.signal.throwIfAborted();
      const events = service.generate(input, { signal: controller.signal });
      id = events.id;
      response.setHeader('x-request-id', id);
      const decoder = tokenizer.decoder();
      try {
        if (input.stream) {
          response.writeHead(200, {
            'content-type': 'text/event-stream; charset=utf-8',
            'cache-control': 'no-cache', 'x-accel-buffering': 'no',
          });
          response.flushHeaders();
          for await (const event of events) {
            const textDelta = event.type === 'token' ? decoder.push(event.token)
              : event.type === 'queued' ? '' : decoder.finish();
            await writeEvent(response, { ...event, textDelta }, controller.signal);
          }
          response.end();
        } else {
          const output: number[] = [];
          let text = '';
          let terminal: GenerationEvent | undefined;
          for await (const event of events) {
            if (event.type === 'token') {
              output.push(event.token);
              text += decoder.push(event.token);
            } else if (event.type !== 'queued') terminal = event;
          }
          text += decoder.finish();
          if (terminal?.type === 'error') throw new GenerationError(terminal.code, terminal.message);
          if (terminal?.type === 'cancelled') throw new HttpError(409, 'cancelled', 'Request was cancelled.');
          if (terminal?.type !== 'finished') throw new Error('Generation ended without a terminal event.');
          json(response, 200, { id, output, text, finishReason: terminal.finishReason, usage: terminal.usage });
        }
      } finally {
        await events.return?.();
      }
    } catch (error) {
      if (id) service.cancel(id);
      if (controller.signal.aborted || response.destroyed) return;
      const { status, ...detail } = errorResponse(error);
      if (response.headersSent) response.end(sse({ type: 'error', ...detail, id }));
      else json(response, status, { ...(id ? { id } : {}), error: detail });
    } finally {
      request.removeListener('aborted', abort);
      response.removeListener('close', abort);
      response.removeListener('error', abort);
    }
  }
}
