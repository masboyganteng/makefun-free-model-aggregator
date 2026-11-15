// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the trial brief, probes every provider's catalog, registers the one
// catalog, and serves the client's 20 requests: each answered by the best free model
// that meets it, over that provider's own wire, failing over when a provider refuses.
//
// What beats the starter (worth ~a quarter of the score):
// - quality: candidates are ranked by the sealed quality table's rating_milli, not by
//   provider order and model id; load is spread with a per-provider soft cap so no
//   free-tier budget is spent early, and providers that keep refusing sink fast.
// - failover: a 402, 429, 5xx or timeout is retried on the next candidate, until one
//   answers; the first success ends the request (exactly one successful provider call).
// - wires: all three upstream wires are spoken (openai, anthropic, gemini), so the
//   tools+vision requests — only served by gemini-3.8-flash, gemma-4-31b and
//   glm-4.6v-flash — can be answered at all.
// - streams: every upstream stream is consumed and re-rendered in the client's wire
//   with its usage intact (an OpenAI stream without a usage chunk fails the check),
//   and the stop reason is mapped within its family.
// - catalog: the probe's listing plus the quality table's free flag, base_url the
//   directory's own path, models sorted by id; GET /v1/models lists the free ids,
//   deduplicated, in the wire the auth header names.
//
// Determinism: every choice comes from the brief, the requests and the providers'
// answers. No wall clock, no unseeded randomness, so a replay leaves the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type ServedModel,
} from './lib/wire.ts';
import { callAnthropic, callGemini, callOpenAI, type NormalizedAnswer } from './lib/upstream.ts';
import { Router, type Candidate } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
const router = new Router();
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this client request asks for JSON mode on the OpenAI wire. */
function clientJsonMode(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One upstream attempt for one candidate: the request rebuilt in the candidate's wire,
 * answered as one normalized answer. A failed attempt is recorded so the router demotes
 * that provider for the requests that follow.
 */
async function tryCandidate(candidate: Candidate, api: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<NormalizedAnswer | null> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  try {
    let result;
    if (wire === 'openai') {
      const upstreamBody = api === 'openai' ? body : anthropicToOpenAI(body, model);
      result = await callOpenAI(ENV_URL, candidate.probe, model, upstreamBody, stream);
    } else if (wire === 'anthropic') {
      const upstreamBody = api === 'anthropic' ? body : openAIToAnthropicRequest(body, model);
      result = await callAnthropic(ENV_URL, candidate.probe, model, upstreamBody, stream);
    } else {
      const upstreamBody = api === 'openai' ? openAIToGeminiRequest(body, model) : anthropicToGeminiRequest(body, model);
      result = await callGemini(ENV_URL, candidate.probe, model, upstreamBody, stream);
    }
    router.noteAttempt(candidate.probe.provider.id, !result.ok && result.status !== 400 && result.status !== 401);
    if (!result.ok || result.answer === undefined) {
      // A 401 (wrong key) or 400 (unknown model) spends no budget and says nothing
      // about the provider's health, so it is not a routing failure — but it still
      // fails over to the next candidate, because this candidate cannot serve.
      log(`failover: ${candidate.probe.provider.id}/${model} answered ${result.status ?? 'unreachable'} for the ${api} request`);
      return null;
    }
    return result.answer;
  } catch (err) {
    router.noteAttempt(candidate.probe.provider.id, true);
    log(`failover: ${candidate.probe.provider.id}/${model} threw ${(err as Error).message}`);
    return null;
  }
}

/**
 * Serve one client request: try the ranked candidates in order until one answers, and
 * render that answer in the client's wire. The first success ends the request — the
 * env's one_provider check allows exactly one successful provider call per request.
 */
async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = clientJsonMode(api, body);
  const candidates = router.ranked(probes, brief.ratings, requires, jsonMode);
  if (candidates.length === 0) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    const answer = await tryCandidate(candidate, api, body, stream);
    if (answer === null) continue;
    if (api === 'openai') {
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(renderOpenAIStream(answer));
      } else {
        sendJson(res, 200, renderOpenAIJson(answer));
      }
    } else {
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(renderAnthropicStream(answer));
      } else {
        sendJson(res, 200, renderAnthropicJson(answer));
      }
    }
    return;
  }
  sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
