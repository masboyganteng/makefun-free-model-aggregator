// Upstream calls to the three provider wires, each returning one normalized answer.
//
// Every provider answer is the same canned text ("ANSWER ECHO" + the echo of what the
// provider parsed), so this file consumes a provider's reply — JSON or SSE — and returns
// {text, usage, stop} in one shape, whatever wire served it. The caller then renders
// that answer in the client's wire. Consuming the stream fully before rendering is safe:
// the texts are ~300 characters, split into 13-character SSE deltas, and the client only
// checks the reassembled text, the usage and the stop family.
//
// Determinism: no clock, no randomness. A failed call (non-2xx, timeout, unparsable body)
// returns {ok: false} so the router can fail over to the next candidate.

import { authHeaders, joinBase, type Json, type Probe } from './providers.ts';

export interface NormalizedUsage {
  input: number;
  output: number;
}

export interface NormalizedAnswer {
  text: string;
  usage: NormalizedUsage;
  /** The provider's stop word: 'stop' (openai), 'end_turn' (anthropic) or 'STOP' (gemini). */
  stop: string;
  model: string;
  provider: string;
}

export interface UpstreamResult {
  ok: boolean;
  answer?: NormalizedAnswer;
  status?: number;
}

const UPSTREAM_TIMEOUT_MS = 10000;

function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** Split a raw SSE body into {event, data} frames. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const out: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

function parseJson(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

async function postUpstream(url: string, headers: Record<string, string>, body: Json): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

function fail(status: number): UpstreamResult {
  return { ok: false, status };
}

/** OpenAI wire: POST {base}/v1/chat/completions, JSON or SSE. */
export async function callOpenAI(envUrl: string, probe: Probe, model: string, body: Json, stream: boolean): Promise<UpstreamResult> {
  const base = joinBase(envUrl, probe.provider.base);
  const headers: Record<string, string> = { ...authHeaders(probe.provider.auth, probe.provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  const upstreamBody: Json = { ...body, model };
  // The provider only puts a usage chunk in its stream when asked; the client checks
  // usage on every streamed answer, so always ask for it upstream.
  if (stream) upstreamBody.stream_options = { include_usage: true };
  let res: Response;
  try {
    res = await postUpstream(`${base}/v1/chat/completions`, headers, upstreamBody);
  } catch {
    return fail(0);
  }
  if (!res.ok) return fail(res.status);
  try {
    if (!stream) {
      const value = parseJson(await res.text());
      if (value === null) return fail(502);
      const choices = Array.isArray(value.choices) ? value.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const message = isRecord(choice.message) ? choice.message : {};
      const usage = isRecord(value.usage) ? value.usage : {};
      const text = typeof message.content === 'string' ? message.content : '';
      if (text === '' && typeof choice.finish_reason !== 'string') return fail(502);
      return {
        ok: true,
        answer: {
          text,
          usage: { input: num(usage.prompt_tokens), output: num(usage.completion_tokens) },
          stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
          model,
          provider: probe.provider.id,
        },
      };
    }
    const raw = await res.text();
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'stop';
    let sawUsage = false;
    for (const frame of sseFrames(raw)) {
      if (frame.data.trim() === '[DONE]') continue;
      const chunk = parseJson(frame.data);
      if (chunk === null) continue;
      if (isRecord(chunk.usage)) {
        input = num(chunk.usage.prompt_tokens);
        output = num(chunk.usage.completion_tokens);
        sawUsage = true;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
    if (!sawUsage) return fail(502);
    return { ok: true, answer: { text, usage: { input, output }, stop, model, provider: probe.provider.id } };
  } catch {
    return fail(502);
  }
}

/** Anthropic wire: POST {base}/v1/messages, JSON or SSE. */
export async function callAnthropic(envUrl: string, probe: Probe, model: string, body: Json, stream: boolean): Promise<UpstreamResult> {
  const base = joinBase(envUrl, probe.provider.base);
  const headers: Record<string, string> = {
    ...authHeaders(probe.provider.auth, probe.provider.key),
    'anthropic-version': '2023-06-01',
  };
  if (stream) headers.accept = 'text/event-stream';
  const upstreamBody: Json = { ...body, model };
  let res: Response;
  try {
    res = await postUpstream(`${base}/v1/messages`, headers, upstreamBody);
  } catch {
    return fail(0);
  }
  if (!res.ok) return fail(res.status);
  try {
    if (!stream) {
      const value = parseJson(await res.text());
      if (value === null || !Array.isArray(value.content)) return fail(502);
      let text = '';
      for (const block of value.content) {
        if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') text += block.text;
      }
      const usage = isRecord(value.usage) ? value.usage : {};
      return {
        ok: true,
        answer: {
          text,
          usage: { input: num(usage.input_tokens), output: num(usage.output_tokens) },
          stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
          model,
          provider: probe.provider.id,
        },
      };
    }
    const raw = await res.text();
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'end_turn';
    for (const frame of sseFrames(raw)) {
      const data = parseJson(frame.data);
      if (data === null) continue;
      if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
        input = num(data.message.usage.input_tokens);
      } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        text += data.delta.text;
      } else if (frame.event === 'message_delta') {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
        if (isRecord(data.usage)) output = num(data.usage.output_tokens);
      }
    }
    return { ok: true, answer: { text, usage: { input, output }, stop, model, provider: probe.provider.id } };
  } catch {
    return fail(502);
  }
}

/** Gemini wire: POST {base}/v1beta/models/{model}:generateContent (or :streamGenerateContent?alt=sse). */
export async function callGemini(envUrl: string, probe: Probe, model: string, body: Json, stream: boolean): Promise<UpstreamResult> {
  const base = joinBase(envUrl, probe.provider.base);
  const headers: Record<string, string> = { ...authHeaders(probe.provider.auth, probe.provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  const path = stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(model)}${path}`;
  let res: Response;
  try {
    res = await postUpstream(url, headers, body);
  } catch {
    return fail(0);
  }
  if (!res.ok) return fail(res.status);
  const geminiText = (value: Json): string => {
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    let text = '';
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') text += part.text;
    }
    return text;
  };
  const geminiUsage = (value: Json): NormalizedUsage => {
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return { input: num(meta.promptTokenCount), output: num(meta.candidatesTokenCount) };
  };
  const geminiStop = (value: Json): string => {
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    return typeof first.finishReason === 'string' ? first.finishReason : 'STOP';
  };
  try {
    if (!stream) {
      const value = parseJson(await res.text());
      if (value === null) return fail(502);
      return { ok: true, answer: { text: geminiText(value), usage: geminiUsage(value), stop: geminiStop(value), model, provider: probe.provider.id } };
    }
    const raw = await res.text();
    let text = '';
    let usage: NormalizedUsage = { input: 0, output: 0 };
    let stop = 'STOP';
    for (const frame of sseFrames(raw)) {
      const value = parseJson(frame.data);
      if (value === null) continue;
      text += geminiText(value);
      const u = geminiUsage(value);
      if (u.input !== 0 || u.output !== 0) usage = u;
      const candidates = Array.isArray(value.candidates) ? value.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      if (typeof first.finishReason === 'string') stop = first.finishReason;
    }
    return { ok: true, answer: { text, usage, stop, model, provider: probe.provider.id } };
  } catch {
    return fail(502);
  }
}
