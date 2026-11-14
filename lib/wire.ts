// The client wires and the three upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This entry forwards each request to
// the best free model on any of the ten providers, in that provider's own wire, and translates the answer
// back into the wire the client used:
//
//   - openai client -> openai provider: the body passes through (with the routed model and
//     stream_options.include_usage so a streamed answer carries its usage chunk);
//   - anthropic client -> openai provider: anthropicToOpenAI / openAIToAnthropic /
//     anthropicEventsFromOpenAI (the starter's translators);
//   - either client -> anthropic provider: the canonical request rebuilt as a Messages body;
//   - either client -> gemini provider: the canonical request rebuilt as a generateContent body.
//
// A request is first reduced to one canonical shape (system text, user text, one image, tools,
// max_tokens, stops, JSON mode), so every upstream builder starts from the same fields and the provider's
// echo — system, user, tools, max_tokens, stops, image, JSON mode — matches what the client sent on every
// wire. An answer is reduced the same way (its whole text, its stop family, its usage counts) and rendered
// in the client's wire. The provider's answer text ("ANSWER ECHO" plus its echo JSON) is always returned
// whole: a field rewritten or dropped loses the request.
//
// One wire cannot carry JSON mode: an Anthropic-wire provider always echoes json_mode false, so a request
// that asks for JSON mode is never routed to one (main.ts excludes them from its candidates). Everything
// here is a pure function of its arguments: no clock and no randomness, so the same body and the same
// provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// The canonical request: one shape for either client wire, and the builders
// that spell it in each provider wire. The builders keep, in order, the fields
// the provider echoes back (system, user text, image, tools, max_tokens, stops,
// JSON mode), so the echo matches the client's request on every wire.
// ---------------------------------------------------------------------------

/** One image in a request: base64 bytes (with their media type) or a URL. */
export type CanonicalImage =
  | { kind: 'base64'; media: string; data: string }
  | { kind: 'url'; url: string };

/** One tool definition, in the shape every wire's builder can spell. */
export interface CanonicalTool {
  name: string;
  description: string;
  parameters: Json;
}

/** A client request reduced to the fields every wire carries. */
export interface CanonicalRequest {
  system: string;
  userText: string;
  image: CanonicalImage | null;
  tools: CanonicalTool[];
  toolChoice: 'auto' | 'none' | { name: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function stringListOf(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The image of an OpenAI user message's content parts, if it holds one. */
function openAIImageOf(content: unknown): CanonicalImage | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
    const url = block.image_url.url;
    if (typeof url !== 'string' || url === '') continue;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { kind: 'base64', media: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
    return { kind: 'url', url };
  }
  return null;
}

/** The image of an Anthropic user message's content blocks, if it holds one. */
function anthropicImageOf(content: unknown): CanonicalImage | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { kind: 'base64', media: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', url: source.url };
  }
  return null;
}

function openAIToolsOf(body: Json): CanonicalTool[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

function anthropicToolsOf(body: Json): CanonicalTool[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

function openAIToolChoiceOf(body: Json): CanonicalRequest['toolChoice'] {
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none') return choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') return 'auto';
    if (choice.type === 'none') return 'none';
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') return { name: choice.function.name };
  }
  return null;
}

function anthropicToolChoiceOf(body: Json): CanonicalRequest['toolChoice'] {
  const choice = body.tool_choice;
  if (!isRecord(choice)) return null;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'none') return 'none';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { name: choice.name };
  return null;
}

/** An OpenAI Chat Completions request as the canonical shape. */
export function canonicalOfOpenAI(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => textOf(m.content))
    .filter((t) => t !== '')
    .join('\n');
  let userText = '';
  let image: CanonicalImage | null = null;
  for (const m of messages) {
    if (m.role !== 'user') continue;
    userText = textOf(m.content);
    image = openAIImageOf(m.content) ?? image;
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system,
    userText,
    image,
    tools: openAIToolsOf(body),
    toolChoice: openAIToolChoiceOf(body),
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: stringListOf(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages request as the canonical shape. */
export function canonicalOfAnthropic(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let image: CanonicalImage | null = null;
  for (const m of messages) {
    if (m.role !== 'user') continue;
    userText = textOf(m.content);
    image = anthropicImageOf(m.content) ?? image;
  }
  return {
    system: textOf(body.system),
    userText,
    image,
    tools: anthropicToolsOf(body),
    toolChoice: anthropicToolChoiceOf(body),
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: stringListOf(body.stop_sequences),
    // The Anthropic wire carries no JSON mode: its echo is always json_mode false.
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAIUpstream(req: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  if (req.image === null) {
    messages.push({ role: 'user', content: req.userText });
  } else if (req.image.kind === 'base64') {
    messages.push({
      role: 'user',
      content: [
        { type: 'text', text: req.userText },
        { type: 'image_url', image_url: { url: `data:${req.image.media};base64,${req.image.data}` } },
      ],
    });
  } else {
    messages.push({
      role: 'user',
      content: [
        { type: 'text', text: req.userText },
        { type: 'image_url', image_url: { url: req.image.url } },
      ],
    });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (req.toolChoice === 'auto' || req.toolChoice === 'none') out.tool_choice = req.toolChoice;
  else if (req.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.stream) {
    out.stream = true;
    // Ask for the usage chunk: a streamed OpenAI answer must carry the provider's counts.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function buildAnthropicUpstream(req: CanonicalRequest, model: string): Json {
  const content: Json[] = [{ type: 'text', text: req.userText }];
  if (req.image !== null) {
    if (req.image.kind === 'base64') {
      content.push({ type: 'image', source: { type: 'base64', media_type: req.image.media, data: req.image.data } });
    } else {
      content.push({ type: 'image', source: { type: 'url', url: req.image.url } });
    }
  }
  const out: Json = { model, messages: [{ role: 'user', content }] };
  if (req.system !== '') out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'none') out.tool_choice = { type: 'none' };
  else if (req.toolChoice !== null) out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body. */
export function buildGeminiUpstream(req: CanonicalRequest): Json {
  const parts: Json[] = [{ text: req.userText }];
  if (req.image !== null) {
    if (req.image.kind === 'base64') {
      parts.push({ inlineData: { mimeType: req.image.media, data: req.image.data } });
    } else {
      parts.push({ fileData: { mimeType: 'image/png', fileUri: req.image.url } });
    }
  }
  const out: Json = { contents: [{ role: 'user', parts }] };
  if (req.system !== '') out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (req.toolChoice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  else if (req.toolChoice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (req.toolChoice !== null) out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [req.toolChoice.name] } };
  return out;
}

// ---------------------------------------------------------------------------
// The canonical answer: a provider's reply reduced to its whole text and its
// usage counts, then rendered in the client's wire. The text is always the
// provider's answer whole ("ANSWER ECHO" and its echo JSON included).
// ---------------------------------------------------------------------------

/** A provider's answer: its whole text and the usage counts it reported. */
export interface CanonicalAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  stop: string | null;
}

/**
 * The usage the env's providers compute, for when an answer carries none: 1 plus a quarter of the
 * echoed prompt's characters in, 1 plus a quarter of the answer text's characters out.
 */
export function usageOfRequest(req: CanonicalRequest, text: string): { input: number; output: number } {
  const prompt = req.system.length + req.userText.length + req.tools.map((t) => t.name).join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

/** Split a stream body into its SSE frames: each frame's event name and data lines. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/** An OpenAI Chat Completions JSON answer as the canonical shape. */
export function parseOpenAIJson(body: Json): CanonicalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage)
    ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) }
    : null;
  return { text, usage, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
}

/** An OpenAI Chat Completions SSE stream as the canonical shape. */
export function parseOpenAIStream(text: string): CanonicalAnswer {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, usage, stop };
}

/** An Anthropic Messages JSON answer as the canonical shape. */
export function parseAnthropicJson(body: Json): CanonicalAnswer {
  const text = textOf(body.content);
  const usage = isRecord(body.usage)
    ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) }
    : null;
  return { text, usage, stop: typeof body.stop_reason === 'string' ? body.stop_reason : null };
}

/** An Anthropic Messages SSE stream as the canonical shape. */
export function parseAnthropicStream(text: string): CanonicalAnswer {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 }, stop };
}

function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

function geminiUsageOf(body: Json): { input: number; output: number } | null {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  if (meta === null) return null;
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent JSON answer as the canonical shape. */
export function parseGeminiJson(body: Json): CanonicalAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  return { text: geminiTextOf(body), usage: geminiUsageOf(body), stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null };
}

/** A Gemini streamGenerateContent SSE stream as the canonical shape. */
export function parseGeminiStream(text: string): CanonicalAnswer {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const body = jsonOrNull(frame.data);
    if (body === null) continue;
    out += geminiTextOf(body);
    const u = geminiUsageOf(body);
    if (u !== null) usage = u;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
  }
  return { text: out, usage, stop };
}

/** A provider stop as an OpenAI finish_reason: the providers only ever stop cleanly. */
function openAIStopOf(stop: string | null): string {
  if (stop === 'length' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Split text the way the providers' streams do: 13 characters a chunk. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** The canonical answer as an OpenAI Chat Completions JSON object. */
export function renderOpenAIJson(answer: CanonicalAnswer, model: string, requestId: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: `chatcmpl-${requestId}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** The canonical answer as an OpenAI Chat Completions SSE stream. */
export function renderOpenAIStream(answer: CanonicalAnswer, model: string, requestId: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${requestId}`,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStopOf(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: `chatcmpl-${requestId}`,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The canonical answer as an Anthropic Messages JSON object. */
export function renderAnthropicJson(answer: CanonicalAnswer, model: string, requestId: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: `msg_${requestId}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** The canonical answer as an Anthropic Messages SSE stream. */
export function renderAnthropicStream(answer: CanonicalAnswer, model: string, requestId: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${requestId}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
