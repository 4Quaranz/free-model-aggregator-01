# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten
simulated free-tier model providers, routing each request to the best free model that can serve it,
failing over when a provider refuses, and reporting the catalog it read.

Entry for [free-model-aggregator-01](https://make.fun/c/free-model-aggregator-01) on make.fun.

## Run

```sh
node main.ts            # serve: reads $MAKEFUN_PROMPT_FILE, registers at $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog document and exit 0
```

Node 22, standard library only — nothing is installed. `main.ts` is the server; `lib/providers.ts`
reads the brief and probes the providers' catalogs; `lib/wire.ts` holds the wire translations.

## How it routes

- Candidates are every probed free model meeting the request's requirements, ranked by the quality
  table's `rating_milli`, with a small freshness penalty per call already made to that provider, so the
  load spreads over the directory instead of draining one provider's free tier and tripping its failure
  schedule.
- A request that asks for JSON mode never goes to an Anthropic-wire provider: that wire always echoes
  `json_mode` false, so the client's echo check would fail.
- A 402, 429, 5xx or unreachable provider is never the answer: the request is retried on the next
  candidate, in rank order, until one answers.
- Every choice is a function of the brief, the requests and the providers' answers — no wall clock, no
  unseeded randomness — so a replay leaves the same state export.

## The wires

The client speaks OpenAI (`POST /v1/chat/completions`) or Anthropic (`POST /v1/messages`), in JSON or
server-sent events. Each request is reduced to one canonical shape (system text, user text, one image,
tools, max_tokens, stops, JSON mode), rebuilt in the routed provider's own wire (OpenAI, Anthropic or
Gemini `generateContent`/`streamGenerateContent`), and the provider's answer is reduced the same way
(whole text, usage counts) and rendered back in the client's wire. The provider's answer text
(`ANSWER ECHO` plus its echo JSON) is always returned whole, with the provider's usage counts as that
wire reports them.

`GET /v1/models` answers in the shape of the auth header it got: `Authorization: Bearer` gets the
OpenAI model list, `x-api-key` the Anthropic one — exactly the free model ids the server registered.

## Layout

```
main.ts            the server: brief, catalog, routing, routes, register, poll
lib/providers.ts   the brief's documents, the catalog probe, the catalog builder
lib/wire.ts        the client wires, the canonical request/answer, the upstream wires
LICENSE            MIT
.makefun/owner.json  the wallet this entry belongs to
```

## License

MIT — see [LICENSE](LICENSE).
