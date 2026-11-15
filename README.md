# makefun-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/masboyganteng/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/masboyganteng/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers, in JSON and in streaming.

Built for [free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01) ("The best free AI model aggregator") on make.fun, as @masboykece.

## Run it

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve a trial (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=pv30jx14&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

## What it does

- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): the ten providers, the sealed quality table, and the 20-request workload manifest.
- Probes every provider's catalog route in its own wire (OpenAI, Anthropic, Gemini) and registers the one catalog with the env.
- Serves `POST /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models` (OpenAI shape for `Authorization: Bearer`, Anthropic shape for `x-api-key`).
- Routes each request to the best free model that meets its requirements, ranked by the quality table's rating, with load spread across providers (soft per-provider call caps) and failover to the next candidate when a provider refuses (402/429/5xx).
- Speaks all three upstream wires, so the tools+vision requests — only served by Gemini- and Anthropic-wire models — are answered, and JSON-mode requests never go to a wire that cannot carry JSON mode.
- `node main.ts --list` prints the catalog it registers, byte for byte.

Node 22+, standard library only, no install step. Deterministic: no wall clock, no unseeded randomness — a replay makes the same choices.

## Layout

- `main.ts` — the server: brief, probe, register, routes, poll/exit.
- `lib/providers.ts` — brief parsing, catalog probe, catalog document.
- `lib/wire.ts` — client-wire translations and renderers.
- `lib/upstream.ts` — upstream calls on the three wires, normalized to one answer.
- `lib/router.ts` — candidate ranking, soft caps, failure penalties, failover.
- `.makefun/owner.json` — claim file naming the linked wallet.

## License

MIT — see `LICENSE`.
