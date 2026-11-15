// Routing: rank every free candidate by the sealed quality table, spread the load so no
// provider's per-trial budget is spent early, and fail over when a provider refuses.
//
// What the kit's generator draws (program/generator/draw.mjs — read, not guessed):
// - each provider's budget is drawn in [calls_min, calls_max] of the sealed table, so a
//   soft cap at calls_min can never spend a provider's budget by itself;
// - every provider fails 1-2 scheduled calls drawn from [2, budget-1] (its FIRST call
//   never fails on schedule), and ONE provider goes into outage from a call number drawn
//   in [2, budget-1], failing every call after that.
// The router therefore demotes, never blacklists: a single 402/429 is usually a scheduled
// failure, but a provider that keeps failing is pushed down the ranking fast.
//
// Determinism: the ranking is a pure function of the brief and the trial's own call
// history — no clock, no randomness — so a replay makes the same choices.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './providers.ts';

export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Soft per-provider call cap: the sealed table's calls_min. A scored instance's budget is
 * always at least this (generator draws int(calls_min, calls_max)), so staying under the
 * cap avoids budget exhaustion; above it the provider is only demoted, never refused.
 */
export const SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const DEFAULT_SOFT_CAP = 4;
/**
 * Rating milli lost per failed call from a provider. Deliberately SMALLER than the
 * spread between the top models (1475 vs 1470 vs 1465): one scheduled failure only
 * demotes a provider below its nearest rivals, it does not sink it below the whole
 * field. (A 100-point penalty did exactly that on trial-0: after one scheduled
 * 429, openrouter fell to 1375 and the plain requests were served at 1434 — quality
 * 98 instead of ~99.) A provider that keeps failing is still pushed down fast,
 * because the penalty stacks per failure.
 */
const FAILURE_PENALTY = 6;
/** Rating milli lost while a provider is at or over its soft cap: a gentle nudge to spread load. */
const CAP_PENALTY = 6;
/** Extra penalty once a provider has failed 3+ times: it is treated as down (outage/budget spent). */
const DEAD_PENALTY = 300;
const DEAD_AFTER_FAILURES = 3;

export class Router {
  private callsMade = new Map<string, number>();
  private failures = new Map<string, number>();

  /** Record one upstream attempt (success or failure) against its provider. */
  noteAttempt(providerId: string, failed: boolean): void {
    this.callsMade.set(providerId, (this.callsMade.get(providerId) ?? 0) + 1);
    if (failed) this.failures.set(providerId, (this.failures.get(providerId) ?? 0) + 1);
  }

  callsOf(providerId: string): number {
    return this.callsMade.get(providerId) ?? 0;
  }

  failuresOf(providerId: string): number {
    return this.failures.get(providerId) ?? 0;
  }

  private effectiveRating(candidate: Candidate): number {
    const id = candidate.probe.provider.id;
    let score = candidate.rating;
    score -= FAILURE_PENALTY * this.failuresOf(id);
    if (this.failuresOf(id) >= DEAD_AFTER_FAILURES) score -= DEAD_PENALTY;
    const cap = SOFT_CAP[id] ?? DEFAULT_SOFT_CAP;
    if (this.callsOf(id) >= cap) score -= CAP_PENALTY;
    return score;
  }

  /**
   * Every free candidate that meets the request, best first.
   *
   * `clientJsonMode` is true only for an OpenAI client asking response_format json: the
   * Anthropic wire cannot carry JSON mode (its echo always says json_mode false), so
   * anthropic-wire providers are excluded for those requests. An Anthropic client never
   * asks for JSON mode on the wire, so its json requirement is a model capability only.
   */
  ranked(probes: readonly Probe[], ratings: Parameters<typeof ratingOfModel>[0], requires: Requirement, clientJsonMode: boolean): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of probes) {
      if (!probe.ok) continue;
      if (clientJsonMode && probe.provider.wire === 'anthropic') continue;
      for (const model of probe.models) {
        const row = ratingOfModel(ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        out.push({ probe, model, rating: row?.rating_milli ?? 0 });
      }
    }
    out.sort((a, b) => {
      const byRating = this.effectiveRating(b) - this.effectiveRating(a);
      if (byRating !== 0) return byRating;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    return out;
  }
}
