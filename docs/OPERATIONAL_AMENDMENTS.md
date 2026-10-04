# Operational amendments and collection-engineering log

This file records post-registration **operational** changes made to the
collection pipeline after production data collection began.

These entries are intentionally separated from scientific protocol amendments.
They document engineering changes that affect how the fixed study is executed,
resumed, monitored, and protected against provider/runtime failures. They do
not silently change the preregistered scientific instrument or analysis.

Scientific settings that remain fixed unless explicitly documented as a
protocol amendment include the model checkpoints, prompts, temperature, seed,
reasoning settings, max_tokens, vignette dataset, runs per cell, hypotheses,
scoring rules, and statistical analysis plan.

---

## 2026-09-19 — Retry ceiling reduced from 6 to 3

**Change commit:** `877a4d680d78095d8a094c9bed060ed9b27b6113`  
**Affected file:** `config/protocol.yaml`  
**Field:** `rate_limits.max_retries`  
**Before:** 6 retries after the initial attempt, up to 7 attempts per cell  
**After:** 3 retries after the initial attempt, up to 4 attempts per cell

### Triggering production evidence

Scheduled production run `35422141274` collected successfully for Model A but
showed severe retry pressure for Model B.

Observed Model B outcome:

- 67 new responses written;
- 1 non-retryable HTTP 400 cell failure;
- 107 retries;
- 175/175 daily attempts consumed;
- collection stopped at the per-model daily safety cap;
- 107 planned cells within that dispatch window never received a first attempt
  because retries consumed the remaining daily attempt budget.

The failed cell was
`B V002 female_south_asian run=3`. The provider returned an HTTP 400
`json_validate_failed` response indicating that the maximum completion-token
budget had been reached before a valid JSON document was produced.

The run log did **not** contain enough per-attempt HTTP telemetry to prove the
cause of the 107 retryable failures. Therefore no claim is made that they were
specifically 429, TPM throttling, provider capacity, transport errors, or any
other single mechanism.

### Rationale

Retries count against the same fixed daily request-attempt cap as first attempts.
With `max_retries: 6`, one difficult cell could consume as many as seven daily
attempts. Under a 175-attempt/model/day safety cap, that can prevent many
previously unattempted cells from being reached.

Reducing the retry ceiling to 3 limits the worst-case cost of a persistently
failing cell to four total attempts while still allowing multiple opportunities
for genuine transient failures to recover.

The change is applied symmetrically to both model slots. The available evidence
showed heavy retry pressure only for Model B on this run, but there was not
enough evidence to justify a model-specific retry policy.

### Scientific impact assessment

This is an **operational collection-engineering change**, not a change to the
scientific comparison or inference settings.

Unchanged:

- Model A: `openai/gpt-oss-120b`
- Model B: `qwen/qwen3.8-27b`
- provider endpoint family
- system and user prompts
- temperature: 0
- seed: 20260914
- reasoning settings
- max_tokens: 1024
- vignette dataset and wording
- 120-vignette × 6-condition design
- 5 runs per cell
- 7,200 total target responses
- scoring rules
- hypotheses H1-H4
- final statistical analysis plan
- per-model daily cap: 175 attempts

The response cache identity is unchanged. Already-collected responses are not
regenerated, modified, or discarded because of this engineering change.

### Reproducibility note

Production responses collected before this change were obtained under
`max_retries: 6`; responses collected afterward are obtained under
`max_retries: 3`.

The retry ceiling affects **how many provider attempts are allowed before an
uncached cell is deferred to a later collection run**. It does not alter the
payload sent on a successful attempt. A successful response under either
setting receives the same locked model, prompt, temperature, seed, reasoning
configuration, and max-token setting.

For auditability, the Git history preserves both configurations and the exact
transition commit.

---

## 2026-09-19 — Raw responses committed before downstream QC

**Change commit:** `96efd82ab104ecfd73976a177b101b3b72571dc0`

The collection workflow was changed so newly written files under
`results/raw/` are committed to `main` immediately after collection and
before scoring, control checks, validation-sample refresh, or final analysis.

### Rationale

A model response represents already-spent provider quota and primary study data.
It should not depend on downstream scoring/QC code succeeding before it becomes
durable. Git is now the authoritative persistence checkpoint; the Actions cache
remains only a performance/resume aid.

This did not alter any model call, prompt, response, score, or statistical
decision.

---

## 2026-09-19 — Partial-success workflow semantics and retry telemetry

**Change commit:** `877a4d680d78095d8a094c9bed060ed9b27b6113`

The collection code and workflow were updated so productive partial collection
does not automatically become a generic workflow failure.

Operational additions include:

- structured retry telemetry for future provider diagnosis;
- run classifications including `SUCCESS`, `PARTIAL_SUCCESS`,
  `CAPPED_CLEAN`, and `FAILURE`;
- Model B collection remains eligible even if Model A has an independent
  collection-step failure;
- downstream scoring/QC can continue after productive partial collection;
- Notion run status can represent a partial collection outcome;
- uncached failed cells remain pending and are naturally retried later.

The preregistered inferential analysis remains gated until the complete
7,200-response corpus exists.

---

## 2026-10-04 — Per-model request pacing (Model B: 6 -> 2 req/min)

**Change commit:** (this commit)  
**Affected files:** `config/protocol.yaml`, `src/run.py`  
**Field:** `rate_limits.requests_per_minute_by_model` (new, optional)

**Before:** one process-wide `RateLimiter` shared by both model slots at
`requests_per_minute: 6`.  
**After:** one `RateLimiter` per model slot. Slot A keeps the default
(6 req/min, unchanged). Slot B is overridden to 2 req/min via
`requests_per_minute_by_model: {B: 2}`.

### Triggering production evidence

The 2026-09-19 retry-telemetry change could only infer the retry cause from
wall-clock timing. By 2026-10-04, production collection had accumulated 3,984
raw responses (2,830 on slot A, 1,154 on slot B) carrying that telemetry,
which proves it directly:

- Slot A: 0 of 2,830 written responses carry any `retry_events` entry.
- Slot B: 994 of 1,154 (86%) carry at least one `retry_events` entry, 1,366
  retry events in total, **every one of them HTTP 429**. Zero 5xx, zero
  transport errors, on either slot.

Groq is rate-limiting slot B's `qwen/qwen3.8-27b` endpoint specifically, not
slot A's `openai/gpt-oss-120b` endpoint, and the mechanism is confirmed as
provider-side rate limiting rather than server errors or network issues.

### Rationale

With a single shared rate limiter, slowing the request pace enough to stop
slot B's 429s would also have slowed slot A, which has never needed it and
was collecting cleanly at 6 req/min. Giving each model slot its own
`RateLimiter` instance lets the pace be set per the evidence for that slot
alone. `max_concurrent: 1` is unchanged, so requests still run strictly
sequentially; a 2 req/min slot-B limiter spaces consecutive slot-B requests
>= 30s apart without affecting slot A's spacing (>= 10s apart, from the
unchanged default).

### Scientific impact assessment

Operational collection-pacing change only. Unchanged: both model checkpoints,
provider endpoint family, system/user prompts, temperature (0), seed,
reasoning settings, `max_tokens` (1024, locked), the 120-vignette x
6-condition x 5-run design, 7,200-response target, `max_retries` (3),
`daily_request_cap_per_model` (175), `max_collection_minutes` (300), scoring
rules, hypotheses H1-H4, and the final statistical analysis plan (still
gated on the complete 7,200-response corpus).

Already-collected responses are not regenerated, modified, or discarded.
Nothing about a successful call's payload changes -- only how long the
collector waits between slot B's requests.

### Reversibility

Delete `rate_limits.requests_per_minute_by_model` (or set `B: 6`) to return
both slots to a single shared rate. No code change is required to revert;
`src/run.py` falls back to `requests_per_minute` for any slot without an
override.

---

## Interpretation rule

Entries in this file should be treated as part of the provenance record for the
study. They explain **how** the fixed study was executed under real provider and
CI constraints.

If a future change would alter what a model is asked, which model is used, the
locked inference payload, the dataset, the scoring definition, an exclusion
rule, a hypothesis, or the final statistical test, it should not be recorded
only here. It must be handled explicitly as a scientific protocol amendment.
