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

## 2026-10-07 — Per-model request pacing tuned (Model B: 2 -> 1 req/min)

**Change commit:** (this commit)  
**Affected file:** `config/protocol.yaml`  
**Field:** `rate_limits.requests_per_minute_by_model`

**Before:** `{B: 2}` (slot A unchanged at the default 6 req/min).  
**After:** `{B: 1}` (slot A still unchanged at the default 6 req/min). No
`src/run.py` change: the per-slot `RateLimiter` introduced on 2026-10-04
already resolves `requests_per_minute_by_model` overrides at the per-task
level, so only the config value moves.

### Triggering production evidence

Three full nights at `B: 2` (2026-10-05, -06, -07) were measured directly
from the committed raw responses and their `retry_events` telemetry:

| night | slot B written | slot B retries | slot B attempts | retry share | HTTP status |
|---|---|---|---|---|---|
| 1 (`c02c590`) | 95 | 80 | 175 | 46% | 429 only |
| 2 (`ab07959`) | 118 | 57 | 175 | 33% | 429 only |
| 3 (`94bf656`) | 117 | 58 | 175 | 33% | 429 only |

The 2026-10-04 fix is working -- slot B's fresh-response rate roughly
doubled (+95/+118/+117 per night vs +65/night before any per-model pacing)
-- but on the two settled nights roughly one third of slot B's 175 daily
attempts are still spent on retries rather than new cells, and every single
logged retry across all three nights is HTTP 429. Zero 5xx, zero transport
errors. Slot A is unaffected: it hit its 175/day cap on all three nights.

### Rationale

Every retry being HTTP 429 with no other error class confirms the
remaining waste is still provider-side rate limiting on slot B's
`qwen/qwen3.8-27b` endpoint, and wider per-request spacing is the only free
lever left against it (the org-level Groq limits for that model --
30 RPM / 1K req/day / 8K TPM / 200K tokens/day -- leave room to go slower
without approaching a different ceiling). Halving slot B's rate from 2 to 1
req/min does not raise `daily_request_cap_per_model`; 175 fresh
responses/day stays the hard ceiling for slot B regardless of how the pace
is tuned, since that cap is set by Groq's daily token budget, not by
request pacing.

Runtime sanity: 175 attempts at 1/min is approximately 3 hours for slot B,
plus roughly 0.5 hours for slot A while it is still collecting, for a
~3.5-hour total -- inside `max_collection_minutes: 300` and the workflow's
350-minute job timeout. Once slot A completes (expected ~2026-10-08),
nightly runs become slot-B-only at roughly 3 hours, comfortably inside both
budgets. No `.github/workflows/run.yml` change was needed or made.

### Scientific impact assessment

Operational collection-pacing change only. Unchanged: both model
checkpoints, provider endpoint family, system/user prompts, temperature
(0), seed, reasoning settings, `max_tokens` (1024, locked), the 120-vignette
x 6-condition x 5-run design, 7,200-response target, `max_retries` (3),
`daily_request_cap_per_model` (175), `max_collection_minutes` (300), slot
A's pacing (6 req/min, unchanged), scoring rules, hypotheses H1-H4, and the
final statistical analysis plan (still gated on the complete 7,200-response
corpus).

Already-collected responses are not regenerated, modified, or discarded.
Nothing about a successful call's payload changes -- only how long the
collector waits between slot B's requests.

### Reversibility

Set `requests_per_minute_by_model: {B: 2}` (or delete the key entirely, for
a single shared 6 req/min rate on both slots) to revert. No code change is
required either way.

---

## 2026-10-15 — H3 paired-aware sensitivity analysis (supplemental, post-lock)

**Change commit:** (this commit)  
**Affected files:** `src/sensitivity_h3.py` (new), `src/analyze.py` (additive),
`tests/test_sensitivity_h3.py` (new)  
**This is not a `PROTOCOL AMENDMENT`.** It does not change what H3 tests, how
it is scored, or any locked design/statistics field in
`config/protocol.yaml`. It adds a second, descriptive, non-preregistered
view of the same question, reported alongside the preregistered test.

### Spec source

Notion card "Specify a paired-aware H3 sensitivity analysis, clearly
labeled supplemental/post-lock" (statistical design authored by ChatGPT,
2026-10-15), handed off via a Boost (research orchestration) implementation
request on 2026-10-15, P0 ahead of the ~2026-10-20 corpus freeze.

### Why

The preregistered H3 test (`src/analyze.py` section 5) is a chi-square of
`action_category` by `culture` computed directly on individual responses --
it treats the 5 repeated runs of a cell as 5 independent observations. They
are not: the 5 repeats measure decoding instability at temperature 0, the
same reason sections 3-4 collapse repeats to a cell mean before testing
H1/H2 at all. A reviewer could reasonably flag the implicit n-inflation in
the H3 chi-square's denominator.

### What was added

A new, independently testable module, `src/sensitivity_h3.py`, implementing
the four specified steps:

1. `aggregate_proportions` -- collapse each (model, vignette, gender,
   culture) cell's repeats into a proportion vector over the 5 action
   categories.
2. `paired_vignette_differences` -- per vignette, the north_american minus
   south_asian proportion difference per category, averaged over the 3
   gender conditions.
3. `category_bootstrap_summary` / `bootstrap_mean_ci` -- percentile
   bootstrap CI per category, resampling whole vignettes (never individual
   responses), mirroring `analyze.bootstrap_ci`'s resampling unit and
   `default_rng(seed)` convention. Reimplemented locally rather than
   imported, since no `src/` module currently imports another.
4. `omnibus_permutation_test` -- a single joint statistic across all
   categories (sum of squared across-vignette mean differences), with a
   Monte Carlo null built by independently sign-flipping each vignette's
   whole diff vector (= swapping its culture labels across all three gender
   variants at once, preserving the within-vignette pairing). See the
   module's docstring for why the statistic squares the across-vignette
   mean rather than each vignette's own value first -- the latter reading
   is invariant under the permutation and would make the test degenerate.

`src/analyze.py` imports `sensitivity_h3` and calls
`compute_sensitivity_h3(good, action_categories, resamples, level, seed)`
using the same `good` (analysable, exclusion-filtered) frame, and the same
`resamples`/`level`/`seed` config values, that the existing H1/H2 bootstrap
and H3 chi-square already use -- nothing about exclusion rules, the design
config, or the random seed is reinterpreted for this analysis. The result is
rendered as a new `## 5b. H3 paired-aware sensitivity analysis (supplemental,
post-lock)` section in `results/tables/results.md`, positioned between the
existing section 5 (preregistered H3 chi-square) and section 6 (H4). It
reports, per model slot: a table of mean paired difference + 95% CI per
action category, and the omnibus permutation statistic, p-value, and the
planned null stated in words. It is explicitly marked supplemental/not
preregistered in its own text, repeats the classifier-validity caveat from
section 5, and states plainly that it is not Holm-corrected and not folded
into the secondary-family correction table.

### Verification

- `python -m unittest tests.test_sensitivity_h3 -v` -- 11 tests, all pass:
  proportion-vector aggregation (sums to 1, unused categories are 0 not
  missing), paired-difference averaging over genders (correct sign and
  averaging; a missing culture cell gives NaN, not a silent 0), bootstrap CI
  shape/sanity (contains the point estimate on symmetric synthetic data;
  degenerate n=1 input returns NaN bounds without crashing; deterministic
  given the same seed), the permutation null (a genuinely null synthetic
  dataset does not collapse to p~0; a strong, consistent synthetic signal is
  detected at p<0.01; an explicit check that the rejected "per-vignette
  squared-then-averaged" statistic reading is sign-invariant, which is why
  the implementation instead squares the across-vignette mean).
- `python src/analyze.py` run on the current cache (6,379 raw responses,
  6,378 analysable) completes in ~3.4s and appends the new section.
- Preregistered-output check, run on the *same* cache both ways (the
  previously committed `results.md` predates most of the current cache, so
  comparing against it directly would have shown a stale-cache diff, not a
  code diff): ran the pre-change `src/analyze.py` and the post-change
  version against the identical `results/tables/scored.csv`, then diffed
  section 5 between the two outputs line-for-line. **Identical.** The full
  file outside the new section 5b block is also identical between the two
  runs (H1/H2 numbers, figures, H4 section, everything).

### Scientific impact assessment

Additive, descriptive supplemental analysis. Unchanged: the preregistered
H3 chi-square's code and output, H1/H2/H4, the statistical plan's Holm
correction and secondary family, `config/protocol.yaml` (not edited at
all), `src/run.py`, `src/score.py`, the collection pipeline, the vignette
dataset, and every locked design/inference setting. The new section applies
no new significance threshold, is not Holm-corrected, and its content (which
categories, which statistic, 10,000 resamples) was fixed by the written
spec before this cache was analysed -- not selected after seeing the
result.

### Next step (not done here)

Per the handover: Boost revises the manuscript (Methods 3.7) to describe
this supplemental analysis, and ChatGPT reviews the statistical
implementation against the original spec.

---

## 2026-10-15 — H3 sensitivity analysis: missing-data and complete-case correction

**Change commit:** (this commit)  
**Reviewer:** ChatGPT (review of commit `a9cceb1`, verdict: CHANGES REQUESTED)  
**Affected files:** `src/sensitivity_h3.py`, `src/analyze.py` (call site and
rendering only), `tests/test_sensitivity_h3.py`  
**Not a `PROTOCOL AMENDMENT`.** Corrects two implementation bugs in the
2026-10-15 supplemental analysis; does not change the preregistered H3
chi-square, any locked design/statistics field, or what the supplemental
analysis is testing.

### Issue 1 — observed and permutation statistics could disagree on which vignettes counted

`omnibus_permutation_test()` computed the observed statistic via
`omnibus_statistic(diffs_df, categories)`, which filtered each category's
NaNs **independently**, before the row mask used for the permutation
distribution existed. For data produced by this module's own
`paired_vignette_differences` the two happened to select the same rows (a
given vignette's NaN-ness never differed by category there), so this was
latent rather than observed in commit `a9cceb1`'s reported numbers -- but it
was not provably correct in isolation, and an empty/all-NaN input returned
`0.0` ("no shift detected") instead of `NaN` ("nothing to measure").

**Fix:** a new `complete_case_matrix(diffs_df, categories)` establishes the
eligible complete-case vignette set once, before either statistic is
touched; `omnibus_statistic(values)` now takes that already-filtered matrix
directly (and returns `NaN`, not `0.0`, when it is empty); and
`omnibus_permutation_test()` returns a 4th value, `n_complete_vignettes`, so
callers can report the exact sample size both statistics were actually
computed from. **Regression tests added** (`CompleteCaseMatrixRegressionTest`
in `tests/test_sensitivity_h3.py`) construct a `diffs_df` with a
category-specific NaN by hand and confirm: the pre-fix per-category-filter
style of computing the observed statistic disagrees with the corrected,
single-complete-case-set version on that exact input; the corrected
`omnibus_permutation_test` reports `n_complete` consistent with
`complete_case_matrix`'s own row count; and an all-missing input returns
`NaN` for both the observed statistic and the p-value, with an empty
permutation distribution, rather than `0.0`/a crash.

### Issue 2 — paired-gender completeness was not enforced

`paired_vignette_differences()` averaged a vignette's NA-minus-SA contrast
over however many of the three gender pairs happened to have both culture
cells collected, silently reweighting a vignette with 1 available gender
pair the same as one with all 3 -- no distinction was made, and nothing
reported how many vignettes this affected.

**Fix:** the primary supplemental analysis is now complete-case only. A
vignette is included only if all three gender conditions each have both the
north_american and south_asian cell present (6 of 6 required cells); a
vignette missing even one is excluded entirely, not partially averaged. A
new `vignette_completeness()` reports, per model slot, how many vignettes
were included versus excluded out of the total with any data at all, so the
exclusion is never silent. Partial-pair results are not retained as a
separate descriptive table -- out of scope per the review ("if partial-pair
results are retained, separate them clearly"; this implementation does not
retain them). `src/analyze.py`'s section 5b now reports the included/
excluded/total counts per slot before the category table.

On the current cache (6,379 raw responses, checked 2026-10-15): slot A is
unaffected (120/120 vignettes complete); slot B has 92/93 included, 1
excluded for an incomplete gender/culture pairing. This will matter more as
collection continues and more partially-collected vignettes exist.

### Testing performed

`python -m unittest tests.test_sensitivity_h3 -v` -- **24 tests, 24 passed,
0 failed**, covering, per the review's required list: complete paired data
producing the expected contrasts; a single missing name-cue condition for
one gender (whole vignette excluded); an entire missing gender pair (whole
vignette excluded); missing/unclassifiable action-category values (excluded
from the numerator, still counted in the denominator); no complete paired
vignettes at all, at both the `paired_vignette_differences` and
`compute_sensitivity_h3` orchestration level (NaN results, not a crash);
observed and permutation statistics on identical vignette sets (the
regression tests above); deterministic results given a fixed seed (both
`bootstrap_mean_ci` and `omnibus_permutation_test`); and existing
preregistered H3 output remaining unchanged, tested directly as
`compute_sensitivity_h3` not mutating its input `good` frame in place
(`test_does_not_mutate_input_good_frame`) -- the live-cache diff below is
the complementary real-data confirmation of the same property.

Full existing suite: `tests/test_sensitivity_h3.py` is the only test module
in this repository (no other `tests/*.py` exists), so the full suite and the
relevant suite are the same 24 tests.

`python src/analyze.py` run on the current cache (6,379 responses, 6,378
analysable): completes cleanly, writes an updated section 5b. Preregistered-
output check, same method as the 2026-10-15 entry (pre-change and
post-change `analyze.py` run against the identical `results/tables/scored.csv`,
diffed): section 5 (the preregistered H3 chi-square) is **byte-identical**;
the full file outside the 5b block is **byte-identical**. Rerunning
`analyze.py` a second time on its own output is also byte-identical
(deterministic).

### Scientific safeguards confirmed unchanged

`config/protocol.yaml` (not edited), frozen hypotheses H1-H4, the
preregistered H3 chi-square's code and output, `src/run.py`, `src/score.py`,
the collection pipeline, and `data/vignettes.jsonl`. The supplemental
analysis remains labelled supplemental / post-lock / non-preregistered in
its own `results.md` text and is still not folded into the secondary-family
Holm correction. The current cache (6,379/7,200) is incomplete; nothing
here interprets or presents these numbers as final findings.

### Residual limitations

- **Correction (same day, later commit):** the bullet originally here named
  `B V002 female_south_asian run=3` as an example of a vignette this
  complete-case requirement would exclude. That was imprecise in two ways,
  worth stating plainly rather than just quietly fixing: (1) exclusion in
  `paired_vignette_differences`/`vignette_completeness` is decided per
  (gender, culture) **cell**, not per individual repeat -- a cell counts as
  present as soon as it has at least 1 of its 5 repeats, so naming a single
  `run=3` could be misread as "one missing repeat excludes the vignette,"
  which is not how the code behaves; and (2) that specific example was
  already stale by the time this entry was written -- as of 2026-10-15 all
  5 of that cell's repeats are collected (the 2026-09-19 concern that it
  would keep failing non-retryably did not play out that way in practice),
  so it was never actually excluded under this fix.

  The real, current example on today's cache: slot B's vignette `D008` is
  excluded because its `(neutral, south_asian)` cell has **zero** collected
  responses -- that cell simply does not exist yet in `results/raw`. Its
  sibling cell, `(neutral, north_american)`, has only 4 of 5 repeats
  collected and still counts as present, which is exactly the contrast that
  matters: a cell missing repeats is still "present" (1-4 of 5 responses is
  enough); a cell missing entirely (0 of 5) is what triggers exclusion of
  the whole vignette for that model slot.

- The complete-case requirement will reduce `n` further as collection
  continues if any vignette ends up with a persistently empty cell (zero
  responses, for whatever reason) -- that vignette will be excluded from
  this supplemental analysis for that model slot until/unless the missing
  cell is eventually collected, even though it is not necessarily excluded
  from the preregistered analyses, which use different per-test pairing.
- Partial-pair vignettes are excluded, not reported as a separate descriptive
  diagnostic; if ChatGPT's review wants that retained, it is a follow-up
  scope decision, not implemented here.

Returned to ChatGPT for review per the handover; not marked Done here.

---

## 2026-10-15 — H1/H2 complete-case matched-pair effect-size sensitivity (supplemental, post-lock)

**Change commit:** (this commit)  
**Affected files:** `src/sensitivity_h1h2.py` (new), `src/analyze.py`
(additive), `tests/test_sensitivity_h1h2.py` (new)  
**This is not a `PROTOCOL AMENDMENT`.** Does not change what H1/H2 test,
how they are scored, or any locked design/statistics field in
`config/protocol.yaml`. Adds a second, descriptive, non-preregistered view
of the same questions, reported alongside the preregistered test.

### Spec source

Task assigned 2026-10-15 following approval of the H3 supplemental
analysis (commit `09e5645`/`9a7112a`): "Implement H1/H2 paired effect-size
sensitivity: Add the supplemental matched-pair effect-size calculation,
tests, and documentation, preserving the preregistered analysis." No
detailed written spec (no Notion-card equivalent to H3's) accompanied this
item. Given the stakes of guessing wrong on a primary-hypothesis-adjacent
change, the specific design below was proposed and confirmed before
implementation, rather than assumed.

### Why

While confirming there was no written spec to follow, direct inspection of
`src/analyze.py`'s `paired_gaps()` -- the function the PREREGISTERED H1/H2
test is built from -- found the exact same bug pattern the ChatGPT review
had just caught in the H3 supplemental analysis (`09e5645`): each gender's
value is computed as `np.nanmean(list({culture: cell(...) for culture in
(...)}.values()))`, so a vignette with only 1 of 2 culture cells collected
for "male" is silently treated the same as one with both. This exists in
the preregistered pipeline itself, not a supplemental module -- worth
surfacing explicitly rather than silently working around.

### What was added

`src/sensitivity_h1h2.py`, mirroring `sensitivity_h3.py`'s structure and
conventions (no `src/` module imports another; small independently-testable
reimplementations of `analyze.py`'s `cliffs_delta`/`wilcoxon`/bootstrap
logic):

- `vignette_completeness()` / `complete_case_paired_gaps()` -- a vignette is
  included only if male AND female are both present in both
  north_american and south_asian (4 cells; unlike H3, `neutral` never
  enters the H1/H2 gap formula at all, so it is irrelevant to this
  completeness check). Computes the identical `gap` /
  `gap_north_american` / `gap_south_asian` / `gap_culture_diff` quantities
  `paired_gaps()` does, using a plain mean (never `nanmean`) since
  completeness is already guaranteed.
- `compute_sensitivity_h1h2()` -- per model slot, recomputes the H1 ("gap")
  and H2 ("gap_culture_diff") Wilcoxon stat/p, Cliff's delta, mean, and
  bootstrap CI (unadjusted p, never Holm-corrected, never folded into the
  primary family) on the complete-case subset.

`src/analyze.py` imports `sensitivity_h1h2` and renders a new
`## 3b. H1/H2 complete-case matched-pair effect-size sensitivity
(supplemental, post-lock)` section, positioned between the preregistered
section 3 and section 4, reporting included/excluded counts per slot and
the H1/H2 table alongside the preregistered one.

### Verification

- `python -m unittest discover -s tests` -- **33 tests, 33 passed, 0
  failed** (9 new in `test_sensitivity_h1h2.py`, 24 existing in
  `test_sensitivity_h3.py`, unaffected). Covers: a direct parity check --
  `complete_case_paired_gaps()` against `analyze.paired_gaps()` on the same
  complete synthetic data, asserting the gap/gap_culture_diff numbers match
  to 9 decimal places, proving this is a faithful recomputation of the
  preregistered formula and not a different one; a vignette missing one
  culture cell being excluded here while `paired_gaps()` would silently
  nanmean over the rest; no complete-case vignettes at all (NaN, not a
  crash); determinism under a fixed seed; and `compute_sensitivity_h1h2()`
  not mutating its input `cells` frame (the same frame section 3's
  preregistered test reads from).
- Same-cache diff method as the H3 entries: ran the pre-change and
  post-change `src/analyze.py` against the identical current
  `results/tables/scored.csv` and diffed -- the preregistered section 3
  (H1/H2) is **byte-identical**, as is the preregistered section 5 (H3
  chi-square) and the rest of the file outside the two supplemental
  blocks. Reran `analyze.py` a second time on its own output: also
  byte-identical (deterministic).
- On the current cache (6,379 responses): both slots show **0 excluded
  vignettes** (120/120 for A, 93/93 for B) and the complete-case H1/H2
  numbers are identical to the preregistered ones -- the current
  preregistered H1/H2 result happens to already be complete-case-robust,
  because every vignette with any male/female data already has both
  cultures for both genders (the only current gap, vignette `D008`'s
  missing `neutral`/`south_asian` cell, doesn't affect H1/H2, which never
  uses `neutral`). This sensitivity check exists for ongoing/future
  partial collection, not because today's numbers were already in doubt.

### Scientific safeguards confirmed unchanged

`config/protocol.yaml` (not edited), frozen hypotheses H1-H4, the
preregistered H1/H2 code and output (`paired_gaps()`, the Wilcoxon/Cliff's
delta/Holm/bootstrap pipeline in section 3), `src/run.py`, `src/score.py`,
the collection pipeline, `data/vignettes.jsonl`. Labelled
supplemental/post-lock/non-preregistered in its own `results.md` text, and
its p-values are not folded into the primary-family Holm correction. The
current cache (6,379/7,200) is incomplete; nothing here interprets or
presents these numbers as final findings.

### Residual limitations

- Like the H3 supplemental analysis, the complete-case requirement here
  will exclude more vignettes as collection continues if any vignette ends
  up with a persistently empty male/female culture cell. None currently
  exist for either slot.
- Mirrors the H3 fix's choice: incomplete-pair vignettes are excluded, not
  retained as a separate descriptive diagnostic table.

---

## 2026-10-15 — H1/H2 matched-pair sign statistic (supplemental, post-lock)

**Change commit:** (this commit)  
**Affected files:** `src/sensitivity_h1h2.py` (additive), `src/analyze.py`
(additive), `tests/test_sensitivity_h1h2.py` (additive)  
**Not a `PROTOCOL AMENDMENT`.** Adds a further supplemental, descriptive
statistic to the already-approved H1/H2 complete-case sensitivity analysis
(prior entry, this file); changes nothing preregistered and nothing already
reported in that prior entry's section.

### Why

ChatGPT's review of `d667431` approved the complete-case effect-size
sensitivity analysis as additional supplemental work, but noted it did not
implement an item the H1/H2 Notion card specifies exactly: the matched-pair
sign statistic, `(n_positive - n_negative) / (n_positive + n_negative)`,
with positive/negative/tied/total/non-tied counts, and defined handling for
empty and all-tied samples.

### What was added

`sign_statistic(values)` in `src/sensitivity_h1h2.py`: counts the sign of
each paired difference (positive, negative, tied), ignoring magnitude --
deliberately complementary to the already-reported Cliff's delta/Wilcoxon,
which are magnitude-sensitive. Ties are excluded from the statistic's own
denominator (`n_nontied`, the classical sign-test convention) but always
counted and returned (`n_tied`), never silently dropped. Returns `NaN` --
never `0.0`, never a division error -- whenever `n_nontied == 0`: both an
empty sample and an all-tied sample hit this path, and conflating either
with "signs balanced out" (what `0.0` would communicate) would misreport
"nothing to measure" as a null result.

`compute_sensitivity_h1h2()` now also returns `h1_sign`/`h2_sign` per model
slot, computed on the *same* complete-case `gap`/`gap_culture_diff` values
already used for `h1`/`h2` -- not a different dataset -- but kept as
separate dict keys so nothing about the existing `h1`/`h2` results changes.
`src/analyze.py` renders this as its own, separately labeled
`## 3c. H1/H2 matched-pair sign statistic (supplemental, post-lock)`
section, after the existing `## 3b.` section (left untouched) and before
section 4.

### Testing performed

`python -m unittest discover -s tests` -- **43 tests, 43 passed, 0
failed** (24 existing H3 + 19 H1/H2, 10 of which are new: `SignStatisticTest`
covers known counts producing the exact specified formula; an all-positive
sample giving +1; an all-negative sample giving -1; an empty sample
returning NaN with all counts at 0; an all-tied sample returning NaN (not
0.0) with `n_tied == n_total`; NaN values dropped before counting; and a
direct regression test proving the denominator is `n_nontied`, not
`n_total` -- asserting the statistic differs from what a naive
total-denominator implementation would compute on the same ties-containing
input. `ComputeSensitivityH1H2SignIntegrationTest` covers `h1_sign`/`h2_sign`
being present and consistent with `h1`/`h2`'s own sample size, correct NaN
propagation when there are no complete-case vignettes, and an explicit
guard that the existing `h1`/`h2` dicts are unchanged by this addition.

Same-cache diff method as every prior entry: ran the pre-change and
post-change `src/analyze.py` against the identical current
`results/tables/scored.csv` and diffed. **Four separate blocks checked,
all byte-identical:** the preregistered section 3 (H1/H2), the
preregistered section 5 (H3 chi-square), the *existing* section 3b
(complete-case effect-size sensitivity, from the prior entry -- confirming
it was genuinely preserved, not just left alone by inspection), and the
entire rest of the file outside the one new section 3c block. Reran
`analyze.py` a second time on its own output: also byte-identical.

On the current cache: Slot A -- H1 77 positive/17 negative/26 tied
(n=120), sign statistic 0.638; H2 42/46/32, -0.045. Slot B -- H1
66/14/13 (n=93), 0.650; H2 47/33/13, 0.175.

### Scientific safeguards confirmed unchanged

`config/protocol.yaml` (not edited), frozen hypotheses H1-H4, the
preregistered H1/H2/H3 code and output, the existing H1/H2 complete-case
effect-size sensitivity section from the prior entry, `src/run.py`,
`src/score.py`, the collection pipeline, `data/vignettes.jsonl`. The new
section is explicitly labeled supplemental/post-lock/non-preregistered and
is not folded into any Holm correction or family.

---

## 2026-10-16 — Statistical audit edge cases: Wilcoxon insufficient-n, H1 partial pairing, severity exclusion

**Change commit:** (this commit)  
**Reviewer:** ChatGPT (statistical audit, 15 Oct 2026, Notion task "Review
JSON parsing, scoring, excluded keys, Wilcoxon ties, missingness and
incomplete paired-cell handling"; implementation task "Claude Code: resolve
statistical audit edge cases (Wilcoxon, H1 paired coverage, severity
exclusions)", P0, due 2026-10-19)  
**Not a `PROTOCOL AMENDMENT`.** `config/protocol.yaml` is not edited.
Addresses three specific edge cases the audit identified in
`src/analyze.py`; two get a code fix, one gets a documented determination
plus additive reporting, consistent with "align only with documented frozen
rule."

### Issue 1 — `wilcoxon()` reported p=1.0 for both insufficient-n and all-tied

**Before:** `len(differences) < 2 or nonzero == 0` both returned `(NaN, 1.0,
nonzero)` -- conflating "cannot test at all" (0 or 1 paired observation)
with "tested, every difference was exactly zero" (a real, meaningful
degenerate result, the convention this function's own docstring already
disclosed).

**Fix:** split the two cases. `len(differences) < 2` now returns
`(NaN, NaN, nonzero)` -- `NaN` renders as "n/a" via the existing `fmt_p()`,
distinct from a real `1.0`. `nonzero == 0` with `len >= 2` is unchanged:
still `(NaN, 1.0, nonzero)`. The scipy-level `ValueError` fallback (a
separate, pre-existing degenerate path the audit did not flag) is
unchanged, still `1.0`, since changing it would need its own separate
justification.

**Consequence requiring its own fix:** `holm()` takes a list of p-values
across a whole family (e.g. H1 and H2 together) and would silently corrupt
the OTHER real test's adjusted p-value if one entry in the list became
`NaN` -- `NaN` propagating through Python's `sorted`/`max` does not
reliably "sort last." `holm()` now filters `NaN` entries out of the
correction itself (they don't count toward the family size `m` and don't
enter the step-down ordering), returning `(NaN, False)` for them --
"untested," never a false "no" -- while every real entry's adjusted
p-value is computed exactly as if the untestable one were never in the
list. When every input is a real number (every case this study has
produced so far), the result is byte-identical to the pre-amendment
implementation.

Both render sites that show a "reject at a=..." yes/no column (section 3's
primary table, section 6/7's secondary-family table) now show `"n/a"` for
an untested entry instead of a misleading `"no"`. `sensitivity_h1h2.py`'s
own `wilcoxon()` reimplementation (which explicitly mirrors `analyze.py`'s)
got the identical fix for consistency, even though it does not call
`holm()` and so has no sibling-corruption risk of its own.

### Issue 2 — H1 (and H2) partial name-cue pairing not reported where the preregistered test lives

The 2026-10-15 entry already built a supplemental complete-case sensitivity
check for exactly this bug (`sensitivity_h1h2.py`, section 3b) -- the audit
is right that it was not visible from the preregistered section 3 itself.
**Fix:** section 3 now also renders a small coverage table, computed by
`sensitivity_h1h2.vignette_completeness()` (already built, tested, and
proven not to alter any preregistered output) directly beneath the
preregistered H1/H2 table -- per model slot, how many vignettes have
complete male+female x north_american+south_asian coverage versus partial
(nanmean'd) coverage, pointing the reader at section 3b. `paired_gaps()`
and the preregistered table's own numbers are untouched; this is reporting
only. On the current cache, 0 vignettes are partial for either slot (H1/H2
never uses the `neutral` gender, and every vignette with any male/female
data already has both cultures for both genders -- the one current gap,
`D008`'s missing `neutral`/`south_asian` cell, doesn't reach this check).

### Issue 3 — `severity` has no range-exclusion equivalent to `fault_scroller`/`fault_partner`

**Investigated, not silently changed.** `config/protocol.yaml`'s frozen
exclusion rule reads: "Responses missing any required key, or with
out-of-range **fault** values" -- scoped by name to `fault_scroller`/
`fault_partner`, not to `severity` or ratings generically.
`analysable()`'s `missing_keys == ""` check already excludes a response
whose `severity` KEY is entirely absent, same as every other required key
(`score.py`'s `missing = [k for k in required if k not in parsed]` checks
key presence for all 6 required keys uniformly). The gap is narrower than
"severity has no exclusion at all": a response whose `severity` key is
present but holds a value `coerce_score()` cannot parse or that falls
outside 0-10 is not caught by anything, since `missing_keys` only checks
presence, not value validity -- that response's `severity` becomes `NaN` in
`scored.csv` while still passing every current `analysable()` filter.

**Determination: `analysable()` is left unchanged.** The protocol's literal
wording does not extend the range-exclusion to `severity`, and "align only
with documented frozen rule" means not guessing at a stricter rule the text
does not state. This determination is enforced by a test
(`AnalysableSeverityDeterminationTest` in `tests/test_analyze_statistics.py`),
not left as prose that a future change could silently contradict.

**Additive reporting fix:** section 2's per-model quality table gained an
`n with valid severity` column (count of `severity.notna()` within each
model's analysable set), with a note explaining exactly why it can differ
from `n` and confirming it currently does not (both slots: `n with valid
severity == n`, 3,599/3,599 for A, 2,779/2,779 for B -- the gap is latent,
not currently biasing `mean severity`). Section 1's exclusion-rule summary
line was also tightened from "out-of-range ratings" (its own prior
wording, which overclaimed beyond what the protocol text actually says) to
"out-of-range **fault** values specifically," matching the frozen wording
exactly.

### Testing performed

- `python -m unittest discover -s tests` -- **56 tests, 56 passed, 0
  failed** (43 existing, unaffected, + 13 new in
  `tests/test_analyze_statistics.py`): `WilcoxonDegenerateInputTest`
  (0 and 1 observations -> NaN p, not 1.0; 2+ all-tied -> still the
  disclosed 1.0; normal input unaffected; NaN padding dropped before the
  n-check); `HolmNaNHandlingTest` (a known hand-checked Holm case with no
  NaNs reproduces the exact pre-amendment arithmetic; a direct regression
  test proving a NaN entry does not change the adjusted p-values of the
  real entries at all, compared to running `holm()` on just those real
  entries alone -- the exact bug a naive "sort NaN last" implementation
  would have; all-NaN input; empty input; NaN position not mattering);
  `AnalysableSeverityDeterminationTest` (the Issue 3 determination, plus
  confirming the missing-key and out-of-range-fault paths are both still
  correctly excluded). One existing test in `test_sensitivity_h1h2.py`
  (`test_no_complete_paired_vignettes_reports_nan_not_a_crash`) was updated
  from asserting the old `p == 1.0` to the corrected `NaN` -- it was
  exercising exactly the insufficient-n case this amendment fixes.
- Same-cache diff method as every prior entry: ran the pre-change and
  post-change `src/analyze.py` against the identical current
  `results/tables/scored.csv` and diffed. **Byte-identical:** sections 3b,
  3c, 4, 5, 5b in full, and -- critically -- every numeric table row in
  sections 3 and 6/7 (the actual H1/H2/H3/H4 test output: same n, mean, W,
  p, p (Holm), Cliff's delta, and reject/no-reject for every row), proving
  the fix changes zero currently-reported numbers. Only the new,
  additive report text (sections 1, 2, 3's coverage additions) differs.
  Reran `analyze.py` a second time on its own output: also byte-identical.

### Scientific safeguards confirmed unchanged

`config/protocol.yaml` (not edited), frozen hypotheses H1-H4,
`paired_gaps()` itself (only how its surrounding coverage is *reported*
changed, not computed), `analysable()`'s exclusion logic (Issue 3's
determination keeps it exactly as-is), `src/run.py`, `src/score.py`, the
collection pipeline, `data/vignettes.jsonl`, and every number in the
preregistered H1-H4 tables on the current cache.

---

## 2026-10-21 — PROTOCOL AMENDMENT cross-reference: max_tokens 1024 -> 2048

**This is a scientific `PROTOCOL AMENDMENT`, not an operational one.**
Per this file's own interpretation rule below and the header of
`config/protocol.yaml` itself ("Changes to the design, metrics or
statistical plan after `locked: true` get their own commit with a message
beginning 'PROTOCOL AMENDMENT:'"), the authoritative record of this change
is the dedicated commit and the inline comment it left in
`config/protocol.yaml`, not this file. This entry exists only as a
cross-reference, so a reader of this log is pointed at the real record
instead of concluding no one wrote one down.

**Change commit:** `4cf1148` (`PROTOCOL AMENDMENT: raise max_tokens
1024 -> 2048 to finish slot B`)  
**Affected file:** `config/protocol.yaml`, `design.max_tokens` only  
**Handover:** "Claude Code: Protocol amendment to finish slot B
collection" (direct instruction, 2026-10-21)

### Why

Corpus stuck at 7,195/7,200: slot A complete (3,600/3,600), slot B missing
exactly 5 cells (control vignettes C002, C010, C014). Verified directly
against the production job log for dispatch `39552501344`
(job `119064642747`) before making any change: all 5 fail every attempt
with Groq HTTP 400 `json_validate_failed` -- 3 with the explicit
`"max completion tokens reached before generating a valid document"`
message, 2 with the same error code and an empty `failed_generation`
field (same underlying symptom, slightly different diagnostic text).
`src/run.py` treats HTTP 400 as non-retryable by design, so these 5 cells
fail identically on every future dispatch regardless of how many times
collection resumes -- this is the same failure mode smoke run #1 hit at
500 tokens (documented in the prior comment this amendment sits next to
in `protocol.yaml`), now recurring at 1,024 for these specific control
vignettes.

### What changed, and what provably did not

Only `design.max_tokens`: 1024 -> 2048. Nothing else in
`config/protocol.yaml` was touched (verified: `git diff` on this commit
touches exactly one field). No prompt, model, temperature, seed, vignette,
condition, hypothesis, or scoring/analysis rule changed.

This is also mechanically incapable of altering any already-collected
response, not merely asserted to be: `src/run.py`'s `Task.cache_key`
hashes `model_id`, `reasoning_effort`, `reasoning_format`, `run_index`,
`system_prompt` and `user_prompt` -- `max_tokens` is not part of it, so a
cell with an existing cache file is never re-sent regardless of this
change. Verified empirically, not just by reading the code: ran
`python src/run.py --dry-run` immediately before and immediately after the
edit -- both runs report the identical 5 pending cells with the identical
cache keys (e.g. `B C002 male_north_american run=3 key=37f388d2432d`,
unchanged). The other 7,195 responses completed successfully within the
1,024-token budget (none were truncated -- truncation is exactly the
failure mode being fixed), so raising the cap could not have changed their
content even if it had been in effect retroactively.

### Residual risk -- flagged, not absorbed

These 5 cells will collect under a larger token budget (2,048) than the
other 7,195 responses (1,024). A longer reasoning allowance is a possible,
though unproven, confound specific to these 5 responses relative to the
rest of the corpus for the same model and condition family. This is stated
here plainly rather than silently treated as equivalent collection
conditions. [Filled in after collection: see the dispatch result below for
whether this remained a theoretical risk or changed any reported number.]

### Collection result

[Filled in after the post-amendment dispatch completes -- see the
Claude Code report for commit SHAs, final raw/per-model counts, and
finalize-workflow status.]

---

## Interpretation rule

Entries in this file should be treated as part of the provenance record for the
study. They explain **how** the fixed study was executed under real provider and
CI constraints.

If a future change would alter what a model is asked, which model is used, the
locked inference payload, the dataset, the scoring definition, an exclusion
rule, a hypothesis, or the final statistical test, it should not be recorded
only here. It must be handled explicitly as a scientific protocol amendment.
