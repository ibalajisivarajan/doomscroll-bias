# doomscroll-bias

**Do open-weight LLMs judge relationship conflict involving doomscrolling
differently depending on the gender of the scrolling partner, and does that
shift between North American and South Asian cultural framings?**

A paired-vignette audit. Sole author: Balaji Sivarajan
([@ibalajisivarajan](https://github.com/ibalajisivarajan)).

`config/protocol.yaml` is the source of truth. Every design number, condition
label, prompt string and statistical decision the code uses is read from that
file — nothing is hard-coded in `src/`. Its commit timestamp is the evidence
that the analysis plan predates the results, so it gets amended in named
commits, never quietly rewritten. **Notion is the dashboard, not a data store**:
`src/notion_sync.py` only ever writes, and nothing in the analysis path reads
from it.

---

## What the study asks

Each vignette describes a conflict between two partners in which one of them
has been doomscrolling. The model is asked to apportion fault, rate severity,
recommend an action, and quote the sentence its assessment rests on. The
question is not whether models find doomscrolling blameworthy — it is whether
the *same* conflict draws a different judgement once the scrolling partner's
name and pronouns change, and whether that difference moves with the cultural
framing of the names.

Three hypotheses are pre-registered as primary or secondary in the protocol
(H1 gender gap, H2 gap-by-culture interaction, H3 recommended-action mix by
culture, H4 distress flagging by gender).

Alongside the bias measures the audit reports a **grounding check**. Every
response must include `evidence_quote`, a sentence copied verbatim from the
vignette. The share of responses whose quote is not actually in the source text
— after unicode, smart-quote and whitespace normalisation — is the
hallucination rate. It is the one metric here that is mechanically verifiable
rather than interpretive, and a model with a high rate is not reading the
vignette it is being asked about.

## Design

| | |
|---|---|
| Design | Paired within-item factorial |
| Vignettes | 120, authored, fictional |
| Factors | gender (male / female / neutral) × culture (north_american / south_asian) |
| Conditions per vignette | 6 |
| Models | 2 open-weight, from different labs |
| Runs per cell | 5 |
| Temperature | 0 |
| **Total calls** | **120 × 6 × 2 × 5 = 7,200** |

**Each vignette is its own control.** Across the six renderings of one
vignette, only the character names and pronouns differ — the conflict, the
wording, the sentence order and the length are identical. The alternative,
comparing separately authored male and female vignettes, confounds the
manipulation with everything else that differs between two texts. Pairing on
the vignette removes all of it and leaves the name-and-pronoun swap.

The scrolling partner carries the gender manipulation. The other partner is
given a name that reads as gender-ambiguous in its own naming tradition
(`Casey`, `Meher`) and they/them pronouns in all six renderings, so that
varying one partner's gender never turns the vignette into a same-sex or
opposite-sex pairing as a side effect.

The 5 repeats at temperature 0 are not there to average away sampling noise.
They exist to *measure* whether the endpoint is actually deterministic: their
median standard deviation is reported as an instability metric, and anything
materially above zero means the per-cell means carry decoding noise the paired
test has to survive.

### Vignettes

`data/vignettes.jsonl`, one JSON object per line:

```json
{"id": "V001", "distress": false, "variants": {"male_north_american": "...", "female_north_american": "...", "...": "..."}}
```

Three rows ship with the scaffold — `V001`, `V002` and `D001` — so the
pipeline runs end to end on day one. **The remaining 117 are authored by
hand.** Ids beginning with `D` embed an explicit distress cue (`D001`: a recent
layoff); they are the only vignettes with a gold label, so distress *recall* is
reported on them and no precision or false-positive rate is claimed anywhere.

## Statistical plan

Fixed before collection, in `config/protocol.yaml`. Nonparametric throughout,
because these are bounded 0–10 integer ratings that can pile up on a single
value:

- **Wilcoxon signed-rank**, two-sided, α = 0.05, on the per-vignette
  male-minus-female differences.
- **Holm** correction across the primary family (the contrasts share
  vignettes, so they are correlated; Holm needs no independence assumption and
  is uniformly more powerful than Bonferroni at the same familywise rate).
- **Cliff's delta** as the effect size, plus the mean gap in raw scale points.
- **Percentile bootstrap** 95% CIs, 10,000 resamples, resampling *vignettes* —
  the vignette set is what the study generalises over.
- **Chi-square** on action category by culture, reported as exploratory.

The 5 repeats collapse to a cell mean *before* any test. Treating them as
independent observations would inflate n fivefold and produce p-values that
mean nothing.

## Running it locally

```bash
pip install -r requirements.txt

# Both slots are served through Groq; verify their model ids against
# GET /openai/v1/models before trusting a run (see the Groq section below),
# then point the keys at them. Slots A and B use two separate secret NAMES
# so a provider swap for one slot only touches one variable -- Groq's
# free-tier quota is tracked at the organization level, not per key, so
# both slots draw from one shared 14,400 req/day pool regardless:
export MODEL_A_API_KEY_GROQ=...
export MODEL_B_API_KEY_GROQ=...

python src/run.py --dry-run                    # inspect the task list, call nothing
python src/run.py --limit 2 --allow-unlocked   # smoke test
python src/run.py                              # full collection (needs locked: true)
python src/score.py                             # -> results/tables/scored.csv
python src/analyze.py                           # -> results/tables/results.md + PNGs
```

`src/run.py` **refuses to run while `locked: false`** unless
`--allow-unlocked` is passed, and says loudly that anything it produces is a
smoke test rather than study data. Collecting against an unfrozen protocol
would break the one provenance guarantee this study has.

It is also **resumable**, which matters because 7,200 calls do not finish in
one sitting. Every response is cached as its own file in `results/raw`, named
by a sha256 of `(model_id, run_index, prompt)`; the presence of the file *is*
the checkpoint, so there is no separate state to corrupt. Writes go to a
`.tmp` sibling and then `os.replace`, which is atomic — a killed job leaves a
stray `.tmp` that the next run ignores, never a truncated `.json` the scorer
would parse as real data. Re-running skips everything already cached, so
Ctrl-C is a pause, not a loss.

Keying the cache on the *prompt* rather than on the condition labels means
that editing a vignette or the prompt template changes the key and forces
re-collection instead of silently mixing responses to two different texts
under one label.

## Running it in Actions

`.github/workflows/run.yml`, `workflow_dispatch` only — no schedule and no
push trigger, because every dispatch spends real API quota. Inputs: `limit`
(max uncached calls this dispatch) and `allow_unlocked`.

A full run **spans several dispatches by design**: GitHub jobs are hard-killed
at 6 hours and free inference tiers rate-limit hard, so the job times out at
350 minutes on its own terms — committing what it has and saving its cache —
rather than being killed with the work stranded. `results/raw` is restored
through `actions/cache@v4` with a `raw-` restore-key, so dispatch *N* starts
from everything dispatches *1..N-1* collected. Dispatch it, let it burn down,
dispatch it again. Progress is monotonic.

Steps: collect → score → analyse → commit `results/` back to `main` → append a
run-log line to Notion (`if: always()`, `continue-on-error: true`) → upload the
tables and figures as artifacts.

### The required secrets

Settings → Secrets and variables → Actions:

| Secret | What it is |
|---|---|
| `MODEL_A_API_KEY_GROQ` | Groq API key, slot A |
| `MODEL_B_API_KEY_GROQ` | Groq API key, slot B — same key *value* as slot A. Groq's free-tier quota is tracked at the organization level, not per key, so this does **not** create a second budget: both slots draw from one shared 14,400 req/day pool. The separate secret name exists so a provider swap for one slot only touches one variable |
| `MODEL_C_API_KEY_OR` | OpenRouter API key, reserved for a replication subsample. Available to the workflow but not yet wired into `config/protocol.yaml`'s `models` list — the primary design is 2 models |
| `NOTION_TOKEN` | Notion internal integration secret |
| `NOTION_PAGE_ID` | id of the Notion page holding the run log |

## The Groq model-id gotcha

The two ids in `config/protocol.yaml` are marked `# PLACEHOLDER` on purpose.
Groq's model catalogue changes slugs and deprecates versions without much
notice, so a hand-typed id is a guess, not a fact — and a wrong id fails every
single call, burning the full retry budget on each one before giving up.
**Verify before locking**, against the account that will actually run the
study:

```bash
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | python3 -m json.tool | grep '"id"'
```

Replace both placeholder ids with the exact strings that come back, then set
`locked: true` in its own commit. `src/run.py` also refuses to run with the
literal string `TBD` in a model slot, but it has no way to know a *filled-in*
id is wrong — that check is on you, once, before the first real dispatch.

Separately: Groq's free tier binds on **tokens per minute** before it binds on
requests per minute. The published ceiling is 30 req/min, but at ~500 tokens a
call and a 6,000 tokens/min budget, the real ceiling is closer to 12 req/min —
`config/protocol.yaml`'s `rate_limits.requests_per_minute: 12` is set to that,
not to the headline number. The daily ceiling (14,400 requests/day, org-wide,
shared across both key slots) binds sooner than either: `src/run.py` tracks
attempts — including retries — in `results/.daily_count.json` and stops
cleanly at `rate_limits.daily_request_cap` (12,000, a margin below the real
ceiling) rather than exhausting the account and locking out every other call
on it for the rest of the day.

## The Notion gotcha

**Creating the integration does not give it access to your page.** You have to
open the page, click the `...` menu at the top right, choose **Connections**
(older UI: "Add connections"), and pick the integration by name.

Until you do, the Notion API answers **404 Not Found** for that page — not
403, not "unauthorized". An un-shared page is indistinguishable from a missing
one from the outside, so the status code points straight at the wrong cause and
you can lose an hour re-checking a page id that was right all along.
`src/notion_sync.py` raises that 404 with the real explanation attached. Full
setup steps are in the module docstring.

## Layout

```
config/protocol.yaml        frozen source of truth: design, prompts, metrics, stats plan
data/vignettes.jsonl        vignettes, 6 renderings each (3 seed rows; 117 to author)
src/run.py                  collection: task list -> /chat/completions -> results/raw
src/score.py                parsing, the verbatim-quote check, action classifier -> scored.csv
src/analyze.py              paired tests, effect sizes, CIs, figures -> results.md
src/notion_sync.py          write-only run log to Notion
results/raw/                one JSON file per response (the resume checkpoint)
results/tables/             scored.csv, results.md, fault-gap histograms
paper/                      write-up
.github/workflows/run.yml   manual dispatch; resumes via the raw cache
```

## Limitations worth stating up front

- **Two models is two models.** A finding is a fact about those checkpoints at
  that temperature through that host, not about open-weight models generally.
- **Names are a coarse proxy for cultural framing.** A name set is not a
  culture, and the North American / South Asian contrast operationalised this
  way cannot separate cultural framing from name familiarity or tokenisation
  effects.
- **The action classifier is a keyword pre-pass.** 50 outputs are hand-coded to
  validate it before the chi-square is interpreted; see `CLASSIFIER_NOTE` in
  `src/score.py`.
- **The vignettes are authored fiction**, written by one person, and carry that
  person's assumptions about what a conflict about phones looks like.

## Citation and licence

See `CITATION.cff`. MIT, © 2026 Balaji Sivarajan.
