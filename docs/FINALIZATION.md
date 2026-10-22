# Finalization automation

After the production cache reaches exactly 7,200 raw responses, the
finalize-study workflow starts automatically after the collection workflow.

Automatic steps before the human gate:

1. rerun deterministic scoring and control QC;
2. rerun the preregistered final analysis from the complete corpus;
3. create SHA-256 manifests for all 7,200 raw responses and the frozen inputs;
4. commit the frozen collection state;
5. expose the deterministic 50-response blinded validation worksheets (one per coder).

Human gate — two independent coders:

Two independent human coders (A and B) must each fill `action_category` for
all 50 rows, in their own worksheet:

- `results/tables/manual_validation_50_coder_a.csv`
- `results/tables/manual_validation_50_coder_b.csv`

Each coder works from an identical sample (same 50 rows, same seed) but sees
only their own file: neither sees the other coder's labels, and neither sees
the automated classifier's labels (held separately in
`manual_validation_50_key.csv`). This blinding, on both axes, is what makes
the resulting statistic a genuine inter-rater reliability check rather than
one person's labels compared to an algorithm. `sample_manual_validation.py`
refuses to regenerate either worksheet once it already has coded rows, so a
re-run can't silently erase in-progress or completed coding.

This step is intentionally not automated — the preregistered procedure
requires genuine independent human coding. Committing both completed files
automatically resumes finalization.

Automatic steps after the human gate (not held back — these don't create new
identity-revealing public artifacts, just commit results to the already
existing repo):

1. compute agreement: **primary** is Cohen's kappa between coder A and coder
   B (the actual inter-rater reliability statistic); **secondary** is each
   coder's agreement with the automated classifier, reported for reference
   only and never presented as inter-rater reliability;
2. commit the agreement summary, confusion matrices, and both coder
   worksheets.

## Public-release gate

Everything after that — reserving/publishing the Zenodo DOI, creating the
`study-final-v1.0.0` git tag, creating the GitHub Release, and disabling the
daily collection workflow — is held behind a **manual** `workflow_dispatch`
confirmation (`confirm_public_release: true`). It will not fire automatically
just because both coder worksheets are committed.

Why: those steps create new, search-indexable public artifacts (a titled
GitHub Release, a published Zenodo DOI record) under this real,
author-identified repo (`ibalajisivarajan/doomscroll-bias`). Doing that while
the study is under FAccT double-anonymous review would let a reviewer
discover the author's identity via a plain web search, independent of
whether the submitted paper itself links to any of it. Run the workflow
manually with `confirm_public_release=true` only once that risk is gone —
typically after the review decision, at camera-ready.

Steps gated behind this confirmation, once run:

1. reserve a Zenodo DOI;
2. write the reserved DOI into `CITATION.cff` and `paper/RELEASE_METADATA.md`;
3. create the immutable Git tag `study-final-v1.0.0`;
4. create a GitHub Release containing an archive of that exact tag;
5. upload that tagged archive to Zenodo and publish the DOI;
6. update the GitHub Release and release metadata with the final DOI and OSF link;
7. disable the daily collection workflow.

## One-time setup

Create a GitHub Actions repository secret named `ZENODO_TOKEN` containing a
Zenodo personal access token with deposit/write permission.

For a sandbox test, create repository variable `ZENODO_API_URL` with value:
`https://sandbox.zenodo.org/api`

Leave `ZENODO_API_URL` unset for production Zenodo.

The OSF preregistration remains immutable. The GitHub release, Zenodo record,
and paper release metadata all link back to https://osf.io/ndvw8/ — note that
page currently displays the author's name publicly, so for the anonymous
submission itself, use OSF's "Create anonymized link" feature instead of this
direct URL.
