# Finalization automation

After the production cache reaches exactly 7,200 raw responses, the
finalize-study workflow starts automatically after the collection workflow.

Automatic steps before the human gate:

1. rerun deterministic scoring and control QC;
2. rerun the preregistered final analysis from the complete corpus;
3. create SHA-256 manifests for all 7,200 raw responses and the frozen inputs;
4. commit the frozen collection state;
5. expose the deterministic 50-response blinded validation worksheet.

Human gate:

The author must fill human_action_category for all 50 rows in
results/tables/manual_validation_50.csv. This is intentionally not automated
because the preregistered procedure requires genuine human coding. Committing
that completed file automatically resumes finalization.

Automatic steps after the human gate:

1. compute exact agreement and Cohen's kappa;
2. reserve a Zenodo DOI;
3. write the reserved DOI into CITATION.cff and paper/RELEASE_METADATA.md;
4. create the immutable Git tag study-final-v1.0.0;
5. create a GitHub Release containing an archive of that exact tag;
6. upload that tagged archive to Zenodo and publish the DOI;
7. update the GitHub Release and release metadata with the final DOI and OSF link;
8. disable the daily collection workflow.

One-time setup:

Create a GitHub Actions repository secret named ZENODO_TOKEN containing a
Zenodo personal access token with deposit/write permission.

For a sandbox test, create repository variable ZENODO_API_URL with value:
https://sandbox.zenodo.org/api

Leave ZENODO_API_URL unset for production Zenodo.

The OSF preregistration remains immutable. The GitHub release, Zenodo record,
and paper release metadata all link back to https://osf.io/ndvw8/.
