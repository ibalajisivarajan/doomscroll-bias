#!/usr/bin/env python3
"""Create and publish the final Zenodo record for the frozen study release.

Set ZENODO_TOKEN in GitHub Actions secrets. For testing, set
ZENODO_API_URL=https://sandbox.zenodo.org/api and use a sandbox token.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

DEFAULT_API = "https://zenodo.org/api"


def req(method: str, url: str, token: str, **kwargs: Any) -> requests.Response:
    params = dict(kwargs.pop("params", {}) or {})
    params["access_token"] = token
    r = requests.request(method, url, params=params, timeout=120, **kwargs)
    if r.status_code >= 400:
        raise SystemExit(f"Zenodo {method} {url} -> {r.status_code}: {r.text[:1000]}")
    return r


def output(path: str | None, **values: str) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in values.items():
            fh.write(f"{k}={v}\n")


def reserve(token: str, api: str, github_output: str | None) -> int:
    dep = req("POST", f"{api}/deposit/depositions", token, json={}).json()
    dep_id = str(dep["id"])
    doi = (
        dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
        or dep.get("metadata", {}).get("doi")
        or ""
    )
    bucket = dep.get("links", {}).get("bucket", "")
    if not doi or not bucket:
        raise SystemExit("Zenodo did not return a reserved DOI and upload bucket")
    output(github_output, deposit_id=dep_id, doi=doi, bucket=bucket)
    print(json.dumps({"deposit_id": dep_id, "doi": doi, "bucket": bucket}, indent=2))
    return 0


def publish(token: str, api: str, dep_id: str, archive: Path, tag: str,
            github_output: str | None) -> int:
    dep_url = f"{api}/deposit/depositions/{dep_id}"
    dep = req("GET", dep_url, token).json()
    bucket = dep["links"]["bucket"]
    doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")

    metadata = {
        "title": "Gendered and Cultural Name-Cue Effects in LLM Judgments of Doomscrolling-Related Relationship Conflict",
        "upload_type": "dataset",
        "description": (
            "Frozen final release of the preregistered Doomscroll Bias study: "
            "protocol, 120-vignette dataset, 7,200 raw model responses, scoring, "
            "quality-control outputs, preregistered analysis, and manual-validation results."
        ),
        "creators": [{"name": "Sivarajan, Balaji"}],
        "access_right": "open",
        "license": "mit",
        "keywords": [
            "large language models", "algorithmic audit", "gender bias",
            "cultural name cues", "doomscrolling", "paired vignette study"
        ],
        "related_identifiers": [
            {
                "identifier": "https://osf.io/ndvw8/",
                "relation": "isSupplementTo",
                "scheme": "url"
            },
            {
                "identifier": f"https://github.com/ibalajisivarajan/doomscroll-bias/releases/tag/{tag}",
                "relation": "isIdenticalTo",
                "scheme": "url"
            }
        ],
    }
    req("PUT", dep_url, token, json={"metadata": metadata})

    with archive.open("rb") as fh:
        req("PUT", f"{bucket}/{archive.name}", token, data=fh)

    published = req("POST", f"{dep_url}/actions/publish", token).json()
    final_doi = published.get("doi") or published.get("metadata", {}).get("doi") or doi
    record_url = published.get("links", {}).get("html") or f"https://zenodo.org/records/{published.get('id', dep_id)}"
    output(github_output, published_doi=final_doi, zenodo_url=record_url)
    print(json.dumps({"doi": final_doi, "zenodo_url": record_url}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.environ.get("ZENODO_API_URL", DEFAULT_API))
    p.add_argument("--github-output")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reserve")
    pub = sub.add_parser("publish")
    pub.add_argument("--deposit-id", required=True)
    pub.add_argument("--archive", type=Path, required=True)
    pub.add_argument("--tag", required=True)
    args = p.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        raise SystemExit("ZENODO_TOKEN is not configured")

    api = (args.api or DEFAULT_API).strip() or DEFAULT_API
    api = api.rstrip("/")
    if args.cmd == "reserve":
        return reserve(token, api, args.github_output)
    return publish(token, api, args.deposit_id, args.archive, args.tag, args.github_output)


if __name__ == "__main__":
    raise SystemExit(main())
