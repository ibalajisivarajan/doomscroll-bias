#!/usr/bin/env python3
"""Prepare and freeze the study once all 7,200 production responses exist.

This script is idempotent. Before 7,200 responses it reports "not complete"
and exits cleanly. At 7,200 it reruns deterministic scoring/QC/analysis,
regenerates the seeded 50-response manual-validation worksheet, writes SHA-256
manifests for the raw corpus and frozen study inputs, and reports whether the
human-coded validation gate is complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "protocol.yaml"
FINAL_DIR = ROOT / "results" / "final"
MANUAL = ROOT / "results" / "tables" / "manual_validation_50.csv"
VALID_ACTIONS = {"seek_help", "boundary", "accommodate", "negotiate", "unclassified"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=ROOT, check=True)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True
        ).stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def manual_status() -> tuple[bool, int, list[str]]:
    if not MANUAL.exists():
        return False, 0, ["manual-validation worksheet does not exist"]
    with MANUAL.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    problems: list[str] = []
    if len(rows) != 50:
        problems.append(f"expected 50 rows, found {len(rows)}")
    coded = 0
    seen: set[str] = set()
    for row in rows:
        sid = (row.get("sample_id") or "").strip()
        label = (row.get("human_action_category") or "").strip()
        if sid in seen:
            problems.append(f"duplicate sample_id {sid}")
        seen.add(sid)
        if label:
            coded += 1
            if label not in VALID_ACTIONS:
                problems.append(f"{sid}: invalid human_action_category={label!r}")
    if coded != 50:
        problems.append(f"manual coding incomplete: {coded}/50 labelled")
    return not problems, coded, problems


def write_output(path: str | None, key: str, value: str) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{key}={value}\n")


def prepare(github_output: str | None = None) -> int:
    cfg = load_config()
    raw_dir = ROOT / cfg.get("paths", {}).get("raw", "results/raw")
    files = sorted(raw_dir.glob("*.json"))
    target = int(cfg["design"]["total_calls"])
    count = len(files)
    print(f"production cache: {count}/{target}")
    write_output(github_output, "complete", "true" if count == target else "false")
    write_output(github_output, "raw_count", str(count))

    if count < target:
        print("collection is not complete; finalization is not due yet")
        return 0
    if count > target:
        raise SystemExit(f"refusing to finalize: found {count} raw responses, expected exactly {target}")
    if cfg.get("locked") is not True:
        raise SystemExit("refusing to finalize an unlocked protocol")

    run(sys.executable, "src/score.py")
    run(sys.executable, "src/control_check.py")
    run(sys.executable, "src/sample_manual_validation.py")
    run(sys.executable, "src/analyze.py")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in files]
    (FINAL_DIR / "raw_sha256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    dataset = ROOT / cfg.get("paths", {}).get("vignettes", "data/vignettes.jsonl")
    inputs = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head_at_freeze": git_head(),
        "raw_response_count": count,
        "expected_response_count": target,
        "dataset": {
            "path": dataset.relative_to(ROOT).as_posix(),
            "sha256": sha256(dataset),
        },
        "protocol": {
            "path": CONFIG.relative_to(ROOT).as_posix(),
            "sha256": sha256(CONFIG),
            "locked": True,
        },
        "raw_manifest": "results/final/raw_sha256.txt",
        "raw_manifest_sha256": sha256(FINAL_DIR / "raw_sha256.txt"),
    }
    (FINAL_DIR / "collection_manifest.json").write_text(
        json.dumps(inputs, indent=2) + "\n", encoding="utf-8"
    )

    ready, coded, problems = manual_status()
    status = [
        "# Finalization status",
        "",
        f"- Collection: **{count}/{target} complete**",
        "- Raw SHA-256 manifest: **written**",
        "- Final scoring/QC/analysis: **rerun from the complete frozen corpus**",
        f"- Manual validation: **{coded}/50 coded**",
    ]
    if ready:
        status.append("- Release gate: **READY**")
    else:
        status.append("- Release gate: **WAITING FOR HUMAN CODING**")
        status.append("")
        status.append("Complete human_action_category in results/tables/manual_validation_50.csv.")
        if problems:
            status += ["", "Current gate notes:"] + [f"- {p}" for p in problems]
    (FINAL_DIR / "STATUS.md").write_text("\n".join(status) + "\n", encoding="utf-8")

    write_output(github_output, "manual_ready", "true" if ready else "false")
    write_output(github_output, "manual_coded", str(coded))
    print(f"manual validation: {coded}/50; release_ready={ready}")
    return 0


def write_release_metadata(doi: str, zenodo_url: str, tag: str) -> int:
    doi = doi.strip()
    if not doi:
        raise SystemExit("DOI is required")
    cfg = load_config()
    paper = ROOT / "paper"
    paper.mkdir(exist_ok=True)
    osf = "https://osf.io/ndvw8/"
    github_release = f"https://github.com/ibalajisivarajan/doomscroll-bias/releases/tag/{tag}"
    text = f"""# Final release metadata

- **Study:** Gendered and Cultural Name-Cue Effects in LLM Judgments of Doomscrolling-Related Relationship Conflict
- **Final response count:** {cfg["design"]["total_calls"]}
- **GitHub release:** {github_release}
- **Zenodo DOI:** https://doi.org/{doi}
- **Zenodo record:** {zenodo_url}
- **OSF preregistration:** {osf}

The OSF registration remains immutable. This file links the frozen preregistration
to the final public code/data/results release.
"""
    (paper / "RELEASE_METADATA.md").write_text(text, encoding="utf-8")

    cff = ROOT / "CITATION.cff"
    cff_text = cff.read_text(encoding="utf-8")
    lines = [line for line in cff_text.splitlines() if not line.startswith("doi:")]
    insert_at = next((i for i, line in enumerate(lines) if line.startswith("repository-code:")), len(lines))
    lines.insert(insert_at, f'doi: "{doi}"')
    cff.write_text("\n".join(lines) + "\n", encoding="utf-8")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "release.json").write_text(json.dumps({
        "tag": tag,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "zenodo_url": zenodo_url,
        "github_release": github_release,
        "osf_preregistration": osf,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--github-output")
    meta = sub.add_parser("write-release-metadata")
    meta.add_argument("--doi", required=True)
    meta.add_argument("--zenodo-url", required=True)
    meta.add_argument("--tag", required=True)
    args = p.parse_args()
    if args.cmd == "prepare":
        return prepare(args.github_output)
    return write_release_metadata(args.doi, args.zenodo_url, args.tag)


if __name__ == "__main__":
    raise SystemExit(main())
