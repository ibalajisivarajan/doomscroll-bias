#!/usr/bin/env python3
"""Append a run-log entry to the study's Notion page. Write-only, one way.

Notion is the dashboard, not a data store. Nothing in the analysis path ever
reads from it: `config/protocol.yaml` is the source of truth for the design and
`results/` is the source of truth for the data, so if this page were deleted
tomorrow the study would lose nothing but its human-readable log. That is
deliberate -- a two-way sync would make a Notion edit capable of silently
changing what the analysis believes, and there would be no commit trail for it.

SETUP (once)
------------
1. Create the integration.
   https://www.notion.so/my-integrations -> "New integration".
   Give it a name, pick the workspace, and grant *Insert content*. It does not
   need read or update capability -- this script only appends.

2. Copy the Internal Integration Secret (it starts with "ntn_" or "secret_")
   and export it as NOTION_TOKEN. In GitHub, add it as the repository secret
   NOTION_TOKEN.

3. Create the page that will hold the run log. Copy its id from the URL:

       https://www.notion.so/My-Study-Log-1a2b3c4d5e6f7080b1c2d3e4f5061728
                                          ^-- this 32-char hex string

   Export it as NOTION_PAGE_ID (dashes optional -- this script normalises it).
   In GitHub, add it as the repository secret NOTION_PAGE_ID.

4. *** SHARE THE PAGE WITH THE INTEGRATION. *** This is the step everybody
   misses, including me, twice. Creating the integration does not give it
   access to anything. Open the page, click the "..." menu at the top right,
   choose **Connections** (older UI: "Add connections" / "Share"), and pick
   your integration by name.

   Until you do this, the API returns **404 Not Found** for the page -- not
   403, not "unauthorized". A 404 here almost never means a bad token or a
   mistyped id; it means the integration cannot see the page. The error this
   script raises on a 404 says so, because reading it as "wrong page id" costs
   an hour of checking a page id that was right all along.

Usage:
    NOTION_TOKEN=... NOTION_PAGE_ID=... python src/notion_sync.py
    python src/notion_sync.py --note "first smoke test on the new endpoint"
    python src/notion_sync.py --dry-run     # print the block, send nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "protocol.yaml"

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
# Notion rejects any single rich_text item longer than 2000 characters, and
# does it with a 400 that does not name the offending field.
RICH_TEXT_LIMIT = 2000


class NotionError(RuntimeError):
    """Raised with an operator-readable explanation, not a raw API dump."""


def normalise_page_id(raw: str) -> str:
    """Accept a bare id, a dashed uuid, or a pasted page URL.

    People paste whichever of the three is on their clipboard; rejecting two of
    them is a pointless failure mode.
    """
    text = raw.strip()
    hexes = re.findall(r"[0-9a-fA-F]{32}", text.replace("-", ""))
    if not hexes:
        raise NotionError(
            f"could not find a 32-character page id in NOTION_PAGE_ID={raw!r}.\n"
            "  Open the page in Notion and copy the id from the URL: "
            "https://www.notion.so/Title-<32 hex chars>"
        )
    h = hexes[-1].lower()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# building the log entry
# ---------------------------------------------------------------------------
def git_describe() -> str:
    """Short commit sha plus dirty marker, or "unknown" outside a checkout.

    The sha is the whole point of the log line: it ties a set of numbers to the
    exact protocol and code that produced them.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def summarise(config: dict[str, Any]) -> str:
    """One paragraph describing the state of the collection right now.

    Deliberately computed from the files on disk rather than passed in by the
    caller, so the log cannot claim a run finished that did not.
    """
    paths = config.get("paths", {})
    raw_dir = ROOT / paths.get("raw", "results/raw")
    scored = ROOT / paths.get("scored_csv", "results/tables/scored.csv")
    design = config["design"]

    cached = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    target = int(design.get("total_calls", 0))
    pct = f"{cached / target * 100:.1f}%" if target else "n/a"

    parts = [
        f"run-log {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
        f"commit {git_describe()}",
        f"protocol locked={config.get('locked')}",
        f"cached responses {cached}/{target} ({pct})",
    ]

    if scored.exists():
        total = grounded = quoted = parsed = 0
        slots: dict[str, list[int]] = {}
        with scored.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                total += 1
                if row.get("parse_ok") == "True":
                    parsed += 1
                mode = row.get("quote_match_mode", "")
                if mode and mode != "empty":
                    quoted += 1
                    ok = row.get("quote_verbatim") == "True"
                    grounded += int(ok)
                    bucket = slots.setdefault(row.get("model_slot", "?"), [0, 0])
                    bucket[0] += 1
                    bucket[1] += int(ok)
        parts.append(f"scored {total} rows, parsed {parsed}")
        if quoted:
            parts.append(f"hallucination rate {1 - grounded / quoted:.3f}")
            for slot, (n, ok) in sorted(slots.items()):
                parts.append(f"slot {slot} hallucination {1 - ok / n:.3f} (n={n})")
    else:
        parts.append("no scored.csv yet")

    models = ", ".join(
        f"{m['slot']}={m.get('id', 'TBD')}" for m in config.get("models", [])
    )
    parts.append(f"models {models}")
    return " | ".join(parts)


def paragraph_block(text: str) -> dict[str, Any]:
    """A single paragraph block, chunked to Notion's rich_text length limit."""
    chunks = [
        text[i : i + RICH_TEXT_LIMIT] for i in range(0, max(len(text), 1), RICH_TEXT_LIMIT)
    ] or [""]
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": chunk}} for chunk in chunks
            ]
        },
    }


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------
def append_blocks(
    page_id: str, token: str, blocks: list[dict[str, Any]], timeout: float = 30.0
) -> dict[str, Any]:
    """PATCH /v1/blocks/{page_id}/children -- append, never replace.

    PATCH with `children` appends to the end of the page's block list, so the
    log grows and nothing already on the page is touched. There is no call in
    this module that can delete or overwrite anything.
    """
    url = f"{NOTION_API}/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    try:
        resp = requests.patch(
            url, headers=headers, json={"children": blocks}, timeout=timeout
        )
    except requests.RequestException as exc:
        raise NotionError(f"could not reach the Notion API: {exc}") from exc

    if resp.status_code == 404:
        # The message that saves the hour. Notion returns 404 rather than 403
        # for a page the integration has not been granted access to, so the
        # status code points at the wrong cause by default.
        raise NotionError(
            f"Notion returned 404 for page {page_id}.\n"
            "\n"
            "  This is almost certainly NOT a bad token and NOT a mistyped "
            "page id.\n"
            "  Notion answers 404 -- not 403 -- when the integration has not "
            "been given\n"
            "  access to the page, so an un-shared page is indistinguishable "
            "from a\n"
            "  missing one from the outside.\n"
            "\n"
            "  Fix: open the page in Notion, click the '...' menu at the top "
            "right,\n"
            "  choose 'Connections' (older UI: 'Add connections'), and select "
            "this\n"
            "  integration by name. Then re-run.\n"
            "\n"
            "  Only if the page is definitely shared: check that NOTION_PAGE_ID "
            "is the\n"
            "  page you shared, and that NOTION_TOKEN belongs to the same "
            "workspace.\n"
            f"\n  API said: {resp.text[:300]}"
        )
    if resp.status_code == 401:
        raise NotionError(
            "Notion returned 401 Unauthorized: NOTION_TOKEN is missing, "
            "expired, or revoked.\n"
            "  Regenerate the Internal Integration Secret at "
            "https://www.notion.so/my-integrations\n"
            f"  API said: {resp.text[:300]}"
        )
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "unknown")
        raise NotionError(
            f"Notion rate-limited the request (429); Retry-After={retry_after}s. "
            "The run log is not on the critical path -- re-run the sync later."
        )
    if resp.status_code >= 400:
        raise NotionError(
            f"Notion returned {resp.status_code}: {resp.text[:400]}"
        )
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--note", default="", help="free-text note appended to the log entry"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the block that would be sent and exit",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    notion_cfg = config.get("notion", {})
    token_env = notion_cfg.get("token_env", "NOTION_TOKEN")
    page_env = notion_cfg.get("page_id_env", "NOTION_PAGE_ID")

    text = summarise(config)
    if args.note:
        text = f"{text} | note: {args.note}"
    block = paragraph_block(text)

    if args.dry_run:
        print(json.dumps(block, indent=2))
        return 0

    token = os.environ.get(token_env, "").strip()
    raw_page = os.environ.get(page_env, "").strip()
    missing = [
        name for name, value in ((token_env, token), (page_env, raw_page)) if not value
    ]
    if missing:
        # Non-zero, but the caller (the workflow) marks this step
        # continue-on-error: a dashboard that is not configured must never fail
        # a collection run that already succeeded.
        print(
            f"error: {' and '.join(missing)} not set in the environment; "
            "skipping the Notion sync.\n"
            "  See the setup steps in the docstring at the top of this file.",
            file=sys.stderr,
        )
        return 1

    try:
        page_id = normalise_page_id(raw_page)
        append_blocks(page_id, token, [block])
    except NotionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"appended run-log paragraph to Notion page {page_id}")
    print(f"  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
