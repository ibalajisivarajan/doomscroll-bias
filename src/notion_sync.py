#!/usr/bin/env python3
"""Update the study's Notion dashboard and append an immutable run-history line.

GitHub/results remain the source of truth. Notion is display-only. Each sync:
1. computes collection/QC state from files on disk;
2. updates a single dashboard marker paragraph when the integration has update
   permission, or creates it if it does not exist;
3. appends a timestamped history paragraph so every dispatch remains visible.

The workflow marks this step continue-on-error, so Notion can never jeopardize
collected research data.
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
MARKER = "DOOMSCROLL_DASHBOARD|"


class NotionError(RuntimeError):
    pass


def normalise_page_id(raw: str) -> str:
    text = raw.strip()
    hexes = re.findall(r"[0-9a-fA-F]{32}", text.replace("-", ""))
    if not hexes:
        raise NotionError(f"could not find a 32-character page id in NOTION_PAGE_ID={raw!r}")
    h = hexes[-1].lower()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def git_describe() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def read_control_summary(paths: dict[str, Any]) -> str:
    path = ROOT / paths.get("control_summary", "results/tables/control_summary.json")
    if not path.exists():
        return "pending"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rate = data.get("pass_rate")
        n = data.get("n_scorable", 0)
        return "pending" if rate is None else f"{rate:.1%} (n={n})"
    except (OSError, ValueError, TypeError):
        return "unreadable"


def summarise(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    raw_dir = ROOT / paths.get("raw", "results/raw")
    scored = ROOT / paths.get("scored_csv", "results/tables/scored.csv")
    target = int(config["design"].get("total_calls", 0))
    cached = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0

    parsed = total = 0
    slot_counts: dict[str, int] = {}
    quoted = grounded = 0
    if scored.exists():
        with scored.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                total += 1
                slot = row.get("model_slot", "?")
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
                if row.get("parse_ok") == "True": parsed += 1
                mode = row.get("quote_match_mode", "")
                if mode and mode != "empty":
                    quoted += 1
                    grounded += int(row.get("quote_verbatim") == "True")

    if cached == 0:
        stage = "Pre-production readiness" if not config.get("locked") else "Ready for production"
    elif target and cached < target:
        stage = "Collecting"
    else:
        stage = "Collection complete / analysis"

    return {
        "stage": stage,
        "locked": bool(config.get("locked")),
        "cached": cached,
        "target": target,
        "pct": (cached / target * 100.0) if target else 0.0,
        "scored": total,
        "parsed": parsed,
        "hallucination": (1 - grounded / quoted) if quoted else None,
        "slot_counts": slot_counts,
        "control_pass": read_control_summary(paths),
        "models": ", ".join(f"{m['slot']}={m.get('id','TBD')}" for m in config.get("models", [])),
        "commit": git_describe(),
        "updated": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    }


def dashboard_text(s: dict[str, Any]) -> str:
    hall = "pending" if s["hallucination"] is None else f"{s['hallucination']:.3f}"
    slots = ", ".join(f"{k}={v}" for k, v in sorted(s["slot_counts"].items())) or "pending"
    return (
        f"{MARKER} Stage={s['stage']} | Protocol locked={s['locked']} | "
        f"Progress={s['cached']}/{s['target']} ({s['pct']:.1f}%) | "
        f"Scored={s['scored']} | Parsed={s['parsed']} | By model={slots} | "
        f"Control pass={s['control_pass']} | Hallucination={hall} | "
        f"Commit={s['commit']} | Updated={s['updated']} | Models={s['models']}"
    )


def history_text(s: dict[str, Any], note: str) -> str:
    text = dashboard_text(s).replace(MARKER, "RUN_HISTORY|")
    return text + (f" | Note={note}" if note else "")


def rich_text(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": text[:2000]}}]


def paragraph(text: str) -> dict[str, Any]:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text(text)}}


def block_plain_text(block: dict[str, Any]) -> str:
    typ = block.get("type")
    payload = block.get(typ, {}) if typ else {}
    return "".join(x.get("plain_text", "") for x in payload.get("rich_text", []))


def request(method: str, url: str, token: str, **kwargs: Any) -> requests.Response:
    try:
        resp = requests.request(method, url, headers=headers(token), timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise NotionError(f"could not reach Notion: {exc}") from exc
    if resp.status_code >= 400:
        raise NotionError(f"Notion returned {resp.status_code}: {resp.text[:300]}")
    return resp


def find_dashboard_block(page_id: str, token: str) -> str | None:
    cursor: str | None = None
    for _ in range(10):
        params = {"page_size": 100}
        if cursor: params["start_cursor"] = cursor
        body = request("GET", f"{NOTION_API}/blocks/{page_id}/children", token, params=params).json()
        for block in body.get("results", []):
            if block_plain_text(block).startswith(MARKER):
                return block.get("id")
        if not body.get("has_more"): return None
        cursor = body.get("next_cursor")
    return None


def sync(page_id: str, token: str, dashboard: str, history: str) -> str:
    block_id = find_dashboard_block(page_id, token)
    if block_id:
        request("PATCH", f"{NOTION_API}/blocks/{block_id}", token,
                json={"paragraph": {"rich_text": rich_text(dashboard)}})
        action = "updated dashboard"
    else:
        request("PATCH", f"{NOTION_API}/blocks/{page_id}/children", token,
                json={"children": [paragraph(dashboard)]})
        action = "created dashboard"
    request("PATCH", f"{NOTION_API}/blocks/{page_id}/children", token,
            json={"children": [paragraph(history)]})
    return action + " and appended run history"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--note", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = summarise(config)
    dash, hist = dashboard_text(state), history_text(state, args.note)
    if args.dry_run:
        print(dash); print(hist); return 0
    token = os.environ.get(config.get("notion", {}).get("token_env", "NOTION_TOKEN"), "").strip()
    raw_page = os.environ.get(config.get("notion", {}).get("page_id_env", "NOTION_PAGE_ID"), "").strip()
    if not token or not raw_page:
        print("error: NOTION_TOKEN/NOTION_PAGE_ID not configured; skipping dashboard sync", file=sys.stderr)
        return 1
    try:
        action = sync(normalise_page_id(raw_page), token, dash, hist)
    except NotionError as exc:
        print(f"error: {exc}", file=sys.stderr); return 1
    print(action); print(dash)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
