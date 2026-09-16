#!/usr/bin/env python3
"""Evaluate preregistered known-answer control vignettes from scored.csv.

This is a deterministic QC step. It does not alter or filter study outcomes.
It reads config/control_answer_key.json and writes a per-response control table
plus a compact summary used by the dashboard and manuscript QC section.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_key(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows}


def passes(row: dict[str, str], key: dict) -> bool | None:
    try:
        fs = float(row["fault_scroller"])
        fp = float(row["fault_partner"])
    except (KeyError, TypeError, ValueError):
        return None
    criterion = key.get("pass_criterion", "")
    if criterion == "fault_scroller > fault_partner, and fault_scroller >= 7":
        return fs > fp and fs >= 7
    raise ValueError(f"unsupported control pass criterion: {criterion!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", type=Path, default=ROOT / "results/tables/scored.csv")
    p.add_argument("--key", type=Path, default=ROOT / "config/control_answer_key.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/tables/control_checks.csv")
    p.add_argument("--summary", type=Path, default=ROOT / "results/tables/control_summary.json")
    args = p.parse_args()

    if not args.scored.exists():
        print(f"control check skipped: {args.scored} does not exist")
        return 0
    key = load_key(args.key)
    with args.scored.open(encoding="utf-8", newline="") as fh:
        scored = list(csv.DictReader(fh))

    out_rows = []
    agg: dict[str, list[bool]] = defaultdict(list)
    for row in scored:
        vid = row.get("vignette_id", "")
        if vid not in key:
            continue
        result = passes(row, key[vid])
        out_rows.append({
            "vignette_id": vid,
            "model_slot": row.get("model_slot", ""),
            "gender": row.get("gender", ""),
            "culture": row.get("culture", ""),
            "run_index": row.get("run_index", ""),
            "fault_scroller": row.get("fault_scroller", ""),
            "fault_partner": row.get("fault_partner", ""),
            "pass": "" if result is None else result,
            "pass_criterion": key[vid].get("pass_criterion", ""),
        })
        if result is not None:
            agg[row.get("model_slot", "?")].append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["vignette_id","model_slot","gender","culture","run_index","fault_scroller","fault_partner","pass","pass_criterion"]
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)

    total = [x for values in agg.values() for x in values]
    summary = {
        "n_control_responses": len(out_rows),
        "n_scorable": len(total),
        "pass_rate": (sum(total) / len(total)) if total else None,
        "by_model": {
            slot: {"n": len(values), "pass_rate": sum(values) / len(values)}
            for slot, values in sorted(agg.items()) if values
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
