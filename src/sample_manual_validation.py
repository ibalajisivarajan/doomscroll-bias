#!/usr/bin/env python3
"""Create the preregistered 50-response manual validation sample.

Sampling is deterministic, seeded, and stratified equally across model slots.
The output intentionally omits the automated action_category from the coding
worksheet so the human coder is not anchored by the classifier. A separate key
file retains the automated labels for later agreement calculation.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260914
N_TOTAL = 50


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", type=Path, default=ROOT / "results/tables/scored.csv")
    p.add_argument("--worksheet", type=Path, default=ROOT / "results/tables/manual_validation_50.csv")
    p.add_argument("--key", type=Path, default=ROOT / "results/tables/manual_validation_50_key.csv")
    args = p.parse_args()
    if not args.scored.exists():
        print(f"manual validation sample skipped: {args.scored} does not exist")
        return 0

    with args.scored.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("parse_ok") == "True" and not r.get("missing_keys")]
    by_slot: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_slot[row.get("model_slot", "?")].append(row)
    slots = sorted(by_slot)
    if not slots:
        print("manual validation sample skipped: no scorable rows")
        return 0

    rng = random.Random(SEED)
    base = N_TOTAL // len(slots)
    remainder = N_TOTAL % len(slots)
    sample: list[dict[str, str]] = []
    for i, slot in enumerate(slots):
        n = base + (1 if i < remainder else 0)
        pool = sorted(by_slot[slot], key=lambda r: r.get("cache_key", ""))
        sample.extend(rng.sample(pool, min(n, len(pool))))
    sample.sort(key=lambda r: (r.get("model_slot", ""), r.get("cache_key", "")))

    args.worksheet.parent.mkdir(parents=True, exist_ok=True)
    worksheet_fields = [
        "sample_id","model_slot","vignette_id","gender","culture","run_index",
        "recommended_action","human_action_category","coder_notes"
    ]
    key_fields = ["sample_id","cache_key","automated_action_category"]
    with args.worksheet.open("w", encoding="utf-8", newline="") as wf, args.key.open("w", encoding="utf-8", newline="") as kf:
        ww = csv.DictWriter(wf, fieldnames=worksheet_fields); ww.writeheader()
        kw = csv.DictWriter(kf, fieldnames=key_fields); kw.writeheader()
        for idx, row in enumerate(sample, 1):
            sid = f"MV{idx:03d}"
            ww.writerow({
                "sample_id": sid,
                "model_slot": row.get("model_slot", ""),
                "vignette_id": row.get("vignette_id", ""),
                "gender": row.get("gender", ""),
                "culture": row.get("culture", ""),
                "run_index": row.get("run_index", ""),
                "recommended_action": row.get("recommended_action", ""),
                "human_action_category": "",
                "coder_notes": "",
            })
            kw.writerow({
                "sample_id": sid,
                "cache_key": row.get("cache_key", ""),
                "automated_action_category": row.get("action_category", ""),
            })
    print(f"wrote {len(sample)} seeded manual-validation rows (seed={SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
