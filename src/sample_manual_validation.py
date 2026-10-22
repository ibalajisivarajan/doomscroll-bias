#!/usr/bin/env python3
"""Create the preregistered 50-response manual validation sample.

Sampling is deterministic, seeded, and stratified equally across model slots.
Two identical, independent coding worksheets are produced -- one per human
coder -- so Cohen's kappa can be computed between two independent human
raters (the primary reliability statistic). Neither worksheet contains the
automated action_category, so neither coder is anchored by the classifier.
A separate key file retains the automated labels for a secondary
human-vs-automated agreement check.
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

WORKSHEET_FIELDS = [
    "sample_id", "model_slot", "vignette_id", "gender", "culture", "run_index",
    "recommended_action", "action_category", "coder_notes"
]
KEY_FIELDS = ["sample_id", "cache_key", "automated_action_category"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", type=Path, default=ROOT / "results/tables/scored.csv")
    p.add_argument("--worksheet-a", type=Path,
                    default=ROOT / "results/tables/manual_validation_50_coder_a.csv")
    p.add_argument("--worksheet-b", type=Path,
                    default=ROOT / "results/tables/manual_validation_50_coder_b.csv")
    p.add_argument("--key", type=Path, default=ROOT / "results/tables/manual_validation_50_key.csv")
    args = p.parse_args()
    if not args.scored.exists():
        print(f"manual validation sample skipped: {args.scored} does not exist")
        return 0

    # Never regenerate once either coder worksheet already has labels: that
    # would erase in-progress or completed human coding.
    for existing in (args.worksheet_a, args.worksheet_b):
        if existing.exists():
            with existing.open(encoding="utf-8", newline="") as fh:
                if any((r.get("action_category") or "").strip() for r in csv.DictReader(fh)):
                    print(f"manual validation sample skipped: {existing} already has coded rows")
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

    args.worksheet_a.parent.mkdir(parents=True, exist_ok=True)
    with args.worksheet_a.open("w", encoding="utf-8", newline="") as wa, \
         args.worksheet_b.open("w", encoding="utf-8", newline="") as wb, \
         args.key.open("w", encoding="utf-8", newline="") as kf:
        writer_a = csv.DictWriter(wa, fieldnames=WORKSHEET_FIELDS); writer_a.writeheader()
        writer_b = csv.DictWriter(wb, fieldnames=WORKSHEET_FIELDS); writer_b.writeheader()
        kw = csv.DictWriter(kf, fieldnames=KEY_FIELDS); kw.writeheader()
        for idx, row in enumerate(sample, 1):
            sid = f"MV{idx:03d}"
            shared = {
                "sample_id": sid,
                "model_slot": row.get("model_slot", ""),
                "vignette_id": row.get("vignette_id", ""),
                "gender": row.get("gender", ""),
                "culture": row.get("culture", ""),
                "run_index": row.get("run_index", ""),
                "recommended_action": row.get("recommended_action", ""),
                "action_category": "",
                "coder_notes": "",
            }
            writer_a.writerow(shared)
            writer_b.writerow(dict(shared))
            kw.writerow({
                "sample_id": sid,
                "cache_key": row.get("cache_key", ""),
                "automated_action_category": row.get("action_category", ""),
            })
    print(f"wrote {len(sample)} seeded manual-validation rows (seed={SEED}) "
          f"to two independent coder worksheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
