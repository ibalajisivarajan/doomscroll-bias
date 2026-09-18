#!/usr/bin/env python3
"""Compare the 50 blinded human action labels with automated categories."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSHEET = ROOT / "results/tables/manual_validation_50.csv"
KEY = ROOT / "results/tables/manual_validation_50_key.csv"
OUT = ROOT / "results/final"
CATS = ["seek_help", "boundary", "accommodate", "negotiate", "unclassified"]


def main() -> int:
    with WORKSHEET.open(encoding="utf-8", newline="") as fh:
        human = {r["sample_id"]: r["human_action_category"].strip() for r in csv.DictReader(fh)}
    with KEY.open(encoding="utf-8", newline="") as fh:
        auto = {r["sample_id"]: r["automated_action_category"].strip() for r in csv.DictReader(fh)}

    ids = sorted(set(human) & set(auto))
    if len(ids) != 50:
        raise SystemExit(f"expected 50 matched validation rows, found {len(ids)}")
    invalid = [(sid, human[sid]) for sid in ids if human[sid] not in CATS]
    if invalid:
        raise SystemExit(f"manual validation has invalid/unfilled labels: {invalid[:5]}")

    pairs = [(human[sid], auto[sid]) for sid in ids]
    exact = sum(a == b for a, b in pairs)
    n = len(pairs)
    po = exact / n
    hc, ac = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((hc[c] / n) * (ac[c] / n) for c in CATS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    OUT.mkdir(parents=True, exist_ok=True)
    confusion = OUT / "manual_validation_confusion.csv"
    with confusion.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["human\\automated"] + CATS)
        for h in CATS:
            w.writerow([h] + [sum(1 for x, y in pairs if x == h and y == a) for a in CATS])

    summary = {
        "n": n,
        "exact_agreement_n": exact,
        "exact_agreement_rate": po,
        "cohens_kappa": kappa,
        "categories": CATS,
    }
    (OUT / "manual_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "manual_validation_summary.md").write_text(
        "# Manual validation agreement\n\n"
        f"- N: **{n}**\n"
        f"- Exact agreement: **{exact}/{n} ({po:.1%})**\n"
        f"- Cohen's kappa: **{kappa:.3f}**\n\n"
        "See manual_validation_confusion.csv for the full confusion matrix.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
