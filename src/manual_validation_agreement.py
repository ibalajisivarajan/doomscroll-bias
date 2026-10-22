#!/usr/bin/env python3
"""Compute manual-validation agreement for the 50-row blinded sample.

Primary statistic: Cohen's kappa between the two independent human coders
(coder A and coder B), who labelled the same 50 rows blind to each other and
blind to the automated classifier. This is the preregistered inter-rater
reliability check.

Secondary statistic: each human coder's agreement with the automated
action-category classifier, reported for reference but not presented as
inter-rater reliability.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSHEET_A = ROOT / "results/tables/manual_validation_50_coder_a.csv"
WORKSHEET_B = ROOT / "results/tables/manual_validation_50_coder_b.csv"
KEY = ROOT / "results/tables/manual_validation_50_key.csv"
OUT = ROOT / "results/final"
CATS = ["seek_help", "boundary", "accommodate", "negotiate", "unclassified"]


def _load_labels(path: Path, column: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["sample_id"]: r[column].strip() for r in csv.DictReader(fh)}


def _validate(labels: dict[str, str], label_name: str) -> None:
    if len(labels) != 50:
        raise SystemExit(f"{label_name}: expected 50 rows, found {len(labels)}")
    invalid = [(sid, v) for sid, v in labels.items() if v not in CATS]
    if invalid:
        raise SystemExit(f"{label_name}: invalid/unfilled labels present: {invalid[:5]}")


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, int]:
    n = len(pairs)
    exact = sum(a == b for a, b in pairs)
    po = exact / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in CATS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return kappa, po, exact


def write_confusion(path: Path, pairs: list[tuple[str, str]], row_label: str, col_label: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"{row_label}\\{col_label}"] + CATS)
        for r in CATS:
            w.writerow([r] + [sum(1 for x, y in pairs if x == r and y == c) for c in CATS])


def main() -> int:
    coder_a = _load_labels(WORKSHEET_A, "action_category")
    coder_b = _load_labels(WORKSHEET_B, "action_category")
    auto = {r["sample_id"]: r["automated_action_category"].strip()
            for r in csv.DictReader(KEY.open(encoding="utf-8", newline=""))}

    _validate(coder_a, "coder A worksheet")
    _validate(coder_b, "coder B worksheet")

    ids_ab = sorted(set(coder_a) & set(coder_b))
    if len(ids_ab) != 50:
        raise SystemExit(f"expected 50 matched coder A/B rows, found {len(ids_ab)}")
    ids_auto = sorted(set(coder_a) & set(auto))
    if len(ids_auto) != 50:
        raise SystemExit(f"expected 50 matched coder/automated rows, found {len(ids_auto)}")

    OUT.mkdir(parents=True, exist_ok=True)

    # Primary: inter-rater reliability between the two independent human coders.
    ab_pairs = [(coder_a[sid], coder_b[sid]) for sid in ids_ab]
    kappa_ab, po_ab, exact_ab = cohens_kappa(ab_pairs)
    write_confusion(OUT / "manual_validation_confusion_ab.csv", ab_pairs, "coder_a", "coder_b")

    # Secondary: each coder vs. the automated classifier, for reference only.
    a_auto_pairs = [(coder_a[sid], auto[sid]) for sid in ids_auto]
    b_auto_pairs = [(coder_b[sid], auto[sid]) for sid in ids_auto]
    kappa_a_auto, po_a_auto, exact_a_auto = cohens_kappa(a_auto_pairs)
    kappa_b_auto, po_b_auto, exact_b_auto = cohens_kappa(b_auto_pairs)
    write_confusion(OUT / "manual_validation_confusion_a_auto.csv", a_auto_pairs, "coder_a", "automated")
    write_confusion(OUT / "manual_validation_confusion_b_auto.csv", b_auto_pairs, "coder_b", "automated")

    summary = {
        "n": 50,
        "categories": CATS,
        "primary_inter_rater": {
            "description": "Cohen's kappa between the two independent human coders (A and B), "
                            "each blind to the other and to the automated classifier.",
            "exact_agreement_n": exact_ab,
            "exact_agreement_rate": po_ab,
            "cohens_kappa": kappa_ab,
        },
        "secondary_classifier_agreement": {
            "description": "Each human coder vs. the automated action-category classifier. "
                            "Reported for reference only; not inter-rater reliability.",
            "coder_a_vs_automated": {
                "exact_agreement_n": exact_a_auto,
                "exact_agreement_rate": po_a_auto,
                "cohens_kappa": kappa_a_auto,
            },
            "coder_b_vs_automated": {
                "exact_agreement_n": exact_b_auto,
                "exact_agreement_rate": po_b_auto,
                "cohens_kappa": kappa_b_auto,
            },
        },
    }
    (OUT / "manual_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "manual_validation_summary.md").write_text(
        "# Manual validation agreement\n\n"
        "## Primary: inter-rater reliability (coder A vs. coder B)\n\n"
        f"- N: **50**\n"
        f"- Exact agreement: **{exact_ab}/50 ({po_ab:.1%})**\n"
        f"- Cohen's kappa: **{kappa_ab:.3f}**\n\n"
        "See manual_validation_confusion_ab.csv for the full confusion matrix.\n\n"
        "## Secondary: human vs. automated classifier (reference only)\n\n"
        f"- Coder A vs. automated: {exact_a_auto}/50 ({po_a_auto:.1%}), kappa = {kappa_a_auto:.3f}\n"
        f"- Coder B vs. automated: {exact_b_auto}/50 ({po_b_auto:.1%}), kappa = {kappa_b_auto:.3f}\n\n"
        "These are not inter-rater reliability measures and should not be reported as such.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
