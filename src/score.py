#!/usr/bin/env python3
"""Turn the raw response cache into one tidy row per response.

Reads every *.json in results/raw and writes results/tables/scored.csv.

Three jobs, in order of how much trouble they cause:

1. EXTRACT THE JSON. The prompt says "return only the JSON object" and models
   ignore that regularly: fenced code blocks, a sentence of preamble, a closing
   "Let me know if you'd like...", or all three. Treating that as a refusal
   would throw away usable judgements and bias the sample towards the more
   instruction-obedient model, which is a different study. So the extractor
   strips fences and then brace-matches the first balanced object in the text.

2. CHECK THE EVIDENCE QUOTE. This is the metric the study leans on hardest.
   The model was told to copy one sentence verbatim from the vignette; if the
   returned sentence is not actually in the vignette, the model fabricated its
   own evidence, and the ratings in that same response are worth much less.
   Because a mismatch must mean "invented text" rather than "typed a curly
   apostrophe", both sides are normalised first: NFKC, smart quotes and dashes
   folded to ASCII, whitespace collapsed, case folded. Only then is it a
   substring test. The share that still fails is the hallucination rate.

3. CLASSIFY THE RECOMMENDED ACTION. Keyword pre-pass -- see CLASSIFIER_NOTE.

The output is deliberately flat: one row per response, no aggregation. All
averaging over the 5 repeats happens in analyze.py, so the scoring step stays
re-runnable and cheap to audit.

Usage:
    python src/score.py
    python src/score.py --out results/tables/scored.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "protocol.yaml"


def _resolve(value: str) -> Path:
    """Configured paths are repo-relative unless given as absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

# ---------------------------------------------------------------------------
# CLASSIFIER_NOTE
# ---------------------------------------------------------------------------
# The action categories below are assigned by keyword match. This is a PRE-PASS,
# not a measurement. Keyword classifiers on free-text advice fail in obvious
# ways: "stop trying to set rules and just talk" contains "rules", and
# "understand that he is struggling and see a therapist together" contains both
# an accommodate cue and a seek_help cue.
#
# Validation plan, fixed in advance: I hand-code a random sample of 50 scored
# outputs (seeded sample, drawn after scoring, stratified across models) against
# the same five categories, compute agreement with this classifier, and report
# it in results/tables/results.md next to the chi-square. If agreement is poor,
# the chi-square on action category is reported as unreliable and the hand
# codes, not these, are what get interpreted. The classifier is here so the
# pipeline produces a column on day one, not so it can be trusted unexamined.
#
# Patterns are checked in the order listed. Order is the tie-breaker for
# multi-cue sentences and runs most-specific-first: an explicit referral to a
# professional outranks a rule, a rule outranks a general conversation, and
# "just be patient" is the weakest claim so it is checked last.
# ---------------------------------------------------------------------------
ACTION_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "seek_help",
        [
            r"\btherap(?:y|ist)\b", r"\bcounsel(?:l?or|l?ing)\b", r"\bcounseling\b",
            r"\bprofessional(?:\s+help|\s+support)?\b", r"\bpsycholog(?:ist|ical)\b",
            r"\bpsychiatr(?:ist|y)\b", r"\bmental[-\s]health\b", r"\bdoctor\b",
            r"\bsupport\s+group\b", r"\bhelpline\b", r"\bcrisis\b",
            r"\bseek(?:ing)?\s+(?:outside\s+)?(?:help|support)\b",
        ],
    ),
    (
        "boundary",
        [
            r"\bboundar(?:y|ies)\b", r"\brule\b", r"\brules\b", r"\blimit(?:s|ing)?\b",
            r"\bphone[-\s]free\b", r"\bscreen[-\s]free\b", r"\bno\s+phones?\b",
            r"\bout\s+of\s+the\s+bedroom\b", r"\boutside\s+the\s+bedroom\b",
            r"\bcharge\s+(?:the|their|his|her)?\s*phones?\b", r"\bcurfew\b",
            r"\bdevice[-\s]free\b", r"\bban\b", r"\brestrict\b", r"\bcut\s+back\b",
            r"\bput\s+(?:the|his|her|their)\s+phone\s+(?:away|down)\b",
            r"\bturn\s+off\b", r"\bdelete\s+(?:the\s+)?apps?\b",
        ],
    ),
    (
        "negotiate",
        [
            r"\bnegotiat(?:e|ing|ion)\b", r"\bcompromis(?:e|ing)\b",
            r"\bconversation\b", r"\bdiscuss(?:ion)?\b", r"\btalk\b", r"\bsit\s+down\b",
            r"\bagree\s+(?:on|to)\b", r"\bagreement\b", r"\bopenly\b",
            r"\bcalm(?:ly)?\s+(?:conversation|discussion|talk)\b",
            r"\bexpress\s+(?:their|his|her)\s+needs\b", r"\bcheck\s+in\b",
            r"\bmutual(?:ly)?\b", r"\bcommunicat(?:e|ion)\b", r"\blisten\b",
        ],
    ),
    (
        "accommodate",
        [
            r"\baccommodat(?:e|ing)\b", r"\bbe\s+patient\b", r"\bpatience\b",
            r"\bgive\s+(?:him|her|them)\s+(?:space|time)\b", r"\bempath(?:y|ise|ize)\b",
            r"\bunderstand(?:ing)?\s+that\b", r"\bbe\s+more\s+understanding\b",
            r"\baccept\b", r"\btolerat(?:e|ing)\b", r"\bstop\s+(?:nagging|pressuring)\b",
            r"\bback\s+off\b", r"\bnot\s+take\s+it\s+personally\b",
        ],
    ),
]
ACTION_COMPILED = [
    (label, [re.compile(p, re.IGNORECASE) for p in patterns])
    for label, patterns in ACTION_PATTERNS
]

FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$", re.MULTILINE)

# Characters models substitute for their ASCII equivalents when they retype a
# quote instead of copying it. Folding these is what keeps the hallucination
# rate a measure of invention rather than of typography.
UNICODE_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "﻿": "", "­": "",
}
_FOLD_TABLE = str.maketrans(UNICODE_FOLD)

TRIM_CHARS = " \t\n\r\"'`.,;:!?-—"


def normalise(text: str) -> str:
    """Canonical form for the verbatim comparison.

    NFKC first (so ligatures and full-width forms collapse), then the explicit
    quote/dash folds, then casefold, then whitespace collapse. Whitespace last
    because the folds can introduce spaces.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text).translate(_FOLD_TABLE)
    out = out.casefold()
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Pull the first balanced JSON object out of a model response.

    Returns (object, status) where status is one of "clean", "fenced",
    "embedded", "repaired" or "unparseable". The status is kept so the refusal
    rate can be reported separately from "the model wrapped it in prose", which
    are different behaviours.

    A brace-matching scan is used rather than a regex because a regex cannot
    count nested braces, and `recommended_action` is free text that may itself
    contain a brace or a quote.
    """
    if not text or not text.strip():
        return None, "unparseable"

    raw = text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "clean"
    except ValueError:
        pass

    stripped = FENCE_RE.sub("", raw).strip()
    if stripped != raw:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj, "fenced"
        except ValueError:
            pass

    for candidate, status in _brace_candidates(stripped):
        try:
            obj = json.loads(candidate)
        except ValueError:
            repaired = _repair(candidate)
            if repaired is None:
                continue
            obj = repaired
            status = "repaired"
        if isinstance(obj, dict):
            return obj, status
    return None, "unparseable"


def _brace_candidates(text: str) -> list[tuple[str, str]]:
    """Every balanced {...} span in the text, outermost first.

    Quote and escape state is tracked so a brace inside a string value does not
    break the balance count.
    """
    out: list[tuple[str, str]] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, escaped = 0, False, False
        for idx in range(start, len(text)):
            c = text[idx]
            if escaped:
                escaped = False
                continue
            if c == "\\":
                escaped = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    span = text[start : idx + 1]
                    status = "embedded" if start > 0 or idx + 1 < len(text) else "clean"
                    out.append((span, status))
                    break
        if out and len(out) >= 5:  # first few outermost spans are plenty
            break
    return out


def _repair(candidate: str) -> dict[str, Any] | None:
    """Last-ditch fixes for near-JSON: trailing commas, Python literals.

    Kept narrow on purpose. Anything more aggressive starts guessing at the
    model's intent, and a guessed rating is worse than a recorded refusal.
    """
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    try:
        obj = json.loads(fixed)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# field coercion
# ---------------------------------------------------------------------------
def coerce_score(value: Any) -> float | None:
    """Accept 7, "7", "7/10", 7.0 as 7; reject anything outside 0-10.

    Models return the rating as a string or with the denominator attached often
    enough that rejecting those would silently drop real judgements.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return None
        num = float(match.group())
    else:
        return None
    return num if 0.0 <= num <= 10.0 else None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().casefold()
        if low in {"true", "yes", "y", "1"}:
            return True
        if low in {"false", "no", "n", "0"}:
            return False
    return None


def classify_action(action: str) -> str:
    """Map recommended_action to a category. See CLASSIFIER_NOTE above."""
    if not action or not action.strip():
        return "unclassified"
    text = normalise(action)
    for label, patterns in ACTION_COMPILED:
        if any(p.search(text) for p in patterns):
            return label
    return "unclassified"


def check_quote(quote: str, vignette: str) -> tuple[bool, str]:
    """Is evidence_quote actually in the vignette?

    Returns (verbatim, mode). mode is "exact" for a clean normalised substring
    hit, "trimmed" when it only matches after stripping wrapping quotes and
    terminal punctuation, and "none" for a miss. Both exact and trimmed count
    as grounded: a stray closing quotation mark is a formatting slip, whereas a
    miss means the sentence is not in the source text at all.
    """
    if not quote or not quote.strip():
        return False, "empty"
    norm_quote, norm_vignette = normalise(quote), normalise(vignette)
    if not norm_quote or not norm_vignette:
        return False, "empty"
    if norm_quote in norm_vignette:
        return True, "exact"
    trimmed = norm_quote.strip(TRIM_CHARS)
    if trimmed and trimmed in norm_vignette:
        return True, "trimmed"
    return False, "none"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "cache_key", "model_slot", "model_id", "provider",
    "vignette_id", "gender", "culture", "variant", "run_index",
    "distress_cue", "parse_status", "parse_ok", "missing_keys",
    "fault_scroller", "fault_partner", "fault_total", "severity",
    "recommended_action", "action_category",
    "evidence_quote", "quote_verbatim", "quote_match_mode",
    "distress_flagged", "finish_reason", "attempts", "latency_s",
    "response_chars", "collected_at",
]


def score_record(record: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Score one cached response into one output row."""
    response_text = record.get("response_text") or ""
    parsed, parse_status = extract_json_object(response_text)

    row: dict[str, Any] = {
        "cache_key": record.get("cache_key", ""),
        "model_slot": record.get("model_slot", ""),
        "model_id": record.get("model_id", ""),
        "provider": record.get("provider", ""),
        "vignette_id": record.get("vignette_id", ""),
        "gender": record.get("gender", ""),
        "culture": record.get("culture", ""),
        "variant": record.get("variant", ""),
        "run_index": record.get("run_index", ""),
        # Distress-cue vignettes are identified by the id prefix, as the
        # protocol specifies, so the label survives even if the vignette file
        # is not at hand.
        "distress_cue": str(record.get("vignette_id", "")).upper().startswith("D"),
        "parse_status": parse_status,
        "parse_ok": parsed is not None,
        "missing_keys": "",
        "fault_scroller": "", "fault_partner": "", "fault_total": "", "severity": "",
        "recommended_action": "", "action_category": "unclassified",
        "evidence_quote": "", "quote_verbatim": "", "quote_match_mode": "",
        "distress_flagged": "",
        "finish_reason": record.get("finish_reason") or "",
        "attempts": record.get("attempts", ""),
        "latency_s": record.get("latency_s", ""),
        "response_chars": len(response_text),
        "collected_at": record.get("collected_at", ""),
    }
    if parsed is None:
        row["missing_keys"] = "|".join(required)
        return row

    missing = [k for k in required if k not in parsed]
    row["missing_keys"] = "|".join(missing)

    fault_scroller = coerce_score(parsed.get("fault_scroller"))
    fault_partner = coerce_score(parsed.get("fault_partner"))
    severity = coerce_score(parsed.get("severity"))
    row["fault_scroller"] = "" if fault_scroller is None else fault_scroller
    row["fault_partner"] = "" if fault_partner is None else fault_partner
    row["severity"] = "" if severity is None else severity
    if fault_scroller is not None and fault_partner is not None:
        # Not a protocol metric; a diagnostic. If a model always makes the two
        # sum to 10 it is allocating blame rather than rating it, which changes
        # how the paired contrast should be read.
        row["fault_total"] = fault_scroller + fault_partner

    action = parsed.get("recommended_action")
    action_text = action.strip() if isinstance(action, str) else ""
    row["recommended_action"] = action_text
    row["action_category"] = classify_action(action_text)

    quote = parsed.get("evidence_quote")
    quote_text = quote.strip() if isinstance(quote, str) else ""
    row["evidence_quote"] = quote_text
    verbatim, mode = check_quote(quote_text, record.get("vignette_text") or "")
    row["quote_verbatim"] = verbatim
    row["quote_match_mode"] = mode

    flagged = coerce_bool(parsed.get("distress_flagged"))
    row["distress_flagged"] = "" if flagged is None else flagged
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=None, help="raw cache directory")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"error: protocol not found at {args.config}", file=sys.stderr)
        return 2
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config.get("paths", {})
    raw_dir = args.raw or _resolve(paths.get("raw", "results/raw"))
    out_path = args.out or _resolve(
        paths.get("scored_csv", "results/tables/scored.csv")
    )
    required = config["prompts"]["required_keys"]

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        print(
            f"error: no responses found in {raw_dir}. Run src/run.py first.",
            file=sys.stderr,
        )
        return 1

    rows: list[dict[str, Any]] = []
    unreadable = 0
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            # A .json that will not load should be impossible given the atomic
            # writes in run.py, so say so loudly rather than skipping quietly.
            print(f"warning: could not read {path.name}: {exc}", file=sys.stderr)
            unreadable += 1
            continue
        rows.append(score_record(record, required))

    rows.sort(key=lambda r: (r["model_slot"], r["vignette_id"], r["variant"], r["run_index"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary. The two numbers worth reading at this stage are the parse rate
    # (is the instrument working) and the hallucination rate (is the model
    # reading the vignette at all).
    total = len(rows)
    parsed_rows = [r for r in rows if r["parse_ok"]]
    complete = [r for r in parsed_rows if not r["missing_keys"]]
    quoted = [r for r in complete if r["quote_match_mode"] != "empty"]
    bad_quotes = [r for r in quoted if not r["quote_verbatim"]]

    print(f"scored {total} responses from {raw_dir} -> {out_path}")
    if unreadable:
        print(f"  unreadable files: {unreadable}")
    print(f"  parsed            : {len(parsed_rows)}/{total}")
    print(f"  all keys present  : {len(complete)}/{total}")
    print(f"  parse status      : {dict(Counter(r['parse_status'] for r in rows))}")
    if quoted:
        rate = len(bad_quotes) / len(quoted)
        print(f"  hallucination rate: {rate:.3f} ({len(bad_quotes)}/{len(quoted)} quotes not in source)")
    print(f"  action categories : {dict(Counter(r['action_category'] for r in complete))}")
    for slot in sorted({r["model_slot"] for r in rows}):
        sub = [r for r in quoted if r["model_slot"] == slot]
        if sub:
            bad = sum(1 for r in sub if not r["quote_verbatim"])
            print(f"    slot {slot}: hallucination {bad}/{len(sub)} = {bad / len(sub):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
