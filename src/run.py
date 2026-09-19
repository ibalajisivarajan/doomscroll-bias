#!/usr/bin/env python3
"""Collect model judgements for the doomscroll-bias audit.

Builds the full task list (model x vignette x gender x culture x run_index)
from config/protocol.yaml and data/vignettes.jsonl, calls each model's
OpenAI-compatible /chat/completions endpoint, and writes one JSON file per
response into results/raw/.

Design notes, because the reasons matter more than the mechanics:

* One file per response, never one big file. A 7,200-call collection will not
  finish in one sitting: free inference tiers throttle, GitHub Actions jobs die
  at six hours, laptops sleep. A directory of immutable single-response files is
  the cheapest possible resumable store -- the presence of the file IS the
  checkpoint, so there is no separate state to corrupt or to disagree with the
  cache.

* The filename is sha256(model_id, inference options, run_index, prompt).
  Keying on the prompt AND the model-specific inference options means that
  editing a vignette, changing the prompt template, or changing reasoning mode
  forces a re-collection instead of silently mixing responses produced under
  different conditions. That is a feature: it makes stale data impossible to
  reuse by accident.

* Write to <name>.tmp, fsync, then os.replace. Rename within a filesystem is
  atomic, so a job killed mid-write leaves a .tmp file that the next run
  ignores, never a truncated .json that the scorer would parse as real data.

* Retries cover 429 and 5xx with exponential backoff plus jitter. Without
  jitter, four workers that all hit a rate limit sleep for the same interval
  and stampede the endpoint again in lockstep.

Usage:
    python src/run.py                      # full collection (needs locked: true)
    python src/run.py --limit 2 --allow-unlocked   # smoke test
    python src/run.py --model A            # one slot only
    python src/run.py --dry-run            # build the task list, call nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "protocol.yaml"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    """Load the frozen protocol. Every design number comes from here."""
    if not path.exists():
        die(f"protocol not found at {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_vignettes(path: Path) -> list[dict[str, Any]]:
    """Read data/vignettes.jsonl -> [{id, distress, variants}, ...]."""
    if not path.exists():
        die(f"vignette file not found at {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                die(f"{path}:{lineno} is not valid JSON: {exc}")
            for key in ("id", "distress", "variants"):
                if key not in row:
                    die(f"{path}:{lineno} is missing required key {key!r}")
            rows.append(row)
    if not rows:
        die(f"{path} contains no vignettes")
    return rows


def die(msg: str, code: int = 2) -> None:
    """Exit with a one-line explanation instead of a traceback.

    An operator reading CI logs at 2am needs the reason, not a stack trace from
    inside a thread pool.
    """
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def display_path(path: Path) -> str:
    """Repo-relative path for logs, falling back to the absolute one.

    Paths handed in on the command line need not live under the repo (a config
    or cache in /tmp is normal for a test), and Path.relative_to raises rather
    than returning None when they do not. Formatting a log line must never be
    the thing that kills a collection run.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    """Interpret a configured path relative to the repo unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    """One planned API call: a fully resolved cell of the design."""

    model_slot: str
    model_id: str
    provider: str
    base_url: str
    env_key: str
    reasoning_effort: str | None
    reasoning_format: str | None
    vignette_id: str
    gender: str
    culture: str
    run_index: int
    vignette_text: str
    system_prompt: str
    user_prompt: str

    @property
    def variant(self) -> str:
        return f"{self.gender}_{self.culture}"

    @property
    def cache_key(self) -> str:
        """sha256 over model, inference mode, run index and prompt.

        Provider-controlled reasoning settings can materially change model
        behaviour even when the model id and prompt are identical. They are
        therefore part of the cache identity, so changing reasoning mode can
        never silently reuse responses collected under another mode.
        """
        h = hashlib.sha256()
        h.update(self.model_id.encode("utf-8"))
        h.update(b"\x00")
        h.update((self.reasoning_effort or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((self.reasoning_format or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(str(self.run_index).encode("utf-8"))
        h.update(b"\x00")
        h.update(self.system_prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.user_prompt.encode("utf-8"))
        return h.hexdigest()

    def cache_path(self, raw_dir: Path) -> Path:
        return raw_dir / f"{self.cache_key}.json"


def build_tasks(
    config: dict[str, Any],
    vignettes: list[dict[str, Any]],
    only_slots: Iterable[str] | None = None,
) -> list[Task]:
    """Expand the design into the full cross product of planned calls.

    Ordering is model -> vignette -> gender -> culture -> run_index, which means
    --limit N takes a slice that still spans several conditions of the first
    vignettes rather than 5 identical repeats of one cell. A smoke test that
    only ever exercises one condition is not a smoke test.
    """
    design = config["design"]
    prompts = config["prompts"]
    genders = design["factors"]["gender"]
    cultures = design["factors"]["culture"]
    runs = int(design["runs_per_cell"])
    key_fmt = design.get("variant_key_format", "{gender}_{culture}")

    slots = {s.upper() for s in only_slots} if only_slots else None
    tasks: list[Task] = []
    for model in config["models"]:
        slot = str(model["slot"]).upper()
        if slots and slot not in slots:
            continue
        for vignette in vignettes:
            for gender in genders:
                for culture in cultures:
                    variant_key = key_fmt.format(gender=gender, culture=culture)
                    text = vignette["variants"].get(variant_key)
                    if text is None:
                        die(
                            f"vignette {vignette['id']} has no variant "
                            f"{variant_key!r}; the design requires all "
                            f"{len(genders) * len(cultures)} renderings"
                        )
                    user_prompt = prompts["user_template"].format(vignette=text)
                    for run_index in range(1, runs + 1):
                        tasks.append(
                            Task(
                                model_slot=slot,
                                model_id=str(model["id"]),
                                provider=str(model.get("provider", "")),
                                base_url=str(model["base_url"]),
                                env_key=str(model["env_key"]),
                                reasoning_effort=(
                                    str(model["reasoning_effort"])
                                    if model.get("reasoning_effort") is not None
                                    else None
                                ),
                                reasoning_format=(
                                    str(model["reasoning_format"])
                                    if model.get("reasoning_format") is not None
                                    else None
                                ),
                                vignette_id=str(vignette["id"]),
                                gender=gender,
                                culture=culture,
                                run_index=run_index,
                                vignette_text=text,
                                system_prompt=prompts["system"],
                                user_prompt=user_prompt,
                            )
                        )
    return tasks


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
def check_lock(config: dict[str, Any], allow_unlocked: bool) -> None:
    """Refuse to spend money and quota against an unfrozen protocol.

    The whole pre-registration argument rests on the protocol being final
    before collection starts. Running against `locked: false` produces data
    whose provenance cannot be defended, so it takes an explicit flag and a
    loud warning.
    """
    if config.get("locked") is True:
        return
    if not allow_unlocked:
        die(
            "protocol is not locked (config/protocol.yaml: locked: false).\n"
            "  Collecting against an unfrozen protocol would break the "
            "pre-registration guarantee that the analysis plan predates the "
            "results.\n"
            "  Finish authoring the vignettes and the prompts, set "
            "locked: true, and commit that change on its own.\n"
            "  For a smoke test only, re-run with --allow-unlocked.",
            code=3,
        )
    print(
        "WARNING: running against an UNLOCKED protocol (--allow-unlocked).\n"
        "         Output from this run is a smoke test, not study data. "
        "Delete results/raw before the real collection.",
        file=sys.stderr,
    )


def check_models(tasks: list[Task]) -> None:
    """Fail cleanly on unfilled model slots and missing credentials."""
    problems: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task.model_slot in seen:
            continue
        seen.add(task.model_slot)
        label = f"model slot {task.model_slot}"
        placeholders = [
            name
            for name, value in (
                ("id", task.model_id),
                ("provider", task.provider),
                ("base_url", task.base_url),
            )
            if not value or value.strip().upper() == "TBD"
        ]
        if placeholders:
            problems.append(
                f"{label}: {', '.join(placeholders)} still set to TBD in "
                f"config/protocol.yaml"
            )
            continue
        if not os.environ.get(task.env_key):
            problems.append(
                f"{label} ({task.model_id}): environment variable "
                f"{task.env_key} is not set"
            )
    if problems:
        die(
            "cannot start collection:\n  - "
            + "\n  - ".join(problems)
            + "\n\nFill in the model slots in config/protocol.yaml and export "
            "the matching API keys, then re-run. Use --dry-run to inspect "
            "the task list without calling any endpoint.",
            code=4,
        )


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """Process-wide requests-per-minute gate shared by all workers."""

    def __init__(self, requests_per_minute: float) -> None:
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if wait > 0:
            time.sleep(wait)


@dataclass
class Stats:
    """Counters for the end-of-run summary."""

    cached: int = 0
    written: int = 0
    failed: int = 0
    retries: int = 0
    capped: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, attr: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + n)


# --------------------------------------------------------------------------
# daily request cap
# --------------------------------------------------------------------------
class DailyCapExceeded(RuntimeError):
    pass


class TimeBudgetExceeded(RuntimeError):
    pass


class DailyCapTracker:
    """Persists a count of request attempts made today."""

    def __init__(self, path: Path, cap: int) -> None:
        self.path = path
        self.cap = cap
        self._lock = threading.Lock()
        self._date, self._count = self._load()

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load(self) -> tuple[str, int]:
        today = self._today()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("date") == today:
                    return today, int(data.get("count", 0))
            except (ValueError, OSError, TypeError):
                pass
        return today, 0

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"date": self._date, "count": self._count}), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def try_consume(self) -> bool:
        with self._lock:
            today = self._today()
            if today != self._date:
                self._date, self._count = today, 0
            if self._count >= self.cap:
                return False
            self._count += 1
            self._save()
            return True

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


# --------------------------------------------------------------------------
# the call
# --------------------------------------------------------------------------
def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def call_model(
    task: Task,
    session: requests.Session,
    config: dict[str, Any],
    limiter: RateLimiter,
    stats: Stats,
    daily_cap_tracker: "DailyCapTracker | None" = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """POST one completion, retrying transient failures. Returns the record."""
    limits = config["rate_limits"]
    design = config["design"]
    prompts = config["prompts"]

    max_retries = int(limits["max_retries"])
    base = float(limits["backoff_base_seconds"])
    backoff_max = float(limits.get("backoff_max", 120))
    jitter = float(limits.get("jitter", 0.5))
    retry_on = set(limits.get("retry_on_status", [429, 500, 502, 503, 504]))
    timeout = float(limits.get("request_timeout", 120))

    payload: dict[str, Any] = {
        "model": task.model_id,
        "messages": [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_prompt},
        ],
        "temperature": float(design.get("temperature", 0.0)),
        "max_tokens": int(design.get("max_tokens", 500)),
    }
    if design.get("seed") is not None:
        payload["seed"] = int(design["seed"])
    if prompts.get("response_format") == "json_object":
        payload["response_format"] = {"type": "json_object"}
    if task.reasoning_effort is not None:
        payload["reasoning_effort"] = task.reasoning_effort
    if task.reasoning_format is not None:
        payload["reasoning_format"] = task.reasoning_format

    headers = {
        "Authorization": f"Bearer {os.environ.get(task.env_key, '')}",
        "Content-Type": "application/json",
    }
    url = chat_completions_url(task.base_url)

    last_error = "unknown"
    next_delay_override: float | None = None
    # Structured per-attempt telemetry. Run 35422141274 (2026-09-19) showed
    # Model B burning 107 retries for 67 successes with nothing in the log to
    # prove whether that was 429/TPM throttling, 5xx, or transport timeouts --
    # every one of those explanations was inference from wall-clock timing,
    # not evidence. retry_events is kept small (bounded by max_retries) and
    # carries no header/secret material, only what is needed to classify the
    # retry: attempt number, HTTP status or transport-exception class, any
    # Retry-After value the provider sent, and the delay actually slept.
    retry_events: list[dict[str, Any]] = []
    cell = f"{task.model_slot} {task.vignette_id} {task.variant} run={task.run_index}"
    for attempt in range(max_retries + 1):
        if daily_cap_tracker is not None and not daily_cap_tracker.try_consume():
            raise DailyCapExceeded(
                f"daily request cap of {daily_cap_tracker.cap} reached "
                f"({daily_cap_tracker.count}/{daily_cap_tracker.cap} attempts today)"
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeBudgetExceeded(
                "collection deadline reached; stopping with margin before "
                "the job timeout rather than racing a hard cancellation"
            )
        if attempt:
            if next_delay_override is not None:
                delay = next_delay_override + random.uniform(0, jitter)
                next_delay_override = None
            else:
                delay = min(base**attempt, backoff_max) + random.uniform(0, jitter)
            stats.bump("retries")
            print(
                f"  retry   {cell}  attempt={attempt + 1}/{max_retries + 1}  "
                f"{retry_events[-1]['outcome']} status={retry_events[-1]['http_status']} "
                f"retry_after={retry_events[-1]['retry_after_s']}  sleeping={delay:.1f}s"
            )
            time.sleep(delay)
        limiter.acquire()
        started = time.time()
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"transport error: {exc.__class__.__name__}: {exc}"
            retry_events.append({
                "attempt": attempt + 1,
                "outcome": "transport_error",
                "http_status": None,
                "error_class": exc.__class__.__name__,
                "retry_after_s": None,
            })
            continue

        if resp.status_code in retry_on:
            retry_after_hdr = resp.headers.get("Retry-After")
            retry_after_s: float | None = None
            if retry_after_hdr:
                try:
                    retry_after_s = min(float(retry_after_hdr), backoff_max)
                    next_delay_override = retry_after_s
                except ValueError:
                    pass
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            retry_events.append({
                "attempt": attempt + 1,
                "outcome": "retryable_http",
                "http_status": resp.status_code,
                "error_class": None,
                "retry_after_s": retry_after_s,
            })
            continue

        if resp.status_code >= 400:
            raise RuntimeError(
                f"HTTP {resp.status_code} (not retryable): {resp.text[:300]}"
            )

        try:
            body = resp.json()
        except ValueError:
            last_error = f"response was not JSON: {resp.text[:300]}"
            retry_events.append({
                "attempt": attempt + 1,
                "outcome": "malformed_response",
                "http_status": resp.status_code,
                "error_class": None,
                "retry_after_s": None,
            })
            continue

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            last_error = f"unexpected response shape: {json.dumps(body)[:300]}"
            retry_events.append({
                "attempt": attempt + 1,
                "outcome": "malformed_response",
                "http_status": resp.status_code,
                "error_class": None,
                "retry_after_s": None,
            })
            continue

        return {
            "cache_key": task.cache_key,
            "model_slot": task.model_slot,
            "model_id": task.model_id,
            "provider": task.provider,
            "vignette_id": task.vignette_id,
            "gender": task.gender,
            "culture": task.culture,
            "variant": task.variant,
            "run_index": task.run_index,
            "temperature": payload["temperature"],
            "reasoning_effort": task.reasoning_effort,
            "reasoning_format": task.reasoning_format,
            "system_prompt": task.system_prompt,
            "user_prompt": task.user_prompt,
            "vignette_text": task.vignette_text,
            "response_text": content,
            "finish_reason": (body["choices"][0] or {}).get("finish_reason"),
            "usage": body.get("usage"),
            "http_status": resp.status_code,
            "attempts": attempt + 1,
            "retry_events": retry_events,
            "latency_s": round(time.time() - started, 3),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    if retry_events:
        print(f"  exhausted retries on {cell}:")
        for ev in retry_events:
            print(
                f"    attempt={ev['attempt']} outcome={ev['outcome']} "
                f"status={ev['http_status']} error={ev['error_class']} "
                f"retry_after={ev['retry_after_s']}"
            )
    raise RuntimeError(f"exhausted {max_retries} retries; last error: {last_error}")


def write_atomic(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def process(
    task: Task,
    raw_dir: Path,
    session: requests.Session,
    config: dict[str, Any],
    limiter: RateLimiter,
    stats: Stats,
    daily_cap_tracker: "DailyCapTracker | None" = None,
    deadline: float | None = None,
) -> tuple[Task, str, str | None]:
    path = task.cache_path(raw_dir)
    if path.exists():
        stats.bump("cached")
        return task, "cached", None
    try:
        record = call_model(task, session, config, limiter, stats, daily_cap_tracker, deadline)
    except (DailyCapExceeded, TimeBudgetExceeded) as exc:
        stats.bump("capped")
        return task, "capped", str(exc)
    except Exception as exc:  # noqa: BLE001
        stats.bump("failed")
        return task, "failed", f"{exc.__class__.__name__}: {exc}"
    write_atomic(path, record)
    stats.bump("written")
    return task, "written", None


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect model judgements for the doomscroll-bias audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="protocol path")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="stop after N uncached calls (smoke test)",
    )
    p.add_argument(
        "--allow-unlocked",
        action="store_true",
        help="run even though the protocol is not locked; smoke tests only",
    )
    p.add_argument(
        "--model",
        action="append",
        metavar="SLOT",
        help="restrict to model slot A or B (repeatable)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build and summarise the task list without calling any endpoint",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        metavar="MIN",
        help=(
            "stop collecting cleanly after MIN minutes (overrides "
            "rate_limits.max_collection_minutes in the protocol); mainly "
            "for smoke-testing the clean-stop path itself"
        ),
    )
    return p.parse_args(argv)


def classify_run_status(written: int, failed: int, capped: int) -> str:
    """Decide SUCCESS / PARTIAL_SUCCESS / CAPPED_CLEAN / FAILURE for this dispatch.

    Run 35422141274 (2026-09-19) turned a day that wrote 67 real responses
    and safely hit its daily cap into a red GitHub Actions job, because one
    non-retryable HTTP 400 on a single cell made the whole command exit 1 and
    every downstream QC/Notion step (lacking always()) got skipped by
    GitHub's implicit success() gate. That conflates two very different
    situations: a single bad cell in an otherwise-productive run, and a run
    where nothing worked at all (the signature of a revoked key or a broken
    slot config, which check_models() cannot catch since it only checks that
    the env var is *set*, not that the provider accepts it).

    `attempted` deliberately excludes `cached` (no call was made) and
    `capped` (no call was made; the daily budget was already spent) -- both
    are "we chose not to attempt this", not "we attempted and something went
    wrong". Only cells that actually reached the network count.

    written == 0 with attempted > 0 is always systemic: every attempted cell
    failed, so there is no cell-specific explanation left. With more than a
    handful of attempts, a failure rate at or above 50% is treated the same
    way even if a few cells did succeed, on the reasoning that isolated bad
    luck should look like today's 1-failure-in-68 (~1.5%), not like half the
    attempts going bad. Both thresholds are judgement calls, not measured
    constants; they are conservative enough that today's actual run
    (written=67, failed=1, attempted=68) lands clearly in PARTIAL_SUCCESS.
    """
    attempted = written + failed
    if attempted == 0:
        return "CAPPED_CLEAN" if capped else "SUCCESS"
    failure_rate = failed / attempted
    systemic = written == 0 or (attempted >= 5 and failure_rate >= 0.5)
    if systemic:
        return "FAILURE"
    if failed > 0:
        return "PARTIAL_SUCCESS"
    return "CAPPED_CLEAN" if capped else "SUCCESS"


def emit_run_summary(
    status: str,
    *,
    model_slots: str,
    planned: int,
    cached: int,
    written: int,
    failed: int,
    retries: int,
    capped: int,
    pending_remaining: int,
) -> None:
    """Surface the outcome to GitHub Actions without changing run.yml's gating.

    The exit code this run returns is what actually decides whether the
    downstream score/QC/Notion steps run -- they already key off GitHub's
    implicit success() for every step that lacks always(), so making
    classify_run_status() return something other than FAILURE for a
    productive-but-imperfect day is the whole fix for that cascade. This
    function only adds visibility into *why* the exit code was what it was:
    step outputs (for the workflow to fold into the Notion sync note) and a
    step summary table (for a human glancing at the Actions run).
    """
    print(
        f"\nsummary : status={status} slots={model_slots} planned={planned} "
        f"cached={cached} written={written} failed={failed} retries={retries} "
        f"capped={capped} pending_remaining={pending_remaining}"
    )

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        try:
            with open(gh_output, "a", encoding="utf-8") as fh:
                for key, value in (
                    ("status", status),
                    ("model_slots", model_slots),
                    ("cached", cached),
                    ("written", written),
                    ("failed", failed),
                    ("retries", retries),
                    ("capped", capped),
                    ("pending_remaining", pending_remaining),
                ):
                    fh.write(f"{key}={value}\n")
        except OSError as exc:
            print(f"warning: could not write GITHUB_OUTPUT: {exc}", file=sys.stderr)

    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        try:
            with open(gh_summary, "a", encoding="utf-8") as fh:
                fh.write(f"### Collect {model_slots}: {status}\n\n")
                fh.write(
                    "| planned | cached | written | failed | retries | capped "
                    "| pending remaining |\n|---|---|---|---|---|---|---|\n"
                )
                fh.write(
                    f"| {planned} | {cached} | {written} | {failed} | {retries} "
                    f"| {capped} | {pending_remaining} |\n\n"
                )
        except OSError as exc:
            print(f"warning: could not write GITHUB_STEP_SUMMARY: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    paths = config.get("paths", {})
    vignette_path = resolve_path(paths.get("vignettes", "data/vignettes.jsonl"))
    vignettes = load_vignettes(vignette_path)
    raw_dir = resolve_path(paths.get("raw", "results/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(config, vignettes, args.model)
    if not tasks:
        die("task list is empty; check --model and config/protocol.yaml")

    planned = config["design"].get("total_calls")
    print(f"protocol : {display_path(args.config)}  (locked={config.get('locked')})")
    print(f"vignettes: {len(vignettes)} loaded from {display_path(vignette_path)}")
    print(
        f"tasks    : {len(tasks)} planned "
        f"({len(vignettes)} vignettes x "
        f"{len(config['design']['factors']['gender'])} genders x "
        f"{len(config['design']['factors']['culture'])} cultures x "
        f"{config['design']['runs_per_cell']} runs x "
        f"{len({t.model_slot for t in tasks})} models)"
    )
    if planned and len(vignettes) != config["design"]["n_vignettes"]:
        print(
            f"note     : the protocol's full design is {planned} calls over "
            f"{config['design']['n_vignettes']} vignettes; "
            f"{len(vignettes)} are authored so far."
        )

    pending = [t for t in tasks if not t.cache_path(raw_dir).exists()]
    initial_cached = len(tasks) - len(pending)
    print(f"cached   : {initial_cached} already in {display_path(raw_dir)}")

    if args.dry_run:
        print(f"pending  : {len(pending)} (dry run: nothing called)")
        for task in pending[: args.limit or 5]:
            print(
                f"  {task.model_slot} {task.vignette_id} {task.variant} "
                f"run={task.run_index} key={task.cache_key[:12]}"
            )
        return 0

    check_lock(config, args.allow_unlocked)
    check_models(tasks)

    model_slots = ",".join(sorted({t.model_slot for t in tasks}))

    if not pending:
        print("nothing to do: every planned call is already cached.")
        emit_run_summary(
            "SUCCESS", model_slots=model_slots, planned=len(tasks),
            cached=len(tasks), written=0, failed=0, retries=0, capped=0,
            pending_remaining=0,
        )
        return 0
    if args.limit is not None:
        held_back = max(0, len(pending) - max(0, args.limit))
        pending = pending[: max(0, args.limit)]
        if held_back:
            print(f"limit    : {held_back} pending call(s) held back by --limit {args.limit}")
        if not pending:
            print(f"nothing collected: --limit {args.limit} leaves no calls to make.")
            emit_run_summary(
                "SUCCESS", model_slots=model_slots, planned=len(tasks),
                cached=initial_cached, written=0, failed=0,
                retries=0, capped=0, pending_remaining=held_back,
            )
            return 0
    print(f"pending  : {len(pending)} to collect\n")

    limits = config["rate_limits"]
    concurrency = max(1, int(limits["max_concurrent"]))
    limiter = RateLimiter(float(limits["requests_per_minute"]))
    stats = Stats()
    failures: list[tuple[Task, str]] = []

    # Groq publishes daily limits per selected model, so quota accounting must
    # also be per model. A single shared counter would let the first slot consume
    # most of the day's allowance and artificially starve the second slot.
    daily_cap_trackers: dict[str, DailyCapTracker] = {}
    daily_cap = limits.get("daily_request_cap_per_model", limits.get("daily_request_cap"))
    if daily_cap:
        daily_count_base = resolve_path(paths.get("daily_count", "results/.daily_count.json"))
        for slot in sorted({t.model_slot for t in tasks}):
            daily_count_path = daily_count_base.with_name(
                f"{daily_count_base.stem}_{slot}{daily_count_base.suffix}"
            )
            tracker = DailyCapTracker(daily_count_path, int(daily_cap))
            daily_cap_trackers[slot] = tracker
            if tracker.count >= tracker.cap:
                print(
                    f"daily cap {slot}: already at {tracker.count}/{daily_cap} attempts "
                    f"today ({display_path(daily_count_path)}); that model will not collect "
                    f"again until the UTC date rolls over."
                )
            else:
                print(
                    f"daily cap {slot}: {tracker.count}/{daily_cap} attempts spent today "
                    f"({display_path(daily_count_path)})"
                )

    max_minutes = args.max_minutes
    if max_minutes is None:
        max_minutes = limits.get("max_collection_minutes")
    deadline: float | None = None
    if max_minutes:
        deadline = time.monotonic() + float(max_minutes) * 60
        print(f"deadline : stopping collection cleanly after {float(max_minutes):.1f} minutes")

    session = requests.Session()
    cap_hit_announced = False
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    process, t, raw_dir, session, config, limiter, stats,
                    daily_cap_trackers.get(t.model_slot), deadline,
                )
                for t in pending
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                task, status, error = future.result()
                if status == "capped":
                    if not cap_hit_announced:
                        cap_hit_announced = True
                        print(f"\n[{done}/{len(pending)}] CAPPED  {error}")
                        print("stopping cleanly. Remaining tasks stay uncached for the next run.")
                    continue
                if status == "failed":
                    failures.append((task, error or "unknown"))
                    marker = "FAIL"
                else:
                    marker = status.upper()
                print(
                    f"[{done}/{len(pending)}] {marker:7s} {task.model_slot} "
                    f"{task.vignette_id} {task.variant} run={task.run_index}"
                    + (f"  {error}" if error else "")
                )
    except KeyboardInterrupt:
        print("\ninterrupted; completed responses are cached. Re-run to resume.")
        return 130
    finally:
        session.close()

    print(
        f"\ndone: {stats.written} written, {stats.cached} cached, "
        f"{stats.failed} failed, {stats.retries} retries"
        + (f", {stats.capped} capped" if stats.capped else "")
    )
    if failures:
        print("\nfailed cells (re-run to retry; cached work is not repeated):")
        for task, error in failures[:20]:
            print(
                f"  {task.model_slot} {task.vignette_id} {task.variant} "
                f"run={task.run_index}: {error}"
            )
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    status = classify_run_status(stats.written, stats.failed, stats.capped)
    pending_remaining = len(pending) - stats.written - stats.failed
    emit_run_summary(
        status, model_slots=model_slots, planned=len(tasks),
        cached=initial_cached + stats.cached, written=stats.written,
        failed=stats.failed, retries=stats.retries, capped=stats.capped,
        pending_remaining=pending_remaining,
    )
    if status == "FAILURE":
        print(
            "\nFAILURE: every attempted call this dispatch failed "
            f"({stats.failed}/{stats.written + stats.failed} attempted); "
            "this looks systemic (bad credential, broken slot, or a "
            "provider-side outage), not one unlucky cell. Failing the job "
            "loudly rather than reporting partial success."
        )
        return 1
    if status == "PARTIAL_SUCCESS":
        print(
            f"\nPARTIAL_SUCCESS: {stats.written} real response(s) written "
            f"despite {stats.failed} failed cell(s). Exiting 0 so downstream "
            "scoring/QC/Notion sync still run on the cumulative data; failed "
            "cells remain uncached and will be retried on the next dispatch."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
