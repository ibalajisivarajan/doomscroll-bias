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

* The filename is sha256(model_id, run_index, prompt). Keying on the prompt
  rather than on the condition labels means that editing a vignette, or the
  prompt template, changes the key and forces a re-collection instead of
  silently mixing responses to two different texts under one label. That is a
  feature: it makes stale data impossible to reuse by accident.

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
        """sha256 over (model_id, run_index, prompt).

        The prompt is included verbatim so that any change to the vignette or
        the template invalidates the cache rather than silently reusing a
        response to different text.
        """
        h = hashlib.sha256()
        h.update(self.model_id.encode("utf-8"))
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
    """Fail cleanly on unfilled model slots and missing credentials.

    The protocol ships with id/provider/base_url set to "TBD" on purpose, so
    this is the expected state of a fresh clone. It must read as a to-do list,
    not as a crash.
    """
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
            continue  # no point checking the key for a slot with no endpoint
        if not os.environ.get(task.env_key):
            problems.append(
                f"{label} ({task.model_id}): environment variable "
                f"{task.env_key} is not set"
            )
    if problems:
        die(
            "cannot start collection:\n  - "
            + "\n  - ".join(problems)
            + "\n\nFill in the model slots in config/protocol.yaml (id, "
            "provider, base_url) and export the matching API keys, then "
            "re-run. Use --dry-run to inspect the task list without "
            "calling any endpoint.",
            code=4,
        )


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """Process-wide requests-per-minute gate shared by all workers.

    Concurrency alone does not bound request rate: a handful of workers
    against a fast endpoint can burn a whole per-minute allowance in seconds
    and then spend the rest of the run in backoff. This spaces request starts
    so the steady state stays under the documented limit.
    """

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
    """Raised by call_model once today's attempt budget is spent.

    Deliberately its own type rather than a plain RuntimeError, so process()
    can tell "today's quota is gone" apart from "this one cell failed" and
    the run can stop quietly instead of logging a FAIL line per remaining
    task.
    """


class DailyCapTracker:
    """Persists a count of request ATTEMPTS made today, shared across workers
    and across dispatches.

    Counts attempts, not successes. A 429 that gets retried five times spends
    five requests against Groq's org-wide daily ceiling even though it never
    produces a cached response, so a guard that only counted writes would not
    trip until the real ceiling had already been hit -- at which point every
    model call in the account, not just this study's, starts failing for the
    rest of the day. `daily_request_cap` in the protocol is set below that
    ceiling specifically to leave that margin.

    The counter is keyed by UTC calendar date (Groq's ceiling resets at UTC
    midnight) and written to disk after every attempt via the same
    tmp-then-replace pattern as the response cache, so:
      * it survives a killed process -- a run resumed five minutes later does
        not get to recount from zero;
      * a second dispatch later the same UTC day sees what the first one
        already spent, instead of both independently believing they have the
        full daily allowance.
    """

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
                pass  # corrupt or missing counter file: start today at zero
        return today, 0

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"date": self._date, "count": self._count}), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def try_consume(self) -> bool:
        """Spend one unit of today's budget. False once the cap is reached.

        Rolls the counter over to zero itself on a UTC date change, so a run
        that happens to straddle midnight keeps working under the new day's
        allowance rather than staying stuck at yesterday's cap.
        """
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
    """Join the configured API root to the chat-completions path.

    Hosts are inconsistent about trailing slashes and about whether the root
    already ends in /v1, so normalise here instead of in the config.
    """
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
) -> dict[str, Any]:
    """POST one completion, retrying transient failures. Returns the record.

    Raises RuntimeError once retries are exhausted; the caller records the
    failure and moves on, because one dead cell must not abort 7,199 others.
    Raises DailyCapExceeded instead, without making a request, once
    daily_cap_tracker reports today's attempt budget is spent.
    """
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
        # Honoured by some hosts, ignored by others. Harmless either way, and
        # when it is honoured it makes the 5 repeats genuinely reproducible.
        payload["seed"] = int(design["seed"])
    if prompts.get("response_format") == "json_object":
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {os.environ.get(task.env_key, '')}",
        "Content-Type": "application/json",
    }
    url = chat_completions_url(task.base_url)

    last_error = "unknown"
    # Retry-After, when the host sends one, overrides the NEXT sleep rather
    # than being slept on top of it. Groq sends this header on 429s and it
    # is more precise than our own exponential guess; the earlier version of
    # this loop slept for Retry-After immediately AND for the exponential
    # delay on the following iteration, which double-waited on every
    # rate-limited retry and roughly halved real throughput against a
    # strict per-minute budget.
    next_delay_override: float | None = None
    for attempt in range(max_retries + 1):
        # Checked first, before any sleep or network activity: a denied
        # attempt must cost nothing. Every iteration of this loop is one
        # attempt against Groq's daily ceiling whether or not it succeeds,
        # so the guard consumes budget here -- not after a response comes
        # back -- which is what "count attempts, not successes" requires.
        if daily_cap_tracker is not None and not daily_cap_tracker.try_consume():
            raise DailyCapExceeded(
                f"daily request cap of {daily_cap_tracker.cap} reached "
                f"({daily_cap_tracker.count}/{daily_cap_tracker.cap} attempts today)"
            )
        if attempt:
            if next_delay_override is not None:
                delay = next_delay_override + random.uniform(0, jitter)
                next_delay_override = None
            else:
                # Exponential backoff with jitter. The jitter is what keeps
                # the workers from retrying in lockstep after a shared 429.
                delay = min(base**attempt, backoff_max) + random.uniform(0, jitter)
            stats.bump("retries")
            time.sleep(delay)
        limiter.acquire()
        started = time.time()
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"transport error: {exc.__class__.__name__}: {exc}"
            continue

        if resp.status_code in retry_on:
            # Respect Retry-After when the host sends one; it knows its own
            # window better than our backoff curve does. Recorded here and
            # applied at the top of the next iteration instead of slept on
            # the spot, so it replaces rather than stacks with the
            # exponential delay above.
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    next_delay_override = min(float(retry_after), backoff_max)
                except ValueError:
                    pass
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            continue

        if resp.status_code >= 400:
            # 400/401/403/404 are configuration errors, not weather. Retrying
            # a bad model id or a rejected key just wastes the quota.
            raise RuntimeError(
                f"HTTP {resp.status_code} (not retryable): {resp.text[:300]}"
            )

        try:
            body = resp.json()
        except ValueError:
            last_error = f"response was not JSON: {resp.text[:300]}"
            continue

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            last_error = f"unexpected response shape: {json.dumps(body)[:300]}"
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
            "system_prompt": task.system_prompt,
            "user_prompt": task.user_prompt,
            # The vignette text is stored alongside the response so the
            # verbatim-quote check in score.py never has to guess which text
            # produced this output, even if data/vignettes.jsonl later changes.
            "vignette_text": task.vignette_text,
            "response_text": content,
            "finish_reason": (body["choices"][0] or {}).get("finish_reason"),
            "usage": body.get("usage"),
            "http_status": resp.status_code,
            "attempts": attempt + 1,
            "latency_s": round(time.time() - started, 3),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    raise RuntimeError(f"exhausted {max_retries} retries; last error: {last_error}")


def write_atomic(path: Path, record: dict[str, Any]) -> None:
    """Write JSON via a .tmp sibling plus os.replace.

    os.replace is atomic within a filesystem, so a reader (or a killed job)
    only ever sees the complete file or no file at all. fsync before the rename
    means a machine that loses power does not leave a rename pointing at empty
    bytes.
    """
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
) -> tuple[Task, str, str | None]:
    """Fetch one task unless it is already cached. Never raises."""
    path = task.cache_path(raw_dir)
    if path.exists():
        # The file's existence is the checkpoint. This is what makes the run
        # resumable across dispatches at zero bookkeeping cost.
        stats.bump("cached")
        return task, "cached", None
    try:
        record = call_model(task, session, config, limiter, stats, daily_cap_tracker)
    except DailyCapExceeded as exc:
        # Distinct from an ordinary failure: nothing is wrong with this cell,
        # today's budget is just gone. main() handles this status specially so
        # a queue of thousands of already-submitted tasks does not each print
        # their own FAIL line on the way to the same conclusion.
        stats.bump("capped")
        return task, "capped", str(exc)
    except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the run
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    paths = config.get("paths", {})
    # Config paths may be absolute (a test cache in /tmp) or repo-relative.
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
    print(f"cached   : {len(tasks) - len(pending)} already in {display_path(raw_dir)}")

    if args.dry_run:
        print(f"pending  : {len(pending)} (dry run: nothing called)")
        for task in pending[: args.limit or 5]:
            print(
                f"  {task.model_slot} {task.vignette_id} {task.variant} "
                f"run={task.run_index} key={task.cache_key[:12]}"
            )
        return 0

    # Lock and credential checks run before anything is sent, so a
    # misconfigured clone fails in a second rather than after 40 timeouts.
    check_lock(config, args.allow_unlocked)
    check_models(tasks)

    if not pending:
        print("nothing to do: every planned call is already cached.")
        return 0
    if args.limit is not None:
        # Report the limit separately from an empty queue: "nothing to do"
        # must mean the collection is complete, never "you passed --limit 0".
        held_back = max(0, len(pending) - max(0, args.limit))
        pending = pending[: max(0, args.limit)]
        if held_back:
            print(f"limit    : {held_back} pending call(s) held back by --limit {args.limit}")
        if not pending:
            print(f"nothing collected: --limit {args.limit} leaves no calls to make.")
            return 0
    print(f"pending  : {len(pending)} to collect\n")

    limits = config["rate_limits"]
    concurrency = max(1, int(limits["max_concurrent"]))
    limiter = RateLimiter(float(limits["requests_per_minute"]))
    stats = Stats()
    failures: list[tuple[Task, str]] = []

    daily_cap_tracker: DailyCapTracker | None = None
    daily_cap = limits.get("daily_request_cap")
    if daily_cap:
        daily_count_path = resolve_path(paths.get("daily_count", "results/.daily_count.json"))
        daily_cap_tracker = DailyCapTracker(daily_count_path, int(daily_cap))
        if daily_cap_tracker.count >= daily_cap_tracker.cap:
            print(
                f"daily cap: already at {daily_cap_tracker.count}/{daily_cap} attempts "
                f"today ({display_path(daily_count_path)}); nothing will be collected "
                f"until the UTC date rolls over."
            )
        else:
            print(
                f"daily cap: {daily_cap_tracker.count}/{daily_cap} attempts spent today "
                f"({display_path(daily_count_path)})"
            )

    session = requests.Session()
    cap_hit_announced = False
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(process, t, raw_dir, session, config, limiter, stats, daily_cap_tracker)
                for t in pending
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                task, status, error = future.result()
                if status == "capped":
                    # One clean message, not one FAIL line per remaining
                    # queued task: every task submitted above will discover
                    # the same exhausted budget in turn, and printing that
                    # thousands of times over would bury the summary.
                    if not cap_hit_announced:
                        cap_hit_announced = True
                        print(f"\n[{done}/{len(pending)}] CAPPED  {error}")
                        print("stopping: today's attempt budget is spent. "
                              "Remaining tasks stay uncached for the next run.")
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
        # Already-written files stay valid, so a Ctrl-C is a pause, not a loss.
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
        return 1
    # A cap hit is a clean stop, not a failure: nothing is broken, the day's
    # budget is just spent. Exit 0 so a CI step chain (score -> analyze ->
    # commit) keeps going over whatever was collected before the cap hit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
