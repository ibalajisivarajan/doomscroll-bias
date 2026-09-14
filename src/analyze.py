#!/usr/bin/env python3
"""Run the pre-registered analysis over results/tables/scored.csv.

Writes results/tables/results.md and one fault-gap histogram PNG per model.

The pipeline, and why each step is shaped the way it is:

1. AVERAGE THE REPEATS FIRST. The 5 runs of a cell are not 5 independent
   observations -- they are 5 samples of the same (model, vignette, condition)
   point, collected to measure decoding instability. Treating them as
   independent would inflate n by 5x and produce p-values that mean nothing. So
   they collapse to a cell mean before any test, and their spread is reported
   separately as the instability metric.

2. PAIR WITHIN VIGNETTE. One male-minus-female difference per vignette per
   model. Each vignette is its own control, so the paired difference removes
   everything about the vignette -- its severity, its wording, its length --
   and leaves only the effect of the name and pronoun swap.

3. NONPARAMETRIC TESTS ONLY. These are bounded 0-10 integer ratings that can
   pile up on one value. Wilcoxon signed-rank for the test, Cliff's delta for
   the effect size, bootstrap over vignettes for the interval. No normality is
   assumed anywhere.

4. HOLM, NOT BONFERRONI. The primary contrasts share vignettes and are
   therefore correlated; Holm controls the familywise error rate without
   assuming independence and is uniformly more powerful at the same alpha.

Usage:
    python src/analyze.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

# Agg before pyplot: this runs headless in CI, where importing an interactive
# backend fails outright.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "protocol.yaml"


def _resolve(value: str) -> Path:
    """Configured paths are repo-relative unless given as absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    Computed by rank rather than by the O(n*m) pairwise loop so it stays cheap
    when called on every condition. Ordinal and bounded, which suits a rating
    scale where the distance between 6 and 7 is not necessarily the distance
    between 2 and 3.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    rank_sum_a = ranks[:n].sum()
    # Mann-Whitney U from the rank sum, then delta = 2U/(nm) - 1.
    u = rank_sum_a - n * (n + 1) / 2
    return float(2.0 * u / (n * m) - 1.0)


def delta_label(delta: float, thresholds: dict[str, float]) -> str:
    """Romano-convention magnitude label. Labels only -- never a decision."""
    if np.isnan(delta):
        return "n/a"
    size = abs(delta)
    if size < thresholds.get("negligible", 0.147):
        return "negligible"
    if size < thresholds.get("small", 0.33):
        return "small"
    if size < thresholds.get("medium", 0.474):
        return "medium"
    return "large"


def bootstrap_ci(
    values: np.ndarray, resamples: int, level: float, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean, resampling vignettes.

    The resampling unit is the vignette, because the vignette set is what the
    study generalises over. Resampling responses instead would treat the 120
    authored items as the population and understate the interval.
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo = (1 - level) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, 100 - lo))


def wilcoxon(differences: np.ndarray, zero_method: str = "wilcox") -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank on the paired differences.

    Returns (statistic, p, n_nonzero). Degenerate inputs -- fewer than 2 pairs,
    or every difference exactly zero -- are reported as p = 1.0 rather than
    allowed to raise, because "the model gave identical answers to both
    renderings" is a real and interesting result, not an error.
    """
    differences = np.asarray(differences, float)
    differences = differences[~np.isnan(differences)]
    nonzero = int(np.count_nonzero(differences))
    if len(differences) < 2 or nonzero == 0:
        return float("nan"), 1.0, nonzero
    try:
        result = stats.wilcoxon(differences, zero_method=zero_method, alternative="two-sided")
    except ValueError:
        return float("nan"), 1.0, nonzero
    return float(result.statistic), float(result.pvalue), nonzero


def holm(pvalues: list[float], alpha: float) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down. Returns [(adjusted_p, reject), ...] in input order.

    Adjusted p-values are made monotone non-decreasing in the sorted order,
    which is what makes "adjusted p < alpha" equivalent to the step-down
    decision rule.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        candidate = min(1.0, (m - rank) * pvalues[i])
        running = max(running, candidate)
        adjusted[i] = running
    return [(adjusted[i], adjusted[i] < alpha) for i in range(m)]


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
def load_scored(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(
            f"error: {path} not found. Run src/score.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    df = pd.read_csv(path)
    if df.empty:
        print(f"error: {path} has no rows.", file=sys.stderr)
        raise SystemExit(1)
    for col in ("fault_scroller", "fault_partner", "severity"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in ("parse_ok", "quote_verbatim", "distress_flagged", "distress_cue"):
        # Cast to the nullable boolean dtype, not object. A column of Python
        # bools in an object-dtype Series makes `~series` call int.__invert__
        # elementwise (~True == -2), which silently turns a rate into nonsense
        # instead of raising. Blank cells stay missing rather than becoming
        # False, so "the model did not answer" never counts as "answered no".
        df[col] = (
            df.get(col)
            .map({True: True, False: False, "True": True, "False": False})
            .astype("boolean")
        )
    df["missing_keys"] = df.get("missing_keys").fillna("")
    return df


def analysable(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the protocol's exclusion rules and say what was dropped."""
    keep = (
        df["parse_ok"].fillna(False).astype(bool)
        & (df["missing_keys"] == "")
        & df["fault_scroller"].notna()
        & df["fault_partner"].notna()
    )
    return df[keep].copy()


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the repeated runs to one row per (model, vignette, condition).

    Step 1 of the pipeline. `sd` is retained here and only here -- it becomes
    the instability metric and is never used as a weight or an error term.
    """
    grouped = df.groupby(
        ["model_slot", "model_id", "vignette_id", "gender", "culture"], dropna=False
    )
    out = grouped.agg(
        fault_scroller=("fault_scroller", "mean"),
        fault_scroller_sd=("fault_scroller", lambda s: s.std(ddof=0)),
        fault_partner=("fault_partner", "mean"),
        severity=("severity", "mean"),
        severity_sd=("severity", lambda s: s.std(ddof=0)),
        n_runs=("fault_scroller", "size"),
    ).reset_index()
    return out


def paired_gaps(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One male-minus-female difference per (model, vignette).

    The two cultural framings are averaged within each gender first, so the
    H1 contrast is a single number per vignette and H2 has the per-culture
    gaps to compare.
    """
    wide = cells.pivot_table(
        index=["model_slot", "vignette_id"],
        columns=["gender", "culture"],
        values=metric,
        aggfunc="mean",
    )
    rows = []
    for (slot, vid), row in wide.iterrows():
        def cell(gender: str, culture: str) -> float:
            try:
                value = row[(gender, culture)]
            except KeyError:
                return float("nan")
            return float(value) if pd.notna(value) else float("nan")

        male = {c: cell("male", c) for c in ("north_american", "south_asian")}
        female = {c: cell("female", c) for c in ("north_american", "south_asian")}
        neutral = {c: cell("neutral", c) for c in ("north_american", "south_asian")}
        male_mean = np.nanmean(list(male.values()))
        female_mean = np.nanmean(list(female.values()))
        rows.append(
            {
                "model_slot": slot,
                "vignette_id": vid,
                "male": male_mean,
                "female": female_mean,
                "neutral": np.nanmean(list(neutral.values())),
                "gap": male_mean - female_mean,
                "gap_north_american": male["north_american"] - female["north_american"],
                "gap_south_asian": male["south_asian"] - female["south_asian"],
            }
        )
    gaps = pd.DataFrame(rows)
    if not gaps.empty:
        # H2 quantity: does the gender gap itself move with cultural framing?
        gaps["gap_culture_diff"] = gaps["gap_north_american"] - gaps["gap_south_asian"]
    return gaps


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    return "n/a" if np.isnan(value) else f"{value:.{digits}f}"


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Markdown table. tabulate when available, hand-rolled otherwise.

    tabulate is a listed dependency, but results.md should still be writable in
    a stripped environment -- an analysis that cannot report is useless.
    """
    body = [[fmt(c) if not isinstance(c, str) else c for c in row] for row in rows]
    try:
        from tabulate import tabulate

        return tabulate(body, headers=headers, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join("---" for _ in headers) + " |"]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
        return "\n".join(lines)


def histogram(gaps: pd.DataFrame, slot: str, model_id: str, out_path: Path) -> None:
    """Fault-gap histogram for one model.

    A single mean and p-value hides the shape that matters: a small mean gap
    from a tight distribution centred near zero is a different finding from a
    small mean gap made of large offsetting swings on individual vignettes.
    The histogram is how that difference stays visible.
    """
    values = gaps["gap"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    if len(values):
        span = max(1.0, float(np.nanmax(np.abs(values))))
        bins = np.linspace(-span, span, 25)
        ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)
        mean = float(np.nanmean(values))
        ax.axvline(0, color="#444444", linewidth=1.0, linestyle="-", label="no gap")
        ax.axvline(
            mean, color="#C44E52", linewidth=1.6, linestyle="--",
            label=f"mean = {mean:+.3f}",
        )
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "no analysable pairs", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
    ax.set_xlabel("paired fault gap  (male minus female, scale points)")
    ax.set_ylabel("vignettes")
    ax.set_title(f"Slot {slot} — {model_id}\nper-vignette fault gap (n={len(values)})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scored", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="results.md path")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config.get("paths", {})
    stat_cfg = config["statistics"]
    alpha = float(stat_cfg["alpha"])
    ci_cfg = stat_cfg["confidence_intervals"]
    resamples = int(ci_cfg["resamples"])
    level = float(ci_cfg["level"])
    seed = int(config["design"].get("seed", 0))
    thresholds = next(
        (e.get("thresholds", {}) for e in stat_cfg["effect_sizes"]
         if e["name"] == "cliffs_delta"),
        {},
    )

    scored_path = args.scored or _resolve(
        paths.get("scored_csv", "results/tables/scored.csv")
    )
    out_path = args.out or _resolve(
        paths.get("results_md", "results/tables/results.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_scored(scored_path)
    good = analysable(df)
    cells = cell_means(good)

    lines: list[str] = []
    w = lines.append
    w(f"# {config['study']['title']}")
    w("")
    w(f"Generated by `src/analyze.py` from `{scored_path.name}`. "
      f"Protocol `config/protocol.yaml` (locked: **{config.get('locked')}**).")
    if config.get("locked") is not True:
        w("")
        w("> **These are smoke-test numbers.** The protocol is not locked, so this "
          "run is not study data.")
    w("")
    w("## 1. Collection and data quality")
    w("")
    n_vignettes = good["vignette_id"].nunique()
    w(md_table(
        ["quantity", "value"],
        [
            ["responses in cache", len(df)],
            ["parsed as JSON", int(df["parse_ok"].fillna(False).astype(bool).sum())],
            ["analysable (all keys, valid ratings)", len(good)],
            ["excluded", len(df) - len(good)],
            ["refusal / unparseable rate",
             fmt(1 - float(df["parse_ok"].fillna(False).astype(bool).mean()))],
            ["vignettes represented", n_vignettes],
            ["design target", f"{config['design']['n_vignettes']} vignettes, "
                              f"{config['design']['total_calls']} calls"],
        ],
    ))
    w("")
    w("Exclusions follow `statistics.exclusions` in the protocol: unparseable "
      "responses, responses missing a required key, and out-of-range ratings.")
    w("")

    # ---- per-model descriptives -------------------------------------------
    w("## 2. Per-model measurement quality")
    w("")
    quality_rows = []
    for slot, sub in good.groupby("model_slot"):
        quoted = sub[sub["quote_match_mode"] != "empty"]
        grounded = quoted["quote_verbatim"].fillna(False).astype(bool)
        halluc = float(1.0 - grounded.mean()) if len(quoted) else float("nan")
        cell_sub = cells[cells["model_slot"] == slot]
        instability = float(cell_sub["fault_scroller_sd"].median()) if len(cell_sub) else float("nan")
        quality_rows.append([
            slot,
            sub["model_id"].iloc[0],
            len(sub),
            fmt(halluc),
            fmt(instability),
            fmt(float(sub["fault_scroller"].mean()), 2),
            fmt(float(sub["fault_partner"].mean()), 2),
            fmt(float(sub["severity"].mean()), 2),
        ])
    w(md_table(
        ["slot", "model", "n", "hallucination rate", "instability (median SD)",
         "mean fault_scroller", "mean fault_partner", "mean severity"],
        quality_rows,
    ))
    w("")
    w("*Hallucination rate* is the share of responses whose `evidence_quote` is "
      "not present verbatim in the vignette it was shown, after unicode and "
      "whitespace normalisation — i.e. the model invented its own supporting "
      "sentence. *Instability* is the median across cells of the SD of "
      "`fault_scroller` over the 5 repeats; at temperature 0 it should be ~0, "
      "and anything larger means the endpoint is not deterministic and the "
      "cell means carry decoding noise.")
    w("")

    # ---- primary tests ----------------------------------------------------
    w("## 3. Primary contrasts (H1, H2)")
    w("")
    w("One paired difference per vignette per model: mean `fault_scroller` in "
      "the male-coded renderings minus the female-coded renderings, after "
      "averaging the 5 repeats within each cell. Wilcoxon signed-rank, "
      "two-sided; Holm-corrected across the whole primary family.")
    w("")

    tests: list[dict[str, Any]] = []
    gap_frames: dict[str, pd.DataFrame] = {}
    for slot, cell_sub in cells.groupby("model_slot"):
        model_id = cell_sub["model_id"].iloc[0]
        gaps = paired_gaps(cell_sub, "fault_scroller")
        gap_frames[slot] = gaps
        if gaps.empty:
            continue
        for label, column, hypothesis in (
            ("fault gap (male - female)", "gap", "H1"),
            ("gap difference (NA - SA framing)", "gap_culture_diff", "H2"),
        ):
            values = gaps[column].dropna().to_numpy()
            stat, p, nonzero = wilcoxon(values, stat_cfg["primary_test"]["zero_method"])
            lo, hi = bootstrap_ci(values, resamples, level, seed + len(tests))
            if column == "gap":
                delta = cliffs_delta(gaps["male"].to_numpy(), gaps["female"].to_numpy())
            else:
                delta = cliffs_delta(
                    gaps["gap_north_american"].to_numpy(),
                    gaps["gap_south_asian"].to_numpy(),
                )
            tests.append({
                "hypothesis": hypothesis, "slot": slot, "model_id": model_id,
                "label": label, "n": len(values), "nonzero": nonzero,
                "mean": float(np.nanmean(values)) if len(values) else float("nan"),
                "median": float(np.nanmedian(values)) if len(values) else float("nan"),
                "ci": (lo, hi), "stat": stat, "p": p, "delta": delta,
            })

    corrected = holm([t["p"] for t in tests], alpha)
    for test, (adjusted, reject) in zip(tests, corrected):
        test["p_holm"], test["reject"] = adjusted, reject

    if tests:
        w(md_table(
            ["H", "slot", "contrast", "n", "mean", "median",
             f"{int(level * 100)}% CI", "W", "p", "p (Holm)", "Cliff's d", "magnitude",
             f"reject at a={alpha}"],
            [[
                t["hypothesis"], t["slot"], t["label"], t["n"],
                fmt(t["mean"]), fmt(t["median"]),
                f"[{fmt(t['ci'][0])}, {fmt(t['ci'][1])}]",
                fmt(t["stat"], 1), fmt_p(t["p"]), fmt_p(t["p_holm"]),
                fmt(t["delta"]), delta_label(t["delta"], thresholds),
                "yes" if t["reject"] else "no",
            ] for t in tests],
        ))
        w("")
        w(f"CIs are percentile bootstrap over vignettes, {resamples} resamples. "
          "A positive mean gap means more fault was assigned to the male-coded "
          "scrolling partner.")
    else:
        w("_No analysable pairs yet._")
    w("")

    # ---- per-culture and neutral ------------------------------------------
    w("## 4. Gaps by cultural framing, and the neutral condition")
    w("")
    culture_rows = []
    for slot, gaps in gap_frames.items():
        if gaps.empty:
            continue
        for column, label in (
            ("gap_north_american", "north_american"),
            ("gap_south_asian", "south_asian"),
        ):
            values = gaps[column].dropna().to_numpy()
            _, p, _ = wilcoxon(values)
            lo, hi = bootstrap_ci(values, resamples, level, seed)
            culture_rows.append([
                slot, label, len(values), fmt(float(np.nanmean(values))) if len(values) else "n/a",
                f"[{fmt(lo)}, {fmt(hi)}]", fmt_p(p),
            ])
        culture_rows.append([
            slot, "neutral mean fault", int(gaps["neutral"].notna().sum()),
            fmt(float(np.nanmean(gaps["neutral"].to_numpy()))), "—", "—",
        ])
    w(md_table(
        ["slot", "framing", "n", "mean gap", f"{int(level * 100)}% CI", "p (uncorrected)"],
        culture_rows,
    ) if culture_rows else "_No analysable pairs yet._")
    w("")
    w("These per-framing tests are descriptive decomposition of H2, not "
      "additional family members; the corrected H2 test is in section 3. The "
      "neutral row is the mean `fault_scroller` in the they/them renderings, "
      "reported so a gap can be read as a shift from, not merely between, the "
      "gendered conditions.")
    w("")

    # ---- action category by culture ---------------------------------------
    w("## 5. Recommended action by cultural framing (H3, exploratory)")
    w("")
    w("> The `action_category` column is a **keyword pre-pass**, not a "
      "validated coding. Per the protocol I hand-code 50 sampled outputs and "
      "report agreement before this chi-square is interpreted. See "
      "`CLASSIFIER_NOTE` in `src/score.py`.")
    w("")
    secondary_p: list[float] = []
    secondary_meta: list[dict[str, Any]] = []
    for slot, sub in good.groupby("model_slot"):
        table = pd.crosstab(sub["action_category"], sub["culture"])
        w(f"**Slot {slot}** — counts by category and framing:")
        w("")
        w(md_table(
            ["action_category"] + [str(c) for c in table.columns],
            [[str(idx)] + [int(v) for v in row] for idx, row in table.iterrows()],
        ))
        w("")
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(table.to_numpy())
            small = int((expected < 5).sum())
            note = (
                f" {small} expected cell(s) below 5 — treat with caution; "
                "the protocol prefers Fisher's exact where computable."
                if small else ""
            )
            w(f"chi-square = {chi2:.3f}, dof = {dof}, p = {fmt_p(p)}.{note}")
            secondary_p.append(p)
            secondary_meta.append({"slot": slot, "test": "action x culture"})
        else:
            w("_Not enough categories or framings present for a chi-square._")
        w("")

    # ---- distress recall --------------------------------------------------
    w("## 6. Distress flagging on cue vignettes (H4, exploratory)")
    w("")
    w("Restricted to vignettes whose id begins with `D`, which embed an "
      "explicit distress cue. There is no gold label on the other vignettes, "
      "so only recall is reported — no precision and no false-positive rate.")
    w("")
    distress = good[good["distress_cue"].fillna(False).astype(bool)]
    if distress.empty:
        w("_No distress-cue vignettes in the cache yet._")
    else:
        rows = []
        for (slot, gender), sub in distress.groupby(["model_slot", "gender"]):
            flags = sub["distress_flagged"].fillna(False).astype(bool)
            rows.append([slot, gender, len(sub), int(flags.sum()), fmt(float(flags.mean()))])
        w(md_table(["slot", "gender", "n", "flagged", "recall"], rows))
        w("")
        for slot, sub in distress.groupby("model_slot"):
            table = pd.crosstab(
                sub["distress_flagged"].fillna(False).astype(bool), sub["gender"]
            )
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                chi2, p, dof, _ = stats.chi2_contingency(table.to_numpy())
                w(f"Slot {slot}: distress_flagged x gender chi-square = {chi2:.3f}, "
                  f"dof = {dof}, p = {fmt_p(p)}.")
                secondary_p.append(p)
                secondary_meta.append({"slot": slot, "test": "distress x gender"})
            else:
                w(f"Slot {slot}: flagging did not vary — no test computable.")
    w("")

    if secondary_p:
        w("### Secondary-family Holm correction")
        w("")
        w(md_table(
            ["slot", "test", "p", "p (Holm, secondary family)", f"reject at a={alpha}"],
            [[m["slot"], m["test"], fmt_p(p), fmt_p(adj), "yes" if rej else "no"]
             for m, p, (adj, rej) in zip(secondary_meta, secondary_p,
                                         holm(secondary_p, alpha))],
        ))
        w("")

    # ---- figures ----------------------------------------------------------
    w("## 7. Figures")
    w("")
    figures: list[str] = []
    for slot, gaps in gap_frames.items():
        model_id = str(cells[cells["model_slot"] == slot]["model_id"].iloc[0])
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_id)[:40]
        png = out_path.parent / f"fault_gap_hist_slot{slot}_{safe}.png"
        histogram(gaps, str(slot), model_id, png)
        figures.append(png.name)
        w(f"![Fault gap histogram, slot {slot}]({png.name})")
        w("")

    w("---")
    w("")
    w("Reproduce with `python src/run.py && python src/score.py && "
      "python src/analyze.py`. The protocol is the source of truth; its commit "
      "timestamp is the evidence that this analysis plan predates these "
      "numbers.")
    w("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    for name in figures:
        print(f"wrote {out_path.parent / name}")
    print(f"  analysable responses: {len(good)}/{len(df)}")
    print(f"  cells: {len(cells)}  models: {cells['model_slot'].nunique()}")
    for test in tests:
        print(
            f"  {test['hypothesis']} slot {test['slot']}: mean={fmt(test['mean'])} "
            f"p={fmt_p(test['p'])} p_holm={fmt_p(test['p_holm'])} "
            f"delta={fmt(test['delta'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
