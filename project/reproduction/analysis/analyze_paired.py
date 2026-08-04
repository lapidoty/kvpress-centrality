# SPDX-License-Identifier: Apache-2.0
"""Paired per-example statistics over two kvpress prediction runs (RULER / LongBench).

Implements the paired per-example comparison (bootstrap CI + Wilcoxon).

At a fixed seed the kvpress evaluation harness scores the *same* examples in the *same* order, so
two ``evaluate.py`` runs (e.g. our press vs. its base press) are **paired** row-for-row. This script
loads two ``predictions.csv`` files, recomputes the per-example score with kvpress's own metric
(RULER ``string_match``, LongBench token-``f1``), and reports, for the difference ``A - B``:

  * the mean per-example score delta (in accuracy percentage points),
  * a bootstrap 95% confidence interval (10,000 resamples), and
  * a Wilcoxon signed-rank p-value.

A comparison is flagged significant (``*``) iff the bootstrap CI excludes 0.

Design notes
------------
* It deliberately **reuses kvpress's own library scorers** rather than reimplementing them
  (§3.2 / §4.1): ``benchmarks.ruler.calculate_metrics.{string_match_all,string_match_part}`` and
  ``benchmarks.longbench.calculate_metrics.qa_f1_score``, applied one example at a time. If those
  modules cannot be imported (a stranger's minimal env may lack ``jieba``/``rouge``/``fuzzywuzzy``,
  which the LongBench module pulls in), it falls back to byte-identical inline copies so the numbers
  are unchanged.
* The gold references in ``predictions.csv`` are serialized numpy-array reprs such as
  ``"['DCQHV' 'UQBFO' 'EJNEL']"`` -- space-separated, no commas, sometimes line-wrapped. Parsing
  them with ``ast.literal_eval`` silently *glues* the space-separated string literals into a single
  token (``'DCQHVUQBFO'``), which would zero every multi-reference task. ``parse_refs`` below extracts each quoted token individually to avoid it.

Example
-------
    python analyze_paired.py \\
        --a results/.../centrality_ppr_knorm_d0.15__0.50.../predictions.csv \\
        --b results/.../knorm__0.50.../predictions.csv \\
        --metric string_match --out results/paired_ppr_vs_knorm_c0.5.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Pinned defaults (bootstrap CI + Wilcoxon).
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_CI = 0.95
DEFAULT_SEED = 42  # project seed; makes the bootstrap CI reproducible run-to-run

# Control characters stripped from predictions before RULER string-match scoring -- identical to the
# regex in ``benchmarks/ruler/calculate_metrics.py``.
_CTRL_CHARS = re.compile(r"[\x00-\x1f]")

# One quoted token: single- or double-quoted, honouring backslash escapes.
_QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


# --------------------------------------------------------------------------------------------------
# Reference parsing (the §3.2 numpy-repr fix)
# --------------------------------------------------------------------------------------------------
def parse_refs(raw: object) -> List[str]:
    """Parse a ``predictions.csv`` gold-answer cell into a list of reference strings.

    Handles: an already-materialized list/tuple/ndarray; a numpy-array repr with *space*-separated
    quoted tokens (``"['A' 'B']"``, possibly line-wrapped); a Python-list repr with commas
    (``"['a', 'b']"``); and a bare scalar string. Crucially it extracts each quoted token on its own
    instead of ``ast.literal_eval``-ing the whole bracketed string, which would concatenate
    space-separated string literals into one glued reference (the §3.2 bug).
    """
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(r) for r in list(raw)]
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        quoted = _QUOTED.findall(inner)
        if quoted:
            # findall yields (single_group, double_group) tuples; exactly one is non-empty.
            return [a if a != "" else b for a, b in quoted]
        # No quoted tokens (e.g. a numeric list): safe to literal_eval, else split on commas.
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, np.ndarray)):
                return [str(v) for v in list(val)]
            return [str(val)]
        except (ValueError, SyntaxError):
            return [t.strip().strip("'\"") for t in inner.split(",") if t.strip()]
    return [s]


# --------------------------------------------------------------------------------------------------
# Metric functions -- prefer kvpress's library scorers; fall back to byte-identical inline copies
# --------------------------------------------------------------------------------------------------
def _inline_string_match_all(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    score = (
        sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref) for p, ref in zip(preds, refs))
        / len(preds)
        * 100
    )
    return round(score, 2)


def _inline_string_match_part(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    score = (
        sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref) for p, ref in zip(preds, refs))
        / len(preds)
        * 100
    )
    return round(score, 2)


def _inline_normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        import string

        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _inline_qa_f1_score(prediction: str, ground_truth: str, **_: object) -> float:
    pred_tokens = _inline_normalize_answer(prediction).split()
    gt_tokens = _inline_normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _find_eval_dir(override: str | None) -> Path | None:
    """Locate the kvpress ``evaluation/`` dir (which holds the ``benchmarks`` package)."""
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "evaluation")
        candidates.append(parent)  # in case cwd already is evaluation/
    seen = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "benchmarks" / "ruler" / "calculate_metrics.py").exists():
            return cand
    return None


def resolve_scorers(metric: str, eval_dir_override: str | None) -> tuple[dict, str]:
    """Return the metric callables, preferring kvpress's library implementations."""
    eval_dir = _find_eval_dir(eval_dir_override)
    if eval_dir is not None and str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    if metric == "string_match":
        try:
            from benchmarks.ruler.calculate_metrics import string_match_all, string_match_part

            return {"all": string_match_all, "part": string_match_part}, "kvpress.benchmarks.ruler"
        except Exception:  # noqa: BLE001 -- any import failure -> use the identical inline copy
            return {"all": _inline_string_match_all, "part": _inline_string_match_part}, "inline(ruler)"

    # metric == "f1"
    try:
        from benchmarks.longbench.calculate_metrics import qa_f1_score

        return {"f1": qa_f1_score}, "kvpress.benchmarks.longbench"
    except Exception:  # noqa: BLE001
        return {"f1": _inline_qa_f1_score}, "inline(longbench)"


def per_example_scores(df: pd.DataFrame, metric: str, ref_col: str, scorers: dict) -> np.ndarray:
    """Score every row, on a 0-100 scale, reusing the library metric one example at a time."""
    preds = df["predicted_answer"].tolist()
    refs_raw = df[ref_col].tolist()
    tasks = df["task"].tolist() if "task" in df.columns else [""] * len(df)
    out = np.empty(len(df), dtype=float)

    for i, (pred, raw, task) in enumerate(zip(preds, refs_raw, tasks)):
        refs = parse_refs(raw)
        if not refs:
            out[i] = 0.0
            continue
        if metric == "string_match":
            # calculate_metrics strips control chars from the prediction before matching.
            p = _CTRL_CHARS.sub("", str(pred).strip()).strip()
            fn = scorers["part"] if str(task).split("_")[0] == "qa" else scorers["all"]
            out[i] = float(fn([p], [refs]))
        else:  # f1 -- scorer() lstrips the prediction, then maxes over references
            p = str(pred).lstrip()
            out[i] = 100.0 * max(float(scorers["f1"](p, str(g))) for g in refs)
    return out


# --------------------------------------------------------------------------------------------------
# Pairing + statistics
# --------------------------------------------------------------------------------------------------
def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two runs row-for-row on the examples they share.

    A fixed seed yields identical example sets in identical order, so pairing is positional; we make
    it order-robust by sorting both frames on their shared identity columns (everything except the
    run-dependent ``predicted_answer`` / ``compression_ratio``) and then verifying the example keys
    match exactly.
    """
    if len(a) != len(b):
        raise SystemExit(
            f"Row-count mismatch: A has {len(a)} rows, B has {len(b)}. Paired stats require the same "
            "examples (same dataset / seed / fraction)."
        )
    id_cols = [c for c in a.columns if c in b.columns and c not in {"predicted_answer", "compression_ratio"}]
    if not id_cols:
        return a.reset_index(drop=True), b.reset_index(drop=True)

    ka = a[id_cols].astype(str).agg("\x1f".join, axis=1)
    kb = b[id_cols].astype(str).agg("\x1f".join, axis=1)
    a2 = a.assign(_k=ka).sort_values("_k", kind="stable").reset_index(drop=True)
    b2 = b.assign(_k=kb).sort_values("_k", kind="stable").reset_index(drop=True)
    if not np.array_equal(a2["_k"].to_numpy(), b2["_k"].to_numpy()):
        raise SystemExit(
            "The two files do not contain the same examples; cannot pair. Ensure both runs used the "
            "same dataset, seed and fraction."
        )
    return a2.drop(columns="_k"), b2.drop(columns="_k")


def bootstrap_ci(delta: np.ndarray, n_boot: int, ci: float, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``delta`` (chunked to bound memory)."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    alpha = (1.0 - ci) / 2.0
    boots = np.empty(n_boot, dtype=float)
    chunk = 2000
    for i in range(0, n_boot, chunk):
        j = min(i + chunk, n_boot)
        idx = rng.integers(0, n, size=(j - i, n))
        boots[i:j] = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank p-value on the paired samples."""
    if np.allclose(a - b, 0.0):
        # scipy raises when every difference is zero; there is no evidence of a shift.
        return float("nan"), 1.0
    try:
        stat, p = wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), 1.0


def _resolve_predictions_path(p: str) -> Path:
    path = Path(p)
    if path.is_dir():
        path = path / "predictions.csv"
    if not path.exists():
        raise SystemExit(f"predictions file not found: {path}")
    return path


def _normalize_metric(metric: str) -> str:
    m = metric.strip().lower().replace("-", "_")
    if m in {"string_match", "stringmatch", "sm", "ruler"}:
        return "string_match"
    if m in {"f1", "qa_f1", "qa_f1_score", "longbench", "token_f1"}:
        return "f1"
    raise SystemExit(f"unknown --metric '{metric}'; expected 'string_match' or 'f1'")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Paired per-example statistics (bootstrap CI + Wilcoxon) over two kvpress "
        "predictions.csv files. Reports A - B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a", required=True, help="predictions.csv (or run dir) for method A")
    parser.add_argument("--b", required=True, help="predictions.csv (or run dir) for the reference B")
    parser.add_argument("--metric", default="string_match", help="'string_match' (RULER) or 'f1' (LongBench)")
    parser.add_argument("--out", default="paired_stats.json", help="output path (.json; a sibling .csv is also written)")
    parser.add_argument("--a-label", default=None, help="label for A in the summary (default: A's run-dir name)")
    parser.add_argument("--b-label", default=None, help="label for B in the summary (default: B's run-dir name)")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="bootstrap resamples")
    parser.add_argument("--ci", type=float, default=DEFAULT_CI, help="CI level (0-1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap RNG seed")
    parser.add_argument("--kvpress-eval-dir", default=None, help="override path to kvpress evaluation/ dir")
    args = parser.parse_args(argv)

    metric = _normalize_metric(args.metric)
    a_path = _resolve_predictions_path(args.a)
    b_path = _resolve_predictions_path(args.b)
    a_label = args.a_label or a_path.resolve().parent.name
    b_label = args.b_label or b_path.resolve().parent.name

    scorers, scorer_src = resolve_scorers(metric, args.kvpress_eval_dir)

    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    a_df, b_df = align(a_df, b_df)

    ref_col = "answer" if metric == "string_match" else "answers"
    if ref_col not in a_df.columns:
        ref_col = "answers" if ref_col == "answer" else "answer"
    if ref_col not in a_df.columns:
        raise SystemExit(f"no gold-answer column ('answer'/'answers') in {a_path}")

    scores_a = per_example_scores(a_df, metric, ref_col, scorers)
    scores_b = per_example_scores(b_df, metric, ref_col, scorers)
    delta = scores_a - scores_b

    mean_delta = float(np.mean(delta))
    ci_low, ci_high = bootstrap_ci(delta, args.n_bootstrap, args.ci, args.seed)
    w_stat, w_p = wilcoxon_signed_rank(scores_a, scores_b)
    significant = bool(ci_low > 0.0 or ci_high < 0.0)  # significant iff the CI excludes 0
    marker = "*" if significant else ""

    result = {
        "a": str(a_path),
        "b": str(b_path),
        "a_label": a_label,
        "b_label": b_label,
        "metric": metric,
        "scorer_source": scorer_src,
        "n": int(len(delta)),
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(delta)),
        "ci_level": args.ci,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "significant": significant,
        "sig_marker": marker,
    }

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".csv":
        json_path, csv_path = out_path.with_suffix(".json"), out_path
    else:
        json_path = out_path if out_path.suffix.lower() == ".json" else out_path.with_suffix(".json")
        csv_path = json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    per_example = pd.DataFrame(
        {
            "task": a_df["task"].to_numpy() if "task" in a_df.columns else np.full(len(delta), ""),
            "score_a": scores_a,
            "score_b": scores_b,
            "delta": delta,
        }
    )
    per_example.to_csv(csv_path, index=False)

    print(
        f"[{metric}] {a_label} vs {b_label}: mean Δ = {mean_delta:+.2f}{marker}  "
        f"{int(args.ci * 100)}% CI [{ci_low:+.2f}, {ci_high:+.2f}]  Wilcoxon p = {w_p:.3g}  "
        f"(n={len(delta)}; mean A={result['mean_a']:.2f}, B={result['mean_b']:.2f})  -> {json_path}"
    )


if __name__ == "__main__":
    main()
