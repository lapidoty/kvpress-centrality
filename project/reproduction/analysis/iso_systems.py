# SPDX-License-Identifier: Apache-2.0
"""Iso-accuracy systems comparison for CentralityPress vs. its base press (16k ctx, batch 8).

Regenerates REPORT.md §4.5 (Iso-accuracy: memory, throughput and latency at equal quality).

The idea (REPORT.md §4.4/§4.5). At a *fixed* compression ratio every press keeps the same number of
tokens, so decode memory / throughput / latency are ratio-determined -- identical for `knorm`,
`snapkv`, `centrality_pure` and our press; the only press-dependent systems cost is prefill overhead.
The operationally decisive question is therefore the *inverse*: for the same task accuracy, how much
smaller is our cache? This script:

  1. reads the RULER accuracy-vs-compression sweep (a directory of kvpress `evaluate.py` runs -- each
     with a `metrics.json` scored by kvpress's own `calculate_metrics` -- or a pre-aggregated CSV) and
     builds, for the base press and for our press, an accuracy(ratio) curve;
  2. inverts each curve to the compression ratio each press needs to hit a set of *target accuracies*
     (piecewise-linear, with guarded extrapolation past the swept ratios);
  3. at those ratios, on a 16k-token / batch-8 / 128-decode-step workload (the KV-bound regime), reports
     the KV-cache size (MB, computed *exactly* from the model config -- linear in kept length), the
     directly measured peak GPU memory (GB) and decode throughput (tok/s), and decode latency
     percentiles; and
  4. reports the memory saving of `centrality_ppr_knorm` vs. the base at equal accuracy -- the report's
     headline "1.6x-4.5x less KV cache and ~13-16% (~6 GB) lower peak GPU memory".

All systems numbers are COMPUTED (measured on the GPU, or read from a prior systems sweep via
`--systems_csv`) and all accuracy inversions are read from real sweep data -- nothing is hard-coded.

Outputs (to --output_dir):
  * results_iso_systems_16k_bs8.csv -- columns: target_acc, press, ratio, kv_cache_mb, peak_gpu_gb, tok_s
  * iso_systems_summary.json        -- per-target savings (cache reduction x, KV/peak-GPU saved) + the
                                       full-cache reference and the headline saving ranges

Pinned environment (REPORT.md §3.1): meta-llama/Llama-3.1-8B-Instruct (32 layers, 8 KV heads,
head_dim 128, GQA), bf16, one A100-80GB, device cuda:0, attn_implementation=flash_attention_2 if
available else sdpa; python>=3.10, torch==2.13.0, transformers==5.2.0, datasets==5.0.0, kvpress 0.5.4.

Examples:
    # Measure on the GPU, reading the RULER sweep from a directory of evaluate.py run folders:
    python analysis/iso_systems.py --results_dir results/ruler_sweep --output_dir results

    # Same, but read a pre-aggregated accuracy CSV (columns press,ratio,macro):
    python analysis/iso_systems.py --accuracy_csv results/ruler_summary.csv --output_dir results

    # No GPU: reuse a prior systems sweep (ratio -> peak_gpu_gb,tok_s[,p50_ms]) instead of measuring:
    python analysis/iso_systems.py --accuracy_csv results/ruler_summary.csv \
        --systems_csv results/results_speed_memory_bs8.csv --output_dir results
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------------------------
# Pinned defaults (REPORT.md §3.1 / §4.5).
# ------------------------------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BASE_PRESS = "knorm"  # the base press CentralityPress wraps (REPORT §2.1)
DEFAULT_OUR_PRESS = "centrality_ppr_knorm_d0.15"  # headline config (REPORT §4.1)
DEFAULT_TARGETS = [40.0, 50.0, 60.0, 70.0, 80.0]  # RULER macro % targets (REPORT §4.5 table)
DEFAULT_SEQ_LEN = 16384  # 16k prompt -- the KV-bound iso-accuracy regime (REPORT §4.5)
DEFAULT_BATCH = 8
DEFAULT_DECODE_STEPS = 128
DEFAULT_WARMUP_DECODE_STEPS = 8
DEFAULT_SEED = 42  # evaluate.py default; determinism of the synthetic workload
DEFAULT_RATIO_CAP = 0.95  # a compression ratio must stay < 1 (keep at least a few tokens)

# Llama-3.1-8B architecture constants -- used only as a fallback for the analytic KV-cache size when
# neither a loaded model nor AutoConfig is available (the --systems_csv, GPU-free path). These are
# fixed model-architecture facts (REPORT §3.1), NOT experimental results.
LLAMA31_8B_LAYERS = 32
LLAMA31_8B_KV_HEADS = 8
LLAMA31_8B_HEAD_DIM = 128


# ================================================================================================
# sys.path / imports
# ================================================================================================
def configure_sys_path(repo_root: Path) -> None:
    """Put the repo root and evaluation/ on sys.path so evaluate_registry + kvpress import.

    evaluate_registry.py does `from benchmarks.ruler.calculate_metrics import ...` (needs evaluation/
    on the path); the presses live in the top-level kvpress package (needs the repo root). Mirrors
    additional_benchmarks/run.py and scripts/sweep_longbench.py.
    """
    for path in (str(repo_root / "evaluation"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


# ================================================================================================
# Accuracy sweep -> accuracy(ratio) curves
# ================================================================================================
def _macro_from_metrics(metrics: dict) -> Optional[float]:
    """Macro accuracy = mean over tasks of the per-task score in a kvpress metrics.json.

    kvpress's ruler `calculate_metrics` writes {task: {"string_match": score}}; longbench writes
    {task: {"f1": score}} (or {"qa_f1_score": ...}). We average whatever scalar each task carries so
    the same reader works for either benchmark. Returns None if no per-task scalar is found.
    """
    vals: List[float] = []
    for _task, entry in metrics.items():
        if isinstance(entry, dict):
            scalar = next((v for v in entry.values() if isinstance(v, (int, float))), None)
            if scalar is not None:
                vals.append(float(scalar))
        elif isinstance(entry, (int, float)):
            vals.append(float(entry))
    if not vals:
        return None
    return sum(vals) / len(vals)


def curves_from_results_dir(results_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Walk a directory of kvpress evaluate.py runs -> {press_name: [(ratio, macro_acc), ...]}.

    Each run folder holds a metrics.json (scored by kvpress's own calculate_metrics -- we do NOT
    reimplement the scorer, per REPORT §4.1) and a config.yaml giving press_name + compression_ratio.
    Duplicate (press, ratio) runs (e.g. different fractions, or the numbered collision subdirs
    evaluate.py creates) are averaged. Falls back to parsing the "__"-joined folder name when
    config.yaml is absent.
    """
    import yaml  # local import: only needed for this path

    collected: Dict[Tuple[str, float], List[float]] = {}
    for dirpath, _dirnames, filenames in os.walk(results_dir):
        if "metrics.json" not in filenames:
            continue
        run = Path(dirpath)
        try:
            with open(run / "metrics.json") as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        macro = _macro_from_metrics(metrics)
        if macro is None:
            continue

        press_name: Optional[str] = None
        ratio: Optional[float] = None
        cfg_path = run / "config.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                press_name = cfg.get("press_name")
                ratio = cfg.get("compression_ratio")
                if ratio is None and cfg.get("press_name") == "no_press":
                    ratio = 0.0
            except (OSError, yaml.YAMLError):
                pass
        if press_name is None or ratio is None:
            parsed = _parse_run_dirname(run.name)
            press_name = press_name or (parsed[0] if parsed else None)
            ratio = ratio if ratio is not None else (parsed[1] if parsed else None)
        if press_name is None or ratio is None:
            continue
        collected.setdefault((str(press_name), float(ratio)), []).append(macro)

    curves: Dict[str, List[Tuple[float, float]]] = {}
    for (press_name, ratio), macros in collected.items():
        curves.setdefault(press_name, []).append((ratio, sum(macros) / len(macros)))
    for press_name in curves:
        curves[press_name].sort(key=lambda rp: rp[0])
    return curves


def _parse_run_dirname(name: str) -> Optional[Tuple[str, float]]:
    """Best-effort (press_name, ratio) from an evaluate.py "__"-joined folder name.

    Layout (evaluate.py EvaluationConfig.get_results_dir): dataset__[data_dir__]model__press__ratio[__...].
    The press is the token right before the first float-looking token (the ratio).
    """
    parts = name.split("__")
    for i, tok in enumerate(parts):
        try:
            ratio = float(tok)
        except ValueError:
            continue
        if 0.0 <= ratio < 1.0 and i > 0:
            return parts[i - 1], ratio
    return None


def curves_from_csv(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Read an accuracy CSV -> {press_name: [(ratio, macro_acc), ...]}.

    Flexible about column names. Requires a press column ('press'/'press_name'/'method') and a ratio
    column ('ratio'/'compression_ratio'/'compression'). Accuracy is taken from the first present of
    ('macro','macro_acc','accuracy','acc','mean','string_match','score','f1'); if the CSV is long
    (per-task or per-example) the macro is the mean over rows within each (press, ratio) group.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}

    def pick(cands: Sequence[str]) -> Optional[str]:
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    press_col = pick(["press", "press_name", "method"])
    ratio_col = pick(["ratio", "compression_ratio", "compression"])
    acc_col = pick(["macro", "macro_acc", "accuracy", "acc", "mean", "string_match", "score", "f1"])
    if press_col is None or ratio_col is None or acc_col is None:
        raise SystemExit(
            f"Could not find press/ratio/accuracy columns in {csv_path} (have {list(df.columns)}). "
            "Expected a press column, a ratio column, and an accuracy/macro column."
        )

    grouped = df.groupby([press_col, ratio_col])[acc_col].mean().reset_index()
    curves: Dict[str, List[Tuple[float, float]]] = {}
    for _, row in grouped.iterrows():
        curves.setdefault(str(row[press_col]), []).append((float(row[ratio_col]), float(row[acc_col])))
    for press_name in curves:
        curves[press_name].sort(key=lambda rp: rp[0])
    return curves


def dedup_curve(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort by ratio and average any duplicate-ratio accuracy points."""
    by_ratio: Dict[float, List[float]] = {}
    for r, a in points:
        by_ratio.setdefault(round(float(r), 6), []).append(float(a))
    return sorted((r, sum(v) / len(v)) for r, v in by_ratio.items())


# ================================================================================================
# Curve inversion: target accuracy -> compression ratio
# ================================================================================================
def invert_accuracy(
    points: List[Tuple[float, float]],
    target: float,
    ratio_cap: float,
    allow_extrapolate: bool,
) -> Tuple[Optional[float], str]:
    """Find the compression ratio at which accuracy == target (piecewise linear).

    Accuracy is (weakly) decreasing in the compression ratio -- more compression, lower accuracy -- so
    we invert acc(ratio) to ratio(target). Returns (ratio, note). `ratio` is None if the target is
    unattainable even at full cache. The ratio is clamped to [0, ratio_cap]:

      * target between the swept accuracies -> linear interpolation within the bracketing segment;
      * target below the most-compressed accuracy -> the press already beats the target at max swept
        compression, so we extrapolate to a *higher* ratio (more compression, less cache);
      * target above the least-compressed accuracy -> extrapolate to a *lower* ratio (less
        compression), clamped at 0 (full cache) -- attainable iff full cache would meet the target.
    """
    pts = dedup_curve(points)
    if len(pts) == 0:
        return None, "no-data"
    if len(pts) == 1:
        r, a = pts[0]
        return (r, "single-point") if abs(a - target) < 1e-9 else (None, "single-point-mismatch")

    ratios = [p[0] for p in pts]
    accs = [p[1] for p in pts]

    # In-range: first segment that brackets the target.
    for i in range(len(pts) - 1):
        r0, a0 = pts[i]
        r1, a1 = pts[i + 1]
        lo, hi = min(a0, a1), max(a0, a1)
        if lo <= target <= hi:
            if a1 == a0:
                return _clamp(r0, 0.0, ratio_cap), "flat-segment"
            r = r0 + (target - a0) * (r1 - r0) / (a1 - a0)
            return _clamp(r, 0.0, ratio_cap), "interp"

    # Out of range.
    if target < accs[-1]:  # below the most-compressed point -> extrapolate to more compression
        if not allow_extrapolate:
            return _clamp(ratios[-1], 0.0, ratio_cap), "clamped-hi"
        (r0, a0), (r1, a1) = pts[-2], pts[-1]
        if a1 == a0:
            return _clamp(ratios[-1], 0.0, ratio_cap), "flat-tail"
        r = r1 + (target - a1) * (r1 - r0) / (a1 - a0)
        note = "extrapolated" if r <= ratio_cap else "extrapolated-capped"
        return _clamp(r, 0.0, ratio_cap), note

    # target > accs[0]: need less compression than the lightest swept ratio.
    if not allow_extrapolate:
        return _clamp(ratios[0], 0.0, ratio_cap), "clamped-lo"
    (r0, a0), (r1, a1) = pts[0], pts[1]
    if a1 == a0:
        return _clamp(ratios[0], 0.0, ratio_cap), "flat-head"
    r = r0 + (target - a0) * (r1 - r0) / (a1 - a0)
    if r < 0.0:
        # Would need a better-than-full-cache budget -> unattainable at any real ratio.
        return None, "unattainable"
    return _clamp(r, 0.0, ratio_cap), "extrapolated"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ================================================================================================
# Systems: analytic KV-cache size + measured peak GPU / throughput / latency
# ================================================================================================
def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """Bytes of KV cache for ONE token across all layers (keys + values)."""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def kv_cache_mb(kept_len: int, batch: int, per_token_bytes: int) -> float:
    """Exact KV-cache size in MB for `kept_len` tokens at batch `batch` (linear in kept length)."""
    return per_token_bytes * kept_len * batch / (1024.0 * 1024.0)


def kept_len_for_ratio(seq_len: int, ratio: float) -> int:
    """Kept tokens after compression -- matches ScorerPress.compress: int(k_len * (1 - ratio))."""
    return int(seq_len * (1.0 - ratio)) if ratio > 0 else seq_len


def _logits_kwarg(model) -> Optional[str]:
    """Which "keep only the last logits" kwarg this transformers version accepts (avoids a 16k x V
    prefill-logits tensor that would OOM). transformers 5.x uses `logits_to_keep`."""
    params = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def measure_systems(
    model,
    press,  # None for no_press / full cache
    ratio: float,
    seq_len: int,
    batch: int,
    decode_steps: int,
    warmup_decode_steps: int,
    device: str,
) -> Dict[str, float]:
    """Directly measure peak GPU memory, decode throughput and latency percentiles for one operating
    point on a synthetic long-prompt workload (REPORT §4.5: 16k prompt, batch 8, 128 decode steps).

    Prefill runs under `with press(model)` so the cache is compressed exactly as in real inference;
    decode then advances greedily over the compressed cache. KV-cache size is computed analytically
    (exact, linear in kept length) rather than probed, per REPORT §4.5.
    """
    import torch
    from transformers import DynamicCache

    if press is not None and hasattr(press, "compression_ratio"):
        press.compression_ratio = ratio

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    per_token_bytes = kv_bytes_per_token(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        head_dim,
        torch.empty(0, dtype=model.dtype).element_size(),
    )
    kept = kept_len_for_ratio(seq_len, ratio)

    logits_kw = _logits_kwarg(model)

    def forward(input_ids, cache):
        kwargs = {"input_ids": input_ids, "past_key_values": cache, "use_cache": True}
        if logits_kw is not None:
            kwargs[logits_kw] = 1  # only materialize the last position's logits
        return model(**kwargs)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    vocab = int(getattr(cfg, "vocab_size", 32000))
    gen = torch.Generator(device=device).manual_seed(0)
    input_ids = torch.randint(0, vocab, (batch, seq_len), device=device, generator=gen)

    cache = DynamicCache()
    press_ctx = press(model) if press is not None else contextlib.nullcontext()

    with torch.inference_mode():
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with press_ctx:
            out = forward(input_ids, cache)
        torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - t0

        next_tok = out.logits[:, -1:].argmax(dim=-1)
        del out

        # Untimed decode warmup (kernel autotune / allocator warmup at the decode shapes).
        for _ in range(warmup_decode_steps):
            out = forward(next_tok, cache)
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            del out
        torch.cuda.synchronize(device)

        step_ms: List[float] = []
        for _ in range(decode_steps):
            torch.cuda.synchronize(device)
            s = time.perf_counter()
            out = forward(next_tok, cache)
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize(device)
            step_ms.append((time.perf_counter() - s) * 1000.0)
            del out

    peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9
    decode_s = sum(step_ms) / 1000.0
    tok_s = (batch * decode_steps) / decode_s if decode_s > 0 else float("nan")

    step_ms_sorted = sorted(step_ms)
    result = {
        "kv_cache_mb": kv_cache_mb(kept, batch, per_token_bytes),
        "peak_gpu_gb": peak_gpu_gb,
        "tok_s": tok_s,
        "prefill_s": prefill_s,
        "p50_ms": _percentile(step_ms_sorted, 50),
        "p95_ms": _percentile(step_ms_sorted, 95),
        "p99_ms": _percentile(step_ms_sorted, 99),
        "kept_len": kept,
    }

    del cache, input_ids, next_tok
    torch.cuda.empty_cache()
    return result


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# --- GPU-free alternative: interpolate a prior systems sweep (ratio -> metrics) -------------------
def load_systems_sweep(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Read a prior systems sweep CSV -> {metric: [(ratio, value), ...]} for interpolation by ratio.

    Reuses e.g. results/results_speed_memory_bs8.csv (REPORT §4.4). Recognizes a ratio column and any
    of peak_gpu_gb / tok_s / p50_ms / p95_ms / p99_ms (matched case-insensitively).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}
    ratio_col = next((lower[c] for c in ("ratio", "compression_ratio", "compression") if c in lower), None)
    if ratio_col is None:
        raise SystemExit(f"--systems_csv {csv_path} needs a ratio column (have {list(df.columns)}).")

    metrics: Dict[str, List[Tuple[float, float]]] = {}
    for canon in ("peak_gpu_gb", "tok_s", "p50_ms", "p95_ms", "p99_ms", "prefill_s"):
        if canon in lower:
            sub = df[[ratio_col, lower[canon]]].dropna()
            pts = sorted((float(r), float(v)) for r, v in zip(sub[ratio_col], sub[lower[canon]]))
            if pts:
                metrics[canon] = pts
    if "peak_gpu_gb" not in metrics or "tok_s" not in metrics:
        raise SystemExit(f"--systems_csv {csv_path} must provide at least peak_gpu_gb and tok_s columns.")
    return metrics


def interp_systems(
    sweep: Dict[str, List[Tuple[float, float]]],
    ratio: float,
    seq_len: int,
    batch: int,
    per_token_bytes: int,
) -> Dict[str, float]:
    """Interpolate the systems metrics of a prior sweep at `ratio`; KV size stays analytic/exact."""
    import numpy as np

    def at(metric: str) -> float:
        pts = sweep.get(metric)
        if not pts:
            return float("nan")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(np.interp(ratio, xs, ys))  # np.interp clamps outside the swept range

    kept = kept_len_for_ratio(seq_len, ratio)
    return {
        "kv_cache_mb": kv_cache_mb(kept, batch, per_token_bytes),
        "peak_gpu_gb": at("peak_gpu_gb"),
        "tok_s": at("tok_s"),
        "prefill_s": at("prefill_s"),
        "p50_ms": at("p50_ms"),
        "p95_ms": at("p95_ms"),
        "p99_ms": at("p99_ms"),
        "kept_len": kept,
    }


# ================================================================================================
# Model / press setup
# ================================================================================================
def set_deterministic_seeds(seed: int) -> None:
    """Replicate evaluate.py's _setup_deterministic_seeds."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(model_name: str, device: str):
    """Load the model in bf16 on `device`, flash_attention_2 if available else sdpa (REPORT §3.1)."""
    import torch
    from transformers import AutoModelForCausalLM

    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, attn_impl


def resolve_press(press_name: str):
    """Look up a press instance from the shipped PRESS_REGISTRY (None for 'no_press')."""
    import evaluate_registry

    registry = evaluate_registry.PRESS_REGISTRY
    if press_name not in registry:
        raise KeyError(f"Press '{press_name}' not in PRESS_REGISTRY (available: {sorted(registry)}).")
    return registry[press_name]


def analytic_dims(model_name: str, cli_layers, cli_kv_heads, cli_head_dim, cli_dtype_bytes):
    """Resolve (num_layers, num_kv_heads, head_dim, dtype_bytes) without loading model weights.

    Prefer AutoConfig; fall back to CLI overrides / Llama-3.1-8B architecture constants. Used only on
    the GPU-free --systems_csv path (the measured path reads these straight off model.config).
    """
    layers, kv_heads, head_dim = cli_layers, cli_kv_heads, cli_head_dim
    if layers is None or kv_heads is None or head_dim is None:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            layers = layers or cfg.num_hidden_layers
            kv_heads = kv_heads or cfg.num_key_value_heads
            head_dim = head_dim or (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        except Exception:  # noqa: BLE001 -- gated/offline: fall back to the pinned architecture
            layers = layers or LLAMA31_8B_LAYERS
            kv_heads = kv_heads or LLAMA31_8B_KV_HEADS
            head_dim = head_dim or LLAMA31_8B_HEAD_DIM
    return int(layers), int(kv_heads), int(head_dim), int(cli_dtype_bytes)


# ================================================================================================
# CLI
# ================================================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True, help="Directory for the CSV + JSON outputs.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results_dir", default=None, help="Dir of kvpress evaluate.py RULER run folders.")
    src.add_argument("--accuracy_csv", default=None, help="Pre-aggregated accuracy CSV (press,ratio,macro).")

    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local path of the model.")
    p.add_argument("--device", default="cuda:0", help="Torch device for measurement.")
    p.add_argument("--base_press", default=DEFAULT_BASE_PRESS, help="Base press registry name.")
    p.add_argument("--our_press", default=DEFAULT_OUR_PRESS, help="CentralityPress registry name.")
    p.add_argument("--base_acc_label", default=None, help="Base press label in the accuracy source (default: --base_press).")
    p.add_argument("--our_acc_label", default=None, help="Our press label in the accuracy source (default: --our_press).")
    p.add_argument("--targets", nargs="+", type=float, default=DEFAULT_TARGETS, help="Target RULER macro accuracies (%).")

    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="Prompt length (16k = KV-bound regime).")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for the workload.")
    p.add_argument("--decode_steps", type=int, default=DEFAULT_DECODE_STEPS, help="Timed decode steps.")
    p.add_argument("--warmup_decode_steps", type=int, default=DEFAULT_WARMUP_DECODE_STEPS, help="Untimed warmup decode steps.")

    p.add_argument("--ratio_cap", type=float, default=DEFAULT_RATIO_CAP, help="Max compression ratio (< 1).")
    p.add_argument("--no_extrapolate", action="store_true", help="Clamp instead of extrapolating past the swept ratios.")
    p.add_argument("--no_full_cache", action="store_true", help="Skip the full-cache (no_press) reference measurement.")

    p.add_argument("--systems_csv", default=None, help="Reuse a prior systems sweep (ratio->metrics) instead of measuring (no GPU).")
    p.add_argument("--num_layers", type=int, default=None, help="Analytic KV dim override (systems_csv path).")
    p.add_argument("--num_kv_heads", type=int, default=None, help="Analytic KV dim override (systems_csv path).")
    p.add_argument("--head_dim", type=int, default=None, help="Analytic KV dim override (systems_csv path).")
    p.add_argument("--dtype_bytes", type=int, default=2, help="Bytes per KV element (bf16 = 2).")

    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Determinism seed.")
    p.add_argument("--repo_root", default=None, help="kvpress repo root (default: parent of analysis/).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parent.parent
    configure_sys_path(repo_root)

    base_label = args.base_acc_label or args.base_press
    our_label = args.our_acc_label or args.our_press

    # 1. Accuracy curves.
    if args.results_dir is not None:
        curves = curves_from_results_dir(Path(args.results_dir))
        acc_source = f"results_dir:{args.results_dir}"
    else:
        curves = curves_from_csv(Path(args.accuracy_csv))
        acc_source = f"accuracy_csv:{args.accuracy_csv}"

    for label in (base_label, our_label):
        if label not in curves:
            raise SystemExit(
                f"Press '{label}' not found in the accuracy source (found: {sorted(curves)}). "
                "Use --base_acc_label/--our_acc_label to match the sweep's naming."
            )

    # 2. Invert accuracy(ratio) at each target.
    allow_extrap = not args.no_extrapolate
    inverted: Dict[float, Dict[str, dict]] = {}
    for target in args.targets:
        inverted[target] = {}
        for role, label in (("base", base_label), ("ours", our_label)):
            ratio, note = invert_accuracy(curves[label], target, args.ratio_cap, allow_extrap)
            inverted[target][role] = {"label": label, "ratio": ratio, "note": note}

    # 3. Backend for systems metrics: measured on GPU, or interpolated from a prior sweep.
    use_measurement = args.systems_csv is None
    if use_measurement:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("No CUDA device: pass --systems_csv to reuse a prior systems sweep instead.")
        set_deterministic_seeds(args.seed)
        model, attn_impl = build_model(args.model, args.device)
        press_objs = {args.base_press: resolve_press(args.base_press), args.our_press: resolve_press(args.our_press)}

        sys_cache: Dict[Tuple[str, float], Dict[str, float]] = {}

        def systems_at(press_name: str, ratio: float) -> Dict[str, float]:
            key = (press_name, round(ratio, 4))
            if key not in sys_cache:
                press = None if press_name == "no_press" else press_objs[press_name]
                sys_cache[key] = measure_systems(
                    model, press, ratio, args.seq_len, args.batch,
                    args.decode_steps, args.warmup_decode_steps, args.device,
                )
            return sys_cache[key]

        press_name_for = {"base": args.base_press, "ours": args.our_press}
    else:
        attn_impl = "n/a (systems_csv)"
        sweep = load_systems_sweep(Path(args.systems_csv))
        layers, kv_heads, head_dim, dtype_bytes = analytic_dims(
            args.model, args.num_layers, args.num_kv_heads, args.head_dim, args.dtype_bytes
        )
        per_token_bytes = kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes)

        def systems_at(press_name: str, ratio: float) -> Dict[str, float]:  # noqa: ARG001 -- ratio-determined
            return interp_systems(sweep, ratio, args.seq_len, args.batch, per_token_bytes)

        press_name_for = {"base": args.base_press, "ours": args.our_press}

    # 4. Full-cache (no_press) reference.
    full_cache_ref: Optional[Dict[str, float]] = None
    if not args.no_full_cache:
        full_cache_ref = systems_at("no_press", 0.0)

    # 5. Build per-target rows + savings.
    csv_rows: List[dict] = []
    per_target: List[dict] = []
    for target in args.targets:
        entry: Dict[str, object] = {"target_acc": target}
        role_metrics: Dict[str, Optional[Dict[str, float]]] = {}
        for role in ("base", "ours"):
            inv = inverted[target][role]
            ratio = inv["ratio"]
            if ratio is None:
                role_metrics[role] = None
                entry[role] = {"label": inv["label"], "ratio": None, "note": inv["note"], "attainable": False}
                continue
            m = systems_at(press_name_for[role], ratio)
            role_metrics[role] = m
            entry[role] = {
                "label": inv["label"],
                "ratio": ratio,
                "kept_frac": round(1.0 - ratio, 6),
                "note": inv["note"],
                "attainable": True,
                "kv_cache_mb": m["kv_cache_mb"],
                "peak_gpu_gb": m["peak_gpu_gb"],
                "tok_s": m["tok_s"],
                "p50_ms": m["p50_ms"],
                "p95_ms": m["p95_ms"],
                "p99_ms": m["p99_ms"],
                "prefill_s": m["prefill_s"],
            }
            csv_rows.append(
                {
                    "target_acc": target,
                    "press": inv["label"],
                    "ratio": round(ratio, 6),
                    "kv_cache_mb": round(m["kv_cache_mb"], 3),
                    "peak_gpu_gb": round(m["peak_gpu_gb"], 4),
                    "tok_s": round(m["tok_s"], 3),
                }
            )

        base_m, ours_m = role_metrics["base"], role_metrics["ours"]
        if base_m is not None and ours_m is not None:
            entry["savings"] = compute_savings(base_m, ours_m)
        per_target.append(entry)

    # 6. Headline ranges (computed from the per-target savings -- never hard-coded).
    headline = summarize_headline(per_target)

    # 7. Write outputs.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_iso_systems_16k_bs8.csv"
    summary_path = output_dir / "iso_systems_summary.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_acc", "press", "ratio", "kv_cache_mb", "peak_gpu_gb", "tok_s"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    summary = {
        "report_section": "4.5",
        "description": "Iso-accuracy systems comparison (CentralityPress vs base) at 16k ctx, batch 8.",
        "model": args.model,
        "attn_implementation": attn_impl,
        "systems_backend": "measured" if use_measurement else f"interpolated({args.systems_csv})",
        "accuracy_source": acc_source,
        "base_press": args.base_press,
        "our_press": args.our_press,
        "workload": {
            "seq_len": args.seq_len,
            "batch": args.batch,
            "decode_steps": args.decode_steps,
            "warmup_decode_steps": args.warmup_decode_steps,
        },
        "ratio_cap": args.ratio_cap,
        "extrapolate": allow_extrap,
        "targets": list(args.targets),
        "full_cache_reference": full_cache_ref,
        "per_target": per_target,
        "headline": headline,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 8. Console summary.
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Accuracy source: {acc_source}  |  systems: {summary['systems_backend']}  |  attn: {attn_impl}")
    print(
        f"Iso-accuracy ({args.our_press} vs {args.base_press}, 16k/bs{args.batch}) -- KV cache "
        f"{_fmt_range(headline.get('kv_cache_reduction_x_range'), 'x')}, "
        f"KV mem saved {_fmt_range(headline.get('kv_mem_saved_pct_range'), '%')}, "
        f"peak GPU saved {_fmt_range(headline.get('peak_gpu_saved_pct_range'), '%')} "
        f"({_fmt_range(headline.get('peak_gpu_saved_gb_range'), ' GB')})"
    )
    for entry in per_target:
        sv = entry.get("savings")
        if not sv:
            print(f"  target {entry['target_acc']:.0f}%: not attainable by both presses")
            continue
        print(
            f"  target {entry['target_acc']:.0f}%: cache {sv['kv_cache_reduction_x']:.2f}x smaller "
            f"({sv['kv_mem_saved_pct']:.0f}% less KV), peak GPU -{sv['peak_gpu_saved_pct']:.0f}% "
            f"(-{sv['peak_gpu_saved_gb']:.1f} GB), throughput {sv['throughput_change_pct']:+.0f}%, "
            f"p50 latency {sv['latency_p50_change_pct']:+.0f}%"
        )


# ================================================================================================
# Savings + headline aggregation
# ================================================================================================
def compute_savings(base: Dict[str, float], ours: Dict[str, float]) -> Dict[str, float]:
    """Ours-vs-base savings at equal accuracy (REPORT §4.5). Memory saved is positive when ours is
    smaller; throughput/latency change is ours relative to base."""
    kv_base, kv_ours = base["kv_cache_mb"], ours["kv_cache_mb"]
    pk_base, pk_ours = base["peak_gpu_gb"], ours["peak_gpu_gb"]
    tp_base, tp_ours = base["tok_s"], ours["tok_s"]
    return {
        "kv_cache_reduction_x": (kv_base / kv_ours) if kv_ours > 0 else float("inf"),
        "kv_mem_saved_pct": (1.0 - kv_ours / kv_base) * 100.0 if kv_base > 0 else float("nan"),
        "peak_gpu_saved_gb": pk_base - pk_ours,
        "peak_gpu_saved_pct": (1.0 - pk_ours / pk_base) * 100.0 if pk_base > 0 else float("nan"),
        "throughput_change_pct": (tp_ours / tp_base - 1.0) * 100.0 if tp_base > 0 else float("nan"),
        "latency_p50_change_pct": (base["p50_ms"] / ours["p50_ms"] - 1.0) * 100.0
        if ours.get("p50_ms")
        else float("nan"),
    }


def summarize_headline(per_target: List[dict]) -> Dict[str, object]:
    """Min/max of each saving across the attainable targets -- the report's headline ranges."""
    savings = [e["savings"] for e in per_target if e.get("savings")]
    if not savings:
        return {"note": "no target attainable by both presses"}

    def rng(key: str) -> List[float]:
        vals = [s[key] for s in savings if s[key] == s[key]]  # drop NaN
        return [round(min(vals), 3), round(max(vals), 3)] if vals else []

    return {
        "kv_cache_reduction_x_range": rng("kv_cache_reduction_x"),
        "kv_mem_saved_pct_range": rng("kv_mem_saved_pct"),
        "peak_gpu_saved_pct_range": rng("peak_gpu_saved_pct"),
        "peak_gpu_saved_gb_range": rng("peak_gpu_saved_gb"),
        "throughput_change_pct_range": rng("throughput_change_pct"),
        "latency_p50_change_pct_range": rng("latency_p50_change_pct"),
    }


def _fmt_range(rng, unit: str) -> str:
    if not rng:
        return "n/a"
    return f"{rng[0]:g}-{rng[1]:g}{unit}"


if __name__ == "__main__":
    main()
