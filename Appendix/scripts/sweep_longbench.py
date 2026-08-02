# SPDX-License-Identifier: Apache-2.0
"""Sweep KV-cache presses on the LongBench multi-hop QA subset and score token-F1.

Reconstruction from REPORT.md §4.2 (LongBench -- multi-hop QA, F1 %, n=210); the original was lost
with devvm50213.

What it does (faithful to REPORT.md §3.1/§3.2/§4.2):
  * Presses: no_press, knorm, snapkv, centrality_ppr_knorm (the headline config), and
    graphkv_knorm (the GraphKV suppression baseline, injected at runtime via DecayPropagationPress,
    exactly as in additional_benchmarks/run.py -- it is NOT part of the shipped registry).
  * Multi-hop subset: hotpotqa, 2wikimqa, musique (200 examples/task); pooled per-task sample at
    fraction 0.35 -> ~70/task -> n=210, mirroring evaluate.py's df.sample(frac, random_state=seed).
  * Compression ratios 0.5 and 0.75 (no_press is the full-cache ceiling, recorded at every ratio).
  * Generation reuses kvpress's own "kv-press-text-generation" pipeline (greedy decoding), the same
    engine evaluate.py drives -- prefill compresses under `with press(model)`, then greedy decode.
  * Scoring reuses kvpress's OWN metric (benchmarks/longbench/calculate_metrics.py): per-example
    token-F1 = qa_f1_score, taken as the max over the reference answers, x100 (no reimplemented
    scorer). This matches the report's per-example paired-statistics protocol.

Outputs (to --output_dir):
  * longbench_results_long.csv  -- per-example scores, columns: press, ratio, example_id, f1
  * longbench_summary.json      -- mean F1 per press x ratio, with n

Pinned environment (REPORT.md §3.1): meta-llama/Llama-3.1-8B-Instruct, bf16, one A100-80GB, device
cuda:0, attn_implementation=flash_attention_2 if available else sdpa; python>=3.10, torch==2.13.0,
transformers==5.2.0, datasets==5.0.0, kvpress 0.5.4. Numbers are COMPUTED, never hard-coded.

Example:
    python scripts/sweep_longbench.py --output_dir results/longbench_multihop

    # or override the model with a local path (the HF id is gated):
    python scripts/sweep_longbench.py --model /path/to/Llama-3.1-8B-Instruct \
        --output_dir results/longbench_multihop
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TASKS = ["hotpotqa", "2wikimqa", "musique"]  # LongBench multi-hop QA (REPORT §3.2)
DEFAULT_PRESSES = [
    "no_press",
    "knorm",
    "snapkv",
    "centrality_ppr_knorm",  # headline config (REPORT §4.2)
    "graphkv_knorm",  # GraphKV suppression baseline (injected at runtime)
]
DEFAULT_RATIOS = [0.5, 0.75]
DEFAULT_FRACTION = 0.35  # 200/task * 0.35 = 70/task -> n=210 (REPORT §3.2)
DEFAULT_SEED = 42  # evaluate.py default; sampling + decoding determinism


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True, help="Directory for the CSV + JSON outputs.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local path of the model.")
    p.add_argument("--device", default="cuda:0", help="Torch device for the pipeline.")
    p.add_argument("--presses", nargs="+", default=DEFAULT_PRESSES, help="Press names to sweep.")
    p.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS, help="Compression ratios.")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="LongBench task configs to pool.")
    p.add_argument("--fraction", type=float, default=DEFAULT_FRACTION, help="Per-task sampling fraction.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for sampling + determinism.")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override decode length; default uses each task's dataset value (32+20=52 for these).",
    )
    p.add_argument(
        "--repo_root",
        default=None,
        help="kvpress repo root (default: parent of this script's directory).",
    )
    return p.parse_args()


def configure_sys_path(repo_root: Path) -> None:
    """Put the repo root and evaluation/ on sys.path so the library modules import.

    evaluate_registry.py does `from benchmarks.longbench.calculate_metrics import ...` (needs
    evaluation/ on the path) and the GraphKV press lives in additional_benchmarks/ (needs the repo
    root on the path). Mirrors additional_benchmarks/run.py.
    """
    for path in (str(repo_root / "evaluation"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def set_deterministic_seeds(seed: int) -> None:
    """Replicate evaluate.py's _setup_deterministic_seeds for reproducible greedy decoding."""
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


def build_dataset(dataset_path: str, tasks, fraction: float, seed: int):
    """Load the LongBench multi-hop configs, per-task subsample, and pool into one DataFrame.

    Each task is a config of the parquet LongBench mirror (DATASET_REGISTRY['longbench'] =
    'Xnhyacinth/LongBench'); the script-based THUDM/LongBench no longer loads under datasets>=5.
    Sampling mirrors evaluate.py: df.sample(frac=fraction, random_state=seed), applied per task so
    the pooled set is balanced (70/70/70 at fraction 0.35).
    """
    import pandas as pd
    from datasets import load_dataset

    frames = []
    for task in tasks:
        df = load_dataset(dataset_path, name=task, split="test").to_pandas()
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=seed)
        df = df.copy()
        df["task"] = task
        if "_id" in df.columns:
            df["example_id"] = task + "::" + df["_id"].astype(str)
        else:
            df["example_id"] = [f"{task}::{i}" for i in range(len(df))]
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def per_example_f1(dataset2metric, task: str, prediction: str, ground_truths, all_classes) -> float:
    """Per-example token-F1 in %, using kvpress's own metric (max over references).

    Replicates benchmarks/longbench/calculate_metrics.scorer for the multi-hop QA tasks: those are
    not in the trec/triviaqa/samsum newline-split special case, so the prediction is only lstrip'd
    and scored with dataset2metric[task] (= qa_f1_score) against every reference.
    """
    metric = dataset2metric[task]
    prediction = str(prediction).lstrip()
    best = 0.0
    for gt in ground_truths:
        best = max(best, float(metric(prediction, str(gt), all_classes=all_classes)))
    return 100.0 * best


def build_pipeline(model: str, device: str):
    """Load the kv-press-text-generation pipeline in bf16, matching evaluate.py's setup."""
    import torch
    from transformers import pipeline

    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model_kwargs = {"dtype": torch.bfloat16, "attn_implementation": attn_impl}
    pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        model_kwargs=model_kwargs,
        device=device,
        trust_remote_code=True,
    )
    pipe.model.eval()
    return pipe, attn_impl


def evaluate_press_at_ratio(pipe, df, dataset2metric, press, ratio, tqdm, torch):
    """Run every example once for a (press, ratio) and return [(example_id, f1), ...]."""
    if press is not None:
        press.compression_ratio = ratio

    rows = []
    desc = f"{getattr(press, '__class__', type(None)).__name__} r={ratio}"
    with torch.inference_mode():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            max_new_tokens = int(row["max_new_tokens"])
            output = pipe(
                row["context"],
                question=row["question"],
                answer_prefix=row["answer_prefix"],
                press=press,
                max_new_tokens=max_new_tokens,
            )
            f1 = per_example_f1(
                dataset2metric,
                row["task"],
                output["answer"],
                row["answers"],
                row.get("all_classes", None),
            )
            rows.append((row["example_id"], f1))
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parent.parent
    configure_sys_path(repo_root)

    # Heavy / repo-local imports happen after sys.path is configured.
    import torch
    from tqdm import tqdm

    import evaluate_registry
    from benchmarks.longbench.calculate_metrics import dataset2metric
    from kvpress import KnormPress, SnapKVPress

    from additional_benchmarks.decay_propagation_press import DecayPropagationPress

    # Runtime-only injection of the GraphKV suppressor (shipped registry left untouched), exactly as
    # additional_benchmarks/run.py does. Same KnormPress base as centrality_ppr_knorm -> clean pairing.
    press_registry = evaluate_registry.PRESS_REGISTRY
    press_registry.setdefault("graphkv_knorm", DecayPropagationPress(base_press=KnormPress()))
    press_registry.setdefault("graphkv_snapkv", DecayPropagationPress(base_press=SnapKVPress()))

    for name in args.presses:
        if name not in press_registry:
            raise KeyError(f"Press '{name}' not in PRESS_REGISTRY (available: {sorted(press_registry)})")

    dataset_path = evaluate_registry.DATASET_REGISTRY["longbench"]

    set_deterministic_seeds(args.seed)
    pipe, attn_impl = build_pipeline(args.model, args.device)
    df = build_dataset(dataset_path, args.tasks, args.fraction, args.seed)
    # Optional decode-length override (default: each task's own dataset value).
    if args.max_new_tokens is not None:
        df["max_new_tokens"] = args.max_new_tokens

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "longbench_results_long.csv"
    summary_path = output_dir / "longbench_summary.json"

    summary = {
        "benchmark": "longbench_multihop",
        "report_section": "4.2",
        "model": args.model,
        "attn_implementation": attn_impl,
        "tasks": list(args.tasks),
        "fraction": args.fraction,
        "seed": args.seed,
        "n": int(len(df)),
        "results": [],  # one entry per (press, ratio): mean F1 + n
    }

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["press", "ratio", "example_id", "f1"])

    def record(press_name, ratio, rows):
        for example_id, f1 in rows:
            writer.writerow([press_name, ratio, example_id, round(f1, 6)])
        csv_file.flush()
        mean_f1 = sum(f1 for _, f1 in rows) / len(rows) if rows else 0.0
        summary["results"].append(
            {"press": press_name, "ratio": ratio, "mean_f1": round(mean_f1, 2), "n": len(rows)}
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    try:
        for press_name in args.presses:
            press = press_registry[press_name]
            if press_name == "no_press":
                # Full-cache ceiling: compression is a no-op, so run once and record at every ratio.
                base_rows = evaluate_press_at_ratio(pipe, df, dataset2metric, None, 0.0, tqdm, torch)
                for ratio in args.ratios:
                    record(press_name, ratio, base_rows)
            else:
                for ratio in args.ratios:
                    rows = evaluate_press_at_ratio(pipe, df, dataset2metric, press, ratio, tqdm, torch)
                    record(press_name, ratio, rows)
    finally:
        csv_file.close()

    print(f"Wrote per-example scores to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    for entry in summary["results"]:
        print(f"  {entry['press']:<30} r={entry['ratio']:<5} F1={entry['mean_f1']:.2f}  (n={entry['n']})")


if __name__ == "__main__":
    main()
