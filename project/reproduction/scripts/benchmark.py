# SPDX-License-Identifier: Apache-2.0
"""Benchmark CentralityPress end-to-end in one command, fast (RULER fraction 0.06).

Runs the report's benchmarks and writes one run directory (config.yaml + metrics.json) per (press, ratio):
  * RULER accuracy (report section 4.1): centrality vs no_press / knorm / snapkv / centrality_pure
  * RULER GraphKV suppressor (section 4.5): graphkv_knorm, injected at runtime by additional_benchmarks/run.py

Override the model / output dir with env vars:  MODEL=<hf-id>  OUT=<dir>  python benchmark.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # scripts -> reproduction -> project -> repo root
EVAL = REPO / "evaluation"
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
OUT = os.environ.get("OUT", "out")
FRACTION = "0.06"                            # ~30 examples/task; fast screen that matches the report tables


def evaluate(script, press, ratio):
    cmd = [sys.executable, str(script), "--dataset", "ruler", "--data_dir", "4096",
           "--model", MODEL, "--press_name", press, "--compression_ratio", ratio,
           "--fraction", FRACTION, "--output_dir", OUT]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=EVAL, check=True)


# 1. RULER accuracy (section 4.1): centrality vs baselines
evaluate("evaluate.py", "no_press", "0.0")
for press in ("knorm", "snapkv", "centrality_ppr_knorm", "centrality_pure"):
    for ratio in ("0.25", "0.5", "0.75"):
        evaluate("evaluate.py", press, ratio)

# 2. RULER GraphKV suppressor (section 4.5): graphkv_knorm, injected at runtime
run_py = REPO / "project" / "additional_benchmarks" / "run.py"
for ratio in ("0.25", "0.5", "0.75"):
    evaluate(run_py, "graphkv_knorm", ratio)

print(f"\nDone. Per-run config.yaml + metrics.json under evaluation/{OUT}/ ;")
print(f"Cross-method board ranks: python {HERE.parent / 'analysis' / 'rank_vs_board.py'}")
