# SPDX-License-Identifier: Apache-2.0
"""Llama board grid for the leaderboard submission: centrality_ppr_knorm on RULER/4096 at the four board
ratios (0.25 / 0.50 / 0.75 / 0.875), FULL fraction, flash_attention_2 (auto-selected when flash-attn is
installed). Sequential on a single A100 (~2.3 h/run).

Override the model / output dir with env vars:  MODEL=<hf-id>  OUT=<dir>  python submission_grid_llama.py
"""
import os
import subprocess
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "evaluation"
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
OUT = os.environ.get("OUT", "submission_results")
(EVAL / OUT).mkdir(parents=True, exist_ok=True)

for ratio in ("0.25", "0.50", "0.75", "0.875"):
    print(f">>> llama c={ratio}")
    subprocess.run([sys.executable, "evaluate.py", "--dataset", "ruler", "--data_dir", "4096",
                    "--model", MODEL, "--press_name", "centrality_ppr_knorm",
                    "--compression_ratio", ratio, "--output_dir", OUT], cwd=EVAL, check=True)
print("board grid done")
