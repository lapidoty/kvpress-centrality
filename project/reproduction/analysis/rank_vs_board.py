# SPDX-License-Identifier: Apache-2.0
"""Rank our press against the public KVPress leaderboard at matched model + matched ratio (Llama-3.1-8B).

For each ratio we rank every board entry (`kvpress_leaderboard_raw.csv`, the submission-time board snapshot),
insert our full-fraction scores (from `results/board_grid/`), and report where the base (Knorm) and our
reinforced press land. Reproduces the report section 4.1 leaderboard table: rank jump 19->8, 20->12, 20->13,
19->11 at c=0.25/0.5/0.75/0.875, and `fwe` 1st at 0.25 (2nd at 0.5)."""
import glob
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BOARD = HERE / "kvpress_leaderboard_raw.csv"
GRID = str(HERE.parent / "results" / "board_grid" / "*centrality_ppr_knorm_d0.15__{r}" / "metrics.json")
SUB = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
       "niah_multikey_3", "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe", "qa_1", "qa_2"]
CATS = {"macro13": SUB, "fwe": ["fwe"], "cwe": ["cwe"]}
RATIOS = [(0.25, "0.25"), (0.5, "0.50"), (0.75, "0.75"), (0.875, "0.875")]


def ours(rstr):
    m = json.load(open(glob.glob(GRID.format(r=rstr))[0]))
    return {s: m[s]["string_match"] for s in SUB}


board = pd.read_csv(BOARD)
L = board[board.model == "meta-llama/Llama-3.1-8B-Instruct"]
for r, rstr in RATIOS:
    sub = L[L.configured_ratio == r]
    o = ours(rstr)
    print(f"\n=== c={r}  ({len(sub) + 1} board entries incl. ours) ===")
    for cat, cols in CATS.items():
        field = [(row.pretty_name, row[cols].mean()) for _, row in sub.iterrows()]
        om = sum(o[c] for c in cols) / len(cols)
        ranked = sorted(field + [("+ centrality (ours)", om)], key=lambda kv: -kv[1])
        orr = next(i for i, (n, s) in enumerate(ranked, 1) if "ours" in n)
        line = f"  {cat:8s} ours={om:5.1f}  rank {orr}/{len(ranked)}"
        if cat == "macro13":
            kr = next(i for i, (n, s) in enumerate(ranked, 1) if n == "Knorm")
            kn = next(s for n, s in field if n == "Knorm")
            line += f"   | base Knorm rank {kr} ({kn:.1f})  ->  +{om - kn:.1f}"
        print(line)
