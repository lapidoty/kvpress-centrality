# SPDX-License-Identifier: Apache-2.0
"""Rank our press against the public KVPress leaderboard at MATCHED model + MATCHED ratio.

Inserts our fraction-0.06 per-subtask scores into the board's Llama per-ratio ranking (macro over the
13 RULER subtasks -- the board's metric, at a single ratio, never averaged). To keep the comparison set
identical across ratios, we rank only over the methods the board evaluates at ALL THREE ratios (plus ours).
This reproduces the report's section 4.1 table: macro 8/11/12 of 20; fwe 1/1/9; cwe 18/15/15."""
import glob
import json

import pandas as pd

BOARD = "/home/lapidoty/kv-dev/kvpress_leaderboard_raw.csv"
OURS_GLOB = "/home/lapidoty/kv-dev/eval_results/*centrality_ppr_knorm_d0.15__{r}__fraction0.060/metrics.json"
SUB = [
    "niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
    "niah_multikey_3", "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe", "qa_1", "qa_2",
]
CATS = {
    "macro13": SUB,
    "fwe": ["fwe"],
    "cwe": ["cwe"],
    "qa": ["qa_1", "qa_2"],
    "niah_multikey": ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"],
}
RATIOS = [(0.25, "0.25"), (0.5, "0.50"), (0.75, "0.75")]

board = pd.read_csv(BOARD)
L = board[board.model == "meta-llama/Llama-3.1-8B-Instruct"]
# Consistent comparison set: methods present at ALL three ratios (so the denominator does not move).
present = {r: set(L[L.configured_ratio == r].press_name) for r, _ in RATIOS}
common = present[0.25] & present[0.5] & present[0.75]


def ours(rstr):
    m = json.load(open(glob.glob(OURS_GLOB.format(r=rstr))[0]))
    return {s: m[s]["string_match"] for s in SUB}


print(f"comparison set: {len(common)} board methods present at all ratios  (+ours = {len(common) + 1})\n")
for r, rstr in RATIOS:
    sub = L[(L.configured_ratio == r) & (L.press_name.isin(common))].drop_duplicates("press_name")
    o = ours(rstr)
    print(f"=== ratio {r} ({len(sub) + 1} entries incl. ours) ===")
    for cat, cols in CATS.items():
        scores = [row[cols].mean() for _, row in sub.iterrows()]
        om = sum(o[s] for s in cols) / len(cols)
        scores.append(om)
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(len(scores) - 1) + 1
        print(f"  {cat:14s} ours={om:5.1f}  rank {rank}/{len(scores)}")
