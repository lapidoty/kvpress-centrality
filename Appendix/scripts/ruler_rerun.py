# SPDX-License-Identifier: Apache-2.0
"""
Corrected RULER re-run. Uses kvpress's kv-press-text-generation pipeline (which ALREADY applies the
Llama-3.1 chat template) PER EXAMPLE — each example gets its own answer_prefix and max_new_tokens,
fixing the original sweep's per-context .iloc[0] handling that zeroed several tasks. Scores every task
(incl. the previously-excluded vt / multivalue / multiquery / cwe / fwe / qa_1) with RULER's own
string_match metric, and writes per-example rows (with predictions) for diagnosis + paired stats.

Usage:
  ruler_rerun.py --mode diagnostic --ctx 4096 --per-task 15                 # no_press ceilings, all tasks
  ruler_rerun.py --mode sweep --ctx 4096 --per-task 30 --ratios 0.25,0.5,0.75
"""
import argparse
import ast
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline

from kvpress import CentralityPress, KnormPress, SnapKVPress

# The GraphKV baseline (DecayPropagationPress) lives in additional_benchmarks/, not the shipped kvpress
# package -- put the repo root on sys.path so it imports, mirroring additional_benchmarks/run.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from additional_benchmarks.decay_propagation_press import DecayPropagationPress  # noqa: E402

MODEL = "/home/lapidoty/models/Llama-3.1-8B-Instruct"
OUTDIR = "/home/lapidoty/kv-dev/rerun_results"
SEED = 42
ALL_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
             "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
CTRL = re.compile(r"[\x00-\x1f]")


def parse_refs(raw):
    # RULER answers are lists/np-arrays of reference strings. NOTE: never ast.literal_eval a numpy
    # repr like "['A' 'B']" -- Python concatenates the adjacent string literals into one glued token
    # ('AB'), silently zeroing every multi-answer task. Take the raw sequence, or extract quoted tokens.
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(x) for x in raw]
    s = str(raw).strip()
    q = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)  # handles space- AND comma-separated quoted tokens
    q = [a or b for a, b in q]
    if q:
        return q
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
    except (ValueError, SyntaxError):
        return [s.strip("[]'\" ")]


def example_score(pred, refs, task):
    pred = CTRL.sub("", str(pred).strip()).strip().lower()
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    if task.split("_")[0] == "qa":
        return max(hits) if hits else 0.0
    return (sum(hits) / len(hits)) if hits else 0.0


def make_press(kind, ratio, d):
    return {
        "no_press": lambda: None,
        "knorm": lambda: KnormPress(compression_ratio=ratio),
        "snapkv": lambda: SnapKVPress(compression_ratio=ratio),
        "graphkv_knorm": lambda: DecayPropagationPress(base_press=KnormPress(), compression_ratio=ratio),
        "pure": lambda: CentralityPress(base_press=None, damping=1.0, compression_ratio=ratio),
        "ppr_knorm": lambda: CentralityPress(base_press=KnormPress(), damping=d, compression_ratio=ratio),
    }[kind]()


def sample_df(ctx, per_task):
    df = load_dataset("simonjegou/ruler", data_dir=str(ctx), split="test").to_pandas()
    parts = []
    for t in ALL_TASKS:
        sub = df[df.task == t]
        if len(sub):
            parts.append(sub.sample(n=min(per_task, len(sub)), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def run_config(pipe, df, label, kind, ratio, d):
    press = make_press(kind, ratio, d)
    rows = []
    with torch.inference_mode():
        for i, r in df.iterrows():
            ans = pipe(r["context"], question=r["question"], answer_prefix=r["answer_prefix"],
                       press=press, max_new_tokens=int(r["max_new_tokens"]))["answer"]
            rows.append({"context_len": df._ctx, "label": label, "kind": kind, "ratio": ratio,
                         "damping": d, "task": r["task"], "idx": int(i),
                         "score": example_score(ans, parse_refs(r["answer"]), r["task"]),
                         "predicted_answer": ans, "answer": r["answer"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diagnostic", "sweep"], default="diagnostic")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--per-task", type=int, default=15)
    ap.add_argument("--ratios", default="0.5")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    print("loading model...", flush=True)
    pipe = pipeline("kv-press-text-generation", model=MODEL, device="cuda:0",
                    model_kwargs={"dtype": torch.bfloat16})
    pipe.model.eval()
    print("chat_template present:", pipe.tokenizer.chat_template is not None, flush=True)

    df = sample_df(args.ctx, args.per_task)
    df._ctx = args.ctx
    print(f"{len(df)} examples across {df.task.nunique()} tasks (ctx {args.ctx})", flush=True)
    ratios = [float(x) for x in args.ratios.split(",")]

    if args.mode == "diagnostic":
        configs = [("no_press", "no_press", 0.0, None)]
    else:
        base = [("knorm", "knorm", None), ("snapkv", "snapkv", None), ("graphkv_knorm", "graphkv_knorm", None),
                ("pure", "pure", None), ("ppr_knorm_d0.15", "ppr_knorm", 0.15), ("ppr_knorm_d0.3", "ppr_knorm", 0.3)]
        configs = [("no_press", "no_press", 0.0, None)]
        for r in ratios:
            configs += [(f"{lab}@{r}", kind, r, d) for lab, kind, d in base]

    out_csv = f"{OUTDIR}/ruler_rerun_{args.mode}_ctx{args.ctx}.csv"
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, usecols=["label"]).label.unique())
        print(f"resume: {len(done)} configs already done", flush=True)
    for lab, kind, ratio, d in configs:
        if lab in done:
            print(f"skip {lab} (done)", flush=True)
            continue
        t0 = time.time()
        res = run_config(pipe, df, lab, kind, ratio, d)
        res.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        pt = res.groupby("task").score.mean().mul(100).round(1).to_dict()
        print(f"DONE {lab:20s} overall={res.score.mean()*100:5.1f}%  ({time.time()-t0:.0f}s)  per-task={pt}", flush=True)
    print("wrote", out_csv, "\nRERUN_DONE", flush=True)


if __name__ == "__main__":
    main()
