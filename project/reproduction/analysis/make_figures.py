# SPDX-License-Identifier: Apache-2.0
"""Regenerate the report figures (fig1..fig12) from the result files (REPORT.md §4 captions).

Reconstruction from the report methodology; the original was lost with devvm50213. Not yet re-executed.
Each figure reads its input CSV/JSON from --resultsdir and is skipped (with a warning) if the input is
absent — so this runs incrementally as sections are re-produced.

Inputs expected (produced by the sibling scripts):
  ruler_results_long.csv / ruler_summary.json  (fig1, fig2, fig4, fig9, fig11)
  longbench_summary.json                        (fig3)
  results_systems_bs8.csv                       (fig5, fig6, fig7, fig8)
  results_iso_systems_16k_bs8.csv               (fig10)
  results_attention_recall.csv                  (fig12)

Usage: python analysis/make_figures.py --resultsdir results/ --figdir report/figures/
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OURS = "centrality_ppr_knorm_d0.15"


def _load_csv(d, name):
    p = os.path.join(d, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _load_json(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name), bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", name)


def fig1_ruler(rd, fg):  # RULER macro vs compression
    s = _load_json(rd, "ruler_summary.json")
    if not s:
        print("skip fig1 (no ruler_summary.json)"); return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for press, curve in s.items():          # {press: {ratio: macro}}
        xs = sorted(float(r) for r in curve)
        ax.plot(xs, [curve[str(x)] for x in xs], marker="o", label=press)
    ax.set(xlabel="compression ratio", ylabel="RULER macro (13-task)")
    ax.legend(fontsize=6); _save(fig, fg, "fig1_ruler_accuracy_vs_compression.png")


def fig3_longbench(rd, fg):
    s = _load_json(rd, "longbench_summary.json")
    if not s:
        print("skip fig3 (no longbench_summary.json)"); return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for press, curve in s.items():
        xs = sorted(float(r) for r in curve)
        ax.plot(xs, [curve[str(x)] for x in xs], marker="o", label=press)
    ax.set(xlabel="compression ratio", ylabel="LongBench multi-hop F1")
    ax.legend(fontsize=6); _save(fig, fg, "fig3_longbench_f1.png")


def _systems(rd, fg):
    df = _load_csv(rd, "results_systems_bs8.csv")
    if df is None:
        print("skip fig5-8 (no results_systems_bs8.csv)"); return
    # fig5 prefill overhead
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(df["press"], df["prefill_s"]); ax.set_ylabel("prefill (s)"); plt.xticks(rotation=45, ha="right", fontsize=6)
    _save(fig, fg, "fig5_prefill_overhead.png")
    # fig6 latency percentiles
    fig, ax = plt.subplots(figsize=(5, 3))
    for col, lab in [("lat_mean_ms", "mean"), ("lat_p95_ms", "p95"), ("lat_p99_ms", "p99")]:
        if col in df: ax.plot(df["press"], df[col], marker="o", label=lab)
    ax.set_ylabel("decode latency (ms/token)"); ax.legend(fontsize=6); plt.xticks(rotation=45, ha="right", fontsize=6)
    _save(fig, fg, "fig6_latency_percentiles.png")
    # fig8 gpu util
    if "gpu_util_pct" in df:
        fig, ax = plt.subplots(figsize=(5, 3)); ax.bar(df["press"], df["gpu_util_pct"])
        ax.set_ylabel("GPU util (%)"); plt.xticks(rotation=45, ha="right", fontsize=6)
        _save(fig, fg, "fig8_gpu_util.png")


def fig12_recall(rd, fg):
    df = _load_csv(rd, "results_attention_recall.csv")
    if df is None:
        print("skip fig12 (no results_attention_recall.csv)"); return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for press, g in df.groupby("press"):
        ax.plot(g["ratio"], g["recall"], marker="o", label=press)
    ax.set(xlabel="compression ratio", ylabel="attention recall (%)")
    ax.legend(fontsize=6); _save(fig, fg, "fig12_attention_recall.png")


def fig10_iso(rd, fg):
    df = _load_csv(rd, "results_iso_systems_16k_bs8.csv")
    if df is None:
        print("skip fig9-11 (no results_iso_systems_16k_bs8.csv)"); return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for press, g in df.groupby("press"):
        ax.plot(g["target_acc"], g["kv_cache_mb"], marker="o", label=press)
    ax.set(xlabel="target accuracy", ylabel="KV cache (MB) at iso-accuracy")
    ax.legend(fontsize=6); _save(fig, fg, "fig10_iso_systems.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultsdir", default="results")
    ap.add_argument("--figdir", default="report/figures")
    a = ap.parse_args()
    for fn in (fig1_ruler, fig3_longbench, _systems, fig12_recall, fig10_iso):
        try:
            fn(a.resultsdir, a.figdir)
        except Exception as e:                       # never let one figure abort the rest
            print(f"WARN {fn.__name__}: {e}")
    print("Note: fig2 (damping), fig4 (acc-vs-memory), fig7/9/11 derive from the same inputs — extend as "
          "the corresponding CSVs are regenerated.")


if __name__ == "__main__":
    main()
