# MANIFEST — report numbers → run directories

Every RULER number in the report is produced by kvpress's own `evaluation/evaluate.py` +
`evaluation/benchmarks/ruler/calculate_metrics.py` (no reimplemented scorer) and is traceable to a run directory
below (each holds `config.yaml` + `metrics.json`; large `predictions.csv` omitted — see note). `macro` is
the unweighted mean over the 13 RULER subtasks (the leaderboard metric), at a **single** ratio.

**Environment.** Runs on this devserver: 1×A100-80GB, Llama-3.1-8B-Instruct / Qwen3-8B, bf16.
`fraction` and `attn` are given per row because they differ across runs.

> **Model provenance.** `config.yaml` records the canonical Hugging Face id
> `meta-llama/Llama-3.1-8B-Instruct`. On the authoring devserver the weights were loaded from a local
> mirror at `/home/lapidoty/models/Llama-3.1-8B-Instruct` (the HF CDN is unreachable from that host and
> the model is gated). Identical weights; only the label differs, and the canonical id is what the
> kvpress leaderboard matches on.

## RULER — §4.1 headline screen (fraction 0.06, sdpa)

| report | claim | macro | run directory | fraction | attn |
|---|---|---|---|---|---|
| §4.1 table | `no_press` ceiling | 95.25 | `results/ruler_screen/…__no_press__0.00__fraction0.060` | 0.06 | sdpa |
| §4.1 table | `knorm` @0.25/0.5/0.75 | 76.5/51.6/29.9 | `results/ruler_screen/…__knorm__0.{25,50,75}__fraction0.060` | 0.06 | sdpa |
| §4.1 table | `snapkv` | 82.3/69.9/40.5 | `results/ruler_screen/…__snapkv__0.{25,50,75}__fraction0.060` | 0.06 | sdpa |
| §4.1 table | **`centrality_ppr_knorm` d=0.15** | **94.5/82.4/58.3** | `results/ruler_screen/…__centrality_ppr_knorm_d0.15__0.{25,50,75}__fraction0.060` | 0.06 | sdpa |
| §4.1 table | `centrality_pure` (d=1) | 28.2/10.4/5.1 | `results/ruler_screen/…__centrality_pure__0.{25,50,75}__fraction0.060` | 0.06 | sdpa |

## RULER — §4.1 board-grade run (full fraction, flash_attention_2) — reproduces the screen

| report | claim | macro | run directory | fraction | attn |
|---|---|---|---|---|---|
| §4.1 repro | ppr_knorm d=0.15 @0.25 | 94.57 (vs screen 94.5) | `results/board_grid/…__centrality_ppr_knorm_d0.15__0.25` | 1.0 | flash_attention_2 |
| §4.1 repro | ppr_knorm d=0.15 @0.75 | 58.04 (vs screen 58.3) | `results/board_grid/…__centrality_ppr_knorm_d0.15__0.75` | 1.0 | flash_attention_2 |
| §4.1 repro | ppr_knorm d=0.15 @0.50 | 81.33 (vs screen 82.4) | `results/board_grid/…__0.50` | 1.0 | flash_attention_2 |
| §4.1 repro | ppr_knorm d=0.15 @0.875 | 41.71 | `results/board_grid/…__0.875` | 1.0 | flash_attention_2 |

## §4.7 — mechanism of the lift (fraction 0.25, sdpa)

| report | claim | macro @0.75 | run directory |
|---|---|---|---|
| §4.7 τ table | std=True τ=1.0 | 31.99 | `results/ablations/…__centrality_ppr_knorm_std_d15__0.75__fraction0.250` |
| §4.7 τ table | std=True τ=0.7 | 53.05 | `results/ablations/…__knorm_ppr_d15_std_t07__0.75__fraction0.250` |
| §4.7 τ table | std=True τ=0.5 | 55.56 | `results/ablations/…__knorm_ppr_d15_std_t05__0.75__fraction0.250` |
| §4.7 τ table | std=True τ=0.3 | 57.49 | `results/ablations/…__knorm_ppr_d15_std_t03__0.75__fraction0.250` |
| §4.7 uniform | uniform teleport d=0.15 | 5.44 | `results/ablations/…__centrality_uniform_d15__0.75__fraction0.250` |

## §4.8 — GraphKV comparison, RULER (fraction 0.06, sdpa)

| report | claim | macro | run directory |
|---|---|---|---|
| §4.8 | `graphkv_knorm` @0.25/0.5/0.75 | 83.6/57.4/38.3 | `results/ruler_screen/…__graphkv_knorm__0.{25,50,75}__fraction0.060` |
| §4.8 | ppr d=0.15 vs graphkv paired Δ | +12.3/+27.8/+21.8 | (paired over the ppr_knorm vs graphkv_knorm screen runs above) |

The GraphKV benchmark code + a reproduction runner are in `additional_benchmarks/`.

## Cross-method ranks (§4.1)

Computed by `analysis/rank_vs_board.py` against the public board's own backing data
(`kvpress_leaderboard_raw.csv`, included under `analysis/`), matched model + matched ratio:
macro **8th/11th/12th of 20**; `fwe` **1st/1st/9th**; `cwe` 15th–18th.

## §4.2 / §4.4 / §4.5 / §4.6 — re-runnable via reconstructed scripts (raw data lost; not yet re-run)

The raw run dirs / CSVs for these sections were produced on the **original devserver (`devvm50213`, lost)**;
only the report text and figures were recovered. The scripts that produced them have been **faithfully
reconstructed from the report methodology** (each carries a `Reconstruction from REPORT.md §X` header) and
are included, so a stranger can regenerate the numbers:

| section | reconstructed script |
|---|---|
| §4.2 LongBench (incl. the LongBench half of the §4.8 GraphKV comparison) | `scripts/sweep_longbench.py` |
| §4.4 systems (prefill/memory/latency/throughput) | `analysis/speed_memory.py` |
| §4.5 iso-accuracy | `analysis/iso_systems.py` |
| §4.6 attention recall | `analysis/recall_probe.py` |
| figures fig1–12 | `analysis/make_figures.py` |
| paired bootstrap CI + Wilcoxon (§3.4) | `analysis/analyze_paired.py` |

These scripts have **not yet been re-executed** (the original data to diff against is gone), so the numbers
above remain report-only until a re-run on the current devserver regenerates them. That re-run is the single
outstanding step to make these four sections fully MANIFEST-traceable.

## Notes

- `predictions.csv` is omitted from every run dir (up to ~4 MB each). Re-generate by re-running the config.
- The venv used here is a Meta-internal torch build (`2.13.0+cu130`); the `Dockerfile` pins a **standard**
  public stack (torch/cu12x + kvpress + flash-attn) for a stranger to reproduce — see `Dockerfile`.
