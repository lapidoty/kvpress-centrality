# CentralityPress: personalized-PageRank KV-cache eviction

A training-free KV-cache eviction policy for [NVIDIA kvpress](https://github.com/NVIDIA/kvpress), developed
as a fork (Apache-2.0). **Our changes to upstream:** the new press
[`kvpress/presses/centrality_press.py`](kvpress/presses/centrality_press.py) (registered in
`kvpress/__init__.py`, `evaluation/evaluate_registry.py`, `tests/default_presses.py`), its tests in
[`tests/presses/test_centrality_press.py`](tests/presses/test_centrality_press.py), and the
[`project/`](project/) folder (report, reproduction, comparison baseline). Everything else is upstream and
unchanged.

## What's new

`CentralityPress` ([`kvpress/presses/centrality_press.py`](kvpress/presses/centrality_press.py)) is a
`ScorerPress` that scores KV pairs by their **personalized-PageRank centrality** in the key-similarity
graph: importance propagates from tokens a base scorer (e.g. `KnormPress`) already rates highly to the
tokens that support them. It is query-free, flash-attention compatible, and its low-rank cosine kernel keeps
the scoring linear in sequence length, so it drops into the existing kvpress evaluation harness unchanged.

| | |
|---|---|
| 📄 Report | [**PDF**](project/report/REPORT.pdf) · [markdown](project/report/REPORT.md) · [baseline](project/report/BASELINE.pdf) · [supplementary](project/report/SUPPLEMENTARY.pdf) |
| 🧩 Press | [`kvpress/presses/centrality_press.py`](kvpress/presses/centrality_press.py) |
| 🔬 Reproduce | [`project/reproduction/`](project/reproduction/README.md): install, then one command |
| 📊 Results | [`project/reproduction/results/`](project/reproduction/results/): per-run `config.yaml` + `metrics.json` (+ `predictions.csv`); the dir name maps each number to its run |
| 🆚 GraphKV | [`project/additional_benchmarks/`](project/additional_benchmarks/): the suppression baseline |

## Submission guide (project brief → where)

| Brief section | Where |
|---|---|
| **§2** Baseline justification | [`report/BASELINE.pdf`](project/report/BASELINE.pdf) |
| **§3** Test suite: workloads, latency/memory/throughput, scripts + CI | report §3.2, §4.3–4.4; [`reproduction/`](project/reproduction/) + [CI](.github/workflows/centrality-ci.yml) |
| **§4** Extension: feature-branch code, unit tests, API/params | [`add-centrality-press`](../../tree/add-centrality-press) branch, [`tests/`](tests/presses); report §2.1 |
| **§5** Evaluation: baseline vs extended, sweeps, ablation | report §4.1–4.5, ablation ladder §3.3 |
| **§6** Reporting: clean repo, README install + benchmark, 8–12pp PDF | this README, [`reproduction/README.md`](project/reproduction/README.md), [`REPORT.pdf`](project/report/REPORT.pdf) |
| **§7** Criteria: correctness / reproducibility / performance / clarity | throughout |

## Repository layout

```text
kvpress-centrality/
├── kvpress/presses/centrality_press.py     # the extension: CentralityPress (a ScorerPress)
│                                            #   (registered in kvpress/__init__.py + evaluation/evaluate_registry.py)
├── tests/presses/test_centrality_press.py  # unit + end-to-end tests
├── evaluation/                             # upstream kvpress evaluation harness (unmodified)
│
├── project/                               # ── all project deliverables ──
│   ├── report/
│   │   ├── REPORT.pdf / .md               # the report
│   │   ├── BASELINE.pdf / .md             # §2 "Selection of Baseline Framework"
│   │   ├── SUPPLEMENTARY.pdf / .md        # per-task tables + systems figures
│   │   └── figures/                       # fig1–fig11
│   ├── reproduction/
│   │   ├── README.md                      # how to benchmark (one command)
│   │   ├── scripts/                       # benchmark.py, submission_grid_llama.py, fetch_eval_datasets.py
│   │   ├── analysis/                      # rank_vs_board.py + the §4.2/4.4/4.5/4.6 regeneration scripts
│   │   └── results/                       # committed runs: config.yaml + metrics.json (+ predictions.csv)
│   └── additional_benchmarks/             # GraphKV suppression baseline (decay_propagation_press.py, run.py)
│
├── README.md                              # this file
└── UPSTREAM_README.md                     # NVIDIA/kvpress README, preserved verbatim
```

## Headline result

RULER (4k context, Llama-3.1-8B-Instruct, 13-task macro `string_match`), fraction-0.06 screen
(~30 examples/task). `CentralityPress` reinforcing `KnormPress`, versus `KnormPress` alone:

| compression ratio | `knorm` (base) | `centrality_ppr_knorm` |
|---|---|---|
| 0.25 | 76.5 | **94.5** (+18.0) |
| 0.50 | 51.6 | **82.4** (+30.8) |
| 0.75 | 29.9 | **58.3** (+28.4) |

Double-digit macro gains at every ratio; at the hardest setting (0.75) centrality nearly **doubles** the
base, 29.9 → 58.3 (~1.95×). Full-fraction, flash-attention board-grade runs reproduce the centrality
column within ~1 point (94.6 / 81.3 / 58.0; see [`project/reproduction/results/board_grid/`](project/reproduction/results/board_grid/)).
The sensitivity analysis (damping, teleport temperature) and honest caveats are in the report.

> **Scope.** The RULER numbers above are reproducible from the committed run configs. The systems and
> iso-accuracy analyses (report §4.3-4.4) are reported from the project's runs, with the scripts that
> regenerate them included. See
> [`project/reproduction/README.md`](project/reproduction/README.md).

## Quick start

```bash
pip install -e ".[eval]"     # kvpress + evaluation deps (from pyproject.toml)

# reproduce the headline point (fraction-0.06 screen, ~30 examples/task, fast):
cd evaluation
python evaluate.py --dataset ruler --data_dir 4096 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --press_name centrality_ppr_knorm --compression_ratio 0.75 --fraction 0.06 --output_dir out
```

`macro` is the mean `string_match` over the 13 RULER subtasks in the run's `metrics.json`. The full
experiment matrix (board-grade full-fraction runs and the GraphKV comparison) is one command each in
[`project/reproduction/README.md`](project/reproduction/README.md).

## This is a fork of NVIDIA/kvpress

Everything except `CentralityPress` and this project's material is upstream and unchanged; GitHub's language
bar and diffs de-emphasize it via `.gitattributes`.

<details>
<summary>Upstream kvpress README (preserved verbatim)</summary>

The original NVIDIA/kvpress README (base install, the full press catalogue, the leaderboard) is kept
verbatim at [`UPSTREAM_README.md`](UPSTREAM_README.md).

</details>
