# CentralityPress — personalized-PageRank KV-cache eviction

A training-free KV-cache eviction policy for [NVIDIA kvpress](https://github.com/NVIDIA/kvpress), developed
as a fork (Apache-2.0). Everything except the new press and this project's material is upstream and
unmodified.

## What's new

`CentralityPress` ([`kvpress/presses/centrality_press.py`](kvpress/presses/centrality_press.py)) is a
`ScorerPress` that scores KV pairs by their **personalized-PageRank centrality** in the key-similarity
graph: importance propagates from tokens a base scorer (e.g. `KnormPress`) already rates highly to the
tokens that support them. It is query-free, flash-attention compatible, and its low-rank cosine kernel keeps
the scoring linear in sequence length, so it drops into the existing kvpress evaluation harness unchanged.

| | |
|---|---|
| 📄 Report | [**PDF**](project/report/REPORT.pdf) · [markdown](project/report/REPORT.md) · [rendered HTML](project/report/REPORT.html) · supplementary [PDF](project/report/SUPPLEMENTARY.pdf) / [`.md`](project/report/SUPPLEMENTARY.md) |
| 🔬 Reproduce | [`project/reproduction/README.md`](project/reproduction/README.md) — install, then one command per experiment |
| 📊 Results | [`project/reproduction/results/`](project/reproduction/results/) — RULER runs, each with `config.yaml` + `metrics.json` |
| 🗺️ Traceability | [`project/reproduction/MANIFEST.md`](project/reproduction/MANIFEST.md) — every reported number → the run that produced it |
| 🧩 The press | [`kvpress/presses/centrality_press.py`](kvpress/presses/centrality_press.py) |
| 🆚 GraphKV comparison | [`project/additional_benchmarks/`](project/additional_benchmarks/) — the suppression baseline it is measured against |

## Headline result

RULER (4k context, Llama-3.1-8B-Instruct, 13-task macro `string_match`), fraction-0.06 screen
(~30 examples/task). `CentralityPress` reinforcing `KnormPress`, versus `KnormPress` alone:

| compression ratio | `knorm` (base) | `centrality_ppr_knorm` |
|---|---|---|
| 0.25 | 76.5 | **94.5** (+18.0) |
| 0.50 | 51.6 | **82.4** (+30.8) |
| 0.75 | 29.9 | **58.3** (+28.4) |

Double-digit macro gains at every ratio; at the hardest setting (0.75) centrality nearly **doubles** the
base — 29.9 → 58.3 (~1.95×). Full-fraction, flash-attention board-grade runs reproduce the centrality
column within ~1 point (94.6 / 81.3 / 58.0; see [`project/reproduction/results/board_grid/`](project/reproduction/results/board_grid/)).
The sensitivity analysis (damping, teleport temperature) and honest caveats are in the report.

> **Scope.** The RULER numbers above are reproducible from the committed run configs. The LongBench,
> systems, iso-accuracy, and attention-recall sections (§4.2 / 4.4 / 4.5 / 4.6) are report-only — their raw
> data was lost with a prior machine; the scripts that produce them are included and marked as
> reconstructions. See [`project/reproduction/MANIFEST.md`](project/reproduction/MANIFEST.md).

## Quick start

```bash
pip install -e .            # kvpress + eval deps (see project/reproduction/Dockerfile / project/reproduction/requirements.txt)

# reproduce the headline point (fraction-0.06 screen, ~30 examples/task — fast):
cd evaluation
python evaluate.py --dataset ruler --data_dir 4096 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --press_name centrality_ppr_knorm --compression_ratio 0.75 --fraction 0.06 --output_dir out
```

`macro` is the mean `string_match` over the 13 RULER subtasks in the run's `metrics.json`. The full
experiment matrix — board-grade full-fraction runs, the τ / damping ablations, and the GraphKV comparison —
is one command each in [`project/reproduction/README.md`](project/reproduction/README.md).

## This is a fork of NVIDIA/kvpress

Everything except `CentralityPress` and this project's material is upstream and unchanged; GitHub's language
bar and diffs de-emphasize it via `.gitattributes`.

<details>
<summary>Upstream kvpress README (preserved verbatim)</summary>

The original NVIDIA/kvpress README — base install, the full press catalogue, the leaderboard — is kept
verbatim at [`UPSTREAM_README.md`](UPSTREAM_README.md).

</details>
