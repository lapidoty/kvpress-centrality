# CentralityPress — project deliverables

Project material for **`CentralityPress`**, a training-free personalized-PageRank KV-cache eviction press
for [NVIDIA kvpress](https://github.com/NVIDIA/kvpress). The press itself lives at
[`../kvpress/presses/centrality_press.py`](../kvpress/presses/centrality_press.py); this folder holds the
report, the reproduction pack, and the comparison baseline.

| folder | what's in it |
|---|---|
| [`report/`](report/) | the report — [`REPORT.pdf`](report/REPORT.pdf) (+ markdown source), `SUPPLEMENTARY`, and `figures/` |
| [`reproduction/`](reproduction/) | how to reproduce — baseline justification, `Dockerfile` + `requirements.txt`, run scripts, the committed RULER results, and [`MANIFEST.md`](reproduction/MANIFEST.md) mapping every report number to its run directory |
| [`additional_benchmarks/`](additional_benchmarks/) | the GraphKV suppression baseline the press is compared against (report §4.7) |

**Start here:** read [`report/REPORT.pdf`](report/REPORT.pdf). **To reproduce the headline result**, follow
[`reproduction/README.md`](reproduction/README.md) — one command reproduces the RULER numbers.
