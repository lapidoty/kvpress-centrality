# CentralityPress — reproduction pack (how to benchmark)

`CentralityPress` is a training-free KV-cache eviction scorer for NVIDIA kvpress: personalized-PageRank
centrality over the key-similarity graph, anchored to `KnormPress` via a teleport term. The press ships at
[`../../kvpress/presses/centrality_press.py`](../../kvpress/presses/centrality_press.py).

## Benchmark — one command
```bash
pip install -e ".[eval]"                                # kvpress + evaluation deps (from pyproject.toml)
python project/reproduction/scripts/benchmark.py        # RULER §4.1 + GraphKV §4.7 + LongBench §4.2, fraction 0.06 (fast)
```
(`MODEL=<hf-id> OUT=<dir> python …/benchmark.py` to override.) One command runs all the report's benchmarks:
`evaluate.py` for RULER accuracy (centrality vs `no_press`/`knorm`/`snapkv`/`centrality_pure`),
`additional_benchmarks/run.py` for the GraphKV suppressor, and `sweep_longbench.py` for multi-hop QA.

## Output & sample logs
Each run writes a directory with `config.yaml` + `metrics.json` (`macro` = mean `string_match` over the 13
RULER subtasks); LongBench writes `longbench_summary.json`. **Sample logs from our runs are committed under
[`results/`](results/):** `board_grid/` (full-fraction, leaderboard-ready, with `predictions.csv`) and
`ruler_screen/` (the fraction-0.06 screen). Every report RULER number maps to a run directory named
`ruler__4096__<model>__<press>__<ratio>`.

## Also here
- `analysis/rank_vs_board.py` — cross-method board ranks (matched model + ratio) vs `kvpress_leaderboard_raw.csv`.
- `analysis/{speed_memory,iso_systems,recall_probe,make_figures}.py` — regenerate the report-only systems /
  iso-accuracy / attention-recall sections (§4.4/4.5/4.6) and figures (heavier; one A100).
- Install: `pip install -e ".[eval]"` (add the `flash-attn` extra for speed, `matplotlib` to regenerate figures).

The §4.2/4.4/4.5/4.6 raw outputs are report-only (not committed); the scripts above regenerate them.
