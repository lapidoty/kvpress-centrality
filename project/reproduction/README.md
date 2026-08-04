# CentralityPress — reproduction pack

`CentralityPress` is a training-free KV-cache eviction scorer for NVIDIA kvpress: personalized-PageRank
centrality over the key-similarity graph, anchored to a base press (`KnormPress`) through a teleport term.
The press ships in this repository at `kvpress/presses/centrality_press.py`; this directory holds the
report, benchmarks, and reproduction material.

## Layout
- `../report/` — `REPORT.md` (+ `.html`), `SUPPLEMENTARY`, `figures/`.
- `MANIFEST.md` — **read first**: every report number → run directory, with `fraction` + `attn`.
- `results/` — `config.yaml` + `metrics.json` per run: `board_grid/` (full-fraction FA2),
  `ruler_screen/` (fraction-0.06 screen).
- `scripts/` — the eval drivers (`submission_grid_llama.sh`, `sweep_longbench.py`, `fetch_eval_datasets.sh`).
- `analysis/` — `rank_vs_board.py` (matched-ratio board ranks) and the §4.2/4.4/4.5/4.6 regeneration scripts.
- `../additional_benchmarks/` — the GraphKV suppression comparison (not part of the shipped press).

## Reproduce the RULER result (one A100)
```bash
pip install -e .                       # kvpress + eval deps (see Dockerfile / requirements.txt)
cd evaluation
# fraction-0.06 screen (fast):
python evaluate.py --dataset ruler --data_dir 4096 --model meta-llama/Llama-3.1-8B-Instruct \
    --press_name centrality_ppr_knorm --compression_ratio 0.75 --fraction 0.06 --output_dir out
# board-grade (full fraction; flash_attention_2 auto-selected if flash-attn is installed):
python evaluate.py --dataset ruler --data_dir 4096 --model meta-llama/Llama-3.1-8B-Instruct \
    --press_name centrality_ppr_knorm --compression_ratio 0.75 --output_dir out
```
`macro` = mean `string_match` over the 13 RULER subtasks in `metrics.json`. The committed board grid covers
four ratios: 0.25 / 0.50 / 0.75 / 0.875.

## GraphKV comparison
`python project/additional_benchmarks/run.py --dataset ruler --data_dir 4096 --model <m> --press_name graphkv_knorm
--compression_ratio 0.5 --output_dir out` — injects `graphkv_knorm` at runtime; does not touch the registry.

## Scope
The RULER results (§4.1, §4.7) are fully traceable to the committed run directories. LongBench, systems,
iso-accuracy, and attention-recall (§4.2/4.4/4.5/4.6) are **report-only**: their raw outputs are not
committed, but the scripts that produce them are included
(`scripts/sweep_longbench.py`, `analysis/{speed_memory,iso_systems,recall_probe,make_figures}.py`) and can
be **run to regenerate the numbers**. See `MANIFEST.md` and `REGENERATE.md`.
