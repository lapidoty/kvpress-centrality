# CentralityPress — reproduction pack

`CentralityPress` is a training-free KV-cache eviction scorer for NVIDIA kvpress: personalized-PageRank
centrality over the key-similarity graph, anchored to a base press (`KnormPress`) through a teleport term.
The press ships in this repository at `kvpress/presses/centrality_press.py`; this directory holds the
report, benchmarks, and reproduction material.

## Layout
- `../report/` — `REPORT.md` (+ `.html`), `SUPPLEMENTARY`, `figures/`.
- `MANIFEST.md` — **read first**: every report number → run directory, with `fraction` + `attn`.
- `results/` — `config.yaml` + `metrics.json` per run: `board_grid/` (full-fraction FA2),
  `ruler_screen/` (fraction-0.06 screen), `ablations/` (τ sweep, uniform, damping).
- `scripts/` — the eval drivers used (`submission_grid_llama.sh`, `tau_sweep.sh`, `eval_cfg_dec.yaml`, …).
- `analysis/` — `rank_vs_board.py` (matched-ratio board ranks), `teleport_entropy.py`.
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
three ratios: 0.25 / 0.50 / 0.75.

## GraphKV comparison
`python additional_benchmarks/run.py --dataset ruler --data_dir 4096 --model <m> --press_name graphkv_knorm
--compression_ratio 0.5 --output_dir out` — injects `graphkv_knorm` at runtime; does not touch the registry.

## Caveat
LongBench / systems / iso-accuracy / attention-recall (§4.2/4.4/4.5/4.6) are **report-only** — their raw
data was lost with the prior devserver. The scripts that produce them are **reconstructed** here
(`scripts/sweep_longbench.py`, `analysis/{speed_memory,iso_systems,recall_probe,make_figures}.py`, each
marked as a reconstruction) and can be **re-run to regenerate the numbers**; they have not been re-executed
yet, so those sections stay report-only until then. See `MANIFEST.md`. All RULER results are traceable now.
