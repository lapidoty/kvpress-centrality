#!/bin/bash
# Llama board grid, remaining ratios (c=0.75 done by the sanity run): centrality_ppr_knorm_d0.15,
# ruler/4096, FULL fraction, flash_attention_2. Sequential on the single A100 (~2.3h/run).
cd /home/lapidoty/kvpress/evaluation
source /home/lapidoty/kv-dev/cuda13_env.sh
PY=/home/lapidoty/kv-dev/venv/bin/python
OUT=/home/lapidoty/kv-dev/submission_results
M=/home/lapidoty/models/Llama-3.1-8B-Instruct
for r in 0.25 0.50 0.875; do
  echo ">>> llama c=$r $(date +%H:%M)"
  $PY evaluate.py --dataset ruler --data_dir 4096 --model "$M" \
    --press_name centrality_ppr_knorm_d0.15 --compression_ratio "$r" \
    --output_dir "$OUT" --device cuda:0 > "$OUT/grid_llama_$r.log" 2>&1
  echo "  rc=$?  attn=$(grep -oE 'flash_attention_2' "$OUT/grid_llama_$r.log" | head -1)"
done
echo LLAMA_GRID_DONE $(date +%H:%M)
