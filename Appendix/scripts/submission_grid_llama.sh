#!/bin/bash
# Llama board grid: centrality_ppr_knorm, ruler/4096, FULL fraction, flash_attention_2 (auto-selected
# when flash-attn is installed). Sequential on a single A100 (~2.3h/run). Override via env: MODEL, OUT, PYTHON.
cd "$(dirname "$0")/../../evaluation"      # repo-root/evaluation
PY="${PYTHON:-python}"
OUT="${OUT:-./submission_results}"; mkdir -p "$OUT"
M="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
for r in 0.25 0.50 0.75 0.875; do
  echo ">>> llama c=$r $(date +%H:%M)"
  $PY evaluate.py --dataset ruler --data_dir 4096 --model "$M" \
    --press_name centrality_ppr_knorm --compression_ratio "$r" \
    --output_dir "$OUT" --device cuda:0 > "$OUT/grid_llama_$r.log" 2>&1
  echo "  rc=$?  attn=$(grep -oE 'flash_attention_2' "$OUT/grid_llama_$r.log" | head -1)"
done
echo LLAMA_GRID_DONE $(date +%H:%M)
