#!/bin/bash
# Benchmark CentralityPress end-to-end in one command, fast (RULER fraction 0.06).
# Runs the report's benchmarks: RULER accuracy (§4.1), GraphKV suppressor (§4.7), LongBench multi-hop (§4.2).
# Override: MODEL=<hf-id> OUT=<dir> bash benchmark.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
M="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUT="${OUT:-out}"
cd "$REPO/evaluation"

echo ">> RULER accuracy (§4.1): centrality vs baselines, fraction 0.06"
python evaluate.py --dataset ruler --data_dir 4096 --model "$M" \
  --press_name no_press --compression_ratio 0.0 --fraction 0.06 --output_dir "$OUT"
for press in knorm snapkv centrality_ppr_knorm centrality_pure; do
  for r in 0.25 0.5 0.75; do
    python evaluate.py --dataset ruler --data_dir 4096 --model "$M" \
      --press_name "$press" --compression_ratio "$r" --fraction 0.06 --output_dir "$OUT"
  done
done

echo ">> RULER GraphKV suppressor (§4.7): graphkv_knorm, fraction 0.06"
for r in 0.25 0.5 0.75; do
  python "$REPO/project/additional_benchmarks/run.py" --dataset ruler --data_dir 4096 --model "$M" \
    --press_name graphkv_knorm --compression_ratio "$r" --fraction 0.06 --output_dir "$OUT"
done

echo ">> LongBench multi-hop QA (§4.2)"
python "$HERE/sweep_longbench.py" --model "$M" --out "$OUT/longbench_summary.json"

echo ">> Done. Per-run config.yaml + metrics.json under evaluation/$OUT/ ;"
echo ">> LongBench summary at evaluation/$OUT/longbench_summary.json ."
echo ">> Cross-method board ranks: python $HERE/../analysis/rank_vs_board.py"
