#!/bin/bash
# tau sweep: is std=True vs std=False just teleport-sharpness reparametrisation?
#   knorm_ppr_d15_std_t{03,05,07}  -- std=True at sharper tau; does some tau recover the std=False 58.27?
#   knorm                          -- base anchor at fraction 0.25 (fixes the 29.88@0.06 cross-sample compare)
# Reference: std=True tau=1.0 -> 31.99 ; std=False tau=1.0 -> 58.27 ; target ~58.
cd /home/lapidoty/kvpress/evaluation   # PERSISTENT repo
unset LD_LIBRARY_PATH
PY=/home/lapidoty/kv-dev/venv/bin/python
CFG=/home/lapidoty/kv-dev/eval_cfg_dec.yaml   # fraction 0.25
for P in knorm_ppr_d15_std_t03 knorm_ppr_d15_std_t05 knorm_ppr_d15_std_t07 knorm; do
  echo ">>> tau-sweep press=$P cr=0.75 $(date +%H:%M)"
  $PY evaluate.py --config_file "$CFG" --press_name "$P" --compression_ratio 0.75 2>&1 \
    | grep -E "string_match|Skipping|Error|Traceback" | tail -2
done
echo TAU_SWEEP_DONE
