#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the GraphKV suppression comparison benchmark WITHOUT modifying the shipped registry.

Injects `graphkv_knorm` / `graphkv_snapkv` into the in-memory PRESS_REGISTRY, then delegates to the
standard evaluation CLI. Example:

    python additional_benchmarks/run.py --dataset ruler --data_dir 4096 \
        --model <path> --press_name graphkv_knorm --compression_ratio 0.5 --output_dir results_graphkv
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # project/additional_benchmarks
PROJECT = os.path.dirname(HERE)                      # project/  (for the `additional_benchmarks` import)
REPO = os.path.dirname(PROJECT)                      # repo root (for kvpress + evaluation/)
sys.path.insert(0, PROJECT)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))

import evaluate_registry  # noqa: E402
from kvpress import KnormPress, SnapKVPress  # noqa: E402

from additional_benchmarks.decay_propagation_press import DecayPropagationPress  # noqa: E402

# Runtime-only injection: the shipped evaluate_registry.py is left untouched.
evaluate_registry.PRESS_REGISTRY.setdefault("graphkv_knorm", DecayPropagationPress(base_press=KnormPress()))
evaluate_registry.PRESS_REGISTRY.setdefault("graphkv_snapkv", DecayPropagationPress(base_press=SnapKVPress()))

# evaluate.py is a Fire CLI that reads sys.argv and `from evaluate_registry import PRESS_REGISTRY`
# (the already-patched module object). Run it from evaluation/ so its relative imports resolve.
os.chdir(os.path.join(REPO, "evaluation"))
runpy.run_path("evaluate.py", run_name="__main__")
