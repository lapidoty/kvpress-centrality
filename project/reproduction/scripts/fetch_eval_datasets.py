# SPDX-License-Identifier: Apache-2.0
"""Warm the HuggingFace cache with the kvpress evaluation datasets (RULER 4k/8k),
so the benchmark can run without re-downloading. Loads each split once."""
from datasets import load_dataset

print("== RULER (simonjegou/ruler) ==")
for ctx in ("4096", "8192"):
    d = load_dataset("simonjegou/ruler", data_dir=ctx, split="test")
    print(f"  cached ruler {ctx}: {len(d)} rows")

print("datasets cached")
