# Additional benchmark — GraphKV suppression (comparison only)

`DecayPropagationPress` is a training-free reimplementation of **GraphKV** (arXiv:2509.00388) — the
*suppression* counterpart to this project's `CentralityPress` (*reinforcement*). It is kept here as a
**comparison benchmark only**; it is **not** part of the shipped contribution (the submitted press is
`CentralityPress` in `kvpress/presses/centrality_press.py`), so it deliberately lives outside the
`kvpress` package.

## Result

`CentralityPress` (reinforcement) **beats** this suppressor on RULER at every compression ratio — paired
per-example deltas vs `graphkv_knorm` of **+12.3 / +27.8 / +21.8** at c = 0.25 / 0.5 / 0.75 (all 95 %
bootstrap-CI significant; see the "GraphKV comparison" section of `report/REPORT.md`). Independently,
`CentralityPress` is a **ranked entry on the public KVPress leaderboard** (mid-pack overall, 1st on FWE at
matched ratios).

## Mechanism

From a non-negative base importance `p = softmax(base/τ)`, multiplicatively decay each token by its
similarity to the top sources: `s_i = p_i · ∏_{j∈sources, j≠i} (1 − relu(cos(k_i, k_j)))`. Two correctness
points (regression-tested): a source is excluded from its own decay, and the product is accumulated in log
space so large redundant clusters do not underflow to a tie.

## Reproduce

```bash
# unit tests (from the repo root)
python -m pytest additional_benchmarks/

# evaluate on RULER — injects graphkv into the registry AT RUNTIME (does not modify the shipped registry)
python additional_benchmarks/run.py --dataset ruler --data_dir 4096 \
    --model <model-path> --press_name graphkv_knorm --compression_ratio 0.5 --output_dir results_graphkv
```
