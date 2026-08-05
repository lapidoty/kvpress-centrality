# Selection of Baseline Framework

**What it is.** kvpress is a mature, actively maintained framework (Apache-2.0) for **KV-cache
compression**: it plugs cache-eviction policies into HuggingFace `transformers` via a forward hook. The
central abstraction is the `ScorerPress`: a subclass implements
`score(...) -> (batch, num_kv_heads, seq_len)` and the framework keeps the top-scoring tokens during
prefill. It ships ~40 published presses (KnormPress, SnapKVPress, ExpectedAttentionPress, the leverage-score
family, …).

**Default / baseline policy.** Independent, per-token scoring: each press ranks tokens on its own signal
(key norm, recent-query attention, expected attention, etc.) and the framework evicts the lowest-scoring
`compression_ratio` fraction. The natural baseline for this project is `KnormPress` (keep low-L2-norm keys), keys-only, essentially free, and the framework's smooth default scorer.

**Why chosen.** It is the de-facto standard harness for this task: a shared evaluation stack (RULER, a public leaderboard, a tiny CPU unit-test model), so a new press can be benchmarked against
many strong baselines under **identical workloads** with minimal glue. Its `ScorerPress` contract makes a
new relational scorer a ~150-line drop-in, and its PR history merges external and even AI-authored scorers, so the contribution has a real upstream path.

**What this project adds.** Existing scorers rank tokens *independently* and ignore the relational structure
of the cache (a token can be individually unremarkable yet crucial because it *supports* an important one).
`CentralityPress` scores tokens by centrality in the key-similarity graph, anchored to `KnormPress` via a
personalized-PageRank teleport, keeping needles in and repairing the base's collapse under compression.
