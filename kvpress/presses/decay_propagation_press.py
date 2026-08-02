# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class DecayPropagationPress(ScorerPress):
    """
    GraphKV-style ablation (arXiv:2509.00388): the SUPPRESSION counterpart to CentralityPress.

    Starting from a base press's importance, multiplicatively DECAY each token by its similarity to
    the most-important ("source") tokens, removing redundancy (a graph MMR):

        p = softmax(base_press.score / importance_temp)          (non-negative importance)
        s_i = p_i * prod_over_sources_j!=i (1 - relu(cos(k_i, k_j)))

    Two correctness details:
    * The base importance is turned into a non-negative distribution first, so the multiplicative
      decay always shrinks scores. This is required for base presses whose raw scores are negative
      (e.g. KnormPress = -||k||): multiplying a negative by (1 - e) < 1 would wrongly RAISE it.
    * A source is EXCLUDED from its own decay product (the j != i above). Otherwise cos(k_i, k_i) = 1
      makes the factor (1 - 1) = 0 and zeroes exactly the highest-importance tokens (the sources),
      inverting the intended behaviour.

    The product is accumulated in log space, which keeps the ordering of tokens in large redundant
    clusters instead of letting many factors underflow to a single zero-valued tie.

    Sharing the SAME base_press as CentralityPress makes the reinforcement-vs-suppression comparison
    clean: both start from softmax(base), then one adds damping * A @ c while the other multiplies by
    prod(1 - e).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    base_press : Optional[ScorerPress], default=None
        Press providing the base importance. Required (asserted non-None).
    num_rounds : int, default=1
        Number of successive decay rounds (GraphKV finds the first round gives the largest gain).
    top_k_sources : int, default=64
        Number of highest-importance tokens used as decay sources each round.
    importance_temp : float, default=1.0
        Softmax temperature over the base scores.
    n_sink : int, default=4
        Number of initial "attention sink" tokens forced to be kept.
    recent_window : int, default=8
        Number of most recent tokens forced to be kept.
    """

    base_press: Optional[ScorerPress] = None
    num_rounds: int = 1
    top_k_sources: int = 64
    importance_temp: float = 1.0
    n_sink: int = 4
    recent_window: int = 8

    def __post_init__(self):
        super().__post_init__()
        assert self.base_press is not None, "DecayPropagationPress requires a base_press"

    def post_init_from_model(self, model):
        # Forward model-time init to the base press (bases that read model config / load weights).
        if self.base_press is not None:
            self.base_press.post_init_from_model(model)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        B, H, S, D = keys.shape
        base = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs).float()
        # log-space importance; log_softmax is monotonic so topk selects the same sources as softmax.
        log_s = F.log_softmax(base / self.importance_temp, dim=-1)  # (B, H, S)
        kn = F.normalize(keys.float(), dim=-1, eps=1e-8)
        positions = torch.arange(S, device=keys.device)

        for _ in range(self.num_rounds):
            k = min(self.top_k_sources, S)
            sources = log_s.topk(k, dim=-1).indices  # (B, H, k)
            k_sources = torch.gather(kn, 2, sources.unsqueeze(-1).expand(-1, -1, -1, D))  # (B, H, k, D)
            e = torch.relu(kn @ k_sources.transpose(-1, -2))  # (B, H, S, k)
            # A source must not suppress itself (cos(k_i, k_i) = 1 -> factor 0 would zero it out).
            e = e.masked_fill(positions.view(1, 1, S, 1) == sources.unsqueeze(2), 0.0)
            # Accumulate the decay in log space to avoid underflowing large clusters to a single tie.
            log_s = log_s + torch.log1p(-e.clamp(max=1.0 - 1e-6)).sum(dim=-1)

        # Protect attention sinks and the recent window (out-of-place to stay autograd-safe).
        if self.n_sink or self.recent_window:
            fill = log_s.amax(dim=-1, keepdim=True) + 1.0
            protected = torch.zeros(S, dtype=torch.bool, device=log_s.device)
            if self.n_sink:
                protected |= positions < self.n_sink
            if self.recent_window:
                protected |= positions >= (S - self.recent_window)
            log_s = torch.where(protected, fill, log_s)

        # Return fp32: the scores are only used for topk selection, and log-space values span a wide
        # range; downcasting to a low-precision keys.dtype (bf16) would collapse the ranking.
        return log_s
