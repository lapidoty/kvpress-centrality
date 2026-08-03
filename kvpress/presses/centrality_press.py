# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class CentralityPress(ScorerPress):
    """
    Training-free KV eviction by PERSONALIZED PAGERANK over the key-similarity graph.

    A token is kept if it is directly important (teleport = a base press's scores) OR connected
    to important tokens (structural reinforcement via propagation over the graph):

        c <- damping * normalize(A @ c) + (1 - damping) * p
        p = softmax(base_press.score / teleport_temp)   (uniform if base_press is None)
        A[i, j] = (1 + cos(k_i, k_j)) / 2                (default "shifted" kernel)

    * damping = 0.0 -> c = p, which recovers the base press exactly (a safe floor).
    * damping = 1.0 -> pure eigenvector-style centrality, which localizes on the densest cluster
      and evicts outliers (kept only as an ablation; the needle/answer token is usually an outlier).

    Efficiency: the shifted/linear kernel is low rank (A = Kn @ Kn^T with Kn the L2-normalized keys),
    so a matvec never materializes S x S:

        A @ c = 0.5 * sum(c) + 0.5 * Kn @ (Kn^T @ c)    -> O(S * head_dim) time and memory.

    Contrast with GraphKV (arXiv:2509.00388), which *suppresses* redundancy by multiplying the base
    score by prod(1 - e_ij); this press *reinforces* connectedness by adding damping * A @ c.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    base_press : Optional[ScorerPress], default=KnormPress()
        Press whose scores define the teleport distribution. None means a uniform teleport,
        i.e. pure centrality (ablation only).
    num_iters : int, default=3
        Number of power-iteration / PageRank steps.
    damping : float, default=0.15
        Interpolation between the base press (0.0) and pure structural centrality (1.0).
    teleport_temp : float, default=1.0
        Softmax temperature over the base scores. Lower sharpens the teleport distribution.
    standardize_teleport : bool, default=False
        Z-score the base scores per (batch, head) before the softmax teleport, making it scale-invariant
        to the base press. Needed for tiny-magnitude scores (e.g. SnapKV's attention weights), which
        otherwise soften to a ~uniform teleport (i.e. degenerate to pure centrality).
    n_sink : int, default=4
        Number of initial "attention sink" tokens forced to be kept.
    recent_window : int, default=8
        Number of most recent tokens forced to be kept.
    similarity : str, default="shifted"
        "shifted": (1 + cos) / 2 (low rank, non-negative, default -- recommended). "linear": raw cos
        (low rank), which can be negative and yields SIGNED, non-probability scores; for anti-correlated
        clusters the kept side can be noise-dependent, so prefer "shifted".
    """

    base_press: Optional[ScorerPress] = field(default_factory=KnormPress)
    num_iters: int = 3
    damping: float = 0.15
    teleport_temp: float = 1.0
    standardize_teleport: bool = False
    n_sink: int = 4
    recent_window: int = 8
    similarity: str = "shifted"

    def __post_init__(self):
        super().__post_init__()
        assert self.similarity in ("shifted", "linear"), f"Unknown similarity: {self.similarity}"
        assert 0.0 <= self.damping <= 1.0, "damping must be in [0, 1]"

    def post_init_from_model(self, model):
        # Forward model-time init to the base press (bases that read model config / load weights).
        if self.base_press is not None:
            self.base_press.post_init_from_model(model)

    def _matvec(self, kn: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # kn: (B, H, S, D) L2-normalized keys ; c: (B, H, S, 1)
        # A @ c via the low-rank factorization -- O(S * head_dim), never materializes S x S.
        low_rank = kn @ (kn.transpose(-1, -2) @ c)
        if self.similarity == "shifted":
            return 0.5 * c.sum(dim=-2, keepdim=True) + 0.5 * low_rank
        return low_rank  # "linear"

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
        kn = F.normalize(keys.float(), dim=-1, eps=1e-8)  # power iteration is unstable in bf16

        # Personalization / teleport distribution p : (B, H, S, 1)
        if self.base_press is not None:
            base = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs).float()
            if self.standardize_teleport:
                # Make the teleport scale-invariant to the base press: some presses (e.g. SnapKV) emit
                # tiny-magnitude scores where softmax(base / temp) collapses to ~uniform (i.e. pure
                # centrality). Z-scoring per (batch, head) restores a meaningful teleport distribution.
                base = (base - base.mean(dim=-1, keepdim=True)) / (base.std(dim=-1, keepdim=True) + 1e-6)
            p = F.softmax(base / self.teleport_temp, dim=-1).unsqueeze(-1)
        else:
            p = torch.full((B, H, S, 1), 1.0 / S, device=keys.device, dtype=torch.float32)

        c = p.clone()
        for _ in range(self.num_iters):
            ac = self._matvec(kn, c)
            ac = ac / (ac.abs().sum(dim=-2, keepdim=True) + 1e-8)  # renormalize propagated mass
            c = self.damping * ac + (1.0 - self.damping) * p

        scores = c.squeeze(-1)  # (B, H, S)

        # Protect attention sinks and the recent window (out-of-place to stay autograd-safe).
        if self.n_sink or self.recent_window:
            fill = scores.amax(dim=-1, keepdim=True) + 1.0
            positions = torch.arange(S, device=scores.device)
            protected = torch.zeros(S, dtype=torch.bool, device=scores.device)
            if self.n_sink:
                protected |= positions < self.n_sink
            if self.recent_window:
                protected |= positions >= (S - self.recent_window)
            scores = torch.where(protected, fill, scores)

        # Return fp32: the per-iteration renormalization packs scores into a ~1/S band, so
        # downcasting to a low-precision keys.dtype (bf16) would collapse the ranking into ties.
        # compress() only uses these for topk + gather, both of which are dtype-agnostic.
        return scores
