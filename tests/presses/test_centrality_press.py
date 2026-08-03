# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from kvpress.presses.centrality_press import CentralityPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.scorer_press import ScorerPress
from tests.fixtures import unit_test_model  # noqa: F401


@dataclass
class _TinyScaleBase(ScorerPress):
    """Emulates a SnapKV-like base: same ranking as KnormPress but ~1e-3 magnitude."""

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return -keys.norm(dim=-1) * 1e-3


def _coherent_haystack_with_needle(S, D, needle_idx, needle_norm=0.02, haystack_norm=1.0, seed=0):
    """
    Build one (1, 1, S, D) key tensor that isolates the two failure modes of pure centrality:

    * The haystack is a COHERENT DIRECTION cluster (a common direction + tiny noise), so every
      haystack token is mutually similar in cosine space.
    * The needle is a distinct direction ORTHOGONAL to the cluster, with a SMALL L2 norm.

    Consequences (the whole point of the thesis test):
    * KnormPress (score = -||k||) ranks the small-norm needle HIGHEST -> teleport keeps it.
    * Cosine centrality is scale-invariant, so the needle's small norm is irrelevant to the graph;
      being orthogonal to the dense cluster it has the LOWEST centrality -> pure centrality drops it.
    """
    g = torch.Generator().manual_seed(seed)
    common = F.normalize(torch.randn(D, generator=g), dim=-1)
    keys = common.unsqueeze(0).repeat(S, 1) + 0.02 * torch.randn(S, D, generator=g)
    keys = F.normalize(keys, dim=-1) * haystack_norm

    needle_dir = F.normalize(torch.randn(D, generator=g), dim=-1)
    needle_dir = needle_dir - (needle_dir @ common) * common  # orthogonalize to the cluster
    needle_dir = F.normalize(needle_dir, dim=-1)
    keys[needle_idx] = needle_dir * needle_norm
    return keys.view(1, 1, S, D)


def test_score_shape_and_finite():
    press = CentralityPress(base_press=None, damping=1.0, compression_ratio=0.5, n_sink=0, recent_window=0)
    keys = torch.randn(2, 3, 40, 8)
    s = press.score(None, None, keys, keys, None, {})
    assert s.shape == (2, 3, 40)
    assert torch.isfinite(s).all()


def test_lowrank_matvec_matches_explicit():
    # The shifted low-rank matvec must equal the explicit A @ c (guarantees no S x S is needed).
    press = CentralityPress(base_press=None, similarity="shifted")
    kn = F.normalize(torch.randn(1, 2, 50, 8), dim=-1)
    c = torch.rand(1, 2, 50, 1)
    A = 0.5 * (1.0 + kn @ kn.transpose(-1, -2))
    assert torch.allclose(A @ c, press._matvec(kn, c), atol=1e-5)

    press_lin = CentralityPress(base_press=None, similarity="linear")
    A_lin = kn @ kn.transpose(-1, -2)
    assert torch.allclose(A_lin @ c, press_lin._matvec(kn, c), atol=1e-5)


def test_pure_centrality_ranks_hub_first():
    # damping=1, no base: the token aligned with the mean key direction is the most central.
    press = CentralityPress(base_press=None, damping=1.0, compression_ratio=0.5, n_sink=0, recent_window=0, num_iters=5)
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(1, 2, 32, 8, generator=g)
    hub_dir = F.normalize(F.normalize(keys, dim=-1).sum(dim=2), dim=-1)  # (1, 2, 8) mean direction per head
    keys[:, :, 0, :] = hub_dir * 3.0  # magnitude is normalized away; only the direction matters
    assert press.score(None, None, keys, keys, None, {}).argmax(-1).eq(0).all()


def test_damping_zero_recovers_base_kept_set():
    # damping=0 => c = softmax(base); softmax is monotonic so the kept set equals the base press's.
    base = KnormPress()
    press = CentralityPress(base_press=base, damping=0.0, compression_ratio=0.5, n_sink=0, recent_window=0)
    keys = torch.randn(1, 2, 48, 8)
    ours = press.score(None, None, keys, keys, None, {}).topk(24, dim=-1).indices.sort(-1).values
    theirs = base.score(None, None, keys, keys, None, {}).topk(24, dim=-1).indices.sort(-1).values
    assert torch.equal(ours, theirs)


def test_personalization_keeps_the_needle():
    # THE THESIS TEST: an outlier needle that the base press flags must survive via teleport,
    # even though pure centrality (which ignores base importance) evicts it as an outlier.
    base = KnormPress()
    keys = _coherent_haystack_with_needle(S=40, D=8, needle_idx=7)
    ppr = CentralityPress(base_press=base, damping=0.5, compression_ratio=0.8, n_sink=0, recent_window=0)
    pure = CentralityPress(base_press=None, damping=1.0, compression_ratio=0.8, n_sink=0, recent_window=0)
    kept_ppr = ppr.score(None, None, keys, keys, None, {}).topk(8, dim=-1).indices[0, 0].tolist()
    kept_pure = pure.score(None, None, keys, keys, None, {}).topk(8, dim=-1).indices[0, 0].tolist()
    assert 7 in kept_ppr  # personalization retains the outlier needle
    assert 7 not in kept_pure  # pure centrality evicts it (documents the failure mode)


def test_sink_and_window_protected():
    press = CentralityPress(base_press=None, compression_ratio=0.9, n_sink=2, recent_window=2)
    keys = torch.randn(1, 1, 20, 4)
    top = press.score(None, None, keys, keys, None, {}).topk(4, dim=-1).indices[0, 0].tolist()
    assert {0, 1, 18, 19}.issubset(set(top))


def test_determinism():
    press = CentralityPress(base_press=KnormPress(), compression_ratio=0.5)
    keys = torch.randn(1, 2, 32, 8)
    a = press.score(None, None, keys, keys, None, {})
    b = press.score(None, None, keys, keys, None, {})
    assert torch.allclose(a, b)


def test_scales_to_long_sequence():
    # The shifted kernel must handle a long sequence without materializing an S x S matrix.
    press = CentralityPress(
        base_press=None, similarity="shifted", num_iters=2, n_sink=0, recent_window=0, compression_ratio=0.5
    )
    keys = torch.randn(1, 1, 8192, 16)
    s = press.score(None, None, keys, keys, None, {})
    assert s.shape == (1, 1, 8192)
    assert torch.isfinite(s).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak-memory scaling test requires CUDA")
def test_peak_memory_scales_linearly():
    # Shifted kernel peak memory must grow ~linearly in S, not ~quadratically (O(S^2) => ~4x for 2x S).
    def peak_for(S):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        press = CentralityPress(
            base_press=None, similarity="shifted", num_iters=3, n_sink=0, recent_window=0, compression_ratio=0.5
        )
        keys = torch.randn(1, 4, S, 64, device="cuda")
        press.score(None, None, keys, keys, None, {})
        return torch.cuda.max_memory_allocated()

    p1, p2 = peak_for(8192), peak_for(16384)
    assert p2 < 3 * p1, f"peak memory grew super-linearly: {p1} -> {p2}"


def test_teleport_temp_controls_sharpness():
    # teleport_temp is the softmax temperature over the base scores. At damping=0 the returned scores ARE
    # the teleport p = softmax(base / teleport_temp), so its effective support (exp of its entropy) must
    # SHRINK as the temperature shrinks. Guards the knob the report's tau (teleport-sharpness) analysis
    # relies on -- without this, teleport_temp could silently regress to a no-op.
    base = KnormPress()
    keys = torch.randn(1, 1, 128, 16, generator=torch.Generator().manual_seed(0))

    def eff_support(temp):
        p = CentralityPress(
            base_press=base, damping=0.0, teleport_temp=temp, n_sink=0, recent_window=0, compression_ratio=0.5
        ).score(None, None, keys, keys, None, {})[0, 0]
        return torch.exp(-(p * (p + 1e-12).log()).sum())  # exp(entropy) = effective # of positions

    sharp, mid, flat = eff_support(0.3), eff_support(1.0), eff_support(3.0)
    assert sharp < mid < flat  # smaller temperature -> sharper teleport -> fewer effective positions


def test_bf16_keys_scores_not_collapsed():
    # Scores are returned in fp32; on bf16 keys the ~1/S band must NOT collapse into a few tied values.
    press = CentralityPress(
        base_press=None,
        damping=1.0,
        similarity="shifted",
        num_iters=3,
        n_sink=0,
        recent_window=0,
        compression_ratio=0.5,
    )
    keys = torch.randn(1, 1, 2048, 16, dtype=torch.bfloat16)
    s = press.score(None, None, keys, keys, None, {})
    assert s.dtype == torch.float32
    assert torch.isfinite(s).all()
    assert s[0, 0].unique().numel() > 1000  # bf16 return would collapse this to a handful of values


def test_degenerate_inputs_finite():
    for keys in (torch.zeros(1, 1, 16, 8), torch.ones(1, 1, 16, 8)):
        s = CentralityPress(base_press=None, damping=1.0, compression_ratio=0.5)
        assert torch.isfinite(s.score(None, None, keys, keys, None, {})).all()
    keys = torch.zeros(1, 1, 16, 8)
    s = CentralityPress(base_press=KnormPress(), damping=0.5, compression_ratio=0.5)
    assert torch.isfinite(s.score(None, None, keys, keys, None, {})).all()


def test_standardize_teleport_rescues_tiny_scale_base():
    # A tiny-magnitude base (like SnapKV) makes softmax(base/τ) ~uniform -> pure centrality drops the
    # needle; z-scoring the teleport restores the base ranking and keeps it.
    keys = _coherent_haystack_with_needle(S=40, D=8, needle_idx=7)
    kw = dict(base_press=_TinyScaleBase(), damping=0.5, compression_ratio=0.8, n_sink=0, recent_window=0)
    kept_std = CentralityPress(standardize_teleport=True, **kw).score(None, None, keys, keys, None, {})
    kept_raw = CentralityPress(standardize_teleport=False, **kw).score(None, None, keys, keys, None, {})
    assert 7 in kept_std.topk(8, dim=-1).indices[0, 0].tolist()  # standardized teleport keeps the needle
    assert 7 not in kept_raw.topk(8, dim=-1).indices[0, 0].tolist()  # tiny-scale base degenerates to pure


def test_end_to_end_cache_shrinks(unit_test_model):  # noqa: F811
    B, S = 5, 256
    press = CentralityPress(base_press=KnormPress(), compression_ratio=0.1)
    with press(unit_test_model):
        ids = torch.randint(0, 3000, (B, S), device=unit_test_model.device)
        pkv = unit_test_model(ids, past_key_values=DynamicCache()).past_key_values
    assert pkv.layers[0].keys.shape[2] == int(S * 0.9)


def test_identity_at_zero_ratio(unit_test_model):  # noqa: F811
    B, S = 5, 128
    press = CentralityPress(base_press=KnormPress(), compression_ratio=0.0)
    with press(unit_test_model):
        ids = torch.randint(0, 3000, (B, S), device=unit_test_model.device)
        pkv = unit_test_model(ids, past_key_values=DynamicCache()).past_key_values
    assert pkv.layers[0].keys.shape[2] == S
