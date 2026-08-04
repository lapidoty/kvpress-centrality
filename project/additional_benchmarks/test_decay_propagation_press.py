# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from additional_benchmarks.decay_propagation_press import DecayPropagationPress
from kvpress.presses.knorm_press import KnormPress


def test_base_press_required():
    with pytest.raises(AssertionError):
        DecayPropagationPress(base_press=None, compression_ratio=0.5)


def test_score_shape_and_finite():
    press = DecayPropagationPress(base_press=KnormPress(), compression_ratio=0.5)
    keys = torch.randn(2, 3, 80, 8)
    s = press.score(None, None, keys, keys, None, {})
    assert s.shape == (2, 3, 80)
    assert torch.isfinite(s).all()


def test_keeps_a_unique_important_source():
    # REGRESSION for the self-suppression bug: a unique, isolated most-important token (a source) must
    # survive. The buggy version put cos(k_i, k_i)=1 into the token's own decay product, zeroing it.
    base = KnormPress()
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(1, 1, 30, 8, generator=g)
    keys[:, :, 5, :] = F.normalize(torch.randn(1, 1, 8, generator=g), dim=-1) * 0.02  # smallest norm -> top knorm
    press = DecayPropagationPress(base_press=base, top_k_sources=8, num_rounds=1, n_sink=0, recent_window=0)
    kept = press.score(None, None, keys, keys, None, {}).topk(15, dim=-1).indices[0, 0].tolist()
    assert 5 in kept  # would be evicted by the self-suppression bug


def test_suppresses_redundant_neighbor():
    # A token redundant with a strong source must score BELOW an equally-important isolated token.
    base = KnormPress()
    g = torch.Generator().manual_seed(1)
    D = 8
    keys = torch.randn(1, 1, 20, D, generator=g)
    u = F.normalize(torch.randn(D, generator=g), dim=-1)
    keys[:, :, 0, :] = u * 0.01  # strong source (smallest norm)
    keys[:, :, 1, :] = u * 0.5  # redundant neighbor of the source (same direction)
    v = F.normalize(torch.randn(D, generator=g), dim=-1)
    keys[:, :, 2, :] = v * 0.5  # isolated token, same norm/importance as token 1
    press = DecayPropagationPress(base_press=base, top_k_sources=4, num_rounds=1, n_sink=0, recent_window=0)
    s = press.score(None, None, keys, keys, None, {})[0, 0]
    assert s[1] < s[2]  # redundant neighbor suppressed below the equally-important isolated token


def test_top_k_sources_exceeds_seq_len():
    # k is clamped to S; scores must stay finite and not collapse to a single tied value.
    base = KnormPress()
    press = DecayPropagationPress(base_press=base, top_k_sources=64, num_rounds=1, n_sink=0, recent_window=0)
    keys = torch.randn(1, 1, 20, 8)
    s = press.score(None, None, keys, keys, None, {})
    assert torch.isfinite(s).all()
    assert s[0, 0].unique().numel() > 5  # not collapsed to noise/ties


def test_sink_and_window_protected():
    press = DecayPropagationPress(base_press=KnormPress(), compression_ratio=0.9, n_sink=2, recent_window=2)
    keys = torch.randn(1, 1, 20, 4)
    top = press.score(None, None, keys, keys, None, {}).topk(4, dim=-1).indices[0, 0].tolist()
    assert {0, 1, 18, 19}.issubset(set(top))


def test_determinism():
    press = DecayPropagationPress(base_press=KnormPress(), compression_ratio=0.5)
    keys = torch.randn(1, 2, 80, 8)
    a = press.score(None, None, keys, keys, None, {})
    b = press.score(None, None, keys, keys, None, {})
    assert torch.allclose(a, b)
