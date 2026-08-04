# SPDX-License-Identifier: Apache-2.0
"""Attention-recall probe for CentralityPress (REPORT.md §4.6 — the "hit-rate" analog).

On a *full, uncompressed* cache with eager attention, measures what fraction of the attention mass the model
places on the context (from the last-32 observation-window queries, reduced per KV-head) lands on the token
set each press would KEEP at ratios 0.25/0.5/0.75. Averaged over N real RULER-4k prompts × all layers × KV
heads. The report's finding: the highest-recall presses (snapkv, knorm) are NOT the most accurate — the
needle is a low-attention outlier — so recall is the wrong objective for retrieval.

Usage:
  python analysis/recall_probe.py --model meta-llama/Llama-3.1-8B-Instruct --n_prompts 14 \
      --out results_attention_recall.csv
"""
import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

RATIOS = [0.25, 0.5, 0.75]
OBS = 32  # observation-window queries (last 32)


def build_presses():
    from kvpress import KnormPress, SnapKVPress, CentralityPress
    presses = {"snapkv": SnapKVPress(), "knorm": KnormPress(),
               "centrality_ppr_knorm_d0.15": CentralityPress(base_press=KnormPress(), damping=0.15),
               "centrality_pure": CentralityPress(base_press=None, damping=1.0)}
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "additional_benchmarks"))
        from decay_propagation_press import DecayPropagationPress
        presses["graphkv_knorm"] = DecayPropagationPress(base_press=KnormPress())
    except Exception:
        pass
    return presses


def ruler_prompts(tok, n):
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", "4096", split="validation")
    out = []
    for i in range(n):
        text = ds[i].get("context") or ds[i].get("input") or ""
        out.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, return_tensors="pt")[:, :4096])
    return out


def kept_mask(press, module, keys, values, attn, ratio):
    """Top-(1-ratio) positions per (batch, kv-head) from the press's own score()."""
    S = keys.shape[2]
    keep = max(1, round((1.0 - ratio) * S))
    try:
        scores = press.score(module, None, keys, values, attn, {}).float()  # (B, n_kv, S)
    except Exception:
        scores = (-keys.float().norm(dim=-1))  # fallback: knorm-like
    idx = scores.topk(keep, dim=-1).indices
    m = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
    return m  # (B, n_kv, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n_prompts", type=int, default=14)
    ap.add_argument("--out", default="results_attention_recall.csv")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="eager").eval()  # eager -> attentions
    presses = build_presses()
    layers = model.model.layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    group = n_q // n_kv  # GQA

    acc = {(p, r): [] for p in presses for r in RATIOS}
    for ids in ruler_prompts(tok, a.n_prompts):
        ids = ids.to(model.device)
        cache = DynamicCache()
        with torch.no_grad():
            out = model(ids, past_key_values=cache, use_cache=True, output_attentions=True)
        for li, attn in enumerate(out.attentions):          # attn: (B, n_q, S, S)
            keys = cache.layers[li].keys                     # (B, n_kv, S, hd)
            values = cache.layers[li].values
            S = keys.shape[2]
            # attention mass from the last-OBS queries over the context, per query-head -> per KV-head
            mass_q = attn[:, :, -OBS:, :].mean(dim=2)         # (B, n_q, S)
            mass = mass_q.view(mass_q.shape[0], n_kv, group, S).mean(dim=2)  # (B, n_kv, S)
            mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            for p, press in presses.items():
                for r in RATIOS:
                    m = kept_mask(press, layers[li].self_attn, keys, values, attn, r).to(mass.device)
                    acc[(p, r)].append((mass * m).sum(dim=-1).mean().item())  # recall on kept set

    rows = []
    for p in presses:
        for r in RATIOS:
            vals = acc[(p, r)]
            rows.append({"press": p, "ratio": r, "recall": round(100 * sum(vals) / len(vals), 1)})
            print(rows[-1])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["press", "ratio", "recall"]); w.writeheader(); w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
