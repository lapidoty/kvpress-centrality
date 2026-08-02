# SPDX-License-Identifier: Apache-2.0
"""Effective teleport support exp(H(p)) for every CentralityPress config, from ONE model pass.

Companion to the tau sweep: if macro is a clean function of teleport entropy across bases AND flags, that
single scatter is stronger than any one delta. We capture real post-RoPE keys on a few RULER-4k prefills
(via a KeyCapture subclass, no library edit), then compute the teleport p EXACTLY as the current press does
(standardize over all positions, softmax over all, protected NOT masked -- matching the actual eval runs)
for each (base, standardize, tau), and report mean exp(H(p)) over layers/heads/examples.
"""
import glob
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import CentralityPress, KeyDiffPress, KnormPress

MODEL = "/home/lapidoty/models/Llama-3.1-8B-Instruct"
N_EX = 3
CTX = 4096
DEV = "cuda:0"

_captured = []  # list of (B,H,S,D) fp32 cpu key tensors, one per layer per example


class _KeyCapture(CentralityPress):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        _captured.append(keys.detach().float().cpu())
        return super().score(module, hidden_states, keys, values, attentions, kwargs)


def _ruler_prompts(tok):
    from datasets import load_dataset

    ds = load_dataset("simonjegou/ruler", "4096", split="validation")
    prompts = []
    for i in range(N_EX):
        row = ds[i]
        text = row.get("context") or row.get("input") or row.get("prompt") or ""
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        prompts.append(ids[:, :CTX])
    return prompts


def _teleport_entropy(base, standardize, tau):
    # base: (B,H,S) fp32 ; replicate the CURRENT press teleport exactly (no protected mask)
    b = base.clone()
    if standardize:
        b = (b - b.mean(dim=-1, keepdim=True)) / (b.std(dim=-1, keepdim=True) + 1e-6)
    p = F.softmax(b / tau, dim=-1)  # (B,H,S)
    H = -(p * (p + 1e-12).log()).sum(dim=-1)  # (B,H)
    return torch.exp(H).mean().item()  # effective # of positions


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    prompts = _ruler_prompts(tok)

    press = _KeyCapture(base_press=KnormPress(), compression_ratio=0.75)  # any base; we only need the keys
    with torch.no_grad(), press(model):
        for ids in prompts:
            model(ids.to(DEV))
    print(f"captured {len(_captured)} key tensors (layers x examples)")

    knorm, keydiff = KnormPress(), KeyDiffPress()
    configs = [
        ("knorm", knorm, False, 1.0), ("knorm", knorm, True, 1.0),
        ("knorm", knorm, True, 0.7), ("knorm", knorm, True, 0.5), ("knorm", knorm, True, 0.3),
        ("keydiff", keydiff, True, 1.0), ("keydiff", keydiff, False, 1.0),
    ]
    ents = {}
    for name, bp, std, tau in configs:
        vals = []
        for keys in _captured:
            base = bp.score(None, None, keys, keys, None, {}).float()
            vals.append(_teleport_entropy(base, std, tau))
        key = f"{name}_std{int(std)}_t{tau}"
        ents[key] = sum(vals) / len(vals)
    S = _captured[0].shape[2]
    ents["uniform"] = float(S)  # p = 1/S -> exp(H) = S

    print("\n=== mean effective teleport support exp(H(p)) ===")
    for k, v in ents.items():
        print(f"  {k:22s} {v:9.1f}   (of S={S})")
    json.dump(ents, open("/home/lapidoty/kv-dev/teleport_entropy.json", "w"), indent=2)
    print("\nwrote teleport_entropy.json")


if __name__ == "__main__":
    main()
