# SPDX-License-Identifier: Apache-2.0
"""Systems measurement for CentralityPress (REPORT.md §4.4).

Regenerates REPORT.md §4.4 from the report methodology; run on one A100 to produce `results_systems_bs8.csv`.

For each press at a fixed compression ratio, on a long-prompt workload (default: 8192-token prompt, 128
decode steps, batch 8, Llama-3.1-8B bf16), measures: prefill time (s), peak GPU memory (GB), kept KV-cache
size (MB), decode throughput (tok/s), per-token decode latency mean/p95/p99 (ms), and mean GPU utilization
(%). The report's central fact: at a *fixed ratio* the kept-cache length — hence decode memory/latency/
throughput — is identical across presses; only **prefill** differs. So prefill is timed carefully.

Usage:
  python analysis/speed_memory.py --model meta-llama/Llama-3.1-8B-Instruct --ratio 0.5 \
      --prompt_len 8192 --decode 128 --batch 8 --out results_systems_bs8.csv
"""
import argparse
import csv
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PRESSES = ["no_press", "knorm", "snapkv", "centrality_pure", "centrality_ppr_knorm_d0.15", "graphkv_knorm"]


def _make_press(name, ratio):
    from kvpress import KnormPress, SnapKVPress, CentralityPress
    if name == "no_press":
        return None
    if name == "graphkv_knorm":
        # comparison benchmark (see additional_benchmarks/)
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "additional_benchmarks"))
        from decay_propagation_press import DecayPropagationPress
        p = DecayPropagationPress(base_press=KnormPress())
    else:
        p = {"knorm": KnormPress(), "snapkv": SnapKVPress(),
             "centrality_pure": CentralityPress(base_press=None, damping=1.0),
             "centrality_ppr_knorm_d0.15": CentralityPress(base_press=KnormPress(), damping=0.15)}[name]
    p.compression_ratio = ratio
    return p


def _gpu_util():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        return float("nan")


def measure(model, ids, press, decode_steps):
    dev = model.device
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    cache = DynamicCache()
    ctx = press(model) if press is not None else _nullctx()
    with torch.no_grad(), ctx:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(); prefill_s = time.perf_counter() - t0
    kept = cache.layers[0].keys.shape[2]
    n_layers = len(cache.layers)
    n_kv, hd = cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[3]
    bytes_per = 2  # bf16
    kv_mb = ids.shape[0] * kept * n_layers * 2 * n_kv * hd * bytes_per / 1e6
    peak_prefill = torch.cuda.max_memory_allocated() / 1e9

    tok = out.logits[:, -1:].argmax(-1)
    lat, utils = [], []
    with torch.no_grad():
        for _ in range(decode_steps):
            torch.cuda.synchronize(); s = time.perf_counter()
            out = model(tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize(); lat.append((time.perf_counter() - s) * 1e3)
            tok = out.logits[:, -1:].argmax(-1)
            utils.append(_gpu_util())
    lat = np.array(lat)
    toks = ids.shape[0] * decode_steps
    return dict(prefill_s=round(prefill_s, 3), peak_gpu_gb=round(peak_prefill, 1), kv_cache_mb=round(kv_mb),
                tok_s=round(toks / (lat.sum() / 1e3)), lat_mean_ms=round(lat.mean(), 1),
                lat_p95_ms=round(np.percentile(lat, 95), 1), lat_p99_ms=round(np.percentile(lat, 99), 1),
                gpu_util_pct=round(np.nanmean(utils)))


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--prompt_len", type=int, default=8192)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results_systems_bs8.csv")
    a = ap.parse_args()

    try:
        from transformers.utils import is_flash_attn_2_available
        attn = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except Exception:
        attn = "sdpa"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation=attn).eval()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, tok.vocab_size, (a.batch, a.prompt_len), generator=g).to(model.device)

    rows = []
    for name in PRESSES:
        r = float(0.0) if name == "no_press" else a.ratio
        row = {"press": name, "ratio": r, **measure(model, ids, _make_press(name, r), a.decode)}
        print(row); rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
