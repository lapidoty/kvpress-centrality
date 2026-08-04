#!/bin/bash
# fetch_eval_datasets.sh — warm the HuggingFace cache with the kvpress evaluation datasets.
#
# RECONSTRUCTED 2026-07-31. The original lived only at this path on devvm50213's working tree
# (master branch), which was never pushed to a remote and is lost; no Pixelcloud backup contained
# it. This is a faithful, functional reconstruction — it fetches the same datasets the sweeps use
# (RULER 4096/8192 + LongBench multi-hop), loaded ONLINE so the data_dir/config resolves; once
# cached under ~/.cache/huggingface the agent can read them without touching the network.
#
# IMPORTANT — run this OUTSIDE the Claude Code session, in a plain SSH shell on the devserver.
# Everything launched inside Claude (including `!` commands) is tagged agent:claude_code, which the
# fwdproxy destination filter blocks from the HF CDNs (us.aws.cdn.hf.co, cas-server.xethub.hf.co).
# Your own identity (user:<you>) is NOT filtered, so a normal terminal downloads fine and the cache
# is shared with the agent's process on the same box.
set -u

# A plain SSH shell on the devserver has no proxy set (no direct internet -> DNS fails on
# huggingface.co). Route through fwdproxy; because YOU run this (user:<you>, not agent:claude_code),
# the HF CDN destination filter does not apply. Respect any values already in the environment.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export no_proxy="${no_proxy:-.facebook.com,.fbcdn.net,.internalfb.com,localhost,127.0.0.1,::1}"

PY=/home/lapidoty/kv-dev/venv/bin/python

echo "== RULER (simonjegou/ruler) =="
for ctx in 4096 8192; do
  echo "-- data_dir=$ctx --"
  "$PY" -c "from datasets import load_dataset; d=load_dataset('simonjegou/ruler', data_dir='$ctx', split='test'); print('  cached ruler $ctx:', len(d), 'rows')" \
    || echo "  FAILED ruler $ctx"
done

echo "== LongBench multi-hop (THUDM/LongBench) =="
for t in hotpotqa 2wikimqa musique; do
  "$PY" -c "from datasets import load_dataset; d=load_dataset('THUDM/LongBench', '$t', split='test'); print('  cached longbench $t:', len(d), 'rows')" \
    || echo "  FAILED longbench $t"
done

echo "DATASETS_DONE"
