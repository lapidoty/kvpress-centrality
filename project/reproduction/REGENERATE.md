# Regenerate the report-only sections (§4.2 / §4.4 / §4.5 / §4.6) — for a machine with a free A100

The scripts under `project/reproduction/scripts/` and `project/reproduction/analysis/` are **faithful reconstructions** from
the report methodology (their headers say so; the originals were lost with devvm50213). They have **not been
re-executed**. Run them on a CUDA-12 A100 to regenerate the data, then rebuild the figures and commit.

```bash
# env  (see project/reproduction/Dockerfile / requirements.txt)
pip install -e ".[eval]" && pip install flash-attn --no-build-isolation
M=meta-llama/Llama-3.1-8B-Instruct
R=project/reproduction/results

# §4.2 LongBench (multi-hop F1)
python project/reproduction/scripts/sweep_longbench.py --model $M --out $R/longbench_summary.json

# §4.4 systems (8192 prompt, 128 decode, batch 8)
python project/reproduction/analysis/speed_memory.py --model $M --ratio 0.5 --out $R/results_systems_bs8.csv

# §4.5 iso-accuracy (16k, batch 8)  — heavy; needs the RULER accuracy sweep as input
python project/reproduction/analysis/iso_systems.py --model $M --out $R/results_iso_systems_16k_bs8.csv

# §4.6 attention recall  — ⚠️ KNOWN ISSUE: recall_probe.py uses eager output_attentions on 4k context,
#   which OOMs an 80GB GPU as written. Fix before running: capture attention per-layer via forward hooks
#   (or reduce context / n_prompts), then:
python project/reproduction/analysis/recall_probe.py --model $M --n_prompts 14 --out $R/results_attention_recall.csv

# figures (regenerates fig3, fig10, fig12, … from the CSVs above; skips any missing input)
python project/reproduction/analysis/make_figures.py --resultsdir $R --figdir project/report/figures

# commit (DCO + agent marker)
git add -A && git commit -s -m "regenerate §4.2/4.4/4.5/4.6 data + figures 🤖🤖🤖" && \
  git -c http.proxy=fwdproxy:8080 push
```

After a successful re-run: confirm the numbers land near the report's tables (within noise), then in
`project/reproduction/MANIFEST.md` move §4.2/4.4/4.5/4.6 out of the "not yet re-run" section and drop the
`Reconstruction …` header from each script that reproduced (replace with `Reproduced <date> <commit>`).
