# Reproduction image for the CentralityPress benchmarks — a STANDARD public stack (no Meta/fwdproxy/
# manifold deps). NOTE: the report's numbers were produced on the authoring devserver using a Meta-internal
# torch build (2.13.0+cu130); this Dockerfile instead pins the public stack kvpress declares
# (torch>=2.3.1,<3, transformers>=4.56,<5.3) so a stranger with one CUDA-12 A100 can rebuild. It is
# provided for reproduction and has not been built on the authoring box.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# torch first (CUDA 12.4 wheels), then kvpress + eval deps, then flash-attn (no build isolation, per
# kvpress's pyproject: no-build-isolation-package = ["flash-attn"]).
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -e ".[eval]"   # eval extras required (rouge/jieba/fuzzywuzzy for LongBench)
RUN pip3 install --no-cache-dir -r Appendix/requirements.txt
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

CMD ["bash"]
