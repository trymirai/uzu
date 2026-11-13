#!/usr/bin/env bash
# Installs HyperQwen at the pinned commit: vLLM 0.32.0 with syv-ai's patches, its requantized Qwen3.8-27B (4-bit
# weights, lm_head and embeddings) and the W4A16 DFlash2 drafter. Prints the directory to start it from.
set -euo pipefail

COMMIT="6ab9bd05cb84b538a6595d634f3e337ac8a80856"
MODEL_REVISION="1f05c441c4e64ae0549de44fa9ea5a6d43610314"
# On WSL's own disk, like the other Linux engines.
DIR="${XDG_CACHE_HOME:-$HOME/.cache}/uzu-benchmarks/hyperqwen"
MODEL="models/Qwen3.8-27B-W4A16-AutoRound"

if [[ ! -d "$DIR/.git" ]]; then
    git clone --quiet https://github.com/syv-ai/HyperQwen.git "$DIR" >&2
fi
if [[ "$(git -C "$DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
    git -C "$DIR" fetch --quiet origin "$COMMIT" >&2
    git -C "$DIR" checkout --quiet --detach "$COMMIT" >&2
    rm -rf "$DIR/venv"
fi
cd "$DIR"

# docs/install.md's packages. The nvcc pins match the CUDA 13.0 runtime, which FlashInfer's JIT requires.
if [[ ! -f venv/.uzu-patched ]]; then
    rm -rf venv
    uv venv --quiet --python 3.12 venv >&2
    uv pip install --quiet --python venv/bin/python vllm==0.32.0 huggingface_hub hf_transfer ninja pandas \
        nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88 >&2
    uv pip install --quiet --python venv/bin/python --no-deps --index-url https://flashinfer.ai/whl/ \
        flashinfer-cubin==0.7.0.post1 >&2
    bash patches/apply.sh venv/lib/python3.12/site-packages/vllm >&2
    touch venv/.uzu-patched
fi
# The pth-flash-attn kernel, which reads the INT8 per-token-head KV cache the adapter serves with. Its installer builds
# the op library for sm_80/86/89 and installs it with the venv's pip.
if [[ ! -f venv/.uzu-pthfa ]]; then
    uv pip install --quiet --python venv/bin/python pip >&2
    bash pthfa/install.sh >&2
    touch venv/.uzu-pthfa
fi

run() {
    uv run --quiet --no-project --python venv/bin/python "$@" >&2
}
if [[ ! -f "$MODEL/.uzu-prepared" ]]; then
    HF_XET_HIGH_PERFORMANCE=1 venv/bin/hf download dbirks/Qwen3.8-27B-W4A16-AutoRound --revision "$MODEL_REVISION" \
        --local-dir "$MODEL" >&2
    run prepare/quant_lm_head.py "$MODEL"
    run prepare/quant_embed.py "$MODEL"
    run prepare/quant_mtp.py "$MODEL"
    run prepare/build_draft_vocab.py "$MODEL" --ids prepare/draft_vocab_ids.json
    # The single-user variant with a 4-bit lm_head, which the launcher serves when it exists.
    run prepare/fetch_fast_variant.py
    run prepare/fetch_dflash2.py
    touch "$MODEL/.uzu-prepared"
fi
echo "$DIR"
