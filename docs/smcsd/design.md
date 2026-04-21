# Sequential Monte Carlo Speculative Decoding in uzu

Port of **SMC-SD** (Abdelfattah Lab, arXiv:2604.15672 — *"Sequential Monte Carlo
Speculative Decoding"*, reference implementation at
<https://github.com/abdelfattah-lab/smcsd>) to the Metal-backed uzu engine.

## Why

Classical speculative decoding (Leviathan et al., Chen et al.) rejects a whole
γ-token draft as soon as any sampled token falls below the per-step accept
threshold. Throughput is capped by the probability that the *worst* token in the
draft is accepted.

SMC-SD reframes the draft as **N importance-weighted particles**, each a length-γ
continuation from the draft model. After the target model rescores all particles
in a single batched forward pass, one particle is **resampled** according to
its importance weight — no rejection, no wasted draft compute. Particles with
zero weight simply don't survive resampling; the ones that do contribute all γ
tokens at once.

Reported gains on Llama-3-8B / TinyLlama drafts are ~1.3–1.7× on top of vanilla
speculative decoding (per the arXiv paper's reported numbers). The upper bound
depends on how well the draft's distribution covers the target's — the same
coupling that makes vanilla SD work, just used better.

## The algorithm (one step)

Given a target model $T$, draft model $D$, prefix $x_{1:t}$, particle count
$N$, draft length $\gamma$:

1. **Draft extend.** For each particle $i \in \{1..N\}$, sample
   $y^{(i)}_{t+1:t+\gamma} \sim D(\cdot | x_{1:t})$ autoregressively.
2. **Target rescore.** In one batched forward pass, compute
   $T(y^{(i)}_j | x_{1:t}, y^{(i)}_{t+1:j-1})$ for all $i, j$.
3. **Importance weights.** $w^{(i)} = \prod_{j=t+1}^{t+\gamma}
   T(y^{(i)}_j | \ldots) / D(y^{(i)}_j | \ldots)$.
4. **Resample.** Draw one index $i^* \sim \mathrm{Categorical}(w / \sum w)$.
   Emit $y^{(i^*)}_{t+1:t+\gamma}$ as the γ accepted tokens.
5. Advance prefix to $x_{1:t+\gamma}$. Repeat.

Effective Sample Size (ESS) monitoring optionally triggers a fallback to a
single-particle greedy step when all weights collapse.

## What uzu has today

| Component | Status |
|---|---|
| Target model forward pass | ✅ `LanguageModelGenerator<B>` |
| Draft model forward pass | ❌ — single-model per session |
| Contiguous KV cache | ✅ `KVCacheLayer<B>` in `forward_pass/kv_cache_layer.rs` |
| Paged KV cache (for N particles sharing prefix) | ❌ |
| Multi-sequence batching | ❌ — `AsyncBuffers` pipelines one seq for latency |
| n-gram speculator | ✅ `speculators/speculator.rs` (trait returns `HashMap<u64, f32>`) |
| Draft-model speculator | ❌ |
| Categorical sampler exposed to caller | Partially — sampling lives inside generator; we'd want per-particle logits |

## Gaps, ranked by pain

1. **Two models, one session.** `ChatSession` assumes one model. Need a
   `SmcSession` that owns two `LanguageModelGenerator<B>` instances.
2. **Shared-prefix KV cache.** With N=4, γ=8 we have 4 divergent suffixes over
   a shared prefix. Without a paged cache we either pay 4× prefix memory or
   snapshot+restore on each particle. Phase 0 will snapshot; Phase 1 refactors
   to paged.
3. **Batched γ rescore.** Target needs to forward N × γ tokens in one pass to
   get the logits matrix. Today uzu runs seq_len ≤ `generate_suffix_length`;
   we'll drive it with suffix_length = γ, once per particle (Phase 0) or
   N × γ in one batched dim (Phase 2).
4. **Draft sampling with logprobs.** SMC needs $D(y|...)$ for each drafted
   token. uzu's sampler consumes logits internally; we'll expose a path that
   returns (token, logprob) per step.

## Phased roadmap

### Phase 0 — Feasibility spike (N=1, serial) — *current*

Goal: end-to-end two-model loop that emits coherent text and lets us measure
baseline latency. No throughput claims yet.

- New module `crates/uzu/src/smc/` (config, error, session, types).
- `SmcSession::new(target_path, draft_path, cfg)` — loads two generators.
- `run_serial`:
  1. Prefill both on the prompt.
  2. Loop: draft γ tokens from D (greedy), rescore with T, emit all γ,
     advance both caches.
  3. Fall back to T-only step on mismatch (Phase 0 skips true SMC resampling —
     this is a *smoke-test skeleton*, not the real algorithm yet).
- `examples/smc_demo.rs` to exercise it against Qwen3.6-35B-A3B (target) +
  a tiny Qwen draft (TBD — candidate: Qwen3-0.5B once converted via lalamo).

**Exit criteria for Phase 0:** coherent output, correct token counts, serial
draft+target latency measured. No speedup expected — this phase is plumbing.

### Phase 1 — Paged KV cache

Refactor `KVCacheLayer<B>` into a block-allocated cache with refcounts so
N particles can share a common prefix's K/V blocks and CoW on divergence.
Big surgery — touches attention encode path, prefix-length math,
`apply_slice` / `get_slice`.

### Phase 2 — N particles, batched rescore

- Add particle scheduler that runs N draft rollouts (serial loop over
  particles in Phase 2a, then parallel via batched forward in Phase 2b).
- Target rescores N × γ tokens in one pass.
- Implement importance weights + categorical resample (CPU; it's trivial).
- ESS fallback.

### Phase 3 — Measurement & tuning

- Benchmark tok/s vs vanilla generation and vs n-gram speculator.
- Sweep (N, γ) grid. Document where it helps vs hurts.
- Decide whether to upstream or keep behind a feature flag.

## Non-goals (for now)

- **Training-time coupling.** SMC-SD works with any draft; we won't distill.
- **Grammar-constrained SMC.** `CompiledGrammar` interacts oddly with
  resampling — defer.
- **Multi-device / distributed.** Single-Mac-GPU only.
