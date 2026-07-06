# DFlash Speculative Decoding — uzu Support Plan

Counterpart to lalamo PR [trymirai/lalamo#307](https://github.com/trymirai/lalamo/pull/307)
(branch `hikettei/speculator-dflash`). Tracks what's left to land DFlash in uzu and how
to split it into reviewable PRs.

## What DFlash is (in one paragraph)

DFlash is an EAGLE-style **block draft model**. It conditions on a concatenation of the
target model's hidden states at a few `target_layer_ids`, projects/normalizes them into a
per-draft-layer **context KV cache**, then each step runs a tiny non-causal transformer over
a *noise block* `[last_committed_token, mask_token × (block_size−1)]` (embedded with the
**target's** embedding). It reads out `block_size` draft tokens via the target's readout,
forms a linear `ChainProposal`, and the target verifies the whole chain in one forward pass.
Accepted tokens are a contiguous prefix of the chain.

## Current status (as of 2026-07-06)

**Already on `main`:**
- `config/dflash.rs` — `DFlashDraftConfig` / `DFlashDraftLayerConfig` / `DFlashAttentionConfig`
  (field-for-field match with lalamo's `modules/dflash.py`). Landed in #530.
- `speculators/chain_acceptance.rs` — `ChainProposal::accept` with EOS/bonus-token semantics.
  Landed in #532. Currently only unit-tested, not wired into the stream.
- `bridge/model_metadata.rs` — `DFlashSpeculatorConfig → ModelSpecialization::Speculation`.
- `AnyModelConfig` already discriminates `DFlashSpeculatorConfig`.

**On this branch (`dflash_hidden_capture`):**
- Commit `7160ce6a` — threads `hidden_feature_layer_indices` through
  `Transformer::encode` / `Decoder::encode`, returning `TransformerEncodeOutput`
  / `DecoderEncodeOutput` with a `hidden_features: Box<[Allocation]>`.
- Uncommitted — `HiddenFeatureCaptureMode { Copy, AllocateOnly }` + `hidden_capture_bench.rs`.

## Load-bearing finding: no new Metal kernels are required

`backends/metal/kernel/attention/mask.h::should_use_key` already expresses DFlash's attention
pattern when `is_causal == false`:
- the KV-cache **prefix** (context) is attended unconditionally (no position gate),
- the **suffix** block is **bidirectional** (the causal `key_position <= q_seq_idx` check is skipped),
- the **symmetric ±W/2 sliding window** (mask.h lines ~62–71) is exactly DFlash's window.

The classifier already runs this non-causal flat path end-to-end. `is_causal` is baked into
`AttentionCores` **per Attention instance** at construction (`mixer/attention/mod.rs:169–201`),
so a DFlash attention builds its own `is_causal=false` cores and the **decoder's cores stay
byte-identical** (verify with `/metal-pso-diff`). The remaining work is Rust composition, not
shaders.

---

# PR breakdown

Five PRs. PR2 and PR3 are independent of PR1 and can be developed in parallel; PR4 needs 1–3;
PR5 is the user-facing surface. Rough sizes are net lines.

| PR | Title | ~Size | Depends on |
|----|-------|------:|-----------|
| 1 | Hidden feature capture (land this branch) | +130 | — |
| 2 | Enhance `Attention`: split Q/KV projections, non-causal state, context-append | +180 / −30 | — |
| 3 | DFlash draft model assembly + loading + parity | +400 | 2 |
| 4 | Stream orchestration (draft → verify → accept) | +350 | 1,2,3 |
| 5 | Public API (shoji session) + example + fleet perf | +150 | 4 |

---

## PR 1 — Hidden feature capture

Land the committed capture work from this branch.

- **Recommend dropping `AllocateOnly`** and the `hidden_feature_capture_mode` parameter from the
  production API. It exists only so the bench can separate allocation cost from copy cost, and it
  otherwise adds a parameter to 5 call sites (`decoder.rs`, `transformer.rs`, `classifier.rs`,
  `stream.rs` ×2) forever. Keep the bench with `none` / `copy` cases; the copy is one
  `suffix × model_dim` blit per captured layer and the bench will confirm it is noise for decode.
- If a zero-copy path is ever wanted, it already exists as a cheaper refactor than `AllocateOnly`:
  split `context_projection` into per-target-layer partial GEMVs, since
  `concat(f₁,f₂)·W = f₁·W₁ + f₂·W₂` — the draft can consume each layer's hidden directly with no
  concat/copy. Don't pre-build for it.
- Deliverable: capture threaded through, bench committed under `unit/`, one recorded overhead number
  in the PR description.

---

## PR 2 — Enhance `Attention` for DFlash  *(elaborated; unified design)*

This is the load-bearing PR. Goal: teach the existing `Attention` mixer the three things DFlash
needs — split Q/KV projection weights, non-causal stateful operation, and a context-append entry
point — **with zero shader changes and the decoder's fused fast path untouched**.

> An earlier revision of this plan proposed a separate `DFlashAttention` mixer on the theory that
> fused QKV couldn't express DFlash. That was wrong, for two reasons discovered on closer reading:
> **(a)** in the draft step Q, K, V are all projected from the *same* input (the noise block) —
> lalamo merely *stores* the weights as separate tensors; the only Q-less operation is
> context-append. **(b)** `attention_prepare.metal` already supports KV-only dispatch:
> `is_query = !has_kv || head_idx < num_q_heads`, so calling it with `num_q_heads = 0` over a
> KV-only buffer writes RoPE'd K/V at `kv_token_offset` and never touches queries. No new kernel,
> no shader branch. Given that, a parallel mixer would duplicate orchestration for no benefit —
> enhancing `Attention` is both smaller and more maintainable.

### 2.1 Split Q/KV projections (draft step)

`Attention` today holds one fused `qkv_projection: Box<dyn Linear>` (`mod.rs:44`), projected once
and split by the prepare kernel. lalamo exports DFlash `query_projection` and
`key_value_projection` as separate tensors, and context-append needs the KV projection standalone
— so represent projections as a small internal enum:

```rust
enum QkvProjection<B> {
    Fused(Box<dyn Linear<B>>),                                  // decoder path, unchanged
    Split { query: Box<dyn Linear<B>>, key_value: Box<dyn Linear<B>> },
}
```

In `encode`, the `Split` arm runs two GEMVs and **two prepare dispatches** instead of one:
prepare over the Q buffer with `has_kv=false` (writes RoPE'd queries), prepare over the KV buffer
with `num_q_heads=0` (writes RoPE'd K/V into the cache suffix). Both dispatches are existing
kernel entry points. The `Fused` arm is byte-for-byte the current code — no perf or PSO change
for the decoder. Two dispatches on a `block_size`-row GEMV is noise for the draft model.

`QKVNorm` already operates per-head-range (`encode_head`), so query-norm/key-norm apply to the
split buffers directly. Constructor: add `Attention::new_split(...)` reading
`query_projection`/`key_value_projection` subtrees, sharing the body with `new` via a helper;
DFlash's `DFlashAttentionConfig` maps onto the existing `AttentionConfig` fields
(`num_heads`, `num_groups`, `head_dim`, `scale`, `is_causal=false`, norms, window).

### 2.2 Non-causal attention may own state

`state.rs:72` asserts `is_causal` in `AttentionState::create_empty`, and the stateless encode
branch forces `prefix_length=0`. Relax with an explicit invariant instead of a fork:

- **causal + window ⇒ `Ring` cache** (today's behavior, mask via ring) — unchanged;
- **non-causal ⇒ `Full` cache always**, window (if any) enforced purely as a mask.

Concretely: the assert becomes the invariant check; `create_empty` picks `Ring` only when
`is_causal && sliding_window_size.is_some()`; for non-causal, `max_prefix_elements` sizes from
`max_context_length` (not the window). In `Attention::new`, the cores get
`is_kv_cache_ring: sliding_window_size.is_some() && is_causal`. Since no non-causal-windowed
instance exists today, **every existing instantiation produces identical flags** — decoder cores
and PSOs are unchanged (verify with `/metal-pso-diff`).

The non-causal mask semantics needed (prefix unconditionally visible, suffix bidirectional,
symmetric ±W/2 window at `mask.h:66–70`) are already in the shared `should_use_key` — the
classifier exercises the stateless flat variant; this PR exercises the stateful one.
**One check:** confirm the window branch is reachable with `is_sliding_window && !is_kv_cache_ring`
on a `Full` cache; if the predicate needs a tweak it's one line in `mask.h` (guard with PSO diff).
Window support is only needed if the shipped DFlash config sets `sliding_window_size` — gate on
the actual export (open question 2).

### 2.3 Context-append: `encode_context_append` (KV-only, from external hidden)

The one genuinely new operation — project *external* hidden states into the cache prefix with no
query and no attend. Backs lalamo's `project_context`/`append_state`; runs per prefill chunk and
once per accepted block. Inherent method on `Attention` (not on the `Mixer` trait):

```rust
fn encode_context_append(&self, hidden, positions_start, count, state, encoder)
```

1. `key_value_projection.encode(hidden)` — the `Split` KV Linear (this is why 2.1 exists),
2. key-norm via `QKVNorm`,
3. `prepare` with `num_q_heads=0`, `kv_token_offset = state.length`, cos/sin rows sliced at
   `positions_start` — the kernel indexes rope tables by row, so absolute-position RoPE costs
   nothing extra,
4. `state.length += count` (host-side bookkeeping).

Plus a `truncate(len)` on `AttentionStateType::Full` (~5 lines) for PR4's optimistic-append
rollback.

**Semantics note:** for a DFlash state, the cache advances *only* via `encode_context_append` of
verified target features. The draft-step noise K/V written into the suffix region are ephemeral —
`encode_accept` is never called on this state, and the next append simply overwrites the suffix
region (ordering is guaranteed within the command buffer).

### 2.4 Files touched

- `mixer/attention/mod.rs` — `QkvProjection` enum + `Split` encode arm + `new_split` +
  `encode_context_append` (~+130).
- `mixer/attention/state.rs` — invariant-based state-type selection + `truncate` (~+15/−5).
- `config/token_mixer/attention.rs` — only if a field mapping helper is cleaner there (~+10).
- **No `.metal` changes** expected; `mask.h` predicate tweak only if the 2.2 check fails.

### 2.5 Tests & risk

- **Standalone unit test:** build a split non-causal `Attention` with synthetic weights, feed a
  known context via `encode_context_append`, then a `block_size` noise block through `encode`, and
  assert against a small CPU reference of `softmax(QKᵀ·scale)V` under the DFlash mask (context
  all-visible, block bidirectional, optional window). This pins mask + RoPE positions + key-norm
  before any model assembly.
- **Decoder untouched:** `/metal-pso-diff` on decoder attention PSOs (must be byte-identical —
  only new instantiations get new flag combinations); `/metal-m1-regression` if anything moves.
- **Perf:** draft attention is `block_size × context` per layer — tiny. Context-append is one
  GEMV + one prepare dispatch per accepted block; prefill appends ride the existing chunk encoder.

---

## PR 3 — DFlash draft model assembly + loading + parity

- **New `encodable_block/dflash.rs`:** `DFlashDraft<B>` = `context_projection` (`Linear`) +
  `context_norm` (`Normalization`) + `layers: Box<[DFlashDraftLayer]>` + `output_norm`. Each
  `DFlashDraftLayer` = `input_norm` → split non-causal `Attention` (PR2) → residual →
  `post_attention_norm` → `DenseMLP` → residual. (Maps onto the existing `TransformerLayer` shape
  with both post-norms `None`; reuse `DenseMLP` and `Normalization` as-is. With the unified
  attention, evaluate composing the whole draft as a `Transformer` + preceding
  `context_projection`/`context_norm` — context-append then iterates the layers, either via a
  defaulted `Mixer` trait method or a downcast to `Attention`.)
- **`DFlashDraftState<B>`** = per-layer `AttentionState` + `context_length`. `empty_state`,
  `append` (context-append across layers), and a `truncate(len)` for rollback.
- **`extract_target_features`** — concat captured hidden features at `target_layer_ids` along the
  channel dim (this is the consumer of PR1's `hidden_features`); feed to `context_projection`.
- **Draft readout** reuses the **target's** `Embedding::encode_readout` — no new head. Noise-block
  embedding reuses the target's `encode_lookup`. Cache the `block_size−1` mask-token embeddings once.
- **`Engine::load_dflash_speculator`** — clone `engine/classifier_model.rs:43` (`ParameterTree::
  subtree("...")`, `assert_all_tensors_validated`). **Open contract:** the safetensors namespace the
  `lalamo speculator convert` command emits — confirm with the lalamo side before finalizing.
- **Parity test** — forward the draft against a lalamo-exported reference trace (a fixed context +
  noise block → expected draft logits/argmax), mirroring existing block parity tests.

---

## PR 4 — Stream orchestration

The actual speculation loop, in `engine/language_model/stream/stream.rs`.

- **Prefill:** each chunk forward captures `target_layer_ids` features and calls the draft's
  context-append in the **same encoder** (no extra sync).
- **Decode step (1 GPU sync, same as today's speculative path):**
  1. draft forward over the noise block → GPU-side `argmax` → `ChainProposal` (static linear-chain shape),
  2. `TokenCopySampledKernel` copies the draft tokens into the verify pass's `token_ids` buffer
     (chain shape is static, so the trie/`BatchTopology` is fixed),
  3. verify forward on the target with capture of `target_layer_ids`, **and** optimistically
     context-append **all** block features to the draft cache — all in one command buffer,
  4. after readback: `ChainProposal::accept` (already on `main`) → `encode_accept` on the target KV +
     **truncate the draft cache to `num_verified_nodes`** (accepts are a contiguous prefix, so rollback
     is a length set — no gather kernel, no cross-encoder scratch-lifetime problem).
- **Plumbing:** add an internal two-variant proposal source alongside the existing
  `LanguageModelStreamOptions.speculator` (heuristic `Speculator` trait vs. DFlash). **Do not** invent
  a new trait abstraction for two cases.
- **Punt:** grammar + DFlash — guard the combination and defer to a follow-up (the current grammar
  path assumes the trie bitmask flow).

---

## PR 5 — Public API + example + fleet perf

- **shoji session:** let a chat session select a DFlash speculator model (extend
  `ChatSpeculationPreset` or add `with_speculator_model`), bridge in `bridge/chat_token_backend.rs`.
  `models_for_speculation()` already exists.
- **Example:** `crates/uzu/examples/chat_speculation_dflash.rs`.
- **Perf validation:** fleet run (accept-rate + decode tok/s) and confirm the **no-speculator path is
  unregressed**, mirroring the parity table in lalamo #307. Use `/fleet-run`.

---

## Open questions to resolve with the lalamo side

1. **Weight namespace** emitted by `lalamo speculator convert` (the only external contract PR3 depends on).
2. Whether shipped DFlash configs set `sliding_window_size` (decides whether PR2 §2.2 is in scope).
3. `block_size` and `target_layer_ids` count for the first shipped model (sizing for the noise-embed
   cache and the context-projection input dim).
