# Big Refactor Parity TODO

Inventory of the current refactor checkout against current main
(`../uzu-main-1`). This tracks remaining broken, stubbed, deleted, or
behaviorally incomplete work before claiming feature parity.

Audited from:

- current checkout files under `crates/backend-uzu/src` and `crates/uzu/src`
- current main reference under `../uzu-main-1/crates/backend-uzu/src`
- static searches for stale `session`, `inference`, generator, mixer, and
  attention TODO surfaces

Current verification state:

- No cargo command was rerun for this refresh.
- Static refresh notes: DeltaNet is now implemented in the new `Mixer` model;
  older entries that described it as a `todo!()` stub were removed or narrowed
  to validation/runtime coverage.
- Static refresh notes: attention now has the unified `AttentionPrepare` path,
  GEMM core dispatch for 64/128/256 head dims, and the 512-head-dim fallback
  path wired for non-trie suffixes longer than 8 tokens. Remaining attention
  items are validation, dtype coverage, trie/speculative behavior, and state
  semantics rather than "core not implemented".
- Static refresh notes: `max_suffix_length` is gone from the state API, but
  attention and short-conv still hardcode a 1024 suffix capacity internally.
- Static refresh notes: matmul B offsets are folded into `BufferArg`; there is
  no separate `b_offset` parity item left.
- Static stale compile surfaces remain: `crates/uzu` imports deleted
  `backend_uzu::inference`, while backend examples/tests/benches still import
  deleted `backend_uzu::session` and old generator APIs.
- `crates/backend-uzu/examples/engine.rs` is a temporary smoke harness, not a
  parity replacement for the old session/generator layer.

Intentional current scope:

- Backend tracing was removed from `backend-uzu` for now and is not tracked here
  as an immediate parity item. A future tracing rewrite should be designed
  separately.
- XGrammar under the new engine is an intentional branch change, but public
  grammar wiring is still listed below.

## Public Runtime API

- [ ] Reconnect the public `uzu` crate to a backend API.
  - `backend_uzu::inference` was deleted.
  - `crates/uzu/src/engine/mod.rs` still expects backend capability objects for
    chat and classification sessions.
  - `crates/uzu/src/registry/local/config.rs` still expects
    `resolve_model_specialization`.
  - Decide whether the new owner is `backend_uzu::engine`, a bridge crate layer,
    or restored inference traits, then update `crates/uzu`, `nagare`, registry,
    CLI, examples, and tests to that shape.

- [ ] Replace or restore the old session layer.
  - Deleted: `session::{ChatSession, ClassificationSession, TtsSession}`,
    decoding/run/speculator/grammar config, input rendering, output parsing,
    stats, roles/messages, context save/restore, and memory checks.
  - Current replacement: `engine::LanguageModel` loads weights and exposes a
    minimal sync token stream.
  - Needed for parity: chat prompt rendering/tokenization/detokenization,
    cancellation, finish reasons, stats, output parsing, thinking toggles,
    grammar config, speculator config, context mode/length, prefill step size,
    async batch size, session state reset/reconfigure, and model-type routing.

- [ ] Restore backend capability discovery and specialization.
  - Deleted `inference/model_metadata.rs`, `inference/container.rs`, and
    specialization resolution.
  - `crates/uzu` still uses model specializations to decide chat vs
    classification vs TTS.

- [ ] Restore model loading and memory policy surfaces.
  - Deleted session memory checker and old model-loading context wrappers.
  - New `Engine::load_language_model` is intentionally narrow and currently has
    no public loading policy/config surface comparable to main.

## Generation, Sampling, and Context

- [ ] Replace the temporary sync generation loop.
  - `engine/language_model/stream/sync.rs` is a slow temporary loop.
  - It drains all queued input plus one seed token every call, samples exactly
    one token, uses flat topology only, and accepts every forwarded row.
  - Needed for parity: prefill chunking, normal generate loop, speculative trie
    suffix, grammar acceptance, context limits, prefix offsets, cache
    registration, async generation, GPU capture hooks, and cancellation.

- [ ] Wire sampling options into the actual stream path and public API.
  - `LanguageModel::default_stream_options` now builds a `SamplingMethod` from
    generation config, but `engine/language_model/stream/sync.rs` ignores
    `options.sampling_method` and still calls greedy sampling.
  - Seeds, grammar bitmasks, repetition context-ring inputs, and token-id inputs
    are not supplied by the public stream.
  - Preserve main's `sampling_start`/`sampling_length` semantics for partial
    prefill and repetition-penalty slicing.

- [ ] Restore speculator and trie generation behavior.
  - Low-level trie/speculator pieces still exist, but the new stream never
    constructs speculative suffix tries or accepts/rejects sampled candidates.
  - Needed for parity: prompt lookup/ngram/fixed-token speculators,
    grammar-aware trie masks, speculative prefill suffix sampling, and
    accepted-index cache updates.

- [ ] Wire low-level grammar support into public generation.
  - Internal `engine::language_model::grammar::{GrammarConfig, CompiledGrammar}`
    and the XGrammar backend exist.
  - Still missing: public grammar options, creating compiled grammar for a
    stream, token bitmask upload, grammar accept/reject/rollback handling,
    grammar termination handling, and passing grammar masks into sampling.

- [ ] Restore context management.
  - Main supports context length resolution from model/default/max/custom,
    sparse-buffer availability, iOS limits, save/restore slices, and session
    reconfiguration from saved context.
  - Current `LanguageModel::create_empty_state` hardcodes
    `max_context_length = None`.
  - Mixer states still have local `suffix_capacity = 1024` TODOs in attention
    and short-conv.

## Attention and KV Cache

- [ ] Finish attention accept/commit semantics.
  - Full-state accept now emits `KVCacheUpdate` copies for non-contiguous
    accepted indices and increments length, but it needs focused tests for
    contiguous decode, speculative holes, bounds, and context overflow.
  - Ring/SWA accept now emits ordered copies from the suffix area into the ring
    tail, but it needs focused tests for wraparound, full-ring overwrite, and
    speculative accepted-index permutations.
  - Trie/speculative accept still needs end-to-end validation: the current copy
    path assumes sorted accepted indices and needs proof against the trie/euler
    ordering used by speculative generation.
  - Preserve main's separation between forward-time K/V publication and
    accept-time row commit/permutation.

- [ ] Restore cache state management around attention.
  - The refactor state model lacks main's cache slice/apply/clone/copy behavior.
  - No `suffix_start` or explicit prefix-registration model exists yet.
  - Sparse mapping only covers full attention state preparation; windowed/ring
    row mapping and accepted-row mapping still need parity review.

- [ ] Validate shared KV cache semantics.
  - `TransformerLayerStateType::Shared` stores an index into another layer's state,
    but construction does not validate that the source is a prior owned layer.
  - Main validates that shared attention geometry matches
    `(num_groups, head_dim, sliding_window_size)`.
  - Main also rejects mismatches between `AttentionConfig.is_kv_sharing` and
    `kv_source_layer_index`; the refactor currently lets those become later
    panics or state-shape assumptions.

- [ ] Complete attention variant coverage and validation.
  - `AttentionPrepare` only has BF16 element variants; main's separate
    `Rope`/`QkUnpack` kernels supported F32, F16, and BF16.
  - GEMM attention is wired for 64/128/256 head dims, and fallback attention is
    wired for non-trie 512-head-dim suffixes longer than 8 tokens.
  - Remaining work: validate GEMM/fallback against main, cover sinks, ring/SWA,
    KV-sharing, stateless attention, and trie/non-trie dispatch boundaries.
  - Fallback currently asserts non-trie and has no trie path; decide whether
    GEMM is enough for trie large-suffix coverage or whether fallback needs a
    trie-capable variant.

- [ ] Validate stateless/classifier attention.
  - Stateless attention is flat-only and rejects KV sharing/trie.
  - The new path uses token-major scratch K/V, unlike main's old no-cache
    classifier layout. This is the desired direction, but it needs focused
    parity tests before classifier is wired back.

- [ ] Add focused mixer tests.
  - Cover `AttentionPrepare` against old RoPE/QK unpack behavior.
  - Cover full and ring accept, including accepted-index permutation.
  - Cover GEMM and fallback attention against the reference paths, including
    sinks, ring/SWA, KV sharing, and stateless/no-cache attention.
  - Cover KV-sharing query-only RoPE and shared source cache reads.
  - Cover stateless/classifier attention after the classifier wrapper is wired.
  - Cover DeltaNet through the new `Mixer`/`Transformer` path: prefill, decode,
    and accept.

## Mixer-Specific Parity

- [ ] Validate DeltaNet in the new `Mixer`/`MixerState` model.
  - Construction, flat decode/prefill, state allocation, and flat accept are now
    implemented in `encodable_block/mixer/delta_net.rs`.
  - Still needs focused runtime coverage through the new transformer/decoder
    path, including prefill followed by decode and accept.
  - DeltaNet remains non-trie in the new mixer contract, so speculative
    generation must either avoid trie mode or provide a compatible fallback.
  - CPU DeltaNet kernels appear to inherit main's F32-state/F32-small-weight
    mirror mismatch for BF16 activation runs; this is not a refactor-only parity
    delta, but should be tested/fixed if CPU DeltaNet is expected to work.

- [ ] Bring short-conv state semantics back to main parity.
  - F32 conv weights/biases are present.
  - State allocation and prepare still hardcode `suffix_capacity = 1024`.
  - Current state handling is simplified: flat accepts must accept only the last
    row, trie accepts copy one suffix row, and there is no equivalent of main's
    suffix-state valid range.
  - Main handles `sampling_start > 0`, partial prefill before trie suffix,
    trie-only suffix rows, and clearing/setting valid suffix ranges.

- [ ] Bring Mamba2 state semantics back to main parity.
  - The flat prefill/decode path exists.
  - Current path rejects trie topology, requires mutable state, and accepts only
    full flat suffixes.
  - Compare against main's `MambaMixer`, `SSMLayer`, and `SSDPrefillBlock`
    behavior, including env-selected prefill mode and zero/optional conv-state
    edge cases.

- [ ] Decide how non-trie mixers interact with speculative generation.
  - `Mamba2::trie_supported()` and `DeltaNet::trie_supported()` are false.
  - The new stream does not consult `Decoder::trie_supported()` to choose a
    compatible generation strategy.

## Model Type Surfaces

- [ ] Restore classifier support as a lightweight wrapper over `Transformer`.
  - Target example: `workspace/models/0.5.12/chat-moderation-router`.
  - Current reusable pieces exist: classifier config, embedding, `Normalization`,
    `Transformer`, `Pooling`, and `ClassifierPredictionHead`.
  - `encodable_block/classifier.rs` is still a placeholder.
  - Before the wrapper is useful, make generic `Transformer` represent the
    classifier/encoder layer shape:
    - layer norms must use `Normalization`, not `RMSNorm` directly, because this
      model uses `subtract_mean: true` LayerNorm-style norms for pre-mixer
      layers 1-21, all pre-MLP norms, and output norm.
    - layer 0 has no `pre_mixer_norm_config`, but current `TransformerLayer`
      requires one.
    - residual adds need to stay semantically equivalent to old classifier
      layers when moving away from the decoder-specific fused `RMSNorm` path.
  - Implement classifier construction under the safetensors `classifier` root:
    `embedding`, `embedding_norm`, `transformer`, `prediction_head.dense`,
    `prediction_head.norm`, and `prediction_head.readout`.
  - Prediction head wiring needs dense `model_dim -> hidden_dim` with
    `use_dense_bias`, norm over `hidden_dim`, and readout `hidden_dim ->
    num_labels`; this example has a readout bias.
  - Runtime encode should be: token ids -> embedding lookup -> embedding norm ->
    `Transformer::encode(..., output_range: Some(0..seq_len),
    token_topology: Flat, state: None)` -> pooling -> prediction head -> logits
    copyout.
  - Add a public engine/session entry point with classifier prompt rendering,
    tokenizer encode, positions `0..tokens.len()`, sigmoid probability mapping,
    output labels, and model-type routing.
  - Add constructor-time rejects/errors for unsupported classifier inputs:
    KV-sharing attention, trie/stateful-only mixers, embedding readout hadamard,
    and other cases that would otherwise become runtime panics.
  - Attention no-cache/classifier mode still needs focused tests before the
    classifier wrapper is treated as parity-complete.

- [ ] Restore or replace text-to-speech support.
  - Old TTS sessions and inference surfaces were removed.
  - `lib.rs` comments out `audio`, so nanocodec/audio runtime is not exported.
  - Main TTS behavior included FishAudio decoder runtime, TTS prompt rendering,
    streaming audio chunks, run config validation, and backend capability wiring.

## Tests, Benches, and Examples

- [ ] Port or delete stale backend examples.
  - `crates/backend-uzu/examples/fun1.rs` imports removed
    `backend_uzu::session`.
  - `crates/backend-uzu/examples/engine.rs` is temporary smoke code and should
    not become the final public example as-is.

- [ ] Port integration/unit tests from old sessions to the new API.
  - Session chat tests still import removed session APIs.
  - Main coverage included text sessions, sampling, thinking toggle, context
    mode, grammar JSON schema, and TTS nanocodec runtime.

- [ ] Port benchmarks.
  - `language_model_generator_bench.rs` still expects
    `LanguageModelGenerator`, `RunModelResult`, `DecodingConfig`, and old
    session sampling APIs.
  - Session benches still expect `ChatSession`.

## API and Cleanup

- [ ] Replace reachable `todo!()`, `assert!`, and `panic!` boundaries with typed
  errors where model/config input can trigger them.
  - Attention accept now uses copy kernels for full and ring states, but still
    relies on asserts and unchecked shape/order assumptions.
  - State downcasts plus unsupported topology/state access in DeltaNet, Mamba2,
    short-conv, and stateless attention still panic/assert.
  - Layer KV-sharing mismatches are no longer validated as config errors.
  - PLE subtree creation still uses `expect`.

- [ ] Make state acceptance explicit across mixers.
  - Main distinguished prefill cache registration, suffix acceptance, and async
    decode acceptance.
  - Current state APIs take accepted row indices only and have no
    `suffix_start`/prefix-registration model.

- [ ] Decide the final public token type boundary.
  - Embedding lookup still uses `u64` token ids, sampling returns `u32`, and
    `embedding.rs` carries a TODO about matching those dtypes.
  - Public `shoji` token types need a clear conversion boundary.

- [ ] Remove unused transitional code once the real owners exist.
  - Current warnings include many unused exports/types for classifier, pooling,
    prediction heads, sampling variants, and helper enums.
  - Keep future owners only if follow-up tasks actually wire them in.
