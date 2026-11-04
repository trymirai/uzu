# ANE prefill offload

Runs the heavy matmuls of prefill (dense MLPs and the DeltaNet / attention projections) on the Apple
Neural Engine, while the GPU keeps the mixers, norms and everything else. Decode is untouched.
Metal backend only; the shared code only gains defaulted hooks (see "Shared-code delta").

Credit: the ANE runtime approach (private `AppleNeuralEngine.framework` client calls, IOSurface
layout, constant blob) and the W8A8 FFN MIL program are adapted from Splash
(<https://github.com/incoai/splash>, commit 031a25a, Apache License 2.0): `runtime/ane/Program.mm`
and `runtime/ops/AneFfn.cpp`. Not affiliated with Splash.

## Results

Qwen3.5-9B-M (W4 asym g32, RHT), base M4, prefill tok/s:

| prompt | GPU only (`UZU_ANE=0`) | all linears on the ANE, int8 weights (default) | fp16 linear weights (`UZU_ANE_FP16_WEIGHTS=all`) |
|-------:|-----------------------:|-----------------------------------------------:|-------------------------------------------------:|
| 1k     | 225                    | 703                                            | 628                                              |
| 2k     | 224                    | 827                                            | 742                                              |
| 4k     | 222                    | 806                                            | 731                                              |

Mean KL of the prompt logits against GPU-only (`prefill_kl`, single chunk): 1.31e-3 with int8
weights (the default, matching the A8W8/A8W4 precision of the M5 GPU path), 9.78e-4 with fp16 linear
weights (7.12e-4 pipelined). Decode throughput is unchanged (`decode_throughput`).

## Architecture

- `runtime.rs` `AneRuntime`: resolves the framework's classes and client once, checks them.
- `program.rs` `AneProgram`: a compiled, loaded multi-function MIL program (one function per row
  count, `ffn{rows}` / `lin{rows}`; one multi-function program is much cheaper for the ANE service
  than one per count). `evaluate` is synchronous (tests); `enqueue` queues an evaluation that waits
  on a Metal shared event value and signals another. `AneBinding` binds a function's inputs and
  output to surfaces.
- `surface.rs` `AneSurface`: an IOSurface in the ANE's channel-major layout (`[1,1,rows,width]`, rows
  padded to 64 B, allocation to 16 KiB), aliased zero-copy by a Metal buffer (`buffer/external.rs`).
- `mil.rs`: MIL text and constant blob of the W8A8 FFN program and of the linear program.
- `ffn.rs` `AneFfn` (one program per MLP shape, per-layer `AneLayer` constants), `mlp.rs` `AneMlp`
  (wraps a `DenseMlp`; only plain SiLU without clipping), `linear.rs` `AneLinears` (per-lane
  surfaces and GPU kernels shared by all linear programs), `AneLinearProgram` (one per output width
  and weight element type), `AneLinear` (wraps a `Linear`).
- `shared.rs`: what the executors share. `AneKernels` (the GPU kernels around a program),
  `AneExecutor` (program + bindings + per-lane listeners), `submit` (the handoff, below), and
  `AneHealth` (one flag per device: the first failed request retires every executor).
- `trace.rs`: opt-in request/segment timeline (`UZU_ANE_TRACE`), consumed by the test harness only.
- `fake.rs` (test only): fp32 CPU reference of the programs, the oracle of `unit/.../ane/mod.rs`.

GPU kernels (`kernel/ane/ane_ffn.metal`), per call:
1. `AneStageUpRows` / `AneStageDownRows`: dequantize the layer's W4 rows, fold rotations in, requantize
   to int8 (or keep fp16 for linear weights) with one scale per row into the ANE weight surfaces.
   Weights are restaged on every call (no persistent ANE copy of the weights).
2. `AnePack` + `AneTransposeCodes`: `a = H4096(D_a x)`, per-token int8 codes (or fp16 values) and
   token scales, transposed token-major to the channel-major surface.
3. (ANE runs the program.)
4. `AneUnpack`: scales the output, applies the output rotation `D_out H32`, transposes back to
   token-major bf16.

Rotation algebra: mirai weights hold `R_out W R_in^T` with block-32 Hadamard times signs. Staged
up rows are `r' = H_hidden(D_a D_in H32(r))`, so with packed activations `a = H_hidden(D_a x)` the
ANE computes the original product in a rotated basis whose int8 error is near-Gaussian. Down rows
are `r' = D_m H_512(D_din H32(r))`, with the intermediate rotation `D_m H_512` applied on the ANE
before per-token requantization. A linear layer is the FFN's up projection alone. Derivation and an
exact emulation: `scripts/ane_prefill/numerics/pipeline_v1.py` (`ane_emulate.py`, `mlp_ref.py`).

Which layers qualify (`stageable`, `supported_output`): mirai W4 (U4, scale + zero point, group 32,
plain row-major codes), input and output RHT signs present, no biases, input width 4096 (MLP down:
up to 3 x 4096), output width a multiple of 32 up to 16384. Batches under `MIN_TOKENS` (128) or
without a compiled row count (256/512/768/1024) stay on the GPU, as does everything once the ANE
has failed.

## Timeline of one prefill

Prompts are cut into chunks (`pipelined_prefill_chunk_lengths` in `stream.rs`: short and medium
prompts into a large first and a smaller second chunk, so the unhidden tail is small). Chunk c's ANE
work overlaps chunk c+1's GPU work, on two lanes.

- Lane: a `MetalLane` is a command queue with its own shared `event`, plain `order` and `done`
  events and a `relay_queue`. Chunks alternate lanes. `begin_independent_chunk` commits the current
  segment, signals the lane's `order` at a tail value, switches queue and scratch pool, and records
  what the previous chunk reached per layer.
- `order_after_previous_chunk(layer)`: a chunk's layer-L mixer reads state the previous chunk's
  layer-L mixer wrote, so every ANE call site (and every call site that stays on the GPU) makes the
  queue wait for the previous chunk's layer-L ready value (or its tail). This is the only ordering
  between chunks, which is why pipelining needs every dense MLP and layer to qualify.
- `handoff_to_ane(lane, ready, done)` (via `submit`): the encoding is cut into a command buffer
  ending in `signal(lane.event, ready)` and `signal(lane.order, ready)`, and the queue then waits
  for `lane.done >= done`. Metal does not reliably order GPU work across queues through a shared
  event, hence the separate plain `order` event.
- Enqueue: `submit` registers a listener (`notify_listener_at_value`) on `lane.event` at `ready`; when
  the GPU gets there the callback calls `AneProgram::enqueue` with wait=`ready`, signal=`done`. Listeners
  fire in readiness order, so the ANE (in-order) sees requests in the order they became ready, not
  the order they were encoded.
- Done relay (workaround): the ANE signals `lane.event` at `done` itself, but a queue other than the
  first did not reliably wait for that signal (its next command buffer ran before the ANE finished).
  The completion handler therefore calls `MetalLane::relay_done`, which signals the plain `done`
  event from `relay_queue`, and the lane's queue waits on that. The root cause is unknown; the
  workaround was found empirically.
- Failure: any enqueue/completion failure retires `AneHealth`, records the error in the encoding's
  `Handoffs` sink, and still relays `done` so the GPU does not hang. Awaiting the encoding then
  returns an error (`FAILED`). Nothing is replayed; later prefills run entirely on the GPU.

## Shared-code delta

All defaulted, so other backends and models are unaffected (verified against the code):

- `Context` (`backends/common/context.rs`):
  `fn wrap_mlp(&self, mlp: DenseMlp<Self::Backend>, _config: &DenseMLPConfig) -> Box<dyn Mlp<Self::Backend>>`
  (default: `Box::new(mlp)`, called from `dyn Mlp::new`);
  `fn wrap_linear(&self, linear: Box<dyn Linear<Self::Backend>>, _role: LinearRole) -> Box<dyn Linear<Self::Backend>>`
  (default: unchanged; called for the DeltaNet in/out and attention in/out projections);
  `fn pipelines_prefill_chunks(&self) -> bool` (default `false`).
- `Linear` (`encodable_block/linear/mod.rs`): `fn weights(&self) -> Option<LinearWeights<'_, B>>`
  (default `None`; implemented by `matmul.rs` and `rht_wrapper.rs`). `LinearWeights { matrix, has_biases,
  input_rht_signs, output_rht_signs }`. `enum LinearRole { DeltaNetInput, DeltaNetOutput,
  AttentionInput, AttentionOutput }`.
- `DenseMlp::weights(&self) -> Option<DenseMlpWeights<'_, B>>` (inherent, `mlp/dense.rs`): `up` and
  `down` views; the down input signs come from the gate block when the down linear has none.
- `CommandBufferEncoding::begin_independent_chunk(&mut self) -> Result<(), <<Self::CommandBuffer as
  CommandBuffer>::Backend as Backend>::Error>` (default no-op).
- `Transformer::supports_layerwise_prefill_pipelining(&self) -> bool` and
  `Decoder::supports_layerwise_prefill_pipelining(&self) -> bool`: true when every layer's mixer
  state is append-only KV (full attention, no KV sharing) or in-place DeltaNet, with no mixer/MLP conv
  and no `kv_source_layer_index` (`layer_pipelines_prefill`).
- `stream.rs` driver: `prefill_chunk_lengths(prompt_length, pipelined)`; pipelining is used only when
  `speculator.is_none()`, no suffix-repetition sampling, `decoder.supports_layerwise_prefill_pipelining()`
  and `context.pipelines_prefill_chunks()`; it then calls `begin_independent_chunk` before every chunk
  after the first. Without pipelining the chunking is the old fixed 1024. The metal
  `MetalCommandBufferEncoding::begin_independent_chunk` itself falls back to single-lane if the ANE is
  unavailable (`ane_pipelining_available`).
- Metal-only plumbing: multi-segment encodings (`handoff`, `handoff_to_ane`, `failure_sink`, per-lane
  sparse sync) in `metal/command_buffer.rs`, lanes and `AneState` in `metal/context.rs`, timestamp span
  prefixes in `timestamp_recorder.rs`.

## Environment switches

| variable | meaning |
|---|---|
| `UZU_ANE=0` | disable the ANE entirely (GPU-only; also the baseline for KL tests) |
| `UZU_ANE_PIPELINE=0` | keep ANE offload but no cross-lane chunk pipelining (fixed 1024 chunks) |
| `UZU_ANE_LINEARS` | linear roles on the ANE: comma list of `deltanet`, `attention`, `deltanet_in`, `deltanet_out`, `attention_in`, `attention_out`, `all`; `none` for none; default all |
| `UZU_ANE_FP16_WEIGHTS` | roles with fp16 weights (same syntax); unset = none (int8 weights for every role); roles not named use int8 weights |
| `UZU_ANE_ACTIVATIONS=int8` | int8 activations for the linear programs (default fp16) |
| `UZU_ANE_FFN_PRECISION` | `int8` (default), `fp16_in`, `fp16_mid`, `fp16`: unrounded FFN input and/or intermediate |
| `UZU_ANE_WEIGHT_CLIP` | `peak` (default) or a number of std-devs the staged int8 rows cover (experiment; did not help) |
| `UZU_ANE_FFN_WEIGHT_CLIP` | same, for the FFN only; takes precedence over `UZU_ANE_WEIGHT_CLIP` there |
| `UZU_ANE_TRACE=1` | record the timeline (request marks, command-buffer GPU times, named spans) |
| `UZU_ANE_FAULT=enqueue:K` / `completion:K` | test-only: the K-th ANE request (from 1) fails to enqueue / reports failure |

(`UZU_ANE_BLOCKS` does not exist.) Harness inputs: `UZU_PREFILL_MODEL` (model dir), `UZU_PREFILL_LENGTHS`,
`UZU_TIMELINE_LENGTHS`, `UZU_KL_TOKENS`, `UZU_DECODE_PROMPT`, `UZU_FAULT_KINDS`, `UZU_FAULT_LENGTHS`.

## Harness

All are `#[ignore]` and need `UZU_PREFILL_MODEL=~/models/Qwen3.5-9B-M`. Use `--exact` with the full
path, because `prefill_kl` is a prefix of three other tests. The KL tests load the model twice: run
them one at a time (`--test-threads=1` or separate invocations).

```
cargo test --release -p uzu-engine --lib -- --exact --ignored --nocapture \
  uzu_engine::engine::language_model::prefill_kl_test::prefill_kl
```

- `uzu_engine::engine::language_model::prefill_breakdown_test::` `prefill_breakdown` (tok/s and
  per-class GPU time; compare with `UZU_ANE=0`), `prefill_coherence` (greedy continuation),
  `decode_throughput` (run with and without `UZU_ANE=0`), `prefill_timeline` (needs
  `UZU_ANE_TRACE=1`; GPU/ANE overlap analysis and a Chrome trace).
- `uzu_engine::engine::language_model::prefill_kl_test::` `prefill_kl`, `prefill_kl_pipelined`,
  `prefill_kl_split_single_chunk`, `prefill_fault` (failure injection: retirement and error, no
  hang), `ane_linear_numerics` (per-role ANE vs GPU error on real activations).
- `uzu_engine::backends::metal::ane::tests::` `ane_ffn_perf_9b_shape`, `ane_linear_perf_9b_shape`
  (program speed at the 9B shape); `uzu_engine::backends::metal::kernel::ane_ffn_test::` for the
  GPU kernels.

## Limitations

- M4 / non-MXU only. On MXU devices (M5) the GPU re-tiles W4 codes (`GemmBLayout::MxuTiled`, signed
  codes), `stageable` rejects them, and the engine silently stays GPU-only. Whether the ANE would
  still win there: `scripts/ane_prefill/M5_FEASIBILITY.md`.
- One active stream per context: lane events, value counters and the ANE surfaces are per
  `MetalContext`, so concurrent prefills on one context are not supported.
- ANE failure: the prefill in flight errors out, the ANE is retired for the context, no replay.
- The done relay is a workaround with an unknown root cause.
- Memory: two lanes each hold surfaces sized for the widest layer (x, tx, plus per-lane weight/scale/
  output surfaces; fp16-weight programs have their own, `output` rows tall), allocated when the first
  layer is wrapped, in addition to the model.
- Private framework: the API can change with macOS; `AneRuntime::new` fails closed and the engine
  falls back to the GPU.
- Numerics: int8 activations/weights in the FFN cost accuracy (see KL); linear weights default to
  fp16 for that reason.
