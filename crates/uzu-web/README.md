# Uzu in the browser

`uzu-web` exposes the existing `uzu-engine` language model to a dedicated browser
worker. The engine keeps its synchronous backend and command-buffer interfaces.
A separate GPU worker owns WebGPU; shared completion signals let the inference
worker wait without blocking the page or the GPU worker's event loop.

From `apps/mirai-gui`:

```sh
rustup target add wasm32-unknown-unknown
# Install the wasm-bindgen-cli version recorded in Cargo.lock.
cargo install wasm-bindgen-cli --locked --version 0.2.126
pnpm web:setup
pnpm web:wasm
pnpm web:dev
```

The app fetches the production Mirai model catalog and downloads pinned model
files directly, with pause/resume and checksum verification. No model files are
bundled with the app. Downloaded models, settings and chats remain in browser
storage; clearing site data removes them. See the [app setup](../../apps/mirai-gui/README.md#browser-setup)
for deployment, storage and catalog details.

WebGPU needs a secure context (`https` or localhost). The server must send
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`; Vite's web mode sets both. The browser
must expose WebGPU in workers, `SharedArrayBuffer`, and nested workers. A CSP must
allow the blob GPU worker. Chrome on Apple Silicon is exercised by the integration
tests; other browser/device combinations need their own validation.

## Worker API

```js
import init, { createSession, inspectEncoding, inspectModelConfig } from "/wasm/uzu_web.js";
await init();
// Use the encoding supplied by the model registry, as the desktop app does.
const encoding = { type: "hanashi", name: "qwen3.5" };
const capabilities = inspectEncoding(encoding);
inspectModelConfig(configBytes); // no GPU allocation or weights required
const session = await createSession({
  config: configBytes, tokenizer: tokenizerBytes, weights: weightsBlob,
  encoding, seed: 42, // uses the engine's recommended context
});
await session.run({
  repoId: "local/model",
  messages: [{ role: "user", content: "Hello" }],
  samplingPolicy: { type: "Greedy" },
  reasoningEffort: "disabled",
  maxOutputTokens: 128,
}, event => postMessage(event));
session.cancel();
session.dispose();
```

Use this API inside a dedicated worker. `run` accepts the GUI's existing chat
parameters and emits its transcript/delta/chunk/done events. Errors reject the
promise. Registry models use the same Hanashi templates, parser, reasoning modes,
and tool encodings as desktop. `inspectEncoding` derives capabilities from that
encoding without loading a model. `inspectModelConfig` rejects unsupported
architectures before downloading weights; tensor formats and device allocation
limits are checked during model loading.

Config and tokenizer inputs are `Uint8Array`s. Weights accept either a `Blob`
or a `Uint8Array`; the app composes a Blob from its verified cached download
chunks. The loader reads at most 4 MiB at a time and waits for each GPU upload
before releasing that chunk. Config and tokenizer inputs are limited to 16 MiB
and 128 MiB respectively; weights must have a nonzero, safe-integer byte length.
Weight tensors stay on the GPU without persistent
WASM copies. A later CPU access creates a retained mirror only for the tensor
being accessed. File offsets remain 64-bit on wasm32, so loading does not require
a single allocation large enough to hold the checkpoint. `supportsBlobWeights()`
lets the app reject an outdated WASM bundle before opening model files.

The optional `seed` is a nonnegative JavaScript safe integer. It initializes each
fresh inference state for reproducible sampling; a continued conversation retains
its random-number state along with its KV cache. Omitting it uses the current time.

To cancel model construction from another worker or the page, pass
`loadCancellation: new Int32Array(new SharedArrayBuffer(4))` in `createSession`.
Set its first element to a nonzero value with `Atomics.store` to cancel. The loader
checks between bounded reads and completed GPU uploads, releases partial model
allocations, and rejects construction. The app uses this path when stopping a
response while its model is still loading. A cancelled flag stays set; reset it
to zero or supply a new flag before another construction attempt.

Matching complete token prefixes reuse the KV cache and recurrent state. The
session retains internal tool messages across consecutive completed turns.
Edited, cancelled or differently configured histories safely rebuild their state
when their token prefix no longer matches. The Qwen Hanashi integration test
reuses 47 cached tokens on its second turn. For standalone converted models, the
optional encoding can be omitted to use the model's Jinja template; that fallback
can rebuild prefixes when the template rewrites old assistant reasoning and does
not parse tool calls.

Cancellation yields between decoded tokens and each 256-token prefill batch, and
also releases a run awaiting a tool promise. Already submitted GPU work completes
before buffers are released. Concurrent runs reject, and `dispose` cancels a run
before releasing its model. The app also supports title cancellation and automatic
model ejection after the configured idle period.

## Chat features

The browser uses the shared desktop UI for saved chats, regeneration, response
streaming, reasoning controls, sampling parameters and text-file attachments.
It preserves tool settings and uses actual local inference for chat titles.
The browser worker registers the same date/time, chart and chat-name tools,
validates their arguments, executes them locally, and feeds results back to the
model for automatic continuation. Hidden naming calls stay out of the transcript;
charts use the shared data-only chart validator and renderer. Model encodings
and checkpoint capability flags determine whether reasoning and tool controls
are available; a checkpoint that disables a capability takes precedence.

For direct integrations, pass tool schemas in `run({ tools: [...] }, onEvent,
onTool)`. `onTool({ name, arguments })` may return a value or a promise resolving
to `{ value, failed?, chart?, chatName? }`; `value` is the tool result inserted in
model history. A failed or disabled call is not silently treated as successful.

The existing pure-Rust xgrammar engine also runs in WASM. A direct run can pass
`grammar: { type: "JsonSchema", schema: {...} }`,
`{ type: "Regex", pattern: "yes|no" }`, or `{ type: "BuiltinJson" }` to constrain
GPU token sampling. The grammar starts after an opening reasoning section ends,
when the model supplies the configured thinking delimiter. Grammar and tool
execution must be separate runs. Neither desktop nor browser currently exposes
a grammar editor in the chat UI.

The default context comes from the engine's `recommended_context_length()`, as in
the desktop app. For models with a finite context, this is normally at most
16,384 tokens on WebGPU because this backend does not expose sparse buffers.
An explicit `contextLength` can request up to 65,536 tokens, subject to the model
and device memory. Omitting `maxOutputTokens` leaves generation bounded by EOS,
the context and cancellation; an explicit limit accepts 1–65,536 tokens.
A conversation that exceeds its context reports an
error rather than silently truncating messages. Weights currently use one
safetensors file. Each individual buffer must fit both the adapter's storage-buffer
binding limit and the wasm32 CPU-view limit of 2,147,483,584 bytes. This preserves
the engine's ability to access any buffer through a stable CPU view; it does not
limit the aggregate checkpoint size. The model plus inference state must fit
available GPU memory.
The checkpoint remains in browser storage; tokenizer, metadata and inference
bookkeeping still use WASM memory.

Actual cached Qwen3.5 checkpoints of 2,441,821,768 bytes (4B) and 5,189,121,136 bytes
(9B) loaded from composite Blobs in Chrome, each with 186,384,384 bytes of WASM
linear memory after construction in the final benchmark harness. This measures WASM memory only, excluding GPU
weights, JavaScript and browser Blob storage. Separate 64-token generation runs
also matched native Metal, and cancellation/recovery/disposal checks passed. This validates
loading beyond both 2 GB and 4 GiB; it is separate from the live registry/download
test of the 124,955,416-byte LFM checkpoint.

## Coverage and current limits

The backend supports dense causal/GQA attention, LFM2 short convolution,
Qwen3.5 gated delta networks (GDN), F32/F16/BF16 tensor storage, and packed U4/U8
matrix multiplication with the engine's supported quantization layouts. BF16 is
stored as packed integers and converted by the Slang binding adapter; shader-f16
support is not required. Decode uses packed subgroup matrix-vector kernels with
fused H32 output transforms, parallel attention, coalesced GDN state access, a
two-stage greedy argmax, and parallel selection for small top-k sampling policies.
The top-k path retains full-vocabulary normalization before nucleus filtering
and uses the existing Philox token mapping. Short W4 prefills reuse the packed
subgroup kernel for up to eight rows, or up to 32 rows for projections smaller
than 2,048 in either matrix dimension. Larger W4 prefills reuse packed weights across
four rows and inputs across four columns within each subgroup; other formats
retain workgroup tiled matrix multiplication. Buffers,
shaders, pipelines and bind groups persist between submissions; scratch buffers
are recycled. Small GPU results are copied with the submission that writes them,
so token readback does not wait behind the following token's computation. These
snapshots are bounded, invalidated on writes, and retained across read-only use;
larger buffers are read back only on CPU access.

Speculative model loading and Harmony encodings are not yet available in WASM.
Catalog downloads select the base model's config, tokenizer and weights. Optional
speculator assets are excluded while speculative decoding is unavailable.
The browser does not support Mamba2, mixture-of-experts, per-layer embeddings, or
general transformer convolution architectures yet. Unsupported models show a
reason and cannot be selected for chat. Raw Hugging Face/PyTorch weights still
need conversion to Uzu's model format.

## Reproducible checks

Run these checks from the repository root with a local Chrome installation
(`CHROME_BIN` can override the executable). The bridge test exercises actual
WebGPU, including unaligned/overlapping copies, fills, persistent pipelines and
shader error propagation:

```sh
node crates/uzu-engine/tests/webgpu-bridge.mjs
UZU_WORKER_SCRIPT=crates/uzu-engine/tests/webgpu-readback.mjs \
  node crates/uzu-engine/tests/webgpu-bridge.mjs
cargo test -p uzu-web --lib
cargo test -p uzu-engine --no-default-features --features cpu bytes_
```

To validate numerical kernels in the browser, set `SLANGC` to the pinned compiler
installed by `pnpm web:setup`, or put that compiler on `PATH`:

```sh
cargo build -p uzu-web --release --target wasm32-unknown-unknown --features kernel-tests
wasm-bindgen target/wasm32-unknown-unknown/release/uzu_web.wasm \
  --target web --out-dir /tmp/uzu-wasm-tests --out-name uzu_web
UZU_WASM_DIR=/tmp/uzu-wasm-tests node crates/uzu-engine/tests/webgpu-bridge.mjs
```

The feature adds test exports only. The numerical cases check packed/aliased
tensor arithmetic; quantized decode and prefill matrix multiplication; strided
full-precision matrix multiplication; causal GQA at head sizes 64 and 256;
in-place short convolution; BF16 residual normalization and H32 transforms;
dense and quantized embeddings; GDN update, multi-token prefill, convolution and
recurrent state; clipped gated activation; softmax with and without sinks;
packed subgroup matrix-vector multiplication and its fallback; changing shader
cache uniforms and aliases; and greedy and stochastic sampling against CPU semantics,
including large vocabularies, masks and special floating-point values. References
use independent scalar formulas, including a direct Walsh matrix for H32. Set
`UZU_FORCE_NO_SUBGROUPS=1` on the test command to exercise the portable path.
Sampling checks cover fixed seeds, top-k ties, exact nucleus boundaries, masks,
temperature overflow and multi-stage selection. Parallel floating-point sums
can round a near-boundary nucleus cutoff differently from a serial CPU sum.

For full-model parity, run the native reference and pass its JSON to the browser:

```sh
cargo run -p uzu-web --release --example cpu-parity -- \
  /path/to/model "Explain why the sky is blue in detail." 64 > /tmp/uzu-cpu.json
UZU_WASM_DIR=/tmp/uzu-wasm-tests UZU_MODEL_DIR=/path/to/model \
  UZU_PROMPT="Explain why the sky is blue in detail." UZU_MAX_OUTPUT_TOKENS=64 \
  UZU_EXPECTED_TOKENS=/tmp/uzu-cpu.json \
  node crates/uzu-engine/tests/webgpu-bridge.mjs
```

Set `UZU_EXPECT_REUSED_TOKENS=positive`, an exact count, or `0` to assert the
expected next-turn cache behavior. Full-model checks also exercise cancellation,
concurrent-run rejection, recovery after a callback throws, and disposal during
generation. Failed tests retain the results of completed checks.

For bounded loading, source offsets above 4 GiB, failed-upload cleanup and
cancellation during construction:

```sh
UZU_WASM_DIR=/tmp/uzu-wasm-tests \
  UZU_WORKER_SCRIPT=crates/uzu-web/tests-browser-weights.mjs \
  node crates/uzu-engine/tests/webgpu-bridge.mjs
UZU_WORKER_SCRIPT=crates/uzu-engine/tests/webgpu-weights.mjs \
  node crates/uzu-engine/tests/webgpu-bridge.mjs
UZU_WASM_DIR=/tmp/uzu-wasm-tests UZU_MODEL_DIR=/path/to/qwen3.5-0.8b \
  UZU_WORKER_SCRIPT=crates/uzu-web/tests-browser-load-cancellation.mjs \
  node crates/uzu-engine/tests/webgpu-bridge.mjs
```

For the desktop-compatible chat path, use a Qwen3.5 model with the same test WASM:

```sh
UZU_WASM_DIR=/tmp/uzu-wasm-tests UZU_MODEL_DIR=/path/to/qwen3.5-0.8b \
  UZU_WORKER_SCRIPT=crates/uzu-web/tests-browser-chat.mjs \
  node crates/uzu-engine/tests/webgpu-bridge.mjs
```

This reusable test performs actual inference for Hanashi turns, cache reuse,
titles, reasoning, date/chart/naming tools and their continuations, JSON schema,
regular expressions, built-in JSON, cancellation and recovery, and cancellation
of a pending tool promise. It also validates sampling seeds, reproduces seeded
stochastic generation, and rejects invalid model dimensions before GPU allocation.
`pnpm web:fixture /path/to/model --model qwen3.5-0.8b` in the app
directory can prepare local test files under `.web-test-models/`; those files
are not deployed with the browser app.

## Performance and profiling

On an Apple M5 Max, Chrome 152.0.7977.83 exposed the Metal adapter with 32-lane
subgroups. Generation rates exclude loading and prefill. First-token time includes
prompt processing and host work, measured from fresh inference state.

| Model | WebGPU decode | Metal decode | WebGPU first token | Metal first token |
| --- | ---: | ---: | ---: | ---: |
| LFM2.5-230M (Mirai M4) | 1,107 tokens/s | 1,577 tokens/s | 7.03 ms | 3.20 ms |
| Qwen3.5-0.8B (Mirai M4) | 374 tokens/s | 542 tokens/s | 19.33 ms | 6.67 ms |
| Qwen3.5-4B (Mirai M4) | 118 tokens/s | 154 tokens/s | 65.88 ms | 15.39 ms |
| Qwen3.5-9B (Mirai M4) | 71 tokens/s | 96 tokens/s | 102.33 ms | 18.92 ms |

These are medians of three fresh-state trials after a one-token warmup, with
profiling disabled. Browser and native runs use identical cached model files,
a 256-token context, greedy sampling, seed 42 and no speculative model. LFM uses
“Explain why the sky is blue in detail.” with a 64-token cap (19 input, 64 output).
Qwen uses “Say hello in one sentence.”: 0.8B has 18 input tokens and ends at EOS
after 28 output tokens; 4B/9B have 16 input tokens and a 64-token cap. The larger
models' 64 tokens are reasoning prefixes, not completed answers. These short
measurements describe this machine and browser, not other devices. Trial data,
artifact hashes and validation results are in
[`benchmarks/apple-m5-max.json`](benchmarks/apple-m5-max.json).

Metal speed parity has not been reached: measured WebGPU decode is 69–77% of
Metal across these four workloads. The gap is larger in prompt processing.
A separate matched 118-token LFM prompt took 22.81 ms in WebGPU versus 3.95 ms
in Metal; the same text produced 121 Qwen0.8B input tokens and took 72.44 ms
versus 9.85 ms. These are three-trial medians through receipt of the first token,
with the same context and an eight-token generation cap. All final browser
measurements and checks use Chrome's default WebGPU settings. The harness only
adds the unsafe WebGPU flag when explicitly requested with `UZU_UNSAFE_WEBGPU=1`;
it is not needed on the validated machine.

Native Metal has SIMD-group matrix and M5 TensorOps implementations. The current
W4 prefill WebGPU shader reuses packed weights across rows and reduces scalar
FP32 partial sums within subgroups. The
[subgroup-matrix proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md)
is still experimental; Chromium gates it behind unsafe WebGPU, and the pinned
[Slang cooperative-matrix capability](https://github.com/shader-slang/slang/blob/v2026.19/source/slang/slang-capabilities.capdef)
does not include WGSL. This difference is a likely contributor, not a measurement
of its share of the slowdown or a proof that the current kernels are optimal.

The final browser outputs match the CPU reference at all 64 LFM and 28 Qwen0.8B
tokens. Native Metal's Qwen0.8B output differs at one token (“assist” versus “help”).
LFM and both larger Qwen checkpoints match Metal at all 64 tested tokens; both
long-prompt eight-token checks also match. All models passed cancellation,
callback-error recovery, concurrent-run rejection and disposal checks. The final
build passes 177 numerical cases, plus 175 portable cases with two explicitly
skipped subgroup-only grid tests. Browser chat, load cancellation and real
2540-token prompt tests pass; the shared frontend passes 533 tests, lint and
typecheck. Native CPU/Metal normalization regressions also pass.

Seeded stochastic runs are reproducible within this backend. Floating-point
differences can change a sampled token across CPU, Metal and WebGPU even when
the sampling policy and seed match; cross-backend stochastic token identity is
not guaranteed. Set `UZU_SAMPLING_POLICY=null` in either benchmark to measure
checkpoint defaults, or provide the same sampling-policy JSON to both. The
browser's elapsed durations use the monotonic, high-resolution performance clock.

After building the test WASM above, reproduce the browser timings with:

```sh
UZU_BENCH_TRIALS=3 UZU_WASM_DIR=/tmp/uzu-wasm-tests \
  UZU_MODEL_DIR=/path/to/lfm2.5-230m \
  UZU_PROMPT="Explain why the sky is blue in detail." UZU_MAX_OUTPUT_TOKENS=64 \
  node crates/uzu-engine/tests/webgpu-bridge.mjs > /tmp/uzu-lfm-webgpu.json
UZU_BENCH_TRIALS=3 UZU_WASM_DIR=/tmp/uzu-wasm-tests \
  UZU_MODEL_DIR=/path/to/qwen3.5-0.8b \
  UZU_PROMPT="Say hello in one sentence." UZU_MAX_OUTPUT_TOKENS=32 \
  node crates/uzu-engine/tests/webgpu-bridge.mjs > /tmp/uzu-qwen-webgpu.json
```

Leave `UZU_WEBGPU_PROFILE` unset for throughput measurements. Add
`UZU_WEBGPU_PROFILE=1` to either command for host shader/command encoding time,
GPU timestamps when available, worker encoding, submissions, and readback
counters. Profiling skips the numerical fixture prelude. Queue/readback waits
overlap GPU execution, so these durations must not be summed as independent
costs. The JSON `profiles` array contains each trial's timing and optional
counters. Use the median `stats.generateStats.tokensPerSecond` across trials.

The native example uses the same warmup/trial protocol and reports wall time
and process user/system CPU time separately for prefill and generation:

```sh
cargo run -p uzu-web --release --features native-metal --example native-benchmark -- \
  metal /path/to/lfm2.5-230m "Explain why the sky is blue in detail." 64 \
  > /tmp/uzu-lfm-metal.json
cargo run -p uzu-web --release --features native-metal --example native-benchmark -- \
  metal /path/to/qwen3.5-0.8b "Say hello in one sentence." 32 \
  > /tmp/uzu-qwen-metal.json
```

## Slang

Production kernels are written in
[`src/backends/webgpu/kernel/slang`](../uzu-engine/src/backends/webgpu/kernel/slang)
and compiled at build time with Slang 2026.19. This follows the Vulkan branch's
Slang compilation approach, targeting WGSL instead of SPIR-V. The browser runs
the generated WGSL without downloading a Slang compiler.

From `apps/mirai-gui`, `pnpm web:setup` downloads the official pinned compiler and
checks its SHA-256 before installation in the user's cache. `pnpm web:wasm`
discovers it automatically. Direct Cargo builds accept `SLANGC=/path/to/slangc`
or `SLANG_DIR=/path/to/slang`, or find `slangc` on `PATH`. The build caches compiled
variants by source, ABI, compiler version, build logic and specialization width.

`abi.slang` declares typed tensor and parameter accessors using Slang's
[target interoperation mechanism](https://shader-slang.org/slang/user-guide/a1-04-interop.html).
The existing backend supplies storage-format conversion, physical-buffer aliases,
byte offsets and per-dispatch uniforms. This small WGSL binding adapter preserves
the engine's buffer and kernel interfaces; numerical algorithms live in Slang.
Buffers with overlapping views share a physical binding. Packed F16/BF16 writes
remain atomic, and scalar CPU reads are transferred only when needed.

The compiler emits subgroup reductions and shuffles for compatible devices.
Attention and H32 transform variants use a known fixed subgroup width; other
adapters retain workgroup implementations. Shader sources, pipelines, bind groups
and recycled scratch allocations persist across tokens. The correctness suite
covers both subgroup and portable paths, changing aliases and offsets, and
packed tensor formats. Floating-point compilation uses precise mode; throughput
claims require measured model runs and are not inferred from the source language.
