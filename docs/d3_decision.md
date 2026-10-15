# D3 decision inference

The standalone D3 adapter runs a converted text decision model and returns its
code logits. The initial target is `vllm-sr/d3-flash` at revision
`521525effa79469d899a570726547dd7f00da3af`. Other D3 checkpoints require their own
conversion, contract checks and prediction validation before support is claimed.

This path uses `uzu-engine` directly. It does not register a model with the
high-level chat API or download checkpoint files. The source HF checkpoint must
first be converted into a native `D3DecisionModelConfig` bundle containing
`config.json` and `model.safetensors`. The companion
[Lalamo conversion contribution](https://github.com/trymirai/lalamo/pull/386)
describes the pinned source and verified INT4/RHT32 export. Uzu loads the native
bundle; it does not perform quantization.

## Execution contract

The adapter validates the D3-Flash geometry: a 4096-dimensional, 32-layer Qwen3.5
text trunk with 24 DeltaNet layers, eight full-attention layers, and 255 decision
codes. Activations and the final normalized text hidden state are BF16. That
last-position hidden state is widened to FP32 before the separate, unquantized
FP32 decision readout. Native quantized trunk and embedding weights can be used
without quantizing the readout. Normalization and DeltaNet state parameters
retain their FP32 representation and configured upcasting behavior. The source
vision tower is outside this path.

Each `infer` call evaluates **one complete prompt**. Full-attention layers are
noncausal, so chunking the input as ordinary causal chat prefill would change the
result. The adapter allocates attention capacity for the complete input,
including prompts longer than the usual 1024-token prefill chunk. Every call
starts with fresh attention and DeltaNet state; no prompt prefix or recurrent
state is reused between requests. Existing chat prefill uses its existing state
construction path.

Configuration checks reject incompatible activation/readout types, causal or
sliding-window attention, inconsistent code mappings, and incompatible RMS norm
precision/scale conventions. These are format and execution checks. They do not
verify that an arbitrary bundle was derived from the pinned source revision;
retain and verify the converter's source provenance separately.

## Rust API

On a supported Apple device, create a Metal engine and load a converted bundle:

```rust,no_run
use std::{error::Error, path::Path};

use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::Engine,
};

fn run<B: Backend>() -> Result<(), Box<dyn Error>> {
    let engine = Engine::<B>::new().map_err(|error| error.to_string())?;
    let model = engine.load_d3_decision_model(Path::new("/path/to/converted-d3"))?;

    // Transport example only: use the source D3 formatter/tokenizer in an app.
    let output = model.infer(&[1, 2, 3], false)?;
    assert_eq!(output.logits.len(), model.num_codes());
    assert!(output.hidden.is_none());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    struct Run;
    impl BackendSelection for Run {
        type Output = ();
        type Error = Box<dyn Error>;

        fn select<B: Backend>(self) -> Result<(), Self::Error> {
            if B::NAME != "metal" {
                return Err("this example requires the Metal backend".into());
            }
            run::<B>()
        }
    }
    select_backend(Run, "Metal backend is unavailable".into())
}
```

`infer(token_ids, return_hidden)` accepts nonempty `u32` token IDs. It rejects
out-of-vocabulary IDs and inputs beyond the effective model/configuration
context bound. It returns:

| Field | Meaning |
| --- | --- |
| `logits` | 255 finite FP32 readout values, in `config.decision.codes` order. |
| `hidden` | The last-position normalized BF16 hidden state widened to 4096 FP32 values, present only when `return_hidden` is true. |
| `input_tokens` | The accepted input length. |
| `forward_ms` | Time from per-call state allocation through completed backend execution; excludes model loading and client formatting/answer processing. |
| `peak_memory_bytes` | Optional backend-reported memory high-water mark; not a per-request memory delta or host RSS measurement. |

The loader validates configuration before loading the weight payload and checks
that the native parameter inventory is fully consumed. Model loading and
inference return typed errors rather than returning an answer after a failed
validation or nonfinite output.

### Logits and answer probabilities

Uzu returns raw logits, without temperature scaling or softmax. The client must
preserve the checkpoint's D3 formatter, tokenizer, decision-code order,
temperature and typed-answer rules. In particular, readout row `i` corresponds
to `decision.codes[i]`; `decision.token_ids[i]` records its source token mapping,
not a row index into the 255-element output.

For a choice question, select its assigned code logits in answer order, divide
by the positive source temperature, then apply a numerically stable FP32
softmax **over those valid choices**. Normalizing all 255 codes changes the
question's probability contract. Score, null-choice and multi-question behavior
must likewise follow the source runtime. Prompt construction, answer mapping
and application-specific routing remain client responsibilities; the example
below does not implement them.

## Metal NDJSON example

Run these commands from the repository root with the repository's nightly Rust
toolchain and Apple Metal build prerequisites installed:

```sh
cargo +nightly build -p uzu-engine --release --example d3_decision \
  --no-default-features --features metal
./target/release/examples/d3_decision /path/to/converted-d3
```

The executable requires Metal; selecting `UZU_BACKEND=cpu` produces an explicit
error. It loads one model and writes a flushed `ready` object to stdout
with `type`, `backend`, `device`, `load_seconds`, `model_dim`, `num_codes` and
`vocab_size`. A startup/load failure exits with an error before `ready`. After
`ready`, send one JSON object per stdin line:

```json
{"id":"request-1","token_ids":[1,2,3],"return_hidden":false}
```

The token IDs above illustrate the transport schema, not a meaningful D3
prompt. `id` and `token_ids` are required; `return_hidden` defaults to false.
Unknown fields and malformed JSON are rejected. Successful responses contain
`id`, `logits`, `forward_ms`, `input_tokens` and `peak_memory_bytes`; `hidden` is
included only when requested. There is no generated-text stream.

Request failures produce one JSON response with `id` and
`error: {"code": ..., "message": ...}`. Codes distinguish `invalid_request`,
`invalid_input`, `backend_error` and `nonfinite_output`. An invalid request has a
null `id` when no string ID can be recovered. The worker continues to the next
input line after these request errors and exits when stdin closes. Keep stdout
for this protocol and inspect stderr/process status for startup or I/O failures.

## Regression checks

The configuration, input-bound and BF16-to-FP32 cast regressions run without
Metal or model downloads:

```sh
cargo +nightly test -p uzu-engine --lib --no-default-features --features cpu d3_
```

These small checks cover contract rejection, request bounds and cast behavior.
They do not establish prediction equivalence or full-model Metal correctness.
For a native-model comparison, preserve identical token IDs, answer ordering,
temperature and source revision, and compare both logits/probabilities and final
answers against a source reference. Measurements made with an earlier Uzu
revision do not validate this implementation.

The [Apple M6 / 32 GiB verification report](reports/d3_flash_apple_m6_2026-10-15.md)
records the actual 0.6.8-based build and its limits. It includes 1,356 same-weight
engine regression requests (zero errors, identical choices and probabilities),
native 1024/1025-token boundary checks and the CPU/Metal build/test selections.
The [measurement JSON](reports/d3_flash_apple_m6_2026-10-15.json) retains exact
scores and artifact hashes; model weights and benchmark payloads are not included.
