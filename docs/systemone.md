# Decisions with a loaded language model

Uzu can use a loaded chat model to answer decision questions without generating text.
`DecisionSession` uses either the existing D3 backend or a chat-token scorer.
The HTTP request and response formats remain those provided by upstream Uzu.

## Share one loaded model

```rust
let loaded = engine.chat_instance(model, config).await?;
let chat = engine.chat_with_instance(&loaded).await?;
let decisions = uzu::session::decision::DecisionSession::from_chat_instance(&loaded)?;
let output = decisions.decide(input).await?;
```

The sessions share model weights, not conversation history or decoder state.
Each question gets a fresh decoder state. The chat session retains its cache.
Chat sessions created from the same `ChatInstance` retain shared access during inference.
A decision waits for active chat inference and takes exclusive access until its scoring finishes.
Chat access is released before registered tools run, so a tool can request a decision.
A decision worker retains exclusive access even if its caller stops waiting.
Queued or subsequent questions are skipped when their response receiver is dropped.

The native scorer reads the final prompt logits, without sampling or a generation loop.
Each question processes its full prompt independently. This is not shared-prefix batching.

## Run the server

```sh
cargo +nightly run --locked -p cli -- server --model /absolute/path/to/model --host 127.0.0.1 --port 8000 --prefix-cache
```

Supported chat models expose the existing `/systemone` and `/v1/systemone` handlers beside chat completions.
The existing D3 server and its aliases remain unchanged.

```json
{"state":"I was charged twice. Please refund me today.","questions":{"category":{"type":"choice","instructions":"Classify the ticket.","criteria":{"billing":"Payments and refunds","technical":"Technical support","other":null}},"urgent":{"type":"noul","instructions":"Does the customer ask for action today?"},"priority":{"type":"score","instructions":"Rate urgency.","criteria":["Routine","This week","Today"]}}}
```

## Supported language-model profile

The initial profile requires native token scoring and a Hanashi encoding.
Reasoning must be absent or explicitly disableable. Harmony is not supported by this adapter.
Each answer label must extend the exact prompt by one distinct token. Unsupported boundaries fail instead of generating an answer.
Choice supports 1 to 26 distinct nonempty option names. Score supports 2 to 10 ordered levels.
Each prompt must fit the model and configured context limits. Prepared prompts have an aggregate limit of 65,536 tokens.

The existing upstream API accepts string state, string instructions, and string score levels.
This port does not extend those types to structured JSON or nullable score levels.
It does not add image, box, compression, model discovery, or transport timeout behavior.

The returned probabilities are conditional on the allowed labels at temperature 1.
`noul` is the probability of yes. `score` is the expected zero-based level index.
The existing upstream response computes confidence from normalized entropy.
Confidence is not a calibrated probability of correctness. The scorer retains label mass internally but does not add a response field.
Output token usage is zero. Input usage counts each prepared question prompt.
Session statistics retain upstream prefill duration and optional energy readings.
The LLM prefill duration covers native command preparation through GPU completion, before logit readback.

## Verification

```sh
sh scripts/verify-systemone.sh --model /absolute/path/to/LFM2.5-350M-L --backend metal --output /new/evidence/directory
```

The command validates the pinned fixture, runs native final-logit parity, application tests, real HTTP, and SDK 0.7.0.
It compares a cached chat continuation with and without interleaved decisions, and repeats decisions in A-B-A order.
The SDK test uses the existing string-only contract, the actual served model ID, and `compression: false`.
This is not full TypeSafe protocol parity or a task-quality benchmark.
CPU, WebGPU, Harmony, KV-shared trailing layers, and calibration need their own evidence before support is claimed.
No model weights, SDK dependencies, or acceptance logs belong in Git.
