# D3-Flash verification on Uzu 0.6.8

On 2026-10-15, the native D3 adapter at code commit `bf8a3b43e1ef310cf08d4270d40bd9503437faff` was built from Uzu 0.6.8 (`16afb3dd4c62f6992748afc8e25da42c299d1389`) and checked on Apple M6 / 32 GiB with Metal. The [measurement JSON](d3_flash_apple_m6_2026-10-15.json) preserves unrounded metrics and artifact identities.

## Same-weight engine regression

The retained INT4/group32/RHT32 D3-Flash artifact completed the same **1,356 frozen choice requests with zero errors** on the updated engine. The baseline is the prior local D3 adapter based on `432da1b7`. Scores use the pinned official scorer; the table shows metric values multiplied by 100.

| Benchmark | Requests | Metric | Previous engine | Uzu 0.6.8 adapter |
| --- | ---: | --- | ---: | ---: |
| When2Call MCQ | 200 | accuracy | 88.50 | 88.50 |
| BANKING77 | 154 | macro-F1 | 88.23 | 88.23 |
| CLINC150+OOS | 302 | macro-F1 | 97.22 | 97.22 |
| iSarcasmEval | 400 | sarcastic positive F1 | 52.75 | 52.75 |
| ANLI | 100 | macro-F1 | 60.05 | 60.05 |
| WinoGrande | 100 | accuracy | 86.00 | 86.00 |
| HellaSwag | 100 | accuracy | 94.00 | 94.00 |

| Suite | Same top label | Maximum option-probability difference | Previous measured p50/p95 (ms) | Updated measured p50/p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| roles | 1056/1056 | 0 | 655.8/1139.9 | 637.6/1104.3 |
| regression | 300/300 | 0 | 130.4/225.6 | 125.7/219.4 |

These are partial Decision Index subsets, not a full index/leaderboard result. The scorer is pinned to `9eb2dbe2a358004c8782c66e40a83ac07b953fec`. BANKING77 and CLINC use two rows per class; gold labels are used only for stratification. When2Call has no direct-answer A positives (B/C/D counts 76/65/59), and iSarcasm uses positive-class F1 (51 positives, 349 negatives), not majority-class accuracy. All cases are single-choice questions; noul/score/multi-question and Korean production accuracy are outside this quality sample.

Every request uses the same tokenization/prompt hash, code order, temperature and unchanged model weights. Timing is `python_prepare_ms + engine_call_wall_ms + python_postprocess_ms`: input preparation, synchronized native request/IPC and typed answer construction. Probability normalization and trace bookkeeping between these regions, scorer checks, result serialization, startup and warmup are excluded. These are instrumented component sums, not complete caller-observed latency. The single runs do not establish a statistical speedup or throughput result.

## Boundary and code checks

- Both old and final new binaries passed eight native protocol requests each: short input, 1024 and 1025 tokens, repeat-after-long input, empty/out-of-vocabulary/unknown-field failures, and repeat-after-errors. At the three paired valid lengths, all 255 logits and 4096 last-position hidden values were exactly equal between builds. Repeat outputs were exact and the process continued after invalid requests. Synthetic token IDs test transport/prefill/isolation, not semantic accuracy.
- CPU-only D3 regressions: **16 passed**, none ignored. The same 16 also passed with `cpu,metal`; this is not 32 unique cases. Tests cover configuration/input rejection and a small two-layer BF16 Transformer, with an independent noncausal scalar oracle and causal control, 1024/1025 prefill, fresh state and FP32 readout precision. The synthetic fixture does not bypass or relax production D3-Flash geometry validation.
- Two existing attention-state regressions passed. Workspace formatting, release Clippy for the library/example (`cpu,metal`, `-D warnings`), the Metal-only release example build and compilation of the documented Rust API example passed.
- Metal-only lib tests encounter existing unconditional CPU references in the upstream test helpers. CPU-only and combined-feature lib tests are the passing configurations used here; a Metal-only lib-test pass is not claimed.
- Local application adapter regressions passed separately (76 Python tests plus 21 subtests); those tests and application-specific fixtures are outside this engine contribution.

## Artifact provenance and limits

- Source checkpoint: `vllm-sr/d3-flash` at `521525effa79469d899a570726547dd7f00da3af`.
- Tested native binary SHA-256: `5a3dfc06fd3d2022049ab47307c7450f45d85e374c4cb8c25770f3cb61de8ef3`.
- Retained model weights SHA-256: `03abceda58f76b7c7097f3925314e4ecfe59f0bb37d6c6ae299fd012500a8d8d`; final checks confirmed weights, configuration, readout and conversion manifest remained unchanged.
- The artifact was created with the modified Lalamo checkout based on `831accb8`: INT4/group32/RHT32 text/embedding weights, BF16 scales and an FP32 readout, without Fisher/QAD. This update evaluates the retained artifact; it does not demonstrate a fresh full-model conversion using the current Lalamo PR branch.
- This same-weight engine comparison is separate from the [BF16/MLX/Mirai quantization measurements](https://github.com/backryun/lalamo/blob/d3-support/examples/d3_flash/reports/apple-m6-threeway-2026-10-15.md), which used the previous engine. Matching this regression sample does not establish universal equivalence or quantization parity with original BF16.
- The repository publishes aggregates and hashes. Model weights, benchmark payloads, application details and full raw response archives are retained locally and are not distributed here.
