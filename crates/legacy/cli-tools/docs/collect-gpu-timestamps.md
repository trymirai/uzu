# collect-gpu-timestamps

Measures how long one forward pass takes as a function of two sizes: how many tokens are already in the KV cache (the prefix, P) and how many new tokens the pass processes (the suffix, S). The numbers feed the choice of the speculative-decoding tree budget: verifying a draft tree of N nodes is one forward pass of N tokens, so its cost is `t(N, P)`. On an M5 Max a tree-shaped pass cost the same as a flat pass of the same size within noise (N = 4 was the one exception, about 15% faster), so only flat passes are collected.

## Running

```
cargo tools collect-gpu-timestamps <model_dir> <output_dir> [--spans]
```

- `<model_dir>` is a local model directory (`config.json`, `model.safetensors`, …), for example one under `~/.cache/mirai/models/mirai/`.
- `--spans` also records every debug group and kernel inside each pass. It roughly doubles the run time and inflates the passes it stamps (see Overhead).
- Metal only. Keep the machine otherwise idle: builds and other GPU work show up in the data.
- `cargo tools` runs the dev profile (`opt-level = 1`, overflow checks), which is fine for a smoke test but inflates `wall_us` on the CPU side. For a dataset use the release build: `cargo run --release -p cli-tools -- collect-gpu-timestamps …`. GPU time is the same in both.
- Run time on an M5 Max with Qwen3.8 27B: about 2.5 h without `--spans`, about 5 h with. Small models take minutes.

## Protocol

- Grid: P ∈ {256, 512, 1023, 1024, 2048, 4096, 5120, 8192, 16384, 30720, 32768, 51200, 61440, 65536, 131072, 196608, max − 1024}, keeping only values with P + 1024 ≤ the model's context length; S ∈ {1 … 64} ∪ {72, 80, …, 1024}. The P points bracket the attention kernel switch points; S is dense where the matmul routing changes (around 2, 9, 16, 32 and 64).
- The KV cache holds noise. Before every pass the state is told it contains P tokens (`LanguageModelState::set_context_length`); nothing is prefilled. GPU time depends on the sizes, not on the values: an assumption, consistent with a real-data comparison that stayed inside the run-to-run noise.
- Token ids are `position mod vocab_size`, sampling is greedy, and each pass is a normal prefill: all S tokens through every layer, readout and sampling for the last row, accept of all S tokens.
- Rounds outermost: round 0 is a warm-up; rounds 1–5 each visit every (P, S) point once. A point's repeats are spread over the whole run, so slow drift is shared across points instead of baked into one P.
- With `--spans`, each point runs twice back-to-back per round: a clean pass (no stamps), then a stamped pass.

## Output

```
<output_dir>/
  run.json                  uzu version, device, OS, CPU, memory, model, grid, start time
  model/config.json         model architecture: layer types and sizes
  model/weights.json        safetensors header: every tensor's dtype and shape; __metadata__ holds the per-weight quantization specs
  model/speculator/…        speculator config and tree shapes, when the model has one
  totals/round_<r>.parquet  one row per pass
  spans/round_<r>.parquet   one row per span of each stamped pass (--spans only)
```

Each round is one Parquet file closed when the round ends, so finished rounds are readable while the run continues and a crash loses at most one round. `polars.read_parquet("totals/*.parquet")` loads a run.

### totals

| column | meaning |
|---|---|
| `prefix` | P |
| `tokens` | S |
| `repeat` | round; 0 is the warm-up |
| `stamped` | 1 if per-block timestamps were recorded during the pass |
| `elapsed_s` | seconds since the run started |
| `wall_us` | wall-clock time of the pass: CPU encoding, GPU execution and the wait |
| `gpu_us` | GPU execution time from the command buffer's commit feedback (start to end on the GPU), summed over the pass's command buffers; costs no stamps |

### spans

| column | meaning |
|---|---|
| `prefix`, `tokens`, `repeat` | as above |
| `command_buffer` | index of the command buffer within the pass; 0 is the forward pass |
| `name` | full path of the debug group or kernel, e.g. `decoder/transformer layer 0/delta net/DeltaNetPrefill` |
| `start_us`, `end_us` | microseconds from the first span of the same command buffer |

Spans nest: a layer's span contains its blocks' spans, so parents and children must not be added together.

## Using the data

- Cost model: the clean passes (`stamped = 0`), repeats 1–5, median or min per (P, S).
- Attribution: the stamped passes show which block takes the time. Take each block's fraction of its stamped pass and apply it to the clean total.
- Overhead: every stamp costs GPU time. Per point, `(gpu_us[stamped] − gpu_us[clean]) / (2 × span count)` is the cost per stamp. An earlier calibration on an M5 Max measured about 1.3 µs per stamp plus a stall of about 250 µs after every 80th stamp, so the overhead is not a clean constant per stamp.
- Noise (M5 Max, Qwen3.8 27B): about 6% within a point (p90 18%) and ±10% across hours. Round-robin repeats put that drift into `elapsed_s` instead of into P.

## Known effects (M5 Max, Qwen3.8 27B mirai m-4)

- S = 3, 4 and 5 are slower than S = 8 (about 80 ms against 42 ms at P = 512): the matmul route for 3–7 rows.
- Sizes that are not multiples of 8 cost more than the next multiple: 8-row tiles.
- Time grows with P mainly in attention; the P grid brackets its kernel switch points (1024, 5120, 16384, 32768, 51200, 61440, 131072).

## Not measured

- Draft time (speculator and Weaver).
- Accept cost as a function of the number of accepted tokens.
- Acceptance statistics: how many draft tokens survive depends on the text, not on the GPU.
- Real token values in the KV cache.
