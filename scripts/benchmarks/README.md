# Inference benchmarks

Benchmark adapters for inference engines. Each runner loads one model, reads requests from standard input, and writes generated text, timing, throughput, energy, and process memory measurements as JSON Lines. A process can handle multiple requests without reloading its model.

Interaction with every engine follows the same loop: after launch, it loads its model and writes a `ReadyResponse` to stdout, then reads requests from stdin and answers each with one line on stdout. Every message carries a `type` field naming it. A `BenchRequest` runs inference once and is answered with a `BenchResponse`. A `CooldownRequest` waits until both CPU and GPU average temperatures are strictly below its `temperature` (default 60°C) and is answered with a `CooldownResponse`. With `only_if_capped`, it waits only if macOS thermally capped the GPU since the previous such request, so a Mac that never throttles never waits; requests are never cooled down implicitly. Send the task again for repeated measurements, and send it first to warm up. Requests share no caches: every request prefills its whole prompt.

Each engine includes adapter source files that collect the same set of benchmark metrics. Memory usage is measured using the shared API in [memory_counters.h](common-cpp/src/memory_counters.h), and energy with [energy_counters.h](common-cpp/src/energy_counters.h).

Every runner takes a positional `model.json` with engine-specific settings. Requests are JSON Lines on stdin; diagnostics go to stderr. For example, from this directory:

```bash
printf '%s\n' '{"model":"unsloth/Qwen3.5-0.8B-GGUF:Q4_K_S","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
./engine-llamacpp/run.sh model.json <<'EOF'
{"type": "BenchRequest", "messages": [{"role": "user", "content": "Tell me about London"}], "max_output_tokens": 128, "reasoning_effort": "default"}
EOF
```

Cooldown polls the temperatures every five seconds through [apple_temp_sensors.h](common-cpp/src/apple_temp_sensors.h), which also reads from IOReport how long the thermal manager held the GPU below the performance state it would otherwise run at. If readings are unavailable or cooldown is interrupted, the runner exits with an error.

## Model files and catalogs

Every model file has `model`, `specdec`, `sampling` and `reasoning_efforts`. Engines add their own fields:

| Engine | Further fields in `model.json` |
| --- | --- |
| Uzu | none |
| MLX, Splash, Magnitude | optional `revision` |
| MTPLX | optional `revision`, optional `speculative_depth` |
| llama.cpp | optional `draft_model`, optional `speculative_depth` |
| oMLX, mlx-serve | optional `revision`, optional `draft_model`, optional `draft_revision`, optional `speculative_depth` |
| TensorFold | optional `revision`, optional `draft_model`, optional `draft_revision` |
| ML Drift | optional `revision`, optional `precision` (`q4_0` by default, or `f16`) |

`specdec` says whether the engine decodes speculatively with this model file, however the engine arranges that: a draft model, the model's own heads, a depth. An engine that would do otherwise exits with an error, at startup or, for the Uzu server, on the first reply that shows it.

`model` and `draft_model` accept the same identifiers and local paths described below. Relative paths are relative to the runner's working directory. Uzu and Splash server runners use their native engine's model files.

`revision` and `draft_revision` pin the Hugging Face commit of `model` and `draft_model`; without them, downloads follow `main`. Catalogs record the commit `main` named when they were refreshed. Three downloads can't be pinned: Uzu resolves models through its registry, llama.cpp's downloader always follows `main`, and Splash takes its draft from the draft repository's default branch.

Each `engine-*` directory has `models.json`, mapping Uzu repository identifiers to opaque, engine-specific configurations that can be saved directly as `model.json`. It contains only models with a selected Uzu counterpart. Run `./update_models.py` to refresh all catalogs, Uzu first, or an engine directory's `./update_models.py` to refresh just that engine. These scripts need Python 3 and download metadata only; they do not build engines or fetch weights.

| Engine | Catalog source |
| --- | --- |
| Uzu | Mirai registry, using this checkout's toolchain version |
| MLX, oMLX | Hugging Face repositories from mlx-community and LiquidAI, excluding known non-text pipelines |
| llama.cpp | GGUF variants from Unsloth and LiquidAI on Hugging Face |
| Splash | Upstream suggested models TSV |
| MTPLX | Upstream official model catalog |
| mlx-serve | Upstream CLI's MLX aliases, excluding embedding models |
| Magnitude | Upstream supported models and quantization variants |
| TensorFold | Hugging Face MLX repositories and upstream suggested Metal checkpoints |
| ML Drift | Upstream sample documentation plus the benchmark's Qwen3 14B preset |

Upstream source catalogs are read at the pinned engine versions.

`model_mapping.json` is maintained manually: Uzu repository identifier → upstream catalog ID, or an object with that ID in `model` and a draft's in `draft_model`. The updater resolves these entries into `models.json`, adding `specdec` and the engine's own speculation settings as the engine's updater derives them from the model and its draft; unmapped upstream models are omitted. Updates fail before replacing the catalog if any mapped target or draft disappears. Uzu's catalog maps its own identifiers directly to configurations.

Mappings preserve the checkpoint and quantization choices used in the [published benchmarks](https://trymirai.com/models) where a baseline exists. Other entries select the same base model, with 4-bit variants for Uzu M and 8-bit for L where available. Splash, mlx-serve and TensorFold's L mappings also use their selected 4-bit checkpoints. These mappings do not imply equivalent quantizations. ML Drift has no models matching the current Uzu catalog.

For example, select a catalog entry with `jq`:

```bash
jq '."trymirai/Qwen3.5-0.8B-M"' engine-llamacpp/models.json > model.json
./engine-llamacpp/run.sh model.json
```

Sampling belongs in `model.json` and stays fixed for the process. Use `"sampling": {"greedy": true}` for greedy decoding, or a stochastic object with all six fields: `temp`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, and `suffix_repetition_length`. Set unused controls to explicit `null`; this disables them and does not select engine defaults. Repetition penalties require an explicit suffix window. The variants are exclusive: unknown fields, missing stochastic fields, and mixing `greedy` with stochastic fields are errors. `temp: 0` is an error; use the greedy variant.

Generated entries copy sampling from the exact Uzu revision's `generation_config`, including model-specific repetition controls. Refresh `engine-uzu/models.json` first, then update the other engines. No engine-specific sampling overrides are applied: adapters reject unsupported controls. Uzu, llama.cpp and MLX support the repetition penalty and suffix window; the other adapters currently reject them. ML Drift accepts only greedy sampling. Requests on stdin carry no sampling controls, only an optional `seed`.

`reasoning_effort` is required in a request: `disabled`, `default`, `low`, `medium`, `high` or `xhigh`, applied through the engine's native reasoning controls. `default` leaves them unset; an effort the engine or model can't apply is an error. ML Drift accepts only `default`.

`reasoning_efforts` in a model file says, for each of those six efforts a caller may want, which effort to request from this model instead. Engines don't apply it: whoever builds the requests does, so that engines only get efforts the model supports. The Uzu catalog derives it from the model's Hanashi encoding, and the other catalogs copy it as they copy sampling:

- An effort the encoding renders maps to itself.
- `default` maps to the effort that renders the same as the default does: `xhigh` for Qwen3.8, `high` for Muse Glimmer. Where none does, it stays `default`.
- A level the model lacks maps to the next one up, else to the highest it has: `high` is `xhigh` for Qwen3.8. A model with no levels gets `default` for all of them, so Qwen3.5 and Qwen3.6 reason as they do by default.
- `disabled` on a model that can't stop reasoning maps to its lowest level (`low` for Muse Glimmer), and on a model that never reasons to `default`.

`seed` in a request is an integer from 0 to 2³² − 1 that keys the request's sampling, so the same seed reproduces a stochastic draw; without it the engine seeds on its own. An engine that can't apply the seed rejects the request: ML Drift any seed, llama.cpp 2³² − 1 (its random-seed value), and MTPLX seeds of 2³¹ and above, which its speculative path truncates.

`finish_reason` is `stop` for a stop token and `length` for the output limit. `input_tokens_count` counts the rendered prompt. `energy_gpu` is the whole GPU's energy in joules over `duration`, read without privileges from IOReport; its CPU, ANE and DRAM rails don't update on current macOS.

`memory_used` is the engine's own account of the memory a request took, in bytes, since no single counter is fair to every engine: reusable caches inflate the process counters of some, and llama.cpp's mapped weights are missing from them.

| Engine | `memory_used` |
| --- | --- |
| Uzu | Peak Metal allocation since the process started |
| MLX, MTPLX, oMLX, mlx-serve | MLX's peak active memory during the request, which leaves out its buffer cache |
| llama.cpp | Size of the target and draft models plus the size of their contexts' state after the request |
| Splash | What it holds in Metal after the request, with the KV pages the request's tokens fill in place of its whole KV pool and cached states, which are a prefix cache of earlier requests |
| TensorFold | Its allocator's peak since the server started |
| Magnitude | The bytes its own ledger says the loaded model holds in host and GPU memory after the request: weights, context, compute buffers and optional heads |
| ML Drift, Uzu server | The kernel's count of the process's graphics memory, as neither reports its own |

`memory_phys_footprint`, `memory_resident` and `memory_graphics_total` are the process counters of [memory_counters.h](common-cpp/src/memory_counters.h), read when graphics memory peaked during the request.

`tokens_per_forward_pass` is the average number of output tokens per speculative decoding round. It is `null` without speculative decoding, and for Magnitude, whose API does not expose it.

When `max_output_tokens` is omitted, `null`, or `0`, decoding continues until an EOS token. `messages` is required; raw prompt input is not supported.

## Engines

### [llama.cpp](https://github.com/ggml-org/llama.cpp)

```bash
printf '%s\n' '{"model":"unsloth/Qwen3.5-0.8B-GGUF:Q4_K_S","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
./engine-llamacpp/run.sh model.json
```

The catalog enables MTP for models from Unsloth's `-MTP-` repositories (Qwen3.5-9B and Qwen3.6-27B) and DFlash for models mapped with a draft (Muse Glimmer), at the legacy benchmark depth of 2. Without `speculative_depth`, or with `0`, decoding is ordinary; a `draft_model` requires a positive depth.

DFlash drafts can use an exact repository filename:

```bash
printf '%s\n' '{"model":"unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL","draft_model":"unsloth/Muse-Glimmer-30B-GGUF:dflash-kquant.gguf","speculative_depth":2,"specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
./engine-llamacpp/run.sh model.json
```

### [ML Drift](https://github.com/google-ai-edge/ml-drift)

```bash
printf '%s\n' '{"model":"Qwen/Qwen3-0.6B","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
./engine-mldrift/run.sh model.json
```

Requires Python 3, `bazelisk` (`brew install bazelisk`) and, for Hugging Face ids, the `hf` CLI. The Metal LLM sample runner supports the `gemma3:270m`, `gemma3:1b`, `gemma4:12b`, `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:8b`, and `qwen3:14b` presets (the last added by the benchmark patch), selected from the checkpoint's `config.json`. The first run of a checkpoint converts its float weights with upstream `extract_weights_hf` through Bazel: to Q4_0 (block size 32) by default, or unquantized with `"precision": "f16"` in the model file.

Converted weights are cached under `${XDG_CACHE_HOME:-$HOME/.cache}/uzu-benchmarks/mldrift/`.

Prompts are tokenized by ML Drift's own tokenizer. Chats use its turn markers for user, assistant and system turns (Gemma 3 has no system turn); tools, tool calls and reasoning content are rejected. Decoding requires `"sampling": {"greedy": true}` in the model file. The runner is compiled for a prompt length and context size, and rebuilt before inference when a request changes either.

### [MLX](https://github.com/ml-explore/mlx-lm)
```bash
printf '%s\n' '{"model":"mlx-community/Qwen3.5-0.8B-4bit","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project engine-mlx bench-mlx model.json
```

### [mlx-serve](https://github.com/ddalcu/mlx-serve)

Requires Zig 0.17, pkg-config and libwebp.

```bash
printf '%s\n' '{"model":"mlx-community/Qwen3.5-0.8B-4bit","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
./engine-mlxserve/run.sh model.json
```

Supported native MTP heads are detected and loaded automatically from the target checkpoint, including sidecars. A detected head that fails to load or bind produces an error. Models without a supported head use ordinary autoregressive decoding.

Pass a compatible DFlash or Gemma assistant checkpoint as `draft_model` in the model file to use external speculative decoding. Both paths accept local directories or Hugging Face repositories; an explicit draft model takes precedence over MTP.

```bash
printf '%s\n' '{"model":"mlx-community/Qwen3.6-27B-4bit","draft_model":"z-lab/Qwen3.6-27B-DFlash","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
./engine-mlxserve/run.sh model.json
```

Omit `speculative_depth` in the model file for the decoder's default depth, set it to `0` for ordinary decoding, or supply a positive depth within the decoder's supported range. DFlash uses upstream's hardware-dependent default block size. Depth counts proposed tokens; a DFlash verification block contains one additional target token. Upstream's adaptive acceptance gates can reduce speculation during a request. Inference runs directly through Zig sources, without HTTP.

Each run starts with fresh generation caches and adaptive timing/acceptance history.

### [MTPLX](https://github.com/youssofal/mtplx)
```bash
printf '%s\n' '{"model":"Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project engine-mtplx bench-mtplx model.json
```

MTPLX uses the model's multi-token prediction (MTP) heads to draft tokens and verifies them with the target model: this is speculative decoding without a separate draft model. Omit `speculative_depth` in the model file to use the model pack's recommended depth (fallback: 3), set it to `0` for ordinary autoregressive decoding, or supply a positive depth to override it. Positive depths require matching MTP weights.

### [oMLX](https://github.com/jundot/omlx)
```bash
printf '%s\n' '{"model":"mlx-community/SmolLM-135M-Instruct-4bit","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project engine-omlx bench-omlx model.json
```

Native MTP is enabled automatically when the target checkpoint has embedded prediction heads supported by oMLX's text engine.

```bash
printf '%s\n' '{"model":"TheWirelessPhoenix/Qwen3.5-4B-oQ4e-mtp","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project engine-omlx bench-omlx model.json
```

DFlash is selected in the catalog for Qwen3.5-9B, Qwen3.6-27B, and Qwen3.8-27B through their matching `draft_model` checkpoints. `speculative_depth` in the model file overrides the drafter's or MTP heads' default depth; `0` disables speculation.

```bash
printf '%s\n' '{"model":"mlx-community/Qwen3.6-27B-4bit","draft_model":"z-lab/Qwen3.6-27B-DFlash","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project engine-omlx bench-omlx model.json
```

### [splash](https://github.com/incoai/splash)

```bash
printf '%s\n' '{"model":"unsloth/Qwen3.8-27B-GGUF:UD-IQ1_S","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project engine-splash bench-splash model.json
```

Splash drafts with DFlash at its fixed depth of 7.

Prepared models and runtime state are stored under `~/Library/Application Support/Splash/`.

### [uzu](https://github.com/trymirai/uzu)

```bash
printf '%s\n' '{"model":"trymirai/LFM2.5-230M-M","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
cargo run --release -p benchmarks-uzu -- model.json
```

A model that ships a speculator decodes with DFlash:
```bash
printf '%s\n' '{"model":"trymirai/Qwen3.5-9B-M","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
cargo run --release -p benchmarks-uzu -- model.json
```

## iOS

[platform-ios](platform-ios) runs these engines inside the [iOS app](../../apps/benchmarks-ios) on an iPhone or iPad:
```bash
printf '%s\n' '{"model":"unsloth/Qwen3.5-0.8B-GGUF:Q4_K_S","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
uv run --project platform-ios bench-ios run -e llamacpp model.json
```
Setup, device selection, model downloads and pushes are in the [app's README](../../apps/benchmarks-ios/README.md). `pytest engines` leaves it out because it needs a device; test it with `uv run pytest --engine ios --ios-engine ENGINE --ios-model MODEL`, adding `--ios-device UDID` or `--ios-host HOST` as needed.

## Server

The shared server adapter keeps a server alive across stdin requests. It translates each request to a non-streaming OpenAI completion and writes the same response schema as the native runners.

```bash
printf '%s\n' '{"model":"trymirai/Qwen3.5-4B-M","specdec":true,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > model.json
jq -c '{type: "BenchRequest"} + .' benchman/src/datasets/oneoff/ai-history-summary.json | uv run bench-server uzu model.json
```

Other subcommands are `splash`, `magnitude`, and `tensorfold`. Explicit speculative depth is not supported by these server adapters. Magnitude gets no draft, so `specdec` selects its `mtp` or `plain` method. TensorFold requires Zig 0.17. Closing stdin stops the server.

TensorFold's `draft_model` selects and downloads the drafter before starting its server. The catalog pairs Qwen3.8-27B with `incoai/Qwen3.8-27B-DFlash2`; configurations without a draft explicitly disable drafting.

## Fleet runs

`benchman` benchmarks this checkout on the Mac fleet through Prefect: it ships the source, builds each engine with Nix on the worker and runs the datasets against each model, one flow run per queue, engine and model.

```bash
uv run --project benchman benchman run -o macs-fleet/host-m5-max-001 -m Qwen3.8-27B-M LFM2.5-1.2B-Instruct-M
uv run --project benchman benchman run -b -o macs-fleet/host-m5-{pro,max}-001 -e uzu mlx -m LFM2.5-350M-M -d mt-bench gsm8k
uv run --project benchman benchman results --wait FLOW_RUN...
```

`--on`, `--model`, `--engine` and `--dataset` each take several values, separated by spaces or commas or given again. Models are keys of the engines' `models.json`, with `trymirai/` implied, and a dataset is `NAME` or one prompt of it as `NAME/KEY`. Each request's `reasoning_effort` is replaced through the model's `reasoning_efforts` before the flow run is submitted; datasets that have no efforts ask for `default`. Without `--dataset`, models that decode speculatively run `mirai-mixture-fast` and the others `oneoff/london-summary`. `--background` prints the flow runs' names and exits. `results` shows what flow runs have measured so far, as means over the measured runs and the peak `memory_used`, with one row per model and queue and a line in it for each engine; the best value among the engines is underlined. It exits with 0 once all completed, 1 if one failed, and 75 while some are unfinished.

## Updating engines

Run `./update_engines.py` to refresh engine releases, source hashes, and Python engine lockfiles, or an engine directory's `./update_engine.py` to update just that engine. Shared C++ dependencies have their own `common-cpp/update_dependencies.py`, also called by the top-level updater. All updaters accept `--dry-run` to show proposed pins without changing files. ML Drift follows `main`; other engines use stable releases. Uzu uses this checkout's source.

Review upstream compatibility and patches after an update, then build and test the affected engines. Model catalogs are refreshed separately with `./update_models.py` or each engine's `update_models.py`.

Regenerate checked-in Python bindings with `./common-py/generate_bindings.sh` after `uv sync`, and `./engine-splash/generate_bindings.sh` after `./engine-splash/bootstrap.sh`. Both use uv and Xcode.

## Tests
All engines integration tests
```bash
uv run pytest engines
```

Specific engine test
```bash
uv run pytest --engine uzu
```
