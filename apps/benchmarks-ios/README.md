# Benchmarks for iOS

Runs the inference benchmarks of [scripts/benchmarks](../../scripts/benchmarks) on an iPhone or iPad. A Mac controls the app through [platform-ios](../../scripts/benchmarks/platform-ios), which behaves like any other engine process, so the benchmark harness can target the device. On the device, the Log tab follows the current run, the Models tab lists the cached models and their sizes, and the bar at the bottom shows the server state.

Engines, the same as in `scripts/benchmarks`, each started through its command-line entry inside the app:

- `llamacpp`: `bench-llamacpp`, built with CMake as `BenchmarksLlamacpp.framework`.
- `mldrift`: `engine-mldrift`, built with Bazel as `BenchmarksMldrift.framework`.
- `mlxserve`: `engine-mlxserve`, built with Zig and linked with MLX and mlx-c into `BenchmarksMlxserve.framework`. [patches/zig-ios-null-file.patch](patches/zig-ios-null-file.patch) backports a Zig 0.17.0 fix for iOS.
- `mlx`, `mtplx`, `omlx`, `splash`: the Python adapters on CPython 3.13, each in its own environment installed from its `uv.lock`. Splash's native engine runs in the app through `engine-splash/patches/in-process.patch`; its only models, Qwen3.8-27B and Qwen3.6-35B-A3B, do not fit a 12 GB iPhone.
- `uzu`: `engine-uzu`, built as `BenchmarksUzu.framework`.

## Set up

On the Mac:

- Xcode 27 or newer, signed in with an account of the Mirai team and selected for the command line: `sudo xcode-select -s /Applications/Xcode.app`. Then its Metal Toolchain component: `xcodebuild -downloadComponent MetalToolchain`.
- [Tuist](https://tuist.dev), [uv](https://docs.astral.sh/uv/), CMake, bazelisk and the Hugging Face CLI: `brew install tuist uv cmake bazelisk hf`. Also install Zig 0.17, pkg-config and libwebp for mlx-serve.
- rustup from [rustup.rs](https://rustup.rs), in its default `~/.cargo`; the build finds tools only in `~/.cargo/bin`, `~/.local/bin` and `/opt/homebrew/bin`. Python packages with Rust code build with the stable toolchain: `rustup toolchain install stable && rustup target add aarch64-apple-ios --toolchain stable`.

On the iPhone or iPad:

- iOS 26.4 or newer, Developer Mode on (Settings, Privacy & Security), and paired with the Mac: connect it once with a cable and trust the Mac.
- The first run of the `Benchmarks` scheme from Xcode registers the device with the Mirai team and installs the app. After that, `bench-ios` launches the installed app.

## Build

```bash
cd apps/benchmarks-ios
tuist generate
```

Then run the `Benchmarks` scheme on a device; simulators are not supported.

`Build Python Packages` runs [bootstrap.sh](bootstrap.sh), which builds the engine frameworks, cross-compiles the native Python packages for iOS, MLX with [patches/mlx.patch](patches/mlx.patch), and installs each Python engine's `uv.lock` into `deps/envs/<engine>`, without `hf-xet`, whose downloads stall on iOS. The first build takes several minutes. `Install Python` embeds Python, the engine frameworks and the environments into the app.

## Run from a Mac

`bench-ios` benchmarks on one device, chosen by its UDID. Pass it with `--device UDID`, or let `bench-ios` ask: the first command without `--device` lists the paired iPhones and iPads and asks which one to use. The choice is saved in `~/.cache/bench-ios/device` and used until another is chosen with `bench-ios device` or `--device`. Without a terminal, as in CI, pass `--device`.

The app listens on port 7171 while it is open and keeps the screen awake. `bench-ios` reaches it over USB, over Wi-Fi to a paired device, through the CoreDevice tunnel, or at an address given with `--host`. Unless `--host` is given, commands that talk to the app bring it to the foreground first, since iOS suspends it in the background, and `run` and `download` start a fresh app process, as a new engine process starts on macOS. The positional model JSON file is sent to the app and passed to the engine there. Keep the device unlocked.

The model file contains `model` as each engine expects it: MLX, mlx-serve, MTPLX and oMLX a Hugging Face repository; llama.cpp and Splash a GGUF repository with a quantization, such as `unsloth/Qwen3.5-0.8B-GGUF:Q4_K_S`; ML Drift a pushed directory; uzu a registry identifier such as `trymirai/LFM2.5-230M-M`.

Before `run` and `download`, `bench-ios` skips a model that does not fit, with the rule of the macOS benchmark runner: the model's Hugging Face files, or for llama.cpp any GGUF file of the repository, must take at most 75% of the device's memory. For Splash it counts the chosen GGUF variant and the larger of Splash's draft models, since the draft follows the target's configuration. Pushed models and other local paths are not checked.

```bash
cd scripts/benchmarks
uv run --project platform-ios bench-ios device
uv run --project platform-ios bench-ios push trymirai/LFM2.5-230M-M
printf '%s\n' '{"model":"LiquidAI/LFM2.5-230M-MLX-4bit","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > mlx-model.json
uv run --project platform-ios bench-ios download -e mlx mlx-model.json
uv run --project platform-ios bench-ios models
printf '%s\n' '{"model":"trymirai/LFM2.5-230M-M","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > uzu-model.json
uv run --project platform-ios bench-ios run -e uzu uzu-model.json <<'EOF'
{"messages": [{"role": "user", "content": "Tell me about London"}], "max_output_tokens": 256}
EOF
uv run pytest --engine ios --ios-engine uzu
```

`push` copies a model that is already on this Mac, so the device does not download it: a uzu registry identifier from `~/.cache/mirai`, after the model has run on this Mac; a Hugging Face repository from the Hugging Face cache, after `hf download` or a run on this Mac; or a directory, which lands in the app's Documents folder and is referenced as `models/<name>`. `delete` takes a path that `models` reports.

ML Drift's weight converter does not run on the device, so `mldrift` runs only pushed models. Convert one with `engine-mldrift/run.sh` on this Mac, then push the converted weights together with the checkpoint's other files:

```bash
printf '%s\n' '{"model":"Qwen/Qwen3-0.6B","specdec":false,"sampling":{"greedy":true},"reasoning_efforts":{"disabled":"default","default":"default","low":"default","medium":"default","high":"default","xhigh":"default"}}' > mldrift-model.json
./engine-mldrift/run.sh mldrift-model.json </dev/null
staging="$(mktemp -d)/Qwen3-0.6B-mldrift"
cp -R "$(ls -td "${XDG_CACHE_HOME:-$HOME/.cache}"/uzu-benchmarks/mldrift/*/q4_0/* | head -1)" "$staging"
rsync -aL --exclude '*.safetensors' "$(hf download --quiet Qwen/Qwen3-0.6B)/" "$staging/"
uv run --project platform-ios bench-ios push "$staging"
uv run pytest --engine ios --ios-engine mldrift --ios-model models/Qwen3-0.6B-mldrift
```
