# Benchmarks for iOS

Runs the Python benchmark engines from [scripts/benchmarks](../../scripts/benchmarks) on an iPhone or iPad. The app embeds CPython 3.13 and the `engine-mlx` adapter, with every native dependency cross-compiled for `arm64-apple-ios`. A request goes through the same `run_loop` as `bench-mlx` on macOS.

## Build

```bash
tuist generate
```

Then run the `Benchmarks` scheme on a device. Simulators are not supported. Signing uses the Mirai team, so Xcode needs an account of that team.

The target has two script phases:

- `Build Python Packages` runs [bootstrap.sh](bootstrap.sh) in a clean environment. It downloads the BeeWare Python 3.13 iOS support package into `deps/`, builds iOS wheels for the native packages in a cross-compiling virtual environment and installs everything pinned by `engine-mlx/uv.lock` into `deps/app_packages`. MLX is built from source with [patches/mlx.patch](patches/mlx.patch), which compiles its Metal kernels, including the Neural Accelerator ones, for the iOS SDK. `hf-xet` is left out because its downloads stall on iOS; Hugging Face downloads use plain HTTP instead. The first build takes several minutes; later builds reuse the wheels and spend a few seconds here.
- `Install Python` embeds `Python.framework`, copies the standard library and `deps/app_packages` into the app and turns every extension module into a framework, as iOS requires.

## Run

Enter a Hugging Face repository or a local path and one JSON request per line in the `BenchRequest` format of the macOS engines. Models download to `Library/Caches` in the app container. Local models can be copied into the app's Documents folder with Finder and referenced by their path relative to it.

From a Mac, the same run can be launched with arguments. The engine writes JSON responses to stdout and logs to stderr, as on macOS, and the app exits when it is done:

```bash
xcrun devicectl device process launch --console --terminate-existing --device <device> -- com.mirai.tech.benchmarks \
    -model mlx-community/Qwen3.5-0.8B-4bit \
    -request '{"prompt_text": "Tell me about London", "max_tokens": 256, "num_runs": 3}' \
    -run YES
```
