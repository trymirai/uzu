# Kernel Benchmarks

Criterion microbenchmarks for Metal kernels, runnable on macOS (native) and on a
physical iPhone (built + installed with `cargo-dinghy`). macOS results are written
under `target/criterion/<label>/`; on-device results stay on the phone (see below).

## Prerequisites

- Toolchain: the repo pins nightly via `rust-toolchain.toml` (the bench target uses
  `#![feature(custom_test_frameworks)]`), so no manual toolchain step is needed.
- For iPhone runs:
    - Apple target: `rustup target add aarch64-apple-ios`.
    - Official `cargo-dinghy` (from crates.io): `cargo install cargo-dinghy`.
    - A physical device connected via **USB**, **unlocked**, with **Developer Mode**
      enabled (Settings → Privacy & Security → Developer Mode).
    - Start an RSD tunnel in a separate terminal (needed for dinghy launch):
      `sudo pymobiledevice3 remote tunneld`
    - Identifiers:
        - `cargo-dinghy` matches the **hardware ECID** — copy it from
          `cargo dinghy all-devices` (e.g. `Eugene's iPhone (00008150-… aarch64 27.0)`).
        - `xcrun devicectl` uses the **CoreDevice UUID** — copy the Identifier column
          from `xcrun devicectl list devices`.

## Bench targets

- **`--bench main`** (`benches/main.rs`) — the target that runs on device. Groups:
  `Model loading` (needs the test model) and `Metal/Kernel/A8W8`.
- **`--lib`** — the other Metal-kernel microbenches are `#[uzu_bench]` cases woven
  into the library test harness. List them with
  `cargo bench -p backend-uzu --lib -- --list` and run them with `--lib`. They run on
  macOS (they are not part of the `--bench main` target used for the phone).

## Running on macOS

Use an **absolute** `CRITERION_HOME` so it does not resolve relative to the package dir.

```bash
# A8W8 self-skips on GPUs without MXU (i.e. M1–M4):
CRITERION_HOME="$PWD/target/criterion/m2_max" \
  cargo bench -p backend-uzu --bench main -- "Metal/Kernel/A8W8"

# a --lib kernel microbench (use `--lib -- --list` to discover filters):
CRITERION_HOME="$PWD/target/criterion/m2_max" \
  cargo bench -p backend-uzu --lib -- "<filter>"
```

Set `UZU_CAPTURE_BENCH=1` to capture the first matching command buffer as a Metal
`.gputrace`. `UZU_CAPTURE_BENCH_FILTER` narrows by benchmark-path substring;
`UZU_CAPTURE_BENCH_DIR` sets the output directory (defaults to the current directory).

## Running on iPhone

With `tunneld` running and the phone unlocked:

```bash
ECID=<from `cargo dinghy all-devices`>        # e.g. 00008150-001965E22282401C

cargo dinghy -d "$ECID" bench -p backend-uzu --bench main -- "Metal/Kernel/A8W8"
```

If dinghy’s launch step fails after install, launch the installed app with
`xcrun devicectl device process launch --console --terminate-existing --device <UUID> org.zoy.kali.Dinghy …`.

## RHT A8W8 vs RHT ABF16W8 (`Metal/Kernel/A8W8`)

[`benches/kernels/a8w8_bench.rs`](kernels/a8w8_bench.rs) is the **primary KPI**: full RHT-enabled
linear on both sides, same W8 weights, GPU timestamps.

| Function id | Timed work |
|-------------|------------|
| `linear_a8w8` | Dyn act prepare (`RHT + quant`) + int8×W8 MXU GEMM |
| `linear_abf16w8` | Input RHT + BF16×W8 MXU GEMM (scale-symmetric dequant B) |

Activation quantization is **runtime-only** on the A8W8 arm. Weights, scales, and RHT
signs are preloaded. Both arms share the same shapes and RHT factors.

Correctness is covered only by unit tests (`a8w8_*`), not by this bench. On non-MXU
GPUs the group self-skips.

## Viewing reports (macOS)

```bash
open target/criterion/report/index.html          # all labels
open target/criterion/<label>/report/index.html  # a single label
```

## Cold GEMV

GEMV-class benches cycle through enough quant-buffer copies to cover a 256 MiB weight
working set before reusing one, so kernels are not ranked on cache-warm weights; pools
allocate lazily, so criterion filters skip their cost.
