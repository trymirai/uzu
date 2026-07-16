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
    - Two different identifiers are used:
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
# a --bench main group (A8W8 self-skips on GPUs without MXU, i.e. M1–M4):
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

`cargo-dinghy` builds and installs the bench app via `xcrun devicectl`, but its own
launch step shells out to `pymobiledevice3` (RSD), which on iOS 17 needs a root tunnel
and is sensitive to the installed `pymobiledevice3` CLI version. The reliable path is
to let dinghy build + install, then launch the installed app with `devicectl` directly
— no `pymobiledevice3`, no tunnel, no `sudo`. Run one group at a time to avoid the iOS
watchdog killing the app.

```bash
ECID=<from `cargo dinghy all-devices`>        # e.g. 00008150-001965E22282401C
UUID=<from `xcrun devicectl list devices`>    # e.g. DD4B4CF2-F102-5B4F-A616-5D824E55B6FF

# 1) Build (release, aarch64-apple-ios) + install the app (bundle id org.zoy.kali.Dinghy)
#    via devicectl. dinghy then exits non-zero at its pymobiledevice3 launch step —
#    that is expected; the app is already installed by that point.
cargo dinghy -d "$ECID" bench -p backend-uzu --bench main -- "Metal/Kernel/A8W8" || true

# 2) Launch the installed bench directly; stdout/stderr stream over --console.
xcrun devicectl device process launch --console --terminate-existing \
  --device "$UUID" \
  -e '{"CRITERION_HOME":"target/criterion/a19"}' \
  org.zoy.kali.Dinghy  Metal/Kernel/A8W8 --save-baseline a8w8_a19 --bench
```

Criterion output stays on the phone under `Documents/target/criterion/a19`; pull it
with `xcrun devicectl device copy from` if you want the host-side HTML report.

## A8W8 int8 GEMM (A19 only)

`Metal/Kernel/A8W8` (`benches/kernels/a8w8_bench.rs`) self-skips on GPUs without MXU.
On M5/A19 it first runs a correctness check — the int8×int8→int32 GEMM against a
plain-Rust A8W8 reference; a mismatch panics — then times int8 vs bf16. The int8
timing includes the fused RHT + activation-preparation dispatch. Because the correctness
asserts stream over `--console` before any timing, a panic there means the int8 result
diverged from the reference.

## Viewing reports (macOS)

```bash
open target/criterion/report/index.html          # all labels
open target/criterion/<label>/report/index.html  # a single label
```

## Cold GEMV

GEMV-class benches cycle through enough quant-buffer copies to cover a 256 MiB weight
working set before reusing one, so kernels are not ranked on cache-warm weights; pools
allocate lazily, so criterion filters skip their cost.
