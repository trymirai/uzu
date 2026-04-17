# Kernel Benchmarks

Criterion-based microbenchmarks for Metal kernels. Runs on macOS (native)
and on iPhone (via `cargo-dinghy`). Results from both are consolidated
under a single `target/criterion/<label>/` tree so you can compare
baselines side by side.

## Prerequisites

- Rust nightly: `rustup toolchain install nightly`
- For iOS runs:
  - Apple target: `rustup target add aarch64-apple-ios`
  - `cargo-dinghy` built from the [trymirai/dinghy](https://github.com/trymirai/dinghy/tree/mirai):

    ```bash
    cargo install \
      --git https://github.com/trymirai/dinghy \
      --branch mirai \
      cargo-dinghy
    ```

  - A physical device connected via USB with a device ID from
    `xcrun devicectl list devices`.

## Available benchmark groups

| Group id                       | Filter                         |
| ------------------------------ | ------------------------------ |
| `Metal/Kernel/Matmul/GEMM_MPP` | `Metal/Kernel/Matmul/GEMM_MPP` |
| `Metal/Kernel/RMSNorm`         | `Metal/Kernel/RMSNorm`         |
| `Metal/Kernel/Sampling/Argmax` | `Metal/Kernel/Sampling/Argmax` |

## Output layout

Every run writes into `target/criterion/<label>/…`, where `<label>` is a
free-form name you choose (e.g. `m2_max`, `a18`). The Criterion baseline
you saved lives at `target/criterion/<label>/<benchmark-path>/<baseline-name>/`.

## Running on macOS

From the repo root. Use an **absolute** `CRITERION_HOME` so it doesn't
resolve relative to the package dir:

```bash
CRITERION_HOME="$PWD/target/criterion/m2_max" cargo bench \
  -p uzu \
  --bench kernel -- "Metal/Kernel/RMSNorm" \
  --save-baseline rms_norm_baseline_m2_max
```

## Running on iPhone (via `cargo-dinghy`)

Run one benchmark group at a time to avoid the iOS watchdog killing the
app.

Key flags:

- `-e CRITERION_HOME=target/criterion/a18` — on-device env var. Path is
  relative to the app's cwd (`Documents/`), so this becomes
  `Documents/target/criterion/a18/` on device.
- `--copy-back "Documents/target=$(pwd)/target"` — after the run,
  `cargo-dinghy` pulls `Documents/target` from the device into your
  repo's `target/`. `$(pwd)` is required (absolute DST) because the
  cargo runner is launched with cwd set to the package dir, not the
  workspace root.

```bash
DEVICE=<DEVICE_ID>

cargo dinghy \
  -d "$DEVICE" \
  -e CRITERION_HOME=target/criterion/a18 \
  --copy-back "Documents/target=$(pwd)/target" \
  bench -p uzu --bench kernel -- \
    "Metal/Kernel/Matmul/GEMM_MPP" \
    --save-baseline matmul_gemm_mpp_baseline_a18
```

After the run completes you'll have
`target/criterion/a18/Metal/Kernel/Matmul/GEMM_MPP/…/matmul_gemm_mpp_baseline_a18/`
on the host, next to any `m2_max/` baselines.

## Viewing reports

Open the Criterion HTML report:

```bash
open target/criterion/report/index.html
```

To inspect a specific label only:

```bash
open target/criterion/m2_max/report/index.html
open target/criterion/a18/report/index.html
```
