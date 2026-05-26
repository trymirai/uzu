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

| Group id                          | Filter                            |
| --------------------------------- | --------------------------------- |
| `Metal/Kernel/Matmul/GEMM`        | `Metal/Kernel/Matmul/GEMM`        |
| `Metal/Kernel/Matmul/GEMM_MXU`    | `Metal/Kernel/Matmul/GEMM_MXU`    |
| `Metal/Kernel/QmmTransposed/...`  | `Metal/Kernel/QmmTransposed`      |
| `Metal/Kernel/QmvFast/...`        | `Metal/Kernel/QmvFast`            |
| `Metal/Kernel/RMSNorm`            | `Metal/Kernel/RMSNorm`            |
| `Metal/Kernel/Sampling/Argmax`    | `Metal/Kernel/Sampling/Argmax`    |

The prefix `Metal/Kernel/Matmul` runs both `GEMM` and `GEMM_MXU` in one pass.

## Output layout

Every run writes into `target/criterion/<label>/…`, where `<label>` is a
free-form name you choose (e.g. `m2_max`, `a19`). The Criterion baseline
you saved lives at `target/criterion/<label>/<benchmark-path>/<baseline-name>/`.

## Running on macOS

From the repo root. Use an **absolute** `CRITERION_HOME` so it doesn't
resolve relative to the package dir:

```bash
CRITERION_HOME="$PWD/target/criterion/m2_max" cargo bench \
  -p backend-uzu \
  --bench kernel -- "Metal/Kernel/Matmul" \
  --save-baseline matmul_baseline_m2_max
```

## Running on iPhone (via `cargo-dinghy`)

Run one benchmark group at a time to avoid the iOS watchdog killing the
app.

Key flags (this is the trymirai/dinghy fork — upstream cargo-dinghy does
not support `--sync-dirs`):

- `-e CRITERION_HOME=criterion/a19` — on-device env var. Path is
  relative to the app's cwd (`Documents/`), so this becomes
  `Documents/criterion/a19/` on device.
- `--sync-dirs "$(pwd)/target/criterion=Documents/criterion"` —
  bidirectional sync of one host directory with one device directory.
  Before launch, `cargo-dinghy` runs `xcrun devicectl device copy to`
  to push the host directory to the device (overwriting the device
  copy). After launch, it runs `xcrun devicectl device copy from` to
  pull it back.

> **Important:** scope `--sync-dirs` to `target/criterion`, **not** the
> full `target/`. A workspace `target/` directory routinely grows past
> 20 GB (build artefacts, dinghy runner intermediates), and
> `devicectl copy` performs an unconditional whole-tree push and pull
> with `--remove-existing-content true`. Pushing 20 GB over USB takes
> 5–10 minutes per direction, before any benchmark code runs.
> `target/criterion` is typically <100 MB.
>
> The first push on a fresh device is skipped automatically (when the
> device path does not yet exist), but every subsequent run pushes
> whatever is at the host path.

```bash
DEVICE=<DEVICE_ID>

cargo dinghy \
  -d "$DEVICE" \
  -e CRITERION_HOME=criterion/a19 \
  --sync-dirs "$(pwd)/target/criterion=Documents/criterion" \
  bench -p backend-uzu --bench kernel -- \
    "Metal/Kernel/Matmul" \
    --save-baseline matmul_baseline_a19
```

After the run completes you'll have
`target/criterion/a19/Metal/Kernel/Matmul/<GEMM|GEMM_MXU>/…/matmul_baseline_a19/`
on the host, next to any `m2_max/` baselines.

### Tips

- Trim or wipe old baselines from `target/criterion/` before runs you do
  not need shipped to the device — they are part of the push payload.
- The first device-side run with a fresh `Documents/criterion` skips the
  initial push. To reset, delete it: `xcrun devicectl device process
  launch ...` is not needed; reinstalling the Dinghy app or wiping its
  data container achieves the same.
- Use a tighter Criterion regex filter to keep on-device time short and
  the resulting baseline directory small, e.g.
  `"Metal/Kernel/UnifiedQuantizedGemm/(Simdgroup|Mxu)/ScaleBias_BF16_gs64/M\[(4|8|16|32|64)\]K\[4096\]N\[(4096|14336)\]"`
  matches 20 cases (~3–4 min on device).

## Viewing reports

Open the Criterion HTML report:

```bash
open target/criterion/report/index.html
```

To inspect a specific label only:

```bash
open target/criterion/m2_max/report/index.html
open target/criterion/a19/report/index.html
```
