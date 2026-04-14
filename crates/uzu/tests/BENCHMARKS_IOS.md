# Running Kernel Benchmarks on iOS

## Prerequisites

- iPhone connected via USB (iOS 26+)
- Rust nightly toolchain: `rustup toolchain install nightly`
- iOS target: `rustup target add aarch64-apple-ios`
- cargo-dinghy: `cargo install cargo-dinghy`
- Device ID: find with `xcrun devicectl list devices`

## Available benchmarks

| Benchmark group                       | Filter pattern     | MXU required |
| ------------------------------------- | ------------------ | ------------ |
| `Metal/Kernel/Matmul/GEMM_MPP`        | `GEMM_MPP/`        | yes          |
| `Metal/Kernel/RMSNorm`                | `RMSNorm/`         | no           |
| `Metal/Kernel/Sampling/Argmax`        | `Argmax/`          | no           |

## Running benchmarks

Run one benchmark group at a time to avoid iOS watchdog killing the app:

```bash
DEVICE=<DEVICE_ID>

# RMSNorm
cargo +nightly dinghy -d "$DEVICE" bench -p uzu --bench kernel -- "RMSNorm/"
```

## Retrieving Criterion reports

Criterion saves HTML reports inside the Dinghy app's sandbox on the device.
To pull them to your local machine:

```bash
xcrun devicectl device copy from \
  --device <DEVICE_ID> \
  --domain-type appDataContainer \
  --domain-identifier org.zoy.kali.Dinghy \
  --source Documents/target/criterion \
  --destination ./criterion-ios
```

Open the report:

```bash
open criterion-ios/report/index.html
```
