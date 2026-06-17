# M5 Small-M GEMM Notes

Scope: M5, small-M `M=4..32`, auto dispatch on this branch vs `main`.

Kept MXU tiles:

- `Tile16x32x256_Simdgroups1x1`
- `Tile16x128x256_Simdgroups1x4`

Dropped/disabled after tuning: tile8 variants, `16x16`, `16x64`, `32x32`, `32x128`, q4/q8 small-tile auto dispatch, and mixed-type quant small-tile variants.

Benchmark method:

- temporary unit-test harness, removed after collection
- auto dispatch only
- fresh Metal buffers for every measured sample
- one unreported warmup sample per row
- median of 10 GPU execution times
- 45s cooldown before every measured row

Full-precision cold-buffer results:

| Type | M | K | N | Current us | Main us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 2560 | 2560 | 74.541 | 121.125 | 1.62x |
| BF16 | 5 | 9728 | 2560 | 628.062 | 660.667 | 1.05x |
| BF16 | 8 | 9728 | 2560 | 638.542 | 826.833 | 1.29x |
| BF16 | 16 | 4096 | 24576 | 2538.417 | 2557.000 | 1.01x |
| BF16 | 24 | 4096 | 24576 | 2454.208 | 1101.000 | 0.45x |
| BF16 | 32 | 4096 | 12288 | 977.667 | 1047.229 | 1.07x |
| F32 | 4 | 2560 | 2560 | 333.271 | 306.625 | 0.92x |
| F32 | 5 | 9728 | 2560 | 463.708 | 1071.188 | 2.31x |
| F32 | 8 | 9728 | 2560 | 898.292 | 1620.625 | 1.80x |
| F32 | 16 | 4096 | 24576 | 1320.438 | 1723.459 | 1.31x |
| F32 | 24 | 4096 | 24576 | 1305.979 | 1280.584 | 0.98x |
| F32 | 32 | 4096 | 12288 | 1064.562 | 1301.854 | 1.22x |

Quant rows are omitted because no quant-specific tile is kept.

Binary size:

- main `default.metallib`: 49,445,696 bytes, 47.16 MiB
- current `default.metallib`: 50,350,905 bytes, 48.02 MiB
- delta: +905,209 bytes, +0.86 MiB, +1.83%

Verification:

- `cargo fmt`
- `git diff --check`
- `cargo test -p backend-uzu small_m_mxu_tiles_parity -- --nocapture`
- `cargo test -p backend-uzu mxu_quant_parity_bf16 -- --nocapture`
- `cargo check -p backend-uzu --benches`
- `cargo bench -p backend-uzu --lib --no-run`
