# GDN Tree Gram Benchmarks

Local reference on M5 Max, `Hg=16 HV=48 K=128`, BF16 inputs, Criterion
`sample_size=10`, 2026-06-26.

Run through the existing `uzu_bench` lib harness:

```bash
cargo bench -p backend-uzu --lib -- "BuildTreeGram/.*/B1_T64"
cargo bench -p backend-uzu --lib -- "BuildTreeGram"
```

The current bench uses the repo-standard `iter_encode_loop_named` GPU-time
harness. The tables below are historical per-dispatch wall-time notes from the
optimization pass.

## Current

Current Ainv path is the scalar 16x16 column solve. The block-wise variant had
isolated MXU large-batch wins, but the simpler code is kept.

Final median time after full-tile unsafe load/store fast paths and
zero-initialized Ainv columns:

| B | Simdgroup T33 | Simdgroup T49 | Simdgroup T64 | Simdgroup T128 | Simdgroup T256 | Simdgroup T512 |
|---|---------------|---------------|---------------|----------------|----------------|----------------|
| 1 | 175.24 us | 177.65 us | 173.69 us | 201.87 us | 301.10 us | 679.74 us |
| 2 | 179.83 us | 183.96 us | 179.33 us | 246.44 us | 426.66 us | 1.1229 ms |
| 4 | 197.03 us | 218.57 us | 210.26 us | 299.20 us | 689.34 us | 2.0174 ms |
| 8 | 234.17 us | 262.53 us | 257.28 us | 438.20 us | 1.1379 ms | 3.8258 ms |

| B | MXU T33 | MXU T49 | MXU T64 | MXU T128 | MXU T256 | MXU T512 |
|---|---------|---------|---------|----------|----------|----------|
| 1 | 173.09 us | 172.65 us | 173.77 us | 183.50 us | 257.54 us | 448.55 us |
| 2 | 173.33 us | 173.81 us | 174.56 us | 201.17 us | 291.87 us | 747.12 us |
| 4 | 175.41 us | 182.72 us | 183.16 us | 258.75 us | 457.23 us | 1.3470 ms |
| 8 | 195.12 us | 221.28 us | 219.26 us | 327.69 us | 830.41 us | 2.7338 ms |

## Variants

B1/T64 variant check, all correctness-tested:

| Variant | Simdgroup | MXU | Kept |
|---|---:|---:|---|
| Baseline | 195.98 us | 180.58 us | no |
| zip-map only | 188.41 us | 177.34 us | no |
| zip-map + staged metadata | 182.08 us | 174.79 us | yes |
| staged + factored exp | 182.08 us | 174.72 us | no, no real gain |
| upper-tile skip only | 192.61 us | 182.10 us | no, MXU regressed |
| staged + factored exp + upper skip | 182.82 us | 175.35 us | no |
| staged metadata + full-tile unsafe loads | 165.53 us | 163.81 us | yes |

Pre epilogue-staging diagonal forward-substitution median time:

| B | MXU T33 | MXU T64 | MXU T128 | MXU T256 |
|---|---------|---------|----------|----------|
| 1 | 173.99 us | 179.70 us | 205.85 us | 301.06 us |
| 2 | 174.77 us | 183.59 us | 250.96 us | 421.89 us |
| 4 | 179.39 us | 217.04 us | 311.54 us | 672.27 us |
| 8 | 212.27 us | 258.01 us | 444.67 us | 1.1549 ms |

| B | Simdgroup T33 | Simdgroup T64 | Simdgroup T128 | Simdgroup T256 |
|---|---------------|---------------|----------------|----------------|
| 1 | 181.29 us | 193.49 us | 246.31 us | 424.49 us |
| 2 | 181.72 us | 198.02 us | 300.56 us | 639.55 us |
| 4 | 199.13 us | 251.43 us | 416.97 us | 1.0192 ms |
| 8 | 256.82 us | 307.09 us | 653.66 us | 1.8139 ms |

After co-walking `kk_acc`/`qk_acc` and staging per-tile prefix/beta/trie
metadata:

| B | MXU T64 | MXU T128 | MXU T256 | MXU T512 |
|---|---------|----------|----------|----------|
| 1 | 175.22 us | 189.40 us | 259.47 us | 487.78 us |
| 2 | 178.05 us | 215.88 us | 338.31 us | 822.90 us |
| 4 | 194.95 us | 264.85 us | 514.95 us | 1.4937 ms |
| 8 | 228.32 us | 355.72 us | 889.52 us | 2.8619 ms |

| B | Simdgroup T64 | Simdgroup T128 | Simdgroup T256 | Simdgroup T512 |
|---|---------------|----------------|----------------|----------------|
| 1 | 183.25 us | 222.01 us | 360.24 us | 883.53 us |
| 2 | 189.73 us | 274.71 us | 529.30 us | 1.5476 ms |
| 4 | 232.40 us | 378.08 us | 901.71 us | 2.8888 ms |
| 8 | 280.61 us | 546.91 us | 1.5795 ms | 5.5700 ms |

## Rejected Ainv Paths

| Variant | Simdgroup B1 T64 | Simdgroup B8 T256 | MXU B1 T64 | MXU B8 T128 | MXU B8 T256 |
|---|---:|---:|---:|---:|---:|
| scalar column Ainv | 173.69 us | 1.1379 ms | 173.77 us | 327.69 us | 830.41 us |
| 8x8 block Ainv + simdgroup matmul | 174.69 us | 1.1317 ms | 175.36 us | 320.86 us | 814.78 us |

| Variant | Simdgroup B1 T64 | Simdgroup B8 T256 | Simdgroup B8 T512 | MXU B1 T64 | MXU B8 T256 |
|---|---:|---:|---:|---:|---:|
| 8x8 block Ainv + scalar diagonal halves | 174.69 us | 1.1317 ms | 3.8325 ms | 175.36 us | 814.78 us |
| 8x8 block Ainv + Neumann diagonal halves | 174.51 us | 1.1749 ms | 4.0259 ms | 173.10 us | 837.44 us |

## Old Neumann Baseline

Speedup over `0600f367` diagonal Neumann. That old kernel has
`TREE_GRAM_MAX_T=64`, so T128/T256 are current-only.

| B | MXU T33 | MXU T64 | Simdgroup T33 | Simdgroup T64 |
|---|---------|---------|---------------|---------------|
| 1 | 1.04x | 1.06x | 1.01x | 1.01x |
| 2 | 1.04x | 1.23x | 1.03x | 1.17x |
| 4 | 1.17x | 1.20x | 1.10x | 1.05x |
| 8 | 1.21x | 1.25x | 1.02x | 1.12x |

Trace `tree_gram_b1_t64_mxu_debug_source.gputrace` showed low occupancy and
low unit use, not a memory-bandwidth wall: occupancy 7.61%, neural accelerator
utilization 6.95%, ALU utilization 10.74%, instruction-throughput utilization
9.52%. The largest source-line buckets were Gram post-processing, then
fragment loads/MMA, then Ainv fill and the small forward substitution.
