// Declarations for GEMV kernels (Metal).
// Implementations are in gemv.metal; names must match Rust dispatch.

#pragma once

#ifdef __METAL_VERSION__
// No host-side declarations needed; compiled by Metal.
#else
// Host-side string literals for kernel names.
#define GEMV_F16_ROWS2 "gemv_f16_rows2"
#define GEMV_F16_ROWS4 "gemv_f16_rows4"
#define GEMV_F16_ROWS8 "gemv_f16_rows8"

#define GEMV_BF16_ROWS2 "gemv_bf16_rows2"
#define GEMV_BF16_ROWS4 "gemv_bf16_rows4"
#define GEMV_BF16_ROWS8 "gemv_bf16_rows8"
#endif


