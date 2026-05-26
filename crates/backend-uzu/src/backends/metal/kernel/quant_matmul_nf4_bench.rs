//! Public bench-only dispatcher for the experimental NF4 (NormalFloat-4-bit)
//! Metal kernels. These kernels are non-PUBLIC in the DSL sense (no CPU impl),
//! so their auto-generated wrappers have `pub(crate)` constructors/encoders.
//! This module re-exposes them for the bench test crate.

use crate::{
    DataType,
    backends::{
        common::Encoder,
        metal::{
            Metal, MetalContext,
            error::MetalError,
            kernel::{
                Nf4QmmConstantMetalKernel, Nf4QmmE4m3MetalKernel, Nf4QmmTgMetalKernel, Nf4QmmZpMetalKernel,
                Nf4QmvByte256DupMetalKernel, Nf4QmvByte256MetalKernel, Nf4QmvConstantMetalKernel,
                Nf4QmvE4m3MetalKernel, Nf4QmvSelectMetalKernel, Nf4QmvShuffleMetalKernel, Nf4QmvTgIlpMetalKernel,
                Nf4QmvTgMetalKernel, Nf4QmvTgNoBarrierMetalKernel, Nf4QmvTgReplicatedMetalKernel,
                Nf4QmvTgSimdbarDevbufMetalKernel, Nf4QmvTgSimdbarMetalKernel, Nf4QmvTgVec4MetalKernel,
                Nf4QmvZpMetalKernel, QmvFastNf4PrecomputedMetalKernel, QmvFastTemplateAwqLutMetalKernel,
                QmvFastTemplateNf4LutMetalKernel,
            },
        },
    },
};

/// Which NF4 lookup strategy a kernel uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4Variant {
    Constant,
    Tg,
    /// 4x replicated threadgroup codebook (128 B = 32 banks). Each lane reads
    /// from copy `simd_lane >> 3` to spread tgmem accesses across all banks,
    /// eliminating the 8-idle-bank waste of plain `Tg`.
    TgReplicated,
    /// vec4-padded replicated TG codebook (`half[16][4]` = 128 B = 32 banks).
    /// Each nibble has 4 interleaved replicas; lane reads
    /// `lut[nibble * 4 + (simd_lane & 3)]`. Alternative bank-spread to
    /// `TgReplicated` (which uses contiguous copies + lane_id >> 3).
    TgVec4,
    /// Same threadgroup 16-entry codebook as `Tg`, but the inner qdot uses
    /// `qdot_nf4_tg_ilp` (multi-accumulator: 2 independent accumulators per
    /// outer iter, 8 TG loads issued back-to-back before any dot consumes
    /// them). Tests whether breaking the per-iter dependency chain closes
    /// the memory-latency-bound gap (Instruction Throughput Limiter ~90% on
    /// plain Nf4QmvTg on M4).
    TgIlp,
    /// PERF-ONLY PROBE: byte-for-byte copy of `Tg` with the
    /// `threadgroup_barrier(mem_threadgroup)` after cooperative codebook init
    /// REMOVED. Outputs are INTENTIONALLY INCORRECT (race on codebook). Purpose
    /// is to bound the kernel-entry cost of the barrier on M4 / decide whether
    /// to invest in a simdgroup-local LUT variant.
    TgNoBarrier,
    /// Simdgroup-local TG codebook (`half[8][16]`). Per-simdgroup cooperative
    /// init within 16 lanes, ordered with `simdgroup_barrier(mem_threadgroup)`
    /// — which only syncs within the simdgroup (cheaper than the full
    /// 256-lane threadgroup barrier in `Tg`). Correctness: bit-equivalent to
    /// `Nf4QmvConstant`.
    TgSimdbar,
    /// Production-flexible variant of `TgSimdbar`: same simdgroup-local TG
    /// codebook + `simdgroup_barrier(mem_threadgroup)` layout, but the 16
    /// codebook entries come from a CPU-provided `const device half*`
    /// buffer (set at dispatch time) instead of the constant
    /// `nf4_codebook[16]`. Requires the codebook device buffer — call
    /// `encode_tg_simdbar_devbuf` (the plain `encode` panics for this
    /// variant).
    TgSimdbarDevbuf,
    E4m3,
    /// Asymmetric: constant codebook + 4-bit per-group zero-point LUT offset.
    /// Requires a `zero_points` buffer — use `encode_zp` (the plain `encode`
    /// has no zero-point arg and will panic for this variant).
    Zp,
    /// Byte-batched 256-entry threadgroup `half2` codebook LUT (mirrors the
    /// int4 `awq-lut256` access pattern). Same math as `Constant`.
    Byte256,
    /// FLUTE's trick: byte-batched 256-entry `half2` codebook LUT replicated
    /// D=8 times in threadgroup memory. Lane picks copy via `simd_lane & 7`.
    /// Same math as `Byte256`; only the bank distribution of the TG load
    /// differs (relieves L1/threadgroup-port saturation).
    Byte256Dup8,
    /// FLUTE's trick: pair-LUT replicated D=16 times. Lane picks copy via
    /// `simd_lane & 15`.
    Byte256Dup16,
    /// FLUTE's trick: pair-LUT replicated D=32 times (one copy per simd lane).
    Byte256Dup32,
    /// Zero-memory register *shuffle* codebook, size 8 (synthetic 3-bit
    /// timing probe). Dequant via `simd_shuffle` of a register-held entry.
    Shuffle8,
    /// Zero-memory register *shuffle* codebook, size 16 (real NF4 codebook;
    /// numerically equivalent to `Constant`).
    Shuffle16,
    /// Zero-memory register *shuffle* codebook, size 32 (synthetic 5-bit
    /// timing probe; weights still 4-bit nibbles).
    Shuffle32,
    /// Zero-memory in-thread *select* codebook: per-nibble switch-of-literals
    /// (all 16 NF4 values as compile-time constants). No memory, no cross-lane.
    /// Numerically equivalent to `Constant`.
    Select,
    /// AWQ-int4 byte-batched LUT path with hardcoded constexpr flags (no
    /// SPECIALIZE function constants). Sanity-check sibling to
    /// `QmvFastTemplateNf4Lut` — should match the SPECIALIZE-based awq-lut256
    /// dispatch within ±2pp. Uses the same buffer set as awq-lut256:
    /// u32-packed weights + bf16 scales + 4-bit packed zero-points.
    QmvFastTemplateAwqLut,
    /// NF4 byte-batched LUT path with hardcoded constexpr flags. The deciding
    /// experiment: does eliminating SPECIALIZE-driven PSO codegen close the
    /// +91% NF4 vs AWQ gap? Same buffer set as nf4-lut-grft: u32-packed
    /// weights + bf16 scales (no zero-points).
    QmvFastTemplateNf4Lut,
    /// NF4 byte-batched LUT path where the 256-entry bfloat2 LUT is
    /// PRECOMPUTED CPU-side and bound as a `device` buffer. The kernel does
    /// NOT include `nf4_common.h`, so the compiler has no compile-time
    /// visibility into the 16 NF4 codebook values. Deciding experiment for
    /// "is compile-time codebook visibility the cause of the +91% gap?".
    /// Requires the precomputed LUT buffer — call `encode_nf4_precomputed`.
    QmvFastNf4Precomputed,
}

/// QMV NF4 kernel set (one for each lookup strategy). Built lazily by `new`.
pub struct Nf4QmvBench {
    constant: Nf4QmvConstantMetalKernel,
    tg: Nf4QmvTgMetalKernel,
    tg_replicated: Nf4QmvTgReplicatedMetalKernel,
    tg_vec4: Nf4QmvTgVec4MetalKernel,
    tg_ilp: Nf4QmvTgIlpMetalKernel,
    tg_nobar: Nf4QmvTgNoBarrierMetalKernel,
    tg_simdbar: Nf4QmvTgSimdbarMetalKernel,
    tg_simdbar_devbuf: Nf4QmvTgSimdbarDevbufMetalKernel,
    e4m3: Nf4QmvE4m3MetalKernel,
    zp: Nf4QmvZpMetalKernel,
    byte256: Nf4QmvByte256MetalKernel,
    byte256_dup8: Nf4QmvByte256DupMetalKernel,
    byte256_dup16: Nf4QmvByte256DupMetalKernel,
    byte256_dup32: Nf4QmvByte256DupMetalKernel,
    shuffle8: Nf4QmvShuffleMetalKernel,
    shuffle16: Nf4QmvShuffleMetalKernel,
    shuffle32: Nf4QmvShuffleMetalKernel,
    select: Nf4QmvSelectMetalKernel,
    tmpl_awq: QmvFastTemplateAwqLutMetalKernel,
    tmpl_nf4: QmvFastTemplateNf4LutMetalKernel,
    nf4_precomputed: QmvFastNf4PrecomputedMetalKernel,
}

impl Nf4QmvBench {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            constant: Nf4QmvConstantMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg: Nf4QmvTgMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg_replicated: Nf4QmvTgReplicatedMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg_vec4: Nf4QmvTgVec4MetalKernel::new(ctx, DataType::BF16, 64)?,
            tg_ilp: Nf4QmvTgIlpMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg_nobar: Nf4QmvTgNoBarrierMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg_simdbar: Nf4QmvTgSimdbarMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg_simdbar_devbuf: Nf4QmvTgSimdbarDevbufMetalKernel::new(ctx, DataType::BF16, 64)?,
            e4m3: Nf4QmvE4m3MetalKernel::new(ctx, DataType::BF16, 64)?,
            zp: Nf4QmvZpMetalKernel::new(ctx, DataType::BF16, 64)?,
            byte256: Nf4QmvByte256MetalKernel::new(ctx, DataType::BF16, 64)?,
            byte256_dup8: Nf4QmvByte256DupMetalKernel::new(ctx, DataType::BF16, 64, 8)?,
            byte256_dup16: Nf4QmvByte256DupMetalKernel::new(ctx, DataType::BF16, 64, 16)?,
            byte256_dup32: Nf4QmvByte256DupMetalKernel::new(ctx, DataType::BF16, 64, 32)?,
            shuffle8: Nf4QmvShuffleMetalKernel::new(ctx, DataType::BF16, 64, 8)?,
            shuffle16: Nf4QmvShuffleMetalKernel::new(ctx, DataType::BF16, 64, 16)?,
            shuffle32: Nf4QmvShuffleMetalKernel::new(ctx, DataType::BF16, 64, 32)?,
            select: Nf4QmvSelectMetalKernel::new(ctx, DataType::BF16, 64)?,
            tmpl_awq: QmvFastTemplateAwqLutMetalKernel::new(ctx, DataType::BF16, 64)?,
            tmpl_nf4: QmvFastTemplateNf4LutMetalKernel::new(ctx, DataType::BF16, 64)?,
            nf4_precomputed: QmvFastNf4PrecomputedMetalKernel::new(ctx, DataType::BF16, 64)?,
        })
    }

    /// Encode the template-AWQ QMV kernel (needs a `zero_points` buffer).
    /// Uses the SAME buffer layout as QmvFast use_lut=true: u32-packed
    /// weights, bf16 scales, 4-bit packed zero-points two-per-byte.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_tmpl_awq(
        &self,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        zero_points: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        self.tmpl_awq.encode(
            weights,
            scales,
            zero_points,
            input,
            output,
            in_vec_size,
            out_vec_size,
            batch_size,
            encoder,
        )
    }

    /// Encode the QmvFastNf4Precomputed QMV kernel (needs a CPU-precomputed
    /// 256-entry `bfloat2` LUT buffer). Same buffer set as nf4-lut-grft for
    /// weights/scales/input/output, but adds the precomputed LUT.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nf4_precomputed(
        &self,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        precomputed_lut: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        self.nf4_precomputed.encode(
            weights,
            scales,
            precomputed_lut,
            input,
            output,
            in_vec_size,
            out_vec_size,
            batch_size,
            encoder,
        )
    }

    /// Encode the `Nf4QmvTgSimdbarDevbuf` QMV kernel (needs a 16-entry
    /// codebook device buffer, half-precision). Same NF4 weights/scales as
    /// `Constant` for buffer layout; codebook values are arbitrary.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_tg_simdbar_devbuf(
        &self,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        codebook_dev: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        self.tg_simdbar_devbuf.encode(
            weights,
            scales,
            codebook_dev,
            input,
            output,
            in_vec_size,
            out_vec_size,
            batch_size,
            encoder,
        )
    }

    /// Encode the NF4-ZP QMV kernel (needs a `zero_points` buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_zp(
        &self,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        zero_points: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        self.zp.encode(weights, scales, zero_points, input, output, in_vec_size, out_vec_size, batch_size, encoder)
    }

    pub fn encode(
        &self,
        variant: Nf4Variant,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        match variant {
            Nf4Variant::Constant => {
                self.constant.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Tg => {
                self.tg.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::TgReplicated => self.tg_replicated.encode(
                weights,
                scales,
                input,
                output,
                in_vec_size,
                out_vec_size,
                batch_size,
                encoder,
            ),
            Nf4Variant::TgVec4 => {
                self.tg_vec4.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::TgIlp => {
                self.tg_ilp.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::TgNoBarrier => {
                self.tg_nobar.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::TgSimdbar => {
                self.tg_simdbar.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::TgSimdbarDevbuf => panic!(
                "Nf4Variant::TgSimdbarDevbuf requires a codebook device buffer; \
                 call Nf4QmvBench::encode_tg_simdbar_devbuf instead"
            ),
            Nf4Variant::E4m3 => {
                self.e4m3.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Byte256 => {
                self.byte256.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Byte256Dup8 => {
                self.byte256_dup8.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Byte256Dup16 => self.byte256_dup16.encode(
                weights,
                scales,
                input,
                output,
                in_vec_size,
                out_vec_size,
                batch_size,
                encoder,
            ),
            Nf4Variant::Byte256Dup32 => self.byte256_dup32.encode(
                weights,
                scales,
                input,
                output,
                in_vec_size,
                out_vec_size,
                batch_size,
                encoder,
            ),
            Nf4Variant::Shuffle8 => {
                self.shuffle8.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Shuffle16 => {
                self.shuffle16.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Shuffle32 => {
                self.shuffle32.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Select => {
                self.select.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::QmvFastTemplateNf4Lut => {
                self.tmpl_nf4.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::QmvFastTemplateAwqLut => panic!(
                "Nf4Variant::QmvFastTemplateAwqLut requires a zero_points buffer; \
                 call Nf4QmvBench::encode_tmpl_awq instead"
            ),
            Nf4Variant::QmvFastNf4Precomputed => panic!(
                "Nf4Variant::QmvFastNf4Precomputed requires a precomputed_lut buffer; \
                 call Nf4QmvBench::encode_nf4_precomputed instead"
            ),
            Nf4Variant::Zp => {
                panic!("Nf4Variant::Zp requires a zero_points buffer; call Nf4QmvBench::encode_zp instead")
            },
        }
    }
}

/// Which BM tile size to use for an NF4 QMM dispatch. Matches the AWQ
/// production dispatcher's pick (BM=8 for batch_dim < 48, BM=64 otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4QmmTile {
    Small, // BM=8,  BK=32, BN=32, WM=WN=1 (32 threads = 1 simdgroup)
    Big,   // BM=64, BK=64, BN=64, WM=WN=2 (128 threads = 4 simdgroups)
}

/// QMM NF4 kernel set. Each variant holds two pipeline instances — a small
/// BM=8 tile and a big BM=64 tile — so the bench can dispatch the same tile
/// AWQ's production dispatcher picks for a given M.
pub struct Nf4QmmBench {
    constant_small: Nf4QmmConstantMetalKernel,
    constant_big: Nf4QmmConstantMetalKernel,
    tg_small: Nf4QmmTgMetalKernel,
    tg_big: Nf4QmmTgMetalKernel,
    e4m3_small: Nf4QmmE4m3MetalKernel,
    e4m3_big: Nf4QmmE4m3MetalKernel,
    zp_small: Nf4QmmZpMetalKernel,
    zp_big: Nf4QmmZpMetalKernel,
}

impl Nf4QmmBench {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            constant_small: Nf4QmmConstantMetalKernel::new(ctx, DataType::BF16, 64, 8, 32, 32, 1, 1)?,
            constant_big: Nf4QmmConstantMetalKernel::new(ctx, DataType::BF16, 64, 64, 64, 64, 2, 2)?,
            tg_small: Nf4QmmTgMetalKernel::new(ctx, DataType::BF16, 64, 8, 32, 32, 1, 1)?,
            tg_big: Nf4QmmTgMetalKernel::new(ctx, DataType::BF16, 64, 64, 64, 64, 2, 2)?,
            e4m3_small: Nf4QmmE4m3MetalKernel::new(ctx, DataType::BF16, 64, 8, 32, 32, 1, 1)?,
            e4m3_big: Nf4QmmE4m3MetalKernel::new(ctx, DataType::BF16, 64, 64, 64, 64, 2, 2)?,
            zp_small: Nf4QmmZpMetalKernel::new(ctx, DataType::BF16, 64, 8, 32, 32, 1, 1)?,
            zp_big: Nf4QmmZpMetalKernel::new(ctx, DataType::BF16, 64, 64, 64, 64, 2, 2)?,
        })
    }

    /// Encode the NF4-ZP QMM kernel (needs a `zero_points` buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_zp(
        &self,
        tile: Nf4QmmTile,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        zero_points: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        let k = match tile {
            Nf4QmmTile::Small => &self.zp_small,
            Nf4QmmTile::Big => &self.zp_big,
        };
        k.encode(weights, scales, zero_points, input, output, in_vec_size, out_vec_size, batch_size, encoder)
    }

    pub fn encode(
        &self,
        variant: Nf4Variant,
        tile: Nf4QmmTile,
        weights: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        scales: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        input: &<Metal as crate::backends::common::Backend>::DenseBuffer,
        output: &mut <Metal as crate::backends::common::Backend>::DenseBuffer,
        in_vec_size: u32,
        out_vec_size: u32,
        batch_size: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        match (variant, tile) {
            (Nf4Variant::Constant, Nf4QmmTile::Small) => self.constant_small.encode(
                weights,
                scales,
                input,
                output,
                in_vec_size,
                out_vec_size,
                batch_size,
                encoder,
            ),
            (Nf4Variant::Constant, Nf4QmmTile::Big) => {
                self.constant_big.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            (Nf4Variant::Tg, Nf4QmmTile::Small) => {
                self.tg_small.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            (Nf4Variant::Tg, Nf4QmmTile::Big) => {
                self.tg_big.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            (Nf4Variant::E4m3, Nf4QmmTile::Small) => {
                self.e4m3_small.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            (Nf4Variant::E4m3, Nf4QmmTile::Big) => {
                self.e4m3_big.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            (Nf4Variant::Byte256, _) => {
                panic!("Nf4Variant::Byte256 is QMV-only; no Nf4QmmByte256 kernel exists")
            },
            (Nf4Variant::Byte256Dup8 | Nf4Variant::Byte256Dup16 | Nf4Variant::Byte256Dup32, _) => {
                panic!("Nf4Variant::Byte256Dup* is QMV-only; no QMM kernel exists")
            },
            (Nf4Variant::TgReplicated, _) => {
                panic!("Nf4Variant::TgReplicated is QMV-only; no Nf4QmmTgReplicated kernel exists")
            },
            (Nf4Variant::TgVec4, _) => {
                panic!("Nf4Variant::TgVec4 is QMV-only; no Nf4QmmTgVec4 kernel exists")
            },
            (Nf4Variant::TgIlp, _) => {
                panic!("Nf4Variant::TgIlp is QMV-only; no Nf4QmmTgIlp kernel exists")
            },
            (Nf4Variant::TgNoBarrier, _) => {
                panic!("Nf4Variant::TgNoBarrier is QMV-only (perf probe); no Nf4QmmTgNoBarrier kernel exists")
            },
            (Nf4Variant::TgSimdbar, _) => {
                panic!("Nf4Variant::TgSimdbar is QMV-only; no Nf4QmmTgSimdbar kernel exists")
            },
            (Nf4Variant::TgSimdbarDevbuf, _) => {
                panic!("Nf4Variant::TgSimdbarDevbuf is QMV-only; no Nf4QmmTgSimdbarDevbuf kernel exists")
            },
            (Nf4Variant::Shuffle8 | Nf4Variant::Shuffle16 | Nf4Variant::Shuffle32, _) => {
                panic!("Nf4Variant::Shuffle* is QMV-only; no Nf4QmmShuffle kernel exists")
            },
            (Nf4Variant::Select, _) => {
                panic!("Nf4Variant::Select is QMV-only; no Nf4QmmSelect kernel exists")
            },
            (Nf4Variant::QmvFastTemplateAwqLut | Nf4Variant::QmvFastTemplateNf4Lut, _) => {
                panic!("Nf4Variant::QmvFastTemplate* is QMV-only; no QMM kernel exists")
            },
            (Nf4Variant::QmvFastNf4Precomputed, _) => {
                panic!("Nf4Variant::QmvFastNf4Precomputed is QMV-only; no QMM kernel exists")
            },
            (Nf4Variant::Zp, _) => {
                panic!("Nf4Variant::Zp requires a zero_points buffer; call Nf4QmmBench::encode_zp instead")
            },
        }
    }
}
