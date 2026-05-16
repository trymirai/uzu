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
                Nf4QmvByte256MetalKernel, Nf4QmvConstantMetalKernel, Nf4QmvE4m3MetalKernel, Nf4QmvSelectMetalKernel,
                Nf4QmvShuffleMetalKernel, Nf4QmvTgMetalKernel, Nf4QmvZpMetalKernel,
            },
        },
    },
};

/// Which NF4 lookup strategy a kernel uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4Variant {
    Constant,
    Tg,
    E4m3,
    /// Asymmetric: constant codebook + 4-bit per-group zero-point LUT offset.
    /// Requires a `zero_points` buffer — use `encode_zp` (the plain `encode`
    /// has no zero-point arg and will panic for this variant).
    Zp,
    /// Byte-batched 256-entry threadgroup `half2` codebook LUT (mirrors the
    /// int4 `awq-lut256` access pattern). Same math as `Constant`.
    Byte256,
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
}

/// QMV NF4 kernel set (one for each lookup strategy). Built lazily by `new`.
pub struct Nf4QmvBench {
    constant: Nf4QmvConstantMetalKernel,
    tg: Nf4QmvTgMetalKernel,
    e4m3: Nf4QmvE4m3MetalKernel,
    zp: Nf4QmvZpMetalKernel,
    byte256: Nf4QmvByte256MetalKernel,
    shuffle8: Nf4QmvShuffleMetalKernel,
    shuffle16: Nf4QmvShuffleMetalKernel,
    shuffle32: Nf4QmvShuffleMetalKernel,
    select: Nf4QmvSelectMetalKernel,
}

impl Nf4QmvBench {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            constant: Nf4QmvConstantMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg: Nf4QmvTgMetalKernel::new(ctx, DataType::BF16, 64)?,
            e4m3: Nf4QmvE4m3MetalKernel::new(ctx, DataType::BF16, 64)?,
            zp: Nf4QmvZpMetalKernel::new(ctx, DataType::BF16, 64)?,
            byte256: Nf4QmvByte256MetalKernel::new(ctx, DataType::BF16, 64)?,
            shuffle8: Nf4QmvShuffleMetalKernel::new(ctx, DataType::BF16, 64, 8)?,
            shuffle16: Nf4QmvShuffleMetalKernel::new(ctx, DataType::BF16, 64, 16)?,
            shuffle32: Nf4QmvShuffleMetalKernel::new(ctx, DataType::BF16, 64, 32)?,
            select: Nf4QmvSelectMetalKernel::new(ctx, DataType::BF16, 64)?,
        })
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
            Nf4Variant::E4m3 => {
                self.e4m3.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
            Nf4Variant::Byte256 => {
                self.byte256.encode(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, encoder)
            },
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
            (Nf4Variant::Shuffle8 | Nf4Variant::Shuffle16 | Nf4Variant::Shuffle32, _) => {
                panic!("Nf4Variant::Shuffle* is QMV-only; no Nf4QmmShuffle kernel exists")
            },
            (Nf4Variant::Select, _) => {
                panic!("Nf4Variant::Select is QMV-only; no Nf4QmmSelect kernel exists")
            },
            (Nf4Variant::Zp, _) => {
                panic!("Nf4Variant::Zp requires a zero_points buffer; call Nf4QmmBench::encode_zp instead")
            },
        }
    }
}
