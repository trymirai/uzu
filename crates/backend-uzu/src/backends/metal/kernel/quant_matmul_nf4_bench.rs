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
                Nf4QmmConstantMetalKernel, Nf4QmmE4m3MetalKernel, Nf4QmmTgMetalKernel, Nf4QmvConstantMetalKernel,
                Nf4QmvE4m3MetalKernel, Nf4QmvTgMetalKernel,
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
}

/// QMV NF4 kernel set (one for each lookup strategy). Built lazily by `new`.
pub struct Nf4QmvBench {
    constant: Nf4QmvConstantMetalKernel,
    tg: Nf4QmvTgMetalKernel,
    e4m3: Nf4QmvE4m3MetalKernel,
}

impl Nf4QmvBench {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            constant: Nf4QmvConstantMetalKernel::new(ctx, DataType::BF16, 64)?,
            tg: Nf4QmvTgMetalKernel::new(ctx, DataType::BF16, 64)?,
            e4m3: Nf4QmvE4m3MetalKernel::new(ctx, DataType::BF16, 64)?,
        })
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
        })
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
        }
    }
}
