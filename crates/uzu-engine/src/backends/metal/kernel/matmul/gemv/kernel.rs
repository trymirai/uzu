use metal::MTLGPUFamily;

use super::policy::{self, DEFAULT_RESULTS_PER_SIMDGROUP, FP_K_BLOCK, trellis_k_block};
use crate::{
    backends::{
        common::{
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{MatmulShape, trellis_format::trellis_params},
        },
        metal::{context::MetalContext, error::MetalError, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

const GEMV_MAX_BATCH: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
    output_row_tile: u32,
    num_simdgroups: u32,
    input_row_tile: u32,
    reduction_lanes: u32,
    group_lanes: u32,
    gathered: bool,
    signed_codes: bool,
    full_tile: bool,
}

impl GemvSpecialization {
    pub fn select_shape(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        gpu_core_count: u32,
        apple_gpu_family: MTLGPUFamily,
    ) -> Option<Self> {
        // THREE trellis geometries. What the decode is short of below M = 5 is
        // THREADS -- at M = 1 it issues ~9.5 instructions per weight against an
        // `instruction_throughput_limiter` of 73% at 25% utilisation, and at
        // M = 4 the two-simdgroup tile still measures 23% occupancy against a
        // 33% target with 90 registers and no spill -- so those widths take a
        // 32-lane, eight-simdgroup tile and only the weight rows per lane move:
        //
        //   * M <= 2 -- `(8, 32, 8)`: one output row per simdgroup, so a lane
        //     owns ONE weight row. Nothing amortises across batch rows here, so
        //     the narrowest tile that fills the machine wins.
        //   * M = 3, 4 -- `(16, 32, 8)`: two output rows per simdgroup, so one
        //     activation vector feeds two rows' worth of decode. Measured
        //     against `(16, 8, 2)` at M = 4: -15.9% on out_proj and -2.3% on
        //     in_proj, i.e. 1.104 -> 0.927 and 1.123 -> 1.100 against the
        //     shipped INT4 route.
        //   * M >= 5 -- `(16, 8, 2)`: past four batch rows the wide threadgroup
        //     stops paying. The same A/B measured `(16, 32, 8)` 6% SLOWER at
        //     M = 6 and M = 8 on in_proj and level on out_proj, so the batch
        //     arm stays where round 10 left it.
        //
        // `(16, 8, 2)` is also the only tile for a K the 32-lane block does not
        // divide. Must agree with `gemv.metal`'s trellis CONSTRAINT.
        if shape.b_prologue == GemmBPrologueKind::Trellis {
            if !(1..=GEMV_MAX_BATCH).contains(&shape.m) {
                return None;
            }
            // Routing rounds M UP to a compiled width rather than compiling one
            // per M: {1, 2, 4, 8} covers 1..=8 in four tiles instead of eight,
            // and a padding row costs only its own arithmetic -- `TrellisSlice`
            // clamps it onto the last real activation row and the epilogue does
            // not store it.
            let batch = shape.m.next_power_of_two();
            // Thirty-two reduction lanes make one K block 512 values wide, so a
            // K that is a whole number of 128s but not of 512s reaches neither
            // 32-lane tile.
            let wide_block = shape.k.is_multiple_of(trellis_k_block(32));
            let tile = match batch {
                1 | 2 if wide_block => policy::GemvTile::quantized_output_tile(8, 8, batch, 32, 1),
                4 if wide_block => policy::GemvTile::quantized_output_tile(8, 16, batch, 32, 1),
                // The eight-lane family is compiled at input row tiles 1, 2 and
                // 8, so a batch of four that lands here walks the grid two rows
                // at a time rather than costing a variant of its own.
                _ => {
                    let rows = if batch == GEMV_MAX_BATCH {
                        batch
                    } else {
                        batch.min(2)
                    };
                    policy::GemvTile::quantized_output_tile(2, 16, rows, 8, 1)
                },
            };
            return Self::select_tile(shape, weights_data_type, input_data_type, output_data_type, tile);
        }
        let is_quant = shape.is_quant();
        let bits = shape.b_bits.unwrap_or(0);
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let tile = if is_quant && shape.gathered {
            policy::gathered_tile(bits, shape.b_group_size.unwrap_or(0), shape.m, shape.n)
        } else if is_quant {
            policy::quantized_tile(
                gpu_core_count,
                apple_gpu_family,
                bits,
                shape.b_group_size.unwrap_or(0),
                shape.m,
                shape.n,
                shape.k,
                shape.d_transform,
                bf16_io,
            )
        } else {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
                return None;
            }
            let input_aligned = shape.k.is_multiple_of(FP_K_BLOCK);
            if shape.d_transform.contains(GemmDTransform::RHT) {
                Some(policy::DEFAULT_TILE)
            } else {
                Some(policy::fp_tile(gpu_core_count, apple_gpu_family, shape.m, shape.n, shape.k, input_aligned))
            }
        };
        Self::select_tile(shape, weights_data_type, input_data_type, output_data_type, tile?)
    }

    pub fn select_tile(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        tile: policy::GemvTile,
    ) -> Option<Self> {
        if !shape.b_transpose || !shape.a_full_precision {
            return None;
        }
        let is_quant = shape.is_quant();
        let bad_leading_dimension = if is_quant {
            shape.b_leading_dimension.is_some()
        } else {
            shape.b_leading_dimension.is_some_and(|ld| ld != shape.k)
        };
        if bad_leading_dimension {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::RHT) && !shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::ACCUMULATE) && !shape.n.is_multiple_of(32) {
            return None;
        }
        let trellis = shape.b_prologue == GemmBPrologueKind::Trellis;
        if trellis
            && (shape.gathered
                || shape.m > GEMV_MAX_BATCH
                || shape.d_transform.contains(GemmDTransform::RHT)
                || !shape.k.is_multiple_of(trellis_k_block(tile.reduction_lanes))
                || [weights_data_type, input_data_type, output_data_type] != [DataType::BF16; 3])
        {
            return None;
        }
        let bits = shape.b_bits.unwrap_or(0);
        if !is_quant {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
                return None;
            }
        }
        let block_size = if !is_quant {
            FP_K_BLOCK
        } else if trellis {
            trellis_k_block(tile.reduction_lanes)
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = shape.k.is_multiple_of(block_size);
        // Gathered quantized rows cannot share one input tile.
        if is_quant && shape.gathered && tile.input_row_tile > 1 {
            return None;
        }
        let specialization = Self {
            b_prologue: shape.b_prologue,
            // A tape has one scale per ROW, so the GEMV has no group at all. The
            // 64 the host reports is the GEMM's staging block (`TRELLIS_BLOCK_K`),
            // which this path does not stage.
            group_size: if trellis {
                0
            } else {
                shape.b_group_size.unwrap_or(0)
            },
            bits,
            output_transform: shape.d_transform,
            input_aligned,
            k_split: tile.k_split,
            output_row_tile: tile.output_row_tile(),
            num_simdgroups: tile.num_simdgroups,
            input_row_tile: tile.input_row_tile,
            reduction_lanes: tile.reduction_lanes,
            group_lanes: tile.group_lanes,
            gathered: shape.gathered,
            signed_codes: shape.signed_codes,
            full_tile: full_tile(shape, tile),
        };
        Some(specialization)
    }

    pub fn output_row_tile(&self) -> u32 {
        self.output_row_tile
    }

    /// Only the trellis routing test reads this.
    #[cfg(test)]
    pub fn reduction_lanes(&self) -> u32 {
        self.reduction_lanes
    }

    fn create_pipeline(
        &self,
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<GemvMetalKernel, MetalError> {
        GemvMetalKernel::new(
            context,
            input_data_type,
            weights_data_type,
            output_data_type,
            self.b_prologue,
            self.group_size,
            self.bits,
            self.k_split,
            self.input_aligned,
            self.input_row_tile,
            self.output_row_tile(),
            self.reduction_lanes,
            self.group_lanes,
            self.num_simdgroups,
            self.output_transform,
            self.gathered,
            self.signed_codes,
            self.full_tile,
        )
    }
}

fn full_tile(
    shape: &MatmulShape,
    tile: policy::GemvTile,
) -> bool {
    shape.m.is_multiple_of(tile.input_row_tile) && shape.n.is_multiple_of(tile.output_row_tile())
}

use std::collections::{HashMap, hash_map::Entry};

use crate::backends::{
    common::{
        BufferArg, Encoder,
        kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError},
    },
    metal::Metal,
};

/// GEMV pipelines compiled on first use.
pub struct GemvKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
}

impl GemvKernel {
    pub fn new(
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = specialization
                    .create_pipeline(context, self.weights_data_type, self.input_data_type, self.output_data_type)
                    .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let soft_cap = arguments.d_transform.soft_cap;

        let MatmulArguments {
            a,
            b,
            d,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;
        let MatmulA::FullPrecision {
            values: a,
            offset: a_offset,
        } = a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "Gemv",
                reason: "prepared int8 activations require GEMM",
            });
        };

        // Preserve each weight buffer's residency range.
        let (scales, zero_points, biases) = match &b {
            MatmulB::FullPrecision {
                ..
            } => (None, None, None),
            MatmulB::ScaleBiasDequant {
                scales,
                biases,
                ..
            } => (Some(*scales), None, Some(*biases)),
            MatmulB::ScaleZeroPointDequant {
                scales,
                zero_points,
                ..
            } => (Some(*scales), Some(*zero_points), None),
            // A trellis tape is the `b` buffer and its per-row scales are the
            // `scales` buffer, so its residency is the symmetric path's.
            MatmulB::ScaleSymmetricDequant {
                scales,
                ..
            }
            | MatmulB::Trellis {
                scales,
                ..
            } => (Some(*scales), None, None),
        };

        let output_group_count = n.div_ceil(specialization.output_row_tile());
        let context = encoder.context();
        let pipeline = self.get_or_create(context, specialization)?;
        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => pipeline.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                &mut *d,
                output_bias,
                rht_factors,
                gather_indices,
                k,
                n,
                m,
                ab_scale,
                output_group_count,
                soft_cap,
                None,
                encoder,
            ),
            MatmulB::ScaleBiasDequant {
                b: weights,
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                b: weights,
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                b: weights,
                ..
            } => pipeline.encode(
                weights,
                scales,
                zero_points,
                biases,
                (a, a_offset),
                &mut *d,
                output_bias,
                rht_factors,
                gather_indices,
                k,
                n,
                m,
                ab_scale,
                output_group_count,
                soft_cap,
                None,
                encoder,
            ),
            MatmulB::Trellis {
                b: weights,
                config,
                ..
            } => {
                let Some(params) = trellis_params(config, k) else {
                    return Err(MatmulError::UnsupportedLayout {
                        path: "Gemv trellis",
                    });
                };
                pipeline.encode(
                    weights,
                    scales,
                    zero_points,
                    biases,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    gather_indices,
                    k,
                    n,
                    m,
                    ab_scale,
                    output_group_count,
                    soft_cap,
                    Some(params),
                    encoder,
                )
            },
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/matmul/gemv/trellis_test.rs"]
mod trellis_tests;
