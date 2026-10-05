use metal::MTLGPUFamily;

use super::{
    super::MatmulOutputWork,
    policy::{self, DEFAULT_RESULTS_PER_SIMDGROUP, FP_K_BLOCK},
};
use crate::{
    backends::{
        common::{
            CommandBufferEncoding,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{MatmulB, MatmulShape},
        },
        metal::{
            command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError,
            kernel::GemvMetalKernel,
        },
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
                bf16_io,
            )
        } else {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
                return None;
            }
            let input_aligned = shape.k.is_multiple_of(FP_K_BLOCK);
            Some(policy::fp_tile(gpu_core_count, apple_gpu_family, shape.m, shape.n, shape.k, input_aligned))
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
        let output_transform = output_transform_for_tile(shape, tile)?;
        if shape.d_transform.contains(GemmDTransform::ACCUMULATE) && !shape.n.is_multiple_of(32) {
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
            group_size: shape.b_group_size.unwrap_or(0),
            bits,
            output_transform,
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

    pub fn fuses_rht(&self) -> bool {
        self.output_transform.contains(GemmDTransform::RHT)
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

fn output_transform_for_tile(
    shape: &MatmulShape,
    tile: policy::GemvTile,
) -> Option<GemmDTransform> {
    let transform = shape.d_transform;
    if !transform.contains(GemmDTransform::RHT) {
        return Some(transform);
    }
    if !shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
        return None;
    }

    let output_rows = tile.output_row_tile();
    let can_fuse = tile.k_split == 1
        && output_rows >= HADAMARD_TRANSFORM_BLOCK_SIZE
        && output_rows.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE);
    if can_fuse {
        Some(transform)
    } else {
        Some(transform.difference(GemmDTransform::RHT | GemmDTransform::BIAS))
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
        BufferMut, BufferRef,
        kernel::matmul::{MatmulA, MatmulArguments, MatmulError},
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

    pub fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        specialization: GemvSpecialization,
        output_work: &MatmulOutputWork,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            b,
            output,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;
        let (mut d, ops) = output.into_contiguous(n, "Gemv")?;
        let ab_scale = ops.ab_scale;
        let output_bias = ops.bias;
        let rht_factors = ops.rht_factors;
        let soft_cap = ops.soft_cap;
        let deferred_factors = rht_factors.filter(|_| !specialization.fuses_rht());
        let (gemv_bias, gemv_rht_factors) = if deferred_factors.is_some() {
            (None, None)
        } else {
            (output_bias, rht_factors)
        };
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

        let a = a.subrange(a_offset..);
        let (scales, biases, zero_points, scale_strides, zero_point_strides) =
            b.quantized().map_or((None, None, None, Default::default(), Default::default()), |quantized| {
                (
                    Some(quantized.scales),
                    quantized.biases(),
                    quantized.zero_points(),
                    quantized.params.scale_strides(),
                    quantized.zero_point_strides(),
                )
            });
        let output_group_count = n.div_ceil(specialization.output_row_tile());
        let context = command_buffer.context();
        let pipeline = self.get_or_create(context, specialization)?;
        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => pipeline.encode(
                weights,
                scales,
                zero_points,
                biases,
                a,
                d.reborrow(),
                gemv_bias,
                gemv_rht_factors,
                gather_indices,
                k,
                n,
                m,
                ab_scale,
                output_group_count,
                scale_strides.output_stride,
                scale_strides.group_stride,
                zero_point_strides.output_stride,
                zero_point_strides.group_stride,
                soft_cap,
                command_buffer,
            ),
            MatmulB::Quantized(quantized) => pipeline.encode(
                quantized.codes,
                scales,
                zero_points,
                biases,
                a,
                d.reborrow(),
                gemv_bias,
                gemv_rht_factors,
                gather_indices,
                k,
                n,
                m,
                ab_scale,
                output_group_count,
                scale_strides.output_stride,
                scale_strides.group_stride,
                zero_point_strides.output_stride,
                zero_point_strides.group_stride,
                soft_cap,
                command_buffer,
            ),
            MatmulB::Trellis {
                ..
            } => {
                return Err(MatmulError::UnsupportedLayout {
                    path: "Gemv",
                });
            },
        }
        .map_err(MatmulError::BackendError)?;

        if let Some(factors) = deferred_factors {
            output_work.apply(d, factors, output_bias, m, n, command_buffer).map_err(MatmulError::BackendError)?;
        }

        Ok(())
    }
}
