use metal::MTLGPUFamily;

use crate::{
    backends::{
        common::{
            CommandBufferEncoding,
            gpu_types::{
                gemm::{GemmBPrologueKind, GemmDTransform},
                hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            },
            kernel::matmul::{MatmulB, MatmulShape},
        },
        metal::{
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::gemm::GemmBLayout,
            kernel::{
                GemvMetalKernel, GemvMxuTiledMetalKernel,
                matmul::{
                    MatmulOutputWork,
                    gemv::policy::{self, DEFAULT_RESULTS_PER_SIMDGROUP, FP_K_BLOCK},
                },
            },
        },
    },
    data_type::DataType,
};

const GEMV_MAX_BATCH: u32 = 8;
const MXU_TILED_KERNEL_MIN_K: u32 = 2048;

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
    gemm_b_layout: GemmBLayout,
    full_tile: bool,
    mxu_tiled_kernel: bool,
}

impl GemvSpecialization {
    pub fn select_shape(
        shape: &MatmulShape,
        gemm_b_layout: GemmBLayout,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        gpu_core_count: u32,
        apple_gpu_family: MTLGPUFamily,
    ) -> Option<Self> {
        let is_quant = shape.is_quant();
        let bits = shape.b_bits.unwrap_or(0);
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let mxu_tiled_kernel = gemm_b_layout == GemmBLayout::MxuTiled
            && shape.b_prologue == GemmBPrologueKind::ScaleSymmetricDequant
            && shape.m == 1
            && !shape.gathered
            && bf16_io
            && weights_data_type == DataType::BF16
            && shape.k >= MXU_TILED_KERNEL_MIN_K
            && shape.n >= 32 * gpu_core_count;
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
            .map(|tile| match gemm_b_layout {
                GemmBLayout::MxuTiled if mxu_tiled_kernel => {
                    policy::GemvTile::quantized_output_tile(8, 32, 1, 32, tile.group_lanes)
                },
                GemmBLayout::MxuTiled if tile.output_row_tile() < 16 => {
                    policy::GemvTile::quantized_output_tile(8, 16, 1, 32, tile.group_lanes)
                },
                _ => tile,
            })
        } else {
            if shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
                return None;
            }
            let input_aligned = shape.k.is_multiple_of(FP_K_BLOCK);
            Some(policy::fp_tile(gpu_core_count, apple_gpu_family, shape.m, shape.n, shape.k, input_aligned))
        };
        Self::select_tile(shape, gemm_b_layout, tile?).map(|specialization| Self {
            mxu_tiled_kernel,
            ..specialization
        })
    }

    pub fn select_tile(
        shape: &MatmulShape,
        gemm_b_layout: GemmBLayout,
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
        let bits = shape.b_bits.unwrap_or(0);
        if !is_quant {
            if shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
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
        // Gathered quantized rows cannot share one input tile, and batched tiles need whole K blocks.
        if is_quant && tile.input_row_tile > 1 && (shape.gathered || !input_aligned) {
            return None;
        }
        let output_transform = output_transform_for_tile(shape, tile)?;
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
            gemm_b_layout,
            full_tile: full_tile(shape, tile),
            mxu_tiled_kernel: false,
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
            self.gemm_b_layout,
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
        kernel::matmul::{MatmulA, MatmulArguments},
    },
    metal::{Metal, kernel::matmul::MatmulMetalError},
};

/// GEMV pipelines compiled on first use.
pub struct GemvKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
    mxu_tiled_pipelines: HashMap<GemvSpecialization, GemvMxuTiledMetalKernel>,
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
            mxu_tiled_pipelines: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulMetalError> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = specialization.create_pipeline(
                    context,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                )?;
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
    ) -> Result<(), MatmulMetalError> {
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
            return Err(MatmulMetalError::IncompatibleA {
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
        if specialization.mxu_tiled_kernel {
            let MatmulB::Quantized(quantized) = b else {
                return Err(MatmulMetalError::UnsupportedLayout {
                    path: "Gemv",
                });
            };
            let pipeline = match self.mxu_tiled_pipelines.entry(specialization) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(GemvMxuTiledMetalKernel::new(
                    context,
                    specialization.group_size,
                    specialization.output_transform,
                    specialization.signed_codes,
                )?),
            };
            pipeline.encode(
                quantized.codes,
                quantized.scales,
                a,
                d.reborrow(),
                gemv_bias,
                gemv_rht_factors,
                k,
                n,
                ab_scale,
                scale_strides.output_stride,
                scale_strides.group_stride,
                soft_cap,
                command_buffer,
            );
            return Ok(());
        }
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
                return Err(MatmulMetalError::UnsupportedLayout {
                    path: "Gemv",
                });
            },
        }

        if let Some(factors) = deferred_factors {
            output_work.apply(d, factors, output_bias, m, n, command_buffer);
        }

        Ok(())
    }
}
