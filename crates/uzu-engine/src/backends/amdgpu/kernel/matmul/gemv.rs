//! GEMV dispatch, as in `backends/metal/kernel/matmul/gemv/kernel.rs` with the AMD tile policy.

use std::collections::{HashMap, hash_map::Entry};

use super::{
    MatmulOutputWork,
    policy::{self, FP_K_BLOCK, GemvTile},
};
use crate::{
    backends::{
        amdgpu::{Amdgpu, context::AmdgpuContext, error::AmdgpuError, kernel::GemvAmdgpuKernel},
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
        },
    },
    data_type::DataType,
};

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
    pub fn select(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Option<Self> {
        if !shape.b_transpose || !shape.a_full_precision {
            return None;
        }
        let is_quant = shape.is_quant();
        let bits = shape.b_bits.unwrap_or(0);
        let tile = if is_quant {
            let group = shape.b_group_size.unwrap_or(0);
            let quant_block = if bits == 4 {
                512
            } else {
                256
            };
            let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
            policy::multi_row_quantized_tile(
                shape.m,
                bits,
                group,
                shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant,
                shape.b_prologue == GemmBPrologueKind::ScaleSymmetricDequant,
                bf16_io,
                shape.k.is_multiple_of(quant_block),
                shape.gathered,
            )
            .filter(|tile| shape.n >= tile.rows_per_lane())
            .map_or_else(|| policy::quantized_tile(bits, group, shape.n, bf16_io), Some)?
        } else {
            // the kernel is compiled for f32 weights only with f32 activations and output
            if weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32)
            {
                return None;
            }
            policy::fp_tile(shape.m, shape.n, shape.k, shape.k.is_multiple_of(FP_K_BLOCK))
        };

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
        let block_size = if !is_quant {
            FP_K_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        Some(Self {
            b_prologue: shape.b_prologue,
            group_size: shape.b_group_size.unwrap_or(0),
            bits,
            output_transform,
            input_aligned: shape.k.is_multiple_of(block_size),
            k_split: tile.k_split,
            output_row_tile: tile.output_row_tile(),
            num_simdgroups: tile.num_simdgroups,
            input_row_tile: tile.input_row_tile,
            reduction_lanes: tile.reduction_lanes,
            group_lanes: tile.group_lanes,
            gathered: shape.gathered,
            signed_codes: shape.signed_codes,
            full_tile: shape.m.is_multiple_of(tile.input_row_tile) && shape.n.is_multiple_of(tile.output_row_tile()),
        })
    }

    fn fuses_rht(&self) -> bool {
        self.output_transform.contains(GemmDTransform::RHT)
    }

    fn create_kernel(
        &self,
        context: &AmdgpuContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<GemvAmdgpuKernel, AmdgpuError> {
        GemvAmdgpuKernel::new(
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
            self.output_row_tile,
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
    tile: GemvTile,
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

/// GEMV kernels created on first use.
pub struct GemvKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    kernels: HashMap<GemvSpecialization, GemvAmdgpuKernel>,
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
            kernels: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &AmdgpuContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvAmdgpuKernel, MatmulError<Amdgpu>> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = specialization
                    .create_kernel(context, self.weights_data_type, self.input_data_type, self.output_data_type)
                    .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Amdgpu,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferMut<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
        >,
        specialization: GemvSpecialization,
        output_work: &MatmulOutputWork,
        command_buffer: &mut <<Amdgpu as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), MatmulError<Amdgpu>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let soft_cap = arguments.d_transform.soft_cap;
        let deferred_factors = rht_factors.filter(|_| !specialization.fuses_rht());
        let (gemv_bias, gemv_rht_factors) = if deferred_factors.is_some() {
            (None, None)
        } else {
            (output_bias, rht_factors)
        };

        let MatmulArguments {
            a,
            b,
            mut d,
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
        let output_group_count = n.div_ceil(specialization.output_row_tile);
        let context = command_buffer.context();
        let kernel = self.get_or_create(context, specialization)?;
        let weights = match b {
            MatmulB::FullPrecision {
                b: weights,
            } => weights,
            MatmulB::Quantized(quantized) => quantized.codes,
        };
        kernel.encode(
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
        );

        if let Some(factors) = deferred_factors {
            output_work.apply(d, factors, output_bias, m, n, command_buffer);
        }

        Ok(())
    }
}
