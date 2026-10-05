//! GEMM through `gemm.metal`, as in `backends/metal/kernel/matmul/gemm/kernel.rs`, reduced to what the
//! AMD backend compiles: the simdgroup engine (no MXU, no int8 activations, no Morton order) on the
//! tilings in build/amdgpu's allowlist, without split-K. Quantized weights are dequantized once per
//! tile and shared by all rows of the M block, unlike GEMV which dequantizes per input row.

use std::collections::{HashMap, hash_map::Entry};

use crate::{
    backends::{
        amdgpu::{Amdgpu, context::AmdgpuContext, error::AmdgpuError, kernel::GemmAmdgpuKernel},
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
            gpu_types::{
                GemmParams,
                gemm::{GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
            },
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
        },
    },
    data_type::DataType,
};

type Encoding = <<Amdgpu as Backend>::CommandBuffer as CommandBuffer>::Encoding;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GemmSpecialization {
    tiling: GemmTiling,
    transpose_b: bool,
    b_prologue: GemmBPrologueKind,
    bits: u32,
    group_size: u32,
    output_transform: GemmDTransform,
    alignment: GemmAlignment,
    signed_codes: bool,
}

/// Tiling for an AMD GEMM, or `None` when the shape needs a variant the AMD build does not compile:
/// bf16 only, and rows of A and B 16-byte aligned (the staged loaders copy them with b128 accesses).
pub fn select_tiling(
    shape: &MatmulShape,
    data_types: [DataType; 3],
) -> Option<GemmTiling> {
    if !shape.a_full_precision || shape.gathered || data_types.iter().any(|&data_type| data_type != DataType::BF16) {
        return None;
    }
    let b_row = if shape.b_transpose || shape.is_quant() {
        shape.k
    } else {
        shape.n
    };
    if !shape.k.is_multiple_of(8)
        || !b_row.is_multiple_of(8)
        || shape.b_leading_dimension.is_some_and(|ld| !ld.is_multiple_of(8))
    {
        return None;
    }
    let tiling = if shape.m < 64 || !shape.n.is_multiple_of(64) {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    } else {
        GemmTiling::Tile64x64x32_Simdgroups2x2
    };
    if shape.is_quant() {
        let group_size = shape.b_group_size?;
        let compiled_group = matches!(group_size, 32 | 64);
        let layout = shape.b_transpose && shape.b_leading_dimension.is_none();
        if !compiled_group || !layout || tiling.simdgroup_block_k() > group_size {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::ACCUMULATE) {
            return None;
        }
    }
    Some(tiling)
}

pub struct GemmKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmAmdgpuKernel>,
}

impl GemmKernel {
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
        specialization: GemmSpecialization,
    ) -> Result<&GemmAmdgpuKernel, AmdgpuError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmAmdgpuKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    false,
                    specialization.b_prologue,
                    specialization.bits,
                    specialization.group_size,
                    GemmAPrologueKind::FullPrecision,
                    0,
                    specialization.output_transform,
                    specialization.alignment,
                    specialization.signed_codes,
                )?;
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
        tiling: GemmTiling,
        command_buffer: &mut Encoding,
    ) -> Result<(), AmdgpuError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let MatmulArguments {
            a,
            b,
            mut d,
            d_transform,
            ..
        } = arguments;
        let MatmulA::FullPrecision {
            values: a,
            offset: a_offset,
        } = a
        else {
            return Err(MatmulError::<Amdgpu>::IncompatibleA {
                path: "Gemm",
                reason: "int8 activations need the MXU path",
            }
            .into());
        };
        let a = a.subrange(a_offset..);
        let (m, n, k) = (shape.m, shape.n, shape.k);

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
        let weights = match b {
            MatmulB::FullPrecision {
                b: weights,
            } => weights,
            MatmulB::Quantized(quantized) => quantized.codes,
        };

        let alignment = GemmAlignment::new(
            m.is_multiple_of(tiling.block_m()),
            n.is_multiple_of(tiling.block_n()),
            k.is_multiple_of(tiling.block_k()),
        );
        let threadgroups_per_row = n.div_ceil(tiling.block_n());
        let threadgroups_per_column = m.div_ceil(tiling.block_m());
        let leading_dimension_b = if shape.is_quant() {
            k
        } else {
            shape.b_leading_dimension.unwrap_or(if shape.b_transpose {
                k
            } else {
                n
            })
        };
        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: k,
            leading_dimension_b,
            leading_dimension_d: n,
            threadgroups_per_row,
            threadgroups_per_column,
            aligned_inner_iterations: k / tiling.block_k(),
            use_morton: false,
            ab_scale: d_transform.ab_scale,
            scale_output_stride: scale_strides.output_stride,
            scale_group_stride: scale_strides.group_stride,
            zero_point_output_stride: zero_point_strides.output_stride,
            zero_point_group_stride: zero_point_strides.group_stride,
        };
        let specialization = GemmSpecialization {
            tiling,
            transpose_b: shape.b_transpose,
            b_prologue: shape.b_prologue,
            bits: shape.b_bits.unwrap_or(0),
            group_size: shape.b_group_size.unwrap_or(0),
            output_transform: d_transform.mask(),
            alignment,
            signed_codes: shape.signed_codes,
        };
        let kernel = self.get_or_create(command_buffer.context(), specialization)?;
        kernel.encode(
            Some(a),
            weights,
            d.reborrow(),
            scales,
            biases,
            zero_points,
            d_transform.bias,
            d_transform.rht_factors,
            None::<&<Amdgpu as Backend>::GlobalBuffer>,
            None::<&<Amdgpu as Backend>::GlobalBuffer>,
            None::<&<Amdgpu as Backend>::GlobalBuffer>,
            std::slice::from_ref(&params),
            threadgroups_per_row,
            threadgroups_per_column,
            1,
            command_buffer,
        );
        Ok(())
    }
}
