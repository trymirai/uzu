use std::collections::{HashMap, hash_map::Entry};

use super::{GemmEngine, GemmPlan, selection::TRELLIS_K_STEP as K_STEP};
use crate::backends::{
    common::{
        BufferMut, BufferRef, CommandBufferEncoding,
        gpu_types::{
            GemmParams,
            gemm::{GemmAlignment, GemmTiling},
        },
        kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, TrellisFormat},
    },
    metal::{
        Metal,
        command_buffer::MetalCommandBufferEncoding,
        error::MetalError,
        kernel::{GemmTrellisMetalKernel, GemmTrellisReduceMetalKernel},
    },
};

#[derive(Default)]
pub(super) struct TrellisGemm {
    kernels: HashMap<(GemmTiling, bool, TrellisFormat, GemmAlignment), GemmTrellisMetalKernel>,
    reduce_kernel: Option<GemmTrellisReduceMetalKernel>,
}

impl TrellisGemm {
    pub(super) fn encode_plan(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        plan: GemmPlan,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let unsupported = || {
            MetalError::from(MatmulError::UnsupportedLayout {
                path: "GemmTrellis",
            })
        };
        let (
            MatmulA::Trellis {
                values: activations,
                column_group_sums,
                scales: activation_scales,
            },
            MatmulB::Trellis {
                codes,
                row_scales,
                codebook,
                format,
            },
        ) = (arguments.a, arguments.b)
        else {
            return Err(unsupported());
        };
        let (m, n, k) = (arguments.m, arguments.n, arguments.k);
        let output_stride = arguments.output.row_stride.unwrap_or(n);
        let (tiling, split_k) = (plan.tiling, plan.split_k);
        if (plan.engine == GemmEngine::Mxu) != tiling.is_mxu_variant()
            || !arguments.output.ops.mask().is_empty()
            || !arguments.b_transpose
            || arguments.b_leading_dimension.is_some()
            || output_stride < n
            || (split_k > 1 && !n.is_multiple_of(4))
        {
            return Err(unsupported());
        }

        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: k,
            leading_dimension_d: output_stride,
            threadgroups_per_row: n.div_ceil(tiling.block_n()),
            threadgroups_per_column: m.div_ceil(tiling.block_m()),
            aligned_inner_iterations: k / split_k / K_STEP,
            ..Default::default()
        };
        let alignment =
            GemmAlignment::new(m.is_multiple_of(tiling.block_m()), n.is_multiple_of(tiling.block_n()), true);
        let use_mxu = plan.engine == GemmEngine::Mxu;
        let key = (tiling, use_mxu, format, alignment);
        let kernel = match self.kernels.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(GemmTrellisMetalKernel::new(
                command_buffer.context(),
                tiling,
                use_mxu,
                alignment,
                format.vector_width,
                format.transition_bits,
                format.restart_columns.unwrap_or(0),
            )?),
        };
        macro_rules! encode_gemm {
            ($destination:expr) => {
                kernel.encode(
                    activations,
                    column_group_sums,
                    activation_scales,
                    codes,
                    row_scales,
                    codebook,
                    $destination,
                    std::slice::from_ref(&params),
                    params.threadgroups_per_row,
                    params.threadgroups_per_column,
                    split_k,
                    command_buffer,
                )
            };
        }
        if split_k == 1 {
            encode_gemm!(arguments.output.values);
            return Ok(());
        }

        let partial_count = split_k as usize * m as usize * n as usize;
        let mut partial_sums = command_buffer.allocate_scratch(partial_count * std::mem::size_of::<i32>())?;
        encode_gemm!(&mut partial_sums);
        let reduce_kernel = match &mut self.reduce_kernel {
            Some(kernel) => kernel,
            empty => empty.insert(GemmTrellisReduceMetalKernel::new(command_buffer.context())?),
        };
        reduce_kernel.encode(
            &partial_sums,
            column_group_sums,
            activation_scales,
            row_scales,
            codebook,
            arguments.output.values,
            m,
            n,
            split_k,
            output_stride,
            command_buffer,
        );
        Ok(())
    }
}
