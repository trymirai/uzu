use std::collections::{HashMap, hash_map::Entry};

use crate::backends::{
    common::{
        BufferMut, BufferRef, CommandBufferEncoding,
        kernel::matmul::{MatmulA, MatmulArguments, MatmulB, TRELLIS_CODES_TAIL_BYTES, TrellisFormat},
    },
    metal::{
        Metal,
        command_buffer::MetalCommandBufferEncoding,
        gpu_types::{
            gemm::{GemmAlignment, GemmTiling, TRELLIS_DECODE_MAX_BATCH},
            matmul::GemmParams,
        },
        kernel::{
            GemmTrellisDecodeMetalKernel, GemmTrellisMetalKernel, GemmTrellisReduceMetalKernel,
            matmul::{
                MatmulMetalError,
                gemm::{GemmEngine, GemmPlan, selection::TRELLIS_K_STEP as K_STEP},
            },
        },
    },
};

const LANE_COLUMNS: u32 = 16;
const MAX_CENTERED_WEIGHT: u32 = 57;
const MAX_ACTIVATION: u32 = 128;
const DECODE_MAX_EXACT_K: u32 =
    ((1 << 24) - 1) / (LANE_COLUMNS * MAX_CENTERED_WEIGHT * MAX_ACTIVATION) * 32 * LANE_COLUMNS;

#[derive(Default)]
pub(super) struct TrellisGemm {
    kernels: HashMap<(GemmTiling, TrellisFormat, GemmAlignment), GemmTrellisMetalKernel>,
    decode_kernels: HashMap<(u32, TrellisFormat), GemmTrellisDecodeMetalKernel>,
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
    ) -> Result<(), MatmulMetalError> {
        let unsupported = || MatmulMetalError::UnsupportedLayout {
            path: "GemmTrellis",
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
        if !arguments.output.ops.mask().is_empty()
            || !arguments.b_transpose
            || arguments.b_leading_dimension.is_some()
            || output_stride < n
            || (split_k > 1 && !n.is_multiple_of(4))
            || codes.size() < (n * format.code_row_bytes(k)) as usize + TRELLIS_CODES_TAIL_BYTES
        {
            return Err(unsupported());
        }

        if plan.engine == GemmEngine::Simdgroup
            && (1..=TRELLIS_DECODE_MAX_BATCH).contains(&m)
            && k <= DECODE_MAX_EXACT_K
        {
            let kernel = match self.decode_kernels.entry((m, format)) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(GemmTrellisDecodeMetalKernel::new(
                    command_buffer.context(),
                    m,
                    format.vector_width,
                    format.transition_bits,
                    format.restart_columns.unwrap_or(0),
                )?),
            };
            kernel.encode(
                activations,
                column_group_sums,
                activation_scales,
                codes,
                row_scales,
                codebook,
                arguments.output.values,
                k,
                n,
                output_stride,
                command_buffer,
            );
            return Ok(());
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
        let key = (tiling, format, alignment);
        let kernel = match self.kernels.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(GemmTrellisMetalKernel::new(
                command_buffer.context(),
                tiling,
                plan.engine == GemmEngine::Mxu,
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
