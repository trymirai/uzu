use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::super::MTLContext;
use crate::{DataType, backends::metal::MTLError};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedMatmulError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Invalid dimensions: M={m}, N={n}, K={k}")]
    InvalidDimensions {
        m: usize,
        n: usize,
        k: usize,
    },
}

pub struct QuantizedMatmulKernel {
    pipeline: MTLComputePipelineState,
    kind: KernelKind,
}

#[derive(Debug)]
pub struct QuantizedMatmulArguments<'a> {
    pub a_buffer: &'a MTLBuffer, // Input A (float)
    pub b_buffer: &'a MTLBuffer, // Input B (quantized)
    pub scales_buffer: &'a MTLBuffer,
    pub zero_points_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub m: i32,
    pub n: i32,
    pub k: i32,
}

impl QuantizedMatmulKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        kernel_name: &str,
    ) -> Result<Self, QuantizedMatmulError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16) {
            return Err(QuantizedMatmulError::UnsupportedDataType(data_type));
        }

        let pipeline = mtl_context
            .compute_pipeline_state(kernel_name, None)
            .map_err(QuantizedMatmulError::MetalError)?;

        let kind = if kernel_name.starts_with("qmv") {
            KernelKind::Qmv
        } else if kernel_name.starts_with("qvm") {
            KernelKind::Qvm
        } else {
            KernelKind::Qmm
        };

        Ok(Self {
            pipeline,
            kind,
        })
    }

    #[allow(dead_code)]
    fn kernel_name_for_config(data_type: DataType) -> String {
        let type_suffix = match data_type {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            _ => unreachable!(),
        };

        format!("qmm_t_{}_gs_64_b_4_batch_0", type_suffix)
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: QuantizedMatmulArguments,
    ) -> Result<(), QuantizedMatmulError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(args.b_buffer), 0);
        encoder.set_buffer(1, Some(args.scales_buffer), 0);
        encoder.set_buffer(2, Some(args.zero_points_buffer), 0);
        encoder.set_buffer(3, Some(args.a_buffer), 0);
        encoder.set_buffer(4, Some(args.output_buffer), 0);

        // Set constants
        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &args.k as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.n as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.m as *const i32 as *const _,
        );

        match self.kind {
            KernelKind::Qmv => {
                let bk = 32;
                let bn = 8;
                let n_tgp_y = (args.n + bn - 1) / bn; // ceiling division
                let threadgroups =
                    MTLSize::new(args.m as u64, n_tgp_y as u64, 1);
                let threads_per_threadgroup = MTLSize::new(bk as u64, 2, 1);
                encoder.dispatch_thread_groups(
                    threadgroups,
                    threads_per_threadgroup,
                );
            },
            KernelKind::Qvm => {
                let bk = 32;
                let bn = 64;
                let n_tgp_y = (args.n + bn - 1) / bn; // ceiling division for columns
                let threadgroups =
                    MTLSize::new(args.m as u64, n_tgp_y as u64, 1);
                let threads_per_threadgroup = MTLSize::new(bk as u64, 2, 1);
                encoder.dispatch_thread_groups(
                    threadgroups,
                    threads_per_threadgroup,
                );
            },
            KernelKind::Qmm => {
                let bm = 32;
                let bn = 32;
                let wm = 2;
                let wn = 2;
                let threads_per_threadgroup = MTLSize::new(32, wn, wm);
                let threadgroups = MTLSize::new(
                    (args.n as u64 + bn - 1) / bn,
                    (args.m as u64 + bm - 1) / bm,
                    1,
                );
                encoder.dispatch_thread_groups(
                    threadgroups,
                    threads_per_threadgroup,
                );
            },
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelKind {
    Qmm,
    Qmv,
    Qvm,
}

pub fn encode_quantized_matmul(
    mtl_context: &MTLContext,
    encoder: &ComputeCommandEncoderRef,
    kernel_data_type: DataType,
    group_size: usize,
    transpose: bool,
    args: QuantizedMatmulArguments,
) -> Result<(), QuantizedMatmulError> {
    let type_suffix = match kernel_data_type {
        DataType::F16 => "f16",
        DataType::BF16 => "bf16",
        other => return Err(QuantizedMatmulError::UnsupportedDataType(other)),
    };

    let kernel_name = match (type_suffix, group_size, transpose) {
        ("f16", 64, false) => "qmm_f16_g64_b4".to_string(),
        ("f16", 64, true) => "qmm_transposed_f16_g64_b4".to_string(),
        ("f16", 128, false) => "qmm_f16_g128_b4".to_string(),
        ("f16", 128, true) => "qmm_transposed_f16_g128_b4".to_string(),
        ("bf16", 64, false) => "qmm_bf16_g64_b4".to_string(),
        ("bf16", 64, true) => "qmm_transposed_bf16_g64_b4".to_string(),
        ("bf16", 128, false) => "qmm_bf16_g128_b4".to_string(),
        ("bf16", 128, true) => "qmm_transposed_bf16_g128_b4".to_string(),
        _ => {
            return Err(QuantizedMatmulError::UnsupportedGroupSize(group_size));
        },
    };

    let kernel = QuantizedMatmulKernel::new(
        mtl_context,
        kernel_data_type,
        &kernel_name,
    )?;
    kernel.encode(encoder, args)
}
