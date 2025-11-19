use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::super::MTLContext;
use crate::{DataType, backends::metal::MTLError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// GroupQuantized style: uses zero-points
    ZeroPoint,
    /// MLX style: uses pre-computed biases
    Mlx,
}

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
    bm: u64,
    bn: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulArguments<'a> {
    pub a_buffer: &'a MTLBuffer, // Input A (float)
    pub b_buffer: &'a MTLBuffer, // Input B (quantized)
    pub scales_buffer: &'a MTLBuffer,
    /// For ZeroPoint quantization: packed zero-points
    /// For MLX quantization: pre-computed biases (deq_biases)
    pub zero_points_or_biases_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub quantization_type: QuantizationType,
}

fn dtype_suffix(data_type: DataType) -> Option<&'static str> {
    match data_type {
        DataType::F16 => Some("f16"),
        DataType::BF16 => Some("bf16"),
        DataType::F32 => Some("f32"),
        _ => None,
    }
}

fn base_qmm_kernel_name(
    type_suffix: &str,
    group_size: usize,
    transpose_infix: &str,
) -> Result<String, QuantizedMatmulError> {
    let kernel_name = match (type_suffix, group_size) {
        ("f16", 32) => format!("qmm{}_f16_g32_b4", transpose_infix),
        ("f16", 64) => format!("qmm{}_f16_g64_b4", transpose_infix),
        ("f16", 128) => format!("qmm{}_f16_g128_b4", transpose_infix),
        ("bf16", 32) => format!("qmm{}_bf16_g32_b4", transpose_infix),
        ("bf16", 64) => format!("qmm{}_bf16_g64_b4", transpose_infix),
        ("bf16", 128) => format!("qmm{}_bf16_g128_b4", transpose_infix),
        ("f32", 32) => format!("qmm{}_f32_g32_b4", transpose_infix),
        ("f32", 64) => format!("qmm{}_f32_g64_b4", transpose_infix),
        ("f32", 128) => format!("qmm{}_f32_g128_b4", transpose_infix),
        _ => {
            return Err(QuantizedMatmulError::UnsupportedGroupSize(group_size));
        },
    };
    Ok(kernel_name)
}

    fn resolve_qmm_kernel_name(
    type_suffix: &str,
    group_size: usize,
    transpose: bool,
    n: i32,
    k: i32,
) -> Result<String, QuantizedMatmulError> {
    let transpose_infix = if transpose {
        "_transposed"
    } else {
        ""
    };
    let mut kernel_name =
        base_qmm_kernel_name(type_suffix, group_size, transpose_infix)?;
    if transpose {
        if n % 32 != 0 {
            kernel_name.push_str("_unaligned");
        } else if type_suffix == "bf16" && group_size == 128 {
            // Use optimized 64x64 kernel for BF16 G128 aligned
            kernel_name.push_str("_64x64");
        }
    } else if k % 32 == 0 {
        kernel_name.push_str("_alignedk");
    }
    Ok(kernel_name)
}

impl QuantizedMatmulKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        kernel_name: &str,
        quantization_type: QuantizationType,
    ) -> Result<Self, QuantizedMatmulError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(QuantizedMatmulError::UnsupportedDataType(data_type));
        }

        // Create function constants for quantization type
        let function_constants = metal::FunctionConstantValues::new();
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);
        function_constants.set_constant_value_at_index(
            &use_mlx_quant as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            40,
        );

        let (pipeline, _) = mtl_context
            .compute_pipeline_state_with_reflection(
                kernel_name,
                Some(&function_constants),
            )
            .map_err(QuantizedMatmulError::MetalError)?;

        let kind = if kernel_name.starts_with("qmv") {
            KernelKind::Qmv
        } else if kernel_name.starts_with("qvm") {
            KernelKind::Qvm
        } else {
            KernelKind::Qmm
        };

        let (bm, bn) = if kernel_name.contains("_64x64") {
            (64, 64)
        } else if kernel_name.contains("_64x128") {
            (64, 128)
        } else if kernel_name.contains("_128x64") {
            (128, 64)
        } else {
            (32, 32)
        };

        Ok(Self {
            pipeline,
            kind,
            bm,
            bn,
        })
    }

    #[allow(dead_code)]
    fn kernel_name_for_config(data_type: DataType) -> String {
        let type_suffix = match data_type {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
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
        encoder.set_buffer(2, Some(args.zero_points_or_biases_buffer), 0);
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
                let bm = self.bm;
                let bn = self.bn;
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

pub fn select_qmm_kernel_name(
    data_type: DataType,
    group_size: usize,
    transpose: bool,
    n: usize,
    k: usize,
) -> Result<String, QuantizedMatmulError> {
    let type_suffix = dtype_suffix(data_type)
        .ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;
    resolve_qmm_kernel_name(
        type_suffix,
        group_size,
        transpose,
        n as i32,
        k as i32,
    )
}

pub fn quantized_kernel_names(
    data_type: DataType,
    group_size: usize,
    n: usize,
    k: usize,
) -> Option<(String, String)> {
    let type_suffix = dtype_suffix(data_type)?;
    if !matches!(group_size, 32 | 64 | 128) {
        return None;
    }

    let mm = select_qmm_kernel_name(data_type, group_size, true, n, k).ok()?;
    let mv = format!("qmv_{type_suffix}_g{group_size}_b4_fast");
    Some((mm, mv))
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
    let kernel_name = select_qmm_kernel_name(
        kernel_data_type,
        group_size,
        transpose,
        args.n as usize,
        args.k as usize,
    )?;

    let kernel = QuantizedMatmulKernel::new(
        mtl_context,
        kernel_data_type,
        &kernel_name,
        args.quantization_type,
    )?;
    kernel.encode(encoder, args)
}
