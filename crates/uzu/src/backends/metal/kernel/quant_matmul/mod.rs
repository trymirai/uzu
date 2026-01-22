use std::{collections::HashMap, mem::size_of};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::super::MTLContext;
use crate::{DataType, backends::metal::MTLError, config::QuantizationMode};

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
    #[error("Unsupported bits: {0}")]
    UnsupportedBits(usize),
}

pub struct QuantizedMatmulKernel {
    pipelines: HashMap<KernelKind, (MTLComputePipelineState, u64, u64)>,
    output_dim: usize,
    weights_transposed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulArguments<'a> {
    pub a_buffer: &'a MTLBuffer,
    /// Byte offset into `a_buffer` (used for slicing the batch dimension).
    pub a_offset: u64,
    pub b_buffer: &'a MTLBuffer,
    pub scales_buffer: &'a MTLBuffer,
    pub zero_points_or_biases_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub batch: i32,
    pub input_dim: i32,
    pub output_dim: i32,
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
    bits: usize,
) -> Result<String, QuantizedMatmulError> {
    if bits != 4 && bits != 8 {
        return Err(QuantizedMatmulError::UnsupportedBits(bits));
    }

    let kernel_name = match (type_suffix, group_size) {
        ("f16", 32) => format!("qmm{}_f16_g32_b{}", transpose_infix, bits),
        ("f16", 64) => format!("qmm{}_f16_g64_b{}", transpose_infix, bits),
        ("f16", 128) => format!("qmm{}_f16_g128_b{}", transpose_infix, bits),
        ("bf16", 32) => format!("qmm{}_bf16_g32_b{}", transpose_infix, bits),
        ("bf16", 64) => format!("qmm{}_bf16_g64_b{}", transpose_infix, bits),
        ("bf16", 128) => format!("qmm{}_bf16_g128_b{}", transpose_infix, bits),
        ("f32", 32) => format!("qmm{}_f32_g32_b{}", transpose_infix, bits),
        ("f32", 64) => format!("qmm{}_f32_g64_b{}", transpose_infix, bits),
        ("f32", 128) => format!("qmm{}_f32_g128_b{}", transpose_infix, bits),
        _ => {
            return Err(QuantizedMatmulError::UnsupportedGroupSize(group_size));
        },
    };
    Ok(kernel_name)
}

impl QuantizedMatmulKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        group_size: usize,
        input_dim: usize,
        output_dim: usize,
        mode: QuantizationMode,
        quantization_type: QuantizationType,
        weights_transposed: bool,
    ) -> Result<Self, QuantizedMatmulError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(QuantizedMatmulError::UnsupportedDataType(data_type));
        }

        let function_constants = metal::FunctionConstantValues::new();
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);
        function_constants.set_constant_value_at_index(
            &use_mlx_quant as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            40,
        );

        let mut pipelines = HashMap::new();

        let kernel_name_mv = select_matrix_vector_kernel_name(
            data_type,
            group_size,
            weights_transposed,
            output_dim,
            input_dim,
            mode,
        )?;

        let cache_key_mv = format!("{}_mlx_{}", kernel_name_mv, use_mlx_quant);
        let (pipeline_mv, _) = mtl_context
            .compute_pipeline_state_with_reflection_cached(
                &cache_key_mv,
                &kernel_name_mv,
                Some(&function_constants),
            )
            .map_err(QuantizedMatmulError::MetalError)?;
        pipelines.insert(KernelKind::MatrixVector, (pipeline_mv, 32, 32));

        let kernel_name_mm = select_qmm_kernel_name(
            data_type,
            group_size,
            weights_transposed,
            output_dim,
            input_dim,
            mode,
        )?;

        let cache_key_mm = format!("{}_mlx_{}", kernel_name_mm, use_mlx_quant);
        let (pipeline_mm, _) = mtl_context
            .compute_pipeline_state_with_reflection_cached(
                &cache_key_mm,
                &kernel_name_mm,
                Some(&function_constants),
            )
            .map_err(QuantizedMatmulError::MetalError)?;

        let (bm, bn) = if kernel_name_mm.contains("_64x64") {
            (64, 64)
        } else if kernel_name_mm.contains("_64x128") {
            (64, 128)
        } else if kernel_name_mm.contains("_128x64") {
            (128, 64)
        } else {
            (32, 32)
        };
        pipelines.insert(KernelKind::MatrixMatrix, (pipeline_mm, bm, bn));

        Ok(Self {
            pipelines,
            output_dim,
            weights_transposed,
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
        let variant = self.select_variant(args.batch as usize);
        let (pipeline, bm, bn) =
            self.pipelines.get(&variant).ok_or_else(|| {
                QuantizedMatmulError::InvalidDimensions {
                    m: args.batch as usize,
                    n: args.output_dim as usize,
                    k: args.input_dim as usize,
                }
            })?;

        encoder.set_compute_pipeline_state(pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(args.b_buffer), 0);
        encoder.set_buffer(1, Some(args.scales_buffer), 0);
        encoder.set_buffer(2, Some(args.zero_points_or_biases_buffer), 0);
        encoder.set_buffer(3, Some(args.a_buffer), args.a_offset);
        encoder.set_buffer(4, Some(args.output_buffer), 0);

        let k: i32 = args.input_dim;
        let m = args.batch;
        let n = args.output_dim;

        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &k as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &n as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &m as *const i32 as *const _,
        );

        match variant {
            KernelKind::MatrixVector => {
                let bk = 32;
                let bn = if self.weights_transposed {
                    8
                } else {
                    64
                };
                let n_tgp_y = (n + bn - 1) / bn;
                let threadgroups = MTLSize::new(m as u64, n_tgp_y as u64, 1);
                let threads_per_threadgroup = MTLSize::new(bk as u64, 2, 1);
                encoder.dispatch_thread_groups(
                    threadgroups,
                    threads_per_threadgroup,
                );
            },
            KernelKind::MatrixMatrix => {
                let wm = 2;
                let wn = 2;
                let threads_per_threadgroup = MTLSize::new(32, wn, wm);
                let threadgroups = MTLSize::new(
                    (n as u64 + bn - 1) / bn,
                    (m as u64 + bm - 1) / bm,
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

    fn select_variant(
        &self,
        batch: usize,
    ) -> KernelKind {
        if batch < 32 || self.output_dim == 1 {
            KernelKind::MatrixVector
        } else {
            KernelKind::MatrixMatrix
        }
    }
}

fn select_matrix_vector_kernel_name(
    data_type: DataType,
    group_size: usize,
    weights_transposed: bool,
    output_dim: usize,
    input_dim: usize,
    mode: QuantizationMode,
) -> Result<String, QuantizedMatmulError> {
    let type_suffix = dtype_suffix(data_type)
        .ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;
    let bits = match mode {
        QuantizationMode::UInt4 => 4,
        QuantizationMode::UInt8 | QuantizationMode::Int8 => 8,
    };

    if weights_transposed {
        let mut name = format!("qmv_{}_g{}_b{}", type_suffix, group_size, bits);
        if output_dim % 8 == 0 && input_dim % 512 == 0 {
            name.push_str("_fast");
        }
        return Ok(name);
    }

    Ok(format!("qvm_{}_g{}_b{}", type_suffix, group_size, bits))
}

fn select_qmm_kernel_name(
    data_type: DataType,
    group_size: usize,
    weights_transposed: bool,
    output_dim: usize,
    input_dim: usize,
    mode: QuantizationMode,
) -> Result<String, QuantizedMatmulError> {
    let type_suffix = dtype_suffix(data_type)
        .ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;
    let bits = match mode {
        QuantizationMode::UInt4 => 4,
        QuantizationMode::Int8 | QuantizationMode::UInt8 => 8,
    };

    let transpose_infix = if weights_transposed {
        "_transposed"
    } else {
        ""
    };
    let mut kernel_name =
        base_qmm_kernel_name(type_suffix, group_size, transpose_infix, bits)?;
    if weights_transposed {
        if output_dim % 32 != 0 {
            kernel_name.push_str("_unaligned");
        } else if type_suffix == "bf16"
            && (group_size == 128 || group_size == 64)
            && (bits == 4 || bits == 8)
        {
            kernel_name.push_str("_64x64");
        }
    } else if input_dim % 32 == 0 {
        kernel_name.push_str("_alignedk");
    }
    Ok(kernel_name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KernelKind {
    MatrixMatrix,
    MatrixVector,
}

/// Arguments for MLP fused quantized GEMV
#[derive(Debug)]
pub struct MlpFusedQmvArguments<'a> {
    /// Weight matrix [2*hidden_dim, input_dim] (up and gate concatenated, quantized)
    pub weights: &'a MTLBuffer,
    /// Scales buffer
    pub scales: &'a MTLBuffer,
    /// Zero points or biases buffer (depends on quantization type)
    pub zero_points_or_biases: &'a MTLBuffer,
    /// Input vector [input_dim]
    pub input: &'a MTLBuffer,
    /// Input byte offset
    pub input_offset: u64,
    /// Output vector [hidden_dim]
    pub output: &'a MTLBuffer,
    /// Input dimension (K)
    pub input_dim: i32,
    /// Hidden dimension (output size, half of weight rows)
    pub hidden_dim: i32,
    /// Batch count
    pub batch_count: i32,
}

/// MLP Fused Quantized GEMV Kernel
/// Computes paired up/gate projections with fused activation: out = up * activation(gate)
pub struct MlpFusedQmvKernel {
    pipeline: MTLComputePipelineState,
}

impl MlpFusedQmvKernel {
    pub fn new(
        context: &MTLContext,
        data_type: DataType,
        group_size: usize,
        mode: QuantizationMode,
        quantization_type: QuantizationType,
    ) -> Result<Self, QuantizedMatmulError> {
        let type_suffix = dtype_suffix(data_type)
            .ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;

        let bits = match mode {
            QuantizationMode::UInt4 => 4,
            QuantizationMode::UInt8 | QuantizationMode::Int8 => 8,
        };

        if bits != 4 {
            return Err(QuantizedMatmulError::UnsupportedBits(bits));
        }

        let kernel_name =
            format!("qmv_mlp_fused_{}_g{}_b{}", type_suffix, group_size, bits);

        let function_constants = metal::FunctionConstantValues::new();
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);
        function_constants.set_constant_value_at_index(
            &use_mlx_quant as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            40,
        );

        let pipeline = context
            .compute_pipeline_state(&kernel_name, Some(&function_constants))
            .map_err(QuantizedMatmulError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: &MlpFusedQmvArguments,
    ) -> Result<(), QuantizedMatmulError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(0, Some(args.weights), 0);
        encoder.set_buffer(1, Some(args.scales), 0);
        encoder.set_buffer(2, Some(args.zero_points_or_biases), 0);
        encoder.set_buffer(3, Some(args.input), args.input_offset);
        encoder.set_buffer(4, Some(args.output), 0);

        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &args.input_dim as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.hidden_dim as *const i32 as *const _,
        );

        // Dispatch: one threadgroup per 8 output rows (num_simdgroups * results_per_simdgroup)
        let rows_per_threadgroup = 8;
        let n_tgp_y = ((args.hidden_dim + rows_per_threadgroup - 1)
            / rows_per_threadgroup) as u64;
        let threadgroups =
            MTLSize::new(args.batch_count.max(1) as u64, n_tgp_y, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 1); // 2 simdgroups, 32 threads each

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}

/// Arguments for MLP fused quantized GEMM (prefill)
#[derive(Debug)]
pub struct MlpFusedQmmArguments<'a> {
    /// Weight matrix [K, 2*hidden_dim] quantized (up and gate concatenated)
    pub weights: &'a MTLBuffer,
    /// Scales buffer
    pub scales: &'a MTLBuffer,
    /// Zero points or biases buffer (depends on quantization type)
    pub zero_points_or_biases: &'a MTLBuffer,
    /// Input activations [M, K]
    pub input: &'a MTLBuffer,
    /// Input byte offset
    pub input_offset: u64,
    /// Output [M, hidden_dim]
    pub output: &'a MTLBuffer,
    /// Batch size (M)
    pub batch: i32,
    /// Input dimension (K)
    pub input_dim: i32,
    /// Hidden dimension (output size, half of weight columns)
    pub hidden_dim: i32,
}

/// MLP Fused Quantized GEMM Kernel for prefill path
/// Computes paired up/gate projections with fused activation: out = up * activation(gate)
pub struct MlpFusedQmmKernel {
    pipeline: MTLComputePipelineState,
}

impl MlpFusedQmmKernel {
    pub fn new(
        context: &MTLContext,
        data_type: DataType,
        group_size: usize,
        mode: QuantizationMode,
        quantization_type: QuantizationType,
    ) -> Result<Self, QuantizedMatmulError> {
        let type_suffix = dtype_suffix(data_type)
            .ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;

        let bits = match mode {
            QuantizationMode::UInt4 => 4,
            QuantizationMode::UInt8 | QuantizationMode::Int8 => 8,
        };

        if bits != 4 {
            return Err(QuantizedMatmulError::UnsupportedBits(bits));
        }

        let kernel_name =
            format!("qmm_mlp_fused_{}_g{}_b{}", type_suffix, group_size, bits);

        let function_constants = metal::FunctionConstantValues::new();
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);
        function_constants.set_constant_value_at_index(
            &use_mlx_quant as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            40,
        );

        let pipeline = context
            .compute_pipeline_state(&kernel_name, Some(&function_constants))
            .map_err(QuantizedMatmulError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: &MlpFusedQmmArguments,
    ) -> Result<(), QuantizedMatmulError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(0, Some(args.weights), 0);
        encoder.set_buffer(1, Some(args.scales), 0);
        encoder.set_buffer(2, Some(args.zero_points_or_biases), 0);
        encoder.set_buffer(3, Some(args.input), args.input_offset);
        encoder.set_buffer(4, Some(args.output), 0);

        encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &args.input_dim as *const i32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.hidden_dim as *const i32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.batch as *const i32 as *const _,
        );

        // Dispatch: BM=32, BN=32
        let bm = 32;
        let bn = 32;
        let tiles_m = ((args.batch + bm - 1) / bm) as u64;
        let tiles_n = ((args.hidden_dim + bn - 1) / bn) as u64;
        let threadgroups = MTLSize::new(tiles_n, tiles_m, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 2); // WM=2, WN=2, 32 threads

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}
