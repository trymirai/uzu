use std::collections::HashMap;

use metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLFunctionConstantValues, MTLSize};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    DataType,
    backends::{
        common::kernel::{
            QmmBf16G128B4AlignedKKernel as QmmBf16G128B4AlignedKKernelTrait,
            QmmBf16G128B4AlignedKMlxKernel as QmmBf16G128B4AlignedKMlxKernelTrait,
            QmmKernel as QmmKernelTrait, QmmTransposedKernel as QmmTransposedKernelTrait,
            QmvFastKernel as QmvFastKernelTrait,
            QmvKernel as QmvKernelTrait, QvmKernel as QvmKernelTrait,
            matmul::{
                QuantizedMatmulArguments as GenericQuantizedMatmulArguments, QuantizedMatmulConfiguration,
                QuantizedMatmulKernel as QuantizedMatmulKernelTrait, QuantizedMatmulType,
            },
        },
        metal::{
            FunctionConstantValuesSetValue, Metal, MetalContext, MetalError, metal_extensions::ComputeEncoderSetValue,
        },
    },
    config::QuantizationMode,
};
use super::dsl::{
    QmmBf16G128B4AlignedKMetalKernel, QmmBf16G128B4AlignedKMlxMetalKernel, QmmMetalKernel, QmmTransposedMetalKernel,
    QmvFastMetalKernel, QmvMetalKernel, QvmMetalKernel,
};

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
    MetalError(#[from] MetalError),
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
    matrix_vector_kernel: MatrixVectorKernel,
    matrix_matrix_kernel: MatrixMatrixKernel,
    output_dim: usize,
}

enum MatrixVectorKernel {
    QmvFast(QmvFastMetalKernel),
    Qmv(QmvMetalKernel),
    Qvm(QvmMetalKernel),
}

enum MatrixMatrixKernel {
    QmmBf16G128B4AlignedK(QmmBf16G128B4AlignedKMetalKernel),
    QmmBf16G128B4AlignedKMlx(QmmBf16G128B4AlignedKMlxMetalKernel),
    Qmm(QmmMetalKernel),
    QmmTransposed(QmmTransposedMetalKernel),
    Legacy((Retained<ProtocolObject<dyn MTLComputePipelineState>>, u64, u64)),
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulArguments<'a> {
    pub a_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset into `a_buffer` (used for slicing the batch dimension).
    pub a_offset: u64,
    pub b_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    pub scales_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    pub zero_points_or_biases_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    pub output_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
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
        mtl_context: &MetalContext,
        data_type: DataType,
        group_size: usize,
        input_dim: usize,
        output_dim: usize,
        mode: QuantizationMode,
        quantization_type: QuantizationType,
        weights_transposed: bool,
    ) -> Result<Self, QuantizedMatmulError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(QuantizedMatmulError::UnsupportedDataType(data_type));
        }

        let group_size_u32 =
            u32::try_from(group_size).map_err(|_| QuantizedMatmulError::UnsupportedGroupSize(group_size))?;
        let bits = match mode {
            QuantizationMode::UInt4 => 4_u32,
            QuantizationMode::UInt8 | QuantizationMode::Int8 => 8_u32,
        };
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);

        let matrix_vector_kernel = if weights_transposed {
            if output_dim % 8 == 0 && input_dim % 512 == 0 {
                MatrixVectorKernel::QmvFast(
                    <QmvFastMetalKernel as QmvFastKernelTrait>::new(
                        mtl_context,
                        data_type,
                        use_mlx_quant,
                        !use_mlx_quant,
                        group_size_u32,
                        bits,
                    )
                    .map_err(QuantizedMatmulError::MetalError)?,
                )
            } else {
                MatrixVectorKernel::Qmv(
                    <QmvMetalKernel as QmvKernelTrait>::new(
                        mtl_context,
                        data_type,
                        use_mlx_quant,
                        !use_mlx_quant,
                        group_size_u32,
                        bits,
                    )
                    .map_err(QuantizedMatmulError::MetalError)?,
                )
            }
        } else {
            MatrixVectorKernel::Qvm(
                <QvmMetalKernel as QvmKernelTrait>::new(
                    mtl_context,
                    data_type,
                    use_mlx_quant,
                    !use_mlx_quant,
                    group_size_u32,
                    bits,
                )
                .map_err(QuantizedMatmulError::MetalError)?,
            )
        };

        let kernel_name_mm =
            select_qmm_kernel_name(data_type, group_size, weights_transposed, output_dim, input_dim, mode)?;

        let (bm, bn) = if kernel_name_mm.contains("_64x64") {
            (64, 64)
        } else if kernel_name_mm.contains("_64x128") {
            (64, 128)
        } else if kernel_name_mm.contains("_128x64") {
            (128, 64)
        } else {
            (32, 32)
        };

        // Use a dedicated DSL flavor for the known regressed BF16 4-bit prefill shape family.
        let use_qmm_bf16_g128_b4_alignedk_flavour = !weights_transposed
            && matches!(data_type, DataType::BF16)
            && matches!(mode, QuantizationMode::UInt4)
            && group_size == 128
            && input_dim == 4096
            && output_dim >= 14336;

        let build_legacy_mm = || -> Result<MatrixMatrixKernel, QuantizedMatmulError> {
            let function_constants = MTLFunctionConstantValues::new();
            function_constants.set_value(&use_mlx_quant, 40);
            let cache_key_mm = format!("{}_mlx_{}", kernel_name_mm, use_mlx_quant);
            let pipeline_mm = mtl_context
                .compute_pipeline_state_cached(&cache_key_mm, &kernel_name_mm, Some(&function_constants))
                .map_err(QuantizedMatmulError::MetalError)?;
            Ok(MatrixMatrixKernel::Legacy((pipeline_mm, bm, bn)))
        };

        let matrix_matrix_kernel = if bm == 32 && bn == 32 {
            if use_qmm_bf16_g128_b4_alignedk_flavour {
                if use_mlx_quant {
                    MatrixMatrixKernel::QmmBf16G128B4AlignedKMlx(
                        <QmmBf16G128B4AlignedKMlxMetalKernel as QmmBf16G128B4AlignedKMlxKernelTrait>::new(
                            mtl_context,
                        )
                        .map_err(QuantizedMatmulError::MetalError)?,
                    )
                } else {
                    MatrixMatrixKernel::QmmBf16G128B4AlignedK(
                        <QmmBf16G128B4AlignedKMetalKernel as QmmBf16G128B4AlignedKKernelTrait>::new(mtl_context)
                        .map_err(QuantizedMatmulError::MetalError)?,
                    )
                }
            } else if weights_transposed {
                MatrixMatrixKernel::QmmTransposed(
                    <QmmTransposedMetalKernel as QmmTransposedKernelTrait>::new(
                        mtl_context,
                        data_type,
                        use_mlx_quant,
                        !use_mlx_quant,
                        group_size_u32,
                        bits,
                        output_dim % 32 == 0,
                    )
                    .map_err(QuantizedMatmulError::MetalError)?,
                )
            } else {
                MatrixMatrixKernel::Qmm(
                    <QmmMetalKernel as QmmKernelTrait>::new(
                        mtl_context,
                        data_type,
                        use_mlx_quant,
                        !use_mlx_quant,
                        group_size_u32,
                        bits,
                        input_dim % 32 == 0,
                    )
                    .map_err(QuantizedMatmulError::MetalError)?,
                )
            }
        } else {
            build_legacy_mm()?
        };

        Ok(Self {
            matrix_vector_kernel,
            matrix_matrix_kernel,
            output_dim,
        })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: QuantizedMatmulArguments,
    ) -> Result<(), QuantizedMatmulError> {
        let variant = self.select_variant(args.batch as usize);

        match variant {
            KernelKind::MatrixVector => {
                let input = (args.a_buffer, args.a_offset as usize);
                let zero_points =
                    matches!(args.quantization_type, QuantizationType::ZeroPoint).then_some(args.zero_points_or_biases_buffer);
                let biases =
                    matches!(args.quantization_type, QuantizationType::Mlx).then_some(args.zero_points_or_biases_buffer);

                match &self.matrix_vector_kernel {
                    MatrixVectorKernel::QmvFast(kernel) => {
                        <QmvFastMetalKernel as QmvFastKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            zero_points,
                            biases,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                    MatrixVectorKernel::Qmv(kernel) => {
                        <QmvMetalKernel as QmvKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            zero_points,
                            biases,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                    MatrixVectorKernel::Qvm(kernel) => {
                        <QvmMetalKernel as QvmKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            zero_points,
                            biases,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                }
            },
            KernelKind::MatrixMatrix => {
                let input = (args.a_buffer, args.a_offset as usize);
                let zero_points =
                    matches!(args.quantization_type, QuantizationType::ZeroPoint).then_some(args.zero_points_or_biases_buffer);
                let biases =
                    matches!(args.quantization_type, QuantizationType::Mlx).then_some(args.zero_points_or_biases_buffer);

                match &self.matrix_matrix_kernel {
                    MatrixMatrixKernel::QmmBf16G128B4AlignedKMlx(kernel) => {
                        <QmmBf16G128B4AlignedKMlxMetalKernel as QmmBf16G128B4AlignedKMlxKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            args.zero_points_or_biases_buffer,
                            args.zero_points_or_biases_buffer,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                    MatrixMatrixKernel::QmmBf16G128B4AlignedK(kernel) => {
                        <QmmBf16G128B4AlignedKMetalKernel as QmmBf16G128B4AlignedKKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            args.zero_points_or_biases_buffer,
                            args.zero_points_or_biases_buffer,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                    MatrixMatrixKernel::Qmm(kernel) => {
                        <QmmMetalKernel as QmmKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            zero_points,
                            biases,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                    MatrixMatrixKernel::QmmTransposed(kernel) => {
                        <QmmTransposedMetalKernel as QmmTransposedKernelTrait>::encode(
                            kernel,
                            args.b_buffer,
                            args.scales_buffer,
                            zero_points,
                            biases,
                            input,
                            args.output_buffer,
                            args.input_dim,
                            args.output_dim,
                            args.batch,
                            encoder,
                        );
                    },
                    MatrixMatrixKernel::Legacy((pipeline, bm, bn)) => {
                        encoder.set_compute_pipeline_state(pipeline);

                        encoder.set_buffer(Some(args.b_buffer), 0, 0);
                        encoder.set_buffer(Some(args.scales_buffer), 0, 1);
                        encoder.set_buffer(Some(args.zero_points_or_biases_buffer), 0, 2);
                        encoder.set_buffer(Some(args.a_buffer), args.a_offset as usize, 3);
                        encoder.set_buffer(Some(args.output_buffer), 0, 4);

                        let k = args.input_dim;
                        let m = args.batch;
                        let n = args.output_dim;
                        encoder.set_value(&k, 5);
                        encoder.set_value(&n, 6);
                        encoder.set_value(&m, 7);

                        let wm = 2;
                        let wn = 2;
                        let threads_per_threadgroup = MTLSize::new(32, wn as usize, wm as usize);
                        let threadgroups =
                            MTLSize::new(((n as u64 + bn - 1) / bn) as usize, ((m as u64 + bm - 1) / bm) as usize, 1);
                        encoder.dispatch_threadgroups(threadgroups, threads_per_threadgroup);
                    },
                }
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

impl QuantizedMatmulKernelTrait for QuantizedMatmulKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, MetalError> {
        let quantization_type = match configuration.quantization_type {
            QuantizedMatmulType::ZeroPoint => QuantizationType::ZeroPoint,
            QuantizedMatmulType::Mlx => QuantizationType::Mlx,
        };

        QuantizedMatmulKernel::new(
            context,
            configuration.data_type,
            configuration.group_size,
            configuration.input_dim,
            configuration.output_dim,
            configuration.mode,
            quantization_type,
            configuration.weights_transposed,
        )
        .map_err(|error| MetalError::Generic(format!("{:?}", error)))
    }

    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: GenericQuantizedMatmulArguments<Metal>,
    ) {
        let quantization_type = match arguments.quantization_type {
            QuantizedMatmulType::ZeroPoint => QuantizationType::ZeroPoint,
            QuantizedMatmulType::Mlx => QuantizationType::Mlx,
        };

        let backend_arguments = QuantizedMatmulArguments {
            a_buffer: arguments.a_buffer,
            a_offset: arguments.a_offset as u64,
            b_buffer: arguments.b_buffer,
            scales_buffer: arguments.scales_buffer,
            zero_points_or_biases_buffer: arguments.zero_points_or_biases_buffer,
            output_buffer: arguments.output_buffer,
            batch: arguments.batch as i32,
            input_dim: arguments.input_dim as i32,
            output_dim: arguments.output_dim as i32,
            quantization_type,
        };

        QuantizedMatmulKernel::encode(self, encoder, backend_arguments).expect("Failed to encode quantized matmul");
    }
}

fn select_qmm_kernel_name(
    data_type: DataType,
    group_size: usize,
    weights_transposed: bool,
    output_dim: usize,
    input_dim: usize,
    mode: QuantizationMode,
) -> Result<String, QuantizedMatmulError> {
    let type_suffix = dtype_suffix(data_type).ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;
    let bits = match mode {
        QuantizationMode::UInt4 => 4,
        QuantizationMode::Int8 | QuantizationMode::UInt8 => 8,
    };

    let transpose_infix = if weights_transposed {
        "_transposed"
    } else {
        ""
    };
    let mut kernel_name = base_qmm_kernel_name(type_suffix, group_size, transpose_infix, bits)?;
    if weights_transposed {
        if output_dim % 32 != 0 {
            kernel_name.push_str("_unaligned");
        } else if type_suffix == "bf16" && (group_size == 128 || group_size == 64) && (bits == 4 || bits == 8) {
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
    pub weights: &'a ProtocolObject<dyn MTLBuffer>,
    /// Scales buffer
    pub scales: &'a ProtocolObject<dyn MTLBuffer>,
    /// Zero points or biases buffer (depends on quantization type)
    pub zero_points_or_biases: &'a ProtocolObject<dyn MTLBuffer>,
    /// Input vector [input_dim]
    pub input: &'a ProtocolObject<dyn MTLBuffer>,
    /// Input byte offset
    pub input_offset: u64,
    /// Output vector [hidden_dim]
    pub output: &'a ProtocolObject<dyn MTLBuffer>,
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
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MlpFusedQmvKernel {
    pub fn new(
        context: &MetalContext,
        data_type: DataType,
        group_size: usize,
        mode: QuantizationMode,
        quantization_type: QuantizationType,
    ) -> Result<Self, QuantizedMatmulError> {
        let type_suffix = dtype_suffix(data_type).ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;

        let bits = match mode {
            QuantizationMode::UInt4 => 4,
            QuantizationMode::UInt8 | QuantizationMode::Int8 => 8,
        };

        if bits != 4 {
            return Err(QuantizedMatmulError::UnsupportedBits(bits));
        }

        let kernel_name = format!("qmv_mlp_fused_{}_g{}_b{}", type_suffix, group_size, bits);

        let function_constants = MTLFunctionConstantValues::new();
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);
        function_constants.set_value(&use_mlx_quant, 40);

        let pipeline = context
            .compute_pipeline_state(&kernel_name, Some(&function_constants))
            .map_err(QuantizedMatmulError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &MlpFusedQmvArguments,
    ) -> Result<(), QuantizedMatmulError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(Some(args.weights), 0, 0);
        encoder.set_buffer(Some(args.scales), 0, 1);
        encoder.set_buffer(Some(args.zero_points_or_biases), 0, 2);
        encoder.set_buffer(Some(args.input), args.input_offset as usize, 3);
        encoder.set_buffer(Some(args.output), 0, 4);

        encoder.set_value(&args.input_dim, 5);
        encoder.set_value(&args.hidden_dim, 6);

        // Dispatch: one threadgroup per 8 output rows (num_simdgroups * results_per_simdgroup)
        let rows_per_threadgroup = 8;
        let n_tgp_y = ((args.hidden_dim + rows_per_threadgroup - 1) / rows_per_threadgroup) as u64;
        let threadgroups = MTLSize::new(args.batch_count.max(1) as usize, n_tgp_y as usize, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 1); // 2 simdgroups, 32 threads each

        encoder.dispatch_threadgroups(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}

/// Arguments for MLP fused quantized GEMM (prefill)
#[derive(Debug)]
pub struct MlpFusedQmmArguments<'a> {
    /// Weight matrix [K, 2*hidden_dim] quantized (up and gate concatenated)
    pub weights: &'a ProtocolObject<dyn MTLBuffer>,
    /// Scales buffer
    pub scales: &'a ProtocolObject<dyn MTLBuffer>,
    /// Zero points or biases buffer (depends on quantization type)
    pub zero_points_or_biases: &'a ProtocolObject<dyn MTLBuffer>,
    /// Input activations [M, K]
    pub input: &'a ProtocolObject<dyn MTLBuffer>,
    /// Input byte offset
    pub input_offset: u64,
    /// Output [M, hidden_dim]
    pub output: &'a ProtocolObject<dyn MTLBuffer>,
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
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MlpFusedQmmKernel {
    pub fn new(
        context: &MetalContext,
        data_type: DataType,
        group_size: usize,
        mode: QuantizationMode,
        quantization_type: QuantizationType,
    ) -> Result<Self, QuantizedMatmulError> {
        let type_suffix = dtype_suffix(data_type).ok_or(QuantizedMatmulError::UnsupportedDataType(data_type))?;

        let bits = match mode {
            QuantizationMode::UInt4 => 4,
            QuantizationMode::UInt8 | QuantizationMode::Int8 => 8,
        };

        if bits != 4 {
            return Err(QuantizedMatmulError::UnsupportedBits(bits));
        }

        let kernel_name = format!("qmm_mlp_fused_{}_g{}_b{}", type_suffix, group_size, bits);

        let function_constants = MTLFunctionConstantValues::new();
        let use_mlx_quant = matches!(quantization_type, QuantizationType::Mlx);
        function_constants.set_value(&use_mlx_quant, 40);

        let pipeline = context
            .compute_pipeline_state(&kernel_name, Some(&function_constants))
            .map_err(QuantizedMatmulError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &MlpFusedQmmArguments,
    ) -> Result<(), QuantizedMatmulError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(Some(args.weights), 0, 0);
        encoder.set_buffer(Some(args.scales), 0, 1);
        encoder.set_buffer(Some(args.zero_points_or_biases), 0, 2);
        encoder.set_buffer(Some(args.input), args.input_offset as usize, 3);
        encoder.set_buffer(Some(args.output), 0, 4);

        encoder.set_value(&args.input_dim, 5);
        encoder.set_value(&args.hidden_dim, 6);
        encoder.set_value(&args.batch, 7);

        // Dispatch: BM=32, BN=32
        let bm = 32;
        let bn = 32;
        let tiles_m = ((args.batch + bm - 1) / bm) as u64;
        let tiles_n = ((args.hidden_dim + bn - 1) / bn) as u64;
        let threadgroups = MTLSize::new(tiles_n as usize, tiles_m as usize, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 2); // WM=2, WN=2, 32 threads

        encoder.dispatch_threadgroups(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}
