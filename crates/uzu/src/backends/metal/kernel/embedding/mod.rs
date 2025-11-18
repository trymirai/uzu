use std::{mem::size_of, rc::Rc};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};
use mpsgraph::CommandBuffer;

use super::super::{
    MTLContext,
    forward_pass::{
        ArrayId, ForwardPassState,
        encodable_with_state::{EncodableWithState, EncodingParameters},
    },
};
use crate::{
    Array, DataType, backends::metal::MTLError, parameters::ParameterTree,
};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedEmbeddingError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
}

pub struct QuantizedEmbeddingLookupKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct QuantizedEmbeddingLookupArguments<'a> {
    pub token_ids_buffer: &'a MTLBuffer, // [batch_size] as U64
    pub weights_buffer: &'a MTLBuffer,   // [vocab_size, model_dim/2] as U8
    pub scales_buffer: &'a MTLBuffer,    // [vocab_size, num_groups]
    pub biases_buffer: &'a MTLBuffer,    // [vocab_size, num_groups]
    pub output_buffer: &'a MTLBuffer,    // [batch_size, model_dim]
    pub batch_size: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub group_size: u32,
}

impl QuantizedEmbeddingLookupKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
    ) -> Result<Self, QuantizedEmbeddingError> {
        let kernel_name = match data_type {
            DataType::F32 => "quantized_embedding_lookup_f32",
            DataType::F16 => "quantized_embedding_lookup_f16",
            DataType::BF16 => "quantized_embedding_lookup_bf16",
            other => {
                return Err(QuantizedEmbeddingError::UnsupportedDataType(
                    other,
                ));
            },
        };

        let (pipeline, _) = mtl_context
            .compute_pipeline_state_with_reflection(kernel_name, None)
            .map_err(QuantizedEmbeddingError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: QuantizedEmbeddingLookupArguments,
    ) -> Result<(), QuantizedEmbeddingError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(args.token_ids_buffer), 0);
        encoder.set_buffer(1, Some(args.weights_buffer), 0);
        encoder.set_buffer(2, Some(args.scales_buffer), 0);
        encoder.set_buffer(3, Some(args.biases_buffer), 0);
        encoder.set_buffer(4, Some(args.output_buffer), 0);

        // Set constants
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &args.batch_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &args.vocab_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &args.model_dim as *const u32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &args.group_size as *const u32 as *const _,
        );

        // Dispatch one thread per output element
        let total_threads = (args.batch_size * args.model_dim) as u64;
        let threads_per_threadgroup = 256u64;
        let threadgroups = (total_threads + threads_per_threadgroup - 1)
            / threads_per_threadgroup;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );

        Ok(())
    }
}

pub struct QuantizedEmbeddingLookupKernelBlock {
    kernel: QuantizedEmbeddingLookupKernel,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    biases_buffer: MTLBuffer,
    vocab_size: u32,
    model_dim: u32,
    group_size: u32,
}

impl QuantizedEmbeddingLookupKernelBlock {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        let kernel =
            QuantizedEmbeddingLookupKernel::new(mtl_context, data_type)?;

        // Load weights [vocab_size, model_dim/2] as U8
        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("output_weights").map_err(|e| {
                QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load weights: {:?}",
                    e
                )))
            })?,
        };

        if weights.data_type() != DataType::U8 {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Expected packed U8 weights, got {:?}",
                    weights.data_type()
                )),
            ));
        }

        // Load scales [vocab_size, num_groups]
        let mut scales = match parameter_tree.leaf("scales") {
            Ok(scales) => scales,
            Err(_) => parameter_tree.leaf("output_scales").map_err(|e| {
                QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load scales: {:?}",
                    e
                )))
            })?,
        };

        // Validate shapes and types
        let num_groups = (model_dim + group_size - 1) / group_size;
        if weights.shape() != [vocab_size, model_dim / 2] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding lookup weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim / 2
                )),
            ));
        }
        if scales.shape() != [vocab_size, num_groups] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding lookup scales shape mismatch: got {:?}, expected [{}, {}]",
                    scales.shape(),
                    vocab_size,
                    num_groups
                )),
            ));
        }
        if scales.data_type() != data_type {
            return Err(QuantizedEmbeddingError::UnsupportedDataType(
                scales.data_type(),
            ));
        }

        // Load or create biases buffer [vocab_size, num_groups] (MLX key: "biases")
        let biases_buffer: MTLBuffer = match parameter_tree
            .leaf("biases")
            .or_else(|_| parameter_tree.leaf("output_biases"))
        {
            Ok(mut deq_biases) => {
                if deq_biases.shape() != [vocab_size, num_groups] {
                    return Err(QuantizedEmbeddingError::MetalError(
                        MTLError::Generic(format!(
                            "Embedding lookup deq_biases shape mismatch: got {:?}, expected [{}, {}]",
                            deq_biases.shape(),
                            vocab_size,
                            num_groups
                        )),
                    ));
                }
                if deq_biases.data_type() != data_type {
                    return Err(QuantizedEmbeddingError::UnsupportedDataType(
                        deq_biases.data_type(),
                    ));
                }
                unsafe { deq_biases.mtl_buffer().to_owned() }
            },
            Err(_) => {
                let elem_size: usize = match data_type {
                    DataType::F16 | DataType::BF16 => 2,
                    DataType::F32 => 4,
                    other => {
                        return Err(
                            QuantizedEmbeddingError::UnsupportedDataType(other),
                        );
                    },
                };
                let size_bytes = (vocab_size * num_groups * elem_size) as u64;
                let buf = mtl_context.device.new_buffer(
                    size_bytes,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    std::ptr::write_bytes(
                        buf.contents(),
                        0,
                        size_bytes as usize,
                    );
                }
                buf
            },
        };

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };
        let scales_buffer = unsafe { scales.mtl_buffer().to_owned() };

        Ok(Self {
            kernel,
            weights_buffer,
            scales_buffer,
            biases_buffer,
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            group_size: group_size as u32,
        })
    }
}

impl EncodableWithState for QuantizedEmbeddingLookupKernelBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBuffer,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let token_ids_array = arrays[0].borrow();
        let batch_size = token_ids_array.shape()[0];
        drop(token_ids_array);

        let mut token_ids_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let token_ids_buffer = unsafe { token_ids_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let root_command_buffer = command_buffer.root_command_buffer();
        let encoder = root_command_buffer.new_compute_command_encoder();

        let args = QuantizedEmbeddingLookupArguments {
            token_ids_buffer,
            weights_buffer: &self.weights_buffer,
            scales_buffer: &self.scales_buffer,
            biases_buffer: &self.biases_buffer,
            output_buffer,
            batch_size: batch_size as u32,
            vocab_size: self.vocab_size,
            model_dim: self.model_dim,
            group_size: self.group_size,
        };

        self.kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized embedding lookup kernel");

        encoder.end_encoding();
    }
}

// Quantized embedding readout block - uses quantized matmul for efficiency
pub struct QuantizedEmbeddingReadoutKernelBlock {
    kernel_mm: super::super::kernel::quant_matmul::QuantizedMatmulKernel,
    kernel_mv: super::super::kernel::quant_matmul::QuantizedMatmulKernel,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    biases_buffer: MTLBuffer,
    vocab_size: usize,
    model_dim: usize,
}

impl QuantizedEmbeddingReadoutKernelBlock {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        // Load weights [vocab_size, model_dim/2] as U8
        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("output_weights").map_err(|e| {
                QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load weights: {:?}",
                    e
                )))
            })?,
        };

        // Load scales [vocab_size, num_groups]
        let mut scales = match parameter_tree.leaf("scales") {
            Ok(scales) => scales,
            Err(_) => parameter_tree.leaf("output_scales").map_err(|e| {
                QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load scales: {:?}",
                    e
                )))
            })?,
        };

        // Validate shapes
        let num_groups = (model_dim + group_size - 1) / group_size;
        if weights.shape() != [vocab_size, model_dim / 2] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding readout weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim / 2
                )),
            ));
        }
        if scales.shape() != [vocab_size, num_groups] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding readout scales shape mismatch: got {:?}, expected [{}, {}]",
                    scales.shape(),
                    vocab_size,
                    num_groups
                )),
            ));
        }
        if scales.data_type() != data_type {
            return Err(QuantizedEmbeddingError::UnsupportedDataType(
                scales.data_type(),
            ));
        }

        // MLX requires per-group biases; if missing, create a zero buffer of shape [vocab_size, num_groups]
        let biases_buffer: MTLBuffer = match parameter_tree
            .leaf("biases")
            .or_else(|_| parameter_tree.leaf("output_biases"))
        {
            Ok(mut deq_biases) => {
                if deq_biases.shape() != [vocab_size, num_groups] {
                    return Err(QuantizedEmbeddingError::MetalError(
                        MTLError::Generic(format!(
                            "Embedding readout deq_biases shape mismatch: got {:?}, expected [{}, {}]",
                            deq_biases.shape(),
                            vocab_size,
                            num_groups
                        )),
                    ));
                }
                if deq_biases.data_type() != data_type {
                    return Err(QuantizedEmbeddingError::UnsupportedDataType(
                        deq_biases.data_type(),
                    ));
                }
                unsafe { deq_biases.mtl_buffer().to_owned() }
            },
            Err(_) => {
                // Allocate zero-initialized biases buffer
                let elem_size: usize = match data_type {
                    DataType::F16 | DataType::BF16 => 2,
                    DataType::F32 => 4,
                    other => {
                        return Err(
                            QuantizedEmbeddingError::UnsupportedDataType(other),
                        );
                    },
                };
                let size_bytes = (vocab_size * num_groups * elem_size) as u64;
                let buf = mtl_context.device.new_buffer(
                    size_bytes,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    std::ptr::write_bytes(
                        buf.contents(),
                        0,
                        size_bytes as usize,
                    );
                }
                buf
            },
        };

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };
        let scales_buffer = unsafe { scales.mtl_buffer().to_owned() };

        use super::super::kernel::quant_matmul::{
            QuantizationType, QuantizedMatmulKernel, quantized_kernel_names,
        };

        let aligned_n = vocab_size % 32 == 0;
        let Some((kernel_name_mm, kernel_name_mv)) =
            quantized_kernel_names(data_type, group_size, aligned_n)
        else {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Unsupported group size {} for transposed {:?} kernel",
                    group_size, data_type
                )),
            ));
        };

        let kernel_mm = QuantizedMatmulKernel::new(
            mtl_context,
            data_type,
            &kernel_name_mm,
            QuantizationType::Mlx,
        )
        .map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to create kernel: {:?}",
                e
            )))
        })?;

        let kernel_mv = QuantizedMatmulKernel::new(
            mtl_context,
            data_type,
            &kernel_name_mv,
            QuantizationType::Mlx,
        )
        .map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to create kernel: {:?}",
                e
            )))
        })?;

        Ok(Self {
            kernel_mm,
            kernel_mv,
            weights_buffer,
            scales_buffer,
            biases_buffer,
            vocab_size,
            model_dim,
        })
    }
}

impl EncodableWithState for QuantizedEmbeddingReadoutKernelBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBuffer,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::Main, ArrayId::Logits]);
        let input_array = arrays[0].borrow();
        let batch_size = input_array.shape()[0];
        drop(input_array);

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let root_command_buffer = command_buffer.root_command_buffer();
        let encoder = root_command_buffer.new_compute_command_encoder();

        // For transposed matmul: input @ weights.T
        // where weights is [vocab_size, model_dim]
        use super::super::kernel::quant_matmul::{
            QuantizationType, QuantizedMatmulArguments,
        };

        let args = QuantizedMatmulArguments {
            a_buffer: input_buffer,
            b_buffer: &self.weights_buffer,
            scales_buffer: &self.scales_buffer,
            zero_points_or_biases_buffer: &self.biases_buffer,
            output_buffer,
            m: batch_size as i32,
            n: self.vocab_size as i32,
            k: self.model_dim as i32,
            quantization_type: QuantizationType::Mlx,
        };
        let kernel = if batch_size == 1 {
            &self.kernel_mv
        } else {
            &self.kernel_mm
        };

        kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized embedding readout kernel");

        encoder.end_encoding();
    }
}
