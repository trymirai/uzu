//! Quantized embedding encodables.

use std::rc::Rc;

use metal::Buffer as MTLBuffer;
use mpsgraph::CommandBuffer;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::{
            embedding::{
                QuantizedEmbeddingLookupArguments,
                QuantizedEmbeddingLookupKernel,
            },
            quant_matmul::{
                QuantizationType, QuantizedMatmulArguments,
                QuantizedMatmulKernel,
            },
        },
    },
    config::QuantizationMode,
    parameters::ParameterTree,
};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedEmbeddingError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
}

pub struct QuantizedEmbeddingLookup {
    kernel: QuantizedEmbeddingLookupKernel,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    biases_buffer: MTLBuffer,
    vocab_size: u32,
    model_dim: u32,
    group_size: u32,
    input_scale: f32,
}

impl QuantizedEmbeddingLookup {
    pub fn new_tied(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        input_scale: f32,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        Self::new_with_names(
            mtl_context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            input_scale,
            "weights",
            "scales",
            "biases",
            parameter_tree,
        )
    }

    pub fn new_untied_input(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        input_scale: f32,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        Self::new_with_names(
            mtl_context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            input_scale,
            "input_weights",
            "input_scales",
            "input_biases",
            parameter_tree,
        )
    }

    fn new_with_names(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        input_scale: f32,
        weights_name: &str,
        scales_name: &str,
        biases_name: &str,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        let packing_divisor = mode.packing_divisor();

        let kernel =
            QuantizedEmbeddingLookupKernel::new(mtl_context, data_type, mode)?;

        // Load weights [vocab_size, model_dim/packing_divisor] as storage_type
        let mut weights = parameter_tree.leaf(weights_name).map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load {weights_name}: {:?}",
                e
            )))
        })?;

        if weights.data_type() != mode.storage_type() {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Expected packed weights of type {:?}, got {:?}",
                    mode.storage_type(),
                    weights.data_type()
                )),
            ));
        }

        // Load scales [vocab_size, num_groups]
        let mut scales = parameter_tree.leaf(scales_name).map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load {scales_name}: {:?}",
                e
            )))
        })?;

        // Validate shapes and types
        let num_groups = (model_dim + group_size - 1) / group_size;
        if weights.shape() != [vocab_size, model_dim / packing_divisor] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding lookup weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim / packing_divisor
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

        // Load or create biases buffer [vocab_size, num_groups]
        let biases_buffer: MTLBuffer = match parameter_tree.leaf(biases_name) {
            Ok(mut deq_biases) => {
                if deq_biases.shape() != [vocab_size, num_groups] {
                    return Err(QuantizedEmbeddingError::MetalError(
                        MTLError::Generic(format!(
                            "Embedding lookup biases shape mismatch: got {:?}, expected [{}, {}]",
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
            input_scale,
        })
    }
}

impl EncodableBlock for QuantizedEmbeddingLookup {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
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
            input_scale: self.input_scale,
        };

        self.kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized embedding lookup kernel");

        encoder.end_encoding();

        if parameters.wait_until_completed {
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
pub struct QuantizedEmbeddingReadout {
    kernel: QuantizedMatmulKernel,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    biases_buffer: MTLBuffer,
    vocab_size: usize,
    model_dim: usize,
}

impl QuantizedEmbeddingReadout {
    pub fn new_tied(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        Self::new_with_names(
            mtl_context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            "weights",
            "scales",
            "biases",
            parameter_tree,
        )
    }

    pub fn new_untied_output(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        Self::new_with_names(
            mtl_context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            "output_weights",
            "output_scales",
            "output_biases",
            parameter_tree,
        )
    }

    fn new_with_names(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        weights_name: &str,
        scales_name: &str,
        biases_name: &str,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        // Load weights [vocab_size, model_dim/2] as U8
        let mut weights = parameter_tree.leaf(weights_name).map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load {weights_name}: {:?}",
                e
            )))
        })?;

        // Load scales [vocab_size, num_groups]
        let mut scales = parameter_tree.leaf(scales_name).map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load {scales_name}: {:?}",
                e
            )))
        })?;

        // Validate shapes
        let num_groups = (model_dim + group_size - 1) / group_size;
        let packing_divisor = mode.packing_divisor();

        // Determine if weights are transposed by checking shape
        let weights_transposed = weights.shape()[0] == vocab_size;

        if weights.shape() != [vocab_size, model_dim / packing_divisor] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding readout weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim / packing_divisor
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
        let biases_buffer: MTLBuffer = match parameter_tree.leaf(biases_name) {
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

        let kernel = QuantizedMatmulKernel::new(
            mtl_context,
            data_type,
            group_size,
            model_dim,
            vocab_size,
            mode,
            QuantizationType::Mlx,
            weights_transposed,
        )
        .map_err(|e| {
            QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to create kernel: {:?}",
                e
            )))
        })?;

        Ok(Self {
            kernel,
            weights_buffer,
            scales_buffer,
            biases_buffer,
            vocab_size,
            model_dim,
        })
    }
}

impl EncodableBlock for QuantizedEmbeddingReadout {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::Main, ArrayId::Logits]);
        let batch_size = state.active_suffix_length();
        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let root_command_buffer = command_buffer.root_command_buffer();
        let encoder = root_command_buffer.new_compute_command_encoder();

        let args = QuantizedMatmulArguments {
            a_buffer: input_buffer,
            b_buffer: &self.weights_buffer,
            scales_buffer: &self.scales_buffer,
            zero_points_or_biases_buffer: &self.biases_buffer,
            output_buffer,
            batch: batch_size as i32,
            input_dim: self.model_dim as i32,
            output_dim: self.vocab_size as i32,
            quantization_type: QuantizationType::Mlx,
        };

        self.kernel
            .encode(encoder, args)
            .expect("Failed to encode quantized embedding readout kernel");

        encoder.end_encoding();

        if parameters.wait_until_completed {
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
