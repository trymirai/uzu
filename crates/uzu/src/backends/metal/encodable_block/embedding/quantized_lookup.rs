use std::rc::Rc;

use metal::{Buffer as MTLBuffer, CommandBufferRef, ComputeCommandEncoderRef};

use super::{
    super::{EncodableBlock, EncodingParameters},
    EmbeddingError,
};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::embedding::{
            QuantizedEmbeddingLookupArguments, QuantizedEmbeddingLookupKernel,
        },
    },
    config::QuantizationMode,
    parameters::ParameterTree,
};

pub struct QuantizedEmbeddingLookup {
    kernel: QuantizedEmbeddingLookupKernel,
    weights_buffer: MTLBuffer,
    scales_buffer: MTLBuffer,
    biases_buffer: MTLBuffer,
    input_scale: f32,
    vocab_size: u32,
    model_dim: u32,
    group_size: u32,
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
    ) -> Result<Self, EmbeddingError> {
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
    ) -> Result<Self, EmbeddingError> {
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
    ) -> Result<Self, EmbeddingError> {
        let packing_divisor = mode.packing_divisor();

        let kernel =
            QuantizedEmbeddingLookupKernel::new(mtl_context, data_type, mode)?;

        // Load weights [vocab_size, model_dim/packing_divisor] as storage_type
        let mut weights = parameter_tree.leaf(weights_name).map_err(|e| {
            EmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load weights: {:?}",
                e
            )))
        })?;

        if weights.data_type() != mode.storage_type() {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Expected packed weights of type {:?}, got {:?}",
                    mode.storage_type(),
                    weights.data_type()
                ),
            )));
        }

        // Load scales [vocab_size, num_groups]
        let mut scales = parameter_tree.leaf(scales_name).map_err(|e| {
            EmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load scales: {:?}",
                e
            )))
        })?;

        // Validate shapes and types
        let num_groups = (model_dim + group_size - 1) / group_size;
        if weights.shape() != [vocab_size, model_dim / packing_divisor] {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Embedding lookup weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim / packing_divisor
                ),
            )));
        }
        if scales.shape() != [vocab_size, num_groups] {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Embedding lookup scales shape mismatch: got {:?}, expected [{}, {}]",
                    scales.shape(),
                    vocab_size,
                    num_groups
                ),
            )));
        }
        if scales.data_type() != data_type {
            return Err(EmbeddingError::UnsupportedDataType(
                scales.data_type(),
            ));
        }

        // Load or create biases buffer [vocab_size, num_groups] (MLX key: "biases")
        let biases_buffer: MTLBuffer = match parameter_tree.leaf(biases_name) {
            Ok(mut deq_biases) => {
                if deq_biases.shape() != [vocab_size, num_groups] {
                    return Err(EmbeddingError::MetalError(MTLError::Generic(
                        format!(
                            "Embedding lookup deq_biases shape mismatch: got {:?}, expected [{}, {}]",
                            deq_biases.shape(),
                            vocab_size,
                            num_groups
                        ),
                    )));
                }
                if deq_biases.data_type() != data_type {
                    return Err(EmbeddingError::UnsupportedDataType(
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
                        return Err(EmbeddingError::UnsupportedDataType(other));
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
            input_scale,
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            group_size: group_size as u32,
        })
    }
}

impl EncodableBlock for QuantizedEmbeddingLookup {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBufferRef,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
        let mut token_ids_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let token_ids_buffer = unsafe { token_ids_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

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
    }
}
