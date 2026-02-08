use super::{
    super::{EncodableBlock, EncodingParameters, Metal},
    EmbeddingError,
};
use crate::{
    DataType,
    backends::{
        common::{Context, kernel::QuantizedEmbeddingLookupKernel},
        metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
            MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
            Retained,
            forward_pass::{ArrayId, ForwardPassState},
            kernel::dsl::QuantizedEmbeddingLookupMetalKernel,
        },
    },
    config::QuantizationMode,
    parameters::ParameterTree,
};

pub struct QuantizedEmbeddingLookup {
    kernel: QuantizedEmbeddingLookupMetalKernel,
    weights_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    scales_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    biases_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mode: QuantizationMode,
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
        parameter_tree: &ParameterTree<MTLContext>,
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
        parameter_tree: &ParameterTree<MTLContext>,
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
        parameter_tree: &ParameterTree<MTLContext>,
    ) -> Result<Self, EmbeddingError> {
        let packing_divisor = mode.packing_divisor();

        let kernel = QuantizedEmbeddingLookupMetalKernel::new(
            mtl_context,
            data_type.into(),
        )?;

        // Load weights [vocab_size, model_dim/packing_divisor] as storage_type
        let weights = parameter_tree.leaf(weights_name).map_err(|e| {
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
        let scales = parameter_tree.leaf(scales_name).map_err(|e| {
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
        let biases_buffer: Retained<ProtocolObject<dyn MTLBuffer>> =
            match parameter_tree.leaf(biases_name) {
                Ok(deq_biases) => {
                    if deq_biases.shape() != [vocab_size, num_groups] {
                        return Err(EmbeddingError::MetalError(
                            MTLError::Generic(format!(
                                "Embedding lookup deq_biases shape mismatch: got {:?}, expected [{}, {}]",
                                deq_biases.shape(),
                                vocab_size,
                                num_groups
                            )),
                        ));
                    }
                    if deq_biases.data_type() != data_type {
                        return Err(EmbeddingError::UnsupportedDataType(
                            deq_biases.data_type(),
                        ));
                    }
                    deq_biases.buffer().to_owned().into()
                },
                Err(_) => {
                    let elem_size: usize = match data_type {
                        DataType::F16 | DataType::BF16 => 2,
                        DataType::F32 => 4,
                        other => {
                            return Err(EmbeddingError::UnsupportedDataType(
                                other,
                            ));
                        },
                    };
                    let size_bytes = vocab_size * num_groups * elem_size;
                    let buf = mtl_context
                        .create_buffer(size_bytes)
                        .expect("Failed to allocate buffer");
                    unsafe {
                        std::ptr::write_bytes(
                            metal::MTLBuffer::contents(&*buf).as_ptr(),
                            0,
                            size_bytes as usize,
                        );
                    }
                    buf
                },
            };

        let weights_buffer = weights.buffer().to_owned().into();
        let scales_buffer = scales.buffer().to_owned().into();

        Ok(Self {
            kernel,
            weights_buffer,
            scales_buffer,
            biases_buffer,
            mode,
            input_scale,
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            group_size: group_size as u32,
        })
    }
}

impl EncodableBlock<Metal> for QuantizedEmbeddingLookup {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
        let token_ids_array_mut = arrays[0].borrow_mut();
        let output_array_mut = arrays[1].borrow_mut();

        let quant_mode = match self.mode {
            QuantizationMode::UInt4 => 0,
            QuantizationMode::Int8 => 1,
            QuantizationMode::UInt8 => 2,
        };
        self.kernel.encode(
            token_ids_array_mut.buffer(),
            &self.weights_buffer,
            &self.scales_buffer,
            &self.biases_buffer,
            output_array_mut.buffer(),
            batch_size as u32,
            self.vocab_size,
            self.model_dim,
            self.group_size,
            self.input_scale,
            quant_mode,
            encoder,
        )
    }
}
