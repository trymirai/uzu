use std::rc::Rc;

use super::{
    super::{EncodableBlock, EncodingParameters},
    EmbeddingError,
};
use crate::{
    Array, DataType,
    backends::{
        common::Context,
        metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
            MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
            Retained,
            forward_pass::{ArrayId, ForwardPassState},
            kernel::quant_matmul::{
                QuantizationType, QuantizedMatmulArguments,
                QuantizedMatmulKernel,
            },
        },
    },
    config::QuantizationMode,
    parameters::ParameterTree,
};

pub struct QuantizedEmbeddingReadout {
    kernel: QuantizedMatmulKernel,
    weights_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    scales_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    biases_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
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
    ) -> Result<Self, EmbeddingError> {
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
    ) -> Result<Self, EmbeddingError> {
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
    ) -> Result<Self, EmbeddingError> {
        // Load weights [vocab_size, model_dim/2] as U8
        let mut weights = parameter_tree.leaf(weights_name).map_err(|e| {
            EmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load weights: {:?}",
                e
            )))
        })?;

        // Load scales [vocab_size, num_groups]
        let mut scales = parameter_tree.leaf(scales_name).map_err(|e| {
            EmbeddingError::MetalError(MTLError::Generic(format!(
                "Failed to load scales: {:?}",
                e
            )))
        })?;

        // Validate shapes
        let num_groups = (model_dim + group_size - 1) / group_size;
        let packing_divisor = mode.packing_divisor();

        // Determine if weights are transposed by checking shape
        let weights_transposed = weights.shape()[0] == vocab_size;

        if weights.shape() != [vocab_size, model_dim / packing_divisor] {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Embedding readout weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim / packing_divisor
                ),
            )));
        }
        if scales.shape() != [vocab_size, num_groups] {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Embedding readout scales shape mismatch: got {:?}, expected [{}, {}]",
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

        // MLX requires per-group biases; if missing, create a zero buffer of shape [vocab_size, num_groups]
        let biases_buffer: Retained<ProtocolObject<dyn MTLBuffer>> =
            match parameter_tree.leaf(biases_name) {
                Ok(mut deq_biases) => {
                    if deq_biases.shape() != [vocab_size, num_groups] {
                        return Err(EmbeddingError::MetalError(
                            MTLError::Generic(format!(
                                "Embedding readout deq_biases shape mismatch: got {:?}, expected [{}, {}]",
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
                    unsafe { deq_biases.mtl_buffer().to_owned().into() }
                },
                Err(_) => {
                    // Allocate zero-initialized biases buffer
                    let elem_size: usize = match data_type {
                        DataType::F16 | DataType::BF16 => 2,
                        DataType::F32 => 4,
                        other => {
                            return Err(EmbeddingError::UnsupportedDataType(
                                other,
                            ));
                        },
                    };
                    let size_bytes =
                        (vocab_size * num_groups * elem_size) as u64;
                    let buf = mtl_context
                        .allocate_buffer(size_bytes)
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

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned().into() };
        let scales_buffer = unsafe { scales.mtl_buffer().to_owned().into() };

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
            EmbeddingError::MetalError(MTLError::Generic(format!(
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
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
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
        let arrays = state.arrays(&[ArrayId::Main, ArrayId::Logits]);
        let batch_size = state.sampling_length();
        if batch_size == 0 {
            return;
        }
        let sampling_start = state.sampling_start();
        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let elem_size = input_array_mut.data_type().size_in_bytes();
        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };
        let a_offset = (sampling_start * self.model_dim * elem_size) as u64;

        let args = QuantizedMatmulArguments {
            a_buffer: input_buffer,
            a_offset,
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
    }
}
