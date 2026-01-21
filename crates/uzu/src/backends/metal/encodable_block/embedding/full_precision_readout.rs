use std::{cell::RefCell, rc::Rc};

use crate::backends::metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, MTLCommandBuffer,
    MTLCommandEncoder,
};

use super::{
    super::{EncodableBlock, EncodingParameters},
    EmbeddingError,
};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::matmul::{MatmulArguments, MatmulKernel},
    },
    parameters::ParameterTree,
};

pub struct FullPrecisionEmbeddingReadout {
    kernel: RefCell<MatmulKernel>,
    weights_buffer: Buffer,
    vocab_size: usize,
    model_dim: usize,
}

impl FullPrecisionEmbeddingReadout {
    pub fn new(
        _mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, EmbeddingError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(EmbeddingError::UnsupportedDataType(data_type));
        }

        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("output_weights").map_err(|e| {
                EmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load weights: {:?}",
                    e
                )))
            })?,
        };

        if weights.shape() != [vocab_size, model_dim] {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Embedding readout weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim
                ),
            )));
        }

        if weights.data_type() != data_type {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Weights dtype mismatch: got {:?}, expected {:?}",
                    weights.data_type(),
                    data_type
                ),
            )));
        }

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };

        let mut kernel =
            MatmulKernel::new(data_type).map_err(EmbeddingError::MetalError)?;
        kernel.precompile(_mtl_context).map_err(EmbeddingError::MetalError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights_buffer: weights_buffer.into(),
            vocab_size,
            model_dim,
        })
    }
}

impl EncodableBlock for FullPrecisionEmbeddingReadout {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: CommandBufferRef<'_>,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer.new_compute_command_encoder()
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
        encoder: ComputeCommandEncoderRef<'_>,
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

        let args = MatmulArguments {
            a: input_buffer,
            a_offset,
            b: &self.weights_buffer,
            c: None,
            d: output_buffer,
            bias: None,
            batch: batch_size as i32,
            input_dim: self.model_dim as i32,
            output_dim: self.vocab_size as i32,
            lda: self.model_dim as i32,
            ldb: self.model_dim as i32,
            ldd: self.vocab_size as i32,
            batch_count: 1,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: true,
        };

        self.kernel
            .borrow_mut()
            .encode(state.mtl_context(), encoder, args)
            .expect("Failed to encode full precision embedding readout kernel");
    }
}
