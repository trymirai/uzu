use std::{cell::RefCell, rc::Rc};

use metal::Buffer as MTLBuffer;
use mpsgraph::CommandBuffer;

use super::{
    super::{EncodableBlock, EncodingParameters},
    QuantizedEmbeddingError,
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
    weights_buffer: MTLBuffer,
    vocab_size: usize,
    model_dim: usize,
}

impl FullPrecisionEmbeddingReadout {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, QuantizedEmbeddingError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(QuantizedEmbeddingError::UnsupportedDataType(
                data_type,
            ));
        }

        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("output_weights").map_err(|e| {
                QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load weights: {:?}",
                    e
                )))
            })?,
        };

        if weights.shape() != [vocab_size, model_dim] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding readout weights shape mismatch: got {:?}, expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim
                )),
            ));
        }

        if weights.data_type() != data_type {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Weights dtype mismatch: got {:?}, expected {:?}",
                    weights.data_type(),
                    data_type
                )),
            ));
        }

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };

        // Weights are [vocab_size, model_dim], we compute input @ weights^T
        // MatmulKernel with transpose_b=true handles this
        let kernel = MatmulKernel::new(mtl_context, data_type, false, true)
            .map_err(QuantizedEmbeddingError::MetalError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights_buffer,
            vocab_size,
            model_dim,
        })
    }
}

impl EncodableBlock for FullPrecisionEmbeddingReadout {
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

        let args = MatmulArguments {
            a: input_buffer,
            b: &self.weights_buffer,
            d: output_buffer,
            batch: batch_size as i32,
            input_dim: self.model_dim as i32,
            output_dim: self.vocab_size as i32,
            lda: self.model_dim as i32,
            ldb: self.model_dim as i32,
            ldd: self.vocab_size as i32,
            batch_count: 1,
        };

        self.kernel
            .borrow_mut()
            .encode(state.mtl_context(), encoder, args)
            .expect("Failed to encode full precision embedding readout kernel");

        encoder.end_encoding();

        if parameters.wait_until_completed {
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
