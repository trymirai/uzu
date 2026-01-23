use std::{cell::RefCell, rc::Rc};

use crate::{
    DataType,
    backends::metal::{
        Buffer, MTLCommandBuffer, MTLCommandEncoder,
        MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
        encodable_block::{EncodableBlock, EncodingParameters},
        forward_pass::{ArrayId, ForwardPassState},
        kernel::matmul::{MatmulArguments, MatmulKernel},
    },
    device::array::Array,
    parameters::ParameterTree,
};

pub struct FullPrecisionLinear {
    kernel: RefCell<MatmulKernel>,
    bias_buffer: Option<Buffer>,
    weights_buffer: Buffer,
    input_dim: usize,
    output_dim: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl FullPrecisionLinear {
    pub fn new(
        _mtl_context: &MTLContext,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, MTLError> {
        if !matches!(precision, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for full precision linear kernel: {:?}",
                precision
            )));
        }

        let mut weights = parameter_tree.leaf("weights").map_err(|e| {
            MTLError::Generic(format!("Failed to load weights: {:?}", e))
        })?;

        let w_shape = weights.shape();
        if w_shape != [output_dim, input_dim] {
            return Err(MTLError::Generic(format!(
                "Unexpected weights shape: got {:?}, expected [{}, {}]",
                w_shape, output_dim, input_dim
            )));
        }

        if weights.data_type() != precision {
            return Err(MTLError::Generic(format!(
                "Weights dtype mismatch: got {:?}, expected {:?}",
                weights.data_type(),
                precision
            )));
        }

        let weights_buffer: Buffer =
            unsafe { weights.mtl_buffer() }.to_owned().into();

        let bias_buffer = match parameter_tree.leaf("biases") {
            Ok(mut biases) => {
                if biases.shape() != [output_dim] {
                    return Err(MTLError::Generic(format!(
                        "Bias shape mismatch: got {:?}, expected [{:?}]",
                        biases.shape(),
                        output_dim
                    )));
                }
                if biases.data_type() != precision {
                    return Err(MTLError::Generic(format!(
                        "Bias dtype mismatch: got {:?}, expected {:?}",
                        biases.data_type(),
                        precision
                    )));
                }
                let bias_buffer: Buffer =
                    unsafe { biases.mtl_buffer() }.to_owned().into();
                Some(bias_buffer)
            },
            Err(_) => None,
        };

        let mut kernel = MatmulKernel::new(precision)?;
        kernel.precompile(_mtl_context)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_buffer,
            weights_buffer,
            input_dim,
            output_dim,
            input_array_id,
            output_array_id,
        })
    }
}

impl EncodableBlock for FullPrecisionLinear {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");

        let args = MatmulArguments {
            a: input_buffer,
            a_offset: 0,
            b: &self.weights_buffer,
            c: None,
            d: output_buffer,
            bias: self.bias_buffer.as_ref().map(|v| v.as_ref()),
            batch: batch_size as i32,
            input_dim: self.input_dim as i32,
            output_dim: self.output_dim as i32,
            lda: self.input_dim as i32,
            ldb: self.input_dim as i32,
            ldd: self.output_dim as i32,
            batch_count: 1,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: true,
        };

        let mut kernel = self.kernel.borrow_mut();
        kernel
            .encode(state.mtl_context(), &encoder, args)
            .expect("Failed to encode matmul kernel");

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
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let args = MatmulArguments {
            a: input_buffer,
            a_offset: 0,
            b: &self.weights_buffer,
            c: None,
            d: output_buffer,
            bias: self.bias_buffer.as_ref().map(|v| v.as_ref()),
            batch: batch_size as i32,
            input_dim: self.input_dim as i32,
            output_dim: self.output_dim as i32,
            lda: self.input_dim as i32,
            ldb: self.input_dim as i32,
            ldd: self.output_dim as i32,
            batch_count: 1,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: true,
        };

        let mut kernel = self.kernel.borrow_mut();
        kernel
            .encode(state.mtl_context(), encoder, args)
            .expect("Failed to encode matmul kernel");
    }
}
