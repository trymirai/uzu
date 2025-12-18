use std::{cell::RefCell, rc::Rc};

use metal::{Buffer as MTLBuffer, CommandBufferRef, ComputeCommandEncoderRef};

use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        encodable_block::{EncodableBlock, EncodingParameters},
        forward_pass::{ArrayId, ForwardPassState},
        kernel::{
            KernelDataType, TensorAddBias,
            matmul::{MatmulArguments, MatmulKernel},
        },
    },
    device::array::Array,
    parameters::ParameterTree,
};

pub struct FullPrecisionLinear {
    kernel: RefCell<MatmulKernel>,
    bias_add_kernel: Option<TensorAddBias>,
    biases_buffer: Option<MTLBuffer>,
    weights_buffer: MTLBuffer,
    input_dim: usize,
    output_dim: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl FullPrecisionLinear {
    pub fn new(
        mtl_context: &MTLContext,
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

        let weights_buffer: MTLBuffer =
            unsafe { weights.mtl_buffer() }.to_owned();

        let (bias_add_kernel, biases_buffer) =
            match parameter_tree.leaf("biases") {
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
                    let bias_add_kernel = Some(TensorAddBias::new(
                        mtl_context,
                        KernelDataType::from(precision),
                    )?);
                    let biases_buffer: MTLBuffer =
                        unsafe { biases.mtl_buffer() }.to_owned();
                    (bias_add_kernel, Some(biases_buffer))
                },
                Err(_) => (None, None),
            };

        let kernel = MatmulKernel::new(mtl_context, precision, false, true)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_add_kernel,
            biases_buffer,
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
        command_buffer: &CommandBufferRef,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let encoder = command_buffer.new_compute_command_encoder();

        let args = MatmulArguments {
            a: input_buffer,
            b: &self.weights_buffer,
            d: output_buffer,
            batch: batch_size as i32,
            input_dim: self.input_dim as i32,
            output_dim: self.output_dim as i32,
            lda: self.input_dim as i32,
            ldb: self.input_dim as i32,
            ldd: self.output_dim as i32,
            batch_count: 1,
        };

        self.kernel
            .borrow_mut()
            .encode(state.mtl_context(), encoder, args)
            .expect("Failed to encode matmul kernel");

        encoder.end_encoding();

        if let (Some(bias_add), Some(bias_buf)) =
            (&self.bias_add_kernel, &self.biases_buffer)
        {
            let total_len = batch_size * self.output_dim;
            bias_add.encode_into_command_buffer(
                &output_buffer,
                bias_buf,
                &output_buffer,
                self.output_dim,
                total_len,
                command_buffer,
                parameters.predicate,
            );
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();

        let mut input_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let input_buffer = unsafe { input_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let args = MatmulArguments {
            a: input_buffer,
            b: &self.weights_buffer,
            d: output_buffer,
            batch: batch_size as i32,
            input_dim: self.input_dim as i32,
            output_dim: self.output_dim as i32,
            lda: self.input_dim as i32,
            ldb: self.input_dim as i32,
            ldd: self.output_dim as i32,
            batch_count: 1,
        };

        self.kernel
            .borrow_mut()
            .encode(state.mtl_context(), encoder, args)
            .expect("Failed to encode matmul kernel");

        if let (Some(bias_add), Some(bias_buf)) =
            (&self.bias_add_kernel, &self.biases_buffer)
        {
            let total_len = batch_size * self.output_dim;
            bias_add.encode_with_encoder(
                &output_buffer,
                bias_buf,
                &output_buffer,
                self.output_dim,
                total_len,
                encoder,
                parameters.predicate,
            );
        }
    }
}
