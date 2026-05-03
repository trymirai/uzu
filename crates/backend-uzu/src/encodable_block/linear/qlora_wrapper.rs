use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::{
            ManualKernels,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
        },
    },
    config::QuantizationConfig,
    encodable_block::{
        Linear,
        linear::{LinearBlockError, QuantizedLinear, QuantizedLinearError},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    prelude::{ParameterLeaf, ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QLoRALinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] Box<LinearBlockError<B>>),
    #[error("Quantized linear error: {0}")]
    QuantizedLinearError(#[from] QuantizedLinearError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[usize]>,
        data_type: DataType,
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
    },
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
}

pub struct QLoRALinearWrapper<B: Backend> {
    base_linear: QuantizedLinear<B>,
    adapter_kernel: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    adapter_down: B::DenseBuffer,
    adapter_up: B::DenseBuffer,
    input_dim: usize,
    output_dim: usize,
    lora_rank: usize,
    lora_scale: f32,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

// TODO: figure out how to make this generic over QLoRAWrapperError::InvalidTensor or make one global "Invalid Tensor" error and make this a common helper
fn validate_tensor<'file, 'context, 'leaf, B: Backend>(
    weights_leaf: &ParameterLeaf<'file, 'context, 'leaf, B::Context>,
    expected_shape: [usize; 2],
    expected_data_type: DataType,
) -> Result<(), QLoRALinearWrapperError<B>> {
    let shape = weights_leaf.shape();
    let data_type = weights_leaf.data_type();

    if (shape, data_type) != (expected_shape.as_ref(), expected_data_type) {
        return Err(QLoRALinearWrapperError::InvalidTensor {
            shape: shape.into(),
            data_type: weights_leaf.data_type(),
            expected_shape: expected_shape.into(),
            expected_data_type,
        });
    }

    Ok(())
}

impl<B: Backend> QLoRALinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        quantization: &QuantizationConfig,
        lora_rank: usize,
        lora_scale: f32,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
        output_quantized_hadamard_factors: Option<B::DenseBuffer>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, QLoRALinearWrapperError<B>> {
        let data_type = quantization.activation_precision.into();

        let base_linear = QuantizedLinear::new(
            context,
            quantization,
            input_dim,
            output_dim,
            parameter_tree,
            input_array_id,
            output_array_id,
            output_quantized_hadamard_factors,
        )?;
        let adapter_kernel =
            RefCell::new(<<B::Kernels as ManualKernels>::MatmulKernel as MatmulKernel>::new(context, data_type)?);

        let adapter_down_leaf = parameter_tree.leaf("down_weights")?;
        validate_tensor(&adapter_down_leaf, [lora_rank as usize, input_dim as usize], data_type)?;
        let adapter_down = adapter_down_leaf.read_buffer()?;

        let adapter_up_leaf = parameter_tree.leaf("up_weights")?;
        validate_tensor(&adapter_up_leaf, [output_dim, lora_rank as usize], data_type)?;
        let adapter_up = adapter_up_leaf.read_buffer()?;

        Ok(Self {
            base_linear,
            adapter_kernel,
            adapter_down,
            adapter_up,
            input_dim,
            output_dim,
            lora_rank,
            lora_scale,
            input_array_id,
            output_array_id,
        })
    }
}

impl<B: Backend> Linear<B> for QLoRALinearWrapper<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), <B as Backend>::Error> {
        self.base_linear.encode(state, encoder)?;

        let mut adapter_kernel = self.adapter_kernel.borrow_mut();
        let batch_dim = state.active_row_count();

        let intermediate_array = state.common_aux.lora_intermediate.as_ref().unwrap();
        let input_array = state.array(self.input_array_id);
        let output_array = state.array(self.output_array_id);

        adapter_kernel.encode(
            state.context(),
            MatmulArguments {
                a: input_array.buffer().borrow().deref(),
                a_offset: 0,
                b: &self.adapter_down,
                ab_scale: 1.0,
                c: MatmulArgumentC::None,
                d: intermediate_array.buffer().borrow_mut().deref_mut(),
                batch_dim: batch_dim as u32,
                input_dim: self.input_dim as u32,
                output_dim: self.lora_rank as u32,
            },
            encoder,
        );

        adapter_kernel.encode(
            state.context(),
            MatmulArguments {
                a: intermediate_array.buffer().borrow().deref(),
                a_offset: 0,
                b: &self.adapter_up,
                ab_scale: self.lora_scale,
                c: MatmulArgumentC::Accumulate,
                d: output_array.buffer().borrow_mut().deref_mut(),
                batch_dim: batch_dim as u32,
                input_dim: self.lora_rank as u32,
                output_dim: self.output_dim as u32,
            },
            encoder,
        );

        Ok(())
    }
}
