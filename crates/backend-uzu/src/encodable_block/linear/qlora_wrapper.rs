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
        linear::{LinearBlockError, LoraAdapter, QuantizedLinear, QuantizedLinearError},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterLeaf,
    prelude::{ParameterLoaderError, ParameterTree},
};

fn load_tensor<'f, 'c, 'l, B: Backend>(
    leaf: ParameterLeaf<'f, 'c, 'l, B::Context>,
    expected_shape: [usize; 2],
    expected_data_type: DataType,
) -> Result<B::Buffer, QLoRALinearWrapperError<B>> {
    if (leaf.shape(), leaf.data_type()) != (expected_shape.as_ref(), expected_data_type) {
        return Err(QLoRALinearWrapperError::InvalidTensor {
            shape: leaf.shape().into(),
            data_type: leaf.data_type(),
            expected_shape: expected_shape.into(),
            expected_data_type,
        });
    }
    Ok(leaf.read_buffer()?)
}

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
    adapter_down: B::Buffer,
    input_dim: usize,
    output_dim: usize,
    lora_rank: usize,
    lora_scale: f32,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    rms_norm_fuses_a_down: bool,
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
        output_quantized_hadamard_factors: Option<B::Buffer>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        rms_norm_fuses_a_down: bool,
    ) -> Result<Self, QLoRALinearWrapperError<B>> {
        let data_type = quantization.activation_precision.into();

        let adapter_kernel =
            RefCell::new(<<B::Kernels as ManualKernels>::MatmulKernel as MatmulKernel>::new(context, data_type)?);

        let adapter_down = load_tensor::<B>(parameter_tree.leaf("down_weights")?, [lora_rank, input_dim], data_type)?;
        let adapter_up = load_tensor::<B>(parameter_tree.leaf("up_weights")?, [output_dim, lora_rank], data_type)?;

        let base_linear = QuantizedLinear::new(
            context,
            quantization,
            input_dim,
            output_dim,
            parameter_tree,
            input_array_id,
            output_array_id,
            output_quantized_hadamard_factors,
            Some(LoraAdapter {
                buffer: adapter_up,
                scale: lora_scale,
                rank: lora_rank as u32,
            }),
        )?;

        Ok(Self {
            base_linear,
            adapter_kernel,
            adapter_down,
            input_dim,
            output_dim,
            lora_rank,
            lora_scale,
            input_array_id,
            output_array_id,
            rms_norm_fuses_a_down,
        })
    }
}

impl<B: Backend> Linear<B> for QLoRALinearWrapper<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), <B as Backend>::Error> {
        let batch_dim = state.active_row_count();

        if !self.rms_norm_fuses_a_down {
            let intermediate_array = state.common_aux.lora_intermediate.as_ref().unwrap();
            let input_buf_rc = state.array(self.input_array_id).buffer();
            let mut adapter_kernel = self.adapter_kernel.borrow_mut();

            adapter_kernel.encode(
                state.context(),
                MatmulArguments {
                    a: input_buf_rc.borrow().deref(),
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
        }

        self.base_linear.encode(state, encoder)?;

        if !self.base_linear.use_qmv_fast_fuse_lora_a_up(batch_dim) {
            let mut adapter_kernel = self.adapter_kernel.borrow_mut();
            let intermediate_array = state.common_aux.lora_intermediate.as_ref().unwrap();
            let output_array = state.array(self.output_array_id);

            adapter_kernel.encode(
                state.context(),
                MatmulArguments {
                    a: intermediate_array.buffer().borrow().deref(),
                    a_offset: 0,
                    b: self.base_linear.lora_adapter_up().expect("A_up buffer missing"),
                    ab_scale: self.lora_scale,
                    c: MatmulArgumentC::Accumulate,
                    d: output_array.buffer().borrow_mut().deref_mut(),
                    batch_dim: batch_dim as u32,
                    input_dim: self.lora_rank as u32,
                    output_dim: self.output_dim as u32,
                },
                encoder,
            );
        }

        Ok(())
    }
}
