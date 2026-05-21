use std::cell::RefCell;

use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            ManualKernels,
            matmul::{MatmulArguments, MatmulB, MatmulDOp, MatmulError, MatmulKernel},
        },
    },
    config::QuantizationConfig,
    encodable_block::{
        Linear,
        linear::{LinearBlockError, LinearMatmul, LinearMatmulError},
    },
    prelude::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QLoRALinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] Box<LinearBlockError<B>>),
    #[error("Linear matmul error: {0}")]
    LinearMatmulError(#[from] LinearMatmulError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
}

pub struct QLoRALinearWrapper<B: Backend> {
    base_linear: LinearMatmul<B>,
    adapter_kernel: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    adapter_down: Allocation<B>,
    adapter_up: Allocation<B>,
    input_dim: usize,
    output_dim: usize,
    lora_rank: usize,
    lora_scale: f32,
    data_type: DataType,
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
        output_quantized_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, QLoRALinearWrapperError<B>> {
        let data_type = quantization.activation_precision.into();

        let base_linear = LinearMatmul::quantized(
            context,
            quantization,
            input_dim,
            output_dim,
            parameter_tree,
            output_quantized_hadamard_factors,
        )?;
        let adapter_kernel =
            RefCell::new(<<B::Kernels as ManualKernels>::MatmulKernel as MatmulKernel>::new(context, data_type)?);

        let adapter_down_leaf = parameter_tree.leaf("down_weights")?;
        adapter_down_leaf.validate_shape(&[lora_rank, input_dim], data_type)?;
        let adapter_down = adapter_down_leaf.read_allocation()?;

        let adapter_up_leaf = parameter_tree.leaf("up_weights")?;
        adapter_up_leaf.validate_shape(&[output_dim, lora_rank], data_type)?;
        let adapter_up = adapter_up_leaf.read_allocation()?;

        Ok(Self {
            base_linear,
            adapter_kernel,
            adapter_down,
            adapter_up,
            input_dim,
            output_dim,
            lora_rank,
            lora_scale,
            data_type,
        })
    }
}

impl<B: Backend> Linear<B> for QLoRALinearWrapper<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, <B as Backend>::Error> {
        let mut adapter_kernel = self.adapter_kernel.borrow_mut();
        let mut intermediate =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.lora_rank], self.data_type))?;

        adapter_kernel
            .encode(
                MatmulArguments {
                    a: &input,
                    a_offset: 0,
                    a_prologue: &[],
                    b: MatmulB::FullPrecision {
                        b: &self.adapter_down,
                    },
                    b_offset: 0,
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut intermediate,
                    d_transform: &[],
                    m: batch_dim as u32,
                    n: self.lora_rank as u32,
                    k: self.input_dim as u32,
                },
                encoder,
            )
            .expect("encode failed");

        let mut output = self.base_linear.encode(input, batch_dim, encoder)?;

        adapter_kernel
            .encode(
                MatmulArguments {
                    a: &intermediate,
                    a_offset: 0,
                    a_prologue: &[],
                    b: MatmulB::FullPrecision {
                        b: &self.adapter_up,
                    },
                    b_offset: 0,
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output,
                    d_transform: &[
                        MatmulDOp::Scale {
                            ab_scale: self.lora_scale,
                        },
                        MatmulDOp::Accumulate,
                    ],
                    m: batch_dim as u32,
                    n: self.output_dim as u32,
                    k: self.lora_rank as u32,
                },
                encoder,
            )
            .expect("encode failed");

        Ok(output)
    }
}
