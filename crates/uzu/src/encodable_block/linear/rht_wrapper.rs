use std::ops::DerefMut;

use thiserror::Error;

use super::{Linear, LinearBlockError};
use crate::{
    backends::common::{
        Backend, Encoder,
        kernel::{HadamardTransformKernel, Kernels},
    },
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RHTLinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] Box<LinearBlockError<B>>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

pub struct RHTLinearWrapper<B: Backend> {
    input_hadamard_kernel: <B::Kernels as Kernels>::HadamardTransformKernel,
    input_factors: B::Buffer,
    inner_linear: Box<dyn Linear<B>>,
    input_dimension: usize,
    input_array_id: ArrayId,
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        block_size: usize,
        inner_config: &LinearConfig,
        input_dimension: usize,
        output_dimension: usize,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        assert!(block_size == 32, "only block size 32 hadamard is supported");

        let data_type = inner_config.activation_precision().into();

        let input_factors_leaf = parameter_tree.leaf("input_factors")?;
        let output_factors_leaf = parameter_tree.leaf("output_factors")?;

        let input_factors = input_factors_leaf.read_buffer()?;
        let output_factors = output_factors_leaf.read_buffer()?;

        let input_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformKernel::new(context, data_type)
            .map_err(RHTLinearWrapperError::BackendError)?;

        let inner_linear_tree = parameter_tree.subtree("inner_linear")?;

        let inner_linear = <dyn Linear<B>>::new_with_output_hadamard(
            context,
            inner_config,
            input_array_id,
            output_array_id,
            &inner_linear_tree,
            output_factors,
            input_dimension,
            output_dimension,
        )
        .map_err(|e| RHTLinearWrapperError::InnerLinearError(Box::new(e)))?;

        Ok(Self {
            input_hadamard_kernel,
            input_factors,
            inner_linear,
            input_dimension,
            input_array_id,
        })
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        // Input Hadamard
        let input_array = state.array(self.input_array_id);
        self.input_hadamard_kernel.encode(
            input_array.buffer().borrow_mut().deref_mut(),
            &self.input_factors,
            self.input_dimension as u32,
            state.active_row_count() as u32,
            encoder,
        );

        // Matmul with fused output Hadamard
        self.inner_linear.encode(state, encoder)
    }
}
