use thiserror::Error;

use super::{Linear, LinearBlockError};
use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{HadamardTransformKernel, Kernels},
    },
    config::LinearConfig,
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
    input_factors: Allocation<B>,
    inner_linear: Box<dyn Linear<B>>,
    input_dimension: usize,
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        block_size: usize,
        inner_config: &LinearConfig,
        input_dimension: usize,
        output_dimension: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        assert!(block_size == 32, "only block size 32 hadamard is supported");

        let data_type = inner_config.activation_precision().into();
        let input_factors = parameter_tree.leaf("input_factors")?.read_allocation()?;
        let output_factors = parameter_tree.leaf("output_factors")?.read_allocation()?;
        let input_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformKernel::new(context, data_type)
            .map_err(RHTLinearWrapperError::BackendError)?;
        let inner_linear_tree = parameter_tree.subtree("inner_linear")?;
        let inner_linear = <dyn Linear<B>>::new_with_output_hadamard(
            context,
            inner_config,
            &inner_linear_tree,
            output_factors,
            input_dimension,
            output_dimension,
        )
        .map_err(|error| RHTLinearWrapperError::InnerLinearError(Box::new(error)))?;

        Ok(Self {
            input_hadamard_kernel,
            input_factors,
            inner_linear,
            input_dimension,
        })
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode(
        &self,
        context: &B::Context,
        input: &mut Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.input_hadamard_kernel.encode(
            input,
            &self.input_factors,
            self.input_dimension as u32,
            batch_dim as u32,
            encoder,
        );
        self.inner_linear.encode(context, input, batch_dim, encoder)
    }
}
