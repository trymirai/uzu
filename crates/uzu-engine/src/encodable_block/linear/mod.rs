use derive_more::Debug;
use thiserror::Error;

use self::transformed_linear::InputPreparationMode;

mod input_rht;
mod matmul;
mod transformed_linear;
mod untied_readout;

pub use matmul::{LinearMatmul, LinearMatmulError};
pub use transformed_linear::TransformedLinear;
pub use untied_readout::UntiedReadout;

use crate::{
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{
            ActivationFormat,
            matmul::{MatmulA, MatmulKernel},
        },
    },
    config::weight_matrix::AnyWeightMatrixSpec,
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Matmul error: {0}")]
    Matmul(#[source] <<B::Kernels as Kernels>::MatmulKernel as MatmulKernel>::Error),
}

pub trait Linear<B: Backend>: Send + Sync {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        self.encode_input(MatmulA::full_precision(input), batch_dim, command_buffer)
    }

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>>;

    fn select_activation_format(
        &self,
        _batch_dim: u32,
        _output_dim: Option<u32>,
        _context: &B::Context,
    ) -> ActivationFormat {
        ActivationFormat::Bf16
    }
}

pub type LinearInput<B> = MatmulA<<B as Backend>::ScratchBuffer>;

pub struct LinearInputPreparation<B: Backend> {
    pub rht_signs: B::GlobalGpuBuffer,
    pub activation_format: ActivationFormat,
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error("LinearMatmul error: {0}")]
    LinearMatmulError(#[from] LinearMatmulError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> dyn Linear<B> {
    pub async fn new_mixed_precision(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            spec @ (AnyWeightMatrixSpec::FullPrecisionSpec(_)
            | AnyWeightMatrixSpec::MLXSpec(_)
            | AnyWeightMatrixSpec::IntSpec(_)) => {
                let matrix = LinearMatmul::load(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    spec,
                    None,
                    &weights_tree,
                    has_biases.then_some(parameter_tree),
                    context,
                )
                .await?;
                Ok(Box::new(matrix))
            },
            AnyWeightMatrixSpec::HybridSpec(spec) => {
                let (linear, _) = TransformedLinear::load(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    has_biases.then_some(parameter_tree),
                    spec,
                    &weights_tree,
                    InputPreparationMode::InLinear,
                    context,
                )
                .await?;
                Ok(linear)
            },
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    pub async fn new(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        Self::new_mixed_precision(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            has_biases,
            parameter_tree,
            context,
        )
        .await
    }

    pub async fn new_with_input_rht_mixed_precision(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::GlobalGpuBuffer>), LinearBlockError<B>> {
        let (linear, preparation) = Self::new_with_input_preparation(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            false,
            parameter_tree,
            context,
        )
        .await?;
        Ok((linear, preparation.map(|preparation| preparation.rht_signs)))
    }

    pub async fn new_for_fused_input(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        Self::new_with_input_preparation(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            has_biases,
            true,
            parameter_tree,
            context,
        )
        .await
    }

    pub async fn new_with_input_rht(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::GlobalGpuBuffer>), LinearBlockError<B>> {
        Self::new_with_input_rht_mixed_precision(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            has_biases,
            parameter_tree,
            context,
        )
        .await
    }

    async fn new_with_input_preparation(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        allow_quantized_activations: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            AnyWeightMatrixSpec::HybridSpec(spec) => {
                TransformedLinear::load(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    has_biases.then_some(parameter_tree),
                    spec,
                    &weights_tree,
                    InputPreparationMode::AllowExternal {
                        allow_quantized_activations,
                    },
                    context,
                )
                .await
            },
            _ => {
                let linear = Self::new_mixed_precision(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    has_biases,
                    parameter_tree,
                    context,
                )
                .await?;
                Ok((linear, None))
            },
        }
    }
}
