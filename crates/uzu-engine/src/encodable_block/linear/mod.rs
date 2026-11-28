use derive_more::Debug;
use thiserror::Error;

mod ane_weights;
mod input_preparation;
mod matmul;
mod output_stack;

pub use ane_weights::AneWeightLayer;
pub use input_preparation::InputRhtParameters;
pub use matmul::{LinearMatmul, LinearMatmulError};
pub use output_stack::{OutputRows, OutputStack};

use crate::{
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{
            ActivationFormat,
            matmul::{MatmulA, MatmulKernel},
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, output_stack::OutputStackError},
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Matmul error: {0}")]
    Matmul(#[source] <<B::Kernels as Kernels>::MatmulKernel as MatmulKernel>::Error),
    #[error("Unsupported linear input: {0}")]
    UnsupportedInput(&'static str),
    #[error("the selected linear input basis is no longer available")]
    InputBasisUnavailable,
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

    fn encode_for_basis(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        basis: InputBasis,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        if basis == InputBasis::AneUnrotated {
            return Err(LinearEncodeError::InputBasisUnavailable);
        }
        self.encode(input, batch_dim, command_buffer)
    }

    fn select_activation_format(
        &self,
        _batch_dim: u32,
        _output_dim: Option<u32>,
        _context: &B::Context,
    ) -> ActivationFormat {
        ActivationFormat::Bf16
    }

    fn input_basis(
        &self,
        _batch_dim: u32,
    ) -> InputBasis {
        InputBasis::GpuExpected
    }

    fn supports_unrotated_input(&self) -> bool {
        false
    }

    /// The compact projection and its source rotations, if the accelerator can restage this layer.
    /// A layer that takes no input rotation itself uses `fused_input_rht` from the preceding block.
    fn ane_weights<'a>(
        &'a self,
        _fused_input_rht: Option<(u32, &'a B::GlobalGpuBuffer)>,
    ) -> Option<AneWeightLayer<'a, B>> {
        None
    }
}

pub type LinearInput<B> = MatmulA<<B as Backend>::ScratchBuffer>;

pub enum InputPreparationMode {
    InLinear,
    AllowExternal {
        allow_quantized_activations: bool,
    },
}

pub enum InputAccess {
    MayOverwrite,
    Preserve,
}

pub enum OutputBasis {
    Original,
    Rotated,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputBasis {
    GpuExpected,
    AneUnrotated,
}

pub struct LinearInputPreparation<B: Backend> {
    pub rht: InputRhtParameters<B>,
    pub activation_format: ActivationFormat,
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error(transparent)]
    OutputStack(#[from] OutputStackError),
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
        output_basis: OutputBasis,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let (linear, _) = Self::new_with_input_preparation(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            InputPreparationMode::InLinear,
            output_basis,
            parameter_tree,
            context,
        )
        .await?;
        Ok(linear)
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
            OutputBasis::Original,
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
    ) -> Result<(Box<dyn Linear<B>>, Option<InputRhtParameters<B>>), LinearBlockError<B>> {
        let (linear, preparation) = Self::new_with_input_preparation(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            InputPreparationMode::AllowExternal {
                allow_quantized_activations: false,
            },
            OutputBasis::Original,
            parameter_tree,
            context,
        )
        .await?;
        Ok((linear, preparation.map(|preparation| preparation.rht)))
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
            InputPreparationMode::AllowExternal {
                allow_quantized_activations: true,
            },
            OutputBasis::Original,
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
    ) -> Result<(Box<dyn Linear<B>>, Option<InputRhtParameters<B>>), LinearBlockError<B>> {
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
        input_preparation_mode: InputPreparationMode,
        output_basis: OutputBasis,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if matches!(output_basis, OutputBasis::Rotated) && !matches!(spec, AnyWeightMatrixSpec::HybridSpec(_)) {
            return Err(LinearBlockError::UnsupportedConfiguration("expected input-output Hadamard transform".into()));
        }
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
                Ok((Box::new(matrix), None))
            },
            spec @ (AnyWeightMatrixSpec::OutputStackSpec(_) | AnyWeightMatrixSpec::HybridSpec(_)) => {
                let (linear, preparation) = OutputStack::load(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    spec,
                    &weights_tree,
                    has_biases.then_some(parameter_tree),
                    input_preparation_mode,
                    InputAccess::MayOverwrite,
                    output_basis,
                    context,
                )
                .await?;
                Ok((Box::new(linear), preparation))
            },
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }
}
