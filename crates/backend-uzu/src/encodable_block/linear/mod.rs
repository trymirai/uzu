mod matmul;
mod qlora_wrapper;
mod rht_wrapper;

pub use matmul::{LinearMatmul, LinearMatmulError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder, gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE},
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        full_precision_spec::FullPrecisionSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Linear<B: Backend>: Send + Sync {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error("LinearMatmul error: {0}")]
    LinearMatmulError(#[from] LinearMatmulError<B>),
    #[error("Output hadamard linear error: {0}")]
    OutputHadamardLinearError(#[from] OutputHadamardLinearError<B>),
    #[error("QLoRALinearWrapper error: {0}")]
    QLoRALinearWrapperError(#[from] QLoRALinearWrapperError<B>),
    #[error("RHTLinearWrapper error: {0}")]
    RHTLinearWrapperError(#[from] RHTLinearWrapperError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

#[derive(Debug, Error)]
pub enum OutputHadamardLinearError<B: Backend> {
    #[error("LinearMatmul error: {0}")]
    LinearMatmulError(#[from] LinearMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> dyn Linear<B> {
    pub fn new_mixed_precision<const N: usize>(
        input_dimension: usize,
        output_dimensions: [usize; N],
        has_biases: bool,
        context: &B::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        let weights_tree = parameter_tree.subtree("weights")?;
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                layout: Layout::OutputInput,
                ..
            }) => {
                let block = LinearMatmul::full_precision(
                    context,
                    input_dimension,
                    output_dimension_sum,
                    has_biases,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    parameter_tree,
                )?;
                Ok(Box::new(block))
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                adapter_spec: None,
                incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
                ..
            }) => Ok(Box::new(RHTLinearWrapper::new(
                context,
                input_dimension,
                output_dimension_sum,
                has_biases,
                weights_data_type,
                input_data_type,
                output_data_type,
                parameter_tree,
            )?)),
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                adapter_spec: Some(adapter_spec),
                incoherence_block_size,
                incoherence_processing_mode,
                ..
            }) => {
                assert!(!has_biases, "QLoRA linear with biases is not supported");
                let adapter_spec = *adapter_spec;
                let AnyWeightMatrixSpec::LowRankSpec(adapter_spec) = adapter_spec else {
                    return Err(LinearBlockError::UnsupportedConfiguration(format!("{adapter_spec:?}")));
                };
                Ok(Box::new(QLoRALinearWrapper::new(
                    context,
                    *quantization_spec,
                    adapter_spec,
                    incoherence_block_size,
                    incoherence_processing_mode,
                    input_dimension,
                    output_dimension_sum,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    &weights_tree,
                )?))
            },
            spec @ (AnyWeightMatrixSpec::MLXSpec(_) | AnyWeightMatrixSpec::IntSpec(_)) => {
                let block = LinearMatmul::quantized(
                    context,
                    spec,
                    input_dimension,
                    output_dimension_sum,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    &weights_tree,
                    has_biases.then_some(parameter_tree),
                    None,
                )?;
                Ok(Box::new(block))
            },
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    pub fn new<const N: usize>(
        input_dimension: usize,
        output_dimensions: [usize; N],
        has_biases: bool,
        context: &B::Context,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        Self::new_mixed_precision(
            input_dimension,
            output_dimensions,
            has_biases,
            context,
            data_type,
            data_type,
            data_type,
            parameter_tree,
        )
    }

    pub fn new_with_output_hadamard_mixed_precision(
        context: &B::Context,
        parameter_tree: &ParameterTree<B>,
        output_factors: Allocation<B>,
        input_dim: usize,
        output_dim: usize,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Box<dyn Linear<B>>, OutputHadamardLinearError<B>> {
        let weights_tree = parameter_tree.subtree("weights")?.subtree("quantized")?;
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            spec @ (AnyWeightMatrixSpec::MLXSpec(_) | AnyWeightMatrixSpec::IntSpec(_)) => {
                Ok(Box::new(LinearMatmul::quantized(
                    context,
                    spec,
                    input_dim,
                    output_dim,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    &weights_tree,
                    has_biases.then_some(parameter_tree),
                    Some(output_factors),
                )?))
            },
            spec => Err(OutputHadamardLinearError::UnsupportedConfiguration(format!(
                "{spec:?} doesn't support fused output hadamard"
            ))),
        }
    }

    pub fn new_extracting_input_hadamard_mixed_precision<const N: usize>(
        input_dimension: usize,
        output_dimensions: [usize; N],
        has_biases: bool,
        context: &B::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Box<dyn Linear<B>>, Option<Allocation<B>>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        let weights_tree = parameter_tree.subtree("weights")?;
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            adapter_spec: None,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        }) = spec
        {
            let input_factors = weights_tree
                .leaf("incoherence_signs.input_signs")?
                .validate(&[input_dimension], DataType::I32)?
                .read_allocation()?;
            let output_factors = weights_tree
                .leaf("incoherence_signs.output_signs")?
                .validate(&[output_dimension_sum], DataType::I32)?
                .read_allocation()?;
            let inner_linear = Self::new_with_output_hadamard_mixed_precision(
                context,
                parameter_tree,
                output_factors,
                input_dimension,
                output_dimension_sum,
                has_biases,
                weights_data_type,
                input_data_type,
                output_data_type,
            )?;
            Ok((inner_linear, Some(input_factors)))
        } else {
            let linear = Self::new_mixed_precision(
                input_dimension,
                output_dimensions,
                has_biases,
                context,
                weights_data_type,
                input_data_type,
                output_data_type,
                parameter_tree,
            )?;
            Ok((linear, None))
        }
    }

    pub fn new_extracting_input_hadamard<const N: usize>(
        input_dimension: usize,
        output_dimensions: [usize; N],
        has_biases: bool,
        context: &B::Context,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Box<dyn Linear<B>>, Option<Allocation<B>>), LinearBlockError<B>> {
        Self::new_extracting_input_hadamard_mixed_precision(
            input_dimension,
            output_dimensions,
            has_biases,
            context,
            data_type,
            data_type,
            data_type,
            parameter_tree,
        )
    }
}
