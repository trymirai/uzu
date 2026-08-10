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
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Linear<B: Backend>: Send + Sync {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error("LinearMatmul error: {0}")]
    LinearMatmulError(#[from] LinearMatmulError<B>),
    #[error("QLoRALinearWrapper error: {0}")]
    QLoRALinearWrapperError(#[from] QLoRALinearWrapperError<B>),
    #[error("RHTLinearWrapper error: {0}")]
    RHTLinearWrapperError(#[from] RHTLinearWrapperError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> dyn Linear<B> {
    pub fn new_mixed_precision<const N: usize>(
        input_dimension: u32,
        output_dimensions: [u32; N],
        has_biases: bool,
        context: &B::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let output_dimension_sum: u32 = output_dimensions.iter().sum();
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            spec @ (AnyWeightMatrixSpec::FullPrecisionSpec(_)
            | AnyWeightMatrixSpec::MLXSpec(_)
            | AnyWeightMatrixSpec::IntSpec(_)) => {
                let block = LinearMatmul::load(
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
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    pub fn new<const N: usize>(
        input_dimension: u32,
        output_dimensions: [u32; N],
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

    pub fn new_extracting_input_hadamard_mixed_precision<const N: usize>(
        input_dimension: u32,
        output_dimensions: [u32; N],
        has_biases: bool,
        context: &B::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Box<dyn Linear<B>>, Option<Allocation<B>>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        if let Some(linear) = RHTLinearWrapper::new_allowing_predecessor_rht(
            context,
            input_dimension,
            output_dimension_sum,
            has_biases,
            weights_data_type,
            input_data_type,
            output_data_type,
            parameter_tree,
        )? {
            return Ok(linear);
        }

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

    pub fn new_extracting_input_hadamard<const N: usize>(
        input_dimension: u32,
        output_dimensions: [u32; N],
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
