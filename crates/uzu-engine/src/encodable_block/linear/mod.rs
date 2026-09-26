use thiserror::Error;

mod input_rht;
mod matmul;
mod qlora_wrapper;
mod rht_wrapper;
mod untied_readout;

pub use matmul::{LinearMatmul, LinearMatmulError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
pub use untied_readout::UntiedReadout;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer,
        gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationQuantization,
            matmul::{ActivationFormat, Int8CodeLayout, MatmulA},
        },
    },
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
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error>;

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        match input {
            LinearInput::FullPrecision(input) => self.encode(input, batch_dim, command_buffer),
            LinearInput::Int8Symmetric {
                ..
            } => {
                panic!("linear does not support pre-quantized activations")
            },
        }
    }

    fn select_activation_format(
        &self,
        _batch_dim: u32,
        _context: &B::Context,
    ) -> ActivationFormat {
        ActivationFormat::Bf16
    }
}

pub enum LinearInput<B: Backend> {
    FullPrecision(B::ScratchBuffer),
    Int8Symmetric {
        values: B::ScratchBuffer,
        scales: B::ScratchBuffer,
        group_sums: Option<B::ScratchBuffer>,
        scale_group_size: u32,
        code_layout: Int8CodeLayout,
    },
}

pub struct Gather<T: BufferRef> {
    pub indices: T,
    pub output_dim: u32,
}

impl<B: Backend> LinearInput<B> {
    fn as_matmul_a(&self) -> MatmulA<&B::ScratchBuffer> {
        match self {
            Self::FullPrecision(values) => MatmulA::FullPrecision {
                values,
                offset: 0,
            },
            Self::Int8Symmetric {
                values,
                scales,
                group_sums,
                scale_group_size,
                code_layout,
            } => MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums: group_sums.as_ref(),
                scale_group_size: *scale_group_size,
                code_layout: *code_layout,
            },
        }
    }
}

pub struct LinearInputPreparation<B: Backend> {
    pub rht_signs: B::GlobalBuffer,
    pub activation_quantization: Option<ActivationQuantization>,
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
    pub fn new_mixed_precision(
        input_dimension: u32,
        output_dimensions: impl AsRef<[u32]>,
        has_biases: bool,
        context: &B::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let output_dimension_sum: u32 = output_dimensions.as_ref().iter().sum();
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
                incoherence_block_size: Some(block_size),
                incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
                ..
            }) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => Ok(Box::new(RHTLinearWrapper::new(
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

    pub fn new(
        input_dimension: u32,
        output_dimensions: impl AsRef<[u32]>,
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

    pub fn new_with_input_rht_mixed_precision(
        input_dimension: u32,
        output_dimensions: impl AsRef<[u32]>,
        has_biases: bool,
        context: &B::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::GlobalBuffer>), LinearBlockError<B>> {
        let output_dimension_sum: u32 = output_dimensions.as_ref().iter().sum();
        if let Some(linear) = RHTLinearWrapper::try_new_with_input_preparation(
            context,
            input_dimension,
            output_dimension_sum,
            has_biases,
            weights_data_type,
            input_data_type,
            output_data_type,
            false,
            parameter_tree,
        )? {
            return Ok((linear.0, linear.1.map(|preparation| preparation.rht_signs)));
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

    pub fn new_for_fused_input(
        input_dimension: u32,
        output_dimensions: impl AsRef<[u32]>,
        has_biases: bool,
        context: &B::Context,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        let output_dimension_sum: u32 = output_dimensions.as_ref().iter().sum();
        if let Some(linear) = RHTLinearWrapper::try_new_with_input_preparation(
            context,
            input_dimension,
            output_dimension_sum,
            has_biases,
            data_type,
            data_type,
            data_type,
            true,
            parameter_tree,
        )? {
            return Ok(linear);
        }

        let linear = Self::new_mixed_precision(
            input_dimension,
            output_dimensions,
            has_biases,
            context,
            data_type,
            data_type,
            data_type,
            parameter_tree,
        )?;
        Ok((linear, None))
    }

    pub fn new_with_input_rht(
        input_dimension: u32,
        output_dimensions: impl AsRef<[u32]>,
        has_biases: bool,
        context: &B::Context,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::GlobalBuffer>), LinearBlockError<B>> {
        Self::new_with_input_rht_mixed_precision(
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
