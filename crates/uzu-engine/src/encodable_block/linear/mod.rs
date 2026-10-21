use derive_more::Debug;
use thiserror::Error;

mod input_rht;
mod matmul;
mod rht_wrapper;
mod untied_readout;

pub use matmul::{LinearMatmul, LinearMatmulError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
pub use untied_readout::UntiedReadout;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, Kernels,
        gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationQuantization,
            matmul::{ActivationFormat, Int8CodeLayout, MatmulA, MatmulKernel},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
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
        self.encode_input(LinearInput::FullPrecision(input), batch_dim, command_buffer)
    }

    #[allow(dead_code)]
    fn encode_ref(
        &self,
        input: &B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>>;

    #[allow(dead_code)]
    fn encode_prerotated_bf16(
        &self,
        rotated_input: &B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        self.encode_ref(rotated_input, batch_dim, command_buffer)
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
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            spec @ (AnyWeightMatrixSpec::FullPrecisionSpec(_)
            | AnyWeightMatrixSpec::MLXSpec(_)
            | AnyWeightMatrixSpec::IntSpec(_)) => {
                let block = LinearMatmul::load(
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
                )?;
                Ok(Box::new(block))
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                incoherence_block_size: Some(block_size),
                incoherence_processing_mode: IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput,
                ..
            }) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => Ok(Box::new(RHTLinearWrapper::new(
                input_dimension,
                output_dimension,
                weights_data_type,
                input_data_type,
                output_data_type,
                has_biases,
                parameter_tree,
                context,
            )?)),
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    pub fn new(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<B>,
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
    }

    pub fn new_with_input_rht_mixed_precision(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::GlobalBuffer>), LinearBlockError<B>> {
        if let Some(linear) = RHTLinearWrapper::try_new_with_input_preparation(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            false,
            parameter_tree,
            context,
        )? {
            return Ok((linear.0, linear.1.map(|preparation| preparation.rht_signs)));
        }

        let linear = Self::new_mixed_precision(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            parameter_tree,
            context,
        )?;
        Ok((linear, None))
    }

    pub fn new_for_fused_input(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        if let Some(linear) = RHTLinearWrapper::try_new_with_input_preparation(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            has_biases,
            true,
            parameter_tree,
            context,
        )? {
            return Ok(linear);
        }

        let linear = Self::new_mixed_precision(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            has_biases,
            parameter_tree,
            context,
        )?;
        Ok((linear, None))
    }

    pub fn new_with_input_rht(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::GlobalBuffer>), LinearBlockError<B>> {
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
    }
}
