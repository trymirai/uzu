use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearEncodeError, LinearInput, LinearInputPreparation, LinearMatmul, LinearMatmulError,
        input_rht::InputRht,
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RHTLinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] LinearMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported RHT linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct RHTLinearWrapper<B: Backend> {
    input_rht: InputRht<B>,
    inner_linear: LinearMatmul<B>,
}

fn has_input_output_rht(spec: &AnyWeightMatrixSpec) -> bool {
    matches!(
        spec,
        AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            incoherence_block_size: Some(block_size),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        }) if *block_size == HADAMARD_TRANSFORM_BLOCK_SIZE
    )
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub(super) async fn new(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if !has_input_output_rht(&spec) {
            return Err(RHTLinearWrapperError::UnsupportedConfiguration(format!("{spec:?}")));
        }

        let (rht_signs, mut inner_linear) = Self::load_inner_with_output_rht(
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
        let activation_quantization =
            inner_linear.optimize_weights_layout(context).map_err(RHTLinearWrapperError::BackendError)?;
        Self::build_self_contained(
            input_data_type,
            LinearInputPreparation {
                rht_signs,
                activation_quantization,
            },
            inner_linear,
            context,
        )
    }

    pub(super) async fn try_new_with_input_preparation(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        allow_prequantized_activation: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Option<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>)>, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if !has_input_output_rht(&spec) {
            return Ok(None);
        }

        let (rht_signs, mut inner_linear) = Self::load_inner_with_output_rht(
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
        let input_preparation = LinearInputPreparation {
            rht_signs,
            activation_quantization: inner_linear
                .optimize_weights_layout(context)
                .map_err(RHTLinearWrapperError::BackendError)?,
        };
        if input_preparation.activation_quantization.is_some() && !allow_prequantized_activation {
            let wrapper = Self::build_self_contained(input_data_type, input_preparation, inner_linear, context)?;
            Ok(Some((Box::new(wrapper), None)))
        } else {
            Ok(Some((Box::new(inner_linear), Some(input_preparation))))
        }
    }

    async fn load_inner_with_output_rht(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(B::GlobalGpuBuffer, LinearMatmul<B>), RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let quantized_weights_tree = weights_tree.subtree("quantized");
        let quantization_spec = quantized_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let rht_signs =
            weights_tree.read_buffer("incoherence_signs.input_signs", &[input_dimension], DataType::I32).await?;
        let output_factors =
            weights_tree.read_buffer("incoherence_signs.output_signs", &[output_dimension], DataType::I32).await?;
        let inner_linear = LinearMatmul::load(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            quantization_spec,
            Some(output_factors),
            &quantized_weights_tree,
            has_biases.then_some(parameter_tree),
            context,
        )
        .await?;
        Ok((rht_signs, inner_linear))
    }

    fn build_self_contained(
        input_data_type: DataType,
        input_preparation: LinearInputPreparation<B>,
        inner_linear: LinearMatmul<B>,
        context: &B::Context,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let input_rht = InputRht::new(input_data_type, input_preparation, true, context)
            .map_err(RHTLinearWrapperError::BackendError)?;

        Ok(Self {
            input_rht,
            inner_linear,
        })
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let input = match input {
            LinearInput::FullPrecision(input) => input,
            input => return self.inner_linear.encode_input(input, batch_dim, command_buffer),
        };
        let command_buffer = &mut *command_buffer.debug_group("linear (rht)");

        let format = self.inner_linear.select_activation_format(batch_dim, command_buffer.context());
        let input = self
            .input_rht
            .prepare_in_place(input, batch_dim, format, command_buffer)
            .map_err(LinearEncodeError::Backend)?;
        let output = self.inner_linear.encode_with_a(
            input.as_matmul_a(),
            batch_dim,
            None::<Gather<&B::ScratchBuffer>>,
            command_buffer,
        )?;

        Ok(output)
    }
}
