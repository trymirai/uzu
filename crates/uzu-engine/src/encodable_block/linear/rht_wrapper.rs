use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder, gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE},
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        Linear, LinearInput, LinearInputPreparation, LinearMatmul, LinearMatmulError, input_rht::InputRht,
    },
    parameters::{ParameterLoaderError, ParameterTree},
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
            adapter_spec: None,
            incoherence_block_size: Some(block_size),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        }) if *block_size == HADAMARD_TRANSFORM_BLOCK_SIZE
    )
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub(super) fn new(
        context: &B::Context,
        input_dimension: u32,
        output_dimension: u32,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if !has_input_output_rht(&spec) {
            return Err(RHTLinearWrapperError::UnsupportedConfiguration(format!("{spec:?}")));
        }

        let (rht_signs, mut inner_linear) = Self::load_inner_with_output_rht(
            context,
            input_dimension,
            output_dimension,
            has_biases,
            input_data_type,
            output_data_type,
            weights_data_type,
            parameter_tree,
        )?;
        let a8_plan = inner_linear.prepare_a8(context);
        Self::build_self_contained(
            context,
            input_data_type,
            LinearInputPreparation {
                rht_signs,
                a8_plan,
            },
            inner_linear,
        )
    }

    pub(super) fn try_new_with_input_preparation(
        context: &B::Context,
        input_dimension: u32,
        output_dimension: u32,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        allow_prequantized_activation: bool,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Option<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>)>, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if !has_input_output_rht(&spec) {
            return Ok(None);
        }

        let (rht_signs, mut inner_linear) = Self::load_inner_with_output_rht(
            context,
            input_dimension,
            output_dimension,
            has_biases,
            input_data_type,
            output_data_type,
            weights_data_type,
            parameter_tree,
        )?;
        let input_preparation = LinearInputPreparation {
            rht_signs,
            a8_plan: inner_linear.prepare_a8(context),
        };
        if input_preparation.a8_plan.is_some() && !allow_prequantized_activation {
            let wrapper = Self::build_self_contained(context, input_data_type, input_preparation, inner_linear)?;
            Ok(Some((Box::new(wrapper), None)))
        } else {
            Ok(Some((Box::new(inner_linear), Some(input_preparation))))
        }
    }

    fn load_inner_with_output_rht(
        context: &B::Context,
        input_dimension: u32,
        output_dimension: u32,
        has_biases: bool,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Allocation<B>, LinearMatmul<B>), RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let quantized_weights_tree = weights_tree.subtree("quantized");
        let quantization_spec = quantized_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let rht_signs = weights_tree
            .leaf("incoherence_signs.input_signs")?
            .validate(&[input_dimension], DataType::I32)?
            .read_allocation()?;
        let output_factors = weights_tree
            .leaf("incoherence_signs.output_signs")?
            .validate(&[output_dimension], DataType::I32)?
            .read_allocation()?;
        let inner_linear = LinearMatmul::load(
            context,
            quantization_spec,
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            &quantized_weights_tree,
            has_biases.then_some(parameter_tree),
            Some(output_factors),
        )?;
        Ok((rht_signs, inner_linear))
    }

    fn build_self_contained(
        context: &B::Context,
        input_data_type: DataType,
        input_preparation: LinearInputPreparation<B>,
        inner_linear: LinearMatmul<B>,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let input_rht = InputRht::new(context, input_data_type, input_preparation, true)
            .map_err(RHTLinearWrapperError::BackendError)?;

        Ok(Self {
            input_rht,
            inner_linear,
        })
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.encode_input(LinearInput::FullPrecision(input), batch_dim, encoder)
    }

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("linear (rht)");

        let input = match input {
            LinearInput::FullPrecision(input) => input,
            input => {
                let output = self.inner_linear.encode_input(input, batch_dim, encoder);
                encoder.pop_debug_group();
                return output;
            },
        };
        let format = self.inner_linear.select_activation_format(batch_dim, encoder.context());
        let input = self.input_rht.prepare_in_place(input, batch_dim, format, encoder)?;
        let output = self.inner_linear.encode_a(input.as_matmul_a(), batch_dim, None, encoder)?;

        encoder.pop_debug_group();
        Ok(output)
    }
}
