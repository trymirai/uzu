use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationTransform,
            matmul::{A8ActivationPlan, ActivationFormat, MatmulA},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{Linear, LinearInput, LinearInputPreparation, LinearMatmul, LinearMatmulError},
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
    bf16_input_rht: ActivationTransform<B>,
    a8_input_rht: Option<ActivationTransform<B>>,
    input_factors: Allocation<B>,
    inner_linear: LinearMatmul<B>,
    input_dimension: u32,
}

fn has_input_output_rht(spec: &AnyWeightMatrixSpec) -> bool {
    matches!(
        spec,
        AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            adapter_spec: None,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        })
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

        let (input_factors, mut inner_linear) = Self::load_inner_with_output_rht(
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
        Self::build_self_contained(context, input_dimension, input_data_type, input_factors, inner_linear, a8_plan)
    }

    pub(super) fn try_new_with_input_preparation(
        context: &B::Context,
        input_dimension: usize,
        output_dimension: usize,
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

        let (input_factors, mut inner_linear) = Self::load_inner_with_output_rht(
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
        if let Some(a8_plan) = a8_plan
            && !allow_prequantized_activation
        {
            let wrapper = Self::build_self_contained(
                context,
                input_dimension,
                input_data_type,
                input_factors,
                inner_linear,
                Some(a8_plan),
            )?;
            Ok(Some((Box::new(wrapper), None)))
        } else {
            Ok(Some((
                Box::new(inner_linear),
                Some(LinearInputPreparation {
                    input_factors,
                    a8_plan,
                }),
            )))
        }
    }

    fn load_inner_with_output_rht(
        context: &B::Context,
        input_dimension: usize,
        output_dimension: usize,
        has_biases: bool,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<(Allocation<B>, LinearMatmul<B>), RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let quantized_weights_tree = weights_tree.subtree("quantized");
        let quantization_spec = quantized_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let input_factors = weights_tree
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
        Ok((input_factors, inner_linear))
    }

    fn build_self_contained(
        context: &B::Context,
        input_dimension: usize,
        input_data_type: DataType,
        input_factors: Allocation<B>,
        inner_linear: LinearMatmul<B>,
        a8_plan: Option<A8ActivationPlan>,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let input_transform = ActivationTransform::input_rht(context, input_data_type, true)
            .map_err(RHTLinearWrapperError::BackendError)?;
        let a8_input_rht = a8_plan
            .map(|plan| {
                ActivationTransform::quantize(
                    context,
                    input_data_type,
                    plan.activation_group_size as usize,
                    plan.sum_group_size,
                )
            })
            .transpose()
            .map_err(RHTLinearWrapperError::BackendError)?;

        Ok(Self {
            bf16_input_rht: input_transform,
            a8_input_rht,
            input_factors,
            inner_linear,
            input_dimension,
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
        batch_dim: usize,
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

        if let Some(a8_input_rht) = &self.a8_input_rht
            && self.inner_linear.select_activation_format(batch_dim, encoder.context()) == ActivationFormat::Int8
        {
            let activation_group_size = a8_input_rht.activation_group_size();
            let scale_groups_per_row = self.input_dimension.div_ceil(activation_group_size);
            let mut values =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dimension], DataType::I8))?;
            let mut scales =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, scale_groups_per_row], DataType::F32))?;
            let mut group_sums = a8_input_rht
                .sum_group_size()
                .map(|group_size| self.input_dimension.div_ceil(group_size as usize))
                .map(|groups| encoder.allocate_scratch(size_for_shape(&[batch_dim, groups], DataType::I32)))
                .transpose()?;

            a8_input_rht.encode_quantize(
                &input,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                &self.input_factors,
                batch_dim,
                self.input_dimension,
                encoder,
            );
            let output = self.inner_linear.encode_with_a(
                MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
                    group_sums: group_sums.as_ref(),
                    group_size: activation_group_size as u32,
                },
                batch_dim,
                encoder,
            )?;

            encoder.pop_debug_group();
            return Ok(output);
        }

        let mut input = input;
        self.bf16_input_rht.encode_fp_in_place(
            &mut input,
            &self.input_factors,
            batch_dim,
            self.input_dimension,
            encoder,
        );
        let output = self.inner_linear.encode(input, batch_dim, encoder)?;

        encoder.pop_debug_group();
        Ok(output)
    }
}
