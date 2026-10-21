use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::matmul::ActivationFormat,
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

fn has_input_rht(spec: &AnyWeightMatrixSpec) -> Option<bool> {
    match spec {
        AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            incoherence_block_size: Some(block_size),
            incoherence_processing_mode:
                mode @ (IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput),
            ..
        }) if *block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => Some(*mode == IncoherenceProcessingMode::InputOutput),
        _ => None,
    }
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub(super) fn new(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let Some(has_output_rht) = has_input_rht(&spec) else {
            return Err(RHTLinearWrapperError::UnsupportedConfiguration(format!("{spec:?}")));
        };

        let (rht_signs, mut inner_linear) = Self::load_inner(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            has_output_rht,
            parameter_tree,
            context,
        )?;
        let activation_quantization = inner_linear.prepare_a8(context);
        if activation_quantization.is_some() {
            inner_linear.optimize_weights_layout(context).map_err(RHTLinearWrapperError::BackendError)?;
        }
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

    pub(super) fn try_new_with_input_preparation(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        allow_prequantized_activation: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Option<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>)>, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let Some(has_output_rht) = has_input_rht(&spec) else {
            return Ok(None);
        };

        let (rht_signs, mut inner_linear) = Self::load_inner(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            has_biases,
            has_output_rht,
            parameter_tree,
            context,
        )?;
        let input_preparation = LinearInputPreparation {
            rht_signs,
            activation_quantization: inner_linear.prepare_a8(context),
        };
        if input_preparation.activation_quantization.is_some() {
            inner_linear.optimize_weights_layout(context).map_err(RHTLinearWrapperError::BackendError)?;
        }
        if input_preparation.activation_quantization.is_some() && !allow_prequantized_activation {
            let wrapper = Self::build_self_contained(input_data_type, input_preparation, inner_linear, context)?;
            Ok(Some((Box::new(wrapper), None)))
        } else {
            Ok(Some((Box::new(inner_linear), Some(input_preparation))))
        }
    }

    fn load_inner(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        has_output_rht: bool,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(B::GlobalBuffer, LinearMatmul<B>), RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let quantized_weights_tree = weights_tree.subtree("quantized");
        let quantization_spec = quantized_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let rht_signs = weights_tree.read_buffer("incoherence_signs.input_signs", &[input_dimension], DataType::I32)?;
        let output_factors = has_output_rht
            .then(|| weights_tree.read_buffer("incoherence_signs.output_signs", &[output_dimension], DataType::I32))
            .transpose()?;
        let inner_linear = LinearMatmul::load(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            quantization_spec,
            output_factors,
            &quantized_weights_tree,
            has_biases.then_some(parameter_tree),
            context,
        )?;
        Ok((rht_signs, inner_linear))
    }

    fn build_self_contained(
        input_data_type: DataType,
        input_preparation: LinearInputPreparation<B>,
        inner_linear: LinearMatmul<B>,
        context: &B::Context,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let input_rht = InputRht::new(input_data_type, input_preparation, false, context)
            .map_err(RHTLinearWrapperError::BackendError)?;

        Ok(Self {
            input_rht,
            inner_linear,
        })
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode_ref(
        &self,
        input: &B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("linear (rht)");

        let format = self.inner_linear.select_activation_format(batch_dim, command_buffer.context());
        let input =
            self.input_rht.prepare(input, batch_dim, format, command_buffer).map_err(LinearEncodeError::Backend)?;
        let output = self.inner_linear.encode_with_a(
            input.as_matmul_a(),
            batch_dim,
            None::<Gather<&B::ScratchBuffer>>,
            command_buffer,
        )?;

        Ok(output)
    }

    fn encode_prerotated_bf16(
        &self,
        rotated_input: &B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("linear (prerotated)");
        self.inner_linear.encode_ref(rotated_input, batch_dim, command_buffer)
    }

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

    // Delegate to inner_linear so the underlying matmul selects its optimal activation format
    // (e.g. Int8 vs FullPrecision) for this batch dimension.
    fn select_activation_format(
        &self,
        batch_dim: u32,
        context: &B::Context,
    ) -> ActivationFormat {
        self.inner_linear.select_activation_format(batch_dim, context)
    }
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    fn make_hybrid_spec(
        mode: &str,
        block_size: Option<u32>,
    ) -> AnyWeightMatrixSpec {
        let block_size_json = match block_size {
            Some(b) => b.to_string(),
            None => "null".to_string(),
        };
        let json = format!(
            r#"{{
                "type": "HybridSpec",
                "quantization_spec": {{
                    "type": "IntSpec",
                    "bits": 4,
                    "group_size": 32,
                    "is_symmetric": false,
                    "layout": "output_input"
                }},
                "adapter_spec": null,
                "incoherence_block_size": {block_size_json},
                "incoherence_processing_mode": "{mode}"
            }}"#
        );
        serde_json::from_str(&json).expect("valid hybrid spec json")
    }

    #[uzu_test]
    fn test_has_input_rht_selection() {
        // Input-only rotation (e.g. Qwen3.5 shared expert gate): has input RHT, no output RHT
        let spec_input = make_hybrid_spec("input", Some(HADAMARD_TRANSFORM_BLOCK_SIZE));
        assert_eq!(has_input_rht(&spec_input), Some(false));

        // InputOutput rotation: has both input and output RHT
        let spec_input_output = make_hybrid_spec("input_output", Some(HADAMARD_TRANSFORM_BLOCK_SIZE));
        assert_eq!(has_input_rht(&spec_input_output), Some(true));

        // Output-only rotation: unsupported by input RHT wrapper
        let spec_output = make_hybrid_spec("output", Some(HADAMARD_TRANSFORM_BLOCK_SIZE));
        assert_eq!(has_input_rht(&spec_output), None);

        // Mismatched block size: unsupported
        let spec_wrong_block = make_hybrid_spec("input", Some(16));
        assert_eq!(has_input_rht(&spec_wrong_block), None);
    }
}
