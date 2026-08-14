use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Context, DeviceCapabilities, Encoder,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod},
        kernel::{
            ActivationTransform,
            activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
            matmul::{MatmulA, MatmulPath},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearMatmul, LinearMatmulError},
        weight_matrix::{ParsedWeightSpec, WeightMatrixError, parse_spec},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

// Temporary, for testing only: flip to false to run linear layers on the bf16
// activation path without touching model configs.
const INT8_ACTIVATIONS_ENABLED: bool = true;

pub(super) fn int8_activations_eligible<B: Backend>(
    context: &B::Context,
    spec: &ParsedWeightSpec,
    input_dimension: u32,
    input_data_type: DataType,
    output_data_type: DataType,
    activation_group_size: u32,
) -> bool {
    if !INT8_ACTIVATIONS_ENABLED {
        return false;
    }
    if !context.device_capabilities().contains(DeviceCapabilities::NATIVE_INT8_MATMUL) {
        return false;
    }
    if input_data_type != DataType::BF16 || output_data_type != DataType::BF16 {
        return false;
    }
    let (Some(info), Layout::OutputInput) = (spec.quantization, &spec.layout) else {
        return false;
    };
    let group_size = info.group_size;
    matches!(group_size, 32 | 64 | 128)
        && input_dimension.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
        && input_dimension.is_multiple_of(group_size)
        && input_dimension.is_multiple_of(activation_group_size)
}

#[derive(Debug, Error)]
pub enum RHTLinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] LinearMatmulError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported RHT linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct RHTLinearWrapper<B: Backend> {
    input_transform: ActivationTransform<B>,
    quantize_transform: Option<ActivationTransform<B>>,
    input_factors: Allocation<B>,
    inner_linear: LinearMatmul<B>,
    input_dimension: u32,
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
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
        let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            adapter_spec: None,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        }) = &spec
        else {
            return Err(RHTLinearWrapperError::UnsupportedConfiguration(format!("{spec:?}")));
        };

        let input_factors = weights_tree
            .leaf("incoherence_signs.input_signs")?
            .validate(&[input_dimension], DataType::I32)?
            .read_allocation()?;
        let output_factors = weights_tree
            .leaf("incoherence_signs.output_signs")?
            .validate(&[output_dimension], DataType::I32)?
            .read_allocation()?;
        let quantized_weights_tree = weights_tree.subtree("quantized");
        let quantization_spec = quantized_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let parsed = parse_spec::<B>(&quantization_spec)?;
        let Some(quantization) = parsed.quantization else {
            return Err(RHTLinearWrapperError::UnsupportedConfiguration("RHT requires a quantized inner spec".into()));
        };
        let activation_group_size = ACTIVATION_SCALE_GROUP_SIZE;

        let input_transform = ActivationTransform::input_rht(context, input_data_type, true)
            .map_err(RHTLinearWrapperError::BackendError)?;

        let mut inner_linear = LinearMatmul::load(
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

        let quantize_transform = if int8_activations_eligible::<B>(
            context,
            &parsed,
            input_dimension,
            input_data_type,
            output_data_type,
            activation_group_size,
        ) {
            let emit_group_sums = !matches!(quantization.method, QuantizationMethod::ScaleSymmetric);
            let transform = ActivationTransform::quantize(
                context,
                input_data_type,
                activation_group_size,
                emit_group_sums.then_some(quantization.group_size.min(activation_group_size)),
            )
            .map_err(RHTLinearWrapperError::BackendError)?;
            inner_linear.make_codes_signed();
            Some(transform)
        } else {
            None
        };

        Ok(Self {
            input_transform,
            quantize_transform,
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
        encoder.push_debug_group("linear (rht)");

        if let Some(quantize_transform) = &self.quantize_transform
            && self.inner_linear.select_path(batch_dim, encoder.context) == MatmulPath::Gemm
        {
            let scale_groups_per_row = self.input_dimension.div_ceil(quantize_transform.activation_group_size);
            let sum_groups_per_row =
                quantize_transform.sum_group_size.map(|group_size| self.input_dimension.div_ceil(group_size));
            let mut values = encoder.allocate_scratch_for_shape(&[batch_dim, self.input_dimension], DataType::I8)?;
            let mut scales = encoder.allocate_scratch_for_shape(&[batch_dim, scale_groups_per_row], DataType::F32)?;
            let mut group_sums = sum_groups_per_row
                .map(|groups| encoder.allocate_scratch_for_shape(&[batch_dim, groups], DataType::I32))
                .transpose()?;

            quantize_transform.encode_quantize(
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
                    group_size: quantize_transform.activation_group_size,
                },
                batch_dim,
                encoder,
            )?;

            encoder.pop_debug_group();

            return Ok(output);
        }

        let mut input = input;
        self.input_transform.encode_fp_in_place(
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
