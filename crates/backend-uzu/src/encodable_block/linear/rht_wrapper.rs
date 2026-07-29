use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Context, DeviceCapabilities, Encoder,
        gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationTransform,
            matmul::{MatmulA, MatmulPath},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        int_spec::IntSpec,
        mlx_spec::MLXSpec,
    },
    data_type::DataType,
    encodable_block::linear::{Linear, LinearMatmul, LinearMatmulError},
    parameters::{ParameterLoaderError, ParameterTree},
};

// Temporary, for testing only: flip to false to run linear layers on the bf16
// activation path without touching model configs.
const INT8_ACTIVATIONS_ENABLED: bool = true;

pub(super) fn int8_activations_eligible<B: Backend>(
    context: &B::Context,
    quantization_spec: &AnyWeightMatrixSpec,
    input_dimension: usize,
    input_data_type: DataType,
    output_data_type: DataType,
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
    let (bits, group_size) = match quantization_spec {
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            group_size,
            layout: Layout::OutputInput,
            ..
        })
        | AnyWeightMatrixSpec::MLXSpec(MLXSpec {
            bits,
            group_size,
            layout: Layout::OutputInput,
            ..
        }) => (*bits, *group_size),
        _ => return false,
    };
    matches!(bits, 4 | 8)
        && matches!(group_size, 32 | 64 | 128)
        && input_dimension.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
        && input_dimension.is_multiple_of(group_size)
}

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

fn weights_need_group_sums(quantization_spec: &AnyWeightMatrixSpec) -> bool {
    !matches!(
        quantization_spec,
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            is_symmetric: true,
            ..
        })
    )
}

pub struct RHTLinearWrapper<B: Backend> {
    input_transform: ActivationTransform<B>,
    quantize_transform: Option<ActivationTransform<B>>,
    input_factors: Allocation<B>,
    inner_linear: LinearMatmul<B>,
    input_dimension: usize,
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        input_dimension: usize,
        output_dimension: usize,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights")?;
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
        let quantized_weights_tree = weights_tree.subtree("quantized")?;
        let quantization_spec = quantized_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

        let input_transform =
            ActivationTransform::input_rht(context, input_data_type).map_err(RHTLinearWrapperError::BackendError)?;

        let quantize_transform = if int8_activations_eligible::<B>(
            context,
            &quantization_spec,
            input_dimension,
            input_data_type,
            output_data_type,
        ) {
            let emit_group_sums = weights_need_group_sums(&quantization_spec);
            Some(
                ActivationTransform::quantize(context, input_data_type, emit_group_sums)
                    .map_err(RHTLinearWrapperError::BackendError)?,
            )
        } else {
            None
        };

        let mut inner_linear = LinearMatmul::quantized(
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
        if quantize_transform.is_some() {
            inner_linear.to_signed_weight_codes();
        }

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
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        if let Some(quantize_transform) = &self.quantize_transform
            && self.inner_linear.select_path(batch_dim, encoder.context()) == MatmulPath::Gemm
        {
            let groups_per_row = self.input_dimension.div_ceil(HADAMARD_TRANSFORM_BLOCK_SIZE);
            let mut values =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dimension], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::F32))?;
            let emit_group_sums = quantize_transform.emit_group_sums();
            let mut group_sums = emit_group_sums
                .then(|| encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::I32)))
                .transpose()?;

            quantize_transform.encode_quantize(
                &input,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                &self.input_factors,
                batch_dim as u32,
                self.input_dimension as u32,
                encoder,
            );
            return self.inner_linear.encode_with_a(
                MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
                    group_sums: group_sums.as_ref(),
                },
                batch_dim,
                encoder,
            );
        }

        let mut transformed = encoder.allocate_scratch(input.size())?;
        self.input_transform.encode_fp(
            &input,
            &mut transformed,
            &self.input_factors,
            batch_dim as u32,
            self.input_dimension as u32,
            encoder,
        );
        self.inner_linear.encode(transformed, batch_dim, encoder)
    }
}
