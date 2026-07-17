use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{ActivationPrepareOps, HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder},
        kernel::{
            ActivationPrepareConfig, ActivationsPrepareKernel, HadamardTransformKernel, Kernels, matmul::MatmulA,
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        int_spec::IntSpec,
    },
    data_type::DataType,
    encodable_block::linear::{Linear, LinearMatmul, LinearMatmulError},
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

struct Int8Preparation<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationsPrepareKernel,
    group_size: u32,
}

pub struct RHTLinearWrapper<B: Backend> {
    input_hadamard_kernel: <B::Kernels as Kernels>::HadamardTransformKernel,
    int8_preparation: Option<Int8Preparation<B>>,
    input_factors: Allocation<B>,
    inner_linear: LinearMatmul<B>,
    input_dimension: usize,
}

pub(super) fn activation_prepare_group_size(
    config: ActivationPrepareConfig,
    input_dimension: usize,
    quantization_spec: &AnyWeightMatrixSpec,
) -> Option<u32> {
    let AnyWeightMatrixSpec::IntSpec(IntSpec {
        bits: 8,
        group_size,
        is_symmetric: true,
        layout: Layout::OutputInput,
        ..
    }) = quantization_spec
    else {
        return None;
    };

    if input_dimension.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) && config.supports_group_size(*group_size) {
        u32::try_from(*group_size).ok()
    } else {
        None
    }
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

        let input_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            input_data_type,
            HadamardTransformOrder::Input,
        )
        .map_err(RHTLinearWrapperError::BackendError)?;

        let config = ActivationPrepareConfig::from_env();
        let int8_preparation = activation_prepare_group_size(config, input_dimension, &quantization_spec)
            .map(|group_size| {
                let ops = ActivationPrepareOps::INPUT_RHT | ActivationPrepareOps::QUANTIZE;
                <B::Kernels as Kernels>::ActivationsPrepareKernel::new(context, input_data_type, ops, config.stat).map(
                    |kernel| Int8Preparation {
                        kernel,
                        group_size,
                    },
                )
            })
            .transpose()
            .map_err(RHTLinearWrapperError::BackendError)?;

        let inner_linear = LinearMatmul::quantized(
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

        Ok(Self {
            input_hadamard_kernel,
            int8_preparation,
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
        if let Some(preparation) = &self.int8_preparation
            && self.inner_linear.supports_int8_symmetric_a(encoder.context(), batch_dim, preparation.group_size)
        {
            let groups_per_row = self.input_dimension.div_ceil(preparation.group_size as usize);
            let mut values =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dimension], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::F32))?;

            preparation.kernel.encode(
                &input,
                Some(&mut values),
                Some(&mut scales),
                Some(&self.input_factors),
                batch_dim as u32,
                self.input_dimension as u32,
                preparation.group_size,
                encoder,
            );

            return self.inner_linear.encode_with_a(
                MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
                    group_size: preparation.group_size,
                },
                batch_dim,
                encoder,
            );
        }

        let mut input = input;
        self.input_hadamard_kernel.encode(
            &mut input,
            &self.input_factors,
            self.input_dimension as u32,
            batch_dim as u32,
            encoder,
        );
        self.inner_linear.encode(input, batch_dim, encoder)
    }
}
