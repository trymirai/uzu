use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder,
        gpu_types::{
            ActivationPrepareOps, ActivationQuantScheme, HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder,
        },
        kernel::{
            ActivationPrepareConfig, ActivationsPrepareKernel, HadamardTransformKernel, Kernels, compute_b_col_sums,
            matmul::MatmulA,
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

enum Int8Scheme<B: Backend> {
    Symmetric,
    Asymmetric { b_col_sums: Allocation<B> },
}

struct Int8Preparation<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationsPrepareKernel,
    group_size: u32,
    scheme: Int8Scheme<B>,
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
        activation_prepare: ActivationPrepareConfig,
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

        let int8_preparation = if let Some(group_size) =
            activation_prepare_group_size(activation_prepare, input_dimension, &quantization_spec)
        {
            let mut ops = ActivationPrepareOps::INPUT_RHT | ActivationPrepareOps::QUANTIZE;
            let scheme = match activation_prepare.scheme {
                ActivationQuantScheme::Symmetric => Int8Scheme::Symmetric,
                ActivationQuantScheme::Asymmetric => {
                    ops |= ActivationPrepareOps::ASYMMETRIC;
                    let weights = quantized_weights_tree
                        .leaf("weights")?
                        .validate(&[output_dimension, input_dimension], DataType::U8)?
                        .read_allocation()?;
                    let host = weights.copyout::<u8>();
                    let sums = compute_b_col_sums(&host, output_dimension, input_dimension, group_size as usize);
                    let mut b_col_sums = context
                        .create_allocation(sums.len() * size_of::<i32>(), AllocationType::Global)
                        .map_err(RHTLinearWrapperError::BackendError)?;
                    b_col_sums.copyin(&sums);
                    Int8Scheme::Asymmetric { b_col_sums }
                },
            };
            let kernel = <B::Kernels as Kernels>::ActivationsPrepareKernel::new(
                context,
                input_data_type,
                ops,
                activation_prepare.statistic,
            )
            .map_err(RHTLinearWrapperError::BackendError)?;
            Some(Int8Preparation {
                kernel,
                group_size,
                scheme,
            })
        } else {
            None
        };

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
            && self.inner_linear.supports_int8_a(encoder.context(), preparation.group_size)
        {
            let groups_per_row = self.input_dimension.div_ceil(preparation.group_size as usize);
            let mut values =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dimension], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::F32))?;
            let mut row_sums = encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::I32))?;

            return match &preparation.scheme {
                Int8Scheme::Symmetric => {
                    preparation.kernel.encode(
                        &input,
                        Some(&mut values),
                        Some(&mut scales),
                        Some(&mut row_sums),
                        None::<&mut Allocation<B>>,
                        Some(&self.input_factors),
                        batch_dim as u32,
                        self.input_dimension as u32,
                        preparation.group_size,
                        encoder,
                    );
                    self.inner_linear.encode_with_a(
                        MatmulA::Int8Symmetric {
                            values: &values,
                            scales: &scales,
                            row_sums: &row_sums,
                            group_size: preparation.group_size,
                        },
                        batch_dim,
                        encoder,
                    )
                },
                Int8Scheme::Asymmetric { b_col_sums } => {
                    let mut zero_points =
                        encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::I8))?;
                    preparation.kernel.encode(
                        &input,
                        Some(&mut values),
                        Some(&mut scales),
                        Some(&mut row_sums),
                        Some(&mut zero_points),
                        Some(&self.input_factors),
                        batch_dim as u32,
                        self.input_dimension as u32,
                        preparation.group_size,
                        encoder,
                    );
                    self.inner_linear.encode_with_a(
                        MatmulA::Int8Asymmetric {
                            values: &values,
                            scales: &scales,
                            zero_points: &zero_points,
                            row_sums: &row_sums,
                            b_col_sums,
                            group_size: preparation.group_size,
                        },
                        batch_dim,
                        encoder,
                    )
                },
            };
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
