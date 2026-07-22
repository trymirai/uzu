use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Context, DeviceCapabilities, Encoder,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder},
        kernel::{
            ActivationsPrepareKernel, HadamardTransformKernel, Kernels,
            matmul::{MatmulA, symmetric_int8_activations::ACTIVATION_QUANTIZATION_GROUP_SIZE},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
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

struct SymmetricInt8Preparation<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationsPrepareKernel,
}

pub struct RHTLinearWrapper<B: Backend> {
    input_hadamard_kernel: <B::Kernels as Kernels>::HadamardTransformKernel,
    symmetric_int8_preparation: Option<SymmetricInt8Preparation<B>>,
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

        let input_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            input_data_type,
            HadamardTransformOrder::Input,
        )
        .map_err(RHTLinearWrapperError::BackendError)?;

        let symmetric_int8_preparation = if context
            .device_capabilities()
            .contains(DeviceCapabilities::HARDWARE_INT8_MATMUL)
        {
            Some(
                <B::Kernels as Kernels>::ActivationsPrepareKernel::new(context, input_data_type)
                    .map(|kernel| SymmetricInt8Preparation {
                        kernel,
                    })
                    .map_err(RHTLinearWrapperError::BackendError)?,
            )
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
            symmetric_int8_preparation,
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
        if let Some(preparation) = &self.symmetric_int8_preparation {
            let groups_per_row = self.input_dimension.div_ceil(ACTIVATION_QUANTIZATION_GROUP_SIZE as usize);
            let mut values =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dimension], DataType::I8))?;
            let mut scales = encoder.allocate_scratch(size_for_shape(&[batch_dim, groups_per_row], DataType::F32))?;

            preparation.kernel.encode(
                &input,
                &mut values,
                &mut scales,
                &self.input_factors,
                batch_dim as u32,
                self.input_dimension as u32,
                ACTIVATION_QUANTIZATION_GROUP_SIZE,
                encoder,
            );
            return self.inner_linear.encode_with_a(
                MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
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
