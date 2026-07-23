use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder},
        kernel::{
            HadamardTransformKernel, Kernels,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, hybrid_spec::IncoherenceProcessingMode, low_rank_spec::LowRankSpec},
    data_type::DataType,
    encodable_block::linear::{Linear, LinearMatmul, LinearMatmulError},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QLoRALinearWrapperError<B: Backend> {
    #[error("LinearMatmul error: {0}")]
    LinearMatmulError(#[from] LinearMatmulError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported QLoRA linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct QLoRALinearWrapper<B: Backend> {
    base_linear: LinearMatmul<B>,
    input_hadamard: Option<(<B::Kernels as Kernels>::HadamardTransformKernel, Allocation<B>)>,
    output_hadamard: Option<(<B::Kernels as Kernels>::HadamardTransformKernel, Allocation<B>)>,
    adapter_down_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    adapter_up_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    adapter_down: Allocation<B>,
    adapter_up: Allocation<B>,
    input_dim: usize,
    output_dim: usize,
    lora_rank: usize,
    weights_data_type: DataType,
    input_data_type: DataType,
}

impl<B: Backend> QLoRALinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        quantization_spec: AnyWeightMatrixSpec,
        adapter_spec: LowRankSpec,
        incoherence_block_size: Option<usize>,
        incoherence_processing_mode: IncoherenceProcessingMode,
        input_dim: usize,
        output_dim: usize,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
    ) -> Result<Self, QLoRALinearWrapperError<B>> {
        let use_incoherence_signs = match (incoherence_block_size, incoherence_processing_mode) {
            (None, _) => false,
            (Some(HADAMARD_TRANSFORM_BLOCK_SIZE), IncoherenceProcessingMode::InputOutput) => true,
            (incoherence_block_size, incoherence_processing_mode) => {
                return Err(QLoRALinearWrapperError::UnsupportedConfiguration(format!(
                    "incoherence block_size={incoherence_block_size:?}, processing_mode={incoherence_processing_mode:?}"
                )));
            },
        };

        let quantized_tree = weights_tree.subtree("quantized")?;
        let base_linear = LinearMatmul::quantized(
            context,
            quantization_spec,
            input_dim,
            output_dim,
            weights_data_type,
            input_data_type,
            output_data_type,
            &quantized_tree,
            None,
            None,
        )?;

        let (input_hadamard, output_hadamard) = if use_incoherence_signs {
            let input_factors = weights_tree
                .leaf("incoherence_signs.input_signs")?
                .validate(&[input_dim], DataType::I32)?
                .read_allocation()?;
            let output_factors = weights_tree
                .leaf("incoherence_signs.output_signs")?
                .validate(&[output_dim], DataType::I32)?
                .read_allocation()?;
            (
                Some((
                    <B::Kernels as Kernels>::HadamardTransformKernel::new(
                        context,
                        input_data_type,
                        HadamardTransformOrder::Input,
                    )
                    .map_err(QLoRALinearWrapperError::BackendError)?,
                    input_factors,
                )),
                Some((
                    <B::Kernels as Kernels>::HadamardTransformKernel::new(
                        context,
                        output_data_type,
                        HadamardTransformOrder::Output,
                    )
                    .map_err(QLoRALinearWrapperError::BackendError)?,
                    output_factors,
                )),
            )
        } else {
            (None, None)
        };

        let adapter_down_kernel = Mutex::new(
            <<B::Kernels as Kernels>::MatmulKernel as MatmulKernel>::new(
                context,
                weights_data_type,
                input_data_type,
                weights_data_type,
            )
            .map_err(QLoRALinearWrapperError::BackendError)?,
        );
        let adapter_up_kernel = Mutex::new(
            <<B::Kernels as Kernels>::MatmulKernel as MatmulKernel>::new(
                context,
                weights_data_type,
                weights_data_type,
                output_data_type,
            )
            .map_err(QLoRALinearWrapperError::BackendError)?,
        );

        let adapter_down = weights_tree
            .leaf("adapter.down_projection")?
            .validate(&[adapter_spec.rank, input_dim], weights_data_type)?
            .read_allocation()?;

        let adapter_up = weights_tree
            .leaf("adapter.up_projection")?
            .validate(&[output_dim, adapter_spec.rank], weights_data_type)?
            .read_allocation()?;

        Ok(Self {
            base_linear,
            input_hadamard,
            output_hadamard,
            adapter_down_kernel,
            adapter_up_kernel,
            adapter_down,
            adapter_up,
            input_dim,
            output_dim,
            lora_rank: adapter_spec.rank,
            weights_data_type,
            input_data_type,
        })
    }
}

impl<B: Backend> Linear<B> for QLoRALinearWrapper<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut intermediate =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.lora_rank], self.weights_data_type))?;

        {
            let mut adapter_kernel = self.adapter_down_kernel.lock();
            adapter_kernel.encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &input,
                        offset: 0,
                    },
                    b: MatmulB::FullPrecision {
                        b: &self.adapter_down,
                    },
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut intermediate,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None,
                    m: batch_dim as u32,
                    n: self.lora_rank as u32,
                    k: self.input_dim as u32,
                },
                encoder,
            )?;
        }

        let base_input = if let Some((input_hadamard_kernel, input_factors)) = &self.input_hadamard {
            let mut base_input =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dim], self.input_data_type))?;
            encoder.encode_copy(&input, .., &mut base_input, ..);
            input_hadamard_kernel.encode(
                &mut base_input,
                input_factors,
                self.input_dim as u32,
                batch_dim as u32,
                encoder,
            );
            base_input
        } else {
            input
        };

        let mut output = self.base_linear.encode(base_input, batch_dim, encoder)?;

        {
            let mut adapter_kernel = self.adapter_up_kernel.lock();
            adapter_kernel.encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &intermediate,
                        offset: 0,
                    },
                    b: MatmulB::FullPrecision {
                        b: &self.adapter_up,
                    },
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output,
                    d_transform: MatmulDOps {
                        accumulate: true,
                        ..MatmulDOps::none()
                    },
                    gather_indices: None,
                    m: batch_dim as u32,
                    n: self.output_dim as u32,
                    k: self.lora_rank as u32,
                },
                encoder,
            )?;
        }

        if let Some((output_hadamard_kernel, output_factors)) = &self.output_hadamard {
            output_hadamard_kernel.encode(
                &mut output,
                output_factors,
                self.output_dim as u32,
                batch_dim as u32,
                encoder,
            );
        }

        Ok(output)
    }
}
