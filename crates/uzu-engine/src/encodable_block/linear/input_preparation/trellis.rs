use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::trellis::{COLUMN_GROUP_COUNT, MAX_DIMENSION, MAX_HADAMARD_SIZE, MIN_HADAMARD_SIZE},
        kernel::{TrellisTransformKernel, matmul::MatmulA, mixing_dimension},
    },
    config::weight_matrix::AnyWeightMatrixSpec,
    data_type::DataType,
    encodable_block::linear::{LinearBlockError, LinearInput},
    safetensors::ParameterTree,
};

pub struct TrellisPreparation<B: Backend> {
    kernel: <B::Kernels as Kernels>::TrellisTransformKernel,
    signs: B::GlobalGpuBuffer,
    orthogonal_factor: B::GlobalGpuBuffer,
    input_dimension: u32,
}

impl<B: Backend> TrellisPreparation<B> {
    pub async fn load(
        input_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        spec: &AnyWeightMatrixSpec,
        weights_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, LinearBlockError<B>> {
        if has_biases {
            return Err(LinearBlockError::UnsupportedConfiguration("Kronecker linear biases".into()));
        }
        if weights_data_type != DataType::BF16
            || input_data_type != DataType::BF16
            || output_data_type != DataType::BF16
        {
            return Err(LinearBlockError::UnsupportedConfiguration("Kronecker linears require BF16".into()));
        }
        let hadamard_size = input_dimension.isolate_lowest_one();
        if !(MIN_HADAMARD_SIZE..=MAX_HADAMARD_SIZE).contains(&hadamard_size) || input_dimension > MAX_DIMENSION {
            return Err(LinearBlockError::UnsupportedConfiguration(format!(
                "Kronecker rotation over {input_dimension} columns"
            )));
        }
        let kernel = <B::Kernels as Kernels>::TrellisTransformKernel::new(context, input_dimension)
            .map_err(LinearBlockError::Backend)?;
        let signs = weights_tree.read_buffer("incoherence_signs.signs", &[input_dimension], DataType::I32).await?;
        let mixing_dimension = mixing_dimension(input_dimension);
        let orthogonal_factor = weights_tree
            .read_buffer("incoherence_signs.orthogonal_factor", &[mixing_dimension, mixing_dimension], DataType::F32)
            .await?;
        let trellis_weights = match spec {
            AnyWeightMatrixSpec::OutputStackSpec(stack) => {
                stack.parts.iter().all(|part| matches!(part.spec, AnyWeightMatrixSpec::QtipGaussianSpec(_)))
            },
            spec => matches!(spec, AnyWeightMatrixSpec::QtipGaussianSpec(_)),
        };
        if !trellis_weights {
            return Err(LinearBlockError::UnsupportedConfiguration("expected Trellis weights".into()));
        }
        Ok(Self {
            kernel,
            signs,
            orthogonal_factor,
            input_dimension,
        })
    }

    pub fn prepare(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        let mut values = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.input_dimension], DataType::I8)?;
        let mut column_group_sums =
            command_buffer.allocate_scratch_for_shape(&[batch_dim, COLUMN_GROUP_COUNT], DataType::F32)?;
        let mut scales = command_buffer.allocate_scratch_for_shape(&[batch_dim], DataType::F32)?;
        self.kernel.encode(
            input,
            &self.signs,
            &self.orthogonal_factor,
            &mut values,
            &mut column_group_sums,
            &mut scales,
            batch_dim,
            command_buffer,
        );
        Ok(MatmulA::Trellis {
            values,
            column_group_sums,
            scales,
        })
    }
}
