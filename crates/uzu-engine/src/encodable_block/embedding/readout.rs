use parking_lot::Mutex;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulKernel},
    },
    config::{embedding::AnyEmbeddingConfig, weight_matrix::AnyWeightMatrixSpec},
    encodable_block::{
        EncodableBlock,
        embedding::{EmbeddingError, EmbeddingResource},
        linear::{Gather, UntiedReadout},
    },
    parameters::ParameterTree,
};

pub struct EmbeddingReadoutInput<'a, B: Backend, H: BufferRef<Backend = B>, G: BufferRef<Backend = B>> {
    pub resource: &'a EmbeddingResource<B>,
    pub hidden: H,
    pub batch_dim: u32,
    pub gather: Option<Gather<G>>,
}

pub enum EmbeddingReadoutKernel<B: Backend> {
    Tied(Mutex<<B::Kernels as Kernels>::MatmulKernel>),
    Untied(UntiedReadout<B>),
}

pub struct EmbeddingReadout<B: Backend> {
    kernel: EmbeddingReadoutKernel<B>,
}

impl<B: Backend> EmbeddingReadout<B> {
    pub fn new(
        context: &B::Context,
        config: &AnyEmbeddingConfig,
        parameter_tree: &ParameterTree<B>,
        resource: &EmbeddingResource<B>,
    ) -> Result<(Self, Option<B::GlobalBuffer>), EmbeddingError<B>> {
        let vocab_size = resource.vocab_size;
        let model_dim = resource.model_dim;
        let data_type = resource.data_type;
        let (kernel, input_hadamard_factors) = match config {
            AnyEmbeddingConfig::TiedEmbeddingConfig(_) => {
                let input_hadamard_factors = resource
                    .output_hadamard_factors
                    .as_ref()
                    .map(|_| {
                        EmbeddingResource::load_output_hadamard_factors(&parameter_tree.subtree("embedding"), model_dim)
                    })
                    .transpose()?;
                let kernel = <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                (EmbeddingReadoutKernel::Tied(Mutex::new(kernel)), input_hadamard_factors)
            },
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => {
                let output_tree = parameter_tree.subtree("output_embedding");
                let output_spec = output_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
                let readout =
                    UntiedReadout::load(context, &output_tree, output_spec, vocab_size, model_dim, data_type)?;
                (EmbeddingReadoutKernel::Untied(readout), None)
            },
        };

        Ok((
            Self {
                kernel,
            },
            input_hadamard_factors,
        ))
    }
}

impl<B: Backend, H: BufferRef<Backend = B>, G: BufferRef<Backend = B>>
    EncodableBlock<B, EmbeddingReadoutInput<'_, B, H, G>> for EmbeddingReadout<B>
{
    type Kernel = EmbeddingReadoutKernel<B>;
    type Output = B::ScratchBuffer;
    type Error = B::Error;

    fn encode(
        &self,
        input: EmbeddingReadoutInput<'_, B, H, G>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error> {
        let output_dim = input.gather.as_ref().map_or(input.resource.vocab_size, |gather| gather.output_dim);
        assert!(input.batch_dim > 0 && output_dim > 0, "Embedding readout requires non-empty dimensions");

        command_buffer.push_debug_group("embedding readout");

        let output = match &self.kernel {
            EmbeddingReadoutKernel::Untied(readout) => {
                readout.encode(input.hidden, input.batch_dim, input.gather, command_buffer)?
            },
            EmbeddingReadoutKernel::Tied(kernel) => {
                let mut output = command_buffer
                    .allocate_scratch_for_shape(&[input.batch_dim, output_dim], input.resource.data_type)?;
                let arguments = MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: input.hidden,
                        offset: 0,
                    },
                    b: input.resource.matrix.matmul_b(),
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output,
                    d_transform: MatmulDOps::none(),
                    gather_indices: input.gather.map(|gather| gather.indices),
                    m: input.batch_dim,
                    n: output_dim,
                    k: input.resource.model_dim,
                };
                kernel.lock().encode(arguments, command_buffer)?;
                output
            },
        };

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
