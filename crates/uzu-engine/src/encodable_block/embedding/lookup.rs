use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{FullPrecisionEmbeddingLookupKernel, QuantizedEmbeddingLookupKernel},
    },
    encodable_block::{EncodableBlock, embedding::EmbeddingResource},
};

pub struct EmbeddingLookupInput<'a, B: Backend, T: BufferRef<Backend = B>> {
    pub resource: &'a EmbeddingResource<B>,
    pub token_ids: T,
    pub batch_dim: u32,
}

pub enum EmbeddingLookupKernel<B: Backend> {
    FullPrecision(<B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel),
    Quantized(<B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel),
}

pub struct EmbeddingLookup<B: Backend> {
    kernel: EmbeddingLookupKernel<B>,
    scale: f32,
}

impl<B: Backend> EmbeddingLookup<B> {
    pub fn new(
        context: &B::Context,
        resource: &EmbeddingResource<B>,
        scale: f32,
    ) -> Result<Self, B::Error> {
        let data_type = resource.data_type;
        let kernel = match resource.matrix.quantization() {
            None => EmbeddingLookupKernel::FullPrecision(
                <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)?,
            ),
            Some(info) => {
                EmbeddingLookupKernel::Quantized(<B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    info.group_size,
                    info.mode,
                    info.method,
                    resource.output_hadamard_factors.is_some(),
                )?)
            },
        };

        Ok(Self {
            kernel,
            scale,
        })
    }
}

impl<B: Backend, T: BufferRef<Backend = B>> EncodableBlock<B, EmbeddingLookupInput<'_, B, T>> for EmbeddingLookup<B> {
    type Kernel = EmbeddingLookupKernel<B>;
    type Output = B::ScratchBuffer;
    type Error = B::Error;

    fn encode(
        &self,
        input: EmbeddingLookupInput<'_, B, T>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error> {
        let matrix = &input.resource.matrix;

        command_buffer.push_debug_group("embedding lookup");

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[input.batch_dim, input.resource.model_dim], input.resource.data_type)?;
        match (&self.kernel, matrix.quantization()) {
            (EmbeddingLookupKernel::FullPrecision(kernel), None) => kernel.encode(
                input.token_ids,
                matrix.values(),
                &mut output,
                input.batch_dim,
                input.resource.vocab_size,
                input.resource.model_dim,
                self.scale,
                command_buffer,
            ),
            (EmbeddingLookupKernel::Quantized(kernel), Some(_)) => kernel.encode(
                input.token_ids,
                matrix.values(),
                matrix.scales().expect("quantized lookup requires scales"),
                matrix.zero_points(),
                matrix.biases(),
                &mut output,
                input.resource.output_hadamard_factors.as_ref(),
                input.batch_dim,
                input.resource.vocab_size,
                input.resource.model_dim,
                self.scale,
                command_buffer,
            ),
            _ => panic!("embedding lookup kernel does not match the resource quantization"),
        }

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
