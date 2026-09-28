use std::sync::Arc;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{FullPrecisionEmbeddingLookupKernel, QuantizedEmbeddingLookupKernel},
    },
    encodable_block::{EncodableBlock, embedding::EmbeddingResource},
};

pub struct EmbeddingLookupInput<T: BufferRef> {
    pub token_ids: T,
    pub batch_dim: u32,
}

pub enum EmbeddingLookupKernel<B: Backend> {
    FullPrecision(<B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel),
    Quantized(<B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel),
}

pub struct EmbeddingLookup<B: Backend> {
    resource: Arc<EmbeddingResource<B>>,
    kernel: EmbeddingLookupKernel<B>,
    scale: f32,
}

impl<B: Backend> EmbeddingLookup<B> {
    pub fn new(
        context: &B::Context,
        resource: Arc<EmbeddingResource<B>>,
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
            resource,
            kernel,
            scale,
        })
    }

    pub fn model_dim(&self) -> u32 {
        self.resource.model_dim
    }
}

impl<B: Backend, T: BufferRef<Backend = B>> EncodableBlock<B, EmbeddingLookupInput<T>> for EmbeddingLookup<B> {
    type Kernel = EmbeddingLookupKernel<B>;
    type Output = B::ScratchBuffer;
    type Error = B::Error;

    fn encode(
        &self,
        input: EmbeddingLookupInput<T>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error> {
        let matrix = &self.resource.matrix;

        command_buffer.push_debug_group("embedding lookup");

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[input.batch_dim, self.resource.model_dim], self.resource.data_type)?;
        match &self.kernel {
            EmbeddingLookupKernel::FullPrecision(kernel) => kernel.encode(
                input.token_ids,
                matrix.values(),
                &mut output,
                input.batch_dim,
                self.resource.vocab_size,
                self.resource.model_dim,
                self.scale,
                command_buffer,
            ),
            EmbeddingLookupKernel::Quantized(kernel) => kernel.encode(
                input.token_ids,
                matrix.values(),
                matrix.scales().expect("quantized lookup requires scales"),
                matrix.zero_points(),
                matrix.biases(),
                &mut output,
                self.resource.output_hadamard_factors.as_ref(),
                input.batch_dim,
                self.resource.vocab_size,
                self.resource.model_dim,
                self.scale,
                command_buffer,
            ),
        }

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
