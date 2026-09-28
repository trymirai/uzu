use std::convert::Infallible;

use crate::{
    backends::common::{Backend, BufferMut, CommandBuffer, Kernels, kernel::LogitTransformKernel},
    config::embedding::AnyEmbeddingConfig,
    data_type::DataType,
    encodable_block::EncodableBlock,
};

pub struct LogitTransformInput<L: BufferMut> {
    pub logits: L,
    pub length: u32,
}

pub struct LogitTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::LogitTransformKernel,
    scale: f32,
    soft_cap: Option<f32>,
}

impl<B: Backend> LogitTransform<B> {
    pub fn new(
        context: &B::Context,
        config: &AnyEmbeddingConfig,
        data_type: DataType,
    ) -> Result<Option<Self>, B::Error> {
        let scale = config.logit_scale().unwrap_or(1.0);
        let soft_cap = *config.logit_soft_cap();
        if scale == 1.0 && soft_cap.is_none() {
            return Ok(None);
        }

        let kernel = <B::Kernels as Kernels>::LogitTransformKernel::new(context, data_type, soft_cap.is_some())?;
        Ok(Some(Self {
            kernel,
            scale,
            soft_cap,
        }))
    }
}

impl<B: Backend, L: BufferMut<Backend = B>> EncodableBlock<B, LogitTransformInput<L>> for LogitTransform<B> {
    type Kernel = <B::Kernels as Kernels>::LogitTransformKernel;
    type Output = ();
    type Error = Infallible;

    fn encode(
        &self,
        input: LogitTransformInput<L>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error> {
        self.kernel.encode(input.logits, input.length, self.scale, self.soft_cap.unwrap_or(0.0), command_buffer);
        Ok(())
    }
}
