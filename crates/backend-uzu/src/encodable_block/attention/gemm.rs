pub use crate::backends::common::kernel::attention_gemm::AttentionGemmArguments;
use crate::{
    backends::common::{
        AsBufferRangeRef, Backend, Buffer, Encoder, Kernels, kernel::attention_gemm::AttentionGemmBackendBlock,
    },
    data_type::DataType,
};

pub struct AttentionGemmBlock<B: Backend>(<B::Kernels as Kernels>::AttentionGemmBlock);

impl<B: Backend> AttentionGemmBlock<B> {
    pub fn new(data_type: DataType) -> Self {
        Self(<B::Kernels as Kernels>::AttentionGemmBlock::new(data_type))
    }

    pub fn encode<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = B>>>(
        &self,
        encoder: &mut Encoder<B>,
        args: AttentionGemmArguments<B, KVBuf>,
    ) -> Result<(), B::Error> {
        self.0.encode(encoder, args)
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/attention_gemm_test.rs"]
mod tests;
