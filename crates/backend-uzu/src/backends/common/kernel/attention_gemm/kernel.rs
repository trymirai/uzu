use std::convert::Infallible;

use crate::{
    backends::common::{Allocation, Backend, BufferArg, Encoder, kernel::Kernels},
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub trait AttentionGemmCore<B: Backend<Kernels: Kernels<AttentionGemmCore = Self>>>: Sized + Send + Sync {
    fn is_supported(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<bool, B::Error>;

    fn new(
        context: &B::Context,
        arguments: &AttentionCoreNewArguments,
    ) -> Result<Self, B::Error>;

    fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

impl<B: Backend<Kernels: Kernels<AttentionGemmCore = Infallible>>> AttentionGemmCore<B> for Infallible {
    fn is_supported(
        _arguments: &AttentionCoreNewArguments,
        _context: &B::Context,
    ) -> Result<bool, B::Error> {
        Ok(false)
    }

    fn new(
        _context: &B::Context,
        _arguments: &AttentionCoreNewArguments,
    ) -> Result<Self, B::Error> {
        unreachable!("unsupported attention core should not be constructed")
    }

    fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        _arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        _encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match *self {}
    }
}
