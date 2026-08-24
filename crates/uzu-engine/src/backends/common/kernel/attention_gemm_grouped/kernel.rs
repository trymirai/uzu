use std::convert::Infallible;

use crate::{
    backends::common::{Allocation, Backend, BufferArg, Encoder, kernel::Kernels},
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub trait AttentionGemmGroupedCore<B: Backend<Kernels: Kernels<AttentionGemmGroupedCore = Self>>>:
    Sized + Send + Sync
{
    fn is_supported(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<bool, B::Error>;

    fn new(
        context: &B::Context,
        arguments: &AttentionCoreNewArguments,
    ) -> Result<Self, B::Error>;

    fn should_encode(
        &self,
        suffix_length: u32,
        kv_length: u32,
    ) -> bool;

    fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

impl<B: Backend<Kernels: Kernels<AttentionGemmGroupedCore = Infallible>>> AttentionGemmGroupedCore<B> for Infallible {
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
        unreachable!("unsupported grouped attention core should not be constructed")
    }

    fn should_encode(
        &self,
        _suffix_length: u32,
        _kv_length: u32,
    ) -> bool {
        match *self {}
    }

    fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        _arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        _encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match *self {}
    }
}
