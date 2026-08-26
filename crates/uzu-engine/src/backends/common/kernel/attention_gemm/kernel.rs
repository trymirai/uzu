use crate::{
    backends::common::{
        Allocation, Backend, BufferArg, Encoder,
        kernel::{Kernels, Unsupported},
    },
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub trait AttentionGemmCore: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionGemmCore = Self>>;

    fn is_supported(
        arguments: &AttentionCoreNewArguments,
        context: &<Self::Backend as Backend>::Context,
    ) -> Result<bool, <Self::Backend as Backend>::Error>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        arguments: &AttentionCoreNewArguments,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<'a, KT: BufferArg<'a, Self::Backend>, VT: BufferArg<'a, Self::Backend>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, Self::Backend, KT, VT>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<Allocation<Self::Backend>, <Self::Backend as Backend>::Error>;
}

impl<B: Backend<Kernels: Kernels<AttentionGemmCore = Unsupported<B>>>> AttentionGemmCore for Unsupported<B> {
    type Backend = B;

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
        match self.never {}
    }
}
