use crate::{
    backends::common::{Backend, CommandBuffer, Kernels, kernel::Unsupported},
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

pub trait DeltaNetTreeVerify: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<DeltaNetTreeVerify = Self>>;

    fn is_supported(context: &<Self::Backend as Backend>::Context) -> bool;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Self::Backend>,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, <Self::Backend as Backend>::Error>;
}

impl<B: Backend<Kernels: Kernels<DeltaNetTreeVerify = Unsupported<B>>>> DeltaNetTreeVerify for Unsupported<B> {
    type Backend = B;

    fn is_supported(_context: &B::Context) -> bool {
        false
    }

    fn new(
        _context: &B::Context,
        _arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, B::Error> {
        unreachable!("unsupported DeltaNet tree verifier should not be constructed")
    }

    fn encode(
        &self,
        _arguments: TreeVerifyEncodeArguments<'_, B>,
        _command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        unreachable!("unsupported DeltaNet tree verifier cannot encode")
    }
}
