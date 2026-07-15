use std::convert::Infallible;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels},
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

pub trait DeltaNetTreeVerify<B: Backend<Kernels: Kernels<DeltaNetTreeVerify = Self>>>: Sized {
    fn is_supported(
        arguments: &TreeVerifyNewArguments,
        context: &B::Context,
    ) -> Result<bool, B::Error>;

    fn new(
        context: &B::Context,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, B::Error>;

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

impl<B: Backend<Kernels: Kernels<DeltaNetTreeVerify = Infallible>>> DeltaNetTreeVerify<B> for Infallible {
    fn is_supported(
        _arguments: &TreeVerifyNewArguments,
        _context: &B::Context,
    ) -> Result<bool, B::Error> {
        Ok(false)
    }

    fn new(
        _context: &B::Context,
        _arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, B::Error> {
        unreachable!("unsupported DeltaNet tree-verify core should not be constructed")
    }

    fn encode(
        &self,
        _arguments: TreeVerifyEncodeArguments<'_, B>,
        _encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        unreachable!("unsupported DeltaNet tree-verify core cannot encode")
    }
}
