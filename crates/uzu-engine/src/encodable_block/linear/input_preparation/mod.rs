use std::sync::Arc;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, kernel::ActivationFormat},
    encodable_block::linear::LinearInput,
};
mod scalar;
mod trellis;
pub use scalar::{InputRhtParameters, ScalarPreparation};
pub use trellis::TrellisPreparation;

pub enum InputPreparation<B: Backend> {
    Scalar(ScalarPreparation<B>),
    Trellis(Arc<TrellisPreparation<B>>),
}

impl<B: Backend> InputPreparation<B> {
    pub fn requires_full_precision(&self) -> bool {
        matches!(self, Self::Scalar(preparation) if preparation.requires_full_precision())
    }

    pub fn supports_owned_input(&self) -> bool {
        match self {
            Self::Scalar(preparation) => preparation.supports_owned_input(),
            Self::Trellis(_) => true,
        }
    }

    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Scalar(preparation) if preparation.is_identity())
    }

    pub fn select_format(
        &self,
        trellis_format: ActivationFormat,
        scalar_format: impl FnOnce() -> ActivationFormat,
    ) -> ActivationFormat {
        match self {
            Self::Scalar(_) => scalar_format(),
            Self::Trellis(_) => trellis_format,
        }
    }

    pub fn prepare(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<LinearInput<B>>, B::Error> {
        match self {
            Self::Scalar(preparation) => preparation.prepare(input, batch_dim, format, command_buffer),
            Self::Trellis(preparation) => preparation.prepare(input, batch_dim, command_buffer).map(Some),
        }
    }

    pub fn prepare_owned(
        &self,
        input: B::ScratchBuffer,
        offset: usize,
        batch_dim: u32,
        format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        match self {
            Self::Scalar(preparation) => preparation.prepare_owned(input, offset, batch_dim, format, command_buffer),
            Self::Trellis(preparation) => preparation.prepare((&input).subrange(offset..), batch_dim, command_buffer),
        }
    }
}
