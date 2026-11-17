use std::sync::Arc;

use super::LinearInput;
use crate::backends::common::{Backend, BufferRef, CommandBuffer, kernel::ActivationFormat};
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
        matches!(self, Self::Scalar(p) if p.requires_full_precision())
    }

    pub fn supports_owned_input(&self) -> bool {
        match self {
            Self::Scalar(p) => p.supports_owned_input(),
            Self::Trellis(_) => true,
        }
    }

    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Scalar(p) if p.is_identity())
    }

    pub fn select_format(
        &self,
        fallback: impl FnOnce() -> ActivationFormat,
    ) -> ActivationFormat {
        match self {
            Self::Scalar(_) => fallback(),
            Self::Trellis(_) => ActivationFormat::Trellis,
        }
    }

    pub fn external_format(
        &self,
        fallback: impl FnOnce() -> ActivationFormat,
    ) -> ActivationFormat {
        match self {
            Self::Scalar(_) => fallback(),
            Self::Trellis(_) => ActivationFormat::Bf16,
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
            Self::Scalar(p) => p.prepare(input, batch_dim, format, command_buffer),
            Self::Trellis(p) => p.prepare(input, batch_dim, command_buffer).map(Some),
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
            Self::Scalar(p) => p.prepare_owned(input, offset, batch_dim, format, command_buffer),
            Self::Trellis(p) => p.prepare((&input).subrange(offset..), batch_dim, command_buffer),
        }
    }
}
