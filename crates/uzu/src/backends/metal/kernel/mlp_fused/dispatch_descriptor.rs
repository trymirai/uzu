use super::{common::MlpFusedArguments, gemm, gemv, split_k};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpFusedKernelVariant {
    Gemv,
    SplitK,
    Gemm,
}

impl MlpFusedKernelVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            MlpFusedKernelVariant::Gemv => "GEMV",
            MlpFusedKernelVariant::SplitK => "Split-K",
            MlpFusedKernelVariant::Gemm => "GEMM",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum MlpFusedDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    SplitK(split_k::DispatchDescriptor),
    Gemm(gemm::DispatchDescriptor),
}

impl MlpFusedDispatchDescriptor {
    pub fn variant(&self) -> MlpFusedKernelVariant {
        match self {
            MlpFusedDispatchDescriptor::Gemv(_) => MlpFusedKernelVariant::Gemv,
            MlpFusedDispatchDescriptor::SplitK(_) => MlpFusedKernelVariant::SplitK,
            MlpFusedDispatchDescriptor::Gemm(_) => MlpFusedKernelVariant::Gemm,
        }
    }
}

pub(crate) fn choose_dispatch_descriptor(
    context: &MTLContext,
    data_type: DataType,
    weights_transposed: bool,
    arguments: &MlpFusedArguments,
) -> Result<MlpFusedDispatchDescriptor, MTLError> {
    if weights_transposed {
        if let Some(descriptor) = gemv::DispatchDescriptor::try_new(context, data_type, arguments)? {
            return Ok(MlpFusedDispatchDescriptor::Gemv(descriptor));
        }

        if let Some(descriptor) = split_k::DispatchDescriptor::try_new(context, data_type, arguments)? {
            return Ok(MlpFusedDispatchDescriptor::SplitK(descriptor));
        }
    }

    Ok(MlpFusedDispatchDescriptor::Gemm(gemm::DispatchDescriptor::new(
        context,
        data_type,
        weights_transposed,
        arguments,
    )?))
}

pub fn determine_kernel_variant(
    context: &MTLContext,
    data_type: DataType,
    weights_transposed: bool,
    arguments: &MlpFusedArguments,
) -> Result<MlpFusedKernelVariant, MTLError> {
    choose_dispatch_descriptor(context, data_type, weights_transposed, arguments).map(|d| d.variant())
}
