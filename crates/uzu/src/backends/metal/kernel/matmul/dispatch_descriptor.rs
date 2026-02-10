use super::{common::MatmulArguments, gemm, gemv, split_k};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulKernelVariant {
    Gemv,
    SplitK,
    Gemm,
}

impl MatmulKernelVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            MatmulKernelVariant::Gemv => "GEMV",
            MatmulKernelVariant::SplitK => "Split-K",
            MatmulKernelVariant::Gemm => "GEMM",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum MatmulDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    SplitK(split_k::DispatchDescriptor),
    Gemm(gemm::DispatchDescriptor),
}

impl MatmulDispatchDescriptor {
    pub fn variant(&self) -> MatmulKernelVariant {
        match self {
            MatmulDispatchDescriptor::Gemv(_) => MatmulKernelVariant::Gemv,
            MatmulDispatchDescriptor::SplitK(_) => MatmulKernelVariant::SplitK,
            MatmulDispatchDescriptor::Gemm(_) => MatmulKernelVariant::Gemm,
        }
    }
}

pub(crate) fn choose_dispatch_descriptor(
    context: &MTLContext,
    data_type: DataType,
    arguments: &MatmulArguments,
) -> Result<MatmulDispatchDescriptor, MTLError> {
    if let Some(descriptor) = gemv::DispatchDescriptor::try_new(context, data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    if let Some(descriptor) = split_k::DispatchDescriptor::try_new(context, data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::SplitK(descriptor));
    }

    Ok(MatmulDispatchDescriptor::Gemm(gemm::DispatchDescriptor::new(context, data_type, arguments)?))
}

pub fn determine_kernel_variant(
    context: &MTLContext,
    data_type: DataType,
    arguments: &MatmulArguments,
) -> Result<MatmulKernelVariant, MTLError> {
    choose_dispatch_descriptor(context, data_type, arguments).map(|d| d.variant())
}
