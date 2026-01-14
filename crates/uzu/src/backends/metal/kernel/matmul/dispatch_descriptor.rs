use super::{common::MatmulArguments, gemm, gemv, split_k};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

#[derive(Debug, Clone)]
pub(crate) enum MatmulDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    SplitK(split_k::DispatchDescriptor),
    Gemm(gemm::DispatchDescriptor),
}

pub(crate) fn choose_dispatch_descriptor(
    context: &MTLContext,
    data_type: DataType,
    arguments: &MatmulArguments,
) -> Result<MatmulDispatchDescriptor, MTLError> {
    if let Some(descriptor) =
        gemv::DispatchDescriptor::try_new(context, data_type, arguments)?
    {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    if let Some(descriptor) =
        split_k::DispatchDescriptor::try_new(context, data_type, arguments)?
    {
        return Ok(MatmulDispatchDescriptor::SplitK(descriptor));
    }

    Ok(MatmulDispatchDescriptor::Gemm(
        gemm::DispatchDescriptor::new(context, data_type, arguments)?,
    ))
}
