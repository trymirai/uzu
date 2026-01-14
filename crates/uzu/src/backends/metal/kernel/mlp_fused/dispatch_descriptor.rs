use super::{common::MlpFusedArguments, gemm, gemv, split_k};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

#[derive(Debug, Clone)]
pub(crate) enum MlpFusedDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    SplitK(split_k::DispatchDescriptor),
    Gemm(gemm::DispatchDescriptor),
}

pub(crate) fn choose_dispatch_descriptor(
    context: &MTLContext,
    data_type: DataType,
    weights_transposed: bool,
    arguments: &MlpFusedArguments,
) -> Result<MlpFusedDispatchDescriptor, MTLError> {
    if weights_transposed {
        if let Some(descriptor) =
            gemv::DispatchDescriptor::try_new(context, data_type, arguments)?
        {
            return Ok(MlpFusedDispatchDescriptor::Gemv(descriptor));
        }

        if let Some(descriptor) =
            split_k::DispatchDescriptor::try_new(context, data_type, arguments)?
        {
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
