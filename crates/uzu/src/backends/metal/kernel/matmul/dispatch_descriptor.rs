use super::gemm;
use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{MatmulArguments, MatmulDispatchDescriptor, gemv, split_k},
        metal::{Metal, context::MetalContext, error::MetalError},
    },
};

pub fn choose_dispatch_descriptor(
    context: &MetalContext,
    data_type: DataType,
    arguments: &MatmulArguments<Metal>,
) -> Result<MatmulDispatchDescriptor, MetalError> {
    if let Some(descriptor) = gemv::DispatchDescriptor::try_new::<Metal>(data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    if let Some(descriptor) = split_k::DispatchDescriptor::try_new::<Metal>(data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::SplitK(descriptor));
    }

    Ok(MatmulDispatchDescriptor::Gemm(gemm::DispatchDescriptor::new(context, data_type, arguments)?))
}
