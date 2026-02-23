use super::{gemm, gemm_mpp, gemm_scalar_int};
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
    let is_int_dtype = matches!(data_type, DataType::I8 | DataType::I16 | DataType::I32);
    if is_int_dtype {
        if context.is_mpp_available() {
            return Ok(MatmulDispatchDescriptor::GemmMpp(
                gemm_mpp::DispatchDescriptor::new(data_type, arguments)?,
            ));
        }
        return Ok(MatmulDispatchDescriptor::GemmScalarInt(
            gemm_scalar_int::DispatchDescriptor::new(data_type, arguments)?,
        ));
    }

    if context.is_mpp_available() && matches!(data_type, DataType::BF16) {
        return Ok(MatmulDispatchDescriptor::GemmMpp(
            gemm_mpp::DispatchDescriptor::new(data_type, arguments)?,
        ));
    }

    if let Some(descriptor) = gemv::DispatchDescriptor::try_new::<Metal>(data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    if let Some(descriptor) = split_k::DispatchDescriptor::try_new::<Metal>(data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::SplitK(descriptor));
    }

    Ok(MatmulDispatchDescriptor::Gemm(gemm::DispatchDescriptor::new(context, data_type, arguments)?))
}
