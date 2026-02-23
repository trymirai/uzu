use super::{gemm, gemm_mpp};
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
    let requires_mpp = matches!(data_type, DataType::I8 | DataType::I32);
    if requires_mpp {
        if !context.is_mpp_available() {
            return Err(MetalError::Generic(format!(
                "Integer matmul requires MPP (M5+ hardware), got dtype {data_type:?}"
            )));
        }
        return Ok(MatmulDispatchDescriptor::GemmMpp(
            gemm_mpp::DispatchDescriptor::new(data_type, arguments)?,
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
