use super::{MatmulArguments, MatmulError};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::matmul::{
            gemm::GemmDispatchDescriptor, gemv::GemvDispatchDescriptor, split_k::SplitKDispatchDescriptor,
        },
    },
};

#[derive(Debug, Clone)]
pub enum MatmulDispatchDescriptor {
    Gemv(GemvDispatchDescriptor),
    SplitK(SplitKDispatchDescriptor),
    Gemm(GemmDispatchDescriptor),
}

impl MatmulDispatchDescriptor {
    pub fn bias_is_fused(&self) -> bool {
        match self {
            MatmulDispatchDescriptor::Gemv(d) => d.bias_is_fused(),
            MatmulDispatchDescriptor::SplitK(_) | MatmulDispatchDescriptor::Gemm(_) => false,
        }
    }
}

pub fn choose_matmul_dispatch_descriptor<B: Backend>(
    context: &B::Context,
    data_type: DataType,
    arguments: &MatmulArguments<B>,
) -> Result<MatmulDispatchDescriptor, MatmulError<B>> {
    if let Some(descriptor) = GemvDispatchDescriptor::try_new::<B>(data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    if let Some(descriptor) = SplitKDispatchDescriptor::try_new::<B>(data_type, arguments)? {
        return Ok(MatmulDispatchDescriptor::SplitK(descriptor));
    }

    Ok(MatmulDispatchDescriptor::Gemm(GemmDispatchDescriptor::try_new::<B>(context, data_type, arguments)?))
}
