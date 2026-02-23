use super::{gemm, gemv, split_k};

#[derive(Debug, Clone)]
pub enum MatmulDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    SplitK(split_k::DispatchDescriptor),
    Gemm(gemm::DispatchDescriptor),
}

impl MatmulDispatchDescriptor {
    pub fn bias_is_fused(&self) -> bool {
        match self {
            MatmulDispatchDescriptor::Gemv(d) => d.bias_is_fused(),
            MatmulDispatchDescriptor::SplitK(_) | MatmulDispatchDescriptor::Gemm(_) => false,
        }
    }
}
