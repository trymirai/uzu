use super::{gemm, gemm_mixed_types_simple, gemm_mpp, gemv, split_k};

#[derive(Debug, Clone)]
pub enum MatmulDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    SplitK(split_k::DispatchDescriptor),
    Gemm(gemm::DispatchDescriptor),
    GemmMpp(gemm_mpp::DispatchDescriptor),
    GemmMixedTypesSimple(gemm_mixed_types_simple::DispatchDescriptor),
}

impl MatmulDispatchDescriptor {
    pub fn bias_is_fused(&self) -> bool {
        match self {
            MatmulDispatchDescriptor::Gemv(d) => d.bias_is_fused(),
            MatmulDispatchDescriptor::SplitK(_)
            | MatmulDispatchDescriptor::Gemm(_)
            | MatmulDispatchDescriptor::GemmMpp(_)
            | MatmulDispatchDescriptor::GemmMixedTypesSimple(_) => false,
        }
    }
}
