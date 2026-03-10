use super::{gemm_mpp, gemv};

#[derive(Debug, Clone)]
pub enum MatmulDispatchDescriptor {
    Gemv(gemv::DispatchDescriptor),
    GemmMpp(gemm_mpp::DispatchDescriptor),
}

impl MatmulDispatchDescriptor {
    pub fn bias_is_fused(&self) -> bool {
        match self {
            MatmulDispatchDescriptor::Gemv(d) => d.bias_is_fused(),
            MatmulDispatchDescriptor::GemmMpp(_) => false,
        }
    }
}
