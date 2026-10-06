use super::{gemm::GemmPlan, gemv::GemvSpecialization};

pub enum MatmulDispatch {
    Gemv(GemvSpecialization),
    Gemm(GemmPlan),
}
