mod routes;

pub(super) use routes::route;

use super::{gemm::GemmPlan, gemv::GemvTile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QmvRoute {
    Tuned(GemvTile),
    MainGemv(GemvTile),
    MainGemm(GemmPlan),
}
