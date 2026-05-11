#![cfg(metal_backend)]

use backend_uzu::backends::metal::MatmulDispatchPath;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchDispatchPath {
    Auto,
    Gemm,
    GemmMpp,
    UnifiedGemm,
    UnifiedGemmMxuMma,
}

impl BenchDispatchPath {
    pub const ALL: &'static [BenchDispatchPath] = &[
        BenchDispatchPath::Auto,
        BenchDispatchPath::Gemm,
        BenchDispatchPath::GemmMpp,
        BenchDispatchPath::UnifiedGemm,
        BenchDispatchPath::UnifiedGemmMxuMma,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BenchDispatchPath::Auto => "auto",
            BenchDispatchPath::Gemm => "gemm",
            BenchDispatchPath::GemmMpp => "gemm_mpp",
            BenchDispatchPath::UnifiedGemm => "unified_gemm",
            BenchDispatchPath::UnifiedGemmMxuMma => "unified_gemm_mxu",
        }
    }

    pub fn requires_mxu(self) -> bool {
        matches!(self, BenchDispatchPath::GemmMpp | BenchDispatchPath::UnifiedGemmMxuMma)
    }

    pub fn metal_dispatch(self) -> Option<MatmulDispatchPath> {
        match self {
            BenchDispatchPath::Auto => None,
            BenchDispatchPath::Gemm => Some(MatmulDispatchPath::Gemm),
            BenchDispatchPath::GemmMpp => Some(MatmulDispatchPath::GemmMpp),
            BenchDispatchPath::UnifiedGemm => Some(MatmulDispatchPath::UnifiedGemm),
            BenchDispatchPath::UnifiedGemmMxuMma => Some(MatmulDispatchPath::UnifiedGemmMxuMma),
        }
    }
}
