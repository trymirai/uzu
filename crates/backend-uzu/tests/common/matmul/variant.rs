#[cfg(metal_backend)]
use backend_uzu::backends::metal::{DeviceExt, MatmulDispatchPath, MetalContext};
use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    #[display("GEMM")]
    Gemm,
    #[display("GEMM_MPP")]
    GemmMpp,
    #[display("UnifiedGEMM")]
    UnifiedGemm,
    #[display("UnifiedGEMM_MXU")]
    UnifiedGemmMxu,
}

impl Variant {
    pub const ALL: &'static [Variant] =
        &[Variant::Gemm, Variant::GemmMpp, Variant::UnifiedGemm, Variant::UnifiedGemmMxu];

    pub const fn requires_mxu(self) -> bool {
        matches!(self, Variant::GemmMpp | Variant::UnifiedGemmMxu)
    }

    #[cfg(metal_backend)]
    pub fn supported(
        self,
        context: &MetalContext,
    ) -> bool {
        !self.requires_mxu() || context.device.supports_mxu()
    }

    #[cfg(metal_backend)]
    pub const fn dispatch_path(self) -> MatmulDispatchPath {
        match self {
            Variant::Gemm => MatmulDispatchPath::Gemm,
            Variant::GemmMpp => MatmulDispatchPath::GemmMpp,
            Variant::UnifiedGemm => MatmulDispatchPath::UnifiedGemm,
            Variant::UnifiedGemmMxu => MatmulDispatchPath::UnifiedGemmMxuMma,
        }
    }
}
