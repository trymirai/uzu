#[cfg(metal_backend)]
use backend_uzu::backends::metal::{DeviceExt, MatmulDispatchPath, MetalContext};
use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    #[display("UnifiedGEMM")]
    UnifiedGemm,
    #[display("UnifiedGEMM_MXU")]
    UnifiedGemmMxu,
}

impl Variant {
    pub const ALL: &'static [Variant] = &[Variant::UnifiedGemm, Variant::UnifiedGemmMxu];

    pub const fn requires_mxu(self) -> bool {
        matches!(self, Variant::UnifiedGemmMxu)
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
            Variant::UnifiedGemm => MatmulDispatchPath::UnifiedGemm,
            Variant::UnifiedGemmMxu => MatmulDispatchPath::UnifiedGemmMxuMma,
        }
    }
}
