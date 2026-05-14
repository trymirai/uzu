#[cfg(metal_backend)]
use backend_uzu::backends::metal::{DeviceExt, MatmulDispatchPath, MetalContext};
use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    #[display("GEMM")]
    Gemm,
    #[display("GEMM_MXU")]
    GemmMxu,
}

impl Variant {
    pub const ALL: &'static [Variant] = &[Variant::Gemm, Variant::GemmMxu];

    pub const fn requires_mxu(self) -> bool {
        matches!(self, Variant::GemmMxu)
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
            Variant::GemmMxu => MatmulDispatchPath::GemmMxu,
        }
    }
}
