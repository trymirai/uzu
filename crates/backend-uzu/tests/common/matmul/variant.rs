#[cfg(metal_backend)]
use backend_uzu::backends::metal::MatmulDispatchPath;
use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Auto-selected FP gemm — picks MXU on capable chips, simdgroup otherwise.
    #[display("GEMM")]
    Gemm,
    /// Force the simdgroup path regardless of hardware capability.
    #[display("GEMM_SIMDGROUP")]
    GemmSimdgroup,
}

impl Variant {
    #[cfg(metal_backend)]
    pub const fn dispatch_path(self) -> MatmulDispatchPath {
        match self {
            Variant::Gemm => MatmulDispatchPath::Gemm,
            Variant::GemmSimdgroup => MatmulDispatchPath::GemmSimdgroup,
        }
    }
}
