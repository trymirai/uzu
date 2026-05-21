#[cfg(metal_backend)]
use backend_uzu::backends::metal::MatmulDispatchPath;
#[cfg(metal_backend)]
use backend_uzu::backends::metal::MetalContext;
use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    #[display("GEMM")]
    Gemm,
}

impl Variant {
    #[cfg(metal_backend)]
    pub fn supported(
        self,
        _context: &MetalContext,
    ) -> bool {
        true
    }

    #[cfg(metal_backend)]
    pub const fn dispatch_path(self) -> MatmulDispatchPath {
        match self {
            Variant::Gemm => MatmulDispatchPath::Gemm,
        }
    }
}
