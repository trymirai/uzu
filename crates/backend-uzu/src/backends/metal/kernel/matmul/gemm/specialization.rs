use crate::backends::{
    common::gpu_types::gemm::{GemmAlignment, GemmDTransform},
    metal::kernel::GemmKey,
};

/// Identifies one GEMM pipeline: the template variant, plus the function constants that
/// are specialized into it. Only the key half decides which kernel was instantiated, so
/// only that half is validated -- see [`GemmKey::validate`], which the build generates
/// from gemm.metal's own CONSTRAINTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) key: GemmKey,
    pub(crate) output_transform: GemmDTransform,
    pub(crate) alignment: GemmAlignment,
}
