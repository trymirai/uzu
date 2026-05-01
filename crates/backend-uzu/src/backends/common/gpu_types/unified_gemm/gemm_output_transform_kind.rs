use debug_display::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmOutputTransformKind {
    Store,
    Scale,
    Accumulate,
    Bias,
    Rht,
    ScaleAccumulate,
    ScaleAccumulateBias,
    ScaleAccumulateBiasRht,
}
