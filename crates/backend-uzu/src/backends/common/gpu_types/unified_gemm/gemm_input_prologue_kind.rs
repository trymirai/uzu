use debug_display::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmInputPrologueKind {
    FullPrecision,
    ExternalRht,
}
