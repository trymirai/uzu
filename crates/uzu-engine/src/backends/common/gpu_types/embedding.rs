/// Storage format used by the input embedding lookup kernel.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EmbeddingTableKind {
    Dense = 0,
    Quantized = 1,
    D4S4 = 2,
}
