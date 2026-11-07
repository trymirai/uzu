/// Int8Gemm's fold metadata entries per array and slot of a stage part, [kind][row or column].
pub const INT8_FOLD_ENTRIES: u32 = 2 * 64;

/// The most input rows a GEMV workgroup takes at once.
pub const GEMV_INPUT_ROW_TILE: u32 = 8;
