use derive_more::Display;

/// The order of quantized B's codes: RowMajor n rows of k codes, or Tiled [n / 8][k / 64][row % 8][column % 64].
#[repr(C)]
#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmBLayout {
    #[default]
    RowMajor,
    Tiled,
}
